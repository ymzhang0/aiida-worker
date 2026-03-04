"""
Real-time process event broadcasting via Server-Sent Events (SSE).

Two core components:
1. BroadcastManager  – fan-out hub for N concurrent web clients.
2. aiida_event_listener – subscribes to AiiDA's message bus (or polls)
   and pushes state-change payloads into the BroadcastManager.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# BroadcastManager (Singleton)
# ──────────────────────────────────────────────────────

class BroadcastManager:
    """Fan-out hub: one producer → N async consumer queues."""

    _instance: BroadcastManager | None = None

    def __new__(cls) -> BroadcastManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        return cls._instance

    # -- client lifecycle --

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new web client and return its personal queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._queues.add(queue)
        logger.info("SSE client connected (total=%d)", len(self._queues))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a client queue on disconnect."""
        self._queues.discard(queue)
        logger.info("SSE client disconnected (total=%d)", len(self._queues))

    @property
    def client_count(self) -> int:
        return len(self._queues)

    # -- broadcasting --

    def broadcast(self, data: dict[str, Any]) -> None:
        """
        Push *data* into every connected client queue.

        Safe to call from **any** thread – when invoked from a non-asyncio
        thread (e.g. an AiiDA callback running on a kiwipy thread) we
        schedule via ``call_soon_threadsafe``.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            self._push(data)
        else:
            # Called from a foreign thread → schedule on the event loop.
            with suppress(RuntimeError):
                asyncio.get_event_loop().call_soon_threadsafe(self._push, data)

    def _push(self, data: dict[str, Any]) -> None:
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for q in self._queues:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._queues.discard(q)
            logger.warning("Dropped slow SSE client (queue full)")


broadcast_manager = BroadcastManager()


# ──────────────────────────────────────────────────────
# SSE event generator (used by the endpoint)
# ──────────────────────────────────────────────────────

async def event_generator(queue: asyncio.Queue[dict[str, Any]]) -> AsyncGenerator[dict[str, str], None]:
    """Yield SSE-formatted dicts from a client queue, with keepalive."""
    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield {
                    "event": data.get("event", "process_state_change"),
                    "data": json.dumps(data),
                }
            except asyncio.TimeoutError:
                # Keepalive comment to prevent proxy/browser timeout
                yield {"comment": "keepalive"}
    except asyncio.CancelledError:
        return


# ──────────────────────────────────────────────────────
# AiiDA Event Listener
# ──────────────────────────────────────────────────────

_POLL_INTERVAL_SECONDS = 5.0


def _try_subscribe_communicator() -> bool:
    """
    Attempt to subscribe to AiiDA's RabbitMQ broadcast channel.
    Returns True if subscription succeeded.
    """
    try:
        from aiida.manage.manager import get_manager

        manager = get_manager()
        communicator = manager.get_communicator()

        if communicator is None:
            logger.info("AiiDA communicator is None – will use polling fallback")
            return False

        def _on_broadcast(body: Any, sender: Any = None, subject: Any = None, correlation_id: Any = None) -> None:
            """Callback fired by kiwipy on any process broadcast."""
            try:
                pk = None
                state = None

                # sender is typically the process UUID
                if sender is not None:
                    # Try to resolve PK from UUID
                    with suppress(Exception):
                        from aiida import orm
                        node = orm.load_node(str(sender))
                        pk = node.pk
                        state = str(getattr(node, "process_state", None) or "unknown")

                # body may carry state info directly
                if isinstance(body, dict):
                    state = state or str(body.get("state", body.get("process_state", "unknown")))
                if isinstance(subject, str) and "state_changed" in subject:
                    pass  # confirmed state change

                if pk is not None:
                    broadcast_manager.broadcast({
                        "event": "process_state_change",
                        "pk": pk,
                        "state": state or "unknown",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception:
                logger.debug("Error processing broadcast callback", exc_info=True)

        communicator.add_broadcast_subscriber(_on_broadcast)
        logger.info("Subscribed to AiiDA communicator broadcasts")
        return True

    except Exception:
        logger.info("Could not subscribe to AiiDA communicator – will use polling fallback", exc_info=True)
        return False


async def _poll_process_changes() -> None:
    """Polling fallback: periodically query for recently changed processes."""
    from aiida import orm

    last_check = time.time()

    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        if broadcast_manager.client_count == 0:
            last_check = time.time()
            continue

        try:
            now = time.time()
            from aiida.orm import QueryBuilder, ProcessNode
            from datetime import datetime as dt, timedelta

            cutoff = dt.fromtimestamp(last_check - 1.0, tz=timezone.utc)
            qb = QueryBuilder()
            qb.append(
                ProcessNode,
                filters={"mtime": {">": cutoff}},
                project=["id", "process_state", "mtime"],
            )
            results = qb.all(flat=False)

            for row in results:
                pk, state, mtime = row[0], row[1], row[2]
                broadcast_manager.broadcast({
                    "event": "process_state_change",
                    "pk": pk,
                    "state": str(state or "unknown"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            last_check = now
        except Exception:
            logger.debug("Polling fallback error", exc_info=True)


async def aiida_event_listener() -> None:
    """
    Main entry point: try the communicator subscription first,
    then keep alive (communicator mode) or fall back to polling.
    """
    # Attempt communicator subscription (runs callback on kiwipy thread)
    has_communicator = _try_subscribe_communicator()

    if has_communicator:
        # Communicator callbacks fire on their own thread.
        # Just keep this task alive so it can be cancelled on shutdown.
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger.info("AiiDA event listener (communicator mode) shutting down")
            return
    else:
        # Polling fallback
        logger.info("Starting AiiDA event listener in polling mode (interval=%.1fs)", _POLL_INTERVAL_SECONDS)
        try:
            await _poll_process_changes()
        except asyncio.CancelledError:
            logger.info("AiiDA event listener (polling mode) shutting down")
            return
