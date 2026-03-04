from __future__ import annotations

from typing import Any

from fastapi import Request
from sse_starlette.sse import EventSourceResponse

from core.engine import SessionCleanupAPIRouter, ensure_profile_loaded
from core.events import broadcast_manager, event_generator
from core.process_utils import get_process_log_payload, inspect_process_payload

process_router = SessionCleanupAPIRouter(prefix="/process", tags=["process"])


# ── SSE stream (must be declared BEFORE /{identifier} catch-all) ────

@process_router.get("/events")
async def process_events(request: Request):
    """SSE stream of real-time process state changes."""
    queue = broadcast_manager.subscribe()

    async def _generate():
        try:
            async for event in event_generator(queue):
                yield event
        finally:
            broadcast_manager.unsubscribe(queue)

    return EventSourceResponse(
        _generate(),
        ping=20,
        ping_message_factory=lambda: {"comment": "keepalive"},
    )


# ── Standard REST endpoints ─────────────────────────────────────────

@process_router.get("/{identifier}")
def inspect_process(identifier: str) -> dict[str, Any]:
    ensure_profile_loaded()
    return inspect_process_payload(identifier)


@process_router.get("/{identifier}/logs")
def inspect_process_logs(identifier: str) -> dict[str, Any]:
    ensure_profile_loaded()
    return get_process_log_payload(identifier)
