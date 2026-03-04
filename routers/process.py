from __future__ import annotations

from typing import Any

from core.engine import SessionCleanupAPIRouter, ensure_profile_loaded
from core.process_utils import get_process_log_payload, inspect_process_payload

process_router = SessionCleanupAPIRouter(prefix="/process", tags=["process"])


@process_router.get("/{identifier}")
def inspect_process(identifier: str) -> dict[str, Any]:
    ensure_profile_loaded()
    # PKs are already cast in the service layer
    return inspect_process_payload(identifier)


@process_router.get("/{identifier}/logs")
def inspect_process_logs(identifier: str) -> dict[str, Any]:
    ensure_profile_loaded()
    return get_process_log_payload(identifier)
