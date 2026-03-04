from __future__ import annotations

from typing import Any

from fastapi import Query

from core.engine import SessionCleanupAPIRouter, ensure_profile_loaded
from core.scripts import get_registered_script, list_registered_scripts, register_script
from models.schemas import ScriptRegisterRequest

registry_router = SessionCleanupAPIRouter(prefix="/registry", tags=["registry"])


@registry_router.get("/list")
def registry_list() -> dict[str, Any]:
    ensure_profile_loaded()
    return list_registered_scripts()


@registry_router.post("/register")
def registry_register(payload: ScriptRegisterRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    return register_script(
        script_name=payload.script_name or "",
        script=payload.script,
        description=payload.description,
        overwrite=payload.overwrite,
    )


@registry_router.get("/workchains/{entry_point:path}/spec")
def registry_workchain_spec(entry_point: str) -> Any:
    """
    Alias to submission/spec.
    """
    from routers.submission import submission_spec
    return submission_spec(entry_point)


@registry_router.get("/{script_name}")
def registry_get(script_name: str, include_content: bool = Query(default=True)) -> dict[str, Any]:
    ensure_profile_loaded()
    return get_registered_script(script_name, include_content=include_content)
