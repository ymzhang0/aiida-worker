from __future__ import annotations

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from aiida import orm
from fastapi import Request

from core.engine import SessionCleanupAPIRouter, ensure_profile_loaded, http_error
from core.scripts import execute_registered_script
from models.schemas import PythonScriptRequest, ScriptExecuteRequest
from repository.analysis.common_utils import (
    capture_saved_artifacts,
    activate_workspace_path,
    persist_plot_artifacts,
    save_artifact,
)

execution_router = SessionCleanupAPIRouter(tags=["execution"])
management_execution_router = SessionCleanupAPIRouter(prefix="/management", tags=["management"])
WORKSPACE_PATH_HEADER = "X-SABR-Active-Workspace-Path"


def _request_workspace_path(request: Request | None) -> str | None:
    if request is None:
        return None
    cleaned = str(request.headers.get(WORKSPACE_PATH_HEADER) or "").strip()
    return cleaned or None


def _execute_python_script(script: str, *, workspace_path: str | None = None) -> dict[str, Any]:
    code = str(script or "")
    if not code.strip():
        raise http_error(400, "Python script is required")

    from aiida import engine as aiida_engine
    from aiida import plugins as aiida_plugins

    exec_globals: dict[str, Any] = {
        "orm": orm,
        "plugins": aiida_plugins,
        "engine": aiida_engine,
        "save_artifact": save_artifact,
        "ACTIVE_WORKSPACE_PATH": workspace_path,
    }
    output_buffer = io.StringIO()

    try:
        with (
            activate_workspace_path(workspace_path),
            capture_saved_artifacts() as saved_artifacts,
            redirect_stdout(output_buffer),
            redirect_stderr(output_buffer),
        ):
            exec(code, exec_globals)
            if not saved_artifacts:
                persist_plot_artifacts(exec_globals, prefix="auto-plot")
        return {
            "success": True,
            "output": output_buffer.getvalue() or "Code executed successfully (No output).",
            "workspace_path": workspace_path,
            "artifacts": saved_artifacts,
        }
    except Exception:  # noqa: BLE001
        return {
            "success": False,
            "output": output_buffer.getvalue(),
            "workspace_path": workspace_path,
            "error": traceback.format_exc(),
        }


@management_execution_router.post("/run-python")
def management_run_python(payload: PythonScriptRequest, request: Request) -> dict[str, Any]:
    ensure_profile_loaded()
    return _execute_python_script(payload.script, workspace_path=_request_workspace_path(request))


@execution_router.post("/execute/{script_name}")
async def execute_script(script_name: str, payload: ScriptExecuteRequest, request: Request) -> dict[str, Any]:
    ensure_profile_loaded()
    return await execute_registered_script(
        script_name,
        params=payload.params,
        workspace_path=_request_workspace_path(request),
    )
