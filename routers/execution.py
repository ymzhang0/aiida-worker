from __future__ import annotations

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from aiida import orm

from core.engine import SessionCleanupAPIRouter, ensure_profile_loaded, http_error
from core.scripts import execute_registered_script
from models.schemas import PythonScriptRequest, ScriptExecuteRequest

execution_router = SessionCleanupAPIRouter(tags=["execution"])
management_execution_router = SessionCleanupAPIRouter(prefix="/management", tags=["management"])


def _execute_python_script(script: str) -> dict[str, Any]:
    code = str(script or "")
    if not code.strip():
        raise http_error(400, "Python script is required")

    from aiida import engine as aiida_engine
    from aiida import plugins as aiida_plugins

    exec_globals: dict[str, Any] = {
        "orm": orm,
        "plugins": aiida_plugins,
        "engine": aiida_engine,
    }
    output_buffer = io.StringIO()

    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            exec(code, exec_globals)
        return {
            "success": True,
            "output": output_buffer.getvalue() or "Code executed successfully (No output).",
        }
    except Exception:  # noqa: BLE001
        return {
            "success": False,
            "output": output_buffer.getvalue(),
            "error": traceback.format_exc(),
        }


@management_execution_router.post("/run-python")
def management_run_python(payload: PythonScriptRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    return _execute_python_script(payload.script)


@execution_router.post("/execute/{script_name}")
async def execute_script(script_name: str, payload: ScriptExecuteRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    return await execute_registered_script(script_name, params=payload.params)
