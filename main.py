from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from core.engine import (
    active_profile_name,
    cleanup_storage_session,
    ensure_profile_loaded,
    ensure_script_registry_dir,
)
from routers.data import (
    data_router,
    management_router,
    plugins_router,
)
from routers.execution import execution_router, management_execution_router
from routers.process import process_router
from routers.registry import registry_router
from routers.submission import submission_router

app = FastAPI(
    title="AiiDA Bridge API",
    description="Bridge API exposing AiiDA management, process/data inspection, and submission workflows.",
    version="2.0.0",
)


@app.middleware("http")
async def _cleanup_storage_session_middleware(request: Request, call_next: Any):
    try:
        response = await call_next(request)
    finally:
        cleanup_storage_session()
    return response


@app.exception_handler(HTTPException)
async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    payload: dict[str, Any] = {"error": "Internal server error", "reason": str(exc)}
    profile_name = active_profile_name()
    if profile_name:
        payload["profile"] = profile_name
    return JSONResponse(status_code=500, content=payload)


@app.on_event("startup")
def _startup() -> None:
    ensure_profile_loaded()
    ensure_script_registry_dir()


app.include_router(management_router)
app.include_router(management_execution_router)
app.include_router(process_router)
app.include_router(data_router)
app.include_router(submission_router)
app.include_router(plugins_router)
app.include_router(registry_router)
app.include_router(execution_router)


@app.get("/")
def root():
    """Root endpoint for health checks and status summary."""
    from core.engine import ensure_profile_loaded, get_system_info_payload
    ensure_profile_loaded()
    info = get_system_info_payload()
    return {
        "message": "AiiDA Bridge API is online",
        "version": "2.0.0",
        "status": info,
    }


@app.get("/status")
def root_status():
    """Root status endpoint for SABR client compatibility."""
    from core.engine import ensure_profile_loaded, get_system_info_payload
    ensure_profile_loaded()
    return get_system_info_payload()


@app.get("/plugins")
def root_plugins():
    """Root plugins endpoint for SABR client compatibility."""
    from aiida.plugins.entry_point import get_entry_point_names
    return sorted(get_entry_point_names("aiida.workflows"))


@app.get("/system/info")
def root_system_info():
    """Root system info endpoint for SABR client compatibility."""
    from core.engine import ensure_profile_loaded, get_system_info_payload
    ensure_profile_loaded()
    return get_system_info_payload()


@app.get("/resources")
def root_resources():
    """Root resources endpoint for SABR client compatibility."""
    from core.engine import ensure_profile_loaded, serialize_codes, serialize_computers
    ensure_profile_loaded()
    return {
        "computers": serialize_computers(),
        "codes": serialize_codes(),
    }





if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
