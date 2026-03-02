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
from models.schemas import JobValidationRequest, JobValidationResponse, ResourcesResponse, SpecResponse, SystemInfoResponse, ValidationResponse
from routers.data import (
    data_router,
    list_workflow_plugins,
    management_database_summary,
    management_resources,
    management_router,
    management_statistics,
    management_system_info,
    plugins_router,
)
from routers.execution import execution_router, management_execution_router
from routers.process import process_router
from routers.registry import registry_router
from routers.submission import (
    submission_router,
    submission_spec,
    submission_submit,
    submission_validate,
    submission_validate_job,
)

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


# Legacy aliases kept for existing clients.
@app.get("/plugins", response_model=list[str])
def legacy_list_workflow_plugins() -> list[str]:
    return list_workflow_plugins()


@app.get("/system/info", response_model=SystemInfoResponse)
def legacy_get_system_info() -> SystemInfoResponse:
    return management_system_info()


@app.get("/status", response_model=SystemInfoResponse)
def legacy_get_status() -> SystemInfoResponse:
    return management_system_info()


@app.get("/resources", response_model=ResourcesResponse)
def legacy_get_resources() -> ResourcesResponse:
    return management_resources()


@app.get("/statistics")
def legacy_get_statistics() -> dict[str, Any]:
    return management_statistics()


@app.get("/database/summary")
def legacy_get_database_summary() -> dict[str, Any]:
    return management_database_summary()


@app.get("/spec/{entry_point:path}", response_model=SpecResponse)
def legacy_get_workflow_spec(entry_point: str) -> SpecResponse:
    return submission_spec(entry_point)


@app.post("/validate", response_model=ValidationResponse)
def legacy_validate_workflow_inputs(payload: dict[str, Any]) -> ValidationResponse:
    return submission_validate(payload)


@app.post("/validate-job", response_model=JobValidationResponse)
def legacy_validate_job(payload: JobValidationRequest) -> JobValidationResponse:
    return submission_validate_job(payload)


@app.post("/submit")
def legacy_submit_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    return submission_submit(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
