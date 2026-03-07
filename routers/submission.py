from __future__ import annotations

import inspect
import json
from typing import Any, Mapping, Sequence, get_args, get_origin

from aiida import orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.engine import submit
from aiida.engine.processes.ports import InputPort, PortNamespace
from aiida.plugins import WorkflowFactory

from core.engine import SessionCleanupAPIRouter, ensure_profile_loaded, http_error
from core.submission_utils import (
    _draft_workchain_builder,
    _extract_valid_types,
    _generate_submission_script,
    _load_workflow,
    _prepare_and_validate,
    _submit_validated_workflow,
    _submit_workchain_builder,
    _validate_job_payload,
    _validate_workchain_builder,
    resolve_generic_inputs,
)
from core.utils import serialize_spec, to_jsonable, type_to_string
from models.schemas import (
    BuilderDraftRequest,
    BuilderDraftResponse,
    BuilderScriptResponse,
    BuilderSubmitRequest,
    JobValidationRequest,
    JobValidationResponse,
    SpecResponse,
    SubmissionScriptRequest,
    SubmitResponse,
    ValidationResponse,
)

submission_router = SessionCleanupAPIRouter(prefix="/submission", tags=["submission"])

_MISSING = object()
_RESERVED_BUILDER_KEYS = {
    "entry_point",
    "workchain",
    "protocol",
    "overrides",
    "intent_data",
    "draft",
    "inputs",
}


@submission_router.get("/spec/{entry_point:path}", response_model=SpecResponse)
def submission_spec(entry_point: str) -> SpecResponse:
    ensure_profile_loaded()
    process = _load_workflow(entry_point)
    spec_payload = serialize_spec(process.spec().inputs)
    return SpecResponse(entry_point=entry_point, inputs=spec_payload)


@submission_router.post("/validate", response_model=ValidationResponse)
def submission_validate(payload: dict[str, Any]) -> ValidationResponse:
    ensure_profile_loaded()
    if not isinstance(payload, Mapping):
        raise http_error(400, "Validation payload must be a JSON object")

    if "entry_point" in payload and "inputs" in payload:
        entry_point = str(payload.get("entry_point") or "").strip()
        raw_inputs = payload.get("inputs", {})
        if raw_inputs is None:
            raw_inputs = {}
        if not isinstance(raw_inputs, Mapping):
            raise http_error(400, "Field 'inputs' must be an object for workflow validation")

        process = _load_workflow(entry_point)

        try:
            _, validation_error = _prepare_and_validate(process, raw_inputs)
        except ValueError as exc:
            raise http_error(400, "Failed to normalize or pre-process inputs", reason=str(exc)) from exc

        if validation_error is not None:
            raise http_error(
                422,
                "Validation failed",
                port=str(getattr(validation_error, "port", "") or ""),
                reason=str(getattr(validation_error, "message", "") or str(validation_error)),
                full_error=str(validation_error),
            )

        return ValidationResponse(success=True, message="Success", errors=[])

    draft_payload: Mapping[str, Any] | None = None
    maybe_draft = payload.get("draft")
    if isinstance(maybe_draft, Mapping):
        draft_payload = maybe_draft
    elif any(key in payload for key in ("workchain", "intent_data", "protocol", "overrides")):
        draft_payload = payload

    if draft_payload is not None:
        result = _validate_workchain_builder(draft_payload)
        return ValidationResponse(
            success=result["success"],
            message=result["message"],
            errors=result["errors"],
            entry_point=result.get("entry_point"),
            missing_ports=result.get("missing_ports", []),
            signature=result.get("signature", []),
            builder_inputs=result.get("builder_inputs", {}),
            pseudo_expectations=result.get("pseudo_expectations", []),
            recovery_plan=result.get("recovery_plan", {}),
        )

    raise http_error(
        400,
        "Unsupported validation payload",
        expected=[
            {"entry_point": "aiida.workflows:...", "inputs": {}},
            {"entry_point": "aiida.workflows:...", "intent_data": {}, "protocol": "moderate", "overrides": {}},
        ],
    )


@submission_router.post("/validate-job", response_model=JobValidationResponse)
def submission_validate_job(payload: JobValidationRequest) -> JobValidationResponse:
    ensure_profile_loaded()
    return JobValidationResponse(
        **_validate_job_payload(
            entry_point=payload.entry_point,
            input_pks=payload.input_pks,
            parameters=payload.parameters,
        )
    )


@submission_router.post("/submit", response_model=SubmitResponse | ValidationResponse)
async def submission_submit(payload: dict[str, Any]) -> Any:
    """
    Unified submission endpoint.
    Supports either:
    - {"entry_point": "...", "inputs": {...}} for direct workflow submission
    - {"draft": {...}} or a builder payload for protocol-driven submission
    """
    ensure_profile_loaded()
    if not isinstance(payload, dict):
        raise http_error(400, "Submission payload must be a JSON object")

    if "entry_point" in payload and "inputs" in payload:
        entry_point = str(payload.get("entry_point") or "").strip()
        raw_inputs = payload.get("inputs", {})
        if raw_inputs is None:
            raw_inputs = {}
        if not isinstance(raw_inputs, Mapping):
            raise http_error(400, "Field 'inputs' must be an object for workflow submission")
        return _submit_validated_workflow(entry_point=entry_point, inputs=raw_inputs)

    draft_payload: Mapping[str, Any] | None = None
    maybe_draft = payload.get("draft")
    if isinstance(maybe_draft, Mapping):
        draft_payload = maybe_draft
    elif any(key in payload for key in ("workchain", "intent_data", "protocol", "overrides")):
        draft_payload = payload

    if draft_payload is not None:
        return _submit_workchain_builder(draft_payload)

    raise http_error(
        400,
        "Unsupported submission payload",
        expected=[
            {"entry_point": "aiida.workflows:...", "inputs": {}},
            {"entry_point": "aiida.workflows:...", "intent_data": {}, "protocol": "moderate", "overrides": {}},
        ],
    )


@submission_router.post("/draft-builder", response_model=BuilderDraftResponse)
def submission_draft_builder(payload: BuilderDraftRequest) -> Any:
    ensure_profile_loaded()
    return _draft_workchain_builder(
        {
            "entry_point": payload.entry_point,
            "workchain": payload.workchain,
            "protocol": payload.protocol,
            "intent_data": payload.intent_data,
            "overrides": payload.overrides,
            "structure_pk": payload.structure_pk,
            "code": payload.code,
        }
    )


@submission_router.post("/submit-builder", response_model=SubmitResponse)
async def submission_submit_builder(payload: BuilderSubmitRequest) -> Any:
    ensure_profile_loaded()
    return _submit_workchain_builder(payload.draft)


@submission_router.post("/generate-script", response_model=BuilderScriptResponse)
def submission_generate_script(payload: SubmissionScriptRequest) -> Any:
    ensure_profile_loaded()
    return _generate_submission_script(
        {
            "entry_point": payload.entry_point,
            "workchain": payload.workchain,
            "protocol": payload.protocol,
            "intent_data": payload.intent_data,
            "overrides": payload.overrides,
        }
    )
