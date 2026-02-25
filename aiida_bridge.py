from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aiida import load_profile, orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.engine import submit
from aiida.engine.processes.ports import InputPort, PortNamespace
from aiida.plugins import WorkflowFactory
from aiida.plugins.entry_point import get_entry_point_names

PROFILE_NAME = os.getenv("AIIDA_PROFILE", "sandbox")
_PROFILE_LOADED = False
_ACTIVE_PROFILE_NAME = ""


def _ensure_profile_loaded() -> None:
    """Load the configured AiiDA profile once."""
    global _ACTIVE_PROFILE_NAME, _PROFILE_LOADED

    if _PROFILE_LOADED:
        return

    configured_profile = PROFILE_NAME.strip()
    candidates: list[str | None] = [configured_profile or "sandbox"]
    if "sandbox" not in candidates:
        candidates.append("sandbox")
    candidates.append(None)  # Final fallback to AiiDA's configured default profile.

    errors: list[dict[str, str]] = []
    for candidate in candidates:
        label = candidate or "<aiida-default>"
        try:
            profile = load_profile(candidate)
            _ACTIVE_PROFILE_NAME = str(getattr(profile, "name", candidate or "default"))
            _PROFILE_LOADED = True
            return
        except Exception as exc:  # noqa: BLE001
            errors.append({"message": f"Profile '{label}' failed: {exc}"})

    raise HTTPException(
        status_code=500,
        detail={
            "message": "Failed to load any AiiDA profile",
            "errors": errors,
        },
    )


def _type_to_string(value: Any) -> str:
    """Render a class/type declaration into a compact string."""
    if value is None:
        return "Any"

    if isinstance(value, tuple):
        return " | ".join(_type_to_string(entry) for entry in value)

    if isinstance(value, type):
        return value.__name__

    return str(value)


def _to_jsonable(value: Any) -> Any:
    """Convert arbitrary python/AiiDA values into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, orm.Node):
        return {
            "pk": int(value.pk),
            "uuid": str(value.uuid),
            "type": value.__class__.__name__,
        }

    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]

    if is_dataclass(value):
        return _to_jsonable(asdict(value))

    if callable(value):
        module = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__", repr(value))
        return f"<callable {module}.{qualname}>"

    return str(value)


def _extract_default(port: InputPort | PortNamespace) -> Any:
    """Return a JSON-safe default value if defined, otherwise None."""
    has_default = False
    has_default_method = getattr(port, "has_default", None)
    if callable(has_default_method):
        try:
            has_default = bool(has_default_method())
        except Exception:  # noqa: BLE001
            has_default = False

    if not has_default:
        return None

    try:
        default_value = port.default
    except Exception:  # noqa: BLE001
        return None

    return _to_jsonable(default_value)


def _extract_help(port: InputPort | PortNamespace) -> str | None:
    try:
        text = port.help
    except Exception:  # noqa: BLE001
        return None
    return str(text) if text else None


def _extract_required(port: InputPort | PortNamespace) -> bool:
    try:
        return bool(port.required)
    except Exception:  # noqa: BLE001
        return False


def serialize_spec(port_or_namespace: InputPort | PortNamespace, name: str = "inputs") -> dict[str, Any]:
    """
    Recursively serialize an AiiDA input port/namespace into a JSON-safe structure.
    """
    payload: dict[str, Any] = {
        "name": name,
        "required": _extract_required(port_or_namespace),
        "default": _extract_default(port_or_namespace),
        "help": _extract_help(port_or_namespace),
    }

    if isinstance(port_or_namespace, PortNamespace):
        payload["type"] = "PortNamespace"
        payload["dynamic"] = bool(getattr(port_or_namespace, "dynamic", False))
        payload["ports"] = {
            child_name: serialize_spec(child_port, child_name)
            for child_name, child_port in port_or_namespace.items()
        }
    else:
        payload["type"] = _type_to_string(getattr(port_or_namespace, "valid_type", None))
        payload["non_db"] = bool(getattr(port_or_namespace, "non_db", False))

    return payload


def _extract_valid_types(port: InputPort | PortNamespace) -> tuple[type[Any], ...]:
    valid_type = getattr(port, "valid_type", None)
    if valid_type is None:
        return ()
    if isinstance(valid_type, tuple):
        return tuple(entry for entry in valid_type if isinstance(entry, type))
    if isinstance(valid_type, type):
        return (valid_type,)
    return ()


def _expects_node(port: InputPort | PortNamespace) -> bool:
    for port_type in _extract_valid_types(port):
        if issubclass(port_type, orm.Node):
            return True
    return False


def _is_node_pk_candidate(port: InputPort | PortNamespace, value: Any) -> bool:
    """Heuristic for auto-resolving integer JSON values to stored AiiDA nodes."""
    if not isinstance(value, int):
        return False

    if not _expects_node(port):
        return False

    return value > 0


def _resolve_node_reference(port: InputPort | PortNamespace, value: Any, path: Sequence[str]) -> Any:
    """Resolve PK-like payloads into loaded AiiDA nodes for node-typed ports."""
    if isinstance(value, Mapping):
        # Explicit node reference format: {"pk": 123}
        if _expects_node(port) and "pk" in value and set(value.keys()).issubset({"pk", "uuid"}):
            raw_pk = value.get("pk")
            if isinstance(raw_pk, int):
                try:
                    return orm.load_node(raw_pk)
                except Exception as exc:  # noqa: BLE001
                    joined = ".".join(path)
                    raise ValueError(f"Could not load node for '{joined}' with pk={raw_pk}: {exc}") from exc
        return value

    if _is_node_pk_candidate(port, value):
        try:
            loaded_node = orm.load_node(value)
        except Exception as exc:  # noqa: BLE001
            # Keep scalar values for scalar-data ports; they are often serialized to nodes by AiiDA.
            scalar_node_types = (orm.Int, orm.Float, orm.Bool, orm.Str, orm.Dict, orm.List)
            if any(port_type in scalar_node_types for port_type in _extract_valid_types(port)):
                return value
            joined = ".".join(path)
            raise ValueError(f"Could not load node for '{joined}' with pk={value}: {exc}") from exc
        valid_types = _extract_valid_types(port)
        if valid_types and not isinstance(loaded_node, valid_types):
            joined = ".".join(path)
            expected = _type_to_string(valid_types)
            raise ValueError(
                f"Loaded node pk={value} for '{joined}' has type {loaded_node.__class__.__name__}, expected {expected}"
            )
        return loaded_node

    return value


def _resolve_inputs_for_namespace(
    namespace: PortNamespace,
    raw_inputs: Mapping[str, Any],
    path: Sequence[str] = ("inputs",),
) -> dict[str, Any]:
    """
    Resolve JSON payload values recursively against a PortNamespace.

    Known node-typed ports support PK auto-resolution.
    Unknown keys are passed through and left for AiiDA validation.
    """
    resolved: dict[str, Any] = {}

    for key, value in raw_inputs.items():
        port = namespace.get(key)
        child_path = (*path, key)

        if isinstance(port, PortNamespace) and isinstance(value, Mapping):
            resolved[key] = _resolve_inputs_for_namespace(port, value, child_path)
            continue

        if port is not None:
            resolved[key] = _resolve_node_reference(port, value, child_path)
        else:
            resolved[key] = value

    return resolved


def _format_validation_error(error: Any) -> dict[str, Any]:
    return {
        "port": str(getattr(error, "port", "") or ""),
        "message": str(getattr(error, "message", "") or str(error)),
        "full_error": str(error),
    }


def _build_error_response(
    message: str,
    *,
    errors: Sequence[dict[str, Any]] | None = None,
    entry_point: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message, "errors": list(errors or [])}
    if entry_point:
        payload["entry_point"] = entry_point
    if _ACTIVE_PROFILE_NAME:
        payload["profile"] = _ACTIVE_PROFILE_NAME
    return payload


def _prepare_and_validate(process: Any, raw_inputs: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Resolve node references, pre-process values, and validate via AiiDA ports."""
    spec_inputs = process.spec().inputs
    resolved_inputs = _resolve_inputs_for_namespace(spec_inputs, raw_inputs)

    try:
        processed_inputs = spec_inputs.pre_process(dict(resolved_inputs))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to pre-process inputs: {exc}") from exc

    validation_error = spec_inputs.validate(processed_inputs)
    return dict(processed_inputs), validation_error


def _load_process(entry_point: str) -> Any:
    try:
        return WorkflowFactory(entry_point)
    except MissingEntryPointError as exc:
        raise HTTPException(
            status_code=400,
            detail=_build_error_response(
                "Workflow entry point not found",
                entry_point=entry_point,
                errors=[{"message": str(exc)}],
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=_build_error_response(
                "Failed to load workflow entry point",
                entry_point=entry_point,
                errors=[{"message": str(exc)}],
            ),
        ) from exc


class WorkflowInputsRequest(BaseModel):
    entry_point: str = Field(..., description="AiiDA workflow entry point name")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Workflow input values")


class SpecResponse(BaseModel):
    entry_point: str
    inputs: dict[str, Any]


class ValidationResponse(BaseModel):
    success: bool
    message: str
    errors: list[dict[str, Any]] = Field(default_factory=list)


class SubmitResponse(BaseModel):
    pk: int
    uuid: str
    state: str


app = FastAPI(
    title="AiiDA Bridge API",
    description="Bridge API exposing AiiDA workflow introspection, validation, and submission.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    _ensure_profile_loaded()


@app.get("/plugins", response_model=list[str])
def list_workflow_plugins() -> list[str]:
    _ensure_profile_loaded()
    return sorted(get_entry_point_names("aiida.workflows"))


@app.get("/spec/{entry_point:path}", response_model=SpecResponse)
def get_workflow_spec(entry_point: str) -> SpecResponse:
    _ensure_profile_loaded()
    process = _load_process(entry_point)
    spec_payload = serialize_spec(process.spec().inputs, name="inputs")
    return SpecResponse(entry_point=entry_point, inputs=spec_payload)


@app.post("/validate", response_model=ValidationResponse)
def validate_workflow_inputs(payload: WorkflowInputsRequest) -> ValidationResponse:
    _ensure_profile_loaded()
    process = _load_process(payload.entry_point)

    try:
        _, validation_error = _prepare_and_validate(process, payload.inputs)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=_build_error_response(
                "Failed to normalize or pre-process inputs",
                entry_point=payload.entry_point,
                errors=[{"port": "", "message": str(exc), "full_error": str(exc)}],
            ),
        ) from exc

    if validation_error is not None:
        raise HTTPException(
            status_code=422,
            detail=_build_error_response(
                "Validation failed",
                entry_point=payload.entry_point,
                errors=[_format_validation_error(validation_error)],
            ),
        )

    return ValidationResponse(success=True, message="Success", errors=[])


@app.post("/submit", response_model=SubmitResponse)
def submit_workflow(payload: WorkflowInputsRequest) -> SubmitResponse:
    _ensure_profile_loaded()
    process = _load_process(payload.entry_point)

    try:
        processed_inputs, validation_error = _prepare_and_validate(process, payload.inputs)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=_build_error_response(
                "Failed to normalize or pre-process inputs",
                entry_point=payload.entry_point,
                errors=[{"port": "", "message": str(exc), "full_error": str(exc)}],
            ),
        ) from exc

    if validation_error is not None:
        raise HTTPException(
            status_code=422,
            detail=_build_error_response(
                "Validation failed",
                entry_point=payload.entry_point,
                errors=[_format_validation_error(validation_error)],
            ),
        )

    try:
        node = submit(process, **processed_inputs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=_build_error_response(
                "Submission failed",
                entry_point=payload.entry_point,
                errors=[{"message": str(exc)}],
            ),
        ) from exc

    process_state = getattr(node, "process_state", None)
    state = process_state.value if hasattr(process_state, "value") else str(process_state or "created")
    return SubmitResponse(pk=int(node.pk), uuid=str(node.uuid), state=state)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("aiida_bridge:app", host="127.0.0.1", port=8001, reload=False)
