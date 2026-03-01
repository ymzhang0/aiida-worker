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
from core.utils import serialize_spec, to_jsonable, type_to_string
from models.schemas import (
    BuilderDraftRequest,
    BuilderSubmitRequest,
    JobValidationRequest,
    JobValidationResponse,
    SpecResponse,
    SubmissionScriptRequest,
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


def _load_workflow(entry_point: str) -> Any:
    cleaned = str(entry_point or "").strip()
    if not cleaned:
        raise http_error(400, "Workflow entry point is required")

    try:
        return WorkflowFactory(cleaned)
    except MissingEntryPointError as exc:
        raise http_error(404, "Workflow entry point not found", entry_point=cleaned, reason=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise http_error(400, "Failed to load workflow entry point", entry_point=cleaned, reason=str(exc)) from exc


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


def _is_node_pk_candidate(
    port: InputPort | PortNamespace,
    value: Any,
    *,
    allow_scalar_pk_resolution: bool = True,
) -> bool:
    if not allow_scalar_pk_resolution:
        return False
    if not isinstance(value, int):
        return False
    if not _expects_node(port):
        return False
    return value > 0


def _resolve_node_reference(
    port: InputPort | PortNamespace,
    value: Any,
    path: Sequence[str],
    *,
    allow_scalar_pk_resolution: bool = True,
) -> Any:
    if isinstance(value, Mapping):
        if _expects_node(port) and "pk" in value and set(value.keys()).issubset({"pk", "uuid"}):
            raw_pk = value.get("pk")
            if isinstance(raw_pk, int):
                try:
                    return orm.load_node(raw_pk)
                except Exception as exc:  # noqa: BLE001
                    joined = ".".join(path)
                    raise ValueError(f"Could not load node for '{joined}' with pk={raw_pk}: {exc}") from exc
        return value

    if _is_node_pk_candidate(port, value, allow_scalar_pk_resolution=allow_scalar_pk_resolution):
        try:
            loaded_node = orm.load_node(value)
        except Exception as exc:  # noqa: BLE001
            scalar_node_types = (orm.Int, orm.Float, orm.Bool, orm.Str, orm.Dict, orm.List)
            if any(port_type in scalar_node_types for port_type in _extract_valid_types(port)):
                return value
            joined = ".".join(path)
            raise ValueError(f"Could not load node for '{joined}' with pk={value}: {exc}") from exc

        valid_types = _extract_valid_types(port)
        if valid_types and not isinstance(loaded_node, valid_types):
            joined = ".".join(path)
            expected = type_to_string(valid_types)
            raise ValueError(
                f"Loaded node pk={value} for '{joined}' has type {loaded_node.__class__.__name__}, expected {expected}"
            )
        return loaded_node

    return value


def _resolve_inputs_for_namespace(
    namespace: PortNamespace,
    raw_inputs: Mapping[str, Any],
    path: Sequence[str] = ("inputs",),
    *,
    allow_scalar_pk_resolution: bool = True,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    for key, value in raw_inputs.items():
        port = namespace.get(key)
        child_path = (*path, key)

        if isinstance(port, PortNamespace) and isinstance(value, Mapping):
            resolved[key] = _resolve_inputs_for_namespace(
                port,
                value,
                child_path,
                allow_scalar_pk_resolution=allow_scalar_pk_resolution,
            )
            continue

        if port is not None:
            resolved[key] = _resolve_node_reference(
                port,
                value,
                child_path,
                allow_scalar_pk_resolution=allow_scalar_pk_resolution,
            )
        else:
            resolved[key] = value

    return resolved


def _prepare_and_validate(process: Any, raw_inputs: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    spec_inputs = process.spec().inputs
    resolved_inputs = _resolve_inputs_for_namespace(spec_inputs, raw_inputs)

    try:
        processed_inputs = spec_inputs.pre_process(dict(resolved_inputs))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to pre-process inputs: {exc}") from exc

    spec = process.spec()
    validation_error = None
    used_process_validate = False
    validate_method = getattr(spec, "validate", None)
    if callable(validate_method):
        try:
            validation_error = validate_method(dict(processed_inputs))
            used_process_validate = True
        except Exception:
            used_process_validate = False
    if not used_process_validate:
        validation_error = spec_inputs.validate(processed_inputs)
    return dict(processed_inputs), validation_error


def _merge_nested_inputs(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = _merge_nested_inputs(existing, value)
        else:
            merged[str(key)] = value
    return merged


def _safe_pk_value(entity: Any) -> int | None:
    raw = getattr(entity, "pk", None)
    if isinstance(raw, int):
        return int(raw)
    return None


def _serialize_computer_summary(computer: orm.Computer) -> dict[str, Any]:
    return {
        "pk": _safe_pk_value(computer),
        "uuid": str(getattr(computer, "uuid", "") or "") or None,
        "label": str(getattr(computer, "label", "") or "") or None,
        "hostname": str(getattr(computer, "hostname", "") or "") or None,
    }


def _serialize_code_summary(code: orm.AbstractCode) -> dict[str, Any]:
    computer_payload: dict[str, Any] | None = None
    try:
        computer = getattr(code, "computer", None)
        if isinstance(computer, orm.Computer):
            computer_payload = _serialize_computer_summary(computer)
    except Exception:  # noqa: BLE001
        computer_payload = None

    return {
        "pk": _safe_pk_value(code),
        "uuid": str(getattr(code, "uuid", "") or "") or None,
        "label": str(getattr(code, "label", "") or "") or None,
        "type": code.__class__.__name__,
        "computer": computer_payload,
    }


def _collect_job_resources(
    value: Any,
    *,
    codes_by_key: dict[str, dict[str, Any]],
    computers_by_key: dict[str, dict[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _collect_job_resources(nested, codes_by_key=codes_by_key, computers_by_key=computers_by_key)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for nested in value:
            _collect_job_resources(nested, codes_by_key=codes_by_key, computers_by_key=computers_by_key)
        return

    if isinstance(value, orm.AbstractCode):
        code_payload = _serialize_code_summary(value)
        code_key = str(code_payload.get("uuid") or code_payload.get("pk") or id(value))
        codes_by_key[code_key] = code_payload

        computer_payload = code_payload.get("computer")
        if isinstance(computer_payload, Mapping):
            computer_key = str(computer_payload.get("uuid") or computer_payload.get("pk") or f"code:{code_key}")
            computers_by_key[computer_key] = dict(computer_payload)
        return

    if isinstance(value, orm.Computer):
        computer_payload = _serialize_computer_summary(value)
        computer_key = str(computer_payload.get("uuid") or computer_payload.get("pk") or id(value))
        computers_by_key[computer_key] = computer_payload


def _build_job_validation_summary(inputs: Mapping[str, Any]) -> dict[str, Any]:
    codes_by_key: dict[str, dict[str, Any]] = {}
    computers_by_key: dict[str, dict[str, Any]] = {}
    _collect_job_resources(inputs, codes_by_key=codes_by_key, computers_by_key=computers_by_key)

    codes = list(codes_by_key.values())
    computers = list(computers_by_key.values())
    primary_code = codes[0] if codes else None
    primary_computer = None

    if isinstance(primary_code, Mapping):
        maybe_primary_computer = primary_code.get("computer")
        if isinstance(maybe_primary_computer, Mapping):
            primary_computer = dict(maybe_primary_computer)
    if primary_computer is None and computers:
        primary_computer = computers[0]

    return {
        "inputs": to_jsonable(inputs),
        "code": primary_code,
        "computer": primary_computer,
        "codes": codes,
        "computers": computers,
    }


def _validate_job_payload(
    *,
    entry_point: str,
    input_pks: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    process = _load_workflow(entry_point)
    spec_inputs = process.spec().inputs
    merged_inputs = _merge_nested_inputs(parameters, input_pks)

    errors: list[dict[str, Any]] = []
    try:
        resolved_parameters = _resolve_inputs_for_namespace(
            spec_inputs,
            parameters,
            allow_scalar_pk_resolution=False,
        )
        resolved_pk_inputs = _resolve_inputs_for_namespace(
            spec_inputs,
            input_pks,
            allow_scalar_pk_resolution=True,
        )
        resolved_inputs = _merge_nested_inputs(resolved_parameters, resolved_pk_inputs)
    except ValueError as exc:
        errors.append({"stage": "resolve_inputs", "message": str(exc)})
        return {
            "success": False,
            "dry_run": True,
            "entry_point": str(entry_point),
            "summary": {
                "inputs": to_jsonable(merged_inputs),
                "code": None,
                "computer": None,
                "codes": [],
                "computers": [],
            },
            "errors": errors,
        }

    try:
        processed_inputs = spec_inputs.pre_process(dict(resolved_inputs))
    except Exception as exc:  # noqa: BLE001
        errors.append({"stage": "pre_process", "message": str(exc)})
        return {
            "success": False,
            "dry_run": True,
            "entry_point": str(entry_point),
            "summary": _build_job_validation_summary(resolved_inputs),
            "errors": errors,
        }

    validation_error = None
    used_process_validate = False
    spec = process.spec()
    validate_method = getattr(spec, "validate", None)
    if callable(validate_method):
        try:
            validation_error = validate_method(dict(processed_inputs))
            used_process_validate = True
        except Exception:
            used_process_validate = False
    if not used_process_validate:
        validation_error = spec_inputs.validate(processed_inputs)

    if validation_error is not None:
        errors.append(
            {
                "stage": "validate",
                "port": str(getattr(validation_error, "port", "") or ""),
                "message": str(getattr(validation_error, "message", "") or str(validation_error)),
                "full_error": str(validation_error),
            }
        )

    return {
        "success": len(errors) == 0,
        "dry_run": True,
        "entry_point": str(entry_point),
        "summary": _build_job_validation_summary(processed_inputs),
        "errors": errors,
    }


# -----------------------------------------------------------------------------
# Dynamic protocol builder inspection and assembly
# -----------------------------------------------------------------------------

def _normalize_lookup_key(value: str) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def _find_intent_value(name: str, intent_data: Mapping[str, Any]) -> Any:
    if name in intent_data:
        direct = intent_data[name]
        if direct is not None:
            return direct

    normalized_target = _normalize_lookup_key(name)
    for key, value in intent_data.items():
        if _normalize_lookup_key(key) == normalized_target and value is not None:
            return value

    suffixes = ("pk", "label", "uuid", "node", "id")
    for key, value in intent_data.items():
        normalized_key = _normalize_lookup_key(key)
        for suffix in suffixes:
            if (
                value is not None
                and (
                    normalized_key == f"{normalized_target}{suffix}"
                    or normalized_target == f"{normalized_key}{suffix}"
                )
            ):
                return value

    return _MISSING


def _is_required_parameter(parameter: inspect.Parameter) -> bool:
    if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
        return False
    return parameter.default is inspect._empty


def _resolve_code_reference(value: Any, *, argument_name: str) -> orm.AbstractCode:
    if isinstance(value, orm.AbstractCode):
        return value
    try:
        return orm.load_code(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to resolve protocol argument '{argument_name}' as AiiDA Code: {exc}") from exc


def _resolve_node_reference_value(
    value: Any,
    *,
    argument_name: str,
    expected_types: tuple[type[Any], ...] | None = None,
) -> orm.Node:
    loaded_node: orm.Node
    if isinstance(value, orm.Node):
        loaded_node = value
    elif isinstance(value, Mapping):
        if "pk" in value:
            loaded_node = orm.load_node(value.get("pk"))
        elif "uuid" in value:
            loaded_node = orm.load_node(value.get("uuid"))
        else:
            raise ValueError(
                f"Protocol argument '{argument_name}' expects node-like reference mapping with 'pk' or 'uuid'."
            )
    else:
        loaded_node = orm.load_node(value)

    if expected_types:
        if not isinstance(loaded_node, expected_types):
            expected = type_to_string(expected_types)
            raise ValueError(
                f"Protocol argument '{argument_name}' resolved node type {loaded_node.__class__.__name__}, expected {expected}."
            )
    return loaded_node


def _extract_annotation_types(annotation: Any) -> tuple[type[Any], ...]:
    if annotation is inspect._empty or isinstance(annotation, str):
        return ()
    if isinstance(annotation, type):
        return (annotation,)

    origin = get_origin(annotation)
    if origin is None:
        return ()

    collected: list[type[Any]] = []
    for arg in get_args(annotation):
        collected.extend(_extract_annotation_types(arg))
    return tuple(collected)


def _annotation_includes_subclass(annotation: Any, target: type[Any]) -> bool:
    for candidate in _extract_annotation_types(annotation):
        try:
            if issubclass(candidate, target):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _is_pseudo_related(path: Sequence[str]) -> bool:
    return "pseudo" in ".".join(path).lower()


def _validate_pseudo_family_group(label: str, *, path: Sequence[str]) -> None:
    cleaned = str(label or "").strip()
    if not cleaned:
        raise ValueError(f"Pseudo family label for '{'.'.join(path)}' cannot be empty.")
    try:
        orm.Group.collection.get(label=cleaned)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Pseudo family group '{cleaned}' for '{'.'.join(path)}' was not found on this worker: {exc}"
        ) from exc


def _sanitize_overrides_for_spec(
    namespace: PortNamespace,
    overrides: Mapping[str, Any],
    path: Sequence[str] = ("overrides",),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sanitized: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    for raw_key, value in overrides.items():
        key = str(raw_key)
        child_path = (*path, key)
        port = namespace.get(key)

        if port is None:
            errors.append(
                {
                    "stage": "invalid_override",
                    "port": ".".join(child_path),
                    "message": "Override key does not exist in workchain input spec",
                }
            )
            continue

        if isinstance(port, PortNamespace):
            if not isinstance(value, Mapping):
                errors.append(
                    {
                        "stage": "invalid_override_type",
                        "port": ".".join(child_path),
                        "message": "Expected object value for PortNamespace override",
                    }
                )
                continue
            nested, nested_errors = _sanitize_overrides_for_spec(port, value, child_path)
            sanitized[key] = nested
            errors.extend(nested_errors)
            continue

        if _is_pseudo_related(child_path):
            valid_types = _extract_valid_types(port)
            expects_string = any(port_type is str for port_type in valid_types)
            if isinstance(value, str) and expects_string and "family" in key.lower():
                try:
                    _validate_pseudo_family_group(value, path=child_path)
                except ValueError as exc:
                    errors.append(
                        {
                            "stage": "invalid_pseudo_family",
                            "port": ".".join(child_path),
                            "message": str(exc),
                        }
                    )
                    continue

        sanitized[key] = value

    return sanitized, errors


def _collect_required_port_paths(namespace: PortNamespace, prefix: Sequence[str] = ()) -> list[tuple[str, ...]]:
    required: list[tuple[str, ...]] = []
    for key, port in namespace.items():
        path = (*prefix, str(key))
        if isinstance(port, PortNamespace):
            required.extend(_collect_required_port_paths(port, path))
            continue
        if bool(getattr(port, "required", False)):
            required.append(path)
    return required


def _path_exists(payload: Mapping[str, Any], path: Sequence[str]) -> bool:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False
        current = current[key]
    return current is not None


def _list_missing_required_ports(namespace: PortNamespace, payload: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for path in _collect_required_port_paths(namespace):
        if path and path[0] == "metadata":
            continue
        if not _path_exists(payload, path):
            missing.append(".".join(path))
    return missing


def _collect_pseudo_expectations(namespace: PortNamespace, prefix: Sequence[str] = ()) -> list[dict[str, Any]]:
    expectations: list[dict[str, Any]] = []
    for key, port in namespace.items():
        path = (*prefix, str(key))
        if _is_pseudo_related(path):
            if isinstance(port, PortNamespace):
                expectations.append({"path": ".".join(path), "mode": "mapping"})
            else:
                valid_types = _extract_valid_types(port)
                if any(port_type is str for port_type in valid_types):
                    expectations.append({"path": ".".join(path), "mode": "pseudo_family"})
                elif any(issubclass(port_type, orm.Node) for port_type in valid_types):
                    expectations.append({"path": ".".join(path), "mode": "pseudo_node"})
        if isinstance(port, PortNamespace):
            expectations.extend(_collect_pseudo_expectations(port, path))
    return expectations


def _coerce_builder_argument(
    name: str,
    raw_value: Any,
    parameter: inspect.Parameter,
    spec_inputs: PortNamespace,
) -> Any:
    normalized_name = name.lower()
    port = spec_inputs.get(name)

    if port is not None and isinstance(port, PortNamespace) and isinstance(raw_value, str) and _is_pseudo_related((name,)):
        raise ValueError(
            f"Protocol argument '{name}' expects a mapping of pseudopotential data, not a string."
        )

    if "pseudo" in normalized_name and "family" in normalized_name and isinstance(raw_value, str):
        _validate_pseudo_family_group(raw_value, path=(name,))
        return raw_value

    expects_code = _annotation_includes_subclass(parameter.annotation, orm.AbstractCode)
    if not expects_code:
        expects_code = normalized_name == "code" or normalized_name.endswith("_code")

    if expects_code:
        return _resolve_code_reference(raw_value, argument_name=name)

    expected_node_types: tuple[type[Any], ...] = ()
    if _annotation_includes_subclass(parameter.annotation, orm.Node):
        expected_node_types = tuple(
            candidate
            for candidate in _extract_annotation_types(parameter.annotation)
            if isinstance(candidate, type) and issubclass(candidate, orm.Node)
        )
    elif port is not None and _expects_node(port):
        expected_node_types = _extract_valid_types(port)

    if expected_node_types:
        return _resolve_node_reference_value(raw_value, argument_name=name, expected_types=expected_node_types)

    return raw_value


def _extract_builder_request(draft_data: Mapping[str, Any]) -> tuple[str, str | None, dict[str, Any], dict[str, Any]]:
    entry_point = str(draft_data.get("entry_point") or draft_data.get("workchain") or "").strip()
    if not entry_point:
        raise http_error(400, "Builder payload requires 'entry_point' or 'workchain'")

    protocol_raw = draft_data.get("protocol", "moderate")
    protocol = str(protocol_raw) if protocol_raw is not None else None

    intent_data: dict[str, Any] = {}
    provided_intent = draft_data.get("intent_data")
    if isinstance(provided_intent, Mapping):
        intent_data.update(dict(provided_intent))

    for key, value in draft_data.items():
        if key in _RESERVED_BUILDER_KEYS:
            continue
        if value is None:
            continue
        intent_data.setdefault(str(key), value)

    # Backward compatibility for older payload style.
    for legacy_key in ("structure_pk", "structure", "code", "code_label"):
        if legacy_key in draft_data and draft_data.get(legacy_key) is not None and legacy_key not in intent_data:
            intent_data[legacy_key] = draft_data[legacy_key]

    raw_overrides = draft_data.get("overrides") or {}
    if not isinstance(raw_overrides, Mapping):
        raise http_error(400, "Builder payload field 'overrides' must be an object")

    return entry_point, protocol, intent_data, dict(raw_overrides)


def _inspect_builder_protocol_signature(workchain: Any) -> dict[str, inspect.Parameter]:
    if not hasattr(workchain, "get_builder_from_protocol"):
        raise http_error(400, "WorkChain does not define get_builder_from_protocol")

    try:
        signature = inspect.signature(workchain.get_builder_from_protocol)
    except Exception as exc:  # noqa: BLE001
        raise http_error(500, "Failed to inspect get_builder_from_protocol signature", reason=str(exc)) from exc

    parameters: dict[str, inspect.Parameter] = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        parameters[name] = parameter
    return parameters


def _build_dynamic_protocol_builder(draft_data: Mapping[str, Any]) -> dict[str, Any]:
    entry_point, protocol, intent_data, overrides = _extract_builder_request(draft_data)
    workchain = _load_workflow(entry_point)
    parameters = _inspect_builder_protocol_signature(workchain)

    spec_inputs = workchain.spec().inputs
    errors: list[dict[str, Any]] = []

    sanitized_overrides, override_errors = _sanitize_overrides_for_spec(spec_inputs, overrides)
    errors.extend(override_errors)

    try:
        resolved_overrides = _resolve_inputs_for_namespace(
            spec_inputs,
            sanitized_overrides,
            allow_scalar_pk_resolution=True,
        )
    except ValueError as exc:
        errors.append(
            {
                "stage": "resolve_overrides",
                "message": str(exc),
            }
        )
        resolved_overrides = sanitized_overrides

    kwargs: dict[str, Any] = {}

    for name, parameter in parameters.items():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue

        if name == "protocol":
            if protocol is not None:
                kwargs[name] = protocol
            continue

        if name == "overrides":
            kwargs[name] = resolved_overrides
            continue

        raw_value = _find_intent_value(name, intent_data)
        if raw_value is _MISSING:
            if _is_required_parameter(parameter):
                errors.append(
                    {
                        "stage": "missing_protocol_argument",
                        "port": name,
                        "message": f"Missing required protocol argument '{name}'",
                    }
                )
            continue

        try:
            kwargs[name] = _coerce_builder_argument(name, raw_value, parameter, spec_inputs)
        except ValueError as exc:
            errors.append(
                {
                    "stage": "resolve_protocol_argument",
                    "port": name,
                    "message": str(exc),
                }
            )

    if "overrides" not in parameters and resolved_overrides:
        errors.append(
            {
                "stage": "unsupported_overrides",
                "port": "overrides",
                "message": "This WorkChain does not accept 'overrides' in get_builder_from_protocol",
            }
        )

    builder = None
    builder_inputs: dict[str, Any] = {}
    missing_ports: list[str] = []

    if not errors:
        try:
            builder = workchain.get_builder_from_protocol(**kwargs)
            builder_inputs = builder._inputs(prune=False)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "stage": "get_builder_from_protocol",
                    "message": str(exc),
                }
            )

    if builder is not None:
        missing_ports = _list_missing_required_ports(spec_inputs, builder_inputs)
        for missing in missing_ports:
            errors.append(
                {
                    "stage": "missing_required_ports",
                    "port": missing,
                    "message": f"Required input '{missing}' is missing after builder construction",
                }
            )

        if not missing_ports:
            try:
                _processed_inputs, validation_error = _prepare_and_validate(workchain, builder_inputs)
                if validation_error is not None:
                    errors.append(
                        {
                            "stage": "validate",
                            "port": str(getattr(validation_error, "port", "") or ""),
                            "message": str(getattr(validation_error, "message", "") or str(validation_error)),
                            "full_error": str(validation_error),
                        }
                    )
            except ValueError as exc:
                errors.append(
                    {
                        "stage": "validate",
                        "message": str(exc),
                    }
                )

    signature_payload: list[dict[str, Any]] = []
    for name, parameter in parameters.items():
        signature_payload.append(
            {
                "name": name,
                "kind": str(parameter.kind),
                "required": _is_required_parameter(parameter),
                "default": None if parameter.default is inspect._empty else to_jsonable(parameter.default),
                "annotation": None
                if parameter.annotation is inspect._empty
                else str(getattr(parameter.annotation, "__name__", parameter.annotation)),
            }
        )

    return {
        "success": len(errors) == 0,
        "entry_point": entry_point,
        "protocol": protocol,
        "intent_data": to_jsonable(intent_data),
        "overrides": to_jsonable(resolved_overrides),
        "protocol_kwargs": to_jsonable(kwargs),
        "signature": signature_payload,
        "pseudo_expectations": _collect_pseudo_expectations(spec_inputs),
        "required_ports": _list_missing_required_ports(spec_inputs, {}),
        "missing_ports": missing_ports,
        "builder": builder,
        "builder_inputs": builder_inputs,
        "errors": errors,
    }


def _draft_workchain_builder(draft_data: Mapping[str, Any]) -> dict[str, Any]:
    result = _build_dynamic_protocol_builder(draft_data)
    if not result["success"]:
        return {
            "status": "DRAFT_INVALID",
            "entry_point": result["entry_point"],
            "protocol": result["protocol"],
            "intent_data": result["intent_data"],
            "overrides": result["overrides"],
            "errors": result["errors"],
            "missing_ports": result["missing_ports"],
            "signature": result["signature"],
            "pseudo_expectations": result["pseudo_expectations"],
        }

    return {
        "status": "DRAFT_READY",
        "entry_point": result["entry_point"],
        "protocol": result["protocol"],
        "intent_data": result["intent_data"],
        "overrides": result["overrides"],
        "signature": result["signature"],
        "pseudo_expectations": result["pseudo_expectations"],
        "preview": f"Ready to submit {result['entry_point']} using protocol '{result['protocol']}'.",
    }


def _submit_workchain_builder(draft_data: Mapping[str, Any]) -> dict[str, Any]:
    result = _build_dynamic_protocol_builder(draft_data)
    if not result["success"]:
        raise http_error(
            422,
            "Builder validation failed",
            errors=result["errors"],
            missing_ports=result["missing_ports"],
            entry_point=result["entry_point"],
        )

    builder = result.get("builder")
    if builder is None:
        raise http_error(500, "Builder construction failed unexpectedly", entry_point=result["entry_point"])

    try:
        node = submit(builder)
    except Exception as exc:  # noqa: BLE001
        raise http_error(500, "WorkChain submission failed", reason=str(exc)) from exc

    process_state = getattr(node, "process_state", None)
    state = process_state.value if hasattr(process_state, "value") else str(process_state or "created")
    return {
        "status": "submitted",
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "state": state,
    }


def _validate_workchain_builder(draft_data: Mapping[str, Any]) -> dict[str, Any]:
    result = _build_dynamic_protocol_builder(draft_data)
    if not result["success"]:
        return {
            "success": False,
            "message": "Builder validation failed",
            "errors": result["errors"],
            "missing_ports": result["missing_ports"],
            "entry_point": result["entry_point"],
            "signature": result["signature"],
            "pseudo_expectations": result["pseudo_expectations"],
        }

    return {
        "success": True,
        "message": "Builder draft validated",
        "errors": [],
        "entry_point": result["entry_point"],
        "signature": result["signature"],
        "pseudo_expectations": result["pseudo_expectations"],
    }


def _render_dynamic_submission_script(
    *,
    entry_point: str,
    protocol: str | None,
    intent_data: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> str:
    entry_json = json.dumps(entry_point)
    protocol_json = json.dumps(protocol)
    intent_json = json.dumps(to_jsonable(dict(intent_data)), indent=2, ensure_ascii=True)
    overrides_json = json.dumps(to_jsonable(dict(overrides)), indent=2, ensure_ascii=True)

    return (
        "from __future__ import annotations\n"
        "\n"
        "import inspect\n"
        "from typing import Any, Mapping\n"
        "\n"
        "from aiida import orm\n"
        "from aiida.engine import submit\n"
        "from aiida.plugins import WorkflowFactory\n"
        "\n"
        f"ENTRY_POINT = {entry_json}\n"
        f"PROTOCOL = {protocol_json}\n"
        f"INTENT_DATA = {intent_json}\n"
        f"OVERRIDES = {overrides_json}\n"
        "\n"
        "\n"
        "def _normalize_key(value: str) -> str:\n"
        "    return ''.join(ch for ch in str(value).lower() if ch.isalnum())\n"
        "\n"
        "\n"
        "def _find_intent_value(name: str, intent_data: Mapping[str, Any]) -> Any:\n"
        "    if name in intent_data:\n"
        "        return intent_data[name]\n"
        "\n"
        "    target = _normalize_key(name)\n"
        "    for key, value in intent_data.items():\n"
        "        if _normalize_key(key) == target:\n"
        "            return value\n"
        "\n"
        "    for key, value in intent_data.items():\n"
        "        normalized = _normalize_key(key)\n"
        "        for suffix in ('pk', 'label', 'uuid', 'node', 'id'):\n"
        "            if normalized == f'{target}{suffix}' or target == f'{normalized}{suffix}':\n"
        "                return value\n"
        "\n"
        "    return None\n"
        "\n"
        "\n"
        "def _coerce_argument(name: str, value: Any) -> Any:\n"
        "    if value is None:\n"
        "        return None\n"
        "\n"
        "    lower = name.lower()\n"
        "    if lower == 'code' or lower.endswith('_code'):\n"
        "        return orm.load_code(value)\n"
        "\n"
        "    if isinstance(value, Mapping) and set(value.keys()).issubset({'pk', 'uuid'}):\n"
        "        if 'pk' in value:\n"
        "            return orm.load_node(value['pk'])\n"
        "        if 'uuid' in value:\n"
        "            return orm.load_node(value['uuid'])\n"
        "\n"
        "    if isinstance(value, int) and value > 0:\n"
        "        try:\n"
        "            return orm.load_node(value)\n"
        "        except Exception:\n"
        "            return value\n"
        "\n"
        "    if 'pseudo' in lower and 'family' in lower and isinstance(value, str):\n"
        "        orm.Group.collection.get(label=value)\n"
        "\n"
        "    return value\n"
        "\n"
        "\n"
        "def _collect_required_ports(namespace, prefix=()):\n"
        "    required = []\n"
        "    for key, port in namespace.items():\n"
        "        path = (*prefix, str(key))\n"
        "        if hasattr(port, 'items'):\n"
        "            required.extend(_collect_required_ports(port, path))\n"
        "        elif getattr(port, 'required', False) and (not path or path[0] != 'metadata'):\n"
        "            required.append(path)\n"
        "    return required\n"
        "\n"
        "\n"
        "def _path_exists(payload: Mapping[str, Any], path: tuple[str, ...]) -> bool:\n"
        "    current: Any = payload\n"
        "    for key in path:\n"
        "        if not isinstance(current, Mapping) or key not in current:\n"
        "            return False\n"
        "        current = current[key]\n"
        "    return current is not None\n"
        "\n"
        "\n"
        "wc_class = WorkflowFactory(ENTRY_POINT)\n"
        "signature = inspect.signature(wc_class.get_builder_from_protocol)\n"
        "kwargs = {}\n"
        "\n"
        "for name, parameter in signature.parameters.items():\n"
        "    if name in {'self', 'cls'}:\n"
        "        continue\n"
        "    if name == 'protocol':\n"
        "        kwargs[name] = PROTOCOL\n"
        "        continue\n"
        "    if name == 'overrides':\n"
        "        kwargs[name] = OVERRIDES\n"
        "        continue\n"
        "    if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):\n"
        "        continue\n"
        "\n"
        "    raw = _find_intent_value(name, INTENT_DATA)\n"
        "    if raw is None:\n"
        "        if parameter.default is inspect._empty:\n"
        "            raise ValueError(f'Missing required protocol argument: {name}')\n"
        "        continue\n"
        "\n"
        "    kwargs[name] = _coerce_argument(name, raw)\n"
        "\n"
        "builder = wc_class.get_builder_from_protocol(**kwargs)\n"
        "builder_inputs = builder._inputs(prune=False)\n"
        "missing = [\n"
        "    '.'.join(path)\n"
        "    for path in _collect_required_ports(wc_class.spec().inputs)\n"
        "    if not _path_exists(builder_inputs, path)\n"
        "]\n"
        "if missing:\n"
        "    raise ValueError(f'Missing required ports after builder construction: {missing}')\n"
        "\n"
        "node = submit(builder)\n"
        "print({'pk': int(node.pk), 'uuid': str(node.uuid)})\n"
    )


def _generate_submission_script(draft_data: Mapping[str, Any]) -> dict[str, Any]:
    entry_point, protocol, intent_data, overrides = _extract_builder_request(draft_data)
    workchain = _load_workflow(entry_point)
    parameters = _inspect_builder_protocol_signature(workchain)

    return {
        "entry_point": entry_point,
        "protocol": protocol,
        "signature": [
            {
                "name": name,
                "required": _is_required_parameter(parameter),
                "kind": str(parameter.kind),
            }
            for name, parameter in parameters.items()
        ],
        "script": _render_dynamic_submission_script(
            entry_point=entry_point,
            protocol=protocol,
            intent_data=intent_data,
            overrides=overrides,
        ),
    }


def _submit_validated_workflow(entry_point: str, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    process = _load_workflow(entry_point)

    try:
        processed_inputs, validation_error = _prepare_and_validate(process, inputs or {})
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

    try:
        node = submit(process, **processed_inputs)
    except Exception as exc:  # noqa: BLE001
        raise http_error(500, "Submission failed", reason=str(exc)) from exc

    process_state = getattr(node, "process_state", None)
    state = process_state.value if hasattr(process_state, "value") else str(process_state or "created")
    return {"pk": int(node.pk), "uuid": str(node.uuid), "state": state}


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
        return ValidationResponse(success=result["success"], message=result["message"], errors=result["errors"])

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


@submission_router.post("/submit")
def submission_submit(payload: dict[str, Any]) -> dict[str, Any]:
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


@submission_router.post("/draft-builder")
def submission_draft_builder(payload: BuilderDraftRequest) -> dict[str, Any]:
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


@submission_router.post("/submit-builder")
def submission_submit_builder(payload: BuilderSubmitRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    return _submit_workchain_builder(payload.draft)


@submission_router.post("/generate-script")
def submission_generate_script(payload: SubmissionScriptRequest) -> dict[str, Any]:
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
