from __future__ import annotations

import inspect
import json
from typing import Any, Mapping, Sequence, get_args, get_origin

from fastapi import HTTPException
from aiida import orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.engine import submit
from aiida.engine.processes.ports import InputPort, PortNamespace
from aiida.plugins import WorkflowFactory

from core.engine import http_error
from core.utils import serialize_spec, to_jsonable, type_to_string


_MISSING = object()
_RESERVED_BUILDER_KEYS = {
    "entry_point",
    "workchain",
    "protocol",
    "overrides",
    "intent_data",
    "draft",
    "inputs",
    "structure_pk",
    "code",
}


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
    if not isinstance(value, (int, str)):
        return False
    # If the value is an empty string, it's not a valid candidate for PK UUID resolution
    if isinstance(value, str) and not value.strip():
        return False
    if not _expects_node(port):
        return False
    if isinstance(value, int) and value <= 0:
        return False
    return True


def _resolve_node_reference(
    port: InputPort | PortNamespace,
    value: Any,
    path: Sequence[str],
    *,
    allow_scalar_pk_resolution: bool = True,
) -> Any:
    # Handle dictionary representation of a Node payload (having 'pk' or 'uuid')
    if isinstance(value, Mapping):
        if _expects_node(port) and ("pk" in value or "uuid" in value) and set(value.keys()).issubset({"pk", "uuid", "type", "label", "node_type"}):
            raw_pk = value.get("pk") or value.get("uuid")
            if raw_pk:
                try:
                    return orm.load_node(raw_pk)
                except Exception as exc:  # noqa: BLE001
                    joined = ".".join(path)
                    raise ValueError(f"Could not load node for '{joined}' with id={raw_pk}: {exc}") from exc
        return value

    # Automatically map scalar ints/strings into AiiDA nodes if the port strictly expects a Node
    if _is_node_pk_candidate(port, value, allow_scalar_pk_resolution=allow_scalar_pk_resolution):
        try:
            # Code nodes are often referenced by "code@computer" labels, which need
            # load_code handling instead of plain load_node resolution.
            valid_types = _extract_valid_types(port)
            expects_code = any(issubclass(t, orm.AbstractCode) for t in valid_types if isinstance(t, type))
            expects_group = any(issubclass(t, orm.Group) for t in valid_types if isinstance(t, type))
            
            if expects_code and isinstance(value, str):
                try:
                    loaded_node = orm.load_code(value)
                except Exception:
                    loaded_node = orm.load_node(value)
            elif expects_group and isinstance(value, str):
                try:
                    loaded_node = orm.load_group(value)
                except Exception:
                    try:
                        loaded_node = orm.Group.collection.get(label=value)
                    except Exception:
                        loaded_node = orm.load_node(value)
            else:
                loaded_node = orm.load_node(value)
        except Exception as exc:  # noqa: BLE001
            # If resolution fails, and the port actually supports scalar DB types (like Int/Str/List)
            # return the raw scalar cleanly instead of crashing as it might just be a standard python value
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


def resolve_generic_inputs(
    namespace: PortNamespace,
    raw_inputs: Mapping[str, Any],
    path: Sequence[str] = ("inputs",),
    *,
    allow_scalar_pk_resolution: bool = True,
) -> dict[str, Any]:
    """
    Recursively traverse raw workflow inputs. Automatically load AiiDA nodes via UUID or PK 
    if the target port is determined to expect an AiiDA Node instead of primitive maps.
    """
    resolved: dict[str, Any] = {}

    for key, value in raw_inputs.items():
        port = namespace.get(key)
        child_path = (*path, key)

        if isinstance(port, PortNamespace) and isinstance(value, Mapping):
            resolved[key] = resolve_generic_inputs(
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



def _prepare_and_validate(process: Any, raw_inputs: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    spec_inputs = process.spec().inputs
    resolved_inputs = resolve_generic_inputs(spec_inputs, raw_inputs)

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
        resolved_parameters = resolve_generic_inputs(
            spec_inputs,
            parameters,
            allow_scalar_pk_resolution=False,
        )
        resolved_pk_inputs = resolve_generic_inputs(
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

        sanitized[key] = value

    return sanitized, errors


def _collect_required_port_paths(
    namespace: PortNamespace, 
    payload: Mapping[str, Any], 
    prefix: Sequence[str] = ()
) -> list[tuple[str, ...]]:
    required: list[tuple[str, ...]] = []
    
    for key, port in namespace.items():
        path = (*prefix, str(key))
        if path and path[0] == "metadata":
            continue
            
        port_present = key in payload and payload[key] is not None
        
        if isinstance(port, PortNamespace):
            # If the namespace is required OR it is present in payload, check its children
            is_required = bool(getattr(port, "required", False))
            if is_required or port_present:
                sub_payload = payload.get(key)
                if not isinstance(sub_payload, Mapping):
                    sub_payload = {}
                required.extend(_collect_required_port_paths(port, sub_payload, path))
        else:
            # Atomic port
            is_required = bool(getattr(port, "required", False))
            if is_required:
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
    required_paths = _collect_required_port_paths(namespace, payload)
    
    for path in required_paths:
        if not _path_exists(payload, path):
            missing.append(".".join(path))
    return missing


def _infer_resource_domain(*values: Any) -> str | None:
    joined = " ".join(str(value or "") for value in values).strip().lower()
    if not joined:
        return None
    if "entry point" in joined or "workflowfactory" in joined:
        return "entry_point"
    if "computer" in joined or "scheduler" in joined:
        return "computer"
    if "code" in joined:
        return "code"
    if "group" in joined:
        return "group"
    if "pseudo" in joined or "upf" in joined:
        return "pseudo"
    if "structure" in joined:
        return "structure"
    if "node" in joined or "pk=" in joined or "uuid" in joined:
        return "node"
    return None


def _build_recovery_plan(
    *,
    entry_point: str,
    errors: Sequence[Mapping[str, Any]],
    missing_ports: Sequence[str],
    required_ports: Sequence[str],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    issue_keys: set[tuple[str, str, str]] = set()
    resource_domains: set[str] = set()

    def append_issue(
        issue_type: str,
        *,
        message: str,
        port: str | None = None,
        stage: str | None = None,
        resource_domain: str | None = None,
    ) -> None:
        normalized_message = str(message or "").strip()
        if not normalized_message:
            return
        key = (issue_type, str(port or "").strip(), normalized_message)
        if key in issue_keys:
            return
        issue_keys.add(key)
        issue: dict[str, Any] = {
            "type": issue_type,
            "message": normalized_message,
        }
        if port:
            issue["port"] = port
        if stage:
            issue["stage"] = stage
        if resource_domain:
            issue["resource_domain"] = resource_domain
            resource_domains.add(resource_domain)
        issues.append(issue)

    if missing_ports:
        preview = ", ".join(str(port) for port in missing_ports[:6])
        if len(missing_ports) > 6:
            preview += ", ..."
        append_issue(
            "missing_required_inputs",
            message=f"Required inputs are still missing after builder construction: {preview}",
            stage="missing_required_ports",
        )

    for error in errors:
        stage = str(error.get("stage") or "").strip()
        port = str(error.get("port") or "").strip() or None
        message = str(error.get("message") or error.get("reason") or error).strip()
        lowered = message.lower()
        resource_domain = _infer_resource_domain(port, message)

        if stage == "missing_protocol_argument":
            append_issue("missing_protocol_argument", message=message, port=port, stage=stage, resource_domain=resource_domain)
            continue
        if stage in {"resolve_protocol_argument", "resolve_overrides"}:
            issue_type = "resource_reference_unresolved" if "could not load" in lowered or "not found" in lowered else "input_resolution_error"
            append_issue(issue_type, message=message, port=port, stage=stage, resource_domain=resource_domain)
            continue
        if stage == "unsupported_overrides":
            append_issue("unsupported_override", message=message, port=port, stage=stage)
            continue
        if "missing entry point" in lowered or resource_domain == "entry_point":
            append_issue("entry_point_unavailable", message=message, port=port, stage=stage or "load_workflow", resource_domain="entry_point")
            continue
        if "could not load" in lowered or "not found" in lowered or "does not exist" in lowered or "no such" in lowered:
            append_issue("resource_reference_unresolved", message=message, port=port, stage=stage, resource_domain=resource_domain)
            continue
        if stage == "validate":
            append_issue("validation_error", message=message, port=port, stage=stage, resource_domain=resource_domain)
            continue
        append_issue("builder_construction_error", message=message, port=port, stage=stage, resource_domain=resource_domain)

    recommended_actions: list[dict[str, Any]] = []
    seen_actions: set[str] = set()

    def append_action(action: str, reason: str) -> None:
        normalized_action = str(action or "").strip()
        normalized_reason = str(reason or "").strip()
        if not normalized_action or normalized_action in seen_actions:
            return
        seen_actions.add(normalized_action)
        recommended_actions.append({"action": normalized_action, "reason": normalized_reason})

    issue_types = {issue["type"] for issue in issues}
    if issue_types & {
        "missing_required_inputs",
        "missing_protocol_argument",
        "input_resolution_error",
        "unsupported_override",
        "validation_error",
    }:
        append_action(
            "inspect_spec",
            "Review the WorkChain spec and builder signature before changing inputs.",
        )
    if "entry_point_unavailable" in issue_types:
        append_action(
            "inspect_available_workchains",
            "Confirm that the requested WorkChain entry point is installed in the current worker.",
        )
    if resource_domains & {"code", "computer"}:
        append_action(
            "inspect_resources",
            "Check the active AiiDA profile for matching computers and codes before retrying.",
        )
    if resource_domains & {"group", "pseudo", "structure", "node"} or missing_ports:
        append_action(
            "inspect_database_inputs",
            "Check whether the required nodes, groups, or data objects exist in the current profile.",
        )
    if issues:
        append_action(
            "ask_user",
            "Do not substitute missing inputs or resources silently; confirm the user's preferred fix.",
        )
        append_action(
            "stop_if_unresolved",
            "If the required input or resource cannot be found, stop the submission path and report the blocker.",
        )

    summary_parts: list[str] = []
    if missing_ports:
        summary_parts.append(f"Missing required inputs: {', '.join(str(port) for port in missing_ports[:5])}")
    elif "entry_point_unavailable" in issue_types:
        summary_parts.append("The requested WorkChain is not available in the current worker.")
    elif resource_domains:
        summary_parts.append("One or more referenced AiiDA resources could not be resolved.")
    elif issues:
        summary_parts.append(str(issues[0].get("message") or "Builder validation failed."))
    else:
        summary_parts.append("Review builder diagnostics before retrying.")

    if required_ports:
        summary_parts.append(f"Required spec ports: {', '.join(str(port) for port in required_ports[:6])}")

    return {
        "status": "blocked" if issues else "ready",
        "entry_point": entry_point,
        "summary": " ".join(summary_parts),
        "issues": issues,
        "missing_ports": list(missing_ports),
        "required_ports": list(required_ports),
        "resource_domains": sorted(resource_domains),
        "recommended_actions": recommended_actions,
        "user_decision_required": bool(issues),
    }


def _failure_message_from_http_exception(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, Mapping):
        return str(detail.get("reason") or detail.get("error") or detail.get("message") or exc)
    return str(detail or exc)


def _build_failed_protocol_builder_result(
    *,
    entry_point: str,
    protocol: str | None,
    intent_data: Mapping[str, Any],
    overrides: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_errors = [dict(item) for item in errors]
    recovery_plan = _build_recovery_plan(
        entry_point=entry_point,
        errors=normalized_errors,
        missing_ports=[],
        required_ports=[],
    )
    return {
        "success": False,
        "entry_point": entry_point,
        "protocol": protocol,
        "intent_data": to_jsonable(intent_data),
        "overrides": to_jsonable(overrides),
        "protocol_kwargs": {},
        "signature": [],
        "pseudo_expectations": [],
        "required_ports": [],
        "missing_ports": [],
        "builder": None,
        "builder_inputs": {},
        "errors": normalized_errors,
        "recovery_plan": recovery_plan,
    }


def _coerce_builder_argument(
    name: str,
    raw_value: Any,
    parameter: inspect.Parameter,
    spec_inputs: PortNamespace,
) -> Any:
    normalized_name = name.lower()
    port = spec_inputs.get(name)

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
    elif port is not None:
        valid_types = _extract_valid_types(port)
        if any(issubclass(t, orm.Node) for t in valid_types if isinstance(t, type)):
            expected_node_types = tuple(t for t in valid_types if isinstance(t, type))

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
    try:
        workchain = _load_workflow(entry_point)
    except HTTPException as exc:
        return _build_failed_protocol_builder_result(
            entry_point=entry_point,
            protocol=protocol,
            intent_data=intent_data,
            overrides=overrides,
            errors=[
                {
                    "stage": "load_workflow",
                    "port": entry_point,
                    "message": _failure_message_from_http_exception(exc),
                }
            ],
        )

    try:
        parameters = _inspect_builder_protocol_signature(workchain)
    except HTTPException as exc:
        return _build_failed_protocol_builder_result(
            entry_point=entry_point,
            protocol=protocol,
            intent_data=intent_data,
            overrides=overrides,
            errors=[
                {
                    "stage": "inspect_protocol_signature",
                    "port": entry_point,
                    "message": _failure_message_from_http_exception(exc),
                }
            ],
        )

    spec_inputs = workchain.spec().inputs
    errors: list[dict[str, Any]] = []
    required_ports = _list_missing_required_ports(spec_inputs, {})

    sanitized_overrides, override_errors = _sanitize_overrides_for_spec(spec_inputs, overrides)
    errors.extend(override_errors)

    try:
        resolved_overrides = resolve_generic_inputs(
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
            builder_inputs = builder._inputs(prune=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "stage": "get_builder_from_protocol",
                    "message": str(exc),
                }
            )

    if not builder_inputs:
        try:
            fallback_builder = workchain.get_builder()
            builder_inputs = fallback_builder._inputs(prune=True)
        except Exception:
            pass

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

    recovery_plan = _build_recovery_plan(
        entry_point=entry_point,
        errors=errors,
        missing_ports=missing_ports,
        required_ports=required_ports,
    )

    return {
        "success": len(errors) == 0,
        "entry_point": entry_point,
        "protocol": protocol,
        "intent_data": to_jsonable(intent_data),
        "overrides": to_jsonable(resolved_overrides),
        "protocol_kwargs": to_jsonable(kwargs),
        "signature": signature_payload,
        "pseudo_expectations": [],  # Generic WorkChains don't have QE specific hardcoding
        "required_ports": required_ports,
        "missing_ports": missing_ports,
        "builder": builder,
        "builder_inputs": to_jsonable(builder_inputs),
        "errors": errors,
        "recovery_plan": recovery_plan,
    }


def _draft_workchain_builder(draft_data: Mapping[str, Any]) -> dict[str, Any]:
    result = _build_dynamic_protocol_builder(draft_data)
    if not result["success"]:
        return {
            "success": False,
            "status": "DRAFT_INVALID",
            "entry_point": result["entry_point"],
            "protocol": result["protocol"],
            "intent_data": result["intent_data"],
            "overrides": result["overrides"],
            "errors": result["errors"],
            "missing_ports": result["missing_ports"],
            "signature": result["signature"],
            "pseudo_expectations": result["pseudo_expectations"],
            "builder_inputs": result.get("builder_inputs", {}),
            "recovery_plan": result["recovery_plan"],
        }

    return {
        "success": True,
        "status": "DRAFT_READY",
        "entry_point": result["entry_point"],
        "protocol": result["protocol"],
        "intent_data": result["intent_data"],
        "overrides": result["overrides"],
        "signature": result["signature"],
        "pseudo_expectations": result["pseudo_expectations"],
        "builder_inputs": result.get("builder_inputs", {}),
        "recovery_plan": result["recovery_plan"],
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
            recovery_plan=result["recovery_plan"],
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
            "builder_inputs": result.get("builder_inputs", {}),
            "recovery_plan": result["recovery_plan"],
        }

    return {
        "success": True,
        "message": "Builder draft validated",
        "errors": [],
        "entry_point": result["entry_point"],
        "signature": result["signature"],
        "pseudo_expectations": result["pseudo_expectations"],
        "builder_inputs": result.get("builder_inputs", {}),
        "recovery_plan": result["recovery_plan"],
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
        "def _coerce_argument(name: str, value: Any, port: Any = None) -> Any:\n"
        "    if value is None:\n"
        "        return None\n"
        "\n"
        "    lower = name.lower()\n"
        "    # Check if port expects Code or Group\n"
        "    expects_code = lower == 'code' or lower.endswith('_code')\n"
        "    expects_group = False\n"
        "\n"
        "    if port is not None and hasattr(port, 'valid_type'):\n"
        "        vt = port.valid_type\n"
        "        if vt:\n"
        "            types = vt if isinstance(vt, tuple) else (vt,)\n"
        "            if any(issubclass(t, orm.AbstractCode) for t in types if isinstance(t, type)):\n"
        "                expects_code = True\n"
        "            if any(issubclass(t, orm.Group) for t in types if isinstance(t, type)):\n"
        "                expects_group = True\n"
        "\n"
        "    if expects_code:\n"
        "        try:\n"
        "            return orm.load_code(value)\n"
        "        except Exception:\n"
        "            return orm.load_node(value)\n"
        "\n"
        "    if expects_group and isinstance(value, str):\n"
        "        try:\n"
        "            return orm.load_group(value)\n"
        "        except Exception:\n"
        "            return orm.Group.collection.get(label=value)\n"
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
        "spec_inputs = wc_class.spec().inputs\n"
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
        "    port = spec_inputs.get(name)\n"
        "    kwargs[name] = _coerce_argument(name, raw, port=port)\n"
        "\n"
        "builder = wc_class.get_builder_from_protocol(**kwargs)\n"
        "builder_inputs = builder._inputs(prune=True)\n"
        "missing = [\n"
        "    '.'.join(path)\n"
        "    for path in _collect_required_ports(spec_inputs)\n"
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
