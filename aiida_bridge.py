from __future__ import annotations

import io
import os
import traceback
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aiida import load_profile, orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.engine import submit
from aiida.engine.daemon.client import get_daemon_client
from aiida.engine.processes.ports import InputPort, PortNamespace
from aiida.manage.configuration import get_config
from aiida.manage.manager import get_manager
from aiida.orm import Group, Node, ProcessNode, QueryBuilder
from aiida.plugins import WorkflowFactory
from aiida.plugins.entry_point import get_entry_point_names
from aiida.storage.sqlite_zip.backend import SqliteZipBackend

PROFILE_NAME = os.getenv("AIIDA_PROFILE", "sandbox")
_PROFILE_LOADED = False
_ACTIVE_PROFILE_NAME = ""
_CURRENT_MOUNTED_ARCHIVE: str | None = None


# -----------------------------------------------------------------------------
# Error + profile helpers
# -----------------------------------------------------------------------------

def _http_error(status_code: int, error: str, **extra: Any) -> HTTPException:
    payload: dict[str, Any] = {"error": error}
    payload.update(extra)
    if _ACTIVE_PROFILE_NAME and "profile" not in payload:
        payload["profile"] = _ACTIVE_PROFILE_NAME
    return HTTPException(status_code=status_code, detail=payload)


def _ensure_profile_loaded() -> None:
    """Load configured AiiDA profile once with robust fallback behavior."""
    global _ACTIVE_PROFILE_NAME, _PROFILE_LOADED

    if _PROFILE_LOADED:
        return

    configured_profile = PROFILE_NAME.strip()
    candidates: list[str | None] = [configured_profile or "sandbox"]
    if "sandbox" not in candidates:
        candidates.append("sandbox")
    candidates.append(None)  # fallback to AiiDA configured default

    errors: list[dict[str, str]] = []
    for candidate in candidates:
        label = candidate or "<aiida-default>"
        try:
            profile = load_profile(candidate)
            _ACTIVE_PROFILE_NAME = str(getattr(profile, "name", candidate or "default"))
            _PROFILE_LOADED = True
            return
        except Exception as exc:  # noqa: BLE001
            errors.append({"profile": label, "error": str(exc)})

    raise _http_error(500, "Failed to load any AiiDA profile", errors=errors)


def _switch_profile(profile_name: str) -> str:
    global _ACTIVE_PROFILE_NAME, _PROFILE_LOADED, _CURRENT_MOUNTED_ARCHIVE

    cleaned = str(profile_name or "").strip()
    if not cleaned:
        raise _http_error(400, "Profile name is required")

    try:
        profile = load_profile(cleaned, allow_switch=True)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(400, "Failed to switch profile", target=cleaned, reason=str(exc)) from exc

    _ACTIVE_PROFILE_NAME = str(getattr(profile, "name", cleaned))
    _PROFILE_LOADED = True
    _CURRENT_MOUNTED_ARCHIVE = None
    return _ACTIVE_PROFILE_NAME


def _load_archive_profile(filepath: str) -> str:
    global _ACTIVE_PROFILE_NAME, _PROFILE_LOADED, _CURRENT_MOUNTED_ARCHIVE

    source = str(filepath or "").strip()
    if not source:
        raise _http_error(400, "Archive path is required")

    archive = Path(source).expanduser()
    if not archive.exists() or not archive.is_file():
        raise _http_error(404, "Archive file not found", path=str(archive))

    if archive.suffix.lower() not in {".aiida", ".zip"}:
        raise _http_error(400, "Unsupported archive format", path=str(archive))

    try:
        archive_profile = SqliteZipBackend.create_profile(filepath=str(archive))
        profile = load_profile(archive_profile, allow_switch=True)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(400, "Failed to load archive profile", path=str(archive), reason=str(exc)) from exc

    _ACTIVE_PROFILE_NAME = str(getattr(profile, "name", archive.stem))
    _PROFILE_LOADED = True
    _CURRENT_MOUNTED_ARCHIVE = str(archive)
    return _ACTIVE_PROFILE_NAME


def _active_profile_name() -> str:
    if _ACTIVE_PROFILE_NAME:
        return _ACTIVE_PROFILE_NAME
    try:
        manager = get_manager()
        profile = manager.get_profile()
        if profile is not None:
            return str(profile.name)
    except Exception:  # noqa: BLE001
        pass
    return PROFILE_NAME.strip() or "unknown"


def _list_profiles_payload() -> dict[str, Any]:
    _ensure_profile_loaded()
    try:
        config = get_config()
        default_profile = str(config.default_profile_name or "") or None
        raw_profiles = config.profiles
        if isinstance(raw_profiles, Mapping):
            profile_names = sorted([str(name) for name in raw_profiles.keys()])
        else:
            profile_names = sorted([str(getattr(profile, "name", profile)) for profile in raw_profiles])
    except Exception as exc:  # noqa: BLE001
        raise _http_error(500, "Failed to list configured profiles", reason=str(exc)) from exc

    current_profile = _active_profile_name()
    profiles = [
        {
            "name": name,
            "is_default": bool(default_profile and name == default_profile),
            "is_active": bool(name == current_profile),
        }
        for name in profile_names
    ]
    return {
        "current_profile": current_profile,
        "default_profile": default_profile,
        "profiles": profiles,
        "mounted_archive": _CURRENT_MOUNTED_ARCHIVE,
    }


# -----------------------------------------------------------------------------
# JSON helpers + spec serialization
# -----------------------------------------------------------------------------

def _type_to_string(value: Any) -> str:
    if value is None:
        return "Any"
    if isinstance(value, tuple):
        return " | ".join(_type_to_string(entry) for entry in value)
    if isinstance(value, type):
        return value.__name__
    return str(value)


def _to_jsonable(value: Any) -> Any:
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
        return _to_jsonable(port.default)
    except Exception:  # noqa: BLE001
        return None


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


def serialize_spec(port_or_namespace: InputPort | PortNamespace) -> dict[str, Any]:
    """
    Recursively serialize an AiiDA input port/namespace into a JSON-safe structure.
    Port names are represented by keys in the parent ``ports`` mapping.
    """
    payload: dict[str, Any] = {
        "required": _extract_required(port_or_namespace),
        "default": _extract_default(port_or_namespace),
        "help": _extract_help(port_or_namespace),
    }

    if isinstance(port_or_namespace, PortNamespace):
        payload["type"] = "PortNamespace"
        payload["dynamic"] = bool(getattr(port_or_namespace, "dynamic", False))
        payload["ports"] = {
            child_name: serialize_spec(child_port)
            for child_name, child_port in port_or_namespace.items()
        }
    else:
        payload["type"] = _type_to_string(getattr(port_or_namespace, "valid_type", None))
        payload["non_db"] = bool(getattr(port_or_namespace, "non_db", False))

    return payload


# -----------------------------------------------------------------------------
# Submission input normalization helpers
# -----------------------------------------------------------------------------

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
    if not isinstance(value, int):
        return False
    if not _expects_node(port):
        return False
    return value > 0


def _resolve_node_reference(port: InputPort | PortNamespace, value: Any, path: Sequence[str]) -> Any:
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

    if _is_node_pk_candidate(port, value):
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


def _prepare_and_validate(process: Any, raw_inputs: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    spec_inputs = process.spec().inputs
    resolved_inputs = _resolve_inputs_for_namespace(spec_inputs, raw_inputs)

    try:
        processed_inputs = spec_inputs.pre_process(dict(resolved_inputs))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to pre-process inputs: {exc}") from exc

    validation_error = spec_inputs.validate(processed_inputs)
    return dict(processed_inputs), validation_error


def _load_workflow(entry_point: str) -> Any:
    cleaned = str(entry_point or "").strip()
    if not cleaned:
        raise _http_error(400, "Workflow entry point is required")

    try:
        return WorkflowFactory(cleaned)
    except MissingEntryPointError as exc:
        raise _http_error(404, "Workflow entry point not found", entry_point=cleaned, reason=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _http_error(400, "Failed to load workflow entry point", entry_point=cleaned, reason=str(exc)) from exc


# -----------------------------------------------------------------------------
# System, management, and node-level serialization helpers
# -----------------------------------------------------------------------------

def _is_daemon_running() -> bool:
    try:
        daemon_client = get_daemon_client()
        return bool(getattr(daemon_client, "is_daemon_running", False))
    except Exception:  # noqa: BLE001
        return False


def _serialize_computers() -> list[dict[str, Any]]:
    computers = sorted(orm.Computer.collection.all(), key=lambda computer: computer.label.lower())
    return [
        {
            "label": str(computer.label),
            "hostname": str(computer.hostname),
            "description": str(computer.description) if computer.description else None,
        }
        for computer in computers
    ]


def _serialize_codes() -> list[dict[str, Any]]:
    codes = sorted(orm.Code.collection.all(), key=lambda code: code.label.lower())
    payload: list[dict[str, Any]] = []
    for code in codes:
        computer_label = None
        try:
            computer = code.computer
            computer_label = str(computer.label) if computer is not None else None
        except Exception:  # noqa: BLE001
            computer_label = None

        payload.append(
            {
                "label": str(code.label),
                "default_plugin": str(getattr(code, "default_calc_job_plugin", "") or "") or None,
                "computer_label": computer_label,
            }
        )
    return payload


def _query_count(cls: type[Any], *, filters: dict[str, Any] | None = None) -> int:
    qb = QueryBuilder()
    qb.append(cls, filters=filters or {})
    return int(qb.count())


def _collect_system_counts() -> dict[str, int]:
    try:
        groups_count = _query_count(Group)
    except Exception:  # noqa: BLE001
        groups_count = 0

    try:
        nodes_count = _query_count(Node)
    except Exception:  # noqa: BLE001
        nodes_count = 0

    try:
        processes_count = _query_count(ProcessNode)
    except Exception:  # noqa: BLE001
        processes_count = 0

    try:
        failed_count = _query_count(ProcessNode, filters={"exit_status": {"!==": 0}})
    except Exception:  # noqa: BLE001
        failed_count = 0

    return {
        "computers": len(orm.Computer.collection.all()),
        "codes": len(orm.Code.collection.all()),
        "workchains": len(get_entry_point_names("aiida.workflows")),
        "groups": groups_count,
        "nodes": nodes_count,
        "processes": processes_count,
        "failed_processes": failed_count,
    }


def _get_system_info_payload() -> dict[str, Any]:
    counts = _collect_system_counts()
    return {
        "profile": _active_profile_name(),
        "counts": {
            "computers": counts["computers"],
            "codes": counts["codes"],
            "workchains": counts["workchains"],
        },
        "daemon_status": _is_daemon_running(),
    }


def _get_statistics_payload() -> dict[str, Any]:
    counts = _collect_system_counts()
    return {
        "profile": _active_profile_name(),
        "daemon_status": _is_daemon_running(),
        "counts": counts,
    }


def _get_database_summary_payload() -> dict[str, Any]:
    counts = _collect_system_counts()
    return {
        "status": "success",
        "profile": _active_profile_name(),
        "node_count": counts["nodes"],
        "process_count": counts["processes"],
        "failed_count": counts["failed_processes"],
    }


def _node_type_name(node: orm.Node) -> str:
    node_type = str(getattr(node, "node_type", node.__class__.__name__))
    if "." in node_type:
        parts = node_type.split(".")
        if len(parts) >= 2:
            return parts[-2]
    return str(node.__class__.__name__)


def _extract_node_payload(node: orm.Node) -> Any:
    payload = None
    try:
        if isinstance(node, orm.Dict):
            payload = node.get_dict()
        elif isinstance(node, orm.FolderData):
            payload = node.list_object_names()
        elif isinstance(node, orm.StructureData):
            payload = node.get_formula()
        elif isinstance(node, orm.BandsData):
            payload = "BandsStructure"
        elif isinstance(node, orm.Code):
            payload = node.full_label
        elif isinstance(node, (orm.Int, orm.Float, orm.Str, orm.Bool)):
            payload = node.value
        elif isinstance(node, orm.KpointsData):
            try:
                mesh, offset = node.get_kpoints_mesh()
                payload = {
                    "mode": "mesh",
                    "mesh": _to_jsonable(mesh),
                    "offset": _to_jsonable(offset),
                }
            except Exception:
                with suppress(Exception):
                    kpoints = node.get_kpoints()
                    payload = {
                        "mode": "list",
                        "num_points": len(kpoints),
                        "points": _to_jsonable(kpoints.tolist()),
                    }
    except Exception:  # noqa: BLE001
        payload = "Error loading content"

    return _to_jsonable(payload)


def _serialize_node(node: orm.Node) -> dict[str, Any]:
    info: dict[str, Any] = {
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "type": _node_type_name(node),
        "full_type": str(getattr(node, "node_type", node.__class__.__name__)),
        "label": str(getattr(node, "label", None) or "N/A"),
        "ctime": node.ctime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "ctime", None) else None,
    }

    payload = _extract_node_payload(node)
    if payload is not None:
        info["payload"] = payload

    if isinstance(node, orm.ProcessNode):
        process_state = getattr(node, "process_state", None)
        info["state"] = process_state.value if hasattr(process_state, "value") else str(process_state or "unknown")
        info["exit_status"] = getattr(node, "exit_status", None)
        info["process_label"] = str(getattr(node, "process_label", None) or "N/A")

    return info


def _get_node_summary(node_pk: int) -> dict[str, Any]:
    try:
        node = orm.load_node(node_pk)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(404, "Node not found", pk=node_pk, reason=str(exc)) from exc

    try:
        incoming_count = len(node.base.links.get_incoming().all())
    except Exception:  # noqa: BLE001
        incoming_count = 0

    try:
        outgoing_count = len(node.base.links.get_outgoing().all())
    except Exception:  # noqa: BLE001
        outgoing_count = 0

    process_state = getattr(node, "process_state", None)
    state_value = process_state.value if hasattr(process_state, "value") else str(process_state or "N/A")

    return {
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "type": _node_type_name(node),
        "full_type": str(getattr(node, "node_type", node.__class__.__name__)),
        "ctime": node.ctime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "ctime", None) else None,
        "label": str(getattr(node, "label", None) or "(No Label)"),
        "state": state_value,
        "exit_status": getattr(node, "exit_status", "N/A"),
        "incoming": incoming_count,
        "outgoing": outgoing_count,
        "attributes": _to_jsonable(node.base.attributes.all),
    }


def _get_structure_formula(node: orm.StructureData) -> str | None:
    for method_name in ("get_formula", "get_chemical_formula"):
        method = getattr(node, method_name, None)
        if not callable(method):
            continue
        try:
            formula = method()
        except TypeError:
            formula = method(mode="hill")
        except Exception:
            continue
        if formula:
            return str(formula)
    return None


_NODE_CLASS_MAP: dict[str, type[Node]] = {
    "ProcessNode": ProcessNode,
    "WorkChainNode": orm.WorkChainNode,
    "StructureData": orm.StructureData,
}


def _resolve_node_class(node_type: str | None) -> type[Node]:
    if not node_type or not str(node_type).strip():
        return Node

    normalized = str(node_type).strip()
    if normalized in _NODE_CLASS_MAP:
        return _NODE_CLASS_MAP[normalized]

    for known_name, node_class in _NODE_CLASS_MAP.items():
        if normalized.lower() == known_name.lower():
            return node_class

    dynamic_node_class = getattr(orm, normalized, None)
    if isinstance(dynamic_node_class, type) and issubclass(dynamic_node_class, Node):
        return dynamic_node_class

    supported = ", ".join(_NODE_CLASS_MAP.keys())
    raise _http_error(400, f"Unsupported node_type '{normalized}'", supported=supported)


def _list_group_labels(search_string: str | None = None) -> list[str]:
    qb = QueryBuilder()
    filters = {"label": {"like": f"%{search_string}%"}} if search_string else {}
    qb.append(Group, project=["label", "*"], filters=filters)

    labels: list[str] = []
    for label, group in qb.all():
        if getattr(group, "type_string", "") == "core.import":
            continue
        labels.append(str(label))

    return sorted(set(labels), key=str.lower)


def _list_groups(search: str | None = None) -> list[dict[str, Any]]:
    qb = QueryBuilder()
    filters = {"label": {"like": f"%{search}%"}} if search else {}
    qb.append(Group, project=["label", "id", "*"], filters=filters)

    items: list[dict[str, Any]] = []
    for label, pk, group in qb.all():
        if getattr(group, "type_string", "") == "core.import":
            continue
        items.append(
            {
                "label": str(label),
                "pk": int(pk),
                "count": int(len(group.nodes)),
            }
        )
    return sorted(items, key=lambda item: item["label"].lower())


def _inspect_group(group_name: str, limit: int = 20) -> dict[str, Any]:
    cleaned = str(group_name or "").strip()
    if not cleaned:
        raise _http_error(400, "Group label is required")

    matches = Group.collection.find(filters={"label": cleaned})
    if not matches:
        similar = Group.collection.find(filters={"label": {"like": f"%{cleaned}%"}})
        suggestions = sorted({str(group.label) for group in similar})
        raise _http_error(404, "Group not found", group=cleaned, suggestions=suggestions)

    group = Group.collection.get(label=cleaned)
    nodes = list(group.nodes)[: max(1, int(limit))]

    serialized_nodes: list[dict[str, Any]] = []
    for node in nodes:
        process_state = getattr(node, "process_state", None)
        state_value = process_state.value if hasattr(process_state, "value") else str(process_state or "N/A")
        serialized_nodes.append(
            {
                "pk": int(node.pk),
                "uuid": str(node.uuid),
                "label": str(node.label or _node_type_name(node)),
                "type": _node_type_name(node),
                "full_type": str(node.node_type),
                "process_label": str(getattr(node, "process_label", "N/A") or "N/A"),
                "process_state": state_value,
                "exit_status": getattr(node, "exit_status", None),
                "ctime": node.ctime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "ctime", None) else None,
                "mtime": node.mtime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "mtime", None) else None,
                "attributes": _to_jsonable(node.base.attributes.all),
                "extras": _to_jsonable(node.base.extras.all),
            }
        )

    return {
        "group": {
            "label": str(group.label),
            "pk": int(group.pk),
            "size": int(len(group.nodes)),
            "type_string": str(getattr(group, "type_string", "")),
        },
        "nodes": serialized_nodes,
    }


def _get_recent_processes(limit: int = 15) -> list[dict[str, Any]]:
    qb = QueryBuilder()
    qb.append(ProcessNode, project=["id", "attributes.process_state", "attributes.process_label", "ctime"], tag="process")
    qb.order_by({"process": {"ctime": "desc"}})
    qb.limit(max(1, int(limit)))

    results: list[dict[str, Any]] = []
    for pk, state, label, ctime in qb.all():
        state_value = state.value if hasattr(state, "value") else str(state or "unknown")
        results.append(
            {
                "pk": int(pk),
                "label": str(label or "Unknown Task"),
                "state": state_value,
                "ctime": ctime.strftime("%Y-%m-%d %H:%M:%S") if ctime else None,
            }
        )
    return results


def _get_recent_nodes(limit: int = 15, group_label: str | None = None, node_type: str | None = None) -> list[dict[str, Any]]:
    node_class = _resolve_node_class(node_type)
    qb = QueryBuilder()

    if group_label and str(group_label).strip():
        qb.append(Group, filters={"label": str(group_label).strip()}, tag="group")
        qb.append(node_class, with_group="group", project=["*"], tag="node")
    else:
        qb.append(node_class, project=["*"], tag="node")

    qb.order_by({"node": {"ctime": "desc"}})
    qb.limit(max(1, int(limit)))

    results: list[dict[str, Any]] = []
    for (node,) in qb.all():
        process_state_value: str | None = None
        if isinstance(node, ProcessNode):
            process_state = getattr(node, "process_state", None)
            process_state_value = (
                process_state.value if hasattr(process_state, "value")
                else (str(process_state) if process_state else "unknown")
            )

        formula_value: str | None = _get_structure_formula(node) if isinstance(node, orm.StructureData) else None

        process_label = getattr(node, "process_label", None)
        if isinstance(node, orm.StructureData):
            label = node.label or formula_value or node.__class__.__name__
        else:
            label = process_label or node.label or node.__class__.__name__

        results.append(
            {
                "pk": int(node.pk),
                "state": process_state_value or "unknown",
                "label": str(label),
                "node_type": str(node.__class__.__name__),
                "process_state": process_state_value,
                "formula": formula_value,
            }
        )
    return results


def _get_unified_source_map(target: str | None = None) -> dict[str, Any]:
    if target and str(target).strip():
        cleaned = str(target).strip()
        candidate = Path(cleaned).expanduser()
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in {".aiida", ".zip"}:
            _load_archive_profile(str(candidate))
        else:
            _switch_profile(cleaned)

    groups = _list_groups()
    source_name = _CURRENT_MOUNTED_ARCHIVE if _CURRENT_MOUNTED_ARCHIVE else _active_profile_name()
    source_type = "archive" if _CURRENT_MOUNTED_ARCHIVE else "profile"

    return {
        "name": str(Path(source_name).name if source_type == "archive" else source_name),
        "type": source_type,
        "current_profile": _active_profile_name(),
        "groups": [{"label": item["label"], "pk": item["pk"]} for item in groups],
    }


def _execute_python_script(script: str) -> dict[str, Any]:
    code = str(script or "")
    if not code.strip():
        raise _http_error(400, "Python script is required")

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


# -----------------------------------------------------------------------------
# Process inspection helpers
# -----------------------------------------------------------------------------

class _ProcessTree:
    """Recursively build and serialize process provenance trees."""

    def __init__(self, node: orm.ProcessNode):
        self.node = node
        self.children: dict[str, _ProcessTree] = {}

        if isinstance(node, orm.WorkChainNode):
            subprocesses = sorted(node.called, key=lambda process: process.ctime)
            counts: defaultdict[str, int] = defaultdict(int)
            for sub in subprocesses:
                raw_label = (
                    sub.base.attributes.all.get("metadata_inputs", {})
                    .get("metadata", {})
                    .get("call_link_label")
                )
                if not raw_label:
                    raw_label = getattr(sub, "process_label", "process")
                label = str(raw_label)
                unique_label = f"{label}_{counts[label]}" if counts[label] > 0 else label
                counts[label] += 1
                self.children[unique_label] = _ProcessTree(sub)

    def to_dict(self) -> dict[str, Any]:
        process_state = getattr(self.node, "process_state", None)
        state_value = process_state.value if hasattr(process_state, "value") else str(process_state or "N/A")
        return {
            "pk": int(self.node.pk),
            "process_label": str(getattr(self.node, "process_label", "N/A") or "N/A"),
            "state": state_value,
            "exit_status": getattr(self.node, "exit_status", None),
            "children": {label: child.to_dict() for label, child in self.children.items()},
        }


def _get_process_log_payload(identifier: int | str) -> dict[str, Any]:
    try:
        node = orm.load_node(identifier)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(404, "Process node not found", identifier=str(identifier), reason=str(exc)) from exc

    if not isinstance(node, orm.ProcessNode):
        raise _http_error(400, "Node is not a ProcessNode", identifier=str(identifier))

    lines: list[str] = []
    reports: list[str] = []

    report_rows: list[tuple[str, Any]] = []
    for log_filter_key in ("dbnode_id", "objpk"):
        try:
            qb = QueryBuilder().append(orm.Log, filters={log_filter_key: int(node.pk)}, project=["message", "time"])
            if qb.count() > 0:
                report_rows = list(qb.order_by({orm.Log: {"time": "asc"}}).all())
                break
        except Exception:  # noqa: BLE001
            continue

    if report_rows:
        lines.append("--- Reports ---")
        for message, timestamp in report_rows:
            formatted = f"[{timestamp.strftime('%H:%M:%S')}] {message}"
            reports.append(formatted)
            lines.append(formatted)

    stderr_excerpt = None
    if isinstance(node, orm.CalcJobNode) and "retrieved" in node.outputs:
        with suppress(Exception):
            stderr = node.get_scheduler_stderr()
            if stderr:
                stderr_excerpt = str(stderr)[-2000:]
                lines.append("--- Stderr ---")
                lines.append(stderr_excerpt)

    return {
        "pk": int(node.pk),
        "lines": lines,
        "reports": reports,
        "stderr_excerpt": stderr_excerpt,
        "text": "\n".join(lines) if lines else "No logs found.",
    }


def _inspect_calculation_node(node: orm.ProcessNode) -> dict[str, Any]:
    if not isinstance(node, (orm.CalcJobNode, orm.CalcFunctionNode)):
        return {"error": f"Node {node.pk} is not a calculation node."}

    payload: dict[str, Any] = {
        "summary": _serialize_node(node),
        "inputs": {},
        "outputs": {},
        "repository_files": [],
    }

    incoming_counts: defaultdict[str, int] = defaultdict(int)
    for link in node.base.links.get_incoming().all():
        label = str(link.link_label)
        unique_label = f"{label}_{incoming_counts[label]}" if incoming_counts[label] > 0 else label
        incoming_counts[label] += 1
        payload["inputs"][unique_label] = _serialize_node(link.node)

    outgoing_counts: defaultdict[str, int] = defaultdict(int)
    for link in node.base.links.get_outgoing().all():
        label = str(link.link_label)
        unique_label = f"{label}_{outgoing_counts[label]}" if outgoing_counts[label] > 0 else label
        outgoing_counts[label] += 1
        payload["outputs"][unique_label] = _serialize_node(link.node)

    with suppress(Exception):
        files = node.base.repository.list_object_names()
        payload["repository_files"] = [entry for entry in files if not str(entry).startswith(".aiida")]

    if isinstance(node, orm.CalcJobNode):
        payload["scheduler_info"] = {
            "remote_workdir": node.get_remote_workdir(),
            "stdout_name": node.get_option("output_filename"),
            "stderr_name": node.get_option("error_filename"),
            "has_retrieved": "retrieved" in node.outputs,
        }

    return payload


def _inspect_workchain_node(node: orm.WorkChainNode) -> dict[str, Any]:
    tree = _ProcessTree(node)
    return {"provenance_tree": tree.to_dict()}


def _inspect_process_payload(identifier: str | int) -> dict[str, Any]:
    try:
        node = orm.load_node(identifier)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(404, "Process node not found", identifier=str(identifier), reason=str(exc)) from exc

    if not isinstance(node, orm.ProcessNode):
        raise _http_error(400, "Node is not a ProcessNode", identifier=str(identifier))

    process_state = getattr(node, "process_state", None)
    summary = {
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "type": _node_type_name(node),
        "state": process_state.value if hasattr(process_state, "value") else str(process_state or "unknown"),
        "exit_status": getattr(node, "exit_status", None),
    }

    payload: dict[str, Any] = {
        "summary": summary,
        "logs": _get_process_log_payload(int(node.pk)),
    }

    if isinstance(node, (orm.CalcJobNode, orm.CalcFunctionNode)):
        payload["calculation"] = _inspect_calculation_node(node)

    if isinstance(node, orm.WorkflowNode):
        payload["workchain"] = _inspect_workchain_node(node)

    return payload


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def _get_bands_plot_data(pk: int) -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(404, "Node not found", pk=pk, reason=str(exc)) from exc

    if not isinstance(node, orm.BandsData) and not hasattr(node, "_matplotlib_get_dict"):
        raise _http_error(400, "Node is not a compatible BandsData type", pk=pk)

    try:
        if hasattr(node, "_matplotlib_get_dict"):
            return {
                "pk": int(node.pk),
                "data": _to_jsonable(node._matplotlib_get_dict()),
            }
    except Exception as exc:  # noqa: BLE001
        raise _http_error(500, "Failed to extract bands data", pk=pk, reason=str(exc)) from exc

    raise _http_error(500, "Bands data extraction failed", pk=pk)


def _list_remote_files(pk: int | str) -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(404, "Node not found", pk=str(pk), reason=str(exc)) from exc

    if not isinstance(node, orm.RemoteData):
        raise _http_error(400, "Node is not RemoteData", pk=str(pk), type=node.__class__.__name__)

    try:
        files = node.listdir()
    except Exception as exc:  # noqa: BLE001
        raise _http_error(500, "Failed to list remote files", pk=str(pk), reason=str(exc)) from exc

    return {
        "pk": int(node.pk),
        "files": _to_jsonable(files),
    }


def _get_remote_file_content(pk: int | str, filename: str) -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(404, "Node not found", pk=str(pk), reason=str(exc)) from exc

    if not isinstance(node, orm.RemoteData):
        raise _http_error(400, "Node is not RemoteData", pk=str(pk), type=node.__class__.__name__)

    target = str(filename or "").strip()
    if not target:
        raise _http_error(400, "Filename is required")

    try:
        with TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            node.getfile(target, str(destination))
            content = destination.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise _http_error(500, "Failed to read remote file", pk=str(pk), filename=target, reason=str(exc)) from exc

    return {
        "pk": int(node.pk),
        "filename": target,
        "content": content,
    }


def _get_node_file_content(pk: int | str, filename: str, source: str = "folder") -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(404, "Node not found", pk=str(pk), reason=str(exc)) from exc

    target = str(filename or "").strip()
    if not target:
        raise _http_error(400, "Filename is required")

    mode = str(source or "folder").strip().lower()

    try:
        if mode in {"repository", "virtual.repository"}:
            raw = node.base.repository.get_object_content(target)
        else:
            raw = node.get_object_content(target)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(
            500,
            "Failed to read node file",
            pk=str(pk),
            filename=target,
            source=source,
            reason=str(exc),
        ) from exc

    if isinstance(raw, bytes):
        content = raw.decode("utf-8", errors="replace")
    else:
        content = str(raw)

    return {
        "pk": int(node.pk),
        "filename": target,
        "source": source,
        "content": content,
    }


# -----------------------------------------------------------------------------
# Submission helpers (builder draft/submit migration)
# -----------------------------------------------------------------------------

def _draft_workchain_builder(
    workchain_label: str,
    structure_pk: int,
    code_label: str,
    protocol: str = "moderate",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        workchain = WorkflowFactory(workchain_label)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(400, "Failed to load WorkChain", workchain=workchain_label, reason=str(exc)) from exc

    if not hasattr(workchain, "get_builder_from_protocol"):
        raise _http_error(400, "WorkChain does not support protocols", workchain=workchain_label)

    try:
        code = orm.load_code(code_label)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(400, "Failed to load code", code=code_label, reason=str(exc)) from exc

    try:
        structure = orm.load_node(structure_pk)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(400, "Failed to load structure node", structure_pk=structure_pk, reason=str(exc)) from exc

    draft_overrides = overrides or {}
    try:
        _ = workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol=protocol,
            overrides=draft_overrides,
        )
    except Exception as exc:  # noqa: BLE001
        raise _http_error(
            400,
            "Builder draft failed",
            workchain=workchain_label,
            structure_pk=structure_pk,
            code=code_label,
            protocol=protocol,
            reason=str(exc),
        ) from exc

    return {
        "status": "DRAFT_READY",
        "workchain": workchain_label,
        "structure_pk": int(structure_pk),
        "code": code_label,
        "protocol": protocol,
        "overrides": _to_jsonable(draft_overrides),
        "preview": f"Ready to submit {workchain_label} using {protocol} protocol.",
    }


def _submit_workchain_builder(draft_data: Mapping[str, Any]) -> dict[str, Any]:
    workchain_name = str(draft_data.get("workchain") or "").strip()
    code_label = str(draft_data.get("code") or "").strip()
    protocol = str(draft_data.get("protocol") or "moderate")
    overrides = draft_data.get("overrides") or {}

    try:
        structure_pk = int(draft_data.get("structure_pk"))
    except (TypeError, ValueError) as exc:
        raise _http_error(400, "Draft payload requires integer 'structure_pk'") from exc

    if not workchain_name or not code_label:
        raise _http_error(400, "Draft payload requires 'workchain' and 'code'")

    try:
        workchain = WorkflowFactory(workchain_name)
        builder = workchain.get_builder_from_protocol(
            code=orm.load_code(code_label),
            structure=orm.load_node(structure_pk),
            protocol=protocol,
            overrides=overrides,
        )
        node = submit(builder)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(500, "WorkChain submission failed", reason=str(exc)) from exc

    process_state = getattr(node, "process_state", None)
    state = process_state.value if hasattr(process_state, "value") else str(process_state or "created")
    return {
        "status": "submitted",
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "state": state,
    }


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

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


class SystemCountsResponse(BaseModel):
    computers: int
    codes: int
    workchains: int


class SystemInfoResponse(BaseModel):
    profile: str
    counts: SystemCountsResponse
    daemon_status: bool


class ComputerResource(BaseModel):
    label: str
    hostname: str
    description: str | None = None


class CodeResource(BaseModel):
    label: str
    default_plugin: str | None = None
    computer_label: str | None = None


class ResourcesResponse(BaseModel):
    computers: list[ComputerResource] = Field(default_factory=list)
    codes: list[CodeResource] = Field(default_factory=list)


class ProfileSwitchRequest(BaseModel):
    profile: str


class ArchiveLoadRequest(BaseModel):
    path: str


class ContextNodesRequest(BaseModel):
    ids: list[int] = Field(default_factory=list)


class PythonScriptRequest(BaseModel):
    script: str


class BuilderDraftRequest(BaseModel):
    workchain: str
    structure_pk: int
    code: str
    protocol: str = "moderate"
    overrides: dict[str, Any] = Field(default_factory=dict)


class BuilderSubmitRequest(BaseModel):
    draft: dict[str, Any]


# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------

management_router = APIRouter(prefix="/management", tags=["management"])
process_router = APIRouter(prefix="/process", tags=["process"])
data_router = APIRouter(prefix="/data", tags=["data"])
submission_router = APIRouter(prefix="/submission", tags=["submission"])


@management_router.get("/profiles")
def list_profiles() -> dict[str, Any]:
    return _list_profiles_payload()


@management_router.post("/profiles/switch")
def switch_profile(payload: ProfileSwitchRequest) -> dict[str, Any]:
    _ensure_profile_loaded()
    current = _switch_profile(payload.profile)
    return {"status": "switched", "current_profile": current}


@management_router.post("/profiles/load-archive")
def load_archive_profile(payload: ArchiveLoadRequest) -> dict[str, Any]:
    _ensure_profile_loaded()
    current = _load_archive_profile(payload.path)
    return {
        "status": "loaded",
        "current_profile": current,
        "source": str(Path(payload.path).expanduser()),
    }


@management_router.get("/archives/local")
def list_local_archives(path: str = ".") -> dict[str, Any]:
    _ensure_profile_loaded()
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        raise _http_error(400, "Archive directory does not exist", path=str(root))

    items = [
        str(entry)
        for entry in sorted(root.iterdir())
        if entry.is_file() and entry.suffix.lower() in {".aiida", ".zip"}
    ]
    return {"path": str(root.resolve()), "archives": items}


@management_router.get("/system/info", response_model=SystemInfoResponse)
def management_system_info() -> SystemInfoResponse:
    _ensure_profile_loaded()
    return SystemInfoResponse(**_get_system_info_payload())


@management_router.get("/resources", response_model=ResourcesResponse)
def management_resources() -> ResourcesResponse:
    _ensure_profile_loaded()
    return ResourcesResponse(
        computers=[ComputerResource(**item) for item in _serialize_computers()],
        codes=[CodeResource(**item) for item in _serialize_codes()],
    )


@management_router.get("/statistics")
def management_statistics() -> dict[str, Any]:
    _ensure_profile_loaded()
    return _get_statistics_payload()


@management_router.get("/database/summary")
def management_database_summary() -> dict[str, Any]:
    _ensure_profile_loaded()
    return _get_database_summary_payload()


@management_router.get("/groups")
def management_groups(search: str | None = Query(default=None)) -> dict[str, Any]:
    _ensure_profile_loaded()
    return {"items": _list_groups(search)}


@management_router.get("/groups/labels")
def management_group_labels(search: str | None = Query(default=None)) -> dict[str, Any]:
    _ensure_profile_loaded()
    return {"items": _list_group_labels(search)}


@management_router.get("/groups/{group_name}")
def management_inspect_group(group_name: str, limit: int = Query(default=20, ge=1, le=500)) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _inspect_group(group_name, limit=limit)


@management_router.get("/recent-processes")
def management_recent_processes(limit: int = Query(default=15, ge=1, le=200)) -> dict[str, Any]:
    _ensure_profile_loaded()
    return {"items": _get_recent_processes(limit)}


@management_router.get("/recent-nodes")
def management_recent_nodes(
    limit: int = Query(default=15, ge=1, le=200),
    group_label: str | None = Query(default=None),
    node_type: str | None = Query(default=None),
) -> dict[str, Any]:
    _ensure_profile_loaded()
    return {"items": _get_recent_nodes(limit=limit, group_label=group_label, node_type=node_type)}


@management_router.get("/nodes/{pk}")
def management_node_summary(pk: int) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _get_node_summary(pk)


@management_router.post("/nodes/context")
def management_context_nodes(payload: ContextNodesRequest) -> dict[str, Any]:
    _ensure_profile_loaded()
    items: list[dict[str, Any]] = []
    for raw_pk in payload.ids[:30]:
        try:
            items.append(_get_node_summary(int(raw_pk)))
        except HTTPException as exc:
            error = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
            items.append({"pk": int(raw_pk), **error})
    return {"items": items}


@management_router.get("/source-map")
def management_source_map(target: str | None = Query(default=None)) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _get_unified_source_map(target)


@management_router.post("/run-python")
def management_run_python(payload: PythonScriptRequest) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _execute_python_script(payload.script)


@process_router.get("/{identifier}")
def inspect_process(identifier: str) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _inspect_process_payload(identifier)


@process_router.get("/{identifier}/logs")
def inspect_process_logs(identifier: str) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _get_process_log_payload(identifier)


@data_router.get("/bands/{pk}")
def data_bands(pk: int) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _get_bands_plot_data(pk)


@data_router.get("/remote/{pk}/files")
def data_remote_files(pk: int) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _list_remote_files(pk)


@data_router.get("/remote/{pk}/files/{filename:path}")
def data_remote_file_content(pk: int, filename: str) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _get_remote_file_content(pk, filename)


@data_router.get("/repository/{pk}/files/{filename:path}")
def data_repository_file_content(pk: int, filename: str, source: str = Query(default="folder")) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _get_node_file_content(pk, filename, source=source)


@submission_router.get("/plugins", response_model=list[str])
def submission_plugins() -> list[str]:
    _ensure_profile_loaded()
    return sorted(get_entry_point_names("aiida.workflows"))


@submission_router.get("/spec/{entry_point:path}", response_model=SpecResponse)
def submission_spec(entry_point: str) -> SpecResponse:
    _ensure_profile_loaded()
    process = _load_workflow(entry_point)
    spec_payload = serialize_spec(process.spec().inputs)
    return SpecResponse(entry_point=entry_point, inputs=spec_payload)


@submission_router.post("/validate", response_model=ValidationResponse)
def submission_validate(payload: WorkflowInputsRequest) -> ValidationResponse:
    _ensure_profile_loaded()
    process = _load_workflow(payload.entry_point)

    try:
        _, validation_error = _prepare_and_validate(process, payload.inputs)
    except ValueError as exc:
        raise _http_error(400, "Failed to normalize or pre-process inputs", reason=str(exc)) from exc

    if validation_error is not None:
        raise _http_error(
            422,
            "Validation failed",
            port=str(getattr(validation_error, "port", "") or ""),
            reason=str(getattr(validation_error, "message", "") or str(validation_error)),
            full_error=str(validation_error),
        )

    return ValidationResponse(success=True, message="Success", errors=[])


def _submit_validated_workflow(entry_point: str, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    process = _load_workflow(entry_point)

    try:
        processed_inputs, validation_error = _prepare_and_validate(process, inputs or {})
    except ValueError as exc:
        raise _http_error(400, "Failed to normalize or pre-process inputs", reason=str(exc)) from exc

    if validation_error is not None:
        raise _http_error(
            422,
            "Validation failed",
            port=str(getattr(validation_error, "port", "") or ""),
            reason=str(getattr(validation_error, "message", "") or str(validation_error)),
            full_error=str(validation_error),
        )

    try:
        node = submit(process, **processed_inputs)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(500, "Submission failed", reason=str(exc)) from exc

    process_state = getattr(node, "process_state", None)
    state = process_state.value if hasattr(process_state, "value") else str(process_state or "created")
    return {"pk": int(node.pk), "uuid": str(node.uuid), "state": state}


@submission_router.post("/submit")
def submission_submit(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Unified submission endpoint.
    Supports either:
    - {"entry_point": "...", "inputs": {...}} for direct workflow submission
    - {"draft": {...}} or direct draft payload for protocol builder submission
    """
    _ensure_profile_loaded()
    if not isinstance(payload, dict):
        raise _http_error(400, "Submission payload must be a JSON object")

    if "entry_point" in payload:
        entry_point = str(payload.get("entry_point") or "").strip()
        raw_inputs = payload.get("inputs", {})
        if raw_inputs is None:
            raw_inputs = {}
        if not isinstance(raw_inputs, Mapping):
            raise _http_error(400, "Field 'inputs' must be an object for workflow submission")
        return _submit_validated_workflow(entry_point=entry_point, inputs=raw_inputs)

    draft_payload: Mapping[str, Any] | None = None
    maybe_draft = payload.get("draft")
    if isinstance(maybe_draft, Mapping):
        draft_payload = maybe_draft
    elif all(key in payload for key in ("workchain", "structure_pk", "code")):
        draft_payload = payload

    if draft_payload is not None:
        return _submit_workchain_builder(draft_payload)

    raise _http_error(
        400,
        "Unsupported submission payload",
        expected=[
            {"entry_point": "aiida.workflows:...", "inputs": {}},
            {"draft": {"workchain": "...", "structure_pk": 123, "code": "..."}},
        ],
    )


@submission_router.post("/draft-builder")
def submission_draft_builder(payload: BuilderDraftRequest) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _draft_workchain_builder(
        workchain_label=payload.workchain,
        structure_pk=payload.structure_pk,
        code_label=payload.code,
        protocol=payload.protocol,
        overrides=payload.overrides,
    )


@submission_router.post("/submit-builder")
def submission_submit_builder(payload: BuilderSubmitRequest) -> dict[str, Any]:
    _ensure_profile_loaded()
    return _submit_workchain_builder(payload.draft)


# -----------------------------------------------------------------------------
# App and backward-compatible aliases
# -----------------------------------------------------------------------------

app = FastAPI(
    title="AiiDA Bridge API",
    description="Bridge API exposing AiiDA management, process/data inspection, and submission workflows.",
    version="2.0.0",
)


@app.exception_handler(HTTPException)
async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    payload: dict[str, Any] = {"error": "Internal server error", "reason": str(exc)}
    if _ACTIVE_PROFILE_NAME:
        payload["profile"] = _ACTIVE_PROFILE_NAME
    return JSONResponse(status_code=500, content=payload)


@app.on_event("startup")
def _startup() -> None:
    _ensure_profile_loaded()


app.include_router(management_router)
app.include_router(process_router)
app.include_router(data_router)
app.include_router(submission_router)


# Legacy aliases kept for existing clients.
@app.get("/plugins", response_model=list[str])
def list_workflow_plugins() -> list[str]:
    return submission_plugins()


@app.get("/system/info", response_model=SystemInfoResponse)
def get_system_info() -> SystemInfoResponse:
    return management_system_info()


@app.get("/resources", response_model=ResourcesResponse)
def get_resources() -> ResourcesResponse:
    return management_resources()


@app.get("/statistics")
def get_statistics() -> dict[str, Any]:
    return management_statistics()


@app.get("/spec/{entry_point:path}", response_model=SpecResponse)
def get_workflow_spec(entry_point: str) -> SpecResponse:
    return submission_spec(entry_point)


@app.post("/validate", response_model=ValidationResponse)
def validate_workflow_inputs(payload: WorkflowInputsRequest) -> ValidationResponse:
    return submission_validate(payload)


@app.post("/submit")
def submit_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    return submission_submit(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("aiida_bridge:app", host="127.0.0.1", port=8001, reload=False)
