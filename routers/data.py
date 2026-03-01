from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Query
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from aiida import orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.engine.daemon.client import get_daemon_client
from aiida.orm import Group, Node, ProcessNode, QueryBuilder
from aiida.plugins import DataFactory
from aiida.plugins.entry_point import get_entry_point_names

from core.engine import (
    SessionCleanupAPIRouter,
    active_profile_name,
    current_mounted_archive,
    db_access_guard,
    ensure_profile_loaded,
    http_error,
    load_archive_profile,
    switch_profile,
)
from core.node_utils import (
    build_node_preview,
    extract_process_state_value,
    get_node_summary,
    get_structure_formula,
    node_type_name,
)
from core.utils import to_jsonable
from models.schemas import (
    ArchiveLoadRequest,
    CodeResource,
    ComputerResource,
    ContextNodesRequest,
    ProfileSwitchRequest,
    ResourcesResponse,
    SystemInfoResponse,
)

management_router = SessionCleanupAPIRouter(prefix="/management", tags=["management"])
data_router = SessionCleanupAPIRouter(prefix="/data", tags=["data"])
plugins_router = SessionCleanupAPIRouter(prefix="/submission", tags=["submission"])

_RECENT_NODES_CACHE_TTL_SECONDS = max(0.5, float(os.getenv("AIIDA_RECENT_NODES_CACHE_TTL_SECONDS", "5.0")))
_RECENT_NODES_CACHE_MAX_ITEMS = max(10, int(os.getenv("AIIDA_RECENT_NODES_CACHE_MAX_ITEMS", "128")))
_RECENT_NODES_CACHE_LOCK = threading.Lock()
_RECENT_NODES_CACHE: dict[tuple[int, str | None, str | None], tuple[float, list[dict[str, Any]]]] = {}

_NODE_CLASS_MAP: dict[str, type[Node]] = {
    "ProcessNode": ProcessNode,
    "WorkChainNode": orm.WorkChainNode,
    "StructureData": orm.StructureData,
}

_DATA_ENTRY_POINT_ALIASES: dict[str, str] = {
    "UpfData": "pseudo.upf",
}


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
    with db_access_guard(f"count:{cls.__name__}"):
        return int(qb.count())


def _collect_system_counts() -> dict[str, int]:
    try:
        groups_count = _query_count(Group)
    except SQLAlchemyTimeoutError:
        groups_count = 0
    except Exception:  # noqa: BLE001
        groups_count = 0

    try:
        nodes_count = _query_count(Node)
    except SQLAlchemyTimeoutError:
        nodes_count = 0
    except Exception:  # noqa: BLE001
        nodes_count = 0

    try:
        processes_count = _query_count(ProcessNode)
    except SQLAlchemyTimeoutError:
        processes_count = 0
    except Exception:  # noqa: BLE001
        processes_count = 0

    try:
        failed_count = _query_count(ProcessNode, filters={"exit_status": {"!==": 0}})
    except SQLAlchemyTimeoutError:
        failed_count = 0
    except Exception:  # noqa: BLE001
        failed_count = 0

    try:
        computers_count = _query_count(orm.Computer)
    except SQLAlchemyTimeoutError:
        computers_count = 0
    except Exception:  # noqa: BLE001
        computers_count = 0

    try:
        codes_count = _query_count(orm.Code)
    except SQLAlchemyTimeoutError:
        codes_count = 0
    except Exception:  # noqa: BLE001
        codes_count = 0

    try:
        workchains_count = len(get_entry_point_names("aiida.workflows"))
    except Exception:  # noqa: BLE001
        workchains_count = 0

    return {
        "computers": computers_count,
        "codes": codes_count,
        "workchains": workchains_count,
        "groups": groups_count,
        "nodes": nodes_count,
        "processes": processes_count,
        "failed_processes": failed_count,
    }


def _get_system_info_payload() -> dict[str, Any]:
    counts = _collect_system_counts()
    return {
        "profile": active_profile_name(),
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
        "profile": active_profile_name(),
        "daemon_status": _is_daemon_running(),
        "counts": counts,
    }


def _get_database_summary_payload() -> dict[str, Any]:
    counts = _collect_system_counts()
    return {
        "status": "success",
        "profile": active_profile_name(),
        "node_count": counts["nodes"],
        "process_count": counts["processes"],
        "failed_count": counts["failed_processes"],
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

    normalized_lower = normalized.lower()
    for attr_name in dir(orm):
        if attr_name.lower() != normalized_lower:
            continue
        candidate = getattr(orm, attr_name, None)
        if isinstance(candidate, type) and issubclass(candidate, Node):
            return candidate

    entry_point_candidates: list[str] = []
    aliased = _DATA_ENTRY_POINT_ALIASES.get(normalized)
    if aliased:
        entry_point_candidates.append(aliased)

    if normalized_lower.endswith("data"):
        base = normalized_lower[:-4]
        if base:
            entry_point_candidates.extend([normalized_lower, f"core.{base}"])

    for entry_point in entry_point_candidates:
        with suppress(MissingEntryPointError, ValueError, TypeError, RuntimeError):
            factory_cls = DataFactory(entry_point)
            if isinstance(factory_cls, type) and issubclass(factory_cls, Node):
                return factory_cls

    supported = ", ".join(_NODE_CLASS_MAP.keys())
    raise http_error(400, f"Unsupported node_type '{normalized}'", supported=supported)


def _list_group_labels(search_string: str | None = None) -> list[str]:
    qb = QueryBuilder()
    filters = {"label": {"like": f"%{search_string}%"}} if search_string else {}
    qb.append(Group, project=["label", "*"], filters=filters)

    with db_access_guard("groups-labels"):
        rows = qb.all()

    labels: list[str] = []
    for label, group in rows:
        if getattr(group, "type_string", "") == "core.import":
            continue
        labels.append(str(label))

    return sorted(set(labels), key=str.lower)


def _list_groups(search: str | None = None) -> list[dict[str, Any]]:
    qb = QueryBuilder()
    filters = {"label": {"like": f"%{search}%"}} if search else {}
    qb.append(Group, project=["label", "id", "*"], filters=filters)

    with db_access_guard("groups"):
        rows = qb.all()

    items: list[dict[str, Any]] = []
    for label, pk, group in rows:
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
        raise http_error(400, "Group label is required")

    matches = Group.collection.find(filters={"label": cleaned})
    if not matches:
        similar = Group.collection.find(filters={"label": {"like": f"%{cleaned}%"}})
        suggestions = sorted({str(group.label) for group in similar})
        raise http_error(404, "Group not found", group=cleaned, suggestions=suggestions)

    group = Group.collection.get(label=cleaned)
    nodes = list(group.nodes)[: max(1, int(limit))]

    serialized_nodes: list[dict[str, Any]] = []
    for node in nodes:
        process_state = getattr(node, "process_state", None)
        state_value = process_state.value if hasattr(process_state, "value") else str(process_state or "N/A")
        preview = build_node_preview(node)
        serialized_nodes.append(
            {
                "pk": int(node.pk),
                "uuid": str(node.uuid),
                "label": str(node.label or node_type_name(node)),
                "type": node_type_name(node),
                "full_type": str(node.node_type),
                "process_label": str(getattr(node, "process_label", "N/A") or "N/A"),
                "process_state": state_value,
                "exit_status": getattr(node, "exit_status", None),
                "ctime": node.ctime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "ctime", None) else None,
                "mtime": node.mtime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "mtime", None) else None,
                "attributes": to_jsonable(node.base.attributes.all),
                "extras": to_jsonable(node.base.extras.all),
                "preview_info": preview,
                "preview": preview,
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
    qb.append(
        ProcessNode,
        project=[
            "id",
            "attributes.process_state",
            "attributes.process_label",
            "attributes.exit_status",
            "ctime",
        ],
        tag="process",
    )
    qb.order_by({"process": {"ctime": "desc"}})
    qb.limit(max(1, int(limit)))

    try:
        with db_access_guard("recent-processes"):
            rows = qb.all()
    except HTTPException:
        raise
    except SQLAlchemyTimeoutError as exc:
        raise http_error(503, "Database temporarily unavailable", endpoint="recent-processes", reason=str(exc)) from exc

    results: list[dict[str, Any]] = []
    for pk, state, label, exit_status, ctime in rows:
        state_value = state.value if hasattr(state, "value") else str(state or "unknown")
        results.append(
            {
                "pk": int(pk),
                "label": str(label or "Unknown Task"),
                "process_label": str(label or "Unknown Task"),
                "state": state_value,
                "exit_status": exit_status,
                "ctime": ctime.strftime("%Y-%m-%d %H:%M:%S") if ctime else None,
            }
        )
    return results


def _clone_recent_nodes_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return deepcopy(items)


def _get_recent_nodes_cached(
    key: tuple[int, str | None, str | None],
    *,
    allow_stale: bool = False,
) -> list[dict[str, Any]] | None:
    with _RECENT_NODES_CACHE_LOCK:
        record = _RECENT_NODES_CACHE.get(key)
    if record is None:
        return None
    cached_at, payload = record
    is_fresh = (time.monotonic() - cached_at) <= _RECENT_NODES_CACHE_TTL_SECONDS
    if not is_fresh and not allow_stale:
        return None
    return _clone_recent_nodes_payload(payload)


def _set_recent_nodes_cached(key: tuple[int, str | None, str | None], payload: list[dict[str, Any]]) -> None:
    with _RECENT_NODES_CACHE_LOCK:
        if len(_RECENT_NODES_CACHE) >= _RECENT_NODES_CACHE_MAX_ITEMS:
            oldest_key = min(_RECENT_NODES_CACHE.items(), key=lambda item: item[1][0])[0]
            _RECENT_NODES_CACHE.pop(oldest_key, None)
        _RECENT_NODES_CACHE[key] = (time.monotonic(), _clone_recent_nodes_payload(payload))


def _get_recent_nodes(limit: int = 15, group_label: str | None = None, node_type: str | None = None) -> list[dict[str, Any]]:
    normalized_limit = max(1, int(limit))
    normalized_group = str(group_label).strip() if group_label and str(group_label).strip() else None
    normalized_node_type = str(node_type).strip() if node_type and str(node_type).strip() else None
    cache_key = (normalized_limit, normalized_group, normalized_node_type)

    cached_payload = _get_recent_nodes_cached(cache_key)
    if cached_payload is not None:
        return cached_payload

    node_class = _resolve_node_class(normalized_node_type)
    qb = QueryBuilder()

    if normalized_group:
        qb.append(Group, filters={"label": normalized_group}, tag="group")
        qb.append(node_class, with_group="group", project=["*"], tag="node", subclassing=True)
    else:
        qb.append(node_class, project=["*"], tag="node", subclassing=True)

    qb.order_by({"node": {"ctime": "desc"}})
    qb.limit(normalized_limit)

    try:
        with db_access_guard("recent-nodes"):
            rows = qb.all()
    except HTTPException as exc:
        stale_payload = _get_recent_nodes_cached(cache_key, allow_stale=True)
        if stale_payload is not None:
            return stale_payload
        raise exc
    except SQLAlchemyTimeoutError as exc:
        stale_payload = _get_recent_nodes_cached(cache_key, allow_stale=True)
        if stale_payload is not None:
            return stale_payload
        raise http_error(503, "Database temporarily unavailable", endpoint="recent-nodes", reason=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        stale_payload = _get_recent_nodes_cached(cache_key, allow_stale=True)
        if stale_payload is not None:
            return stale_payload
        raise http_error(500, "Failed to query recent nodes", endpoint="recent-nodes", reason=str(exc)) from exc

    results: list[dict[str, Any]] = []
    for (node,) in rows:
        process_state_value: str | None = None
        if isinstance(node, ProcessNode):
            process_state_value = extract_process_state_value(node)

        formula_value: str | None = get_structure_formula(node) if isinstance(node, orm.StructureData) else None

        process_label = getattr(node, "process_label", None)
        if isinstance(node, orm.StructureData):
            label = node.label or formula_value or node.__class__.__name__
        else:
            label = process_label or node.label or node.__class__.__name__
        preview = build_node_preview(node)

        results.append(
            {
                "pk": int(node.pk),
                "state": process_state_value or "unknown",
                "label": str(label),
                "node_type": str(node.__class__.__name__),
                "process_state": process_state_value,
                "process_label": str(getattr(node, "process_label", None) or "N/A")
                if isinstance(node, ProcessNode)
                else None,
                "exit_status": getattr(node, "exit_status", None) if isinstance(node, ProcessNode) else None,
                "formula": formula_value,
                "preview_info": preview,
                "preview": preview,
            }
        )
    _set_recent_nodes_cached(cache_key, results)
    return results


def _get_unified_source_map(target: str | None = None) -> dict[str, Any]:
    if target and str(target).strip():
        cleaned = str(target).strip()
        candidate = Path(cleaned).expanduser()
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in {".aiida", ".zip"}:
            load_archive_profile(str(candidate))
        else:
            switch_profile(cleaned)

    groups = _list_groups()
    mounted_archive = current_mounted_archive()
    source_name = mounted_archive if mounted_archive else active_profile_name()
    source_type = "archive" if mounted_archive else "profile"

    return {
        "name": str(Path(source_name).name if source_type == "archive" else source_name),
        "type": source_type,
        "current_profile": active_profile_name(),
        "groups": [{"label": item["label"], "pk": item["pk"]} for item in groups],
    }


def _get_bands_plot_data(pk: int) -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Node not found", pk=pk, reason=str(exc)) from exc

    if not isinstance(node, orm.BandsData) and not hasattr(node, "_matplotlib_get_dict"):
        raise http_error(400, "Node is not a compatible BandsData type", pk=pk)

    try:
        if hasattr(node, "_matplotlib_get_dict"):
            return {
                "pk": int(node.pk),
                "data": to_jsonable(node._matplotlib_get_dict()),
            }
    except Exception as exc:  # noqa: BLE001
        raise http_error(500, "Failed to extract bands data", pk=pk, reason=str(exc)) from exc

    raise http_error(500, "Bands data extraction failed", pk=pk)


def _list_remote_files(pk: int | str) -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Node not found", pk=str(pk), reason=str(exc)) from exc

    if not isinstance(node, orm.RemoteData):
        raise http_error(400, "Node is not RemoteData", pk=str(pk), type=node.__class__.__name__)

    try:
        files = node.listdir()
    except Exception as exc:  # noqa: BLE001
        raise http_error(500, "Failed to list remote files", pk=str(pk), reason=str(exc)) from exc

    return {
        "pk": int(node.pk),
        "files": to_jsonable(files),
    }


def _get_remote_file_content(pk: int | str, filename: str) -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Node not found", pk=str(pk), reason=str(exc)) from exc

    if not isinstance(node, orm.RemoteData):
        raise http_error(400, "Node is not RemoteData", pk=str(pk), type=node.__class__.__name__)

    target = str(filename or "").strip()
    if not target:
        raise http_error(400, "Filename is required")

    try:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            node.getfile(target, str(destination))
            content = destination.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise http_error(500, "Failed to read remote file", pk=str(pk), filename=target, reason=str(exc)) from exc

    return {
        "pk": int(node.pk),
        "filename": target,
        "content": content,
    }


def _get_node_file_content(pk: int | str, filename: str, source: str = "folder") -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Node not found", pk=str(pk), reason=str(exc)) from exc

    target = str(filename or "").strip()
    if not target:
        raise http_error(400, "Filename is required")

    mode = str(source or "folder").strip().lower()

    try:
        if mode in {"repository", "virtual.repository"}:
            raw = node.base.repository.get_object_content(target)
        else:
            raw = node.get_object_content(target)
    except Exception as exc:  # noqa: BLE001
        raise http_error(
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


@management_router.get("/profiles")
def list_profiles() -> dict[str, Any]:
    from core.engine import list_profiles_payload

    return list_profiles_payload()


@management_router.post("/profiles/switch")
def management_switch_profile(payload: ProfileSwitchRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    current = switch_profile(payload.profile)
    return {"status": "switched", "current_profile": current}


@management_router.post("/profiles/load-archive")
def management_load_archive_profile(payload: ArchiveLoadRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    current = load_archive_profile(payload.path)
    return {
        "status": "loaded",
        "current_profile": current,
        "source": str(Path(payload.path).expanduser()),
    }


@management_router.get("/archives/local")
def list_local_archives(path: str = ".") -> dict[str, Any]:
    ensure_profile_loaded()
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        raise http_error(400, "Archive directory does not exist", path=str(root))

    items = [
        str(entry)
        for entry in sorted(root.iterdir())
        if entry.is_file() and entry.suffix.lower() in {".aiida", ".zip"}
    ]
    return {"path": str(root.resolve()), "archives": items}


@management_router.get("/system/info", response_model=SystemInfoResponse)
def management_system_info() -> SystemInfoResponse:
    ensure_profile_loaded()
    return SystemInfoResponse(**_get_system_info_payload())


@management_router.get("/resources", response_model=ResourcesResponse)
def management_resources() -> ResourcesResponse:
    ensure_profile_loaded()
    return ResourcesResponse(
        computers=[ComputerResource(**item) for item in _serialize_computers()],
        codes=[CodeResource(**item) for item in _serialize_codes()],
    )


@management_router.get("/statistics")
def management_statistics() -> dict[str, Any]:
    ensure_profile_loaded()
    return _get_statistics_payload()


@management_router.get("/database/summary")
def management_database_summary() -> dict[str, Any]:
    ensure_profile_loaded()
    return _get_database_summary_payload()


@management_router.get("/groups")
def management_groups(search: str | None = Query(default=None)) -> dict[str, Any]:
    ensure_profile_loaded()
    return {"items": _list_groups(search)}


@management_router.get("/groups/labels")
def management_group_labels(search: str | None = Query(default=None)) -> dict[str, Any]:
    ensure_profile_loaded()
    return {"items": _list_group_labels(search)}


@management_router.get("/groups/{group_name}")
def management_inspect_group(group_name: str, limit: int = Query(default=20, ge=1, le=500)) -> dict[str, Any]:
    ensure_profile_loaded()
    return _inspect_group(group_name, limit=limit)


@management_router.get("/recent-processes")
def management_recent_processes(limit: int = Query(default=15, ge=1, le=200)) -> dict[str, Any]:
    ensure_profile_loaded()
    return {"items": _get_recent_processes(limit)}


@management_router.get("/recent-nodes")
def management_recent_nodes(
    limit: int = Query(default=15, ge=1, le=200),
    group_label: str | None = Query(default=None),
    node_type: str | None = Query(default=None),
) -> dict[str, Any]:
    ensure_profile_loaded()
    return {"items": _get_recent_nodes(limit=limit, group_label=group_label, node_type=node_type)}


@management_router.get("/nodes/{pk}")
def management_node_summary(pk: int) -> dict[str, Any]:
    ensure_profile_loaded()
    return get_node_summary(pk)


@management_router.post("/nodes/context")
def management_context_nodes(payload: ContextNodesRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    items: list[dict[str, Any]] = []
    for raw_pk in payload.ids[:30]:
        try:
            items.append(get_node_summary(int(raw_pk)))
        except HTTPException as exc:
            error = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
            items.append({"pk": int(raw_pk), **error})
    return {"items": items}


@management_router.get("/source-map")
def management_source_map(target: str | None = Query(default=None)) -> dict[str, Any]:
    ensure_profile_loaded()
    return _get_unified_source_map(target)


@plugins_router.get("/plugins", response_model=list[str])
def list_workflow_plugins() -> list[str]:
    ensure_profile_loaded()
    return sorted(get_entry_point_names("aiida.workflows"))


@data_router.get("/bands/{pk}")
def data_bands(pk: int) -> dict[str, Any]:
    ensure_profile_loaded()
    return _get_bands_plot_data(pk)


@data_router.get("/remote/{pk}/files")
def data_remote_files(pk: int) -> dict[str, Any]:
    ensure_profile_loaded()
    return _list_remote_files(pk)


@data_router.get("/remote/{pk}/files/{filename:path}")
def data_remote_file_content(pk: int, filename: str) -> dict[str, Any]:
    ensure_profile_loaded()
    return _get_remote_file_content(pk, filename)


@data_router.get("/repository/{pk}/files/{filename:path}")
def data_repository_file_content(pk: int, filename: str, source: str = Query(default="folder")) -> dict[str, Any]:
    ensure_profile_loaded()
    return _get_node_file_content(pk, filename, source=source)
