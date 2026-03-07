from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, Query, UploadFile, File, Form, Depends
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from aiida import orm
from aiida.common.exceptions import MissingEntryPointError
from aiida.orm import Group, Node, ProcessNode, QueryBuilder
from aiida.plugins import DataFactory
from aiida.plugins.entry_point import get_entry_point_names
from core.engine import (
    SessionCleanupAPIRouter,
    active_profile_name,
    current_mounted_archive,
    db_access_guard,
    ensure_profile_loaded,
    get_system_info_payload,
    http_error,
    load_archive_profile,
    serialize_codes as _serialize_codes,
    serialize_computers as _serialize_computers,
    switch_profile,
)
from core.node_utils import (
    build_node_preview,
    extract_process_state_value,
    get_node_script_payload,
    get_node_summary,
    get_structure_formula,
    node_type_name,
)
from core.data_importers import create_node_from_file
from core.utils import to_jsonable
from models.schemas import (
    ArchiveLoadRequest,
    CodeResource,
    ComputerResource,
    ContextNodesRequest,
    GroupAddNodesRequest,
    GroupCreateRequest,
    GroupRenameRequest,
    InfrastructureSetupRequest,
    InfrastructureExportResponse,
    NodeScriptResponse,
    NodeSoftDeleteRequest,
    ProfileSetupRequest,
    ProfileSwitchRequest,
    ResourcesResponse,
    SSHHostDetails,
    SystemInfoResponse,
    UserInfoResponse,
)
from specializations import get_data_entry_point_aliases

management_router = SessionCleanupAPIRouter(prefix="/management", tags=["management"])
data_router = SessionCleanupAPIRouter(prefix="/data", tags=["data"])
plugins_router = SessionCleanupAPIRouter(tags=["plugins"])

_RECENT_NODES_CACHE_TTL_SECONDS = max(0.5, float(os.getenv("AIIDA_RECENT_NODES_CACHE_TTL_SECONDS", "5.0")))
_RECENT_NODES_CACHE_MAX_ITEMS = max(10, int(os.getenv("AIIDA_RECENT_NODES_CACHE_MAX_ITEMS", "128")))
_RECENT_NODES_CACHE_LOCK = threading.Lock()
_RECENT_NODES_CACHE: dict[tuple[int, str | None, str | None, str | None, str | None, bool], tuple[float, list[dict[str, Any]]]] = {}
_SOFT_DELETED_EXTRA_KEY = "sabr_soft_deleted"
_SOFT_DELETED_AT_EXTRA_KEY = "sabr_soft_deleted_at"

_NODE_CLASS_MAP: dict[str, type[Node]] = {
    "ProcessNode": orm.ProcessNode,
    "WorkChainNode": orm.WorkChainNode,
    "StructureData": orm.StructureData,
}

_DATA_ENTRY_POINT_ALIASES: dict[str, str] = get_data_entry_point_aliases()


def _get_statistics_payload() -> dict[str, Any]:
    from core.engine import _collect_system_counts, _is_daemon_running

    counts = _collect_system_counts()
    return {
        "profile": active_profile_name(),
        "daemon_status": _is_daemon_running(),
        "counts": counts,
    }


def _get_database_summary_payload() -> dict[str, Any]:
    from core.engine import _collect_system_counts

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


def _get_recent_processes(limit: int = 15, *, root_only: bool = True) -> list[dict[str, Any]]:
    qb = QueryBuilder()
    qb.append(
        ProcessNode,
        project=[
            "*",
        ],
        tag="process",
        subclassing=True,
    )
    qb.order_by({"process": {"ctime": "desc"}})
    qb.limit(max(1, int(limit)) * 6)

    try:
        with db_access_guard("recent-processes"):
            rows = qb.all()
    except HTTPException:
        raise
    except SQLAlchemyTimeoutError as exc:
        raise http_error(503, "Database temporarily unavailable", endpoint="recent-processes", reason=str(exc)) from exc

    results: list[dict[str, Any]] = []
    for (node,) in rows:
        if root_only and getattr(node, "caller", None) is not None:
            continue
        state = getattr(node, "process_state", None)
        state_value = state.value if hasattr(state, "value") else str(state or "unknown")
        results.append(
            {
                "pk": int(node.pk),
                "label": str(getattr(node, "process_label", None) or node.label or "Unknown Task"),
                "process_label": str(getattr(node, "process_label", None) or node.label or "Unknown Task"),
                "state": state_value,
                "exit_status": getattr(node, "exit_status", None),
                "ctime": node.ctime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "ctime", None) else None,
            }
        )
        if len(results) >= max(1, int(limit)):
            break
    return results


def _clone_recent_nodes_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return deepcopy(items)


def _get_recent_nodes_cached(
    key: tuple[int, str | None, str | None, str | None, str | None, bool],
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


def _set_recent_nodes_cached(key: tuple[int, str | None, str | None, str | None, str | None, bool], payload: list[dict[str, Any]]) -> None:
    with _RECENT_NODES_CACHE_LOCK:
        if len(_RECENT_NODES_CACHE) >= _RECENT_NODES_CACHE_MAX_ITEMS:
            oldest_key = min(_RECENT_NODES_CACHE.items(), key=lambda item: item[1][0])[0]
            _RECENT_NODES_CACHE.pop(oldest_key, None)
        _RECENT_NODES_CACHE[key] = (time.monotonic(), _clone_recent_nodes_payload(payload))


def _clear_recent_nodes_cache() -> None:
    with _RECENT_NODES_CACHE_LOCK:
        _RECENT_NODES_CACHE.clear()


def _serialize_group_item(group: Group) -> dict[str, Any]:
    return {
        "label": str(group.label),
        "pk": int(group.pk),
        "count": int(len(group.nodes)),
        "type_string": str(getattr(group, "type_string", "")) or None,
    }


def _load_group_by_pk(pk: int) -> Group:
    try:
        return orm.load_group(pk=int(pk))
    except (NotExistent, ValueError, TypeError) as exc:
        raise http_error(404, "Group not found", pk=int(pk)) from exc


def _create_group(label: str) -> dict[str, Any]:
    cleaned = str(label or "").strip()
    if not cleaned:
        raise http_error(400, "Group label is required")

    existing = Group.collection.find(filters={"label": cleaned})
    if existing:
        raise http_error(409, "Group already exists", label=cleaned)

    group = Group(label=cleaned).store()
    _clear_recent_nodes_cache()
    return _serialize_group_item(group)


def _rename_group(pk: int, label: str) -> dict[str, Any]:
    group = _load_group_by_pk(pk)
    cleaned = str(label or "").strip()
    if not cleaned:
        raise http_error(400, "Group label is required")

    if cleaned != str(group.label):
        existing = Group.collection.find(filters={"label": cleaned})
        if any(int(candidate.pk) != int(group.pk) for candidate in existing):
            raise http_error(409, "Group already exists", label=cleaned)
        group.label = cleaned
        _clear_recent_nodes_cache()

    return _serialize_group_item(group)


def _delete_group(pk: int) -> dict[str, Any]:
    group = _load_group_by_pk(pk)
    payload = _serialize_group_item(group)
    Group.collection.delete(int(group.pk))
    _clear_recent_nodes_cache()
    return {"status": "deleted", **payload}


def _add_nodes_to_group(pk: int, node_pks: list[int]) -> dict[str, Any]:
    group = _load_group_by_pk(pk)
    normalized_ids: list[int] = []
    seen: set[int] = set()
    for raw_pk in node_pks:
        try:
            parsed = int(raw_pk)
        except (TypeError, ValueError):
            continue
        if parsed <= 0 or parsed in seen:
            continue
        seen.add(parsed)
        normalized_ids.append(parsed)

    if not normalized_ids:
        return {"group": _serialize_group_item(group), "added": [], "missing": []}

    existing_node_ids = {int(node.pk) for node in group.nodes}
    nodes_to_add: list[Node] = []
    added_ids: list[int] = []
    missing_ids: list[int] = []

    for node_pk in normalized_ids:
        if node_pk in existing_node_ids:
            continue
        try:
            node = orm.load_node(node_pk)
        except (NotExistent, ValueError, TypeError):
            missing_ids.append(node_pk)
            continue
        nodes_to_add.append(node)
        added_ids.append(node_pk)

    if nodes_to_add:
        group.add_nodes(nodes_to_add)
        _clear_recent_nodes_cache()

    return {
        "group": _serialize_group_item(group),
        "added": added_ids,
        "missing": missing_ids,
    }


def _remove_node_from_group(pk: int, node_pk: int) -> dict[str, Any]:
    group = _load_group_by_pk(pk)
    try:
        node = orm.load_node(int(node_pk))
    except (NotExistent, ValueError, TypeError) as exc:
        raise http_error(404, "Node not found", pk=int(node_pk)) from exc

    existing_node_ids = {int(n.pk) for n in group.nodes}
    if int(node.pk) not in existing_node_ids:
        return {"group": _serialize_group_item(group), "removed": False}

    group.remove_nodes([node])
    _clear_recent_nodes_cache()
    return {"group": _serialize_group_item(group), "removed": True}


def _is_soft_deleted(node: Node) -> bool:
    try:
        return bool(node.base.extras.get(_SOFT_DELETED_EXTRA_KEY, False))
    except Exception:  # noqa: BLE001
        return False


def _soft_delete_node(pk: int, *, deleted: bool = True) -> dict[str, Any]:
    try:
        node = orm.load_node(pk=int(pk))
    except (NotExistent, ValueError, TypeError) as exc:
        raise http_error(404, "Node not found", pk=int(pk)) from exc

    if deleted:
        node.base.extras.set(_SOFT_DELETED_EXTRA_KEY, True)
        node.base.extras.set(_SOFT_DELETED_AT_EXTRA_KEY, int(time.time()))
    else:
        with suppress(Exception):
            node.base.extras.delete(_SOFT_DELETED_EXTRA_KEY)
        with suppress(Exception):
            node.base.extras.delete(_SOFT_DELETED_AT_EXTRA_KEY)

    _clear_recent_nodes_cache()
    return {"pk": int(node.pk), "soft_deleted": bool(deleted)}


def _export_group(pk: int) -> dict[str, Any]:
    group = _load_group_by_pk(pk)
    exported_nodes: list[dict[str, Any]] = []

    for node in group.nodes:
        process_state = getattr(node, "process_state", None)
        state_value = process_state.value if hasattr(process_state, "value") else str(process_state or "unknown")
        exported_nodes.append(
            {
                "pk": int(node.pk),
                "uuid": str(node.uuid),
                "label": str(node.label or node_type_name(node)),
                "node_type": str(node.__class__.__name__),
                "process_label": str(getattr(node, "process_label", None) or "N/A"),
                "process_state": state_value,
                "ctime": node.ctime.isoformat() if getattr(node, "ctime", None) else None,
                "mtime": node.mtime.isoformat() if getattr(node, "mtime", None) else None,
            }
        )

    return {
        "group": _serialize_group_item(group),
        "nodes": exported_nodes,
    }


def _get_recent_nodes(
    limit: int = 15,
    group_label: str | None = None,
    node_type: str | None = None,
    label: str | None = None,
    process_state: str | None = None,
    root_only: bool = True,
) -> list[dict[str, Any]]:
    normalized_limit = max(1, int(limit))
    normalized_group = str(group_label).strip() if group_label and str(group_label).strip() else None
    normalized_node_type = str(node_type).strip() if node_type and str(node_type).strip() else None
    normalized_label = str(label).strip() if label and str(label).strip() else None
    normalized_process_state = str(process_state).strip() if process_state and str(process_state).strip() else None
    
    cache_key = (
        normalized_limit,
        normalized_group,
        normalized_node_type,
        normalized_label,
        normalized_process_state,
        bool(root_only),
    )

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

    node_filters: dict[str, Any] = {}
    if normalized_label:
        if normalized_label.isdigit():
            node_filters["or"] = [
                {"id": int(normalized_label)},
                {"label": {"ilike": f"%{normalized_label}%"}},
            ]
        else:
            node_filters["label"] = {"ilike": f"%{normalized_label}%"}
    if normalized_process_state:
        node_filters["attributes.process_state"] = normalized_process_state
    if node_filters:
        qb.add_filter("node", node_filters)

    qb.order_by({"node": {"ctime": "desc"}})
    qb.limit(max(normalized_limit * 4, normalized_limit))

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
        if _is_soft_deleted(node):
            continue
        if root_only and isinstance(node, ProcessNode) and getattr(node, "caller", None) is not None:
            continue
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
        if len(results) >= normalized_limit:
            break
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


def _list_repository_files(pk: int | str, source: str = "folder") -> dict[str, Any]:
    try:
        node = orm.load_node(pk)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Node not found", pk=str(pk), reason=str(exc)) from exc

    mode = str(source or "folder").strip().lower()

    try:
        if mode in {"repository", "virtual.repository"}:
            files = node.base.repository.list_object_names()
        else:
            files = node.list_object_names()
    except Exception as exc:  # noqa: BLE001
        raise http_error(
            500,
            "Failed to list node files",
            pk=str(pk),
            source=source,
            reason=str(exc),
        ) from exc

    return {
        "pk": int(node.pk),
        "source": mode,
        "files": to_jsonable(sorted([str(entry) for entry in files])),
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


@management_router.get("/profiles/current-user-info", response_model=UserInfoResponse)
def management_current_user_info() -> dict[str, Any]:
    ensure_profile_loaded()
    try:
        from aiida.manage.configuration import get_profile
        from aiida import orm

        default_email = get_profile().default_user_email
        if default_email is None:
            # We won't crash, instead return empty defaults if there is no user email yet
            return {"first_name": "", "last_name": "", "email": "", "institution": ""}
        
        user = orm.User.collection.get(email=default_email)
        return {
            "first_name": str(getattr(user, 'first_name', '')),
            "last_name": str(getattr(user, 'last_name', '')),
            "email": str(getattr(user, 'email', '')),
            "institution": str(getattr(user, 'institution', ''))
        }
    except Exception as exc:
        raise http_error(500, "Failed to extract current user info", reason=str(exc))


@management_router.post("/profiles/setup")
def management_setup_profile(payload: ProfileSetupRequest) -> dict[str, Any]:
    try:
        from aiida.manage.configuration import create_profile, get_config
        
        config = get_config()
        
        # Define storage configuration for sqlite_dos
        storage_config = {
            "filepath": str(Path(payload.filepath).expanduser().resolve())
        }
        
        # Optional broker configuration (RabbitMQ)
        broker_backend = "core.rabbitmq"
        broker_config = {
            "host": "localhost",
            "port": 5672,
        }

        create_profile(
            config=config,
            storage_backend=payload.backend,
            storage_config=storage_config,
            broker_backend=broker_backend,
            broker_config=broker_config,
            name=payload.profile_name,
            email=payload.email,
            first_name=payload.first_name,
            last_name=payload.last_name,
            institution=payload.institution,
            is_test_profile=False,
        )
        
        # Store the updated config back to disk
        config.store()
        
        return {"status": "success", "profile_name": payload.profile_name}
    except Exception as exc:
        raise http_error(500, "Failed to setup profile programmatically", reason=str(exc))


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
    return SystemInfoResponse(**get_system_info_payload())


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


@management_router.post("/groups/create")
def management_create_group(payload: GroupCreateRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    return _create_group(payload.label)


@management_router.put("/groups/{pk}/label")
def management_rename_group(pk: int, payload: GroupRenameRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    return _rename_group(pk, payload.label)


@management_router.delete("/groups/{pk}")
def management_delete_group(pk: int) -> dict[str, Any]:
    ensure_profile_loaded()
    return _delete_group(pk)


@management_router.post("/groups/{pk}/nodes")
def management_add_nodes_to_group(pk: int, payload: GroupAddNodesRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    return _add_nodes_to_group(pk, payload.node_pks)


@management_router.delete("/groups/{pk}/nodes/{node_pk}")
def management_remove_node_from_group(pk: int, node_pk: int) -> dict[str, Any]:
    ensure_profile_loaded()
    return _remove_node_from_group(pk, node_pk)


@management_router.get("/groups/{pk}/export")
def management_export_group(pk: int) -> dict[str, Any]:
    ensure_profile_loaded()
    return _export_group(pk)


@management_router.get("/groups/labels")
def management_group_labels(search: str | None = Query(default=None)) -> dict[str, Any]:
    ensure_profile_loaded()
    return {"items": _list_group_labels(search)}


@management_router.get("/groups/{group_name}")
def management_inspect_group(group_name: str, limit: int = Query(default=20, ge=1, le=500)) -> dict[str, Any]:
    ensure_profile_loaded()
    return _inspect_group(group_name, limit=limit)


@management_router.get("/recent-processes")
def management_recent_processes(
    limit: int = Query(default=15, ge=1, le=200),
    root_only: bool = Query(default=True),
) -> dict[str, Any]:
    ensure_profile_loaded()
    return {"items": _get_recent_processes(limit, root_only=root_only)}


@management_router.get("/recent-nodes")
def management_recent_nodes(
    limit: int = Query(default=15, ge=1, le=200),
    group_label: str | None = Query(default=None),
    node_type: str | None = Query(default=None),
    label: str | None = Query(default=None),
    process_state: str | None = Query(default=None),
    root_only: bool = Query(default=True),
) -> dict[str, Any]:
    ensure_profile_loaded()
    return {"items": _get_recent_nodes(
        limit=limit,
        group_label=group_label,
        node_type=node_type,
        label=label,
        process_state=process_state,
        root_only=root_only,
    )}


@management_router.get("/nodes/{pk}")
def management_node_summary(pk: int) -> dict[str, Any]:
    ensure_profile_loaded()
    return get_node_summary(pk)


@management_router.get("/nodes/{pk}/script", response_model=NodeScriptResponse)
def management_node_script(pk: int) -> NodeScriptResponse:
    ensure_profile_loaded()
    return NodeScriptResponse(**get_node_script_payload(pk))


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


@management_router.post("/nodes/{pk}/soft-delete")
def management_soft_delete_node(pk: int, payload: NodeSoftDeleteRequest) -> dict[str, Any]:
    ensure_profile_loaded()
    return _soft_delete_node(pk, deleted=payload.deleted)


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


@data_router.get("/repository/{pk}/files")
def data_repository_files(pk: int, source: str = Query(default="folder")) -> dict[str, Any]:
    ensure_profile_loaded()
    return _list_repository_files(pk, source=source)


@data_router.get("/repository/{pk}/files/{filename:path}")
def data_repository_file_content(pk: int, filename: str, source: str = Query(default="folder")) -> dict[str, Any]:
    ensure_profile_loaded()
    return _get_node_file_content(pk, filename, source=source)


@data_router.post("/import/{data_type}")
async def data_import_node(
    data_type: str,
    file: UploadFile | None = File(None),
    source_type: str = Form("file"),
    raw_text: str | None = Form(None),
    label: str | None = Form(None),
    description: str | None = Form(None),
) -> dict[str, Any]:
    ensure_profile_loaded()
    
    if source_type == "file":
        if not file:
            raise http_error(400, "File is required when source_type is 'file'")
        content = await file.read()
        filename = file.filename or "uploaded_file"
    else:
        if not raw_text:
            raise http_error(400, "raw_text is required when source_type is 'raw_text'")
        content = raw_text
        filename = None

    if not content:
        raise http_error(400, "No content provided for import")
        
    try:
        result = create_node_from_file(
            data_type=data_type,
            file_content=content,
            filename=filename,
            label=label,
            description=description,
            source_type=source_type,
        )
        
        if isinstance(result, dict) and result.get("status") == "success":
            return result

        node = result
        return {
            "status": "success",
            "pk": int(node.pk),
            "uuid": str(node.uuid),
            "type": node_type_name(node),
        }
    except ValueError as exc:
        raise http_error(400, str(exc))
    except Exception as exc:
        raise http_error(500, f"Internal error during import: {str(exc)}")

@management_router.get("/infrastructure")
def get_infrastructure():
    """
    Return a nested structure: List[ComputerSchema].
    Each ComputerSchema includes metadata and a list of associated Codes.
    """
    ensure_profile_loaded()
    
    # Get a fresh User in the current session to avoid SQLAlchemy DetachedInstance errors
    from aiida.manage.configuration import get_profile
    default_email = get_profile().default_user_email
    fresh_user = orm.User.collection.get(email=default_email)
    
    computers = orm.Computer.collection.all()
    codes = orm.Code.collection.all()
    
    computer_list = []
    for computer in computers:
        comp_dict = {
            "pk": int(computer.pk),
            "label": str(computer.label),
            "hostname": str(computer.hostname),
            "description": str(computer.description) if computer.description else None,
            "scheduler_type": str(computer.scheduler_type),
            "transport_type": str(computer.transport_type),
            "is_enabled": bool(computer.is_user_configured(fresh_user)),
            "codes": []
        }
        # Filter codes for this computer
        for code in codes:
            try:
                if code.computer and int(code.computer.pk) == int(computer.pk):
                    comp_dict["codes"].append({
                        "pk": int(code.pk),
                        "label": str(code.label),
                        "description": str(code.description) if code.description else None,
                        "default_calc_job_plugin": str(getattr(code, "default_calc_job_plugin", "") or ""),
                    })
            except Exception:
                continue
        computer_list.append(comp_dict)
    
    return computer_list

@management_router.get("/infrastructure/ssh-config")
def get_ssh_config() -> list[SSHHostDetails]:
    """
    Parse the local ~/.ssh/config file and return a list of available hosts.
    """
    import os
    import paramiko

    ssh_config_path = os.path.expanduser("~/.ssh/config")
    if not os.path.exists(ssh_config_path):
        return []

    config = paramiko.config.SSHConfig()
    try:
        with open(ssh_config_path, "r") as f:
            config.parse(f)
    except Exception as e:
        # Just return empty if unparseable
        return []

    hosts = []
    # config.get_hostnames() returns all defined hosts
    for alias in config.get_hostnames():
        # paramiko default includes a catch-all '*', skip it if it's the only rule
        if alias == '*':
            continue
            
        host_dict = config.lookup(alias)
        
        # Determine port
        port = host_dict.get('port')
        if port:
            try:
                port = int(port)
            except ValueError:
                port = None

        # Build SSHHostDetails
        # Paramiko lookup returns keys in lower case: hostname, user, port, proxyjump, identityfile...
        details = SSHHostDetails(
            alias=alias,
            hostname=host_dict.get('hostname'),
            username=host_dict.get('user'),
            port=port,
            proxy_jump=host_dict.get('proxyjump'),
            proxy_command=host_dict.get('proxycommand'),
            identity_file=host_dict.get('identityfile', [None])[0] if isinstance(host_dict.get('identityfile'), list) else host_dict.get('identityfile')
        )
        hosts.append(details)

    return hosts

from models.schemas import (
    CodeSetupRequest,
    CodeDetailedResponse,
    InfrastructureSetupRequest,
    SSHHostDetails,
)

@management_router.post("/infrastructure/setup-code")
def setup_code(payload: CodeSetupRequest):
    """
    Create and store an orm.InstalledCode for a given computer.
    """
    ensure_profile_loaded()
    try:
        computer = orm.Computer.collection.get(label=payload.computer_label)
        
        # Check if code already exists
        existing_codes = orm.Code.collection.find(filters={'label': payload.label, 'attributes.remote_computer_uuid': computer.uuid})
        if existing_codes:
            code = existing_codes[0]
        else:
            code = orm.InstalledCode(
                label=payload.label,
                description=payload.description or "",
                default_calc_job_plugin=payload.default_calc_job_plugin,
                computer=computer,
                filepath_executable=payload.remote_abspath
            )
        
        if payload.prepend_text:
            code.set_prepend_text(payload.prepend_text)
        if payload.append_text:
            code.set_append_text(payload.append_text)
            
        code.with_mpi = payload.with_mpi
        code.use_double_quotes = payload.use_double_quotes
        
        code.store()
        
        return {
            "status": "success",
            "pk": int(code.pk),
            "label": str(code.label)
        }
    except Exception as exc:
        raise http_error(500, "Failed to setup code", reason=str(exc))

@management_router.get("/infrastructure/computer/{computer_label}/codes", response_model=list[CodeDetailedResponse])
def get_computer_codes(computer_label: str):
    """
    Return detailed information for all codes associated with a computer.
    Used for providing suggestions/templates in the frontend.
    """
    ensure_profile_loaded()
    try:
        computer = orm.Computer.collection.get(label=computer_label)
        codes = orm.Code.collection.find(filters={'attributes.remote_computer_uuid': computer.uuid})
        
        result = []
        for code in codes:
            # We only care about InstalledCode for templates usually, but AiiDA might return others
            # In AiiDA 2.x, orm.Code might be a legacy class or a base class.
            # We'll try to extract what we can.
            result.append({
                "pk": int(code.pk),
                "label": str(code.label),
                "description": str(code.description) if code.description else None,
                "default_calc_job_plugin": str(getattr(code, "default_calc_job_plugin", "") or ""),
                "remote_abspath": str(getattr(code, "filepath_executable", "") or ""),
                "prepend_text": str(code.get_prepend_text()) if hasattr(code, 'get_prepend_text') else None,
                "append_text": str(code.get_append_text()) if hasattr(code, 'get_append_text') else None,
                "with_mpi": bool(getattr(code, 'with_mpi', False)),
                "use_double_quotes": bool(getattr(code, 'use_double_quotes', False))
            })
        return result
    except Exception as exc:
        raise http_error(500, "Failed to fetch computer codes", reason=str(exc))


def _trim_export_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, nested_value in value.items():
            trimmed = _trim_export_value(nested_value)
            if trimmed is None:
                continue
            if isinstance(trimmed, str) and not trimmed:
                continue
            if isinstance(trimmed, dict) and not trimmed:
                continue
            cleaned[str(key)] = trimmed
        return cleaned
    if isinstance(value, list):
        return [_trim_export_value(item) for item in value]
    if value is None:
        return None
    return value


def _dump_export_yaml(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(
        _trim_export_value(payload),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )


def _build_computer_export_response(computer: Any) -> InfrastructureExportResponse:
    exported: dict[str, Any] = {
        "label": str(computer.label),
        "hostname": str(computer.hostname),
        "description": str(computer.description) if computer.description else None,
        "transport": str(computer.transport_type),
        "scheduler": str(computer.scheduler_type),
        "shebang": str(getattr(computer, "get_shebang", lambda: "")() or ""),
        "work_dir": str(getattr(computer, "get_workdir", lambda: "")() or ""),
        "mpiprocs_per_machine": getattr(computer, "get_default_mpiprocs_per_machine", lambda: None)(),
        "mpirun_command": " ".join(getattr(computer, "get_mpirun_command", lambda: [])() or []),
        "default_memory_per_machine": getattr(computer, "get_default_memory_per_machine", lambda: None)(),
        "prepend_text": str(computer.get_prepend_text()) if hasattr(computer, "get_prepend_text") else None,
        "append_text": str(computer.get_append_text()) if hasattr(computer, "get_append_text") else None,
        "use_double_quotes": bool(getattr(computer, "get_use_double_quotes", lambda: False)()),
    }

    with suppress(Exception):
        user = orm.User.collection.get_default()
        authinfo = computer.get_authinfo(user)
        auth_params = authinfo.get_auth_params() if hasattr(authinfo, "get_auth_params") else {}
        if isinstance(auth_params, dict) and auth_params:
            exported["auth"] = {
                "username": auth_params.get("username"),
                "key_filename": auth_params.get("key_filename"),
                "proxy_command": auth_params.get("proxy_command"),
                "proxy_jump": auth_params.get("proxy_jump"),
                "use_login_shell": auth_params.get("use_login_shell"),
                "safe_interval": auth_params.get("safe_interval"),
                "connection_timeout": auth_params.get("connection_timeout"),
            }

    return InfrastructureExportResponse(
        kind="computer",
        label=str(computer.label),
        filename=f"{computer.label}-setup.yaml",
        content=_dump_export_yaml(exported),
    )


@management_router.get(
    "/infrastructure/computer/{computer_label}/export",
    response_model=InfrastructureExportResponse,
)
def export_computer_config(computer_label: str):
    ensure_profile_loaded()
    try:
        computer = orm.Computer.collection.get(label=computer_label)
        return _build_computer_export_response(computer)
    except Exception as exc:
        raise http_error(500, "Failed to export computer config", reason=str(exc))


@management_router.get(
    "/infrastructure/computer/pk/{computer_pk}/export",
    response_model=InfrastructureExportResponse,
)
def export_computer_config_by_pk(computer_pk: int):
    ensure_profile_loaded()
    try:
        computer = orm.Computer.collection.get(pk=computer_pk)
        return _build_computer_export_response(computer)
    except Exception as exc:
        raise http_error(500, "Failed to export computer config", reason=str(exc))


@management_router.get(
    "/infrastructure/code/{code_pk}/export",
    response_model=InfrastructureExportResponse,
)
def export_code_config(code_pk: int):
    ensure_profile_loaded()
    try:
        code = orm.load_code(code_pk)
        exported = {
            "label": str(code.label),
            "computer": str(code.computer.label) if getattr(code, "computer", None) else None,
            "description": str(code.description) if code.description else None,
            "default_calc_job_plugin": str(getattr(code, "default_calc_job_plugin", "") or ""),
            "filepath_executable": str(getattr(code, "filepath_executable", "") or ""),
            "prepend_text": str(code.get_prepend_text()) if hasattr(code, "get_prepend_text") else None,
            "append_text": str(code.get_append_text()) if hasattr(code, "get_append_text") else None,
            "with_mpi": bool(getattr(code, "with_mpi", False)),
            "use_double_quotes": bool(getattr(code, "use_double_quotes", False)),
        }
        filename_label = str(code.label)
        computer_label = str(code.computer.label) if getattr(code, "computer", None) else None
        if computer_label:
            filename_label = f"{filename_label}@{computer_label}"
        return InfrastructureExportResponse(
            kind="code",
            label=str(code.label),
            filename=f"{filename_label}.yaml",
            content=_dump_export_yaml(exported),
        )
    except Exception as exc:
        raise http_error(500, "Failed to export code config", reason=str(exc))

@management_router.post("/infrastructure/setup")
def setup_infrastructure(payload: InfrastructureSetupRequest):
    """
    Create/store an orm.Computer, configure its AuthInfo, test the connection,
    and optionally configure a Code.
    """
    ensure_profile_loaded()
    try:
        # 1. Computer Setup
        try:
            computer = orm.Computer.collection.get(label=payload.computer_label)
        except Exception:
            computer = orm.Computer(
                label=payload.computer_label,
                hostname=payload.hostname,
                description=payload.computer_description or "",
                transport_type=payload.transport_type,
                scheduler_type=payload.scheduler_type,
                workdir=payload.work_dir
            )
        computer.set_default_mpiprocs_per_machine(payload.mpiprocs_per_machine)
        if payload.mpirun_command:
            computer.set_mpirun_command(payload.mpirun_command.split())
        if payload.shebang:
            computer.set_shebang(payload.shebang)
        if payload.default_memory_per_machine is not None:
            computer.set_default_memory_per_machine(payload.default_memory_per_machine)
        computer.set_use_double_quotes(payload.use_double_quotes)
        if payload.prepend_text:
            computer.set_prepend_text(payload.prepend_text)
        if payload.append_text:
            computer.set_append_text(payload.append_text)
        computer.store()
        
        # Configure AuthInfo
        user = orm.User.collection.get_default()
        try:
            authinfo = computer.get_authinfo(user)
        except Exception:
            authinfo = orm.AuthInfo(computer=computer, user=user)
            
        auth_params = {}
        if payload.username:
            auth_params["username"] = payload.username
        if payload.key_filename:
            auth_params["key_filename"] = payload.key_filename
        if payload.proxy_command:
            auth_params["proxy_command"] = payload.proxy_command
        if payload.proxy_jump:
            auth_params["proxy_jump"] = payload.proxy_jump
        if payload.use_login_shell is not None:
            auth_params["use_login_shell"] = payload.use_login_shell
        if payload.safe_interval is not None:
            auth_params["safe_interval"] = payload.safe_interval
        if payload.connection_timeout is not None:
            auth_params["connection_timeout"] = payload.connection_timeout
            
        authinfo.set_auth_params(auth_params)
        authinfo.store()
        
        # Test connection
        connection_status = "success"
        connection_error = None
        try:
            with computer.get_transport() as transport:
                if not transport.is_open:
                    transport.open()
        except Exception as conn_exc:
            connection_status = "failed"
            connection_error = str(conn_exc)
        
        # Configure Code
        code_pk = None
        if payload.code_label and payload.default_calc_job_plugin and payload.remote_abspath:
            # Check if code exists on this computer
            existing_codes = orm.Code.collection.find(filters={'label': payload.code_label, 'attributes.remote_computer_uuid': computer.uuid})
            if existing_codes:
                code = existing_codes[0]
            else:
                code = orm.InstalledCode(
                    label=payload.code_label,
                    description=payload.code_description or "",
                    default_calc_job_plugin=payload.default_calc_job_plugin,
                    computer=computer,
                    filepath_executable=payload.remote_abspath
                )
            if payload.code_prepend_text:
                code.set_prepend_text(payload.code_prepend_text)
            if payload.code_append_text:
                code.set_append_text(payload.code_append_text)
            code.store()
            code_pk = int(code.pk)
            
        return {
            "status": "success", 
            "computer_pk": int(computer.pk),
            "code_pk": code_pk,
            "connection_status": connection_status,
            "connection_error": connection_error
        }
    except Exception as exc:
        raise http_error(500, "Failed to setup infrastructure", reason=str(exc))
