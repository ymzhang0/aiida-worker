from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import Any, Mapping

from aiida import orm
from aiida.orm import QueryBuilder

from core.engine import http_error
from core.node_utils import node_type_name, serialize_node


def _get_process_summary(node: orm.ProcessNode) -> dict[str, Any]:
    process_state = getattr(node, "process_state", None)
    return {
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "type": node_type_name(node),
        "process_label": str(getattr(node, "process_label", None) or "N/A"),
        "state": process_state.value if hasattr(process_state, "value") else str(process_state or "unknown"),
        "exit_status": getattr(node, "exit_status", None),
    }


def _get_links_dict(node: orm.Node, direction: str) -> dict[str, Any]:
    links_dict: dict[str, Any] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    link_manager = node.base.links.get_incoming() if direction == "incoming" else node.base.links.get_outgoing()
    
    for link in link_manager.all():
        label = str(link.link_label)
        unique_label = f"{label}_{counts[label]}" if counts[label] > 0 else label
        counts[label] += 1
        
        serialized = serialize_node(link.node)
        serialized["link_label"] = label
        serialized["pk"] = int(serialized["pk"])
        links_dict[unique_label] = serialized
        
    return links_dict


def _flatten_node_manager(manager: Any, prefix: str = "") -> dict[str, orm.Node]:
    result: dict[str, orm.Node] = {}
    
    # Handle AiiDA's NodeLinksManager (used by node.inputs/node.outputs)
    if hasattr(manager, "_get_keys") and hasattr(manager, "_get_node_by_link_label"):
        try:
            for key in manager._get_keys():
                val = manager._get_node_by_link_label(key)
                if isinstance(val, orm.Node):
                    result[f"{prefix}{key}"] = val
                elif hasattr(val, "_get_keys"): # Nested NodeLinksManager
                    result.update(_flatten_node_manager(val, prefix=f"{prefix}{key}__"))
            return result
        except Exception:
            pass

    # Fallback/Recursive for standard dict-like managers or nested namespaces
    try:
        keys = manager.keys()
    except AttributeError:
        return result
        
    for key in keys:
        try:
            val = manager[key]
            if isinstance(val, orm.Node):
                result[f"{prefix}{key}"] = val
            elif hasattr(val, "keys") or hasattr(val, "_get_keys"):
                result.update(_flatten_node_manager(val, prefix=f"{prefix}{key}__"))
        except Exception:
            pass
    return result


def _get_direct_links_dict(node: orm.Node, direction: str) -> dict[str, Any]:
    manager = getattr(node, "inputs" if direction == "incoming" else "outputs", None)
    if manager is None:
        return {}
        
    links_dict: dict[str, Any] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    flat = _flatten_node_manager(manager)
    for label, port_node in flat.items():
        unique_label = f"{label}_{counts[label]}" if counts[label] > 0 else label
        counts[label] += 1
        
        serialized = serialize_node(port_node)
        serialized["link_label"] = label
        serialized["pk"] = int(serialized["pk"])
        links_dict[unique_label] = serialized
        
    return links_dict

class ProcessTree:
    """Recursively build and serialize process provenance trees."""

    def __init__(self, node: orm.ProcessNode):
        self.node = node
        self.children: dict[str, ProcessTree] = {}

        if isinstance(node, orm.WorkChainNode):
            subprocesses = sorted(node.called, key=lambda process: process.ctime)
            counts: defaultdict[str, int] = defaultdict(int)
            for sub in subprocesses:
                raw_label: Any = None
                metadata_inputs = sub.base.attributes.all.get("metadata_inputs", {})
                if isinstance(metadata_inputs, Mapping):
                    metadata = metadata_inputs.get("metadata", {})
                    if isinstance(metadata, Mapping):
                        raw_label = metadata.get("call_link_label")
                if not raw_label:
                    raw_label = getattr(sub, "process_label", "process")
                label = str(raw_label)
                unique_label = f"{label}_{counts[label]}" if counts[label] > 0 else label
                counts[label] += 1
                self.children[unique_label] = ProcessTree(sub)

    def to_dict(self) -> dict[str, Any]:
        summary = _get_process_summary(self.node)
        summary["inputs"] = _get_links_dict(self.node, "incoming")
        summary["outputs"] = _get_links_dict(self.node, "outgoing")
        summary["direct_inputs"] = _get_direct_links_dict(self.node, "incoming")
        summary["direct_outputs"] = _get_direct_links_dict(self.node, "outgoing")
        summary["children"] = {label: child.to_dict() for label, child in self.children.items()}
        return summary


def get_process_log_payload(identifier_or_node: int | str | orm.ProcessNode) -> dict[str, Any]:
    if isinstance(identifier_or_node, orm.ProcessNode):
        node = identifier_or_node
    else:
        try:
            node = orm.load_node(identifier_or_node)
        except Exception as exc:  # noqa: BLE001
            raise http_error(404, "Process node not found", identifier=str(identifier_or_node), reason=str(exc)) from exc

    if not isinstance(node, orm.ProcessNode):
        raise http_error(400, "Node is not a ProcessNode", identifier=str(getattr(node, "pk", identifier_or_node)))

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


def inspect_calculation_node(node: orm.ProcessNode) -> dict[str, Any]:
    if not isinstance(node, (orm.CalcJobNode, orm.CalcFunctionNode)):
        return {"error": f"Node {node.pk} is not a calculation node."}

    payload: dict[str, Any] = {
        "summary": serialize_node(node),
        "inputs": _get_links_dict(node, "incoming"),
        "outputs": _get_links_dict(node, "outgoing"),
        "direct_inputs": _get_direct_links_dict(node, "incoming"),
        "direct_outputs": _get_direct_links_dict(node, "outgoing"),
        "repository_files": [],
    }

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


def inspect_workchain_node(node: orm.WorkChainNode) -> dict[str, Any]:
    tree = ProcessTree(node)
    return {"provenance_tree": tree.to_dict()}


def inspect_process_payload(identifier: str | int) -> dict[str, Any]:
    try:
        node = orm.load_node(identifier)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Process node not found", identifier=str(identifier), reason=str(exc)) from exc

    if not isinstance(node, orm.ProcessNode):
        raise http_error(400, "Node is not a ProcessNode", identifier=str(identifier))

    payload: dict[str, Any] = {
        "summary": _get_process_summary(node),
        "inputs": _get_links_dict(node, "incoming"),
        "outputs": _get_links_dict(node, "outgoing"),
        "direct_inputs": _get_direct_links_dict(node, "incoming"),
        "direct_outputs": _get_direct_links_dict(node, "outgoing"),
        "logs": get_process_log_payload(node),
    }

    if isinstance(node, (orm.CalcJobNode, orm.CalcFunctionNode)):
        payload["calculation"] = inspect_calculation_node(node)

    if isinstance(node, orm.WorkflowNode):
        payload["workchain"] = inspect_workchain_node(node)

    return payload
