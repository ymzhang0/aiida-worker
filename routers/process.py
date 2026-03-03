from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import Any, Mapping

from aiida import orm
from aiida.orm import QueryBuilder

from core.engine import SessionCleanupAPIRouter, ensure_profile_loaded, http_error
from core.node_utils import node_type_name, serialize_node

process_router = SessionCleanupAPIRouter(prefix="/process", tags=["process"])


class _ProcessTree:
    """Recursively build and serialize process provenance trees."""

    def __init__(self, node: orm.ProcessNode):
        self.node = node
        self.children: dict[str, _ProcessTree] = {}

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
                self.children[unique_label] = _ProcessTree(sub)

    def to_dict(self) -> dict[str, Any]:
        process_state = getattr(self.node, "process_state", None)
        state_value = process_state.value if hasattr(process_state, "value") else str(process_state or "N/A")
        
        inputs: list[dict[str, Any]] = []
        for link in self.node.base.links.get_incoming().all():
            serialized = serialize_node(link.node)
            serialized["link_label"] = str(link.link_label)
            # Ensure pk is int for JSON consistency
            serialized["pk"] = int(serialized["pk"])
            inputs.append(serialized)
            
        outputs: list[dict[str, Any]] = []
        for link in self.node.base.links.get_outgoing().all():
            serialized = serialize_node(link.node)
            serialized["link_label"] = str(link.link_label)
            serialized["pk"] = int(serialized["pk"])
            outputs.append(serialized)

        return {
            "pk": int(self.node.pk),
            "process_label": str(getattr(self.node, "process_label", "N/A") or "N/A"),
            "state": state_value,
            "exit_status": getattr(self.node, "exit_status", None),
            "inputs": inputs,
            "outputs": outputs,
            "children": {label: child.to_dict() for label, child in self.children.items()},
        }


def _get_process_log_payload(identifier: int | str) -> dict[str, Any]:
    try:
        node = orm.load_node(identifier)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Process node not found", identifier=str(identifier), reason=str(exc)) from exc

    if not isinstance(node, orm.ProcessNode):
        raise http_error(400, "Node is not a ProcessNode", identifier=str(identifier))

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
        "summary": serialize_node(node),
        "inputs": {},
        "outputs": {},
        "repository_files": [],
    }

    incoming_counts: defaultdict[str, int] = defaultdict(int)
    for link in node.base.links.get_incoming().all():
        label = str(link.link_label)
        unique_label = f"{label}_{incoming_counts[label]}" if incoming_counts[label] > 0 else label
        incoming_counts[label] += 1
        payload["inputs"][unique_label] = serialize_node(link.node)

    outgoing_counts: defaultdict[str, int] = defaultdict(int)
    for link in node.base.links.get_outgoing().all():
        label = str(link.link_label)
        unique_label = f"{label}_{outgoing_counts[label]}" if outgoing_counts[label] > 0 else label
        outgoing_counts[label] += 1
        payload["outputs"][unique_label] = serialize_node(link.node)

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
        raise http_error(404, "Process node not found", identifier=str(identifier), reason=str(exc)) from exc

    if not isinstance(node, orm.ProcessNode):
        raise http_error(400, "Node is not a ProcessNode", identifier=str(identifier))

    process_state = getattr(node, "process_state", None)
    summary = {
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "type": node_type_name(node),
        "process_label": str(getattr(node, "process_label", None) or "N/A"),
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


@process_router.get("/{identifier}")
def inspect_process(identifier: str) -> dict[str, Any]:
    ensure_profile_loaded()
    return _inspect_process_payload(identifier)


@process_router.get("/{identifier}/logs")
def inspect_process_logs(identifier: str) -> dict[str, Any]:
    ensure_profile_loaded()
    return _get_process_log_payload(identifier)
