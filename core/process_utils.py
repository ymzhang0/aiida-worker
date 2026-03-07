from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import Any, Mapping

from aiida import orm
from aiida.orm import QueryBuilder

from core.engine import http_error
from core.node_utils import get_structure_formula, node_type_name, serialize_node
from core.submission_utils import _load_workflow, _prepare_and_validate
from core.utils import to_jsonable


_PARALLEL_SETTING_KEYS = {
    "num_machines",
    "num_mpiprocs_per_machine",
    "tot_num_mpiprocs",
    "num_cores_per_machine",
    "num_cores_per_mpiproc",
    "npool",
    "nk",
    "ntg",
    "ndiag",
    "withmpi",
    "queue_name",
    "account",
    "qos",
    "max_wallclock_seconds",
}


def _merge_nested_dicts(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = _merge_nested_dicts(existing, value)
        else:
            merged[str(key)] = value
    return merged


def _is_nested_inputs_manager(value: Any) -> bool:
    if hasattr(value, "_get_keys") and hasattr(value, "_get_node_by_link_label"):
        return True
    return hasattr(value, "keys") and not isinstance(value, Mapping)


def _iterate_inputs_manager(manager: Any) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    if hasattr(manager, "_get_keys") and hasattr(manager, "_get_node_by_link_label"):
        try:
            for key in manager._get_keys():
                entries.append((str(key), manager._get_node_by_link_label(key)))
            return entries
        except Exception:
            entries = []

    try:
        keys = manager.keys()
    except Exception:
        return entries

    for key in keys:
        key_text = str(key).strip()
        if not key_text:
            continue
        try:
            entries.append((key_text, manager[key]))
        except Exception:
            continue
    return entries


def _append_pk_map_entry(
    entries: list[dict[str, Any]],
    seen: set[tuple[int, str]],
    *,
    pk: int,
    path: str,
    label: str,
) -> None:
    dedupe_key = (pk, path)
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    entries.append({"pk": int(pk), "path": path, "label": label})


def _append_structure_metadata_entry(
    entries: list[dict[str, Any]],
    seen: set[tuple[int, str]],
    *,
    node: orm.StructureData,
    path: str,
) -> None:
    dedupe_key = (int(node.pk), path)
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)

    atom_count: int | None = None
    with suppress(Exception):
        atom_count = len(node.sites)

    symmetry_label: str | None = None
    with suppress(Exception):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        analyzer = SpacegroupAnalyzer(node.get_pymatgen_structure())
        symbol = str(analyzer.get_space_group_symbol())
        number = int(analyzer.get_space_group_number())
        symmetry_label = f"{symbol} ({number})"

    entries.append(
        {
            "pk": int(node.pk),
            "path": path,
            "label": str(node.label or get_structure_formula(node) or f"Structure #{node.pk}"),
            "formula": get_structure_formula(node),
            "symmetry": symmetry_label,
            "num_atoms": atom_count,
        }
    )


def _serialize_code_input(code: Any) -> str:
    for attr_name in ("full_label",):
        value = getattr(code, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    label = str(getattr(code, "label", "") or "").strip()
    computer = getattr(code, "computer", None)
    computer_label = str(getattr(computer, "label", "") or "").strip() if computer is not None else ""
    if label and computer_label:
        return f"{label}@{computer_label}"
    if label:
        return label
    return str(getattr(code, "pk", "") or "").strip()


def _serialize_submission_node_value(
    node: orm.Node,
    *,
    path: str,
    pk_map_entries: list[dict[str, Any]],
    pk_map_seen: set[tuple[int, str]],
    structure_metadata: list[dict[str, Any]],
    structure_seen: set[tuple[int, str]],
    state: dict[str, Any],
) -> Any:
    if isinstance(node, orm.Dict):
        return to_jsonable(node.get_dict())
    if isinstance(node, orm.List):
        return to_jsonable(node.get_list())
    if isinstance(node, orm.Int):
        return int(node.value)
    if isinstance(node, orm.Float):
        return float(node.value)
    if isinstance(node, orm.Bool):
        return bool(node.value)
    if isinstance(node, orm.Str):
        return str(node.value)

    abstract_code = getattr(orm, "AbstractCode", None)
    if isinstance(abstract_code, type) and isinstance(node, abstract_code):
        computer = getattr(node, "computer", None)
        computer_label = str(getattr(computer, "label", "") or "").strip() if computer is not None else ""
        if computer_label and not state.get("target_computer"):
            state["target_computer"] = computer_label
        return _serialize_code_input(node)
    if isinstance(node, orm.Code):
        computer = getattr(node, "computer", None)
        computer_label = str(getattr(computer, "label", "") or "").strip() if computer is not None else ""
        if computer_label and not state.get("target_computer"):
            state["target_computer"] = computer_label
        return _serialize_code_input(node)

    _append_pk_map_entry(
        pk_map_entries,
        pk_map_seen,
        pk=int(node.pk),
        path=path,
        label=path.rsplit(".", 1)[-1] if path else "pk",
    )

    if isinstance(node, orm.StructureData):
        _append_structure_metadata_entry(
            structure_metadata,
            structure_seen,
            node=node,
            path=path,
        )

    return int(node.pk)


def _serialize_process_inputs(
    manager: Any,
    *,
    prefix: str = "",
    pk_map_entries: list[dict[str, Any]],
    pk_map_seen: set[tuple[int, str]],
    structure_metadata: list[dict[str, Any]],
    structure_seen: set[tuple[int, str]],
    state: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in _iterate_inputs_manager(manager):
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, orm.Node):
            payload[key] = _serialize_submission_node_value(
                value,
                path=path,
                pk_map_entries=pk_map_entries,
                pk_map_seen=pk_map_seen,
                structure_metadata=structure_metadata,
                structure_seen=structure_seen,
                state=state,
            )
            continue
        if _is_nested_inputs_manager(value):
            payload[key] = _serialize_process_inputs(
                value,
                prefix=path,
                pk_map_entries=pk_map_entries,
                pk_map_seen=pk_map_seen,
                structure_metadata=structure_metadata,
                structure_seen=structure_seen,
                state=state,
            )
            continue
        payload[key] = to_jsonable(value)
    return payload


def _collect_parallel_settings(payload: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                lowered = str(key).strip().lower()
                if lowered in _PARALLEL_SETTING_KEYS and lowered not in settings:
                    settings[lowered] = to_jsonable(value)
                if isinstance(value, Mapping):
                    walk(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, (Mapping, list)):
                            walk(item)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (Mapping, list)):
                    walk(item)

    walk(payload)
    return settings


def _build_clone_validation_payload(entry_point: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        workflow = _load_workflow(entry_point)
        _, validation_error = _prepare_and_validate(workflow, inputs)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "VALIDATION_FAILED",
            "is_valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "source": "process.clone_draft",
            "entry_point": entry_point,
        }

    if validation_error is None:
        return {
            "status": "VALIDATION_OK",
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "source": "process.clone_draft",
            "entry_point": entry_point,
        }

    message = str(getattr(validation_error, "message", "") or str(validation_error)).strip()
    return {
        "status": "VALIDATION_FAILED",
        "is_valid": False,
        "errors": [message] if message else [str(validation_error)],
        "warnings": [],
        "source": "process.clone_draft",
        "entry_point": entry_point,
        "port": str(getattr(validation_error, "port", "") or ""),
        "full_error": str(validation_error),
    }


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
        direct_inputs = _get_direct_links_dict(self.node, "incoming")
        direct_outputs = _get_direct_links_dict(self.node, "outgoing")
        summary["inputs"] = direct_inputs
        summary["outputs"] = direct_outputs
        summary["direct_inputs"] = direct_inputs
        summary["direct_outputs"] = direct_outputs
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


def build_process_clone_payload(identifier_or_node: int | str | orm.ProcessNode) -> dict[str, Any]:
    if isinstance(identifier_or_node, orm.ProcessNode):
        node = identifier_or_node
    else:
        try:
            node = orm.load_node(identifier_or_node)
        except Exception as exc:  # noqa: BLE001
            raise http_error(404, "Process node not found", identifier=str(identifier_or_node), reason=str(exc)) from exc

    if not isinstance(node, orm.ProcessNode):
        raise http_error(400, "Node is not a ProcessNode", identifier=str(getattr(node, "pk", identifier_or_node)))
    if not isinstance(node, orm.WorkflowNode):
        raise http_error(
            422,
            "Clone & Edit is only supported for workflow processes",
            identifier=str(getattr(node, "pk", identifier_or_node)),
            node_type=node.__class__.__name__,
        )

    entry_point = str(getattr(node, "process_type", "") or "").strip()
    if not entry_point:
        raise http_error(422, "Workflow entry point is unavailable for this process", identifier=str(node.pk))

    pk_map_entries: list[dict[str, Any]] = []
    pk_map_seen: set[tuple[int, str]] = set()
    structure_metadata: list[dict[str, Any]] = []
    structure_seen: set[tuple[int, str]] = set()
    state: dict[str, Any] = {"target_computer": None}

    inputs = _serialize_process_inputs(
        node.inputs,
        pk_map_entries=pk_map_entries,
        pk_map_seen=pk_map_seen,
        structure_metadata=structure_metadata,
        structure_seen=structure_seen,
        state=state,
    )

    metadata_inputs = node.base.attributes.all.get("metadata_inputs")
    if isinstance(metadata_inputs, Mapping):
        inputs = _merge_nested_dicts(inputs, to_jsonable(dict(metadata_inputs)))

    validation = _build_clone_validation_payload(entry_point, inputs)
    process_label = str(getattr(node, "process_label", "") or entry_point or "AiiDA Workflow").strip()
    parallel_settings = _collect_parallel_settings(inputs)

    return {
        "process_label": process_label or "AiiDA Workflow",
        "entry_point": entry_point,
        "inputs": inputs,
        "recommended_inputs": {},
        "advanced_settings": {},
        "meta": {
            "draft": {
                "entry_point": entry_point,
                "inputs": inputs,
            },
            "entry_point": entry_point,
            "workchain": entry_point,
            "workchain_entry_point": entry_point,
            "pk_map": pk_map_entries,
            "target_computer": state.get("target_computer"),
            "parallel_settings": parallel_settings,
            "structure_metadata": structure_metadata,
            "validation": validation,
            "source_process_pk": int(node.pk),
            "source_process_uuid": str(node.uuid),
        },
    }


def inspect_process_payload(identifier: str | int) -> dict[str, Any]:
    try:
        node = orm.load_node(identifier)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Node not found", identifier=str(identifier), reason=str(exc)) from exc

    # Use serialize_node for a consistent summary
    summary = serialize_node(node)
    
    payload: dict[str, Any] = {
        "summary": summary,
        "inputs": _get_links_dict(node, "incoming"),
        "outputs": _get_links_dict(node, "outgoing"),
        "direct_inputs": _get_direct_links_dict(node, "incoming"),
        "direct_outputs": _get_direct_links_dict(node, "outgoing"),
    }

    if isinstance(node, orm.ProcessNode):
        payload["logs"] = get_process_log_payload(node)
        
        if isinstance(node, (orm.CalcJobNode, orm.CalcFunctionNode)):
            payload["calculation"] = inspect_calculation_node(node)

        if isinstance(node, orm.WorkflowNode):
            payload["workchain"] = inspect_workchain_node(node)
    else:
        # Non-process nodes don't have logs or complex execution info
        payload["logs"] = {"pk": int(node.pk), "text": "No logs for data nodes.", "lines": [], "reports": []}

    return payload
