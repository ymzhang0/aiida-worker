from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from typing import Any

from aiida import orm

from core.engine import http_error
from core.utils import to_jsonable


def node_type_name(node: orm.Node) -> str:
    node_type = str(getattr(node, "node_type", node.__class__.__name__))
    if "." in node_type:
        parts = node_type.split(".")
        if len(parts) >= 2:
            return parts[-2]
    return str(node.__class__.__name__)


def extract_process_state_value(node: orm.Node, *, default: str = "unknown") -> str:
    process_state = getattr(node, "process_state", None)
    return process_state.value if hasattr(process_state, "value") else str(process_state or default)


def compute_process_execution_time_seconds(node: orm.ProcessNode, state: str | None = None) -> float | None:
    ctime = getattr(node, "ctime", None)
    if not isinstance(ctime, datetime):
        return None

    normalized_state = str(state or extract_process_state_value(node, default="unknown")).strip().lower()
    if normalized_state in {"created", "running", "waiting"}:
        end_time = datetime.now(tz=ctime.tzinfo)
    else:
        mtime = getattr(node, "mtime", None)
        end_time = mtime if isinstance(mtime, datetime) else None

    if not isinstance(end_time, datetime):
        return None

    elapsed = (end_time - ctime).total_seconds()
    return round(max(0.0, float(elapsed)), 3)


def get_structure_formula(node: orm.StructureData) -> str | None:
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


def shorten_path_for_preview(path: str | None, *, depth: int = 2) -> str | None:
    cleaned = str(path or "").strip()
    if not cleaned:
        return None
    segments = [segment for segment in cleaned.rstrip("/").split("/") if segment]
    if not segments:
        return cleaned
    return f".../{'/'.join(segments[-max(1, depth):])}"


def build_node_preview(node: orm.Node) -> dict[str, Any] | None:
    if isinstance(node, orm.StructureData):
        atom_count: int | None = None
        with suppress(Exception):
            atom_count = len(node.sites)
        return {
            "formula": get_structure_formula(node),
            "atom_count": atom_count,
        }

    if isinstance(node, orm.BandsData):
        num_bands: int | None = None
        num_kpoints: int | None = None

        with suppress(Exception):
            bands = node.get_bands()
            shape = getattr(bands, "shape", None)
            if shape:
                if len(shape) >= 2:
                    num_kpoints = int(shape[-2])
                    num_bands = int(shape[-1])
                elif len(shape) == 1:
                    num_bands = int(shape[0])

        with suppress(Exception):
            kpoints = node.get_kpoints()
            num_kpoints = len(kpoints)

        return {
            "num_bands": num_bands,
            "num_kpoints": num_kpoints,
        }

    if isinstance(node, orm.ArrayData):
        array_names: list[str] = []
        array_shapes: list[list[int] | None] = []
        with suppress(Exception):
            for name in node.get_arraynames():
                array_names.append(str(name))
                shape: list[int] | None = None
                with suppress(Exception):
                    shape = [int(dimension) for dimension in node.get_shape(name)]
                if shape is None:
                    with suppress(Exception):
                        array = node.get_array(name)
                        shape = [int(dimension) for dimension in getattr(array, "shape", ())]
                array_shapes.append(shape)
        return {
            "arrays": array_names,
            "shapes": array_shapes,
        }

    if isinstance(node, orm.RemoteData):
        computer_label: str | None = None
        with suppress(Exception):
            computer = getattr(node, "computer", None)
            if computer is not None:
                computer_label = str(getattr(computer, "label", None) or getattr(computer, "name", None) or "")
        remote_path: str | None = None
        with suppress(Exception):
            remote_path = str(node.get_remote_path())
        return {
            "computer": computer_label or None,
            "path": shorten_path_for_preview(remote_path),
        }

    if isinstance(node, orm.FolderData):
        file_names: list[str] = []
        with suppress(Exception):
            file_names = sorted([str(name) for name in node.list_object_names()])
        return {
            "file_count": len(file_names),
            "files": file_names[:3],
        }

    if isinstance(node, orm.ProcessNode):
        state = extract_process_state_value(node)
        return {
            "state": state,
            "execution_time_seconds": compute_process_execution_time_seconds(node, state=state),
        }

    if isinstance(node, orm.Dict):
        d = node.get_dict()
        keys = list(d.keys())
        return {
            "keys": keys[:5],
            "count": len(keys),
            "summary": str(d)[:100] + ("..." if len(str(d)) > 100 else "")
        }

    if isinstance(node, (orm.List, orm.ArrayData)):
        try:
            if isinstance(node, orm.List):
                l = node.get_list()
                return {"count": len(l), "summary": str(l)[:100] + ("..." if len(str(l)) > 100 else "")}
        except Exception:
            pass

    if isinstance(node, (orm.Int, orm.Float, orm.Str, orm.Bool)):
        return {"value": str(node.value)}

    return None


def extract_node_payload(node: orm.Node) -> Any:
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
                    "mesh": to_jsonable(mesh),
                    "offset": to_jsonable(offset),
                }
            except Exception:
                with suppress(Exception):
                    kpoints = node.get_kpoints()
                    payload = {
                        "mode": "list",
                        "num_points": len(kpoints),
                        "points": to_jsonable(kpoints.tolist()),
                    }
    except Exception:  # noqa: BLE001
        payload = "Error loading content"

    return to_jsonable(payload)


def serialize_node(node: orm.Node) -> dict[str, Any]:
    type_name = node_type_name(node)
    info: dict[str, Any] = {
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "type": type_name,
        "node_type": type_name,
        "full_type": str(getattr(node, "node_type", node.__class__.__name__)),
        "label": str(getattr(node, "label", None) or ""),
        "ctime": node.ctime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "ctime", None) else None,
    }

    if not info["label"]:
        if isinstance(node, (orm.Int, orm.Float, orm.Str, orm.Bool)):
            info["label"] = str(node.value)
        elif isinstance(node, orm.Code):
            info["label"] = node.label
        else:
            info["label"] = type_name

    payload = extract_node_payload(node)
    if payload is not None:
        info["payload"] = payload

    preview = build_node_preview(node)
    if preview is not None:
        info["preview_info"] = preview
        info["preview"] = preview

    if isinstance(node, orm.ProcessNode):
        info["state"] = extract_process_state_value(node)
        info["exit_status"] = getattr(node, "exit_status", None)
        info["process_label"] = str(getattr(node, "process_label", None) or "N/A")

    return info


def get_node_summary(node_pk: int) -> dict[str, Any]:
    try:
        node = orm.load_node(node_pk)
    except Exception as exc:  # noqa: BLE001
        raise http_error(404, "Node not found", pk=node_pk, reason=str(exc)) from exc

    try:
        incoming_count = len(node.base.links.get_incoming().all())
    except Exception:  # noqa: BLE001
        incoming_count = 0

    try:
        outgoing_count = len(node.base.links.get_outgoing().all())
    except Exception:  # noqa: BLE001
        outgoing_count = 0

    state_value = extract_process_state_value(node, default="N/A")
    preview = build_node_preview(node)

    return {
        "pk": int(node.pk),
        "uuid": str(node.uuid),
        "type": node_type_name(node),
        "full_type": str(getattr(node, "node_type", node.__class__.__name__)),
        "ctime": node.ctime.strftime("%Y-%m-%d %H:%M:%S") if getattr(node, "ctime", None) else None,
        "label": str(getattr(node, "label", None) or "(No Label)"),
        "state": state_value,
        "process_label": str(getattr(node, "process_label", None) or "N/A"),
        "exit_status": getattr(node, "exit_status", "N/A"),
        "incoming": incoming_count,
        "outgoing": outgoing_count,
        "attributes": to_jsonable(node.base.attributes.all),
        "preview_info": preview,
        "preview": preview,
    }
