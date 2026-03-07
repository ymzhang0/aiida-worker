from __future__ import annotations

import contextvars
import io
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ACTIVE_WORKSPACE_ENV_VAR = "ACTIVE_WORKSPACE_PATH"
_WORKSPACE_ENV_LOCK = threading.RLock()
_SAVED_ARTIFACTS: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "saved_artifacts",
    default=None,
)


@contextmanager
def activate_workspace_path(path: str | None):
    cleaned = str(path or "").strip()
    previous = os.environ.get(ACTIVE_WORKSPACE_ENV_VAR)
    with _WORKSPACE_ENV_LOCK:
        if cleaned:
            os.environ[ACTIVE_WORKSPACE_ENV_VAR] = cleaned
        else:
            os.environ.pop(ACTIVE_WORKSPACE_ENV_VAR, None)
        try:
            yield cleaned or None
        finally:
            if previous is None:
                os.environ.pop(ACTIVE_WORKSPACE_ENV_VAR, None)
            else:
                os.environ[ACTIVE_WORKSPACE_ENV_VAR] = previous


@contextmanager
def capture_saved_artifacts():
    token = _SAVED_ARTIFACTS.set([])
    bucket = _SAVED_ARTIFACTS.get()
    try:
        yield bucket
    finally:
        _SAVED_ARTIFACTS.reset(token)


def get_active_workspace_path(*, create: bool = True) -> Path:
    raw_path = str(os.getenv(ACTIVE_WORKSPACE_ENV_VAR) or "").strip()
    workspace = Path(raw_path).expanduser().resolve() if raw_path else Path.cwd().resolve()
    if create:
        workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _is_matplotlib_figure(value: Any) -> bool:
    try:
        from matplotlib.figure import Figure
    except Exception:  # noqa: BLE001
        return False
    return isinstance(value, Figure)


def _is_plotly_figure(value: Any) -> bool:
    try:
        from plotly.basedatatypes import BaseFigure
    except Exception:  # noqa: BLE001
        return False
    return isinstance(value, BaseFigure)


def _resolve_workspace_target(filename: str, data: Any) -> Path:
    cleaned = str(filename or "").strip()
    if not cleaned:
        raise ValueError("filename is required")

    relative = Path(cleaned)
    if relative.is_absolute():
        raise ValueError("filename must be relative to the active workspace")

    default_suffix = ".json"
    if isinstance(data, str):
        default_suffix = ".txt"
    elif isinstance(data, (bytes, bytearray, memoryview)):
        default_suffix = ".bin"
    elif _is_matplotlib_figure(data):
        default_suffix = ".png"
    elif _is_plotly_figure(data):
        default_suffix = ".html"

    if not relative.suffix:
        relative = relative.with_suffix(default_suffix)
    elif _is_plotly_figure(data) and relative.suffix.lower() not in {".html", ".json"}:
        relative = relative.with_suffix(".html")

    workspace = get_active_workspace_path(create=True)
    target = (workspace / relative).resolve()
    target.relative_to(workspace)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_artifact(filename: str, data: Any) -> dict[str, Any]:
    target = _resolve_workspace_target(filename, data)
    payload: bytes | str
    binary = False

    if isinstance(data, str):
        payload = data
    elif isinstance(data, (bytes, bytearray, memoryview)):
        payload = bytes(data)
        binary = True
    elif _is_matplotlib_figure(data):
        buffer = io.BytesIO()
        data.savefig(buffer, format=target.suffix.lstrip(".") or "png", bbox_inches="tight")
        payload = buffer.getvalue()
        binary = True
    elif _is_plotly_figure(data):
        payload = data.to_json() if target.suffix.lower() == ".json" else data.to_html(include_plotlyjs="cdn", full_html=True)
    else:
        payload = json.dumps(_json_ready(data), ensure_ascii=True, indent=2) + "\n"

    if binary:
        target.write_bytes(payload if isinstance(payload, bytes) else str(payload).encode("utf-8"))
    else:
        target.write_text(str(payload), encoding="utf-8")

    artifact = {
        "path": str(target),
        "relative_path": str(target.relative_to(get_active_workspace_path(create=True))),
        "size_bytes": int(target.stat().st_size),
    }
    bucket = _SAVED_ARTIFACTS.get()
    if isinstance(bucket, list):
        bucket.append(artifact)
    return artifact


def persist_plot_artifacts(namespace: Mapping[str, Any], *, prefix: str = "plot") -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    seen: set[int] = set()
    matplotlib_index = 0
    plotly_index = 0

    for value in namespace.values():
        marker = id(value)
        if marker in seen:
            continue
        if _is_matplotlib_figure(value):
            seen.add(marker)
            matplotlib_index += 1
            saved.append(save_artifact(f"{prefix}-matplotlib-{matplotlib_index}.png", value))
            continue
        if _is_plotly_figure(value):
            seen.add(marker)
            plotly_index += 1
            saved.append(save_artifact(f"{prefix}-plotly-{plotly_index}.html", value))

    return saved


def kbar_to_gpa(value):
    """Convert pressure/modulus from kbar to GPa."""
    return value / 10.0


def calculate_pettifor_ratio(c12, c44, modulus):
    """Calculate Pettifor ratio (C12 - C44) / modulus."""
    if modulus == 0:
        return 0.0
    return (c12 - c44) / modulus


def calculate_pugh_ratio(bulk_modulus, shear_modulus):
    """Calculate Pugh ratio B/G."""
    if shear_modulus == 0:
        return 0.0
    return bulk_modulus / shear_modulus


def calculate_cubic_elastic_averages(c11, c12, c44):
    """
    Calculate Voigt, Reuss, and Hill averages for a cubic system.
    Returns a dictionary with bulk_modulus, shear_modulus, young_modulus, and poisson_ratio.
    """
    B = (c11 + 2 * c12) / 3.0
    G_V = (c11 - c12 + 3 * c44) / 5.0
    G_R = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12))
    G_H = (G_V + G_R) / 2.0

    averages = {}
    for label, G in [("voigt", G_V), ("reuss", G_R), ("hill", G_H)]:
        E = 9.0 * B * G / (3.0 * B + G) if (3.0 * B + G) != 0 else 0.0
        nu = (3.0 * B - 2.0 * G) / (2.0 * (3.0 * B + G)) if (3.0 * B + G) != 0 else 0.0
        averages[label] = {
            "bulk_modulus": B,
            "shear_modulus": G,
            "young_modulus": E,
            "poisson_ratio": nu,
            "pugh_ratio": B / G if G != 0 else 0.0,
        }
    return averages


def standardize_modulus_names(modulus_dict):
    """
    Rename keys from thermo_pw style to a standard internal style.
    Handles mapping like 'bulk_modulus_B' -> 'bulk_modulus'.
    """
    mapping = {
        "bulk_modulus_B": "bulk_modulus",
        "shear_modulus_G": "shear_modulus",
        "young_modulus_E": "young_modulus",
        "poisson_ratio_n": "poisson_ratio",
        "pugh_ratio_r": "pugh_ratio",
    }
    new_dict = {}
    for k, v in modulus_dict.items():
        new_key = mapping.get(k, k)
        new_dict[new_key] = v
    return new_dict
