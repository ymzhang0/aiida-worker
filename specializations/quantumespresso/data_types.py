from __future__ import annotations

# Quantum ESPRESSO-specific data entry points live here so router/core code stays generic.
DATA_ENTRY_POINT_ALIASES: dict[str, str] = {
    "UpfData": "pseudo.upf",
}


__all__ = ["DATA_ENTRY_POINT_ALIASES"]
