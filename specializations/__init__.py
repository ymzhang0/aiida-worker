from __future__ import annotations

from contextlib import suppress
from typing import Dict


def get_data_entry_point_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    with suppress(Exception):
        from .quantumespresso.data_types import DATA_ENTRY_POINT_ALIASES as qe_aliases

        aliases.update(qe_aliases)
    return aliases


__all__ = ["get_data_entry_point_aliases"]
