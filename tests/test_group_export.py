from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from aiida import orm

from core.engine import ensure_profile_loaded, reset_storage_backend_caches
from routers.data import _export_group_archive_response


def test_export_group_archive_response_returns_aiida_file() -> None:
    ensure_profile_loaded()
    reset_storage_backend_caches()

    structure = orm.StructureData(cell=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si")
    structure.label = f"codex-group-export-structure-{uuid4().hex[:8]}"
    structure.store()

    group = orm.Group(label=f"codex-group-export-{uuid4().hex[:8]}").store()
    group.add_nodes([structure])

    response = _export_group_archive_response(int(group.pk))
    archive_path = Path(response.path)

    try:
        assert response.filename is not None
        assert response.filename.endswith(".aiida")
        assert archive_path.exists()
        assert archive_path.suffix == ".aiida"
        assert archive_path.stat().st_size > 0
    finally:
        with suppress(FileNotFoundError):
            archive_path.unlink()
