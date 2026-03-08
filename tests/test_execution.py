from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.engine import cleanup_storage_session, ensure_profile_loaded, reset_storage_backend_caches
from routers.execution import (
    WORKSPACE_PATH_HEADER,
    _execute_python_script,
    _request_workspace_path,
)


@pytest.fixture(autouse=True)
def _cleanup_execution_test_state() -> None:
    reset_storage_backend_caches()
    cleanup_storage_session()
    yield
    reset_storage_backend_caches()
    cleanup_storage_session()


def _run_script_with_cleanup(script: str) -> dict[str, object]:
    try:
        return _execute_python_script(script)
    finally:
        cleanup_storage_session()


def test_execute_python_script_supports_backend_user_collection_compatibility() -> None:
    ensure_profile_loaded()
    reset_storage_backend_caches()
    script = """
from aiida.manage.manager import get_manager

storage = get_manager().get_profile_storage()
backend = getattr(storage, '_backend', storage)
default_user = backend.users.get_default()
all_users = backend.users.all()
print(default_user.email)
print(len(all_users))
"""

    result = _run_script_with_cleanup(script)

    assert result["success"] is True
    output = str(result["output"])
    assert "@" in output
    assert any(char.isdigit() for char in output)


def test_execute_python_script_serializes_concurrent_store_operations() -> None:
    ensure_profile_loaded()
    reset_storage_backend_caches()
    script = """
from aiida import orm

structure = orm.StructureData(cell=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
structure.append_atom(position=(0.0, 0.0, 0.0), symbols='Si')
structure.label = 'codex-run-python-concurrency'
structure.store()
print(structure.pk)
"""

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _index: _run_script_with_cleanup(script), range(8)))

    failures = [result for result in results if not result.get("success")]
    assert failures == []


def test_request_workspace_path_accepts_aris_header() -> None:
    assert _request_workspace_path(SimpleNamespace(headers={WORKSPACE_PATH_HEADER: "/tmp/aris-workspace"})) == (
        "/tmp/aris-workspace"
    )
