from __future__ import annotations

import io
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout, suppress
from types import MethodType
from typing import Any

from aiida import orm
from fastapi import Request

from core.engine import (
    SessionCleanupAPIRouter,
    cleanup_storage_session,
    ensure_profile_loaded,
    http_error,
    reset_storage_backend_caches,
)
from core.scripts import execute_registered_script
from models.schemas import PythonScriptRequest, ScriptExecuteRequest
from repository.analysis.common_utils import (
    capture_saved_artifacts,
    activate_workspace_path,
    persist_plot_artifacts,
    save_artifact,
)

execution_router = SessionCleanupAPIRouter(tags=["execution"])
management_execution_router = SessionCleanupAPIRouter(prefix="/management", tags=["management"])
WORKSPACE_PATH_HEADER = "X-ARIS-Active-Workspace-Path"
_RUN_PYTHON_LOCK = threading.Lock()
_COMPAT_ORIGINAL_MISSING = object()
_COMPAT_BACKUP_UNSET = object()
_COMPAT_METHOD_NAMES = ("get_default", "all", "find", "get")


def _request_workspace_path(request: Request | None) -> str | None:
    if request is None:
        return None
    cleaned = str(request.headers.get(WORKSPACE_PATH_HEADER) or "").strip()
    return cleaned or None


def _list_profile_users() -> list[orm.User]:
    collection = orm.User.collection
    list_all = getattr(collection, "all", None)
    if callable(list_all):
        users = list(list_all())
        if users:
            return users

    find_users = getattr(collection, "find", None)
    if callable(find_users):
        users = list(find_users())
        if users:
            return users

    qb = orm.QueryBuilder()
    qb.append(orm.User, project=["*"])
    return [row[0] for row in qb.all()]


def _get_profile_default_user() -> orm.User | None:
    from aiida.manage.configuration import get_profile

    default_email = str(getattr(get_profile(), "default_user_email", "") or "").strip()
    if default_email:
        try:
            return orm.User.collection.get(email=default_email)
        except Exception:  # noqa: BLE001
            pass

    get_default = getattr(orm.User.collection, "get_default", None)
    if callable(get_default):
        try:
            return get_default()
        except Exception:  # noqa: BLE001
            pass

    users = _list_profile_users()
    return users[0] if users else None


def _match_user_filter(user: orm.User, filters: dict[str, Any]) -> bool:
    for raw_key, expected in filters.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        actual = getattr(user, key, None)
        if isinstance(expected, dict):
            if "like" in expected:
                needle = str(expected.get("like") or "")
                haystack = str(actual or "")
                if needle.startswith("%") and needle.endswith("%"):
                    if needle.strip("%") not in haystack:
                        return False
                    continue
                if needle.startswith("%"):
                    if not haystack.endswith(needle[1:]):
                        return False
                    continue
                if needle.endswith("%"):
                    if not haystack.startswith(needle[:-1]):
                        return False
                    continue
                if haystack != needle:
                    return False
                continue
            return False
        if actual != expected:
            return False
    return True


def _find_profile_users(filters: dict[str, Any] | None = None) -> list[orm.User]:
    users = _list_profile_users()
    if not filters:
        return users
    return [user for user in users if _match_user_filter(user, filters)]


def _get_profile_user_by_filters(filters: dict[str, Any]) -> orm.User:
    matches = _find_profile_users(filters)
    if not matches:
        raise ValueError(f"No user matched filters: {filters}")
    if len(matches) > 1:
        raise ValueError(f"Multiple users matched filters: {filters}")
    return matches[0]


def _install_user_collection_compatibility(collection: Any) -> bool:
    if collection is None:
        return False

    installed = False
    method_factories = {
        "get_default": lambda: MethodType(lambda _self: _get_profile_default_user(), collection),
        "all": lambda: MethodType(lambda _self: list(_list_profile_users()), collection),
        "find": lambda: MethodType(
            lambda _self, filters=None: list(_find_profile_users(filters if isinstance(filters, dict) else None)),
            collection,
        ),
        "get": lambda: MethodType(
            lambda _self, **filters: _get_profile_user_by_filters(
                {str(key): value for key, value in filters.items() if str(key).strip()}
            ),
            collection,
        ),
    }

    for method_name, factory in method_factories.items():
        current = getattr(collection, method_name, _COMPAT_ORIGINAL_MISSING)
        if callable(current):
            continue
        setattr(collection, f"_aris_execution_original_{method_name}", current)
        setattr(collection, method_name, factory())
        installed = True

    return installed


def _restore_user_collection_compatibility(collections: tuple[Any, ...]) -> None:
    for collection in collections:
        if collection is None:
            continue
        for method_name in _COMPAT_METHOD_NAMES:
            backup_attr = f"_aris_execution_original_{method_name}"
            original = getattr(collection, backup_attr, _COMPAT_BACKUP_UNSET)
            if original is _COMPAT_BACKUP_UNSET:
                continue
            if original is _COMPAT_ORIGINAL_MISSING:
                with suppress(Exception):
                    delattr(collection, method_name)
            else:
                with suppress(Exception):
                    setattr(collection, method_name, original)
            with suppress(Exception):
                delattr(collection, backup_attr)


def _clear_bound_default_users(exec_globals: dict[str, Any]) -> None:
    for owner_key in ("storage", "backend"):
        owner = exec_globals.get(owner_key)
        if owner is None or not hasattr(owner, "_default_user"):
            continue
        with suppress(Exception):
            setattr(owner, "_default_user", None)


def _install_run_python_compatibility_bindings(exec_globals: dict[str, Any]) -> None:
    from aiida.manage.manager import get_manager

    storage = get_manager().get_profile_storage()
    backend = getattr(storage, "_backend", storage)
    reset_storage_backend_caches()
    default_user = _get_profile_default_user()

    for owner in (storage, backend):
        if owner is None or default_user is None:
            continue
        if hasattr(owner, "_default_user"):
            setattr(owner, "_default_user", default_user)

    patched_collections: list[Any] = []
    for owner in {storage, backend, getattr(storage, "users", None), getattr(backend, "users", None)}:
        if owner is None:
            continue
        users = owner if owner.__class__.__name__.endswith("UserCollection") else getattr(owner, "users", None)
        if users is not None and _install_user_collection_compatibility(users):
            patched_collections.append(users)

    exec_globals["storage"] = storage
    exec_globals["backend"] = backend
    exec_globals["get_default_user"] = _get_profile_default_user
    exec_globals["list_users"] = _list_profile_users
    exec_globals["_patched_user_collections"] = tuple(patched_collections)


def _execute_python_script(script: str, *, workspace_path: str | None = None) -> dict[str, Any]:
    code = str(script or "")
    if not code.strip():
        raise http_error(400, "Python script is required")

    from aiida import engine as aiida_engine
    from aiida import plugins as aiida_plugins

    exec_globals: dict[str, Any] = {
        "orm": orm,
        "plugins": aiida_plugins,
        "engine": aiida_engine,
        "save_artifact": save_artifact,
        "ACTIVE_WORKSPACE_PATH": workspace_path,
    }
    output_buffer = io.StringIO()
    saved_artifacts: list[Any] = []

    with _RUN_PYTHON_LOCK:
        try:
            _install_run_python_compatibility_bindings(exec_globals)
            with (
                activate_workspace_path(workspace_path),
                capture_saved_artifacts() as captured_saved_artifacts,
                redirect_stdout(output_buffer),
                redirect_stderr(output_buffer),
            ):
                saved_artifacts = captured_saved_artifacts
                exec(code, exec_globals)
                if not saved_artifacts:
                    persist_plot_artifacts(exec_globals, prefix="auto-plot")
            return {
                "success": True,
                "output": output_buffer.getvalue() or "Code executed successfully (No output).",
                "workspace_path": workspace_path,
                "artifacts": saved_artifacts,
            }
        except Exception:  # noqa: BLE001
            reset_storage_backend_caches()
            return {
                "success": False,
                "output": output_buffer.getvalue(),
                "workspace_path": workspace_path,
                "error": traceback.format_exc(),
            }
        finally:
            _restore_user_collection_compatibility(tuple(exec_globals.get("_patched_user_collections", ())))
            _clear_bound_default_users(exec_globals)
            cleanup_storage_session()
            reset_storage_backend_caches()


@management_execution_router.post("/run-python")
def management_run_python(payload: PythonScriptRequest, request: Request) -> dict[str, Any]:
    ensure_profile_loaded()
    return _execute_python_script(payload.script, workspace_path=_request_workspace_path(request))


@execution_router.post("/execute/{script_name}")
async def execute_script(script_name: str, payload: ScriptExecuteRequest, request: Request) -> dict[str, Any]:
    ensure_profile_loaded()
    return await execute_registered_script(
        script_name,
        params=payload.params,
        workspace_path=_request_workspace_path(request),
    )
