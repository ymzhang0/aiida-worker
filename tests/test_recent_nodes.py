from __future__ import annotations

from contextlib import nullcontext
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from routers import data as data_router


class _FakeProcessState:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeProcessNode:
    def __init__(
        self,
        *,
        pk: int,
        process_label: str,
        caller: object | None = None,
    ) -> None:
        self.pk = pk
        self.label = ""
        self.process_label = process_label
        self.process_state = _FakeProcessState("finished")
        self.exit_status = 0
        self.ctime = None
        self.caller = caller


class _FakeStructureData:
    def __init__(self, *, pk: int, label: str = "") -> None:
        self.pk = pk
        self.label = label
        self.ctime = None
        self.caller = object()


class _FakeQueryBuilder:
    def __init__(self, rows: list[tuple[object]]) -> None:
        self.rows = rows
        self.append_calls: list[dict[str, object]] = []
        self.filter_calls: list[tuple[str, dict[str, object]]] = []

    def append(self, cls, **kwargs):  # noqa: ANN001
        self.append_calls.append({"cls": cls, **kwargs})
        return self

    def add_filter(self, tag: str, filters: dict[str, object]) -> "_FakeQueryBuilder":
        self.filter_calls.append((tag, filters))
        return self

    def order_by(self, *_args, **_kwargs) -> "_FakeQueryBuilder":
        return self

    def limit(self, *_args, **_kwargs) -> "_FakeQueryBuilder":
        return self

    def all(self) -> list[tuple[object]]:
        return list(self.rows)


class _FakeExtras:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


class _FakeNodeBase:
    def __init__(self, extras: _FakeExtras) -> None:
        self.extras = extras


class _FakeNodeWithExtras:
    def __init__(self, extras: dict[str, object] | None = None, *, pk: int = 99) -> None:
        self.pk = pk
        self.base = _FakeNodeBase(_FakeExtras(extras))


def test_get_recent_processes_filters_root_processes_in_python(monkeypatch) -> None:
    rows = [
        (_FakeProcessNode(pk=1, process_label="PwBandsWorkChain", caller=None),),
        (_FakeProcessNode(pk=2, process_label="PwBaseWorkChain", caller=object()),),
    ]
    fake_qb = _FakeQueryBuilder(rows)

    monkeypatch.setattr(data_router, "QueryBuilder", lambda: fake_qb)
    monkeypatch.setattr(data_router, "db_access_guard", lambda _name: nullcontext())
    monkeypatch.setattr(data_router, "ProcessNode", _FakeProcessNode)

    payload = data_router._get_recent_processes(limit=10, root_only=True)

    assert [item["pk"] for item in payload] == [1]
    assert all("filters" not in call for call in fake_qb.append_calls)


def test_get_recent_nodes_filters_root_processes_in_python(monkeypatch) -> None:
    rows = [
        (_FakeProcessNode(pk=10, process_label="PwBandsWorkChain", caller=None),),
        (_FakeProcessNode(pk=11, process_label="PwBaseWorkChain", caller=object()),),
    ]
    fake_qb = _FakeQueryBuilder(rows)

    monkeypatch.setattr(data_router, "QueryBuilder", lambda: fake_qb)
    monkeypatch.setattr(data_router, "db_access_guard", lambda _name: nullcontext())
    monkeypatch.setattr(data_router, "ProcessNode", _FakeProcessNode)
    monkeypatch.setattr(data_router, "_is_soft_deleted", lambda _node: False)
    monkeypatch.setattr(data_router, "build_node_preview", lambda _node: None)
    monkeypatch.setattr(data_router, "get_structure_formula", lambda _node: None)
    monkeypatch.setattr(data_router, "extract_process_state_value", lambda node: node.process_state.value)
    monkeypatch.setattr(data_router, "_resolve_node_class", lambda _node_type: _FakeProcessNode)
    data_router._clear_recent_nodes_cache()

    payload = data_router._get_recent_nodes(limit=10, node_type="ProcessNode", root_only=True)

    assert [item["pk"] for item in payload] == [10]
    assert fake_qb.filter_calls == []


def test_get_recent_nodes_skips_root_only_filter_for_data_nodes(monkeypatch) -> None:
    rows = [
        (_FakeStructureData(pk=20, label="Si"),),
    ]
    fake_qb = _FakeQueryBuilder(rows)

    monkeypatch.setattr(data_router, "QueryBuilder", lambda: fake_qb)
    monkeypatch.setattr(data_router, "db_access_guard", lambda _name: nullcontext())
    monkeypatch.setattr(data_router, "_is_soft_deleted", lambda _node: False)
    monkeypatch.setattr(data_router, "build_node_preview", lambda _node: None)
    monkeypatch.setattr(data_router, "get_structure_formula", lambda _node: "Si2")
    monkeypatch.setattr(data_router, "ProcessNode", _FakeProcessNode)
    monkeypatch.setattr(data_router, "_resolve_node_class", lambda _node_type: _FakeStructureData)
    data_router._clear_recent_nodes_cache()

    payload = data_router._get_recent_nodes(limit=10, node_type="StructureData", root_only=True)

    assert [item["pk"] for item in payload] == [20]
    assert fake_qb.filter_calls == []


def test_soft_delete_compatibility_reads_legacy_extra_keys() -> None:
    node = _FakeNodeWithExtras({data_router._LEGACY_SOFT_DELETED_EXTRA_KEY: True})

    assert data_router._is_soft_deleted(node) is True


def test_soft_delete_node_writes_and_clears_canonical_and_legacy_keys(monkeypatch) -> None:
    node = _FakeNodeWithExtras()
    monkeypatch.setattr(data_router.orm, "load_node", lambda pk: node)
    monkeypatch.setattr(data_router, "_clear_recent_nodes_cache", lambda: None)

    deleted_payload = data_router._soft_delete_node(99, deleted=True)
    assert deleted_payload == {"pk": 99, "soft_deleted": True}
    assert node.base.extras.values[data_router._SOFT_DELETED_EXTRA_KEY] is True
    assert node.base.extras.values[data_router._LEGACY_SOFT_DELETED_EXTRA_KEY] is True
    assert data_router._SOFT_DELETED_AT_EXTRA_KEY in node.base.extras.values
    assert data_router._LEGACY_SOFT_DELETED_AT_EXTRA_KEY in node.base.extras.values

    restored_payload = data_router._soft_delete_node(99, deleted=False)
    assert restored_payload == {"pk": 99, "soft_deleted": False}
    assert data_router._SOFT_DELETED_EXTRA_KEY not in node.base.extras.values
    assert data_router._LEGACY_SOFT_DELETED_EXTRA_KEY not in node.base.extras.values
    assert data_router._SOFT_DELETED_AT_EXTRA_KEY not in node.base.extras.values
    assert data_router._LEGACY_SOFT_DELETED_AT_EXTRA_KEY not in node.base.extras.values
