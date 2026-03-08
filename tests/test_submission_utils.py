from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core import submission_utils


class _FakeNode:
    def __init__(self, *, pk: int | None = None, uuid: str = "uuid", label: str = "") -> None:
        self.pk = pk
        self.uuid = uuid
        self.label = label


class _FakeDict(_FakeNode):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(pk=None, uuid="dict-uuid", label="dict")
        self._payload = payload

    def get_dict(self) -> dict[str, object]:
        return self._payload


class _FakeList(_FakeNode):
    def __init__(self, payload: list[object]) -> None:
        super().__init__(pk=None, uuid="list-uuid", label="list")
        self._payload = payload

    def get_list(self) -> list[object]:
        return self._payload


class _FakeInt(_FakeNode):
    def __init__(self, value: int) -> None:
        super().__init__(pk=None, uuid="int-uuid", label="int")
        self.value = value


class _FakeFloat(_FakeNode):
    def __init__(self, value: float) -> None:
        super().__init__(pk=None, uuid="float-uuid", label="float")
        self.value = value


class _FakeBool(_FakeNode):
    def __init__(self, value: bool) -> None:
        super().__init__(pk=None, uuid="bool-uuid", label="bool")
        self.value = value


class _FakeStr(_FakeNode):
    def __init__(self, value: str) -> None:
        super().__init__(pk=None, uuid="str-uuid", label="str")
        self.value = value


class _FakeCode(_FakeNode):
    def __init__(self, value: str) -> None:
        label, computer = value.split("@", 1)
        super().__init__(pk=12, uuid="code-uuid", label=label)
        self.full_label = value
        self.computer = SimpleNamespace(label=computer)


class _FakeStructure(_FakeNode):
    pass


class _FakeGroup:
    collection = SimpleNamespace(get=None)

    def __init__(self, *, pk: int | None = None, label: str = "", nodes: list[object] | None = None) -> None:
        self.pk = pk
        self.label = label
        self.nodes = list(nodes or [])


def test_serialize_builder_inputs_payload_preserves_editable_values(monkeypatch) -> None:
    fake_orm = submission_utils.orm
    monkeypatch.setattr(fake_orm, "Node", _FakeNode)
    monkeypatch.setattr(fake_orm, "Dict", _FakeDict)
    monkeypatch.setattr(fake_orm, "List", _FakeList)
    monkeypatch.setattr(fake_orm, "Int", _FakeInt)
    monkeypatch.setattr(fake_orm, "Float", _FakeFloat)
    monkeypatch.setattr(fake_orm, "Bool", _FakeBool)
    monkeypatch.setattr(fake_orm, "Str", _FakeStr)
    monkeypatch.setattr(fake_orm, "Code", _FakeCode)
    monkeypatch.setattr(fake_orm, "AbstractCode", _FakeCode)

    payload = {
        "nbands_factor": _FakeFloat(1.5),
        "clean_workdir": _FakeBool(True),
        "label": _FakeStr("test"),
        "parameters": _FakeDict({"SYSTEM": {"ecutwfc": 60}}),
        "settings": _FakeList([1, 2, 3]),
        "code": _FakeCode("pw@localhost"),
        "structure": _FakeStructure(pk=88, uuid="structure-uuid", label="Si"),
    }

    serialized = submission_utils._serialize_builder_inputs_payload(payload)

    assert serialized["nbands_factor"] == 1.5
    assert serialized["clean_workdir"] is True
    assert serialized["label"] == "test"
    assert serialized["parameters"] == {"SYSTEM": {"ecutwfc": 60}}
    assert serialized["settings"] == [1, 2, 3]
    assert serialized["code"] == "pw@localhost"
    assert serialized["structure"] == {
        "pk": 88,
        "uuid": "structure-uuid",
        "type": "_FakeStructure",
        "label": "Si",
    }


def test_batch_submit_expands_group_structures_and_parameter_grid(monkeypatch) -> None:
    fake_group = _FakeGroup(
        pk=7,
        label="htp-structures",
        nodes=[
            _FakeStructure(pk=22, uuid="structure-22", label="Ge"),
            _FakeStructure(pk=11, uuid="structure-11", label="Si"),
        ],
    )

    def _fake_get(*, label: str) -> _FakeGroup:
        assert label == "htp-structures"
        return fake_group

    monkeypatch.setattr(submission_utils.orm, "Node", _FakeNode)
    monkeypatch.setattr(submission_utils.orm, "Group", _FakeGroup)
    monkeypatch.setattr(submission_utils.orm, "load_group", lambda value: (_ for _ in ()).throw(ValueError(value)))
    monkeypatch.setattr(_FakeGroup, "collection", SimpleNamespace(get=_fake_get))

    observed: list[tuple[int | None, float]] = []

    def _fake_submitter(payload: dict[str, object]) -> dict[str, object]:
        intent_data = payload["intent_data"]
        assert isinstance(intent_data, dict)
        structure = intent_data["structure_pk"]
        assert isinstance(structure, _FakeStructure)
        strain = float(intent_data["shear_strain"])
        observed.append((structure.pk, strain))
        index = len(observed)
        return {"pk": index, "uuid": f"uuid-{index}", "state": "created"}

    result = submission_utils.batch_submit(
        _fake_submitter,
        base_payload={
            "entry_point": "example.workchain",
            "intent_data": {"code": "pw@localhost"},
        },
        batch_data={
            "structures": {"group": "htp-structures"},
            "structure_field": "structure_pk",
            "parameter_grid": {"shear_strain": [0.0, 0.1]},
        },
        default_root="intent_data",
        default_structure_path="intent_data.structure_pk",
    )

    assert observed == [
        (11, 0.0),
        (11, 0.1),
        (22, 0.0),
        (22, 0.1),
    ]
    assert result["status"] == "SUBMITTED_BATCH"
    assert result["submitted_count"] == 4
    assert result["failed_count"] == 0
    assert result["submitted_pks"] == [1, 2, 3, 4]
    assert result["batch_context"]["source_group"]["label"] == "htp-structures"


def test_batch_submit_supports_zip_matrix_mode() -> None:
    observed: list[tuple[float, int]] = []

    def _fake_submitter(payload: dict[str, object]) -> dict[str, object]:
        inputs = payload["inputs"]
        assert isinstance(inputs, dict)
        observed.append((float(inputs["shear_strain"]), int(inputs["replica"])))
        index = len(observed)
        return {"pk": index, "uuid": f"uuid-{index}", "state": "created"}

    result = submission_utils.batch_submit(
        _fake_submitter,
        base_payload={"entry_point": "example.workflow", "inputs": {}},
        batch_data={
            "parameter_grid": {
                "shear_strain": [0.0, 0.1, 0.2],
                "replica": [1, 2, 3],
            },
            "matrix_mode": "zip",
        },
        default_root="inputs",
        default_structure_path="inputs.structure",
    )

    assert observed == [(0.0, 1), (0.1, 2), (0.2, 3)]
    assert result["submitted_count"] == 3
    assert result["failed_count"] == 0


def test_submit_workchain_builder_submits_builder_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_builder = object()

    monkeypatch.setattr(
        submission_utils,
        "_build_dynamic_protocol_builder",
        lambda _draft: {
            "success": True,
            "builder": fake_builder,
            "entry_point": "example.workchain",
            "errors": [],
            "missing_ports": [],
            "recovery_plan": {},
        },
    )
    monkeypatch.setattr(
        submission_utils,
        "_run_submission_script",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected submission script call")),
    )

    class _FakeSubmittedNode:
        pk = 77
        uuid = "builder-uuid"
        process_state = SimpleNamespace(value="created")

    captured: dict[str, object] = {}

    def _fake_submit(builder: object) -> _FakeSubmittedNode:
        captured["builder"] = builder
        return _FakeSubmittedNode()

    monkeypatch.setattr(submission_utils, "submit", _fake_submit)

    result = submission_utils._submit_workchain_builder({"entry_point": "example.workchain"})

    assert captured["builder"] is fake_builder
    assert result == {
        "status": "submitted",
        "pk": 77,
        "uuid": "builder-uuid",
        "state": "created",
    }


def test_batch_submit_cleans_storage_between_items(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(submission_utils, "cleanup_storage_session", lambda: calls.append("cleanup"))
    monkeypatch.setattr(submission_utils, "reset_storage_backend_caches", lambda: calls.append("reset"))
    monkeypatch.setattr(submission_utils, "prime_storage_user_context", lambda: calls.append("prime"))

    observed: list[int] = []

    def _fake_submitter(payload: dict[str, object]) -> dict[str, object]:
        observed.append(int(payload["value"]))
        return {"pk": len(observed), "uuid": f"uuid-{len(observed)}", "state": "created"}

    result = submission_utils.batch_submit(
        _fake_submitter,
        requests=[{"value": 1}, {"value": 2}],
    )

    assert observed == [1, 2]
    assert result["submitted_pks"] == [1, 2]
    assert calls == ["cleanup", "reset", "prime", "cleanup", "reset", "prime"]


def test_submit_workchain_builder_primes_storage_before_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_builder = object()
    calls: list[str] = []

    monkeypatch.setattr(
        submission_utils,
        "_build_dynamic_protocol_builder",
        lambda _draft: {
            "success": True,
            "builder": fake_builder,
            "entry_point": "example.workchain",
            "errors": [],
            "missing_ports": [],
            "recovery_plan": {},
        },
    )
    monkeypatch.setattr(submission_utils, "reset_storage_backend_caches", lambda: calls.append("reset"))
    monkeypatch.setattr(submission_utils, "prime_storage_user_context", lambda: calls.append("prime"))

    class _FakeSubmittedNode:
        pk = 91
        uuid = "uuid-91"
        process_state = SimpleNamespace(value="created")

    monkeypatch.setattr(
        submission_utils,
        "submit",
        lambda builder: calls.append("submit") or _FakeSubmittedNode(),
    )

    result = submission_utils._submit_workchain_builder({"entry_point": "example.workchain"})

    assert result["pk"] == 91
    assert calls == ["reset", "prime", "submit"]
