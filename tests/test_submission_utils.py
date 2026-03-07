from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
