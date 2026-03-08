from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core import node_utils


class _FakeLinkedNode:
    def __init__(self, pk: int, node_type: str) -> None:
        self.pk = pk
        self.node_type = node_type
        self.uuid = f"uuid-{pk}"
        self.label = ""
        self.ctime = None


class _FakeLink:
    def __init__(self, link_label: str, node: _FakeLinkedNode) -> None:
        self.link_label = link_label
        self.node = node


class _FakeLinkCollection:
    def __init__(self, links: list[_FakeLink]) -> None:
        self._links = links

    def all(self) -> list[_FakeLink]:
        return list(self._links)


class _FakeLinksBase:
    def __init__(self, incoming: list[_FakeLink], outgoing: list[_FakeLink]) -> None:
        self._incoming = _FakeLinkCollection(incoming)
        self._outgoing = _FakeLinkCollection(outgoing)

    def get_incoming(self) -> _FakeLinkCollection:
        return self._incoming

    def get_outgoing(self) -> _FakeLinkCollection:
        return self._outgoing


class _FakeAttributes:
    def __init__(self) -> None:
        self.all = {}


class _FakeBase:
    def __init__(self, incoming: list[_FakeLink], outgoing: list[_FakeLink]) -> None:
        self.links = _FakeLinksBase(incoming, outgoing)
        self.attributes = _FakeAttributes()


class _FakeNodeManager:
    def __init__(self, items: dict[str, object]) -> None:
        self._items = dict(items)

    def _get_keys(self) -> list[str]:
        return list(self._items.keys())

    def _get_node_by_link_label(self, key: str) -> object:
        return self._items[key]


class _FakeProcessNode:
    def __init__(self) -> None:
        self.pk = 300
        self.uuid = "uuid-300"
        self.node_type = "process.workflow.workchain.WorkChainNode."
        self.label = ""
        self.ctime = None
        self.process_state = None
        self.exit_status = 0
        self.process_label = "PwBandsWorkChain"
        self.base = _FakeBase(
            incoming=[
                _FakeLink("structure", _FakeLinkedNode(101, "data.core.structure.StructureData.")),
                _FakeLink("child__code", _FakeLinkedNode(202, "data.core.code.Code.")),
            ],
            outgoing=[
                _FakeLink("remote_folder", _FakeLinkedNode(303, "data.core.remote.RemoteData.")),
                _FakeLink("child__retrieved", _FakeLinkedNode(404, "data.core.folder.FolderData.")),
            ],
        )
        self.inputs = _FakeNodeManager(
            {
                "structure": _FakeLinkedNode(101, "data.core.structure.StructureData."),
                "metadata": _FakeNodeManager({"options": _FakeLinkedNode(111, "data.core.dict.Dict.")}),
            }
        )
        self.outputs = _FakeNodeManager(
            {
                "remote_folder": _FakeLinkedNode(303, "data.core.remote.RemoteData."),
            }
        )


def test_get_node_summary_prefers_direct_port_links(monkeypatch) -> None:
    node = _FakeProcessNode()

    monkeypatch.setattr(node_utils.orm, "load_node", lambda _pk: node)
    monkeypatch.setattr(node_utils.orm, "Node", _FakeLinkedNode)
    monkeypatch.setattr(
        node_utils,
        "serialize_node",
        lambda linked: {
            "pk": int(linked.pk),
            "uuid": str(linked.uuid),
            "node_type": node_utils.node_type_name(linked),
            "type": node_utils.node_type_name(linked),
            "label": str(getattr(linked, "label", "") or node_utils.node_type_name(linked)),
        },
    )

    summary = node_utils.get_node_summary(300)

    assert summary["inputs"].keys() == {"structure", "metadata__options"}
    assert summary["direct_inputs"] == summary["inputs"]
    assert summary["outputs"].keys() == {"remote_folder"}
    assert summary["direct_outputs"] == summary["outputs"]
    assert "child__code" not in summary["inputs"]
    assert "child__retrieved" not in summary["outputs"]
    assert summary["incoming"] == 2
    assert summary["outgoing"] == 1


class _FakeStructureSite:
    def __init__(self, kind_name: str, position: list[float]) -> None:
        self.kind_name = kind_name
        self.position = position


class _FakeScaledPositions:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self._rows]


class _FakeAtoms:
    def __init__(self, scaled_positions: list[list[float]]) -> None:
        self._scaled_positions = scaled_positions

    def get_scaled_positions(self, wrap: bool = False) -> _FakeScaledPositions:  # noqa: ARG002
        return _FakeScaledPositions(self._scaled_positions)


class _FakeStructureNode:
    def __init__(self) -> None:
        self.sites = [
            _FakeStructureSite("Si", [0.0, 0.0, 0.0]),
            _FakeStructureSite("Si", [1.3575, 1.3575, 1.3575]),
        ]

    def get_ase(self) -> _FakeAtoms:
        return _FakeAtoms([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ])


def test_structure_preview_includes_fractional_positions_from_backend() -> None:
    serializer = node_utils.StructureDataSerializer()

    preview = serializer.build_preview(_FakeStructureNode())

    assert preview is not None
    assert preview["positions"] == [
        {
            "kind": "Si",
            "position": [0.0, 0.0, 0.0],
            "fractional_position": [0.0, 0.0, 0.0],
        },
        {
            "kind": "Si",
            "position": [1.3575, 1.3575, 1.3575],
            "fractional_position": [0.25, 0.25, 0.25],
        },
    ]
