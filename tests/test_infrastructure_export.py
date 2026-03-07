from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from routers import data as data_router


class _FakeAuthInfo:
    def get_auth_params(self) -> dict[str, object]:
        return {
            "username": "alice",
            "key_filename": "/Users/alice/.ssh/id_ed25519",
            "proxy_jump": "login-node",
            "use_login_shell": True,
        }


class _FakeComputer:
    def __init__(self) -> None:
        self.label = "localhost"
        self.hostname = "localhost"
        self.description = "Local machine"
        self.transport_type = "core.local"
        self.scheduler_type = "core.direct"

    def get_workdir(self) -> str:
        return "/tmp/aiida"

    def get_default_mpiprocs_per_machine(self) -> int:
        return 1

    def get_default_memory_per_machine(self) -> int | None:
        return 262144

    def get_shebang(self) -> str:
        return "#!/bin/bash"

    def get_mpirun_command(self) -> list[str]:
        return ["mpirun", "-np", "{tot_num_mpiprocs}"]

    def get_use_double_quotes(self) -> bool:
        return False

    def get_prepend_text(self) -> str:
        return ""

    def get_append_text(self) -> str:
        return ""

    def get_authinfo(self, _user: object) -> _FakeAuthInfo:
        return _FakeAuthInfo()


class _FakeComputerCollection:
    @staticmethod
    def get(**kwargs: object) -> _FakeComputer:
        assert kwargs in ({"label": "localhost"}, {"pk": 7})
        return _FakeComputer()


class _FakeComputerOrm:
    collection = _FakeComputerCollection()


class _FakeUserCollection:
    @staticmethod
    def get_default() -> object:
        return object()


class _FakeUserOrm:
    collection = _FakeUserCollection()


class _FakeCode:
    def __init__(self) -> None:
        self.label = "pw-7.5"
        self.description = "Quantum ESPRESSO"
        self.default_calc_job_plugin = "quantumespresso.pw"
        self.filepath_executable = "/opt/qe/bin/pw.x"
        self.with_mpi = True
        self.use_double_quotes = False
        self.computer = type("ComputerRef", (), {"label": "localhost"})()

    def get_prepend_text(self) -> str:
        return "module load qe"

    def get_append_text(self) -> str:
        return ""


def test_export_computer_config_returns_yaml(monkeypatch) -> None:
    monkeypatch.setattr(data_router, "ensure_profile_loaded", lambda: None)
    monkeypatch.setattr(data_router.orm, "Computer", _FakeComputerOrm)
    monkeypatch.setattr(data_router.orm, "User", _FakeUserOrm)

    payload = data_router.export_computer_config("localhost")

    assert payload.kind == "computer"
    assert payload.filename == "localhost-setup.yaml"
    assert "hostname: localhost" in payload.content
    assert "username: alice" in payload.content
    assert "proxy_jump: login-node" in payload.content
    assert "shebang: '#!/bin/bash'" in payload.content
    assert "default_memory_per_machine: 262144" in payload.content


def test_export_computer_config_by_pk_returns_yaml(monkeypatch) -> None:
    monkeypatch.setattr(data_router, "ensure_profile_loaded", lambda: None)
    monkeypatch.setattr(data_router.orm, "Computer", _FakeComputerOrm)
    monkeypatch.setattr(data_router.orm, "User", _FakeUserOrm)

    payload = data_router.export_computer_config_by_pk(7)

    assert payload.kind == "computer"
    assert payload.filename == "localhost-setup.yaml"
    assert "hostname: localhost" in payload.content


def test_export_code_config_returns_yaml(monkeypatch) -> None:
    fake_code = _FakeCode()
    monkeypatch.setattr(data_router, "ensure_profile_loaded", lambda: None)
    monkeypatch.setattr(data_router.orm, "load_code", lambda pk: fake_code)

    payload = data_router.export_code_config(12)

    assert payload.kind == "code"
    assert payload.filename == "pw-7.5@localhost.yaml"
    assert "default_calc_job_plugin: quantumespresso.pw" in payload.content
    assert "filepath_executable: /opt/qe/bin/pw.x" in payload.content
