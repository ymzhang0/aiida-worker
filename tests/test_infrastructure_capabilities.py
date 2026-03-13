from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from routers import data as data_router


class _FakeTransport:
    def __init__(self, fields: list[str]) -> None:
        self._fields = fields

    def get_valid_auth_params(self) -> list[str]:
        return list(self._fields)


def test_get_infrastructure_capabilities_prefers_asyncssh_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(data_router, "AIIDA_CORE_VERSION", "2.7.3")
    monkeypatch.setattr(
        data_router,
        "get_entry_point_names",
        lambda group: ["core.local", "core.ssh", "core.ssh_async"] if group == "aiida.transports" else [],
    )
    monkeypatch.setattr(
        data_router,
        "TransportFactory",
        lambda name: _FakeTransport(
            {
                "core.local": [],
                "core.ssh": ["username", "timeout", "proxy_jump", "use_login_shell", "safe_interval"],
                "core.ssh_async": ["host", "max_io_allowed", "script_before", "backend", "use_login_shell", "safe_interval"],
            }[name]
        ),
    )

    payload = data_router.get_infrastructure_capabilities()

    assert payload.aiida_core_version == "2.7.3"
    assert payload.supports_async_ssh is True
    assert payload.recommended_transport == "core.ssh_async"
    assert payload.transport_auth_fields["core.ssh_async"] == [
        "host",
        "max_io_allowed",
        "script_before",
        "backend",
        "use_login_shell",
        "safe_interval",
    ]


def test_build_transport_auth_params_maps_timeout_for_core_ssh() -> None:
    payload = data_router.InfrastructureSetupRequest(
        computer_label="daint",
        hostname="daint.cscs.ch",
        transport_type="core.ssh",
        scheduler_type="core.slurm",
        username="alice",
        proxy_jump="bastion",
        connection_timeout=45,
        use_login_shell=True,
        safe_interval=5.0,
    )

    auth_params = data_router._build_transport_auth_params(  # noqa: SLF001
        payload,
        "core.ssh",
        {"username", "proxy_jump", "timeout", "use_login_shell", "safe_interval"},
    )

    assert auth_params == {
        "username": "alice",
        "proxy_jump": "bastion",
        "timeout": 45,
        "use_login_shell": True,
        "safe_interval": 5.0,
    }


def test_build_transport_auth_params_uses_asyncssh_fields() -> None:
    payload = data_router.InfrastructureSetupRequest(
        computer_label="daint-async",
        hostname="daint.cscs.ch",
        transport_type="core.ssh_async",
        scheduler_type="core.slurm",
        host="daint-login",
        max_io_allowed=12,
        authentication_script="/usr/local/bin/aris-2fa.sh",
        backend="openssh",
        use_login_shell=False,
        safe_interval=2.5,
    )

    auth_params = data_router._build_transport_auth_params(  # noqa: SLF001
        payload,
        "core.ssh_async",
        {"host", "max_io_allowed", "script_before", "backend", "use_login_shell", "safe_interval"},
    )

    assert auth_params == {
        "host": "daint-login",
        "max_io_allowed": 12,
        "script_before": "/usr/local/bin/aris-2fa.sh",
        "backend": "openssh",
        "use_login_shell": False,
        "safe_interval": 2.5,
    }


def test_build_transport_auth_params_rejects_legacy_ssh_fields_for_asyncssh() -> None:
    payload = data_router.InfrastructureSetupRequest(
        computer_label="daint-async",
        hostname="daint.cscs.ch",
        transport_type="core.ssh_async",
        scheduler_type="core.slurm",
        username="alice",
    )

    with pytest.raises(Exception) as exc_info:
        data_router._build_transport_auth_params(  # noqa: SLF001
            payload,
            "core.ssh_async",
            {"host", "max_io_allowed", "script_before", "backend", "use_login_shell", "safe_interval"},
        )

    assert "core.ssh_async uses SSH config style authentication parameters" in str(exc_info.value)


def test_export_auth_payload_renames_async_script_before() -> None:
    computer = type("Computer", (), {"transport_type": "core.ssh_async"})()
    payload = data_router._export_auth_payload(  # noqa: SLF001
        computer,
        {
            "host": "daint-login",
            "script_before": "/usr/local/bin/aris-2fa.sh",
            "backend": "asyncssh",
            "use_login_shell": True,
        },
    )

    assert payload == {
        "host": "daint-login",
        "authentication_script": "/usr/local/bin/aris-2fa.sh",
        "backend": "asyncssh",
        "use_login_shell": True,
    }
