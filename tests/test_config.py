from pathlib import Path

import pytest

from rigless import config
from rigless.config import Connection

POD = "https://abc123-8189.proxy.runpod.net"


@pytest.fixture(autouse=True)
def connection_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CONNECTION_FILE", tmp_path / "app" / "connection.json")
    monkeypatch.setattr(config, "CUSTOM_URL", None)
    monkeypatch.setattr(config, "CUSTOM_KEY", None)


def test_switching_to_this_computer_and_back_keeps_the_pod() -> None:
    config.save_pod(POD, "token")

    config.choose("local")
    assert config.connection() == Connection("local", config.LOCAL_URL)

    config.choose("runpod")
    assert config.connection() == Connection("runpod", POD, "token")


def test_stopped_pod_leaves_runpod_chosen_without_an_address() -> None:
    config.save_pod(POD, "token")

    config.forget_pod()

    assert config.connection() == Connection("runpod", None)


def test_comfy_base_url_in_the_environment_overrides_the_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "CUSTOM_URL", "http://gpu-box:8189")
    config.choose("local")

    assert config.connection() == Connection("custom", "http://gpu-box:8189")
