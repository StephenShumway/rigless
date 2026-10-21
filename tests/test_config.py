import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from rigless import config

TRICKY = r'p@ss "word" \ # not a comment'


def test_saved_settings_read_back_and_the_rest_of_env_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("# my comment\nCOMFY_BASE_URL=http://old\nHF_TOKEN=abc\n", "utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    for name in ("COMFY_BASE_URL", "COMFY_API_KEY", "RIGLESS_PROFILE"):
        monkeypatch.delenv(name, raising=False)

    config.save({"COMFY_BASE_URL": "http://new:8199", "COMFY_API_KEY": TRICKY})

    assert dotenv_values(env) == {
        "COMFY_BASE_URL": "http://new:8199",
        "HF_TOKEN": "abc",
        "COMFY_API_KEY": TRICKY,
    }
    assert env.read_text("utf-8").startswith("# my comment\n")
    assert (config.COMFY_BASE_URL, config.COMFY_API_KEY) == ("http://new:8199", TRICKY)


def test_empty_value_restores_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("RIGLESS_PROFILE", "cloud-48gb")

    config.save({"RIGLESS_PROFILE": ""})

    assert config.PROFILE == "local-10gb"
    assert os.environ["RIGLESS_PROFILE"] == ""


def test_multiline_value_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")

    with pytest.raises(ValueError, match="one line"):
        config.save({"COMFY_API_KEY": "a\nRIGLESS_PASSCODE=x"})
