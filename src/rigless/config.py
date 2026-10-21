"""Settings for rigless, read once from the environment and the repo's `.env`.

This is the only module that reads environment variables, and `save` the only
code that writes `.env` (the app's settings page uses it). Every setting has a
local default, so an empty `.env` works against a local ComfyUI behind
comfy-api-proxy. GPU presets live in `profiles/<name>.toml`.

Typical usage example:

  from rigless import config

  overrides = config.load_profile(config.PROFILE)
"""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
load_dotenv(ENV_FILE)


def _connection() -> tuple[str, str | None, str]:
    """Reads the settings the app can change: ComfyUI's address, key, profile."""
    return (
        os.environ.get("COMFY_BASE_URL") or "http://127.0.0.1:8189",
        os.environ.get("COMFY_API_KEY") or None,
        os.environ.get("RIGLESS_PROFILE") or "local-10gb",
    )


COMFY_BASE_URL, COMFY_API_KEY, PROFILE = _connection()
# Only for downloading gated models for experiments.
HF_TOKEN = os.environ.get("HF_TOKEN") or None
# Required to serve the app beyond this computer; None keeps it open locally.
PASSCODE = os.environ.get("RIGLESS_PASSCODE") or None

# Our workflow: ComfyUI's Wan Animate 2 template export with the pose guidance window
# restored to the full run (pose_end_percent 1.0); the untouched export sits beside it.
WORKFLOW_PATH = REPO_ROOT / "workflows" / "rigless_motion_transfer_api.json"
# Qwen-Image-Edit 2511 (ComfyUI's int8 template, flattened, with a second image input).
COMPOSE_WORKFLOW_PATH = REPO_ROOT / "workflows" / "rigless_compose_reference_api.json"
# Qwen-Image 2.1 (ComfyUI's image-edit template, flattened, without its prompt
# rewriter). Better identity, but its license allows research and evaluation only.
COMPOSE_QWEN21_WORKFLOW_PATH = (
    REPO_ROOT / "workflows" / "rigless_compose_qwen_image_2_1_api.json"
)
PROFILES_DIR = REPO_ROOT / "profiles"
RUNS_DIR = REPO_ROOT / "runs"


def load_profile(name: str) -> dict[str, Any]:
    """Reads a GPU preset: workflow settings that override the template's.

    Raises:
        FileNotFoundError: profiles/ has no preset with that name.
    """
    with (PROFILES_DIR / f"{name}.toml").open("rb") as file:
        return tomllib.load(file)


def save(values: Mapping[str, str]) -> None:
    """Writes settings to `.env` and applies them to this process.

    The rest of `.env`, comments included, is kept. An empty value unsets a
    setting, so its default applies.

    Args:
        values: New values by environment variable name, such as
            {"COMFY_BASE_URL": "http://127.0.0.1:8199"}.

    Raises:
        ValueError: A value spans several lines.
    """
    if any("\n" in value or "\r" in value for value in values.values()):
        raise ValueError("Settings must fit on one line.")
    lines = ENV_FILE.read_text("utf-8").splitlines() if ENV_FILE.exists() else []
    pending = dict(values)
    for index, line in enumerate(lines):
        name = line.split("=", 1)[0].strip()
        if "=" in line and name in pending:
            lines[index] = _env_line(name, pending.pop(name))
    lines += [_env_line(name, value) for name, value in pending.items()]
    ENV_FILE.write_text("\n".join(lines) + "\n", "utf-8")
    os.environ.update(values)
    global COMFY_BASE_URL, COMFY_API_KEY, PROFILE
    COMFY_BASE_URL, COMFY_API_KEY, PROFILE = _connection()


def _env_line(name: str, value: str) -> str:
    """Returns a `.env` line that python-dotenv reads back as exactly value."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{name}="{escaped}"'
