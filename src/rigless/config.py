"""Settings for rigless, read once from the environment and the repo's `.env`.

This is the only module that reads environment variables. Every setting has a
local default, so an empty `.env` works against a local ComfyUI behind
comfy-api-proxy. GPU presets live in `profiles/<name>.toml`.

Typical usage example:

  from rigless import config

  overrides = config.load_profile(config.PROFILE)
"""

import os
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

COMFY_BASE_URL = os.environ.get("COMFY_BASE_URL") or "http://127.0.0.1:8189"
COMFY_API_KEY = os.environ.get("COMFY_API_KEY") or None
PROFILE = os.environ.get("RIGLESS_PROFILE") or "local-10gb"

# Our workflow: ComfyUI's Wan Animate 2 template export with the pose guidance window
# restored to the full run (pose_end_percent 1.0); the untouched export sits beside it.
WORKFLOW_PATH = REPO_ROOT / "workflows" / "rigless_motion_transfer_api.json"
# Qwen-Image-Edit 2511 (ComfyUI's int8 template, flattened, with a second image input).
COMPOSE_WORKFLOW_PATH = REPO_ROOT / "workflows" / "rigless_compose_reference_api.json"
PROFILES_DIR = REPO_ROOT / "profiles"
RUNS_DIR = REPO_ROOT / "runs"


def load_profile(name: str) -> dict[str, Any]:
    """Reads a GPU preset: workflow settings that override the template's.

    Raises:
        FileNotFoundError: profiles/ has no preset with that name.
    """
    with (PROFILES_DIR / f"{name}.toml").open("rb") as file:
        return tomllib.load(file)
