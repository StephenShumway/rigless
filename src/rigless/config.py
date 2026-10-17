"""Settings for rigless, read once from the environment and the repo's `.env`.

This is the only module that reads environment variables. Every setting has a
local default, so an empty `.env` works against a local ComfyUI behind
comfy-api-proxy.

Typical usage example:

  from rigless import config

  runs = config.RUNS_DIR
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

COMFY_BASE_URL = os.environ.get("COMFY_BASE_URL") or "http://127.0.0.1:8189"
COMFY_API_KEY = os.environ.get("COMFY_API_KEY") or None

WORKFLOW_PATH = REPO_ROOT / "workflows" / "wan_animate2_motion_transfer_api.json"
RUNS_DIR = REPO_ROOT / "runs"
