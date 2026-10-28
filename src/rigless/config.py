"""Settings for rigless: the environment, an optional `.env`, and the connection.

This is the only module that reads environment variables. Every setting has a
default, so Rigless runs with no `.env` at all. Where Rigless sends its work,
a RunPod pod or ComfyUI on this computer, is chosen in the app and kept in
`runs/app/connection.json` with the pod's token; COMFY_BASE_URL in the
environment overrides that choice, for a ComfyUI anywhere else.

Typical usage example:

  from rigless import config

  target = config.connection()
  client = Comfy(api_key=target.key)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# Only for downloading models from a private Hugging Face repository.
HF_TOKEN = os.environ.get("HF_TOKEN") or None
# Required to serve the app beyond this computer; None keeps it open locally.
PASSCODE = os.environ.get("RIGLESS_PASSCODE") or None
RUNS_DIR = REPO_ROOT / "runs"
CONNECTION_FILE = RUNS_DIR / "app" / "connection.json"
LOCAL_URL = "http://127.0.0.1:8189"  # comfy-api-proxy's default address
CUSTOM_URL = os.environ.get("COMFY_BASE_URL") or None
CUSTOM_KEY = os.environ.get("COMFY_API_KEY") or None

Choice = Literal["runpod", "local"]


@dataclass(frozen=True)
class Connection:
    """Where Rigless sends its work.

    Attributes:
        target: "runpod" (the app's pod), "local" (ComfyUI on this computer),
            or "custom" (COMFY_BASE_URL in the environment).
        url: The Comfy API v2 address, or None when RunPod is chosen and
            there is no pod.
        key: The address's bearer token, or None if it needs none.
    """

    target: Literal["runpod", "local", "custom"]
    url: str | None
    key: str | None = None


def connection() -> Connection:
    """Returns where Rigless sends its work now. RunPod unless chosen otherwise."""
    if CUSTOM_URL:
        return Connection("custom", CUSTOM_URL, CUSTOM_KEY)
    saved = _saved()
    if saved.get("target") == "local":
        return Connection("local", LOCAL_URL)
    pod = saved.get("pod") or {}
    return Connection("runpod", pod.get("url"), pod.get("token"))


def choose(target: Choice) -> None:
    """Sends Rigless's work to RunPod or to ComfyUI on this computer."""
    _save({**_saved(), "target": target})


def save_pod(url: str, token: str) -> None:
    """Records a new pod and its proxy token, and sends Rigless's work to it."""
    _save({"target": "runpod", "pod": {"url": url, "token": token}})


def forget_pod() -> None:
    """Forgets the pod once it's deleted; RunPod stays chosen if it was."""
    _save({**_saved(), "pod": None})


def _saved() -> dict[str, Any]:
    if not CONNECTION_FILE.exists():
        return {}
    saved: dict[str, Any] = json.loads(CONNECTION_FILE.read_text("utf-8"))
    return saved


def _save(values: dict[str, Any]) -> None:
    """Writes the connection whole, so a crash mid-write can't leave half a file."""
    CONNECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    partial = CONNECTION_FILE.with_suffix(".tmp")
    partial.write_text(json.dumps(values, indent=2), "utf-8")
    partial.replace(CONNECTION_FILE)
