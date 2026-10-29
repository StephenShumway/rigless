"""The user's RunPod GPU for Rigless: at most one pod, with its own disk.

Deploy makes a pod wherever RunPod has the chosen GPU free, with a 100GB disk for
ComfyUI and the models (its environments go on the container disk, which a pause
clears). The pod fetches the latest code, runs
the setup (`deploy/runpod/setup.sh`, mostly downloads, reporting its progress on
port 8189), and then serves ComfyUI behind comfy-api-proxy on that port. Pausing
releases the GPU and keeps the disk (billed while paused) to resume in minutes.
Deleting removes both, so nothing is billed afterwards.

The user's API key comes with each call; Rigless passes it to RunPod and never
stores it. RunPod's REST API rejects calls straight from a browser, so the app's
server makes them.

Typical usage example:

  gpus = runpod.gpus(key)
  pod = runpod.start(key, "NVIDIA A40", env={"PROXY_TOKEN": token})
"""

import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
NAME = "rigless-app"  # only the app uses it, so a pod made by hand is never stopped
# A pod RunPod moves to another machine on resume comes back as "<name>-migration-N".
OURS = re.compile(rf"{re.escape(NAME)}(-migration-\d+)?")
DISK_GB = 100  # ~67GB of models plus ComfyUI and its renders
CONTAINER_DISK_GB = 50  # the two environments and uv's cache, ~15GB (Oct 28)
MIN_VRAM_GB = 48  # the H3 swap's peak, with the text encoder released before sampling
IDLE_PER_GB_MONTH = 0.20  # RunPod's price for a paused pod's disk (Oct 2026)
HOURS_PER_MONTH = 730
# A new pod's setup steps in order, with their seconds on a fast H100 pod (Oct 28),
# for the time-left estimate. The download is timed by its own speed instead.
SETUP_STEPS = {
    "comfyui": 30,
    "torch": 180,
    "packages": 260,
    "rigless": 20,
    "models": 0,
    "start": 120,
}
MODELS_BYTES = 67.1e9  # what scripts/download_models.py fetches
DOWNLOAD_SPEED = (
    250e6  # bytes a second (Oct 28: ~280), until the download measures its own
)
# comfy-kitchen's fast int8 kernels need a CUDA 13 driver (12.8 is ~1.75x slower);
# any 13.x will do, and RunPod's H100s run 13.2 (Oct 26).
MIN_CUDA = "13.0"
IMAGE = "runpod/base:1.4.1-ubuntu2404"
CODE = "https://github.com/StephenShumway/rigless/archive/refs/heads/main.tar.gz"

# Runs as the pod's start command: fetch the code and set up in the background,
# then hand over to the image's own start script (SSH, then sleep). If the code
# can't be fetched, a copy already on the disk (a restarted pod) sets up instead.
# One line without single quotes, since RunPod takes it as `bash -c '...'`.
BOOT = (
    "mkdir -p /workspace/logs; (cd /workspace; "
    f"if curl -fsSL {CODE} | tar xz -C /tmp; "
    "then rm -rf rigless-main && mv /tmp/rigless-main .; fi; "
    "bash rigless-main/deploy/runpod/setup.sh) > /workspace/logs/setup.log 2>&1 & "
    "exec /start.sh"
)


class RunPodError(Exception):
    """RunPod refused a request or couldn't be reached; the message says why."""


def gpus(key: str) -> list[dict[str, Any]]:
    """Returns the GPUs that can run Rigless, cheapest first.

    Each is {"id", "name", "memory", "price" (per hour), "stock"}, where stock
    is "High", "Medium", "Low" or None, counting only Secure Cloud machines
    with the CUDA driver Rigless needs.

    Raises:
        RunPodError: RunPod refused the request or couldn't be reached.
    """
    data = _graphql(
        key,
        "query { gpuTypes { id displayName memoryInGb securePrice "
        "lowestPrice(input: {gpuCount: 1, secureCloud: true, "
        f'minCudaVersion: "{MIN_CUDA}"}}) {{ stockStatus }} }} }}',
    )
    usable = [
        {
            "id": gpu["id"],
            "name": gpu["displayName"],
            "memory": gpu["memoryInGb"],
            "price": gpu["securePrice"],
            "stock": (gpu["lowestPrice"] or {}).get("stockStatus"),
        }
        for gpu in data["gpuTypes"]
        if (gpu["memoryInGb"] or 0) >= MIN_VRAM_GB
        and gpu["securePrice"]
        and gpu["id"].startswith("NVIDIA")
        and "MIG" not in gpu["id"]
    ]
    return sorted(usable, key=lambda gpu: gpu["price"])


def pod(key: str) -> dict[str, Any] | None:
    """Returns Rigless's pod, or None if there isn't one.

    It's the one named NAME, or NAME with RunPod's "-migration-N" after a move.

    Raises:
        RunPodError: RunPod refused the request or couldn't be reached.
    """
    pods = _rest(key, "GET", "/pods?includeMachine=true") or []
    return next((found for found in pods if OURS.fullmatch(found["name"])), None)


def start(key: str, gpu: str, env: Mapping[str, str]) -> dict[str, Any]:
    """Starts Rigless's pod wherever RunPod has the GPU free with a CUDA 13 driver.

    It goes through RunPod's GraphQL API: its REST API only takes exact CUDA
    versions, up to 13.0, so it can't ask for "13.0 or newer".

    Args:
        key: The user's RunPod API key.
        gpu: The GPU type, an "id" from gpus().
        env: The pod's environment, PROXY_TOKEN included.

    Returns:
        The new pod.

    Raises:
        RunPodError: There is already a pod, or RunPod refused the request.
    """
    if pod(key) is not None:
        raise RunPodError("Rigless already has a pod. Delete it first.")
    # The dashboard gives each pod the account's SSH key; the API leaves it out,
    # and without it the image never starts its SSH server.
    account = _graphql(key, "query { myself { pubKey } }")["myself"]
    if account.get("pubKey"):
        env = {**env, "PUBLIC_KEY": account["pubKey"]}
    data = _graphql(
        key,
        "mutation Deploy($input: PodFindAndDeployOnDemandInput) "
        "{ podFindAndDeployOnDemand(input: $input) { id } }",
        {
            "input": {
                "name": NAME,
                "imageName": IMAGE,
                "gpuTypeId": gpu,
                "gpuCount": 1,
                "cloudType": "SECURE",
                "minCudaVersion": MIN_CUDA,
                "volumeInGb": DISK_GB,
                "volumeMountPath": "/workspace",
                "containerDiskInGb": CONTAINER_DISK_GB,
                "ports": "8189/http,22/tcp",
                "env": [{"key": name, "value": value} for name, value in env.items()],
                "dockerArgs": f"bash -c '{BOOT}'",
            }
        },
    )
    created: dict[str, Any] | None = data["podFindAndDeployOnDemand"]
    if created is None:
        raise RunPodError("RunPod didn't start a pod. Try again in a minute.")
    return created


def pause(key: str) -> None:
    """Pauses Rigless's pod: RunPod releases the GPU and keeps the disk.

    Raises:
        RunPodError: There's no pod, or RunPod refused the request.
    """
    _rest(key, "POST", f"/pods/{_pod_id(key)}/stop")


def resume(key: str) -> None:
    """Resumes Rigless's paused pod on its machine.

    Raises:
        RunPodError: There's no pod, or RunPod refused, such as when someone
            else has the machine's GPU.
    """
    _rest(key, "POST", f"/pods/{_pod_id(key)}/start")


def delete(key: str) -> None:
    """Deletes Rigless's pod and its disk, if there is one.

    Raises:
        RunPodError: RunPod refused the request or couldn't be reached.
    """
    found = pod(key)
    if found is not None:
        _rest(key, "DELETE", f"/pods/{found['id']}")


def idle_cost_per_hour(found: Mapping[str, Any]) -> float:
    """Returns what a pod's disk costs an hour while the pod is paused, in dollars."""
    return float(found.get("volumeInGb") or 0) * IDLE_PER_GB_MONTH / HOURS_PER_MONTH


def setup_progress(pod_url: str) -> dict[str, Any] | None:
    """Returns how far a new pod's setup has got, or None if the pod isn't saying.

    While it sets up, the pod serves `progress.json` on the port that
    comfy-api-proxy takes over once ComfyUI is running.

    Returns:
        The stage, one of SETUP_STEPS or "failed", when it began ("since",
        seconds since the epoch), and when the setup began ("started"). For
        "models", also the bytes
        downloaded ("done"), in all ("total"), and already there at "since"
        ("from").
    """
    request = urllib.request.Request(
        f"{pod_url}/progress.json", headers={"User-Agent": "rigless"}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            progress = json.load(response)
    # Not set up yet, or comfy-api-proxy has the port.
    except (OSError, ValueError, http.client.HTTPException):
        return None
    return progress if isinstance(progress, dict) else None


def gpu_name(found: Mapping[str, Any]) -> str | None:
    """Returns a pod's GPU as people call it, such as "H100 SXM", or None if unknown."""
    machine = found.get("machine") or {}
    name = (
        (found.get("gpu") or {}).get("displayName")
        or machine.get("gpuDisplayName")
        or (machine.get("gpuType") or {}).get("displayName")
        or machine.get("gpuTypeId")
    )
    return name.removeprefix("NVIDIA ") if name else None


def proxy_url(pod_id: str) -> str:
    """Returns the address of a pod's comfy-api-proxy through RunPod's HTTPS proxy."""
    return f"https://{pod_id}-8189.proxy.runpod.net"


def setup_estimate(progress: Mapping[str, Any], now: float) -> dict[str, float]:
    """Returns roughly how far a new pod's setup has got, from the progress it reports.

    Args:
        progress: What setup_progress returned.
        now: The current time, in seconds since the epoch.

    Returns:
        {"fraction": the share done, "seconds_left": the time still needed}, or
        an empty dict for a stage it can't place, such as "failed".
    """
    stage = progress.get("stage")
    if stage not in SETUP_STEPS:
        return {}
    since = float(progress.get("since", now))
    started = since if progress.get("started") is None else float(progress["started"])
    done = float(progress.get("done", 0))
    speed = DOWNLOAD_SPEED
    fetched = done - float(progress.get("from", done))
    if stage == "models" and fetched > 0 and now > since:
        speed = fetched / (now - since)
    typical = {**SETUP_STEPS, "models": (progress.get("total") or MODELS_BYTES) / speed}
    steps = list(typical)
    current = steps.index(stage)
    # A pod's disk can make the installs ~9x slower than typical (Oct 28), so the
    # steps are scaled by how slow this pod has been: in the steps it finished, or
    # in all of them counting the current one as done, whichever is slower.
    before = sum(typical[step] for step in steps[:current])
    pace = max(
        1.0,
        (since - started) / before if before else 0.0,
        (now - started) / (before + typical[stage]),
    )
    seconds = {step: t if step == "models" else pace * t for step, t in typical.items()}
    if stage == "models":
        spent = done / speed
    else:
        spent = min(now - since, 0.9 * seconds[stage])  # a slow step isn't over yet
    left = seconds[stage] - spent + sum(seconds[step] for step in steps[current + 1 :])
    if progress.get("started") is None:
        finished = sum(seconds[step] for step in steps[:current]) + spent
    else:
        # The time really taken, as a fast pod beats the typical times by minutes
        # (the pod's clock, which set started, can be a little ahead of ours).
        finished = max(0.0, now - started)
    return {"fraction": finished / (finished + left), "seconds_left": left}


def _pod_id(key: str) -> str:
    found = pod(key)
    if found is None:
        raise RunPodError("Rigless has no pod.")
    pod_id: str = found["id"]
    return pod_id


def _graphql(
    key: str, query: str, variables: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    body = _call(key, "POST", GRAPHQL, {"query": query, "variables": variables})
    if body.get("errors"):
        raise RunPodError(body["errors"][0].get("message", "RunPod refused the query."))
    data: dict[str, Any] = body["data"]
    return data


def _rest(key: str, method: str, path: str, body: Any = None) -> Any:
    return _call(key, method, REST + path, body)


def _call(key: str, method: str, url: str, body: Any = None) -> Any:
    """Sends one request to RunPod and returns its JSON answer, or None if empty."""
    request = urllib.request.Request(
        url,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Cloudflare in front of RunPod's REST API bans Python-urllib's agent.
            "User-Agent": "rigless",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            answer = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RunPodError("RunPod didn't accept the API key.") from exc
        raise RunPodError(_reason(exc)) from exc
    except urllib.error.URLError as exc:
        raise RunPodError(f"Couldn't reach RunPod ({exc.reason}).") from exc
    # urlopen wraps only connection errors in URLError; a timeout or a dropped
    # connection while waiting for the answer comes through as it is.
    except (OSError, http.client.HTTPException) as exc:
        raise RunPodError(f"Couldn't reach RunPod ({exc}).") from exc
    try:
        return json.loads(answer) if answer.strip() else None
    except ValueError as exc:
        raise RunPodError("RunPod's answer wasn't readable. Try again.") from exc


def _reason(exc: urllib.error.HTTPError) -> str:
    """RunPod's own words for a refused request, when it gives any."""
    try:
        detail = json.loads(exc.read())
    except ValueError:
        return f"RunPod refused the request (error {exc.code})."
    if isinstance(detail, dict) and detail.get("error"):
        return str(detail["error"])
    return f"RunPod refused the request (error {exc.code})."
