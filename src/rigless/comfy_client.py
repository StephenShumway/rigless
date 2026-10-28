"""Comfy SDK access: connect, run a workflow to completion, download outputs.

The SDK speaks the Comfy API v2, so the same code drives the local
comfy-api-proxy in front of ComfyUI, a RunPod pod, or Comfy Cloud; config
picks the target.

Typical usage example:

  with connect() as client:
      job = run(client, graph)
      download(job, {"246": Path("result.mp4")})
"""

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
from comfy_sdk import BASE_URL_ENV_VAR, Comfy, ComfyError, Job, JobFailed, Progress

from rigless import config

Graph = dict[str, Any]
"""An API-format ComfyUI workflow: node id -> {"class_type", "inputs"}."""


def connect() -> Comfy:
    """Opens a client for where Rigless sends its work (`config.connection`).

    Raises:
        ValueError: RunPod is chosen and there is no pod.
    """
    target = config.connection()
    if target.url is None:
        raise ValueError(
            "There's no RunPod pod. Deploy one on the ComfyUI tab, "
            "or switch to This computer."
        )
    # The SDK takes its target only from this variable and otherwise falls back
    # to Comfy Cloud, so pin it to ours.
    os.environ[BASE_URL_ENV_VAR] = target.url
    client = Comfy(api_key=target.key)
    # The SDK takes no HTTP client of ours, so the hooks go on its own.
    hooks = client._low._client.event_hooks
    hooks["request"].append(pin_to(target.url, target.key))
    hooks["response"].append(read_refused_file_check)
    return client


def pin_to(base_url: str, key: str | None) -> Callable[[httpx.Request], None]:
    """Returns an httpx request hook that sends API calls to base_url only.

    comfy-api-proxy builds the links it returns (job events, outputs) from the
    address it was reached on, and behind RunPod's HTTPS proxy that is an
    internal one the app can't reach. The hook moves any API request bound
    elsewhere back to base_url, so the token only ever goes there.
    """
    base = httpx.URL(base_url)

    def hook(request: httpx.Request) -> None:
        url = request.url
        here = (url.scheme, url.host, url.port) == (base.scheme, base.host, base.port)
        if here or not url.path.startswith("/api/"):
            return
        request.url = url.copy_with(scheme=base.scheme, host=base.host, port=base.port)
        request.headers["Host"] = request.url.netloc.decode("ascii")
        if key:
            request.headers["Authorization"] = f"Bearer {key}"

    return hook


def read_refused_file_check(response: httpx.Response) -> None:
    """Reads a refused "is this file there?" check as "not yet", so it's uploaded.

    Cloudflare, in front of RunPod's proxy, now and then passes these HEAD
    checks on as GETs, which comfy-api-proxy refuses with 405 (Oct 26). The
    upload that follows is safe either way: the proxy takes duplicates.
    """
    request = response.request
    if (
        request.method == "HEAD"
        and response.status_code == 405
        and "/assets/by-hash/" in request.url.path
    ):
        response.status_code = 404


def run(
    client: Comfy,
    graph: Graph,
    on_progress: Callable[[Progress], None] | None = None,
) -> Job:
    """Submits a workflow and blocks until it finishes.

    SDK assets in the graph are uploaded first.

    Args:
        client: An open client.
        graph: An API-format workflow, patched for this run.
        on_progress: Called with each progress update while the job runs.

    Returns:
        The finished job.

    Raises:
        comfy_sdk.JobFailed: The job failed or was cancelled.
    """
    job = client.submit(client.workflows.from_json(graph))
    for event in job.events():
        if on_progress is not None and isinstance(event, Progress):
            on_progress(event)
    return job.result()


def download(job: Job, destinations: Mapping[str, Path]) -> None:
    """Saves each node's output file to the path given for its node id.

    Raises:
        RuntimeError: A node produced no output.
    """
    for node_id, dest in destinations.items():
        outputs = job.get_outputs(node_id)
        if not outputs:
            raise RuntimeError(f"Job {job.id}: node {node_id} produced no output.")
        # Unless the frame count is 4n + 1, the template's final rebatch leaves a
        # few surplus frames that ComfyUI saves as a second, tiny clip. The full
        # clip comes first.
        outputs[0].to_file(dest)


def try_run(
    client: Comfy,
    graph: Graph,
    outputs: Mapping[str, Path],
    on_progress: Callable[[Progress], None] | None = None,
) -> tuple[str | None, str | None]:
    """Runs one job and saves each output node's file, reporting failure as text.

    Args:
        client: An open client.
        graph: The workflow.
        outputs: Where to save each output node's file, by node id.
        on_progress: Called with each progress update while the job runs.

    Returns:
        A tuple (job_id, error), where error is None on success and job_id is
        None if the job failed.
    """
    try:
        job = run(client, graph, on_progress)
    except JobFailed as exc:
        return None, (f"{exc}: {exc.error}" if exc.error else str(exc))
    except ComfyError as exc:
        return None, str(exc)
    download(job, outputs)
    return job.id, None
