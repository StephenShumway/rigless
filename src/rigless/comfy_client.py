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

from comfy_sdk import BASE_URL_ENV_VAR, Comfy, Job, Progress

from rigless import config
from rigless.workflow import Graph


def connect() -> Comfy:
    """Opens a client for the configured Comfy API v2 endpoint."""
    # The SDK takes its target only from this variable and otherwise falls back
    # to Comfy Cloud, so pin it to ours (the local proxy by default).
    os.environ[BASE_URL_ENV_VAR] = config.COMFY_BASE_URL
    return Comfy(api_key=config.COMFY_API_KEY)


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
        outputs[0].to_file(dest)
