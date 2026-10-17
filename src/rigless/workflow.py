"""Loading and patching the API-format Wan Animate 2 workflow.

The workflow JSON in `workflows/` is a template: each take patches in its own
driving video, reference image, and seed. Nodes are found by class type and by
their place in the graph, never by node id, so a fresh export from ComfyUI
keeps working.

Typical usage example:

  template = load(config.WORKFLOW_PATH)
  graph = patch(template, video="dance.mp4", image="hero.jpg", seed=42)
"""

import copy
import json
from pathlib import Path
from typing import Any

from comfy_sdk import Asset

Graph = dict[str, dict[str, Any]]


def load(path: Path) -> Graph:
    """Reads an API-format workflow.

    Raises:
        ValueError: The file is not an API-format workflow, e.g., a UI export.
    """
    graph = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not all(
        isinstance(node, dict) and "class_type" in node for node in graph.values()
    ):
        raise ValueError(
            f"{path} is not an API-format workflow. "
            "In ComfyUI, use Workflow > Export (API)."
        )
    return graph


def find_node(graph: Graph, class_type: str) -> str:
    """Returns the id of the only node of class_type.

    Raises:
        ValueError: The graph has no such node, or more than one.
    """
    ids = [nid for nid, node in graph.items() if node["class_type"] == class_type]
    if len(ids) != 1:
        raise ValueError(f"Expected one {class_type} node, found {len(ids)}.")
    return ids[0]


def output_size(graph: Graph) -> tuple[int, int]:
    """Returns the (width, height) the workflow renders at."""
    animate = graph[find_node(graph, "WanAnimate2ToVideo")]
    resize_id, _ = animate["inputs"]["pose_video"]
    inputs = graph[resize_id]["inputs"]
    return inputs["resize_type.width"], inputs["resize_type.height"]


def video_outputs(graph: Graph) -> dict[str, str]:
    """Maps each SaveVideo node id to "result" or "comparison".

    The comparison is the side-by-side video stitched from the driving video
    and the result.
    """
    return {
        nid: "comparison" if _fed_by(graph, nid, "ImageStitch") else "result"
        for nid, node in graph.items()
        if node["class_type"] == "SaveVideo"
    }


def patch(graph: Graph, *, video: str | Asset, image: str | Asset, seed: int) -> Graph:
    """Returns a copy of the template set up for one take.

    Output files are saved under `rigless/` in ComfyUI's output folder, named
    by role.

    Args:
        graph: The workflow template. It is not modified.
        video: The driving video, as an uploaded filename or an SDK asset.
        image: The reference image, as an uploaded filename or an SDK asset.
        seed: The sampler's noise seed.
    """
    patched = copy.deepcopy(graph)
    patched[find_node(patched, "LoadVideo")]["inputs"]["file"] = video
    patched[find_node(patched, "LoadImage")]["inputs"]["image"] = image
    patched[find_node(patched, "SamplerCustom")]["inputs"]["noise_seed"] = seed
    for nid, role in video_outputs(patched).items():
        patched[nid]["inputs"]["filename_prefix"] = f"rigless/{role}"
    return patched


def _fed_by(graph: Graph, node_id: str, class_type: str) -> bool:
    """Whether any node upstream of node_id is of class_type."""
    stack, seen = [node_id], set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        if graph[nid]["class_type"] == class_type:
            return True
        stack.extend(
            value[0] for value in graph[nid]["inputs"].values() if _is_link(value)
        )
    return False


def _is_link(value: object) -> bool:
    """Whether an input value is a link: [source node id, output index]."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )
