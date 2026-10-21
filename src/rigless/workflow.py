"""Loading and patching the API-format Wan Animate 2 workflow.

The workflow JSON in `workflows/` is a template and the single source of
default settings: `read_settings` reads them out, and `patch` writes a take's
settings, inputs, and seed back into a copy. Nodes are found by class type,
title, and their links into WanAnimate2ToVideo, never by node id, so a fresh
export from ComfyUI keeps working.

Typical usage example:

  template = load(config.WORKFLOW_PATH)
  settings = dataclasses.replace(read_settings(template), pose_strength=1.2)
  graph = patch(template, settings, video="dance.mp4", image="hero.jpg", seed=42)
"""

import copy
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from comfy_sdk import Asset

Graph = dict[str, dict[str, Any]]

REALISM_LORA = (
    "Real-footage LoRA"  # node title of the optional LoRA trained on phone clips
)
# Node ids that chunk_graph adds.
CONTINUE_FROM = "rigless_continue_from"
LAST_FRAME = "rigless_last_frame"


@dataclass(frozen=True)
class Settings:
    """Generation settings for a take; each one is a single workflow input.

    Attributes:
        positive_prompt: "Character Description: ..." and, on a new line,
            "Background description: ...".
        pose_prompt: A description of the movement.
        pose_strength: How strongly the driving video's pose steers the motion.
        pose_start_percent: Fraction of sampling at which pose guidance starts.
        pose_end_percent: Fraction of sampling at which pose guidance ends.
            Outside this window the model ignores the driving video entirely.
        reference_strength: How strongly the reference image steers the look.
        steps: Sampler steps per chunk.
        lora_strength: Strength of the LightX2V step-distillation LoRA.
        fragment_length: Frames per generated chunk; Wan needs 4n + 1.
        context_windows: Whether each chunk is sampled in overlapping windows.
        model: Diffusion model file in ComfyUI's diffusion_models folder.
        cache_device: Where the pose-branch cache lives: "cpu" or "gpu".
        width: Output width in pixels.
        height: Output height in pixels.
        sampler: The sampling method, such as "lcm" (for the LightX2V LoRA)
            or "euler" (for the distilled model).
        realism_lora_strength: Strength of the real-footage LoRA, trained on
            phone videos; at 2.0 it adds camera-like motion blur and flatter
            light. 0 leaves it out, for ComfyUI installs without its file and
            for workflows without the node.
    """

    positive_prompt: str
    pose_prompt: str
    pose_strength: float
    pose_start_percent: float
    pose_end_percent: float
    reference_strength: float
    steps: int
    lora_strength: float
    fragment_length: int
    context_windows: bool
    model: str
    cache_device: str
    width: int
    height: int
    sampler: str = "lcm"
    realism_lora_strength: float = 0.0

    def __post_init__(self) -> None:
        """Rejects settings the workflow can't run.

        Raises:
            ValueError: fragment_length is not 4n + 1, or the pose window
                starts after it ends.
        """
        length = self.fragment_length
        if length % 4 != 1:
            raise ValueError(
                f"fragment_length must be 4n + 1 (e.g., 81), got {length}."
            )
        if self.pose_start_percent > self.pose_end_percent:
            raise ValueError("pose_start_percent must not exceed pose_end_percent.")


@dataclass(frozen=True)
class Chunk:
    """One piece of a take, generated in a single sampling pass.

    Attributes:
        start: The take's frame index of the chunk's first frame. After the
            first chunk, that frame repeats the last frame of the chunk before,
            which the chunk continues from.
        length: The number of frames generated, 4n + 1, counting that one.
    """

    start: int
    length: int


def plan_chunks(frames: int, fragment_length: int) -> list[Chunk]:
    """Splits a take into chunks exactly as the workflow's chunk loop does.

    The last chunk can run past the end of the take; the extra frames are cut.

    Args:
        frames: The take's length in frames.
        fragment_length: The workflow's chunk length, 4n + 1.

    Raises:
        ValueError: frames is below 2.
    """
    if frames < 2:
        raise ValueError(f"A take needs at least 2 frames, got {frames}.")
    step = fragment_length - 1
    full = (frames - 2) // step
    chunks = [Chunk(index * step, fragment_length) for index in range(full)]
    rest = frames - full * step - 1
    return [*chunks, Chunk(full * step, 4 * math.ceil(rest / 4) + 1)]


def distilled(settings: Settings) -> Settings:
    """Returns the settings switched to Wan Animate 2's distilled model.

    Its makers run it for 10 Euler steps without guidance, so the LightX2V
    speed LoRA, made for the base model, is turned off. The model file keeps
    the same precision as the one in settings.
    """
    return replace(
        settings,
        model=settings.model.replace("wan_animate_2_", "wan_animate_2_distill_", 1),
        steps=10,
        lora_strength=0.0,
        sampler="euler",
    )


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


def find_node(graph: Graph, class_type: str, title: str | None = None) -> str:
    """Returns the id of the only node of class_type, with title if given.

    Raises:
        ValueError: The graph has no such node, or more than one.
    """
    ids = [
        nid
        for nid, node in graph.items()
        if node["class_type"] == class_type
        and (title is None or node.get("_meta", {}).get("title") == title)
    ]
    if len(ids) != 1:
        name = f"{class_type} {title!r}" if title else class_type
        raise ValueError(f"Expected one {name} node, found {len(ids)}.")
    return ids[0]


def read_settings(graph: Graph) -> Settings:
    """Returns the settings a workflow currently holds."""
    values = {
        name: graph[nid]["inputs"][key]
        for name, (nid, key) in _setting_inputs(graph).items()
    }
    return Settings(**values)


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


def patch(
    graph: Graph,
    settings: Settings,
    *,
    video: str | Asset,
    image: str | Asset,
    seed: int,
) -> Graph:
    """Returns a copy of the template set up for one take.

    Output files are saved under `rigless/` in ComfyUI's output folder, named
    by role.

    Args:
        graph: The workflow template. It is not modified.
        settings: The generation settings to write into the copy.
        video: The driving video, as an uploaded filename or an SDK asset.
        image: The reference image, as an uploaded filename or an SDK asset.
        seed: The sampler's noise seed.

    Raises:
        ValueError: settings ask for the real-footage LoRA, but the workflow
            has no node for it.
    """
    patched = copy.deepcopy(graph)
    for name, (nid, key) in _setting_inputs(patched).items():
        patched[nid]["inputs"][key] = getattr(settings, name)
    realism = _realism_lora(patched)
    if realism is None and settings.realism_lora_strength:
        raise ValueError(f"This workflow has no {REALISM_LORA!r} node.")
    if realism is not None and not settings.realism_lora_strength:
        _bypass(patched, realism)
    patched[find_node(patched, "LoadVideo")]["inputs"]["file"] = video
    patched[find_node(patched, "LoadImage")]["inputs"]["image"] = image
    patched[find_node(patched, "SamplerCustom")]["inputs"]["noise_seed"] = seed
    for nid, role in video_outputs(patched).items():
        patched[nid]["inputs"]["filename_prefix"] = f"rigless/{role}"
    return patched


def chunk_graph(graph: Graph, chunk: Chunk, continue_from: str | Asset | None) -> Graph:
    """Returns a copy of a take's graph that renders only one of its chunks.

    The workflow's chunk loop and the comparison video are removed. The chunk
    is saved as a silent result video, and its last frame as an image from the
    LAST_FRAME node, for the next chunk to continue from.

    Args:
        graph: A take's graph, from patch. It is not modified.
        chunk: The chunk to render, from plan_chunks.
        continue_from: The previous chunk's last frame, as an uploaded filename
            or an SDK asset; None for the first chunk.
    """
    # Only nodes and their inputs are edited, so copy just those: the inputs
    # can hold SDK assets, which can't be deep-copied.
    chunked = {
        nid: {**node, "inputs": dict(node["inputs"])} for nid, node in graph.items()
    }
    animate = chunked[find_node(chunked, "WanAnimate2ToVideo")]["inputs"]
    animate["length"] = chunk.length
    if continue_from is None:
        animate["video_frame_offset"] = chunk.start
        animate.pop("continue_motion", None)
    else:
        # The node steps the offset back over the frame it continues from.
        animate["video_frame_offset"] = chunk.start + 1
        chunked[CONTINUE_FROM] = _node(
            "LoadImage", "Continue from", image=continue_from
        )
        animate["continue_motion"] = [CONTINUE_FROM, 0]

    for nid, role in video_outputs(chunked).items():
        if role != "result":
            del chunked[nid]
    [result] = video_outputs(chunked)
    decoded = [find_node(chunked, "VAEDecode"), 0]
    create = chunked[_source(chunked, result, "video")]["inputs"]
    create["images"] = decoded
    create.pop("audio", None)
    chunked["rigless_pick_last"] = _node(
        "ImageFromBatch",
        "Last frame",
        image=decoded,
        batch_index=chunk.length - 1,
        length=1,
    )
    chunked[LAST_FRAME] = _node(
        "SaveImage", "Save last frame", images=["rigless_pick_last", 0],
        filename_prefix="rigless/last",
    )  # fmt: skip
    return _upstream_of_outputs(chunked)


def _node(class_type: str, title: str, **inputs: Any) -> dict[str, Any]:
    """Returns a new API-format node."""
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _upstream_of_outputs(graph: Graph) -> Graph:
    """Returns the graph without nodes that no saved output depends on."""
    stack = [
        nid
        for nid, node in graph.items()
        if node["class_type"] in ("SaveVideo", "SaveImage")
    ]
    keep = set()
    while stack:
        nid = stack.pop()
        if nid not in keep:
            keep.add(nid)
            stack.extend(
                value[0] for value in graph[nid]["inputs"].values() if _is_link(value)
            )
    return {nid: node for nid, node in graph.items() if nid in keep}


def _setting_inputs(graph: Graph) -> dict[str, tuple[str, str]]:
    """Maps each Settings field to the (node id, input name) that holds it."""
    animate = find_node(graph, "WanAnimate2ToVideo")
    cache = find_node(graph, "WanAnimate2Cache")
    pose_resize = _source(graph, animate, "pose_video")
    inputs = {
        "positive_prompt": (_source(graph, animate, "positive"), "text"),
        "pose_prompt": (_source(graph, animate, "positive_pose"), "text"),
        "pose_strength": (animate, "pose_strength"),
        "pose_start_percent": (animate, "pose_start_percent"),
        "pose_end_percent": (animate, "pose_end_percent"),
        "reference_strength": (animate, "reference_image_strength"),
        "steps": (find_node(graph, "BasicScheduler"), "steps"),
        "lora_strength": (
            find_node(graph, "LoraLoaderModelOnly", title="Load LoRA"),
            "strength_model",
        ),
        "fragment_length": (
            find_node(graph, "PrimitiveInt", title="Int (fragment_length)"),
            "value",
        ),
        "context_windows": (_source(graph, cache, "model"), "switch"),
        "model": (find_node(graph, "UNETLoader"), "unet_name"),
        "sampler": (find_node(graph, "KSamplerSelect"), "sampler_name"),
        "cache_device": (cache, "device"),
        "width": (pose_resize, "resize_type.width"),
        "height": (pose_resize, "resize_type.height"),
    }
    realism = _realism_lora(graph)
    if realism is not None:
        inputs["realism_lora_strength"] = (realism, "strength_model")
    return inputs


def _realism_lora(graph: Graph) -> str | None:
    """Returns the id of the real-footage LoRA node, or None if there is none."""
    ids = [
        nid
        for nid, node in graph.items()
        if node["class_type"] == "LoraLoaderModelOnly"
        and node.get("_meta", {}).get("title") == REALISM_LORA
    ]
    return ids[0] if ids else None


def _bypass(graph: Graph, node_id: str) -> None:
    """Removes a model-only node, linking its consumers to its own input."""
    source = graph.pop(node_id)["inputs"]["model"]
    for node in graph.values():
        for name, value in node["inputs"].items():
            if value == [node_id, 0]:
                node["inputs"][name] = source


def _source(graph: Graph, node_id: str, input_name: str) -> str:
    """Returns the id of the node linked into node_id's input_name."""
    source_id, _ = graph[node_id]["inputs"][input_name]
    return source_id


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
