import copy
import threading
from dataclasses import replace
from typing import Any, cast

import pytest
from comfy_sdk import Asset

from rigless import config, workflow


@pytest.fixture
def template() -> workflow.Graph:
    return workflow.load(config.WORKFLOW_PATH)


def _inputs(graph: workflow.Graph, class_type: str) -> dict[str, Any]:
    return graph[workflow.find_node(graph, class_type)]["inputs"]


def _patch(graph: workflow.Graph, settings: workflow.Settings) -> workflow.Graph:
    return workflow.patch(graph, settings, video="dance.mp4", image="hero.png", seed=7)


def test_read_settings_finds_each_setting(template: workflow.Graph) -> None:
    settings = workflow.read_settings(template)

    assert settings.positive_prompt.startswith("Character Description:")
    assert settings.pose_prompt.startswith("A girl doing energetic street dance")
    assert (settings.pose_start_percent, settings.pose_end_percent) == (0, 1.0)
    assert (settings.steps, settings.fragment_length) == (6, 81)
    assert (settings.context_windows, settings.cache_device) == (False, "cpu")
    assert (settings.width, settings.height) == (482, 854)
    assert settings.model == "wan_animate_2_int8_convrot.safetensors"
    assert settings.realism_lora_strength == 2.0


def test_patch_writes_settings_to_their_inputs(template: workflow.Graph) -> None:
    settings = replace(
        workflow.read_settings(template),
        pose_prompt="spins twice",
        pose_strength=1.2,
        reference_strength=0.9,
        context_windows=True,
    )

    graph = _patch(template, settings)

    animate = _inputs(graph, "WanAnimate2ToVideo")
    assert (animate["pose_strength"], animate["reference_image_strength"]) == (1.2, 0.9)
    assert workflow.read_settings(graph) == settings


def test_patch_sets_video_image_and_seed(template: workflow.Graph) -> None:
    graph = _patch(template, workflow.read_settings(template))

    assert _inputs(graph, "LoadVideo")["file"] == "dance.mp4"
    assert _inputs(graph, "LoadImage")["image"] == "hero.png"
    assert _inputs(graph, "SamplerCustom")["noise_seed"] == 7


def test_patch_leaves_template_unchanged(template: workflow.Graph) -> None:
    original = copy.deepcopy(template)

    _patch(template, replace(workflow.read_settings(template), steps=8))

    assert template == original


@pytest.mark.parametrize("length", [80, 82, 83])
def test_fragment_length_must_be_4n_plus_1(
    template: workflow.Graph, length: int
) -> None:
    with pytest.raises(ValueError, match=r"4n \+ 1"):
        replace(workflow.read_settings(template), fragment_length=length)


def test_pose_window_must_not_end_before_it_starts(template: workflow.Graph) -> None:
    with pytest.raises(ValueError, match="pose_start_percent"):
        replace(
            workflow.read_settings(template),
            pose_start_percent=0.5,
            pose_end_percent=0.2,
        )


@pytest.mark.parametrize(
    "profile", sorted(p.stem for p in config.PROFILES_DIR.glob("*.toml"))
)
def test_profile_applies_to_template(template: workflow.Graph, profile: str) -> None:
    replace(workflow.read_settings(template), **config.load_profile(profile))


def test_zero_realism_strength_leaves_the_lora_out(template: workflow.Graph) -> None:
    settings = replace(workflow.read_settings(template), realism_lora_strength=0)

    graph = _patch(template, settings)

    loras = [n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert [n["_meta"]["title"] for n in loras] == ["Load LoRA"]
    links = [
        value[0]
        for node in graph.values()
        for value in node["inputs"].values()
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
    ]
    assert all(source in graph for source in links)


def test_export_without_realism_lora_reads_zero_and_rejects_it() -> None:
    export = workflow.load(
        config.REPO_ROOT / "workflows" / "wan_animate2_motion_transfer_api.json"
    )
    settings = workflow.read_settings(export)

    assert settings.realism_lora_strength == 0
    with pytest.raises(ValueError, match="Real-footage LoRA"):
        _patch(export, replace(settings, realism_lora_strength=2.0))


def test_result_and_comparison_outputs_are_told_apart() -> None:
    export = workflow.load(
        config.REPO_ROOT / "workflows" / "wan_animate2_motion_transfer_api.json"
    )

    assert workflow.video_outputs(export) == {"246": "result", "292": "comparison"}


def test_ui_export_is_rejected() -> None:
    with pytest.raises(ValueError, match="not an API-format workflow"):
        workflow.load(
            config.REPO_ROOT / "workflows" / "wan_animate2_motion_transfer.json"
        )


@pytest.mark.parametrize("count", [0, 2])
def test_find_node_requires_exactly_one_match(count: int) -> None:
    graph = {str(i): {"class_type": "LoadImage", "inputs": {}} for i in range(count)}

    with pytest.raises(ValueError, match=f"found {count}"):
        workflow.find_node(graph, "LoadImage")


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (81, [(0, 81)]),
        (82, [(0, 81), (80, 5)]),
        (150, [(0, 81), (80, 73)]),
        (417, [(0, 81), (80, 81), (160, 81), (240, 81), (320, 81), (400, 17)]),
    ],
)
def test_chunks_match_the_workflow_loop(
    frames: int, expected: list[tuple[int, int]]
) -> None:
    chunks = workflow.plan_chunks(frames, 81)

    assert [(chunk.start, chunk.length) for chunk in chunks] == expected


def test_chunks_cover_every_frame_in_4n_plus_1_lengths() -> None:
    for frames in range(2, 420):
        chunks = workflow.plan_chunks(frames, 81)

        assert chunks[0].start == 0, frames
        assert all(chunk.length % 4 == 1 for chunk in chunks), frames
        assert all(
            later.start == earlier.start + earlier.length - 1
            for earlier, later in zip(chunks, chunks[1:], strict=False)
        ), frames
        last = chunks[-1]
        assert frames <= last.start + last.length < frames + 4, frames


def test_take_needs_two_frames_to_chunk() -> None:
    with pytest.raises(ValueError, match="at least 2 frames"):
        workflow.plan_chunks(1, 81)


def _links(graph: workflow.Graph) -> list[str]:
    return [
        value[0]
        for node in graph.values()
        for value in node["inputs"].values()
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
    ]


def test_chunk_graph_drops_the_loop_and_the_comparison(
    template: workflow.Graph,
) -> None:
    take = _patch(template, workflow.read_settings(template))

    graph = workflow.chunk_graph(take, workflow.Chunk(80, 73), "last.png")

    classes = {node["class_type"] for node in graph.values()}
    assert not classes & {"StartLoop", "EndLoop", "ImageStitch"}
    assert all(source in graph for source in _links(graph))
    assert list(workflow.video_outputs(graph).values()) == ["result"]


def test_first_chunk_starts_at_the_first_frame_from_scratch(
    template: workflow.Graph,
) -> None:
    take = _patch(template, workflow.read_settings(template))

    graph = workflow.chunk_graph(take, workflow.Chunk(0, 81), None)

    animate = _inputs(graph, "WanAnimate2ToVideo")
    assert (animate["length"], animate["video_frame_offset"]) == (81, 0)
    assert "continue_motion" not in animate


def test_later_chunk_continues_from_the_given_frame(template: workflow.Graph) -> None:
    take = _patch(template, workflow.read_settings(template))

    graph = workflow.chunk_graph(take, workflow.Chunk(80, 73), "last.png")

    animate = _inputs(graph, "WanAnimate2ToVideo")
    # The node steps back one frame itself, so the pose lines up with frame 80.
    assert (animate["length"], animate["video_frame_offset"]) == (73, 81)
    [source, _] = animate["continue_motion"]
    assert graph[source]["inputs"]["image"] == "last.png"


def test_chunk_saves_its_last_decoded_frame(template: workflow.Graph) -> None:
    take = _patch(template, workflow.read_settings(template))

    graph = workflow.chunk_graph(take, workflow.Chunk(80, 73), "last.png")

    [pick, _] = graph[workflow.LAST_FRAME]["inputs"]["images"]
    assert graph[pick]["inputs"]["batch_index"] == 72
    [decoded, _] = graph[pick]["inputs"]["image"]
    assert graph[decoded]["class_type"] == "VAEDecode"


def test_chunk_graph_keeps_uploaded_assets_as_they_are(
    template: workflow.Graph,
) -> None:
    # SDK assets hold a lock, so they can't be deep-copied.
    asset = cast(Asset, threading.Lock())
    take = workflow.patch(
        template, workflow.read_settings(template), video=asset, image="a.png", seed=1
    )

    graph = workflow.chunk_graph(take, workflow.Chunk(0, 81), None)

    assert _inputs(graph, "LoadVideo")["file"] is asset


@pytest.mark.parametrize(
    ("model", "distilled_model"),
    [
        (
            "wan_animate_2_int8_convrot.safetensors",
            "wan_animate_2_distill_int8_convrot.safetensors",
        ),
        ("wan_animate_2_bf16.safetensors", "wan_animate_2_distill_bf16.safetensors"),
    ],
)
def test_distilled_recipe_keeps_the_precision_and_reaches_the_graph(
    template: workflow.Graph, model: str, distilled_model: str
) -> None:
    settings = replace(workflow.read_settings(template), model=model)

    graph = _patch(template, workflow.distilled(settings))

    assert _inputs(graph, "UNETLoader")["unet_name"] == distilled_model
    assert _inputs(graph, "KSamplerSelect")["sampler_name"] == "euler"
    assert _inputs(graph, "BasicScheduler")["steps"] == 10
