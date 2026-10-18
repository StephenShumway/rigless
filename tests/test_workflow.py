import copy
from dataclasses import replace
from typing import Any

import pytest

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


def test_result_and_comparison_outputs_are_told_apart(template: workflow.Graph) -> None:
    assert workflow.video_outputs(template) == {"246": "result", "292": "comparison"}


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
