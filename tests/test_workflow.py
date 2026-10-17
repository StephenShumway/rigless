import copy
from typing import Any

import pytest

from rigless import config, workflow


@pytest.fixture
def template() -> workflow.Graph:
    return workflow.load(config.WORKFLOW_PATH)


def _inputs(graph: workflow.Graph, class_type: str) -> dict[str, Any]:
    return graph[workflow.find_node(graph, class_type)]["inputs"]


def test_patch_sets_video_image_and_seed(template: workflow.Graph) -> None:
    graph = workflow.patch(template, video="dance.mp4", image="hero.png", seed=7)

    assert _inputs(graph, "LoadVideo")["file"] == "dance.mp4"
    assert _inputs(graph, "LoadImage")["image"] == "hero.png"
    assert _inputs(graph, "SamplerCustom")["noise_seed"] == 7


def test_patch_leaves_template_unchanged(template: workflow.Graph) -> None:
    original = copy.deepcopy(template)

    workflow.patch(template, video="dance.mp4", image="hero.png", seed=7)

    assert template == original


def test_result_and_comparison_outputs_are_told_apart(template: workflow.Graph) -> None:
    assert workflow.video_outputs(template) == {"246": "result", "292": "comparison"}


def test_output_size_comes_from_pose_video_resize(template: workflow.Graph) -> None:
    assert workflow.output_size(template) == (482, 854)


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
