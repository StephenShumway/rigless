import numpy as np
import pytest
from PIL import Image

from rigless import compose, config, workflow

RED, BLUE = (200, 0, 0), (0, 0, 200)


def _face(x: float, y: float, eye_gap: float) -> np.ndarray:
    """COCO face keypoints (nose, eyes, ears) of a frontal face, eyes at (x, y)."""
    g = eye_gap
    return np.array(
        [
            (x, y + 0.5 * g),
            (x + g / 2, y),
            (x - g / 2, y),
            (x + 1.3 * g, y),
            (x - 1.3 * g, y),
        ]
    )


@pytest.fixture
def template() -> workflow.Graph:
    return workflow.load(config.COMPOSE_WORKFLOW_PATH)


def _patch(graph: workflow.Graph, close_up: str | None = "face.png") -> workflow.Graph:
    return compose.patch(
        graph,
        background="lobby.png",
        character="hero.png",
        close_up=close_up,
        prompt="put him in the lobby",
        seed=7,
    )


def _upstream(graph: workflow.Graph, node_id: str, input_name: str) -> str:
    """Follows links upstream from an input to the image file or prompt feeding it."""
    nid = graph[node_id]["inputs"][input_name][0]
    while True:
        inputs = graph[nid]["inputs"]
        if graph[nid]["class_type"] == "LoadImage":
            return str(inputs["image"])
        if "prompt" in inputs:
            return str(inputs["prompt"])
        nid = (inputs.get("image") or inputs["conditioning"])[0]


def _positive_encoder(graph: workflow.Graph) -> str:
    return workflow.find_node(
        graph, "TextEncodeQwenImageEditPlus", "TextEncodeQwenImageEditPlus (Positive)"
    )


def test_background_is_picture_1_and_sets_the_canvas(
    template: workflow.Graph,
) -> None:
    graph = _patch(template)

    encoder = _positive_encoder(graph)
    pictures = [_upstream(graph, encoder, f"image{i}") for i in (1, 2, 3)]
    assert pictures == ["lobby.png", "hero.png", "face.png"]
    assert _upstream(graph, workflow.find_node(graph, "VAEEncode"), "pixels") == (
        "lobby.png"
    )


def test_prompt_reaches_the_positive_conditioning(template: workflow.Graph) -> None:
    graph = _patch(template)

    sampler = workflow.find_node(graph, "KSampler")
    assert _upstream(graph, sampler, "positive") == "put him in the lobby"
    assert _upstream(graph, sampler, "negative") == ""


def test_without_close_up_no_link_is_left_dangling(template: workflow.Graph) -> None:
    graph = _patch(template, close_up=None)

    links = [
        value[0]
        for node in graph.values()
        for value in node["inputs"].values()
        if isinstance(value, list)
    ]
    assert all(source in graph for source in links)
    assert "image3" not in graph[_positive_encoder(graph)]["inputs"]


def test_wide_photo_keeps_its_height_and_gains_a_floor_band() -> None:
    photo = Image.new("RGB", (600, 900), RED)
    photo.paste(BLUE, (0, 880, 600, 900))

    canvas = compose.fit_background(photo, 720, 1280)

    assert canvas.size == (600, 1067)
    assert canvas.crop((0, 0, 600, 900)).tobytes() == photo.tobytes()
    assert canvas.getpixel((300, 1066)) == BLUE


def test_floor_band_is_at_most_a_quarter_of_the_canvas() -> None:
    landscape = Image.new("RGB", (1920, 1080), RED)

    canvas = compose.fit_background(landscape, 720, 1280)

    assert canvas.size == (810, 1440)


def test_narrow_photo_is_cropped_not_padded() -> None:
    photo = Image.new("RGB", (400, 1000), RED)
    photo.paste(BLUE, (0, 0, 400, 100))

    canvas = compose.fit_background(photo, 720, 1280)

    assert canvas.size == (400, 711)
    assert canvas.getpixel((200, 0)) == RED


def test_similarity_transform_recovers_scale_rotation_and_shift() -> None:
    src = _face(50, 60, 20)
    turn = np.array([[np.cos(0.1), -np.sin(0.1)], [np.sin(0.1), np.cos(0.1)]])
    dst = 0.83 * src @ turn.T + (12, -7)

    matrix = compose.similarity_transform(src, dst)

    np.testing.assert_allclose(src @ matrix[:, :2].T + matrix[:, 2], dst, atol=1e-9)


def test_face_box_near_the_edge_stays_square_and_inside() -> None:
    points = _face(15, 10, 20)

    left, top, right, bottom = compose.face_box(points, (400, 600))

    assert right - left == bottom - top
    assert (left, top) == (0, 0)
    assert right <= 400 and bottom <= 600


def test_blend_face_changes_only_the_face() -> None:
    rng = np.random.default_rng(0)
    reference = Image.fromarray(rng.integers(0, 256, (600, 400, 3), dtype=np.uint8))
    points = _face(200, 200, 30)
    box = compose.face_box(points, reference.size)
    edit = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8))
    edit_points = (points - box[:2]) * 512 / (box[2] - box[0])

    result = compose.blend_face(reference, box, points, edit, edit_points)

    changed = np.any(np.asarray(result) != np.asarray(reference), axis=2)
    ys, xs = np.nonzero(changed)
    assert changed[215, 200]  # the nose
    # Temples to chin (the oval) plus its soft edge, never the hair or backdrop.
    assert xs.min() >= 200 - 1.5 * 30 and xs.max() <= 200 + 1.5 * 30
    assert ys.min() >= 200 - 1.55 * 30 and ys.max() <= 200 + 2.2 * 30
