from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rigless import swap


def _masks(found: list[bool]) -> np.ndarray:
    """Each found frame marks its own index, so a copied mask shows its origin."""
    masks = np.zeros((len(found), 1, len(found)), bool)
    for index, seen in enumerate(found):
        masks[index, 0, index] = seen
    return masks


def test_lost_frames_take_the_nearest_found_mask() -> None:
    masks = _masks([False, True, False, False, False, True])

    filled = swap.fill_gaps(masks)

    assert [int(np.argmax(frame)) for frame in filled] == [1, 1, 1, 1, 5, 5]


def test_person_never_found_is_an_error() -> None:
    with pytest.raises(ValueError, match="Tap them again"):
        swap.fill_gaps(_masks([False, False]))


def test_each_extra_picture_is_named_by_its_number() -> None:
    text = swap.prompt(["front", "face", "view", "back"], notes="He wears a monocle.")

    assert "<Picture 2> is a close-up of the same person's face." in text
    assert "<Picture 3> is another view of them." in text
    assert "<Picture 4> shows them from behind." in text
    assert text.endswith("He wears a monocle.")


def test_one_picture_names_no_others() -> None:
    assert "<Picture 2>" not in swap.prompt(["front"])


def test_pictures_reach_h3_in_order() -> None:
    graph = swap.swap_graph(
        "clip.mp4",
        "masks.mp4",
        ["front.png", "face.png", "back.png"],
        prompt="",
        seed=1,
    )

    inputs = graph["conditioning"]["inputs"]
    given = [
        graph[inputs[f"ref_images.ref_image_{index}"][0]]["inputs"]["image"]
        for index in range(3)
    ]
    assert given == ["front.png", "face.png", "back.png"]


@pytest.mark.parametrize("count", [0, swap.MAX_PICTURES + 1])
def test_picture_count_outside_what_h3_takes_is_refused(count: int) -> None:
    with pytest.raises(ValueError):
        swap.swap_graph("clip.mp4", "masks.mp4", ["p.png"] * count, prompt="", seed=1)


def test_what_the_character_lacks_is_listed_as_absent() -> None:
    text = swap.prompt(["front"], unlike=["sunglasses", "rings", "dreadlocks"])

    assert text.endswith(
        "Unlike the original person, they have no sunglasses, no rings and no "
        "dreadlocks."
    )


def test_back_view_given_first_is_named_as_the_back() -> None:
    text = swap.prompt(["back", "front"])

    assert "<Picture 1> shows them from behind." in text
    assert "<Picture 2> is another view of them." in text


def _pair() -> list[swap.Casting]:
    return [
        swap.Casting("on the left", ("front", "face"), look="in a striped shirt"),
        swap.Casting("on the right", ("back", "front"), unlike=("sunglasses",)),
    ]


def test_second_persons_pictures_follow_the_firsts() -> None:
    text = swap.group_prompt(_pair())

    assert (
        "The person on the left, in a striped shirt, becomes the person in "
        "<Picture 1>. <Picture 2> is a close-up of the same person's face."
    ) in text
    assert (
        "The person on the right becomes the person in <Picture 3>. <Picture 3> "
        "shows them from behind. <Picture 4> is another view of them."
    ) in text
    assert "Unlike the original person on the right, they have no sunglasses." in text


def test_tags_in_the_notes_point_at_each_persons_first_picture() -> None:
    text = swap.group_prompt(_pair(), notes="@2 hands @1 the mic; @3 is nobody.")

    assert text.endswith(
        "the person in <Picture 3> hands the person in <Picture 1> the mic; @3 is "
        "nobody."
    )


def test_lead_in_fits_the_whole_figure_into_a_wide_frame(tmp_path: Path) -> None:
    picture = np.full((200, 100, 3), 128, np.uint8)  # a gray backdrop
    picture[40:180, 30:70] = (20, 40, 200)  # the figure, head to toe
    path = tmp_path / "front.png"
    Image.fromarray(picture).save(path)

    frame, mask = swap.lead_in_frame(path, (320, 100))

    # Halved to fit the height: 50 px wide, centered at x 135-185, figure at
    # x 150-170 and y 20-90.
    assert frame.shape == (100, 320, 3)
    assert frame[55, 160].tolist() == [200, 40, 20]  # BGR, like the clip's frames
    assert frame[50, 5].tolist() == [128, 128, 128]  # the backdrop fills the sides
    assert mask[20:90, 150:170].all()
    assert not mask[:, :100].any() and not mask[:, 220:].any()
