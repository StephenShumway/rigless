import io

import numpy as np
import pytest
from PIL import Image

from rigless import config, pose, sheet
from rigless.sheet import View

FIXTURES = config.REPO_ROOT / "local_fixtures"

# A standing figure facing the camera, in COCO order. The person's left side is
# on the image's right. Arms hang at the sides.
STANDING = np.array(
    [
        (0, -30), (5, -35), (-5, -35), (10, -32), (-10, -32),
        (20, 0), (-20, 0), (24, 30), (-24, 30), (26, 60), (-26, 60),
        (12, 60), (-12, 60), (12, 100), (-12, 100), (12, 140), (-12, 140),
    ],
    dtype=float,
)  # fmt: skip


def _view(keypoints: np.ndarray, full_body: bool = True) -> View:
    scores = np.ones(17)
    if not full_body:
        scores[11:] = 0.1
    return View((0, 0, 1, 1), np.zeros((1, 1), bool), keypoints, scores)


def _turned(turn: float) -> View:
    """The standing figure turned about its spine: 1 faces the camera, -1 away."""
    keypoints = STANDING.copy()
    keypoints[:, 0] *= turn
    return _view(keypoints)


def _close_up(nose_shift: float) -> View:
    """A face close-up; a turned head moves the nose off center between the eyes."""
    keypoints = STANDING * 4
    keypoints[0, 0] += nose_shift
    return _view(keypoints, full_body=False)


def test_front_view_is_the_one_facing_the_camera() -> None:
    views = [_turned(0.7), _turned(1.0), _turned(0.05), _turned(-1.0)]

    assert sheet.pick_front(views) is views[1]


def test_back_view_is_the_one_seen_from_behind() -> None:
    views = [_turned(1.0), _turned(-0.7), _turned(-1.0), _turned(0.05)]

    assert sheet.pick_back(views) is views[2]


def test_profile_is_not_taken_for_the_back() -> None:
    views = [_turned(1.0), _turned(0.05), _turned(-0.05)]

    assert sheet.pick_back(views) is None


def test_sheet_without_full_body_view_is_rejected() -> None:
    with pytest.raises(ValueError, match="full-body"):
        sheet.pick_front([_close_up(0)])


def test_frontal_close_up_beats_a_turned_one() -> None:
    views = [_close_up(nose_shift=15), _turned(1.0), _close_up(nose_shift=0)]

    assert sheet.pick_close_up(views) is views[2]


@pytest.mark.parametrize(
    "wrists",
    [[(4, 60), (-4, 60)], [(-8, 45), (8, 45)]],
    ids=["clasped", "crossed"],
)
def test_hands_on_the_torso_are_warned(wrists: list[tuple[float, float]]) -> None:
    keypoints = STANDING.copy()
    keypoints[[9, 10]] = wrists

    assert sheet.pose_warning(_view(keypoints)) is not None


@pytest.mark.parametrize(
    "wrists",
    [[(26, 60), (-26, 60)], [(4, -60), (-4, -60)]],
    ids=["at_sides", "overhead"],
)
def test_hands_off_the_torso_are_not_warned(wrists: list[tuple[float, float]]) -> None:
    keypoints = STANDING.copy()
    keypoints[[9, 10]] = wrists

    assert sheet.pose_warning(_view(keypoints)) is None


@pytest.mark.parametrize("transpose", [False, True], ids=["side_by_side", "stacked"])
def test_touching_figures_are_cut_where_they_touch(transpose: bool) -> None:
    # Two 100-pixel-tall figures joined by a 10-pixel bridge at columns 40-49.
    mask = np.zeros((100, 90), bool)
    mask[:, :40] = mask[:, 50:] = True
    mask[90:, 40:50] = True
    keypoints = [np.array([[20.0, 50.0]]), np.array([[70.0, 50.0]])]
    if transpose:
        mask = mask.T
        keypoints = [points[:, ::-1] for points in keypoints]

    left, right = sheet.split(mask, keypoints)

    expected_left = np.zeros((100, 90), bool)
    expected_left[:, :40] = True
    np.testing.assert_array_equal(left, expected_left.T if transpose else expected_left)
    np.testing.assert_array_equal(right, mask & ~left)


def test_blob_without_keypoints_is_dropped() -> None:
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:40] = True  # the figure
    mask[90:95, 60:95] = True  # a caption

    (figure,) = sheet.split(mask, [np.array([[25.0, 50.0]])])

    assert not figure[90:95, 60:95].any()
    assert figure[10:90, 10:40].all()


def test_a_photo_of_one_person_is_used_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the shirt stands out from the backdrop, as in a photo whose
    # backdrop isn't plain.
    photo = np.full((100, 60, 3), 180, np.uint8)
    photo[10:50, 20:40] = (200, 0, 0)
    person = pose.Person(
        np.array([0, 0, 60, 100]), STANDING * 0.5 + (30, 25), np.ones(17)
    )
    monkeypatch.setattr(pose, "find_people", lambda image: [person])

    views = sheet.extract([Image.fromarray(photo)])

    np.testing.assert_array_equal(np.asarray(views.front), photo)


def test_crop_paints_out_a_neighbor_in_the_margin() -> None:
    image = np.full((100, 60, 3), 180, np.uint8)
    image[10:90, 10:30] = (200, 0, 0)
    image[10:90, 32:50] = (0, 0, 200)
    red, blue = np.zeros((100, 60), bool), np.zeros((100, 60), bool)
    red[10:90, 10:30] = True
    blue[10:90, 32:50] = True
    views = [
        View((10, 10, 30, 90), red, STANDING, np.ones(17)),
        View((32, 10, 50, 90), blue, STANDING, np.ones(17)),
    ]

    pixels = np.asarray(sheet.crop(Image.fromarray(image), views, views[0]))

    assert not (pixels == (0, 0, 200)).all(axis=2).any()
    assert (pixels == (200, 0, 0)).all(axis=2).sum() == red.sum()


def _inside(box: sheet.Box, outer: sheet.Box, tolerance: int = 4) -> bool:
    return (
        box[0] >= outer[0] - tolerance
        and box[1] >= outer[1] - tolerance
        and box[2] <= outer[2] + tolerance
        and box[3] <= outer[3] + tolerance
    )


@pytest.mark.fixtures
def test_fixture_sheet_views_match_the_hand_crops() -> None:
    path = FIXTURES / "character_sheet.jpg"
    if not path.exists():
        pytest.skip("local_fixtures/character_sheet.jpg is missing")
    views = sheet.find_views(Image.open(path))

    front, close_up, back = (
        sheet.pick_front(views),
        sheet.pick_close_up(views),
        sheet.pick_back(views),
    )

    # The hand crops from HANDOFF.md: whole figure, nothing of the neighbors.
    assert _inside(front.box, (0, 0, 232, 604))
    assert front.box[3] - front.box[1] > 0.9 * 604
    assert close_up is not None
    assert _inside(close_up.box, (0, 604, 340, 1024))
    assert close_up.box[3] - close_up.box[1] > 0.9 * 420
    assert back is not None
    assert back.box[0] > 800  # the rightmost figure of the top row
    assert sheet.pose_warning(front) is None


@pytest.mark.fixtures
def test_clasped_hands_reference_is_warned() -> None:
    path = FIXTURES / "character_solo.jpg"
    if not path.exists():
        pytest.skip("local_fixtures/character_solo.jpg is missing")

    views = sheet.find_views(Image.open(path))

    assert sheet.pose_warning(sheet.pick_front(views)) is not None


def test_a_phone_photo_is_turned_the_way_it_is_shown() -> None:
    photo = Image.new("RGB", (20, 40))
    exif = photo.getexif()
    exif[0x0112] = 6  # stored sideways; shown turned 90 degrees clockwise
    photo.info["exif"] = exif.tobytes()
    photo = _reloaded(photo, "JPEG")

    assert sheet.upright(photo).size == (40, 20)


def test_transparent_parts_become_the_white_backdrop() -> None:
    cutout = Image.new("RGBA", (4, 4), (0, 0, 0, 0))

    assert sheet.upright(cutout).getpixel((0, 0)) == (255, 255, 255)


def _reloaded(image: Image.Image, format: str) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format, exif=image.info.get("exif", b""))
    buffer.seek(0)
    return Image.open(buffer)
