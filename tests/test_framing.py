import numpy as np
import pytest

from rigless.framing import centered, check, reframe
from rigless.pose import PoseTrack

LANDSCAPE = (1920, 1080, 0, 0)
PORTRAIT = 9 / 16
FPS = 30.0
# A standing figure in COCO order, arms down; its torso is 60 units long.
BODY = np.array(
    [
        (0, -30), (-5, -35), (5, -35), (-10, -32), (10, -32),  # face
        (-20, 0), (20, 0), (-22, 30), (22, 30), (-24, 60), (24, 60),  # arms
        (-10, 60), (10, 60), (-10, 100), (10, 100), (-10, 140), (10, 140),  # legs
    ],
    dtype=float,
)  # fmt: skip
WINDOW = (540, 960, 690, 60)  # holds a 3x dancer at (960, 400) with room to spare


def _dancer(
    seconds: float = 4, scale: float = 3, at: tuple[float, float] = (960, 400)
) -> PoseTrack:
    frames = round(seconds * FPS)
    keypoints = np.tile(BODY * scale + np.array(at), (frames, 1, 1))
    return PoseTrack(keypoints, np.ones((frames, 17)), FPS, [np.empty((0, 4))] * frames)


def _inside(window: tuple[int, int, int, int], points: np.ndarray) -> bool:
    width, height, x, y = window
    return bool(
        (points[..., 0] >= x).all()
        and (points[..., 0] <= x + width).all()
        and (points[..., 1] >= y).all()
        and (points[..., 1] <= y + height).all()
    )


def test_window_closes_in_on_an_off_center_dancer_at_the_output_aspect() -> None:
    track = _dancer(at=(400, 500))

    window = reframe(track, LANDSCAPE, PORTRAIT)

    width, height, _, _ = window
    assert width / height == pytest.approx(PORTRAIT, rel=0.01)
    assert _inside(window, track.keypoints)
    assert _inside(LANDSCAPE, np.array([window[2:], np.add(window[2:], window[:2])]))
    assert height < 1080 * 0.75


def test_dancer_crossing_the_frame_gets_the_largest_window_that_fits() -> None:
    track = _dancer()
    track.keypoints[: len(track.keypoints) // 2] += [-700, 0]
    track.keypoints[len(track.keypoints) // 2 :] += [700, 0]

    width, height, x, y = reframe(track, LANDSCAPE, PORTRAIT)

    assert (height, y) == (1080, 0)
    assert x >= 0 and x + width <= 1920


def test_one_frame_keypoint_glitch_does_not_widen_the_window() -> None:
    steady = _dancer()
    glitchy = _dancer()
    glitchy.keypoints[50, 9] = (1850, 400)

    assert reframe(glitchy, LANDSCAPE, PORTRAIT) == reframe(steady, LANDSCAPE, PORTRAIT)


def test_dancer_never_found_gets_the_centered_window() -> None:
    track = _dancer()
    nobody = PoseTrack(track.keypoints, np.zeros_like(track.scores), FPS)

    assert reframe(nobody, LANDSCAPE, PORTRAIT) == centered(LANDSCAPE, PORTRAIT)


def test_well_framed_dancer_gets_no_warnings() -> None:
    assert check(_dancer(), WINDOW) == []


def test_hand_leaving_the_frame_is_reported_with_its_time() -> None:
    track = _dancer()
    track.keypoints[81:96, 10] = (1100, 20)  # raised above the window's top

    assert check(track, WINDOW) == [
        "A hand leaves the frame at 2.7-3.2s; step back from the camera."
    ]


def test_keypoint_jitter_at_the_edge_is_not_reported() -> None:
    track = _dancer()
    track.keypoints[81:83, 10] = (1100, 20)

    assert check(track, WINDOW) == []


def test_flickering_problem_is_reported_as_one_stretch() -> None:
    track = _dancer()
    track.keypoints[30:40, 10] = (1100, 20)
    track.keypoints[45:60, 10] = (1100, 20)

    assert check(track, WINDOW) == [
        "A hand leaves the frame at 1.0-2.0s; step back from the camera."
    ]


def test_feet_cut_off_below_the_ankles_are_reported() -> None:
    track = _dancer()
    feet_bottom = int(track.keypoints[0, 15, 1]) + 10  # ankles inside, soles not

    warnings = check(track, (540, feet_bottom - 60, 690, 60))

    assert warnings == [
        "The feet leave the frame at 0.0-4.0s; step back or tilt the camera down."
    ]


@pytest.mark.parametrize(
    ("box", "reported"),
    [((1000, 300, 1200, 900), True), ((1400, 300, 1600, 900), False)],
)
def test_other_person_is_reported_only_inside_the_window(
    box: tuple[int, int, int, int], reported: bool
) -> None:
    track = _dancer()
    for frame in range(60, 90):
        track.others[frame] = np.array([box], dtype=float)

    warnings = check(track, WINDOW)

    assert warnings == (
        ["Someone else is in the frame at 2.0-3.0s; film the dancer alone."]
        if reported
        else []
    )


def test_blurry_stretch_is_reported() -> None:
    track = _dancer()
    track.scores[30:60] = 0.4

    [warning] = check(track, WINDOW)

    assert warning.startswith("The dancer's pose is hard to make out at 1.0-2.0s")


def test_dancer_lost_for_a_while_is_reported() -> None:
    track = _dancer()
    track.scores[30:60] = 0

    assert check(track, WINDOW) == [
        "The dancer can't be seen at 1.0-2.0s; keep them in the frame throughout."
    ]


def test_dancer_filling_the_frame_keeps_the_whole_frame() -> None:
    frame = (1080, 1920, 0, 0)
    track = _dancer(scale=13.75, at=(540, 600))  # just too big for the frame

    assert reframe(track, frame, 720 / 1280) == frame
