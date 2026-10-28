import numpy as np
import pytest

from rigless.pose import (
    NUM_KEYPOINTS,
    SHOULDERS,
    PoseTrack,
    fit_boxes,
    starts_facing_away,
)


def test_boxes_follow_a_portrait_clip_onto_h3s_cropped_canvas() -> None:
    # 1080x1920 covers 736x1344 at 0.7 (756 wide), so 10 px is cropped per side.
    centered = np.array([490.0, 860.0, 590.0, 1060.0])

    [box, gone] = fit_boxes([centered, None], (1080, 1920), (736, 1344))

    assert box is not None
    assert np.allclose(box, [333, 602, 403, 742])
    assert gone is None


def _shoulders(left_x: float, right_x: float, score: float = 0.9) -> PoseTrack:
    """One second of poses with the shoulders at the given x positions."""
    keypoints = np.zeros((24, NUM_KEYPOINTS, 2))
    keypoints[:, SHOULDERS[0], 0] = left_x
    keypoints[:, SHOULDERS[1], 0] = right_x
    scores = np.zeros((24, NUM_KEYPOINTS))
    scores[:, SHOULDERS] = score
    return PoseTrack(keypoints, scores, fps=24.0)


@pytest.mark.parametrize(
    ("track", "away"),
    [
        (_shoulders(100, 200), True),
        (_shoulders(200, 100), False),
        (_shoulders(100, 200, score=0.4), False),
    ],
    ids=["left shoulder on the left", "facing the camera", "shoulders unsure"],
)
def test_back_to_the_camera_is_told_by_the_shoulders(
    track: PoseTrack, away: bool
) -> None:
    assert starts_facing_away(track) is away
