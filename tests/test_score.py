import numpy as np

from rigless.pose import PoseTrack
from rigless.score import score_take

# COCO keypoints with left and right swapped.
SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _figure(
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    fps: float = 30.0,
    scale: float = 1.0,
    shift: tuple[float, float] = (0.0, 0.0),
) -> PoseTrack:
    """A standing stick figure whose straight arms point at the given angles.

    Angles are in radians per frame, 0 pointing along +x in image coordinates.
    """
    frames = len(left_arm)
    body = {
        0: (0, -30), 1: (-5, -35), 2: (5, -35), 3: (-10, -32), 4: (10, -32),
        5: (-20, 0), 6: (20, 0), 11: (-10, 60), 12: (10, 60),
        13: (-10, 100), 14: (10, 100), 15: (-10, 140), 16: (10, 140),
    }  # fmt: skip
    keypoints = np.zeros((frames, 17, 2))
    for index, point in body.items():
        keypoints[:, index] = point
    for shoulder, elbow, wrist, angles in ((5, 7, 9, left_arm), (6, 8, 10, right_arm)):
        direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        keypoints[:, elbow] = keypoints[:, shoulder] + 30 * direction
        keypoints[:, wrist] = keypoints[:, shoulder] + 60 * direction
    keypoints = keypoints * scale + np.array(shift)
    return PoseTrack(keypoints, np.ones((frames, 17)), fps)


def _swing(seconds: float, fps: float) -> np.ndarray:
    """Arm angles for a lively one-cycle-per-second swing."""
    t = np.arange(round(seconds * fps)) / fps
    return np.pi / 2 + 1.2 * np.sin(2 * np.pi * t)


def test_same_motion_scores_full_marks_regardless_of_framing() -> None:
    source = _figure(_swing(4, 30), np.pi - _swing(4, 30))
    reframed = _figure(_swing(4, 30), np.pi - _swing(4, 30), scale=0.4, shift=(300, 80))

    result = score_take(source, reframed)

    assert result.score > 99
    assert not result.frozen
    assert not result.mirrored


def test_character_standing_still_is_frozen() -> None:
    source = _figure(_swing(4, 30), np.pi - _swing(4, 30))
    still = np.full(120, _swing(4, 30)[0])

    result = score_take(source, _figure(still, np.pi - still))

    assert result.frozen
    assert result.energy_ratio is not None and result.energy_ratio < 0.1


def test_mirrored_take_is_flagged() -> None:
    source = _figure(_swing(4, 30), np.full(120, np.pi / 2))
    flipped = source.keypoints[:, SWAP] * np.array([-1.0, 1.0])

    result = score_take(source, PoseTrack(flipped, source.scores, source.fps))

    assert result.mirrored


def test_limbs_found_with_low_confidence_are_ignored() -> None:
    source = _figure(_swing(4, 30), np.pi - _swing(4, 30))
    keypoints, scores = source.keypoints.copy(), source.scores.copy()
    keypoints[:, 9] = keypoints[:, 7] - (keypoints[:, 9] - keypoints[:, 7])  # backwards
    scores[:, 9] = 0.1

    result = score_take(source, PoseTrack(keypoints, scores, source.fps))

    assert result.score > 99


def test_arms_the_pose_model_misses_dont_make_a_take_frozen() -> None:
    source = _figure(_swing(4, 30), np.pi - _swing(4, 30))
    scores = source.scores.copy()
    scores[:90, 7:11] = 0.1  # elbows and wrists unseen for 3 of the 4 seconds

    result = score_take(source, PoseTrack(source.keypoints, scores, source.fps))

    assert not result.frozen


def test_frames_are_matched_by_time_across_frame_rates() -> None:
    source = _figure(_swing(4, 30), np.pi - _swing(4, 30), fps=30)
    output = _figure(_swing(4, 24), np.pi - _swing(4, 24), fps=24)

    # Nearest-frame matching costs a few points; matching by index scores ~50.
    assert score_take(source, output).score > 90


def test_take_with_nobody_in_it_scores_zero() -> None:
    source = _figure(_swing(4, 30), np.pi - _swing(4, 30))
    empty = PoseTrack(np.zeros_like(source.keypoints), np.zeros((120, 17)), 30)

    assert score_take(source, empty).score == 0
