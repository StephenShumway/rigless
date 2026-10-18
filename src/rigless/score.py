"""Motion-accuracy scoring of a generated take against its driving video.

Takes are compared by limb directions, not positions: the output reframes the
shot and the character's proportions differ from the dancer's, so only the
angles of arms, legs, and torso are comparable. Frames are matched by time, so
videos with different frame rates compare correctly.

Typical usage example:

  result = score_take(pose.estimate(driving), pose.estimate(take))
  print(result.score, result.frozen)
"""

from dataclasses import dataclass

import numpy as np

from rigless.pose import SEEN, PoseTrack

# Limbs as (start, end, weight) in COCO keypoints, plus two virtual points:
# 17 = between the shoulders, 18 = between the hips. Arms count double because
# they carry most of a dance and are where takes freeze.
LIMBS = (
    (5, 7, 2.0), (7, 9, 2.0), (6, 8, 2.0), (8, 10, 2.0),  # arms
    (11, 13, 1.0), (13, 15, 1.0), (12, 14, 1.0), (14, 16, 1.0),  # legs
    (18, 17, 1.0),  # torso
)  # fmt: skip
ARMS = slice(0, 4)
WEIGHTS = np.array([weight for _, _, weight in LIMBS])
# Left and right swapped, for checking whether a take is mirrored.
MIRROR = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)

TOLERANCE_DEG = 45.0  # a limb this far off counts as no match at all
SMOOTH_FRAMES = 5  # averages out keypoint jitter before measuring movement
ACTIVE_ARMS_DEG = 60.0  # arm turning per second above which the dancer is "moving"
FROZEN_RATIO = 0.4  # character's arm movement relative to the dancer's
MIRROR_MARGIN = 5.0  # points by which the mirrored comparison must win


@dataclass(frozen=True)
class TakeScore:
    """A motion-accuracy score for one generated take.

    Attributes:
        score: Overall accuracy from 0 to 100.
        per_frame: Accuracy of each compared frame from 0 to 100, NaN where no
            limb was found in both videos. Drops show chunk seams.
        energy_ratio: Median, over the seconds the dancer's arms move, of how
            much the character's arms move relative to the dancer's; 1.0 means
            equally. None if the dancer's arms never move.
        frozen: True if the character barely moves while the dancer does.
        mirrored: True if the take matches the dancer better flipped left to
            right.
    """

    score: float
    per_frame: np.ndarray
    energy_ratio: float | None
    frozen: bool
    mirrored: bool

    def as_record(self) -> dict[str, object]:
        """Returns the score as JSON-ready values, rounded, with None for NaN."""
        return {
            "score": round(self.score, 1),
            "energy_ratio": None
            if self.energy_ratio is None
            else round(self.energy_ratio, 2),
            "frozen": self.frozen,
            "mirrored": self.mirrored,
            "per_frame": [
                None if np.isnan(v) else round(float(v), 1) for v in self.per_frame
            ],
        }


def score_take(source: PoseTrack, output: PoseTrack) -> TakeScore:
    """Scores how faithfully a take's character follows the source dancer.

    Args:
        source: Poses of the driving video.
        output: Poses of the generated take.
    """
    shown = _source_frames(source, output)
    src_dirs, src_seen = _limb_directions(source)
    src_dirs, src_seen = src_dirs[shown], src_seen[shown]
    out_dirs, out_seen = _limb_directions(output)
    out_dirs, out_seen = out_dirs[: len(shown)], out_seen[: len(shown)]

    per_frame = _frame_scores(src_dirs, src_seen, out_dirs, out_seen)
    score = _mean(per_frame)
    mirror_dirs, mirror_seen = _limb_directions(_mirrored(output))
    mirror_score = _mean(
        _frame_scores(
            src_dirs, src_seen, mirror_dirs[: len(shown)], mirror_seen[: len(shown)]
        )
    )
    ratio = _energy_ratio(src_dirs, src_seen, out_dirs, out_seen, output.fps)
    return TakeScore(
        score=score,
        per_frame=per_frame,
        energy_ratio=ratio,
        frozen=ratio is not None and ratio < FROZEN_RATIO,
        mirrored=mirror_score > score + MIRROR_MARGIN,
    )


def _source_frames(source: PoseTrack, output: PoseTrack) -> np.ndarray:
    """Returns the source frame shown at the same moment as each output frame.

    Output frames past the end of the source are left out.
    """
    times = np.arange(len(output.keypoints)) / output.fps
    shown = np.round(times * source.fps).astype(int)
    return shown[shown < len(source.keypoints)]


def _limb_directions(track: PoseTrack) -> tuple[np.ndarray, np.ndarray]:
    """Returns each limb's unit direction per frame, and whether it was found.

    Returns:
        A tuple (directions, found), where directions is (frames, limbs, 2) and
        found is (frames, limbs) bool.
    """
    points = _smooth(track.keypoints)
    shoulders, hips = points[:, [5, 6]].mean(axis=1), points[:, [11, 12]].mean(axis=1)
    points = np.concatenate([points, shoulders[:, None], hips[:, None]], axis=1)
    scores = np.concatenate(
        [
            track.scores,
            track.scores[:, [5, 6]].min(axis=1, keepdims=True),
            track.scores[:, [11, 12]].min(axis=1, keepdims=True),
        ],
        axis=1,
    )
    starts = [start for start, _, _ in LIMBS]
    ends = [end for _, end, _ in LIMBS]
    vectors = points[:, ends] - points[:, starts]
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    directions = vectors / np.maximum(lengths, 1e-9)
    found = (scores[:, starts] > SEEN) & (scores[:, ends] > SEEN)
    return directions, found


def _smooth(keypoints: np.ndarray) -> np.ndarray:
    """Averages each keypoint over a few neighboring frames."""
    pad = SMOOTH_FRAMES // 2
    padded = np.pad(keypoints, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    windows = [padded[i : i + len(keypoints)] for i in range(SMOOTH_FRAMES)]
    return np.mean(windows, axis=0)


def _mirrored(track: PoseTrack) -> PoseTrack:
    """Returns the track flipped left to right, with left and right swapped."""
    keypoints = track.keypoints[:, MIRROR] * np.array([-1.0, 1.0])
    return PoseTrack(keypoints, track.scores[:, MIRROR], track.fps)


def _frame_scores(
    src_dirs: np.ndarray,
    src_seen: np.ndarray,
    out_dirs: np.ndarray,
    out_seen: np.ndarray,
) -> np.ndarray:
    """Returns each frame's weighted limb agreement, 0 to 100, or NaN."""
    cosine = np.clip((src_dirs * out_dirs).sum(axis=-1), -1.0, 1.0)
    agreement = np.clip(1 - np.degrees(np.arccos(cosine)) / TOLERANCE_DEG, 0.0, 1.0)
    weights = WEIGHTS * (src_seen & out_seen)
    total = weights.sum(axis=1)
    scores = np.full(len(total), np.nan)
    has_limbs = total > 0
    scores[has_limbs] = (
        100 * (agreement * weights).sum(axis=1)[has_limbs] / total[has_limbs]
    )
    return scores


def _mean(per_frame: np.ndarray) -> float:
    """Returns the mean of the scored frames, or 0 if none could be scored."""
    scored = per_frame[~np.isnan(per_frame)]
    return float(scored.mean()) if len(scored) else 0.0


def _energy_ratio(
    src_dirs: np.ndarray,
    src_seen: np.ndarray,
    out_dirs: np.ndarray,
    out_seen: np.ndarray,
    fps: float,
) -> float | None:
    """Compares arm movement second by second while the dancer's arms move.

    The median ignores a single unrelated gesture, so a take that stands still
    except for one random arm raise still reads as frozen.
    """
    src_turning, out_turning = (
        _arm_turning(src_dirs, src_seen),
        _arm_turning(out_dirs, out_seen),
    )
    window = max(1, round(fps))
    ratios = []
    for start in range(0, len(src_turning) - window + 1, window):
        src_turn = src_turning[start : start + window].sum()
        if src_turn >= ACTIVE_ARMS_DEG:
            ratios.append(out_turning[start : start + window].sum() / src_turn)
    return float(np.median(ratios)) if ratios else None


def _arm_turning(directions: np.ndarray, found: np.ndarray) -> np.ndarray:
    """Returns how many degrees the arms turn between each pair of frames."""
    angles = np.arctan2(directions[:, ARMS, 1], directions[:, ARMS, 0])
    step = np.abs(np.angle(np.exp(1j * np.diff(angles, axis=0))))
    seen = found[1:, ARMS] & found[:-1, ARMS]
    return np.degrees(np.where(seen, step, 0.0)).sum(axis=1)
