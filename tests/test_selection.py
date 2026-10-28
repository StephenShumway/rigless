import json

import numpy as np
import pytest

from rigless.pose import NUM_KEYPOINTS, Person, PoseTrack
from rigless.selection import (
    Anchor,
    Segment,
    assemble,
    body_taps,
    head_taps,
    lost_spans,
    merge,
    plan_segments,
    tint,
    track_graph,
)

# A 40x40 frame with a face at (20, 10), shoulders at y=21, 16 px apart (so the
# chest tap goes at (20, 29)), and hips at y=36. The whole person is masked from
# row 5; a shirt-only mask starts at the shoulders; a head-only mask stops there.
FACE = (20, 10)
PERSON = np.zeros((40, 40), bool)
PERSON[5:, 10:30] = True
SHIRT = np.zeros((40, 40), bool)
SHIRT[21:, 10:30] = True
HEAD = np.zeros((40, 40), bool)
HEAD[5:21, 14:26] = True


def _posed(
    frames: int, face: float = 0.9, shoulders: float = 0.9, hips: float = 0.9
) -> PoseTrack:
    """Poses with the given confidences for the face, the shoulders and the hips."""
    keypoints = np.zeros((frames, NUM_KEYPOINTS, 2))
    scores = np.zeros((frames, NUM_KEYPOINTS))
    x, y = FACE
    keypoints[:, :7] = [(x, y), (x - 2, y - 1), (x + 2, y - 1), (x - 4, y), (x + 4, y),
                        (x + 8, 21), (x - 8, 21)]  # fmt: skip
    keypoints[:, 11:13] = [(x + 6, 36), (x - 6, 36)]
    scores[:, :5] = face
    scores[:, 5:7] = shoulders
    scores[:, 11:13] = hips
    return PoseTrack(keypoints, scores, fps=24.0)


def _anchor(frame: int) -> Anchor:
    return Anchor(frame=frame, positive=((10, 20),))


def _numbered_runs(segments: list[Segment]) -> list[np.ndarray]:
    """Runs whose masks say which footage frame they belong to: frame f has f + 1
    pixels set, so a mask that lands on the wrong frame shows."""
    runs = []
    for segment in segments:
        order = (
            range(segment.anchor.frame, -1, -1)
            if segment.backward
            else range(segment.start, segment.start + segment.length)
        )
        run = np.zeros((segment.length, 1, 64), bool)
        for index, frame in enumerate(order):
            run[index, 0, : frame + 1] = True
        runs.append(run)
    return runs


def test_anchor_on_the_first_frame_tracks_forward_only() -> None:
    assert plan_segments([_anchor(0)], frames=48) == [
        Segment(_anchor(0), start=0, length=48, backward=False)
    ]


def test_later_anchors_split_the_clip_and_the_first_also_tracks_back() -> None:
    segments = plan_segments([_anchor(30), _anchor(12)], frames=48)

    assert segments == [
        Segment(_anchor(12), start=0, length=13, backward=True),
        Segment(_anchor(12), start=12, length=18, backward=False),
        Segment(_anchor(30), start=30, length=18, backward=False),
    ]


@pytest.mark.parametrize(
    "anchors",
    [[], [_anchor(5), _anchor(5)], [_anchor(48)], [_anchor(-1)]],
    ids=["none", "two on one frame", "past the end", "before the start"],
)
def test_anchors_that_cannot_be_tracked_are_refused(anchors: list[Anchor]) -> None:
    with pytest.raises(ValueError):
        plan_segments(anchors, frames=48)


def test_assembled_masks_land_on_their_own_frames() -> None:
    segments = plan_segments([_anchor(9), _anchor(25)], frames=40)

    masks = assemble(segments, _numbered_runs(segments), frames=40)

    assert [int(mask.sum()) for mask in masks] == [frame + 1 for frame in range(40)]


def test_short_tracking_result_is_an_error_not_a_gap() -> None:
    segments = plan_segments([_anchor(0)], frames=40)
    short = np.zeros((39, 4, 4), bool)

    with pytest.raises(ValueError, match="39 frames"):
        assemble(segments, [short], frames=40)


def test_lost_spans_cover_the_empty_frames() -> None:
    masks = np.ones((10, 2, 2), bool)
    masks[0] = False
    masks[4:7] = False

    assert lost_spans(masks, fps=2) == [(0.0, 0.5), (2.0, 3.5)]


def test_backward_segment_reads_the_reversed_clip_from_its_anchor() -> None:
    anchor = Anchor(frame=12, positive=((10, 20),), negative=((30, 40),))
    segments = plan_segments([anchor], frames=48)

    graph = track_graph("clip.mp4", "reversed.mp4", segments, frames=48, fps=24)

    backward_part = graph["part_0"]["inputs"]
    assert backward_part["image"] == ["backward", 0]
    assert (backward_part["batch_index"], backward_part["length"]) == (35, 13)
    taps = graph["detect_0"]["inputs"]
    assert json.loads(taps["positive_coords"]) == [{"x": 10, "y": 20}]
    assert json.loads(taps["negative_coords"]) == [{"x": 30, "y": 40}]
    assert {f"save_{index}" for index in range(len(segments))} <= graph.keys()


def test_mask_without_the_head_is_retapped_where_it_starts() -> None:
    masks = np.stack([PERSON] * 5 + [SHIRT] * 15)

    taps = head_taps(_posed(20), masks)

    assert [tap.frame for tap in taps] == [5]
    assert taps[0].positive == (FACE, (28, 21), (12, 21))  # the nose and shoulders


@pytest.mark.parametrize(
    ("masks", "face", "shoulders"),
    [
        ([PERSON] * 20, 0.9, 0.9),
        ([PERSON] * 10 + [SHIRT] * 3 + [PERSON] * 7, 0.9, 0.9),
        ([SHIRT] * 20, 0.1, 0.9),
        ([SHIRT] * 20, 0.9, 0.4),
    ],
    ids=["whole person", "brief miss", "face not seen", "seen from behind"],
)
def test_no_head_retap_unless_a_seen_head_stays_out(
    masks: list[np.ndarray], face: float, shoulders: float
) -> None:
    assert head_taps(_posed(len(masks), face, shoulders), np.stack(masks)) == []


def _person(hips: float = 0.9) -> Person:
    track = _posed(1, hips=hips)
    return Person(np.array([8.0, 2.0, 32.0, 40.0]), track.keypoints[0], track.scores[0])


def test_head_only_mask_gets_taps_down_the_body() -> None:
    taps = body_taps([_person()], HEAD)

    assert taps == (FACE, (28, 21), (12, 21), (20, 29), (26, 36), (14, 36))


@pytest.mark.parametrize(
    ("mask", "hips"), [(PERSON, 0.9), (HEAD, 0.2)], ids=["whole person", "hips unseen"]
)
def test_no_body_taps_unless_seen_hips_are_left_out(
    mask: np.ndarray, hips: float
) -> None:
    assert body_taps([_person(hips)], mask) == ()


def test_no_body_taps_on_a_bystander_when_the_tapped_person_isnt_found() -> None:
    bystander = _person()
    bystander.box[:] = [30.0, 0.0, 40.0, 40.0]  # none of the head mask is inside

    assert body_taps([bystander], HEAD) == ()


def test_taps_on_a_tapped_frame_join_the_users_taps() -> None:
    mine = Anchor(0, ((5, 5),), ((9, 9),))
    extra = [Anchor(0, ((6, 6),), ((8, 8),)), Anchor(12, ((7, 7),))]

    merged = merge([mine, _anchor(30)], extra)

    assert sorted(merged, key=lambda a: a.frame) == [
        Anchor(0, ((5, 5), (6, 6)), ((9, 9), (8, 8))),
        Anchor(12, ((7, 7),)),
        _anchor(30),
    ]


def test_tint_blends_each_person_in_their_colour_and_leaves_the_rest() -> None:
    frames = np.full((1, 2, 2, 3), 100, np.uint8)
    first = np.array([[[True, False], [False, False]]])
    third = np.array([[[False, False], [False, True]]])

    out = tint(frames, [first, None, third])

    # 100 blended 45% toward TINTS[0] and TINTS[2].
    assert out[0, 0, 0].tolist() == [120, 157, 154]
    assert out[0, 1, 1].tolist() == [123, 130, 159]
    assert out[0, 0, 1].tolist() == out[0, 1, 0].tolist() == [100, 100, 100]
