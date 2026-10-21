"""Framing the dancer: auto-reframe and the input quality gate.

Both work from the dancer's pose track in the source video. `reframe` picks one
fixed window around the dancer at the output's aspect ratio, and `check` warns
in plain language about anything in that window that makes the dance hard to
follow: the dancer leaving it, someone else in it, or a pose too blurry to
read. Warnings never block a render.

The window is fixed, not panned with the dancer: the generated background
stands still, so a crop that followed the dancer would turn steps across the
floor into sliding on the spot.

Typical usage example:

  track = pose.estimate(Path("dance.mp4"))
  window = reframe(track, frame=(1920, 1080, 0, 0), aspect=720 / 1280)
  for warning in check(track, window):
      print(warning)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from rigless.pose import SEEN, PoseTrack
from rigless.video_prep import Crop

# Room left around the keypoints, in torso lengths (shoulders to hips): hands,
# feet, and the top of the head reach past the wrists, ankles, and eyes.
PAD_TORSOS = 0.5
# How far each body part reaches past its keypoints, in torso lengths; past the
# window edge, that part is cut off.
PARTS = (
    ("The head leaves", (0, 1, 2), 0.25, "step back or tilt the camera up"),
    ("A hand leaves", (9, 10), 0.25, "step back from the camera"),
    ("The feet leave", (15, 16), 0.15, "step back or tilt the camera down"),
)
MIN_FOUND = 4  # keypoints needed in a frame to count the dancer as found
GLITCH_FRAMES = 5  # running median length that drops single-frame keypoint glitches
LOW_CONFIDENCE = 0.5  # median keypoint confidence below which a pose is hard to read
MIN_SECONDS = 0.2  # problems shorter than this are keypoint jitter, not worth a warning
BLUR_SECONDS = 0.5  # unreadable stretches shorter than this barely affect a take
GAP_SECONDS = 0.5  # a problem that lets up for less than this counts as one
LISTED_SPANS = 3


def reframe(track: PoseTrack, frame: Crop, aspect: float) -> Crop:
    """Picks the window of the source video to render, around the dancer.

    The window holds everything the dancer reaches during the clip, with room
    for hands, feet, and head, at the output's aspect ratio. When that doesn't
    fit in the frame, it's the largest window that does, centered on the
    dancer. A dancer never found gets the centered window.

    Args:
        track: The dancer's poses in the source video.
        frame: The picture area of the source, without black bars.
        aspect: The output's width divided by its height.

    Returns:
        The window, inside frame, with even sizes and offsets.
    """
    extents = _extents(track)
    torso = _torso_length(track)
    if len(extents) == 0 or torso is None:
        return centered(frame, aspect)
    smoothed = np.median(
        sliding_window_view(
            np.pad(extents, ((GLITCH_FRAMES // 2,) * 2, (0, 0)), mode="edge"),
            GLITCH_FRAMES,
            axis=0,
        ),
        axis=-1,
    )
    pad = PAD_TORSOS * torso
    x1, y1 = smoothed[:, :2].min(axis=0) - pad
    x2, y2 = smoothed[:, 2:].max(axis=0) + pad
    return _fit((x1, y1, x2, y2), frame, aspect)


def centered(frame: Crop, aspect: float) -> Crop:
    """Returns the largest window at the aspect ratio centered in the frame.

    This is the part of the frame the workflow keeps when nothing is reframed.
    """
    width, height, x, y = frame
    return _fit((x, y, x + width, y + height), frame, aspect)


def check(track: PoseTrack, window: Crop) -> list[str]:
    """Finds problems with the dancer's footage in the rendered window.

    Args:
        track: The dancer's poses in the source video.
        window: The area of the source that gets rendered.

    Returns:
        Plain-language warnings, each with when it happens and how to fix it
        when filming. Empty if the footage looks fine.
    """
    found = (track.scores > SEEN).sum(axis=1) >= MIN_FOUND
    torso = _torso_length(track)
    if torso is None:
        return ["No dancer was found in the video; film the whole body, head to toe."]
    width, height, left, top = window
    inside = (
        (track.keypoints[..., 0] >= left)
        & (track.keypoints[..., 0] <= left + width)
        & (track.keypoints[..., 1] >= top)
        & (track.keypoints[..., 1] <= top + height)
    )
    warnings = []
    if spans := _spans(~found, track.fps, MIN_SECONDS):
        warnings.append(
            f"The dancer can't be seen at {spans}; keep them in the frame throughout."
        )
    for problem, joints, reach, fix in PARTS:
        points = track.keypoints[:, joints]
        margin = reach * torso
        cut = (
            (points[..., 0] - margin < left)
            | (points[..., 0] + margin > left + width)
            | (points[..., 1] - margin < top)
            | (points[..., 1] + margin > top + height)
        )
        if spans := _spans(found & cut.any(axis=1), track.fps, MIN_SECONDS):
            warnings.append(f"{problem} the frame at {spans}; {fix}.")
    if spans := _spans(_others_inside(track, window), track.fps, MIN_SECONDS):
        warnings.append(
            f"Someone else is in the frame at {spans}; film the dancer alone."
        )
    # Joints outside the window are reported above, so only the ones inside it
    # say whether the footage itself is hard to read.
    visible = found & (inside.sum(axis=1) >= MIN_FOUND)
    confidence = np.where(inside, track.scores, np.nan)
    blurry = np.zeros(len(found), dtype=bool)
    blurry[visible] = np.nanmedian(confidence[visible], axis=1) < LOW_CONFIDENCE
    if spans := _spans(blurry, track.fps, BLUR_SECONDS):
        warnings.append(
            f"The dancer's pose is hard to make out at {spans}, from motion blur or "
            "dim light; film in brighter light."
        )
    return warnings


def _extents(track: PoseTrack) -> np.ndarray:
    """Returns the (x1, y1, x2, y2) box of the found keypoints in each frame.

    Frames with too few keypoints found are left out.
    """
    boxes = []
    for points, scores in zip(track.keypoints, track.scores, strict=True):
        seen = points[scores > SEEN]
        if len(seen) >= MIN_FOUND:
            boxes.append([*seen.min(axis=0), *seen.max(axis=0)])
    return np.array(boxes).reshape(-1, 4)


def _torso_length(track: PoseTrack) -> float | None:
    """Returns the median shoulders-to-hips distance, or None if never seen."""
    seen = (track.scores[:, [5, 6, 11, 12]] > SEEN).all(axis=1)
    if not seen.any():
        return None
    points = track.keypoints[seen]
    shoulders = points[:, [5, 6]].mean(axis=1)
    hips = points[:, [11, 12]].mean(axis=1)
    return float(np.median(np.linalg.norm(shoulders - hips, axis=1)))


def _fit(box: tuple[float, float, float, float], frame: Crop, aspect: float) -> Crop:
    """Returns the smallest window at the aspect ratio holding box, inside frame.

    A box that can't fit gets the largest window that does, centered on it.
    """
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width < height * aspect:
        width = height * aspect
    else:
        height = width / aspect
    frame_width, frame_height, frame_x, frame_y = frame
    scale = min(1.0, frame_width / width, frame_height / height)
    # Rounded to the nearest even size, not down: scaled to fill the frame, a
    # side can come out a hair short, as 1919.9999, and must still fill it.
    width = min(2 * round(width * scale / 2), _even(frame_width))
    height = min(2 * round(height * scale / 2), _even(frame_height))
    x = np.clip((x1 + x2 - width) / 2, frame_x, frame_x + frame_width - width)
    y = np.clip((y1 + y2 - height) / 2, frame_y, frame_y + frame_height - height)
    return width, height, _even(x), _even(y)


def _even(value: float) -> int:
    """Rounds down to an even number, as yuv420 video needs for crops."""
    return 2 * int(value // 2)


def _others_inside(track: PoseTrack, window: Crop) -> np.ndarray:
    """Returns, per frame, whether another person stands inside the window."""
    width, height, left, top = window
    inside = np.zeros(len(track.keypoints), dtype=bool)
    for index, boxes in enumerate(track.others):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        inside[index] = bool(
            (
                (centers[:, 0] >= left)
                & (centers[:, 0] <= left + width)
                & (centers[:, 1] >= top)
                & (centers[:, 1] <= top + height)
            ).any()
        )
    return inside


def _spans(flags: np.ndarray, fps: float, min_seconds: float) -> str:
    """Describes when flags is set, as times like "2.7-3.1s and 5.0-5.4s".

    Runs less than GAP_SECONDS apart are joined, then runs shorter than
    min_seconds are ignored. Returns "" if none remain.
    """
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    runs: list[list[int]] = []
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if runs and (start - runs[-1][1]) / fps < GAP_SECONDS:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    spans = [
        f"{start / fps:.1f}-{end / fps:.1f}s"
        for start, end in runs
        if (end - start) / fps >= min_seconds
    ]
    if len(spans) > LISTED_SPANS:
        more = len(spans) - LISTED_SPANS
        spans = [*spans[:LISTED_SPANS], f"{more} more time{'s' if more > 1 else ''}"]
    if len(spans) <= 2:
        return " and ".join(spans)
    return ", ".join(spans[:-1]) + f", and {spans[-1]}"
