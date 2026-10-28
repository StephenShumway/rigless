"""Choosing the person to replace: taps on a frame, then tracking through the clip.

The user pauses the footage on a frame where the person is clear and taps
them; SAM 3.1 turns the taps into a mask on that frame. The taps on one frame
make an anchor, which can also hold "not them" taps (the page sends only the
user's one tap). Tracking follows the person from each anchor to the next, and
back from the first anchor to the start of the clip, so where tracking slips
the user adds an anchor at that moment.

Typical usage example:

  anchor = Anchor(frame=36, positive=((410, 520),))
  mask = detect(footage, anchor, work_dir)
  masks = track(footage, [anchor], work_dir)
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from comfy_sdk import Asset
from PIL import Image

from rigless import comfy_client, video_prep
from rigless.comfy_client import Graph
from rigless.pose import HIPS, SEEN, SHOULDERS, SURE, Person, PoseTrack

SAM_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
# Each person's tint, in the order they're added: the UI's --mask-1 to --mask-3,
# which app.js's MASK_COLORS matches.
TINTS = ((144, 226, 221), (101, 149, 90), (150, 167, 231))
TINT_ALPHA = 0.45
NOSE = 0  # COCO keypoint
# The mask holds a head if it has at least PART_AREA shoulder widths squared of
# pixels above the shoulder line, raised by NECK shoulder widths past the collar;
# on Shoot 1, masks with the head scored 0.23 and up, shirt-only masks 0.03. It
# holds a body if it has as much below CHEST shoulder widths under the shoulders.
NECK = 0.15
CHEST = 0.5
PART_AREA = 0.1
MISSED_FRAMES = 6  # a quarter second; shorter misses are left alone

Point = tuple[int, int]


@dataclass(frozen=True)
class Anchor:
    """Taps on one frame that mark the person to replace.

    Attributes:
        frame: The frame's index in the footage, from 0.
        positive: Points on the person, in the footage's pixels.
        negative: Points on anything the mask should leave out.
    """

    frame: int
    positive: tuple[Point, ...]
    negative: tuple[Point, ...] = ()


@dataclass(frozen=True)
class Segment:
    """A run of frames tracked from one anchor.

    Attributes:
        anchor: The anchor the run starts from.
        start: The run's first frame in the footage. A backward run starts at
            frame 0 and ends at its anchor.
        length: How many frames the run covers, its anchor's frame included.
        backward: True if the run goes from its anchor back to the clip's start.
    """

    anchor: Anchor
    start: int
    length: int
    backward: bool


def plan_segments(anchors: Sequence[Anchor], frames: int) -> list[Segment]:
    """Splits a clip into runs, each tracked from one anchor.

    Each anchor is tracked forward to the next anchor or the clip's end, and
    the first anchor also back to the clip's start.

    Args:
        anchors: The anchors, in any order.
        frames: The clip's length in frames.

    Raises:
        ValueError: There are no anchors, two share a frame, or one lies
            outside the clip.
    """
    if not anchors:
        raise ValueError("Tap the person to replace first.")
    ordered = sorted(anchors, key=lambda anchor: anchor.frame)
    marked = [anchor.frame for anchor in ordered]
    if len(set(marked)) != len(marked):
        raise ValueError("Each frame can hold only one set of taps.")
    if marked[0] < 0 or marked[-1] >= frames:
        raise ValueError("A tap lies outside the clip.")
    segments = []
    if marked[0] > 0:
        segments.append(Segment(ordered[0], 0, marked[0] + 1, backward=True))
    for anchor, end in zip(ordered, [*marked[1:], frames], strict=True):
        segments.append(Segment(anchor, anchor.frame, end - anchor.frame, False))
    return segments


def assemble(
    segments: Sequence[Segment], runs: Sequence[np.ndarray], frames: int
) -> np.ndarray:
    """Puts tracked runs together into one mask per frame of the clip.

    Args:
        segments: The plan, from plan_segments.
        runs: Each segment's masks as (frames, height, width) bools, in the
            order they were tracked: a backward run starts at its anchor.
        frames: The clip's length in frames.

    Returns:
        The masks as (frames, height, width) bools.

    Raises:
        ValueError: A run is shorter than its segment.
    """
    height, width = runs[0].shape[1:]
    masks = np.zeros((frames, height, width), bool)
    for segment, run in zip(segments, runs, strict=True):
        if len(run) < segment.length:
            raise ValueError(
                f"Tracking returned {len(run)} frames where {segment.length} were sent."
            )
        run = run[: segment.length]
        if segment.backward:
            # The forward run from the same anchor covers the anchor's own frame.
            masks[segment.start : segment.start + segment.length - 1] = run[:0:-1]
        else:
            masks[segment.start : segment.start + segment.length] = run
    return masks


def lost_spans(masks: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """Returns the stretches where the mask is empty, as (start, end) seconds."""
    empty = ~masks.reshape(len(masks), -1).any(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], empty.astype(int), [0]])))
    return [
        (start / fps, end / fps)
        for start, end in zip(edges[::2], edges[1::2], strict=True)
    ]


def head_taps(track: PoseTrack, masks: np.ndarray) -> list[Anchor]:
    """Returns taps on the person wherever the mask has lost their head.

    After a camera cut, tracking can hold on to part of the person, such as a
    shirt, and drop the head, and H3 then can't redraw their hair. Each stretch
    of at least MISSED_FRAMES frames where the person's face and shoulders are
    seen but the mask stops at the shoulders gets taps on the nose and both
    shoulders at its first frame, to track again from there. Seen from behind,
    the pose can put the face on someone else, so the shoulders must be sure.
    The nose rather than the head's keypoints together: a "seen" ear can sit
    far down the neck. Keypoints rather than the mask: its middle can be a tie,
    and SAM then follows the tie.

    Args:
        track: The masked person's poses, from pose.estimate_in with
            pose.mask_boxes(masks).
        masks: (frames, height, width) bools, from track.
    """
    missed = np.array(
        [
            _headless(points, scores, mask)
            for points, scores, mask in zip(
                track.keypoints, track.scores, masks, strict=True
            )
        ]
    )
    edges = np.flatnonzero(np.diff(np.concatenate([[0], missed.astype(int), [0]])))
    taps = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        if end - start < MISSED_FRAMES:
            continue
        points = track.keypoints[start, [NOSE, *SHOULDERS]]
        taps.append(Anchor(int(start), _rounded(points)))
    return taps


def body_taps(people: Sequence[Person], mask: np.ndarray) -> tuple[Point, ...]:
    """Returns taps that add the body when a mask holds only a person's head.

    A tap on someone's face can make SAM mask just the head, and H3 then draws
    the character's face on the performer's body. The person is the one whose
    detected box holds most of the mask; with no such box there are no taps, as
    they'd land on someone else. With their face, shoulders and hips
    seen, and nothing masked below the chest, the taps are their nose,
    shoulders, chest and hips; otherwise there are none. Unseen hips mean the
    body is out of frame or hidden behind someone, whom a chest tap would pull in.

    Args:
        people: Everyone found on the frame, from pose.find_people.
        mask: The frame's mask, (height, width) bools.
    """
    if not people or not mask.any():
        return ()
    person = max(people, key=lambda person: _inside_box(mask, person.box))
    if 2 * _inside_box(mask, person.box) < mask.sum():
        return ()
    points, scores = person.keypoints, person.scores
    if scores[NOSE] < SEEN or (scores[[*SHOULDERS, *HIPS]] < SURE).any():
        return ()
    (x1, y1), (x2, y2) = points[SHOULDERS]
    chest = round(max(y1, y2) + CHEST * abs(x2 - x1))
    if mask[chest:].sum() >= PART_AREA * (x2 - x1) ** 2:
        return ()
    middle = np.array([points[SHOULDERS, 0].mean(), chest])
    return _rounded([points[NOSE], *points[SHOULDERS], middle, *points[HIPS]])


def merge(anchors: Sequence[Anchor], extra: Sequence[Anchor]) -> list[Anchor]:
    """Adds taps to anchors, joining those on the same frame into one anchor."""
    by_frame = {anchor.frame: anchor for anchor in anchors}
    for tap in extra:
        mine = by_frame.get(tap.frame)
        by_frame[tap.frame] = (
            Anchor(
                tap.frame, mine.positive + tap.positive, mine.negative + tap.negative
            )
            if mine
            else tap
        )
    return list(by_frame.values())


def _headless(points: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> bool:
    """Returns whether the face and shoulders show but the mask stops at them."""
    if scores[NOSE] < SEEN or (scores[SHOULDERS] < SURE).any():
        return False
    (x1, y1), (x2, y2) = points[SHOULDERS]
    width = abs(x2 - x1)
    neck = round(min(y1, y2) - NECK * width)
    return bool(mask[: max(neck, 0)].sum() < PART_AREA * width**2)


def _inside_box(mask: np.ndarray, box: np.ndarray) -> int:
    """Returns how many of the mask's pixels lie inside an (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = (max(0, round(v)) for v in box)
    return int(mask[y1:y2, x1:x2].sum())


def _rounded(points: Iterable[np.ndarray]) -> tuple[Point, ...]:
    return tuple((round(x), round(y)) for x, y in points)


def overlay(mask: np.ndarray, color: tuple[int, int, int] = TINTS[0]) -> Image.Image:
    """Returns the mask as a transparent image, tinted where the person is."""
    pixels = np.zeros((*mask.shape, 4), np.uint8)
    pixels[mask] = (*color, round(255 * TINT_ALPHA))
    return Image.fromarray(pixels, "RGBA")


def tint(frames: np.ndarray, people: Sequence[np.ndarray | None]) -> np.ndarray:
    """Returns RGB frames with each person's masks in their tint, for a preview.

    Args:
        frames: (frames, height, width, 3) RGB pixels.
        people: Each person's (frames, height, width) bool masks, in TINTS
            order; None for someone who isn't tinted.
    """
    # Frame by frame and only where someone is: a whole clip in floats runs to GBs.
    out = frames.copy()
    for masks, color in zip(people, TINTS, strict=False):
        if masks is None:
            continue
        shade = np.array(color, np.float32) * TINT_ALPHA + 0.5  # + 0.5 rounds
        for frame, mask in zip(out, masks, strict=True):
            frame[mask] = (frame[mask] * (1 - TINT_ALPHA) + shade).astype(np.uint8)
    return out


def detect_graph(frame: str | Asset, anchor: Anchor) -> Graph:
    """Returns a workflow that finds the anchor's person on one frame image."""
    return {
        "sam": _node("CheckpointLoaderSimple", ckpt_name=SAM_CHECKPOINT),
        "frame": _node("LoadImage", image=frame),
        **_detect("detect", ["frame", 0], anchor),
        "mask_image": _node("MaskToImage", mask=["detect", 0]),
        "save": _node(
            "SaveImage", images=["mask_image", 0], filename_prefix="rigless/select"
        ),
    }


def track_graph(
    footage: str | Asset,
    backward: str | Asset | None,
    segments: Sequence[Segment],
    frames: int,
    fps: float,
) -> Graph:
    """Returns a workflow that tracks every segment and saves each as a mask video.

    Args:
        footage: The clip.
        backward: The clip reversed, needed when a segment runs backward.
        segments: The plan, from plan_segments.
        frames: The clip's length in frames.
        fps: The clip's frame rate.

    Returns:
        The workflow; segment i's masks come from the node "save_<i>".
    """
    graph: Graph = {"sam": _node("CheckpointLoaderSimple", ckpt_name=SAM_CHECKPOINT)}
    for name, video in (("forward", footage), ("backward", backward)):
        if video is not None:
            graph[f"{name}_file"] = _node("LoadVideo", file=video)
            graph[name] = _node("GetVideoComponents", video=[f"{name}_file", 0])
    for index, segment in enumerate(segments):
        # In the reversed clip, the anchor's frame f sits at frames - 1 - f.
        source, start = (
            ("backward", frames - 1 - segment.anchor.frame)
            if segment.backward
            else ("forward", segment.start)
        )
        part, first = f"part_{index}", f"first_{index}"
        graph[part] = _node(
            "ImageFromBatch",
            image=[source, 0],
            batch_index=start,
            length=segment.length,
        )
        graph[first] = _node("ImageFromBatch", image=[part, 0], batch_index=0, length=1)
        graph.update(_detect(f"detect_{index}", [first, 0], segment.anchor))
        graph[f"track_{index}"] = _node(
            "SAM3_VideoTrack",
            images=[part, 0],
            model=["sam", 0],
            initial_mask=[f"detect_{index}", 0],
            detection_threshold=0.5,
            max_objects=1,
            detect_interval=1,
        )
        graph[f"masks_{index}"] = _node(
            "SAM3_TrackToMask", track_data=[f"track_{index}", 0], object_indices=""
        )
        graph[f"mask_images_{index}"] = _node("MaskToImage", mask=[f"masks_{index}", 0])
        graph[f"video_{index}"] = _node(
            "CreateVideo", images=[f"mask_images_{index}", 0], fps=float(fps)
        )
        graph[f"save_{index}"] = _node(
            "SaveVideo",
            video=[f"video_{index}", 0],
            filename_prefix="rigless/track",
            format="auto",
            codec="auto",
            **{"format.codec": "auto"},
        )
    return graph


def detect(footage: Path, anchor: Anchor, work_dir: Path) -> np.ndarray:
    """Finds the anchor's person on its frame of the footage.

    Args:
        footage: The footage the anchor's frame number counts in.
        anchor: The taps.
        work_dir: Where the frame and the found mask are saved.

    Returns:
        The mask as (height, width) bools.

    Raises:
        ValueError: RunPod is chosen and there is no pod.
        comfy_sdk.ComfyError: ComfyUI couldn't run the workflow.
    """
    frame = work_dir / f"frame-{anchor.frame}.png"
    video_prep.extract_frame(footage, anchor.frame, frame)
    found = work_dir / f"found-{anchor.frame}.png"
    with comfy_client.connect() as client:
        job = comfy_client.run(
            client, detect_graph(client.assets.from_file(frame), anchor)
        )
        comfy_client.download(job, {"save": found})
    with Image.open(found) as image:
        return np.asarray(image.convert("L")) > 127


def track(footage: Path, anchors: Sequence[Anchor], work_dir: Path) -> np.ndarray:
    """Tracks the person through the whole footage from the anchors.

    Args:
        footage: The footage to track through.
        anchors: The person's taps, on the footage's frames.
        work_dir: Where the reversed footage and each run's masks are saved.

    Returns:
        The masks as (frames, height, width) bools.

    Raises:
        ValueError: The anchors don't fit the footage (see plan_segments), or
            RunPod is chosen and there is no pod.
        comfy_sdk.ComfyError: ComfyUI couldn't run the workflow.
    """
    info = video_prep.probe(footage)
    segments = plan_segments(anchors, info.frames)
    reversed_clip = None
    if any(segment.backward for segment in segments):
        reversed_clip = work_dir / "reversed.mp4"
        video_prep.reverse(footage, reversed_clip)
    outputs = {
        f"save_{index}": work_dir / f"run-{index}.mp4" for index in range(len(segments))
    }
    with comfy_client.connect() as client:
        graph = track_graph(
            client.assets.from_file(footage),
            client.assets.from_file(reversed_clip) if reversed_clip else None,
            segments,
            info.frames,
            info.fps,
        )
        comfy_client.download(comfy_client.run(client, graph), outputs)
    runs = [video_prep.read_frames(path)[..., 0] > 127 for path in outputs.values()]
    return assemble(segments, runs, info.frames)


def _detect(name: str, image: list[str | int], anchor: Anchor) -> Graph:
    """Returns the SAM 3.1 node that turns an anchor's taps into a mask on image."""
    inputs = {
        "model": ["sam", 0],
        "image": image,
        "positive_coords": _points(anchor.positive),
        "threshold": 0.5,
        "refine_iterations": 2,
        "individual_masks": False,
    }
    if anchor.negative:
        inputs["negative_coords"] = _points(anchor.negative)
    return {name: _node("SAM3_Detect", **inputs)}


def _points(points: Sequence[Point]) -> str:
    """Points as the JSON SAM3_Detect takes: [{"x": 1, "y": 2}, ...]."""
    return json.dumps([{"x": x, "y": y} for x, y in points])


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}
