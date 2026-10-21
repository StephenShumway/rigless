"""Body pose estimation on video frames, on the CPU.

Runs a YOLOX person detector and RTMPose through rtmlib and ONNX Runtime, so it
never competes with ComfyUI for the GPU. Keypoints use the 17-point COCO
layout. The models download once, on first use, to rtmlib's cache folder.

Typical usage example:

  track = estimate(Path("result.mp4"))
  track.keypoints.shape  # (frames, 17, 2)
"""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import Body

NUM_KEYPOINTS = 17
CPU_THREADS = 4
DETECT_EVERY = 10  # frames; in between, the person is followed by their keypoints
SEEN = 0.3  # keypoint confidence above which a keypoint counts as found
# Other people count when at least this tall relative to the main person; smaller
# ones, like people seated far behind the dancer, don't get in the way.
OTHERS_HEIGHT = 0.5


@dataclass(frozen=True)
class PoseTrack:
    """Body keypoints of one person in every frame of a video.

    Attributes:
        keypoints: (frames, 17, 2) COCO keypoint positions in pixels.
        scores: (frames, 17) keypoint confidences, all 0 in frames where
            nobody was found.
        fps: The video's frame rate.
        others: Per frame, a (k, 4) array of (x1, y1, x2, y2) boxes of other
            people at least half the main person's height, as of the
            detector's latest run. Empty if other people weren't tracked.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    fps: float
    others: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class Person:
    """One person found in a still image.

    Attributes:
        box: The detector's (x1, y1, x2, y2) box in pixels; it can be loose.
        keypoints: (17, 2) COCO keypoint positions in pixels.
        scores: (17,) keypoint confidences.
    """

    box: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray


def estimate(video: Path, seconds: float | None = None) -> PoseTrack:
    """Finds the main person's body keypoints in every frame of a video.

    The main person is the largest one detected: the dancer, in a typical
    phone clip with people in the background. Other people are tracked too,
    by their boxes only.

    Args:
        video: The video to read.
        seconds: Stop after this many seconds. None reads the whole video.

    Raises:
        ValueError: The file is missing or isn't a readable video.
    """
    model = _model()
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = round(seconds * fps) if seconds is not None else None
    keypoints: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    others: list[np.ndarray] = []
    box, nearby = None, np.empty((0, 4))
    try:
        while frames is None or len(keypoints) < frames:
            ok, frame = capture.read()
            if not ok:
                break
            if box is None or len(keypoints) % DETECT_EVERY == 0:
                box, nearby = _people(model, frame)
            others.append(nearby)
            if box is None:
                keypoints.append(np.zeros((NUM_KEYPOINTS, 2)))
                scores.append(np.zeros(NUM_KEYPOINTS))
                continue
            found, confidence = model.pose_model(frame, bboxes=[box])
            keypoints.append(found[0])
            scores.append(confidence[0])
            box = _box_around(found[0], confidence[0])
    finally:
        capture.release()
    if not keypoints or fps <= 0:
        raise ValueError(f"{video} is missing or isn't a readable video.")
    return PoseTrack(np.stack(keypoints), np.stack(scores), fps, others)


def face_keypoints(image: np.ndarray) -> np.ndarray:
    """Finds the main person's face keypoints in a still image.

    Args:
        image: An HxWx3 RGB image.

    Returns:
        A (5, 2) array of positions in pixels: nose, left eye, right eye, left
        ear, right ear (COCO order; "left" is the person's own left).

    Raises:
        ValueError: The image shows no person.
    """
    model = _model()
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    box, _ = _people(model, frame)
    if box is None:
        raise ValueError("No person found in the image.")
    found, _ = model.pose_model(frame, bboxes=[box])
    return found[0][:5]


def find_people(image: np.ndarray) -> list[Person]:
    """Finds every person in a still image, with their body keypoints.

    Args:
        image: An HxWx3 RGB image.
    """
    model = _model()
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    boxes = model.det_model(frame)
    if len(boxes) == 0:
        return []
    keypoints, scores = model.pose_model(frame, bboxes=list(boxes))
    return [Person(*found) for found in zip(boxes, keypoints, scores, strict=True)]


@cache
def _model() -> Body:
    """Loads the detector and pose model once per process."""
    body = Body(mode="balanced", backend="onnxruntime", device="cpu")
    # rtmlib's default sessions each spin threads on every core, and called
    # back to back they starve each other: about 3x slower than these.
    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    for part in (body.det_model, body.pose_model):
        part.session = ort.InferenceSession(
            part.onnx_model, sess_options=options, providers=["CPUExecutionProvider"]
        )
    return body


def _people(model: Body, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Detects the people in a frame.

    Returns:
        A tuple (main, others), where main is the (x1, y1, x2, y2) box of the
        largest person, or None if nobody was found, and others is a (k, 4)
        array of the boxes of other people at least OTHERS_HEIGHT as tall.
    """
    boxes = model.det_model(frame)
    if len(boxes) == 0:
        return None, np.empty((0, 4))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    main = np.argmax(areas)
    heights = boxes[:, 3] - boxes[:, 1]
    tall = heights >= OTHERS_HEIGHT * heights[main]
    tall[main] = False
    return boxes[main], boxes[tall]


def _box_around(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
    """Returns a box around the found keypoints, or None if too few were found.

    The box is padded by a quarter on each side, because keypoints stop short of
    the top of the head and the feet, and the person moves before the next frame.
    """
    found = keypoints[scores > SEEN]
    if len(found) < 4:
        return None
    (x1, y1), (x2, y2) = found.min(axis=0), found.max(axis=0)
    pad_x, pad_y = (x2 - x1) / 4, (y2 - y1) / 4
    return np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y])
