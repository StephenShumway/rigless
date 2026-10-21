"""Driving-video checks and preparation before upload to ComfyUI.

Uses the ffmpeg build bundled with imageio-ffmpeg, so no system install is
needed.

Typical usage example:

  info = probe(Path("dance.mp4"))
  crop = detect_crop(Path("dance.mp4"))
  prepare(Path("dance.mp4"), run_dir / "driving.mp4", width=482, height=854, crop=crop)
"""

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg

Crop = tuple[int, int, int, int]
"""A picture area as (width, height, x, y) in source pixels."""

_CROP_LINE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


@dataclass(frozen=True)
class VideoInfo:
    """Basic properties of a video file.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        frames: Number of frames.
    """

    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration(self) -> float:
        """The length in seconds."""
        return self.frames / self.fps


def probe(video: Path) -> VideoInfo:
    """Reads a video's size, frame rate, and frame count.

    Raises:
        ValueError: The file is missing or isn't a readable video.
    """
    capture = cv2.VideoCapture(str(video))
    try:
        info = VideoInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=capture.get(cv2.CAP_PROP_FPS),
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()
    if info.frames <= 0 or info.fps <= 0:
        raise ValueError(f"{video} is missing or isn't a readable video.")
    return info


def length_warning(seconds: float) -> str | None:
    """Returns a warning if a clip is too short or too long to render well."""
    if seconds < 3:
        return f"The clip is {seconds:.1f}s; under 3s gives little motion to follow."
    if seconds > 15:
        return f"The clip is {seconds:.1f}s; over 15s renders take very long. Trim it."
    return None


def detect_crop(video: Path) -> Crop | None:
    """Finds the picture area inside black bars, such as a pillarboxed phone clip.

    Returns:
        The picture area, or None if the video has no black bars.

    Raises:
        RuntimeError: ffmpeg failed, e.g., because video is not a video.
    """
    # cropdetect widens its box over the whole clip, so the last line it logs
    # covers every frame and a dark moment can't crop away real picture.
    log = _ffmpeg(
        "-v", "info", "-i", str(video), "-an", "-vf", "cropdetect=limit=24:round=2",
        "-f", "null", "-",
    )  # fmt: skip
    found = _CROP_LINE.findall(log)
    if not found:
        return None
    width, height, x, y = (int(value) for value in found[-1])
    info = probe(video)
    if (width, height) == (info.width, info.height):
        return None
    return width, height, x, y


def prepare(
    src: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    seconds: float | None = None,
    crop: Crop | None = None,
) -> None:
    """Writes a copy of a driving video sized for the workflow, with its audio.

    ComfyUI decodes every frame of the driving video to float32, several times
    per run, so a full-resolution phone clip can exhaust system RAM. The copy
    is cropped to the picture area, then scaled down to just cover
    width x height, which the workflow crops to anyway, so no detail the model
    would see is lost. The frame rate is kept.

    Args:
        src: The source video.
        dest: Where to write the prepared .mp4.
        width: The workflow's output width.
        height: The workflow's output height.
        seconds: Keep only this many seconds from the start. None keeps it all.
        crop: The picture area to keep, from detect_crop. None keeps the frame.

    Raises:
        FileNotFoundError: src does not exist.
        RuntimeError: ffmpeg failed, e.g., because src is not a video.
    """
    if not src.is_file():
        raise FileNotFoundError(src)
    filters = [f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}"] if crop else []
    filters.append(
        f"scale={width}:{height}:force_original_aspect_ratio=increase"
        ":force_divisible_by=2:flags=area"
    )
    trim = ["-t", str(seconds)] if seconds is not None else []
    _ffmpeg(
        "-v", "error", "-y", "-i", str(src), *trim, "-vf", ",".join(filters),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(dest),
    )  # fmt: skip


def finish(src: Path, dest: Path) -> None:
    """Writes a copy of a rendered take that looks more like phone footage.

    Fast-moving parts get motion blur along their own motion, like a camera
    shutter (frames are interpolated by optical flow to 4x the rate, averaged in
    threes, and brought back), and the picture is graded a little brighter and
    less saturated. Measured on Oct 18, this brought motion blur, sharpness,
    exposure, and saturation close to real phone footage; a plain blend of
    neighboring frames ghosted fast hands instead. Frame rate, frame count, and
    audio are kept. Score the original, not the copy: the blur smooths keypoint
    jitter and inflates the motion score.

    Raises:
        ValueError: src is missing or isn't a readable video.
        RuntimeError: ffmpeg failed.
    """
    fps = probe(src).fps
    filters = (
        f"minterpolate=fps={4 * fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
        f"tmix=frames=3:weights='1 1 1',fps={fps},"
        "eq=brightness=0.035:saturation=0.92:gamma=1.03"
    )
    _ffmpeg(
        "-v", "error", "-y", "-i", str(src), "-map", "0:v", "-map", "0:a?",
        "-vf", filters, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "copy", str(dest),
    )  # fmt: skip


def join(parts: Sequence[Path], dest: Path, *, frames: int, audio: Path) -> None:
    """Writes chunks of a take back to back as one video, with a soundtrack.

    Each part after the first starts with a repeat of the previous part's last
    frame, which is dropped, and the result is cut to frames frames. It takes
    the first part's frame rate.

    Args:
        parts: The chunk videos, in order.
        dest: Where to write the joined .mp4.
        frames: The take's length in frames.
        audio: A video whose soundtrack, if any, the joined video gets.

    Raises:
        ValueError: The first part is missing or isn't a readable video.
        RuntimeError: ffmpeg failed.
    """
    fps = probe(parts[0]).fps
    inputs = [arg for part in parts for arg in ("-i", str(part))]
    trims = "".join(
        f"[{index}:v]trim=start_frame={min(index, 1)},setpts=PTS-STARTPTS[v{index}];"
        for index in range(len(parts))
    )
    joined = "".join(f"[v{index}]" for index in range(len(parts)))
    filters = f"{trims}{joined}concat=n={len(parts)}:v=1:a=0,trim=end_frame={frames}[v]"
    _ffmpeg(
        "-v", "error", "-y", *inputs, "-i", str(audio), "-filter_complex", filters,
        "-map", "[v]", "-map", f"{len(parts)}:a?", "-r", str(fps),
        "-t", str(frames / fps), "-c:v", "libx264", "-crf", "14",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest),
    )  # fmt: skip


def _ffmpeg(*args: str) -> str:
    """Runs the bundled ffmpeg with args and returns its log.

    Raises:
        RuntimeError: ffmpeg exited with an error.
    """
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostats", *args]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
    return result.stderr
