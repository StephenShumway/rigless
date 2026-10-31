"""Footage handling: probing, trimming, frame cuts, and writing videos.

Uses the ffmpeg build bundled with imageio-ffmpeg, so no system install is
needed.

Typical usage example:

  info = probe(Path("clip.mp4"))
  start, end = check_trim(2.0, 10.0, info.duration)
  conform(Path("clip.mp4"), Path("footage.mp4"), start=start, end=end, fps=24,
          max_side=1280)
"""

import collections
import contextlib
import itertools
import math
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

MIN_TRIM_SECONDS = 1.0
MAX_TRIM_SECONDS = 20.0  # about 17 min per take on an H100 at H3's 768p


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


def check_trim(start: float, end: float | None, duration: float) -> tuple[float, float]:
    """Returns the part of a clip to use, as (start, end) in seconds.

    Args:
        start: Where the part starts.
        end: Where it ends. None, or a time past the clip's end, means its end.
        duration: The clip's length.

    Raises:
        ValueError: The part starts outside the clip, or is too short or too long.
    """
    end = duration if end is None else min(end, duration)
    if not math.isfinite(end):
        raise ValueError("The trim's end isn't a time.")
    if not 0 <= start < duration:
        raise ValueError("The trim starts outside the clip.")
    length = end - start
    if length < MIN_TRIM_SECONDS:
        raise ValueError(f"Keep at least {MIN_TRIM_SECONDS:g} second of the clip.")
    if length > MAX_TRIM_SECONDS + 1e-6:
        raise ValueError(
            f"Trim the clip to {MAX_TRIM_SECONDS:g} seconds or less. "
            f"This part is {length:.1f}."
        )
    return start, end


def conform(
    src: Path, dest: Path, *, start: float, end: float, fps: float, max_side: int
) -> None:
    """Writes the part of a clip from start to end at a steady frame rate.

    The copy keeps the clip's shape and audio. It is scaled down so that its
    longer side is at most max_side (smaller clips keep their size), and has
    round((end - start) * fps) frames when the clip is long enough.

    Args:
        src: The clip.
        dest: Where the copy goes.
        start: Where the part starts in the clip, in seconds.
        end: Where it ends, in seconds.
        fps: The copy's frame rate.
        max_side: The longest the copy's longer side may be, in pixels.

    Raises:
        FileNotFoundError: src does not exist.
        RuntimeError: ffmpeg failed, e.g., because src is not a video.
    """
    if not src.is_file():
        raise FileNotFoundError(src)
    # Sized after ffmpeg applies the clip's rotation, so phone clips keep their shape.
    even = f"trunc(min({max_side},{{side}})/2)*2"
    scale = (
        f"scale=w='if(gte(iw,ih),{even.format(side='iw')},-2)'"
        f":h='if(gte(iw,ih),-2,{even.format(side='ih')})':flags=area"
    )
    # Each frame fills the slot it falls in, not the nearest one: a screen
    # recording shows frames late, and to the nearest slot a late frame and the
    # next one can land together, one dropped and a gap filled by a repeat.
    _ffmpeg(
        "-v", "error", "-y", "-ss", str(start), "-i", str(src),
        "-vf", f"fps={fps}:round=down,{scale}",
        "-frames:v", str(round((end - start) * fps)),
        "-t", str(end - start), "-c:v", "libx264", "-crf", "16",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest),
    )  # fmt: skip


def cut_frames(src: Path, dest: Path, start: int, stop: int) -> None:
    """Writes frames start to stop - 1 of a video, without its audio.

    Raises:
        RuntimeError: ffmpeg failed, e.g., because src is not a video.
    """
    # Without the rate stated, the last frame gets no length, and the cut's rate
    # reads as frames / (frames - 1) times the source's.
    _ffmpeg(
        "-v", "error", "-y", "-i", str(src), "-an",
        "-vf", f"select='between(n,{start},{stop - 1})',setpts=N/FRAME_RATE/TB",
        "-r", str(probe(src).fps),
        "-frames:v", str(stop - start), "-c:v", "libx264", "-crf", "12",
        "-pix_fmt", "yuv420p", str(dest),
    )  # fmt: skip


def add_audio(
    video: Path,
    source: Path,
    start: float,
    dest: Path,
    comment: str = "",
    delay: float = 0.0,
) -> None:
    """Writes video with source's sound from start seconds on.

    The copy keeps every frame of video; a source without sound gives a silent
    copy.

    Args:
        video: The video to copy.
        source: The file whose sound is added.
        start: Where in source the sound starts, in seconds.
        dest: Where the copy goes.
        comment: Text for the copy's comment metadata, if any.
        delay: Seconds of silence in the copy before the sound comes in.

    Raises:
        RuntimeError: ffmpeg failed, e.g., because a file is not a video.
    """
    delayed = ["-af", f"adelay={round(delay * 1000)}:all=1"] if delay else []
    _ffmpeg(
        "-v", "error", "-y", "-i", str(video), "-ss", str(start), "-i", str(source),
        "-map", "0:v", "-map", "1:a?", *delayed, "-t", str(probe(video).duration),
        "-c:v", "copy", "-c:a", "aac", "-metadata", f"comment={comment}", str(dest),
    )  # fmt: skip


def extract_frame(video: Path, index: int, dest: Path) -> None:
    """Saves one frame of a video, counted from 0, as an image.

    Raises:
        RuntimeError: ffmpeg failed, or the video has no such frame.
    """
    dest.unlink(missing_ok=True)
    _ffmpeg(
        "-v", "error", "-y", "-i", str(video), "-vf", f"select='eq(n,{index})'",
        "-frames:v", "1", "-fps_mode", "passthrough", str(dest),
    )  # fmt: skip
    if not dest.is_file():
        raise RuntimeError(f"{video} has no frame {index}.")


def reverse(src: Path, dest: Path) -> None:
    """Writes a video's frames in reverse order, without its audio.

    Raises:
        RuntimeError: ffmpeg failed, e.g., because src is not a video.
    """
    _ffmpeg(
        "-v", "error", "-y", "-i", str(src), "-an", "-vf", "reverse",
        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(dest),
    )  # fmt: skip


def padded_order(length: int, head: int, tail: int) -> np.ndarray:
    """Returns the frame indices pad plays for a clip of length frames."""
    return np.r_[
        np.arange(head, 0, -1),
        np.arange(length),
        np.arange(length - 2, length - 2 - tail, -1),
    ]


def pad(
    clip: Path,
    head: int,
    tail: int,
    fps: float,
    dest: Path,
    lead_in: np.ndarray | None = None,
) -> None:
    """Writes a clip between its own ends played backward, without its audio.

    The result starts on frame `head`, plays back to frame 1, plays the whole
    clip, then plays back from the second-to-last frame for `tail` frames, so
    the motion never jumps: the order of padded_order.

    Args:
        clip: The video.
        head: How many frames lead in; at most the clip's length minus one.
        tail: How many frames lead out; at most the clip's length minus one.
        fps: The clip's frame rate.
        dest: Where to write the .mp4.
        lead_in: An HxWx3 BGR frame the clip's size, shown for the `head` frames
            in place of the clip's opening played backward.

    Raises:
        RuntimeError: ffmpeg failed.
    """
    info = probe(clip)

    def played() -> Iterator[np.ndarray]:
        source = iter_frames(clip)
        opening = list(itertools.islice(source, head + 1))
        if lead_in is None:
            yield from opening[head:0:-1]
        else:
            yield from itertools.repeat(lead_in, head)
        ending = collections.deque(opening, maxlen=tail + 1)
        yield from opening
        for frame in source:
            ending.append(frame)
            yield frame
        yield from list(ending)[-2::-1][:tail]

    write_frames(played(), fps, (info.width, info.height), dest)


def write_video(frames: np.ndarray, fps: float, dest: Path) -> None:
    """Encodes frames as a video.

    Args:
        frames: (frames, height, width, 3) RGB or (frames, height, width) gray
            uint8 pixels, with even sides.
        fps: The frame rate.
        dest: Where to write the .mp4.

    Raises:
        RuntimeError: ffmpeg failed.
    """
    count, height, width = frames.shape[:3]
    pixels = "gray" if frames.ndim == 3 else "rgb24"
    with _replacing(dest) as partial:
        _ffmpeg(
            "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pixels,
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-frames:v", str(count), "-c:v", "libx264", "-crf", "12",
            "-pix_fmt", "yuv420p", str(partial),
            stdin=np.ascontiguousarray(frames, np.uint8).tobytes(),
        )  # fmt: skip


def read_frames(video: Path) -> np.ndarray:
    """Decodes a whole video as (frames, height, width, 3) BGR uint8 pixels.

    Raises:
        ValueError: The video has no frames, or can't be read.
    """
    frames = list(iter_frames(video))
    if not frames:
        raise ValueError(f"{video.name} has no frames.")
    return np.stack(frames)


def iter_frames(video: Path) -> Iterator[np.ndarray]:
    """Decodes a video one frame at a time, as (height, width, 3) BGR uint8."""
    capture = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def write_frames(
    frames: Iterable[np.ndarray], fps: float, size: tuple[int, int], dest: Path
) -> None:
    """Encodes BGR frames one at a time, so a long full-size clip fits in memory.

    Args:
        frames: (height, width, 3) BGR uint8 pixels, with even sides.
        fps: The frame rate.
        size: The frames' (width, height).
        dest: Where to write the .mp4.

    Raises:
        RuntimeError: ffmpeg failed.
    """
    width, height = size
    with _replacing(dest) as partial:
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16",
            "-pix_fmt", "yuv420p", str(partial),
        ]  # fmt: skip
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdin and process.stderr  # both piped above
        try:
            for frame in frames:
                # ffmpeg's BGR input shifts colors a few levels; its RGB input doesn't.
                process.stdin.write(
                    np.ascontiguousarray(frame[..., ::-1], np.uint8).tobytes()
                )
        except BrokenPipeError:
            pass  # ffmpeg stopped reading; its log says why
        finally:
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
            log = process.stderr.read().decode("utf-8", errors="replace")
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {log.strip()}")


@contextlib.contextmanager
def _replacing(dest: Path) -> Iterator[Path]:
    """Yields a path to write instead of dest, moved onto dest only on success.

    A video cut short by an error or a crash then never stands in for a whole one.
    """
    partial = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    try:
        yield partial
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def _ffmpeg(*args: str, stdin: bytes | None = None) -> str:
    """Runs the bundled ffmpeg with args, feeding it stdin, and returns its log.

    Raises:
        RuntimeError: ffmpeg exited with an error.
    """
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostats", *args]
    result = subprocess.run(command, input=stdin, capture_output=True)
    log = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {log.strip()}")
    return log
