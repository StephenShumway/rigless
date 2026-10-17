"""Driving-video preparation before upload to ComfyUI.

Uses the ffmpeg build bundled with imageio-ffmpeg, so no system install is
needed.

Typical usage example:

  prepare(Path("dance.mp4"), run_dir / "driving.mp4", width=482, height=854)
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg


def prepare(
    src: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    seconds: float | None = None,
) -> None:
    """Writes a copy of a driving video sized for the workflow, with its audio.

    ComfyUI decodes every frame of the driving video to float32, several times
    per run, so a full-resolution phone clip can exhaust system RAM. The copy
    is scaled down to just cover width x height, which the workflow crops to
    anyway, so no detail the model would see is lost. The frame rate is kept.

    Args:
        src: The source video.
        dest: Where to write the prepared .mp4.
        width: The workflow's output width.
        height: The workflow's output height.
        seconds: Keep only this many seconds from the start. None keeps it all.

    Raises:
        FileNotFoundError: src does not exist.
        RuntimeError: ffmpeg failed, e.g., because src is not a video.
    """
    if not src.is_file():
        raise FileNotFoundError(src)
    trim = ["-t", str(seconds)] if seconds is not None else []
    scale = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase"
        ":force_divisible_by=2:flags=area"
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        *trim,
        "-vf",
        scale,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(dest),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not prepare {src}: {result.stderr.strip()}")
