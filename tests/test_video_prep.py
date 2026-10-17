import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import pytest

from rigless import video_prep

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _make_video(path: Path, width: int, height: int, seconds: float) -> None:
    """Writes a 30fps test pattern with a sine-tone audio track."""
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate=30:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def _has_audio(path: Path) -> bool:
    probe = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in probe.stderr


@pytest.mark.parametrize(("width", "height"), [(1080, 1920), (1920, 1080)])
def test_prepared_video_just_covers_output_size(
    tmp_path: Path, width: int, height: int
) -> None:
    src, dest = tmp_path / "src.mp4", tmp_path / "dest.mp4"
    _make_video(src, width, height, seconds=1)

    video_prep.prepare(src, dest, width=482, height=854)

    video = cv2.VideoCapture(str(dest))
    out_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    out_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert out_w >= 482 and out_h >= 854
    assert out_w <= 484 or out_h <= 856
    assert out_w / out_h == pytest.approx(width / height, rel=0.01)


def test_prepared_video_keeps_audio(tmp_path: Path) -> None:
    src, dest = tmp_path / "src.mp4", tmp_path / "dest.mp4"
    _make_video(src, 1080, 1920, seconds=1)

    video_prep.prepare(src, dest, width=482, height=854)

    assert _has_audio(dest)


def test_seconds_keeps_only_the_start(tmp_path: Path) -> None:
    src, dest = tmp_path / "src.mp4", tmp_path / "dest.mp4"
    _make_video(src, 640, 360, seconds=2)

    video_prep.prepare(src, dest, width=482, height=854, seconds=1)

    assert cv2.VideoCapture(str(dest)).get(cv2.CAP_PROP_FRAME_COUNT) == 30
