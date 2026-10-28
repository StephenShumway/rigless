import subprocess
from collections.abc import Callable
from pathlib import Path

import imageio_ffmpeg
import pytest


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """Returns a factory for short 30fps test clips with a sine-tone soundtrack."""

    def make(
        width: int, height: int, seconds: float = 1, name: str = "src.mp4"
    ) -> Path:
        path = tmp_path / name
        video = f"color=c=0x808080:size={width}x{height}:rate=30:duration={seconds}"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error",
                "-f", "lavfi", "-i", video,
                "-f", "lavfi", "-i", f"sine=duration={seconds}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
            ],
            check=True,
        )  # fmt: skip
        return path

    return make
