import subprocess
from collections.abc import Callable
from pathlib import Path

import imageio_ffmpeg
import pytest


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """Returns a factory for short 30fps test clips with a sine-tone soundtrack.

    pad_to letterboxes or pillarboxes the picture with black bars to that size.
    """

    def make(
        width: int,
        height: int,
        seconds: float = 1,
        pad_to: tuple[int, int] | None = None,
        name: str = "src.mp4",
    ) -> Path:
        path = tmp_path / name
        video = f"color=c=0x808080:size={width}x{height}:rate=30:duration={seconds}"
        if pad_to:
            video += f",pad={pad_to[0]}:{pad_to[1]}:(ow-iw)/2:(oh-ih)/2:black"
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
