import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import imageio_ffmpeg
import pytest

from rigless import video_prep

MakeVideo = Callable[..., Path]


def _size(path: Path) -> tuple[float, float]:
    video = cv2.VideoCapture(str(path))
    return video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)


@pytest.mark.parametrize(("width", "height"), [(1080, 1920), (1920, 1080)])
def test_prepared_video_just_covers_output_size(
    make_video: MakeVideo, tmp_path: Path, width: int, height: int
) -> None:
    dest = tmp_path / "dest.mp4"

    video_prep.prepare(make_video(width, height), dest, width=482, height=854)

    out_w, out_h = _size(dest)
    assert out_w >= 482 and out_h >= 854
    assert out_w <= 484 or out_h <= 856
    assert out_w / out_h == pytest.approx(width / height, rel=0.01)


def test_prepared_video_keeps_audio(make_video: MakeVideo, tmp_path: Path) -> None:
    dest = tmp_path / "dest.mp4"

    video_prep.prepare(make_video(1080, 1920), dest, width=482, height=854)

    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(dest)],
        capture_output=True,
        text=True,
    )
    assert "Audio:" in probe.stderr


def test_seconds_keeps_only_the_start(make_video: MakeVideo, tmp_path: Path) -> None:
    dest = tmp_path / "dest.mp4"

    video_prep.prepare(
        make_video(640, 360, seconds=2), dest, width=482, height=854, seconds=1
    )

    assert video_prep.probe(dest).frames == 30


def test_finished_take_keeps_its_timing(make_video: MakeVideo, tmp_path: Path) -> None:
    src = make_video(96, 160, seconds=1)
    dest = tmp_path / "finished.mp4"

    video_prep.finish(src, dest)

    assert video_prep.probe(dest) == video_prep.probe(src)


def test_finished_take_keeps_its_soundtrack(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    dest = tmp_path / "finished.mp4"

    video_prep.finish(make_video(96, 160, seconds=1), dest)

    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(dest)],
        capture_output=True,
        text=True,
    )
    assert "Audio:" in probe.stderr


def test_black_bars_are_found(make_video: MakeVideo) -> None:
    pillarboxed = make_video(608, 1080, pad_to=(1920, 1080))

    assert video_prep.detect_crop(pillarboxed) == (608, 1080, 656, 0)


def test_no_crop_without_black_bars(make_video: MakeVideo) -> None:
    assert video_prep.detect_crop(make_video(1080, 1920)) is None


def test_crop_is_applied_before_scaling(make_video: MakeVideo, tmp_path: Path) -> None:
    dest = tmp_path / "dest.mp4"

    video_prep.prepare(
        make_video(608, 1080, pad_to=(1920, 1080)),
        dest,
        width=482,
        height=854,
        crop=(608, 1080, 656, 0),
    )

    out_w, out_h = _size(dest)
    assert out_w / out_h == pytest.approx(608 / 1080, rel=0.01)


def test_probe_rejects_a_file_that_is_not_a_video(tmp_path: Path) -> None:
    fake = tmp_path / "dance.mp4"
    fake.write_text("not a video")

    with pytest.raises(ValueError, match="readable video"):
        video_prep.probe(fake)


@pytest.mark.parametrize(("frames", "expected"), [(100, 59), (40, 40)])
def test_joined_chunks_drop_the_repeated_frame_and_cut_to_length(
    make_video: MakeVideo, tmp_path: Path, frames: int, expected: int
) -> None:
    parts = [make_video(96, 160, name="a.mp4"), make_video(96, 160, name="b.mp4")]
    dest = tmp_path / "joined.mp4"

    video_prep.join(parts, dest, frames=frames, audio=parts[0])

    assert video_prep.probe(dest).frames == expected
