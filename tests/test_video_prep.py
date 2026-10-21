import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from rigless import video_prep
from rigless.pose import NUM_KEYPOINTS, PoseTrack

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


def _track(
    frames: int, boxes: list[tuple[int, int, int, int]], found: bool = True
) -> PoseTrack:
    """A pose track whose keypoints run down each frame's (x0, y0, x1, y1) box.

    Keypoint 0 (the nose) is at the top and 16 (an ankle) at the bottom; boxes
    repeat their last entry for the remaining frames.
    """
    keypoints = np.empty((frames, NUM_KEYPOINTS, 2))
    for index in range(frames):
        x0, y0, x1, y1 = boxes[min(index, len(boxes) - 1)]
        keypoints[index, :, 0] = np.linspace(x0, x1, NUM_KEYPOINTS)
        keypoints[index, :, 1] = np.linspace(y0, y1, NUM_KEYPOINTS)
    scores = np.full((frames, NUM_KEYPOINTS), 1.0 if found else 0.0)
    return PoseTrack(keypoints, scores, fps=30.0)


def _scene(frames: int, dancer_cols: list[tuple[int, int]]) -> np.ndarray:
    """Frames of a textured 120x60 background with a dark 40px-tall dancer."""
    background = np.random.default_rng(0).integers(100, 220, (120, 60, 3), np.uint8)
    video = np.tile(background, (frames, 1, 1, 1))
    for index in range(frames):
        x0, x1 = dancer_cols[min(index, len(dancer_cols) - 1)]
        video[index, 40:80, x0:x1] = 40
    return video


@pytest.mark.parametrize("with_track", [False, True])
def test_finished_take_keeps_its_timing(
    make_video: MakeVideo, tmp_path: Path, with_track: bool
) -> None:
    src = make_video(96, 160, seconds=1)
    dest = tmp_path / "finished.mp4"
    track = _track(30, [(30, 40, 60, 120)]) if with_track else None

    video_prep.finish(src, dest, track)

    assert video_prep.probe(dest) == video_prep.probe(src)


@pytest.mark.parametrize("with_track", [False, True])
def test_finished_take_keeps_its_soundtrack(
    make_video: MakeVideo, tmp_path: Path, with_track: bool
) -> None:
    dest = tmp_path / "finished.mp4"
    track = _track(30, [(30, 40, 60, 120)]) if with_track else None

    video_prep.finish(make_video(96, 160, seconds=1), dest, track)

    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(dest)],
        capture_output=True,
        text=True,
    )
    assert "Audio:" in probe.stderr


def test_color_step_after_the_first_chunk_is_undone() -> None:
    video = _scene(8, [(20, 40)])
    video[4:, ..., 1] = (video[4:, ..., 1] * 0.94).round().astype(np.uint8)

    steady = video_prep.steady_colors(
        video, _track(8, [(20, 40, 40, 80)]), first_chunk=4
    )

    later = steady[4:].astype(int) - steady[0].astype(int)
    assert np.abs(later.mean(axis=(1, 2))).max() < 0.5


def test_dancer_moving_over_the_background_leaves_colors_alone() -> None:
    cols = [(5, 15)] * 4 + [(40, 50)] * 4
    video = _scene(8, cols)
    track = _track(8, [(x0, 40, x1, 80) for x0, x1 in cols])

    steady = video_prep.steady_colors(video, track, first_chunk=4)

    assert np.abs(steady.astype(int) - video.astype(int)).max() <= 1


def test_take_whose_dancer_was_never_found_is_unchanged() -> None:
    video = _scene(8, [(20, 40)])
    video[4:] = (video[4:] * 0.94).round().astype(np.uint8)

    steady = video_prep.steady_colors(
        video, _track(8, [(20, 40, 40, 80)], found=False), first_chunk=4
    )

    assert np.array_equal(steady, video)


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
