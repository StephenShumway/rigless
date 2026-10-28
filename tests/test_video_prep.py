import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from rigless import video_prep

MakeVideo = Callable[..., Path]


def test_probe_rejects_a_file_that_is_not_a_video(tmp_path: Path) -> None:
    fake = tmp_path / "dance.mp4"
    fake.write_text("not a video")

    with pytest.raises(ValueError, match="readable video"):
        video_prep.probe(fake)


def test_trim_past_the_clip_end_stops_at_the_end() -> None:
    assert video_prep.check_trim(2.0, 30.0, duration=9.5) == (2.0, 9.5)


@pytest.mark.parametrize(
    ("start", "end"),
    [(31.0, None), (3.0, 3.5), (5.0, 2.0), (0.0, 25.0)],
    ids=["starts after the clip", "too short", "ends before it starts", "too long"],
)
def test_trims_that_cannot_be_rendered_are_refused(
    start: float, end: float | None
) -> None:
    with pytest.raises(ValueError):
        video_prep.check_trim(start, end, duration=30.0)


def test_conformed_footage_has_the_trimmed_frame_count_and_fits_the_size(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    dest = tmp_path / "footage.mp4"

    video_prep.conform(
        make_video(1080, 1920, seconds=3),
        dest,
        start=0.5,
        end=2.5,
        fps=24,
        max_side=640,
    )

    info = video_prep.probe(dest)
    assert info.frames == 48
    assert (info.width, info.height) == (360, 640)


def test_extracted_frame_is_the_one_asked_for(tmp_path: Path) -> None:
    ramp = np.stack([np.full((16, 16), 10 * index, np.uint8) for index in range(12)])
    video = tmp_path / "ramp.mp4"
    video_prep.write_video(ramp, 24, video)

    video_prep.extract_frame(video, 7, tmp_path / "frame.png")

    frame = cv2.imread(str(tmp_path / "frame.png"), cv2.IMREAD_GRAYSCALE)
    assert frame is not None
    assert frame.mean() == pytest.approx(70, abs=4)


def test_added_sound_never_shortens_the_video(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    video = make_video(96, 160, seconds=2, name="take.mp4")
    sound = make_video(96, 160, seconds=1, name="clip.mp4")
    dest = tmp_path / "finished.mp4"

    video_prep.add_audio(video, sound, 0.0, dest)

    assert video_prep.probe(dest).frames == video_prep.probe(video).frames
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(dest)],
        capture_output=True,
        text=True,
    )
    assert "Audio:" in probe.stderr


def test_cut_keeps_the_frame_rate_exact(make_video: Callable[..., Path]) -> None:
    # The cut's last frame had no length, so its rate read 24.057 instead of 24
    # and ComfyUI wrote one mask frame too few (Oct 26).
    src = make_video(64, 64, seconds=1)
    dest = src.with_name("part.mp4")

    video_prep.cut_frames(src, dest, 5, 26)

    info = video_prep.probe(dest)
    assert (info.frames, info.fps) == (21, 30.0)


def test_lead_in_plays_the_opening_backward_then_the_whole_clip(tmp_path: Path) -> None:
    ramp = np.stack([np.full((16, 16), 20 * index, np.uint8) for index in range(6)])
    video = tmp_path / "ramp.mp4"
    video_prep.write_video(ramp, 24, video)

    video_prep.lead_in(video, 3, 24, tmp_path / "led.mp4")

    levels = [
        round(frame.mean() / 20)
        for frame in video_prep.iter_frames(tmp_path / "led.mp4")
    ]
    assert levels == [3, 2, 1, 0, 1, 2, 3, 4, 5]
