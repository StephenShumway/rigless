import subprocess
from collections.abc import Callable, Iterator
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
    [(31.0, None), (3.0, 3.5), (5.0, 2.0), (0.0, 25.0), (2.0, float("nan"))],
    ids=[
        "starts after the clip",
        "too short",
        "ends before it starts",
        "too long",
        "ends at no time",
    ],
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


def test_conforming_uneven_frame_times_keeps_every_frame(tmp_path: Path) -> None:
    # As in a screen recording: after the first frame, each sits 0.6 or 0.4 of a
    # frame past the 24 fps grid, in turn, so rounding to the nearest slot puts
    # pairs together.
    clip = tmp_path / "uneven.mp4"
    uneven = "if(eq(N,0),0,N+0.6-0.2*mod(N+1,2))/24/TB"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24:duration=3",
            "-vf", f"settb=1/600,setpts='{uneven}'", "-fps_mode", "passthrough",
            "-enc_time_base:v", "1/600", "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True,
    )  # fmt: skip
    dest = tmp_path / "footage.mp4"

    video_prep.conform(clip, dest, start=0.0, end=2.0, fps=24, max_side=64)

    frames = video_prep.read_frames(dest).astype(np.int16)
    changes = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))
    assert (changes > 0.3).all()


def test_extracted_frame_is_the_one_asked_for(tmp_path: Path) -> None:
    ramp = np.stack([np.full((16, 16), 10 * index, np.uint8) for index in range(12)])
    video = tmp_path / "ramp.mp4"
    video_prep.write_video(ramp, 24, video)

    video_prep.extract_frame(video, 7, tmp_path / "frame.png")

    frame = cv2.imread(str(tmp_path / "frame.png"), cv2.IMREAD_GRAYSCALE)
    assert frame is not None
    assert frame.mean() == pytest.approx(70, abs=4)


def test_asking_for_a_frame_past_the_end_fails_even_over_an_old_image(
    tmp_path: Path,
) -> None:
    video = tmp_path / "short.mp4"
    video_prep.write_video(np.zeros((4, 16, 16), np.uint8), 24, video)
    video_prep.extract_frame(video, 1, tmp_path / "frame.png")

    with pytest.raises(RuntimeError):
        video_prep.extract_frame(video, 9, tmp_path / "frame.png")


def test_frames_that_fail_part_way_leave_no_video_behind(tmp_path: Path) -> None:
    def frames() -> Iterator[np.ndarray]:
        yield np.zeros((16, 16, 3), np.uint8)
        raise ValueError("the take broke")

    with pytest.raises(ValueError):
        video_prep.write_frames(frames(), 24, (16, 16), tmp_path / "take.mp4")

    assert list(tmp_path.iterdir()) == []


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


def test_delayed_sound_starts_after_the_delay(
    make_video: MakeVideo, tmp_path: Path
) -> None:
    video = make_video(96, 160, seconds=2, name="padded.mp4")
    sound = make_video(96, 160, seconds=2, name="clip.mp4")
    dest = tmp_path / "part.mp4"

    video_prep.add_audio(video, sound, 0.0, dest, delay=0.5)

    rate = 8000
    pcm = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(dest), "-ac", "1"]
        + ["-ar", str(rate), "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    level = np.abs(np.frombuffer(pcm, np.int16).astype(int))
    assert level[: int(0.45 * rate)].max() < 100  # the lead-in is silent
    assert level[int(0.55 * rate) : int(1.5 * rate)].mean() > 1000  # then the tone


def test_cut_keeps_the_frame_rate_exact(make_video: Callable[..., Path]) -> None:
    # The cut's last frame had no length, so its rate read 24.057 instead of 24
    # and ComfyUI wrote one mask frame too few (Oct 26).
    src = make_video(64, 64, seconds=1)
    dest = src.with_name("part.mp4")

    video_prep.cut_frames(src, dest, 5, 26)

    info = video_prep.probe(dest)
    assert (info.frames, info.fps) == (21, 30.0)


def test_pad_plays_the_ends_backward_around_the_whole_clip(tmp_path: Path) -> None:
    ramp = np.stack([np.full((16, 16), 20 * index, np.uint8) for index in range(6)])
    video = tmp_path / "ramp.mp4"
    video_prep.write_video(ramp, 24, video)

    video_prep.pad(video, 3, 2, 24, tmp_path / "padded.mp4")

    levels = [
        round(frame.mean() / 20)
        for frame in video_prep.iter_frames(tmp_path / "padded.mp4")
    ]
    assert levels == [3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3]
    assert levels == video_prep.padded_order(6, 3, 2).tolist()  # the masks' order
