"""Prepares phone clips for training the real-footage LoRA.

Converts each clip listed in a manifest into what the trainer expects. It turns
iPhone video upright, since iPhones store portrait video sideways with a rotation
flag the trainer ignores. It tone-maps HDR clips to standard BT.709 color; read
as-is, they would teach a flat, washed-out look. It also scales to 720 pixels
wide, drops the audio, and encodes H.264. Each clip becomes <name>.mp4 next to
its caption, <name>.txt.

The manifest is TOML with one [[clip]] table per clip, giving its file (relative
to the manifest) and a caption describing what happens in it.

Typical usage example:

  uv run python scripts/prepare_lora_clips.py \
      --manifest local_fixtures/lora/clips.toml --out local_fixtures/lora/clips
"""

import argparse
import subprocess
import tomllib
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# HLG to linear light, BT.2020 to BT.709 primaries, then Mobius tone mapping, which
# keeps the bright, saturated look the phone itself shows in SDR.
TONE_MAP = (
    "zscale=tin=arib-std-b67:min=bt2020nc:pin=bt2020:t=linear:npl=100,"
    "format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,"
)


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Prepare clips for LoRA training.")
    parser.add_argument("--manifest", type=Path, required=True, help="clips.toml")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    return parser.parse_args()


def is_hdr(video: Path) -> bool:
    """True if the video is HDR (HLG or PQ), going by its stream info."""
    info = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(video)], capture_output=True, text=True
    ).stderr
    return "arib-std-b67" in info or "smpte2084" in info


def convert(video: Path, dest: Path) -> None:
    """Writes an upright, SDR, 720-pixel-wide, silent H.264 copy of a video."""
    filters = (TONE_MAP if is_hdr(video) else "") + "scale=720:-2:out_range=tv"
    subprocess.run(
        [
            FFMPEG, "-v", "error", "-y", "-i", str(video),
            "-vf", filters, "-an",
            "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv",
            "-movflags", "+faststart", str(dest),
        ],
        check=True,
    )  # fmt: skip


def main() -> None:
    """Converts every clip in the manifest and writes its caption beside it."""
    args = parse_args()
    with args.manifest.open("rb") as file:
        clips = tomllib.load(file)["clip"]
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        source = args.manifest.parent / clip["file"]
        dest = args.out / f"{source.stem}.mp4"
        convert(source, dest)
        dest.with_suffix(".txt").write_text(clip["caption"].strip() + "\n", "utf-8")
        print(dest)


if __name__ == "__main__":
    main()
