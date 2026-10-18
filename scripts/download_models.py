"""Downloads the Wan Animate 2 model files into a ComfyUI models folder.

All files come from the Comfy-Org/Wan-Animate-2 repository on Hugging Face.
Files already present are skipped, and an interrupted download resumes.

Typical usage example:

  uv run python scripts/download_models.py --models-dir C:/path/to/ComfyUI/models
  uv run python scripts/download_models.py --check
"""

import argparse
import urllib.request
from pathlib import Path

REPO_URL = "https://huggingface.co/Comfy-Org/Wan-Animate-2/resolve/main"

# Paths inside the repository match ComfyUI's model subfolders.
REQUIRED = [
    "diffusion_models/wan_animate_2_int8_convrot.safetensors",
    "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "clip_vision/clip_vision_h.safetensors",
    "vae/Wan2_1_VAE_bf16.safetensors",
]
BF16 = "diffusion_models/wan_animate_2_bf16.safetensors"


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Download the Wan Animate 2 models.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--models-dir", type=Path, help="ComfyUI's models folder")
    target.add_argument("--check", action="store_true", help="only check every link")
    parser.add_argument(
        "--bf16", action="store_true", help="also get the 33GB bf16 model (cloud-48gb)"
    )
    return parser.parse_args()


def check(path: str) -> None:
    """Prints whether a model's download link works, and its size."""
    request = urllib.request.Request(f"{REPO_URL}/{path}", method="HEAD")
    with urllib.request.urlopen(request) as response:
        size = int(response.headers.get("Content-Length", 0))
        print(f"ok  {size / 1e9:6.2f}GB  {path}")


def download(path: str, dest: Path) -> None:
    """Downloads one model to dest, resuming a partial dest.part if present."""
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(f"{REPO_URL}/{path}", headers=headers)
    with urllib.request.urlopen(request) as response:
        if response.status != 206:
            start = 0  # The server sent the whole file, so start over.
        total = start + int(response.headers.get("Content-Length", 0))
        done = start
        with part.open("ab" if start else "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                done += len(chunk)
                print(
                    f"\r  {done / 1e9:6.2f} / {total / 1e9:.2f}GB", end="", flush=True
                )
    print()
    part.rename(dest)


def main() -> None:
    """Checks or downloads every model the chosen settings need."""
    args = parse_args()
    paths = REQUIRED + ([BF16] if args.bf16 else [])
    for path in paths:
        if args.check:
            check(path)
            continue
        dest = args.models_dir / path
        if dest.exists():
            print(f"have {path}")
            continue
        print(f"get  {path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        download(path, dest)


if __name__ == "__main__":
    main()
