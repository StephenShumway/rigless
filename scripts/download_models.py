"""Downloads the model files rigless needs into a ComfyUI models folder.

The Wan Animate 2 set comes from Comfy-Org/Wan-Animate-2 on Hugging Face. The
optional editors compose the character into a background photo: Qwen-Image 2.1
(Comfy-Org/Qwen-Image-2.1; its license allows research and evaluation only) and
Qwen-Image-Edit 2511 (Comfy-Org's Qwen-Image repositories and lightx2v's Lightning
LoRA; Apache-2.0). Files already present are skipped, and an interrupted download
resumes.

Typical usage example:

  uv run python scripts/download_models.py --models-dir C:/path/to/ComfyUI/models
  uv run python scripts/download_models.py --check --bf16 --qwen-image-21
"""

import argparse
import urllib.request
from pathlib import Path

WAN = "https://huggingface.co/Comfy-Org/Wan-Animate-2/resolve/main"
QWEN = "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files"
QWEN_EDIT = (
    "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files"
)
LIGHTNING = (
    "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main"
)
QWEN21 = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"

# Each model is (download URL, subfolder of ComfyUI's models folder).
REQUIRED = [
    (
        f"{WAN}/diffusion_models/wan_animate_2_int8_convrot.safetensors",
        "diffusion_models",
    ),
    (
        f"{WAN}/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "loras",
    ),
    (f"{WAN}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    (f"{WAN}/clip_vision/clip_vision_h.safetensors", "clip_vision"),
    (f"{WAN}/vae/Wan2_1_VAE_bf16.safetensors", "vae"),
]
BF16 = [(f"{WAN}/diffusion_models/wan_animate_2_bf16.safetensors", "diffusion_models")]
# int8, not the template's default fp8mixed: GPUs before Ada (e.g., the A40) have no
# FP8 compute, while comfy-kitchen runs int8 natively.
QWEN_EDIT_SET = [
    (
        f"{QWEN_EDIT}/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors",
        "diffusion_models",
    ),
    (f"{QWEN}/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
    (f"{QWEN}/vae/qwen_image_vae.safetensors", "vae"),
    (
        f"{LIGHTNING}/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "loras",
    ),
]
# The template's optional prompt rewriter (another 9GB model) is left out: our prompts
# are already written out in full.
QWEN21_SET = [
    (
        f"{QWEN21}/diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
        "diffusion_models",
    ),
    (f"{QWEN21}/text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders"),
    (f"{QWEN21}/vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
]


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Download the rigless models.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--models-dir", type=Path, help="ComfyUI's models folder")
    target.add_argument("--check", action="store_true", help="only check every link")
    parser.add_argument(
        "--bf16", action="store_true", help="also get the 33GB bf16 model (cloud-48gb)"
    )
    parser.add_argument(
        "--qwen-edit",
        action="store_true",
        help="also get Qwen-Image-Edit 2511 (31GB), for background photos",
    )
    parser.add_argument(
        "--qwen-image-21",
        action="store_true",
        help="also get Qwen-Image 2.1 (17GB; research license), for background photos",
    )
    return parser.parse_args()


def check(url: str) -> None:
    """Prints whether a model's download link works, and its size."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        size = int(response.headers.get("Content-Length", 0))
        print(f"ok  {size / 1e9:6.2f}GB  {url}")


def download(url: str, dest: Path) -> None:
    """Downloads one model to dest, resuming a partial dest.part if present."""
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(url, headers=headers)
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
    models = REQUIRED + (BF16 if args.bf16 else [])
    models += QWEN_EDIT_SET if args.qwen_edit else []
    models += QWEN21_SET if args.qwen_image_21 else []
    for url, folder in models:
        if args.check:
            check(url)
            continue
        dest = args.models_dir / folder / url.rsplit("/", 1)[1]
        if dest.exists():
            print(f"have {folder}/{dest.name}")
            continue
        print(f"get  {folder}/{dest.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        download(url, dest)


if __name__ == "__main__":
    main()
