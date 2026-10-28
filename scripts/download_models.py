"""Downloads the model files Rigless needs into a ComfyUI models folder.

MiniMax H3 and its text encoder and VAEs come from Comfy-Org/MiniMax-H3, the
FL2V Turbo 8-step LoRA from lightx2v/Minimax-h3-Turbo, SAM 3.1 from
Comfy-Org/sam3.1, Qwen3-VL-8B (the describer) from Comfy-Org/Qwen-Image-2.1,
and the Rigless Swap LoRA from its own repository. About 60GB in all. Files
already present are skipped, and an interrupted download resumes. HF_TOKEN in
`.env` is sent along if set, for a private repository.

Typical usage example:

  uv run python scripts/download_models.py --models-dir /workspace/ComfyUI/models
  uv run python scripts/download_models.py --check
"""

import argparse
import urllib.error
import urllib.request
from pathlib import Path

from rigless import config, describe, selection, swap

H3 = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
TURBO = "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main"
SAM = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main"
QWEN21 = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
SWAP_LORA_REPO = "https://huggingface.co/StephenShumway/rigless-lora/resolve/main"

# Each model is (download URL, subfolder of ComfyUI's models folder).
MODELS = [
    (f"{H3}/diffusion_models/{swap.MODEL}", "diffusion_models"),
    (f"{H3}/text_encoders/{swap.TEXT_ENCODER}", "text_encoders"),
    (f"{H3}/vae/{swap.VIDEO_VAE}", "vae"),
    (f"{H3}/vae/{swap.AUDIO_VAE}", "vae"),
    (f"{TURBO}/{swap.TURBO_LORA}", "loras"),
    (f"{SWAP_LORA_REPO}/{swap.SWAP_LORA}", "loras"),
    (f"{SAM}/checkpoints/{selection.SAM_CHECKPOINT}", "checkpoints"),
    (f"{QWEN21}/text_encoders/{describe.VLM}", "text_encoders"),
]


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Download the Rigless models.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--models-dir", type=Path, help="ComfyUI's models folder")
    target.add_argument("--check", action="store_true", help="only check every link")
    return parser.parse_args()


def request(url: str, **kwargs: str) -> urllib.request.Request:
    """A request for url, with the Hugging Face token when there is one."""
    headers = dict(kwargs)
    if config.HF_TOKEN:
        headers["Authorization"] = f"Bearer {config.HF_TOKEN}"
    return urllib.request.Request(url, headers=headers)


def check(url: str) -> None:
    """Prints whether a model's download link works, and its size."""
    head = request(url)
    head.method = "HEAD"
    try:
        with urllib.request.urlopen(head) as response:
            size = int(response.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as exc:
        print(f"BAD ({exc.code})  {url}")
        return
    print(f"ok  {size / 1e9:6.2f}GB  {url}")


def download(url: str, dest: Path) -> None:
    """Downloads one model to dest, resuming a partial dest.part if present."""
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    resume = {"Range": f"bytes={start}-"} if start else {}
    with urllib.request.urlopen(request(url, **resume)) as response:
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
    """Checks or downloads every model."""
    args = parse_args()
    for url, folder in MODELS:
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
