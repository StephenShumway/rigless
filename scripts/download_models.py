"""Downloads the model files Rigless needs into a ComfyUI models folder.

MiniMax H3 and its text encoder and VAEs come from Comfy-Org/MiniMax-H3, the
FL2V Turbo 8-step LoRA from lightx2v/Minimax-h3-Turbo, SAM 3.1 from
Comfy-Org/sam3.1, Qwen3-VL-8B (the describer) from Comfy-Org/Qwen-Image-2.1,
and the Rigless Swap LoRA from its own repository. About 60GB in all. Files
already present are skipped, and an interrupted download resumes, here (a few
tries per file) or on the next run. HF_TOKEN in
`.env` is sent along if set, for a private repository. With --progress, the
bytes downloaded so far and in all are kept in a JSON file, which a new RunPod
pod serves to the app while it sets up.

Typical usage example:

  uv run python scripts/download_models.py --models-dir /workspace/ComfyUI/models
  uv run python scripts/download_models.py --check
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from rigless import config, describe, selection, swap

H3 = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
TURBO = "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main"
SAM = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main"
QWEN21 = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
SWAP_LORA_REPO = "https://huggingface.co/StephenShumway/rigless-lora/resolve/main"
TIMEOUT = 60  # seconds without data before a download counts as stalled
TRIES = 3
PROGRESS_EVERY = 2  # seconds between progress writes

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
    parser.add_argument("--progress", type=Path, help="a JSON file to keep updated")
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


def size(url: str) -> int:
    """The size of a model's download in bytes, or 0 if the server won't say."""
    head = request(url)
    head.method = "HEAD"
    try:
        with urllib.request.urlopen(head, timeout=TIMEOUT) as response:
            return int(response.headers.get("Content-Length", 0))
    except OSError:
        return 0


def download(
    url: str, dest: Path, on_progress: Callable[[int], None] = lambda done: None
) -> None:
    """Downloads one model to dest, resuming a partial dest.part if present.

    Args:
        url: The model's download link.
        dest: Where the model goes.
        on_progress: Called after each chunk with the bytes of dest so far.

    Raises:
        OSError: The download failed or stopped early; dest.part is kept.
    """
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    resume = {"Range": f"bytes={start}-"} if start else {}
    with urllib.request.urlopen(request(url, **resume), timeout=TIMEOUT) as response:
        if response.status != 206:
            start = 0  # The server sent the whole file, so start over.
        length = response.headers.get("Content-Length")
        total = start + int(length or 0)
        done = start
        with part.open("ab" if start else "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                done += len(chunk)
                on_progress(done)
                print(
                    f"\r  {done / 1e9:6.2f} / {total / 1e9:.2f}GB", end="", flush=True
                )
    print()
    # A dropped connection can end the response early without an error.
    if length is not None and done != total:
        raise OSError(f"{dest.name} stopped at {done} of {total} bytes")
    part.rename(dest)


def fetch_all(models_dir: Path, progress: Path | None = None) -> None:
    """Downloads every model that models_dir doesn't have yet.

    Args:
        models_dir: ComfyUI's models folder.
        progress: A file to keep updated, as {"stage": "models", "done": bytes,
            "total": bytes}, or None.

    Raises:
        OSError: A model failed to download after a few tries.
    """
    wanted = [
        (url, models_dir / folder / url.rsplit("/", 1)[1]) for url, folder in MODELS
    ]
    sizes = [
        dest.stat().st_size if dest.exists() else size(url) for url, dest in wanted
    ]
    total = sum(sizes)
    finished = sum(
        length for (_, dest), length in zip(wanted, sizes, strict=True) if dest.exists()
    )
    written = 0.0

    def report(done: int, now: bool = False) -> None:
        nonlocal written
        if progress is None or not now and time.monotonic() - written < PROGRESS_EVERY:
            return
        written = time.monotonic()
        state = {"stage": "models", "done": finished + done, "total": total}
        part = progress.with_name(progress.name + ".part")
        part.write_text(json.dumps(state))
        part.replace(progress)

    report(0, now=True)
    for url, dest in wanted:
        name = f"{dest.parent.name}/{dest.name}"
        if dest.exists():
            print(f"have {name}")
            continue
        print(f"get  {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, TRIES + 1):
            try:
                download(url, dest, report)
                break
            except OSError as error:
                if attempt == TRIES:
                    raise
                print(f"  {error}; trying again")
                time.sleep(5)
        finished += dest.stat().st_size
    report(0, now=True)


def main() -> None:
    """Checks or downloads every model."""
    args = parse_args()
    if args.check:
        for url, _ in MODELS:
            check(url)
    else:
        fetch_all(args.models_dir, args.progress)


if __name__ == "__main__":
    main()
