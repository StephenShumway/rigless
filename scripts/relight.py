"""Relights a character pasted into a background, keeping the sheet's own pixels.

A dev tool until the app exists. A full compose edit redraws the character and
drifts from the sheet (fabric pattern, colors, face). Here the sheet's cut-out is
pasted into the scene as is, and Qwen-Image-Edit 2511 runs with denoise below 1,
so it starts from the paste and mostly changes light, shadows, and edges. It
needs the same ComfyUI setup as compose.py. Each run writes runs/relight-<time>/
with the paste (composite.png), one relit-d<denoise>-seed<seed>.png per denoise
and seed, and relight.json.

The background must already be the reference's final size and empty of people,
such as a clean plate from compose.py without --character. The default framing
matches Genjutsu's: head to shoes over 73% of the height, soles near the bottom.

Typical usage example:

  uv run python scripts/relight.py --background <clean plate>.png \
      --cutout runs/sheet-<time>/front_cutout.png \
      --character runs/sheet-<time>/front.png \
      --close-up runs/sheet-<time>/close_up.png \
      --prompt-file local_fixtures/relight_prompt.txt --denoise 0.35 0.5 0.65 --full
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from rigless import comfy_client, compose, config, workflow


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Paste and relight a character.")
    parser.add_argument(
        "--background", type=Path, required=True, help="empty scene, final size"
    )
    parser.add_argument(
        "--cutout", type=Path, required=True, help="RGBA cut-out of the front view"
    )
    parser.add_argument(
        "--character", type=Path, required=True, help="front view, for identity"
    )
    parser.add_argument("--close-up", type=Path, help="close-up of the face")
    parser.add_argument(
        "--prompt-file", type=Path, required=True, help="edit instruction, as text"
    )
    parser.add_argument(
        "--denoise", type=float, nargs="+", required=True, help="0-1, one run each"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="default: 1 random seed")
    parser.add_argument(
        "--full", action="store_true", help="no Lightning LoRA: 40 steps, cfg 4"
    )
    parser.add_argument(
        "--height", type=float, default=0.73, help="figure height / image height"
    )
    parser.add_argument(
        "--center", type=float, default=0.5, help="figure center / image width"
    )
    parser.add_argument(
        "--feet", type=float, default=0.965, help="soles' height / image height"
    )
    return parser.parse_args()


def main() -> None:
    """Pastes the character, relights it at each denoise, and records the runs."""
    args = parse_args()
    seeds = args.seeds or [random.randrange(2**53)]
    prompt = args.prompt_file.read_text("utf-8").strip()
    template = workflow.load(config.COMPOSE_WORKFLOW_PATH)
    save_node = workflow.find_node(template, "SaveImage")

    run_dir = config.RUNS_DIR / datetime.now().strftime("relight-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    composite = run_dir / "composite.png"
    with Image.open(args.background) as background, Image.open(args.cutout) as cut:
        pasted = compose.paste(
            background, cut, height=args.height, center=args.center, feet=args.feet
        )
    pasted.save(composite)

    results = []
    with comfy_client.connect() as client:
        composite_asset = client.assets.from_file(composite)
        character = client.assets.from_file(args.character)
        close_up = client.assets.from_file(args.close_up) if args.close_up else None
        for denoise in args.denoise:
            for seed in seeds:
                graph = compose.patch(
                    template,
                    background=composite_asset,
                    character=character,
                    close_up=close_up,
                    prompt=prompt,
                    seed=seed,
                    full_sampling=args.full,
                    denoise=denoise,
                )
                started = time.monotonic()
                job = comfy_client.run(client, graph)
                seconds = round(time.monotonic() - started, 1)
                dest = run_dir / f"relit-d{denoise}-seed{seed}.png"
                comfy_client.download(job, {save_node: dest})
                print(f"denoise {denoise}, seed {seed}: {seconds}s  {dest}")
                results.append(
                    {
                        "denoise": denoise,
                        "seed": seed,
                        "seconds": seconds,
                        "file": dest.name,
                    }
                )

    record = {
        "background": args.background.as_posix(),
        "cutout": args.cutout.as_posix(),
        "character": args.character.as_posix(),
        "close_up": args.close_up.as_posix() if args.close_up else None,
        "placement": {"height": args.height, "center": args.center, "feet": args.feet},
        "prompt": prompt,
        "full_sampling": args.full,
        "results": results,
    }
    (run_dir / "relight.json").write_text(json.dumps(record, indent=2), "utf-8")


if __name__ == "__main__":
    main()
