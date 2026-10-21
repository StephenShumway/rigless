"""Composes reference images: the character edited into a background photo.

A dev tool until the app exists. It needs ComfyUI with the chosen editor's models
behind comfy-api-proxy: Qwen-Image 2.1 (`download_models.py --qwen-image-21`; the
default, research license) or Qwen-Image-Edit 2511 (`--qwen-edit`, Apache-2.0).
Each run writes runs/compose-<time>/ with the fitted background, one
reference-seed<seed>.png per seed, and compose.json recording the inputs and
prompt. Pass the reference you like best to render.py as --image. Without
--character it edits the background alone, e.g., to remove a person for relight.py.

Typical usage example:

  uv run python scripts/compose.py \
      --background local_fixtures/character_background.png \
      --character local_fixtures/character_front.png \
      --close-up local_fixtures/character_close_up.png \
      --prompt-file local_fixtures/compose_prompt_v3.txt --seeds 1 2 3
"""

import argparse
import json
import random
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PIL import Image

from rigless import comfy_client, compose, config, workflow

EDITORS = {
    "qwen-image-2.1": config.COMPOSE_QWEN21_WORKFLOW_PATH,
    "qwen-image-edit-2511": config.COMPOSE_WORKFLOW_PATH,
}


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Compose reference images.")
    parser.add_argument(
        "--background", type=Path, required=True, help="photo of the scene"
    )
    parser.add_argument("--character", type=Path, help="full-body front view")
    parser.add_argument("--close-up", type=Path, help="close-up of the face")
    parser.add_argument(
        "--prompt-file", type=Path, required=True, help="edit instruction, as text"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="default: 3 random seeds")
    parser.add_argument("--editor", choices=EDITORS, default="qwen-image-2.1")
    parser.add_argument(
        "--full",
        action="store_true",
        help="2511 only: no Lightning LoRA, 40 steps, ~20x slower, more photographic",
    )
    parser.add_argument(
        "--profile", default=config.PROFILE, help="preset whose output size to fit"
    )
    args = parser.parse_args()
    if args.full and args.editor != "qwen-image-edit-2511":
        parser.error("--full applies to --editor qwen-image-edit-2511 only")
    return args


def main() -> None:
    """Composes one reference image per seed and records how each was made."""
    args = parse_args()
    seeds = args.seeds or [random.randrange(2**53) for _ in range(3)]
    output = replace(
        workflow.read_settings(workflow.load(config.WORKFLOW_PATH)),
        **config.load_profile(args.profile),
    )
    prompt = args.prompt_file.read_text("utf-8").strip()
    workflow_path = EDITORS[args.editor]
    template = workflow.load(workflow_path)
    save_node = workflow.find_node(template, "SaveImage")

    run_dir = config.RUNS_DIR / datetime.now().strftime("compose-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    background = run_dir / "background.png"
    with Image.open(args.background) as photo:
        canvas = compose.fit_background(photo, output.width, output.height)
    canvas.save(background)

    references = []
    with comfy_client.connect() as client:
        background_asset = client.assets.from_file(background)
        character = client.assets.from_file(args.character) if args.character else None
        close_up = client.assets.from_file(args.close_up) if args.close_up else None
        for seed in seeds:
            graph = compose.patch(
                template,
                background=background_asset,
                character=character,
                close_up=close_up,
                prompt=prompt,
                seed=seed,
                full_sampling=args.full,
            )
            started = time.monotonic()
            job = comfy_client.run(client, graph)
            seconds = round(time.monotonic() - started, 1)
            dest = run_dir / f"reference-seed{seed}.png"
            comfy_client.download(job, {save_node: dest})
            print(f"seed {seed}: {seconds}s  {dest}")
            references.append({"seed": seed, "seconds": seconds, "file": dest.name})

    record = {
        "background": args.background.as_posix(),
        "character": args.character.as_posix() if args.character else None,
        "close_up": args.close_up.as_posix() if args.close_up else None,
        "prompt": prompt,
        "editor": args.editor,
        "full_sampling": args.full,
        "workflow": workflow_path.relative_to(config.REPO_ROOT).as_posix(),
        "references": references,
    }
    (run_dir / "compose.json").write_text(json.dumps(record, indent=2), "utf-8")


if __name__ == "__main__":
    main()
