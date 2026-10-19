"""Re-renders the face of a composed reference in detail, one variant per seed.

A dev tool until the app exists. In a full-body reference the face is small, so
fine details such as a monocle come out wrong, and the video copies them. This
cuts the head out, upscales it, re-edits it with Qwen-Image-Edit (the crop as
Picture 1, the sheet's face close-up as Picture 2), and blends the new face back
in, aligned by face keypoints. Each run writes runs/facedetail-<time>/ with the
upscaled crop, each raw face edit, one reference-seed<seed>.png per seed, and
facedetail.json. Pass the reference you like best to render.py as --image.

Typical usage example:

  uv run python scripts/detail_face.py \
      --reference runs/compose-20261018-191657-clearlens/full-clearlens-seed2.png \
      --close-up local_fixtures/character_close_up.png \
      --prompt-file local_fixtures/face_prompt.txt --seeds 1 2 3
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from rigless import comfy_client, compose, config, pose, workflow

DETAIL_SIZE = 1024  # about the editor's native resolution


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Re-render a reference's face.")
    parser.add_argument(
        "--reference", type=Path, required=True, help="composed reference image"
    )
    parser.add_argument(
        "--close-up", type=Path, required=True, help="close-up of the face"
    )
    parser.add_argument(
        "--prompt-file", type=Path, required=True, help="edit instruction, as text"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="default: 3 random seeds")
    parser.add_argument(
        "--full", action="store_true", help="no Lightning LoRA: 40 steps, ~20x slower"
    )
    return parser.parse_args()


def main() -> None:
    """Re-renders the face once per seed and blends each result back in."""
    args = parse_args()
    seeds = args.seeds or [random.randrange(2**53) for _ in range(3)]
    prompt = args.prompt_file.read_text("utf-8").strip()
    template = workflow.load(config.COMPOSE_WORKFLOW_PATH)
    save_node = workflow.find_node(template, "SaveImage")

    run_dir = config.RUNS_DIR / datetime.now().strftime("facedetail-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    with Image.open(args.reference) as image:
        reference = image.convert("RGB")
    points = pose.face_keypoints(np.asarray(reference))
    box = compose.face_box(points, reference.size)
    crop = run_dir / "crop.png"
    reference.crop(box).resize(
        (DETAIL_SIZE, DETAIL_SIZE), Image.Resampling.LANCZOS
    ).save(crop)

    with comfy_client.connect() as client:
        crop_asset = client.assets.from_file(crop)
        close_up = client.assets.from_file(args.close_up)
        for seed in seeds:
            graph = compose.patch(
                template,
                background=crop_asset,
                character=close_up,
                close_up=None,
                prompt=prompt,
                seed=seed,
                full_sampling=args.full,
            )
            face = run_dir / f"face-seed{seed}.png"
            comfy_client.download(comfy_client.run(client, graph), {save_node: face})
            with Image.open(face) as image:
                edit = image.convert("RGB")
            edit_points = pose.face_keypoints(np.asarray(edit))
            dest = run_dir / f"reference-seed{seed}.png"
            compose.blend_face(reference, box, points, edit, edit_points).save(dest)
            print(f"seed {seed}: {dest}")

    record = {
        "reference": args.reference.as_posix(),
        "close_up": args.close_up.as_posix(),
        "prompt": prompt,
        "full_sampling": args.full,
        "box": box,
        "seeds": seeds,
    }
    (run_dir / "facedetail.json").write_text(json.dumps(record, indent=2), "utf-8")


if __name__ == "__main__":
    main()
