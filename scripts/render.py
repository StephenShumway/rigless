"""Renders one take headlessly and saves it to runs/<run id>/.

A dev tool for testing and overnight experiments; the app is the product. It
needs ComfyUI running with comfy-api-proxy in front of it
(`uv run comfy-api-proxy start`).

Typical usage example:

  uv run python scripts/render.py --video local_fixtures/source_video.mp4 \
      --image local_fixtures/character_solo.jpg --seed 962671185310412 --seconds 5
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from comfy_sdk import Progress

from rigless import comfy_client, config, video_prep, workflow


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Render one motion-transfer take.")
    parser.add_argument("--video", type=Path, required=True, help="driving dance video")
    parser.add_argument(
        "--image", type=Path, required=True, help="character reference image"
    )
    parser.add_argument("--seed", type=int, help="noise seed (random if omitted)")
    parser.add_argument("--seconds", type=float, help="render only the first N seconds")
    parser.add_argument(
        "--workflow",
        type=Path,
        default=config.WORKFLOW_PATH,
        help="API-format workflow",
    )
    return parser.parse_args()


def main() -> None:
    """Renders one take and writes its videos and run.json to a new run folder."""
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(2**53)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True)

    template = workflow.load(args.workflow)
    width, height = workflow.output_size(template)
    driving = run_dir / "driving.mp4"
    video_prep.prepare(
        args.video, driving, width=width, height=height, seconds=args.seconds
    )

    started = time.monotonic()

    def show_progress(progress: Progress) -> None:
        step = f" step {progress.step}/{progress.steps}" if progress.steps else ""
        elapsed = time.monotonic() - started
        print(
            f"\r[{elapsed:5.0f}s] node {progress.current_node}{step}    ",
            end="",
            flush=True,
        )

    print(f"Rendering seed {seed} into {run_dir}")
    with comfy_client.connect() as client:
        graph = workflow.patch(
            template,
            video=client.assets.from_file(driving),
            image=client.assets.from_file(args.image),
            seed=seed,
        )
        job = comfy_client.run(client, graph, on_progress=show_progress)
        outputs = {
            nid: run_dir / f"{role}.mp4"
            for nid, role in workflow.video_outputs(graph).items()
        }
        comfy_client.download(job, outputs)

    run = {
        "run_id": run_id,
        "seed": seed,
        "video": args.video.as_posix(),
        "image": args.image.as_posix(),
        "seconds": args.seconds,
        "workflow": args.workflow.as_posix(),
        "size": [width, height],
        "job_id": job.id,
        "render_seconds": round(time.monotonic() - started, 1),
        "outputs": sorted(path.name for path in outputs.values()),
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nSaved {run_dir}")


if __name__ == "__main__":
    main()
