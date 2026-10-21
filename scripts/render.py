"""Renders takes headlessly, one per seed, into runs/<run id>/seed-<seed>/.

A dev tool for testing and overnight experiments; the app is the product. It
needs ComfyUI running with comfy-api-proxy in front of it
(`uv run comfy-api-proxy start`). Settings come from the workflow, then the
profile, then a --settings file, then any setting flags given here.

Typical usage example:

  uv run python scripts/render.py --video local_fixtures/source_video.mp4 \
      --image local_fixtures/character_solo.jpg \
      --settings local_fixtures/prompts.toml --seconds 5 --takes 3
  uv run python scripts/render.py --video ... --image ... --seeds 962671185310412 \
      --pose-strength 1.2
"""

import argparse
import random
import time
import tomllib
from dataclasses import fields, replace
from pathlib import Path

from comfy_sdk import Progress

from rigless import config, pipeline, workflow
from rigless.workflow import Settings


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Render motion-transfer takes.")
    parser.add_argument("--video", type=Path, required=True, help="driving dance video")
    parser.add_argument(
        "--image", type=Path, required=True, help="character reference image"
    )
    parser.add_argument(
        "--seeds",
        "--seed",
        type=int,
        nargs="+",
        default=[],
        help="seeds to render first",
    )
    parser.add_argument(
        "--takes",
        type=int,
        help="total takes; random seeds fill in after --seeds "
        "(default: one per --seeds value, or 3)",
    )
    parser.add_argument("--seconds", type=float, help="render only the first N seconds")
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="render one take chunk by chunk, keeping each chunk's best seed",
    )
    parser.add_argument(
        "--reframe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="frame the dancer in the output (default: on)",
    )
    parser.add_argument("--profile", default=config.PROFILE, help="preset in profiles/")
    parser.add_argument(
        "--settings", type=Path, help="TOML file of settings, e.g. prompts"
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=config.WORKFLOW_PATH,
        help="API-format workflow",
    )

    settings = parser.add_argument_group("settings (override all of the above)")
    settings.add_argument(
        "--prompt", dest="positive_prompt", help="character + background"
    )
    settings.add_argument("--pose-prompt", help="description of the movement")
    settings.add_argument("--pose-strength", type=float)
    settings.add_argument(
        "--pose-start-percent", type=float, help="when pose guidance starts, 0-1"
    )
    settings.add_argument(
        "--pose-end-percent", type=float, help="when pose guidance ends, 0-1"
    )
    settings.add_argument("--reference-strength", type=float)
    settings.add_argument("--steps", type=int)
    settings.add_argument("--lora-strength", type=float)
    settings.add_argument(
        "--realism-lora-strength", type=float, help="real-footage LoRA; 0 leaves it out"
    )
    settings.add_argument(
        "--realism-lora-start-percent",
        type=float,
        help="fraction of sampling before the real-footage LoRA joins; 0 = all steps",
    )
    settings.add_argument(
        "--fragment-length", type=int, help="frames per chunk, 4n + 1"
    )
    settings.add_argument("--context-windows", action=argparse.BooleanOptionalAction)
    settings.add_argument("--model", help="file in ComfyUI's diffusion_models folder")
    settings.add_argument("--sampler", help='e.g. "euler" or "lcm" (LightX2V)')
    settings.add_argument(
        "--standard",
        action="store_true",
        help="Wan Animate 2's base model instead of the distilled one: "
        "6 lcm steps with LightX2V",
    )
    settings.add_argument("--cache-device", choices=["cpu", "gpu"])
    settings.add_argument("--width", type=int)
    settings.add_argument("--height", type=int)

    args = parser.parse_args()
    if args.takes is not None and args.takes < len(args.seeds):
        parser.error("--takes is smaller than the number of --seeds")
    return args


def main() -> None:
    """Renders the requested takes and reports each one as it finishes."""
    args = parse_args()
    overrides = {
        field.name: getattr(args, field.name)
        for field in fields(Settings)
        if getattr(args, field.name) is not None
    }
    from_file = tomllib.loads(args.settings.read_text("utf-8")) if args.settings else {}
    template = workflow.load(args.workflow)
    settings = replace(
        workflow.read_settings(template),
        **config.load_profile(args.profile),
        **from_file,
    )
    if args.standard:
        settings = workflow.standard(settings)
    settings = replace(settings, **overrides)

    takes = args.takes if args.takes is not None else len(args.seeds) or 3
    seeds = args.seeds + [
        random.randrange(2**53) for _ in range(takes - len(args.seeds))
    ]

    driving = pipeline.prepare_driving(
        args.video,
        pipeline.new_run_dir(),
        settings,
        seconds=args.seconds,
        reframe=args.reframe,
    )
    for warning in driving.warnings:
        print(f"Warning: {warning}")
    length = len(driving.dancer.keypoints) / driving.dancer.fps
    print(
        f"Rendering {len(seeds)} take(s) of {length:.1f}s at "
        f"{settings.width}x{settings.height} from the source's "
        f"{driving.window[0]}x{driving.window[1]} at {driving.window[2:]} "
        f"(profile {args.profile})"
    )

    started = time.monotonic()

    def show_progress(index: int, progress: Progress, chunk: str = "") -> None:
        elapsed = int(time.monotonic() - started)
        doing = (
            f"step {progress.step}/{progress.steps}"
            if progress.steps and progress.steps > 1
            else "working"
        )
        print(
            f"\r[{chunk}seed {index + 1}/{len(seeds)}, "
            f"{elapsed // 60}m{elapsed % 60:02d}s] {doing}    ",
            end="",
            flush=True,
        )

    if args.chunked:
        take = pipeline.render_chunked(
            args.workflow,
            settings,
            driving=driving,
            image=args.image,
            seeds=seeds,
            on_progress=lambda chunk, index, progress: show_progress(
                index, progress, f"chunk {chunk + 1}, "
            ),
        )
        print(f"\nChunked take: {describe(take)} ({take.dir})")
        return

    takes = []
    for take in pipeline.render_takes(
        args.workflow,
        settings,
        driving=driving,
        image=args.image,
        seeds=seeds,
        on_progress=show_progress,
    ):
        takes.append(take)
        print(f"\nseed {take.seed}: {describe(take)} ({take.dir})")

    if best := pipeline.pick_best(takes):
        print(f"\nBest take: seed {best.seed}, {describe(best)}\n  {best.result}")


def describe(take: pipeline.Take) -> str:
    """Summarizes a take's outcome in a few words."""
    if take.error or take.score is None:
        return f"FAILED: {take.error}"
    flags = [flag for flag in ("frozen", "mirrored") if getattr(take.score, flag)]
    return f"score {take.score.score:.1f}" + (f" ({', '.join(flags)})" if flags else "")


if __name__ == "__main__":
    main()
