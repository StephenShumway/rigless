"""Rendering takes: prepare the driving video once, then one scored take per seed.

Each sweep creates `runs/<run id>/` holding the prepared driving video and one
`seed-<seed>/` folder per take, with its videos and a `run.json` recording
everything needed to reproduce it, including its motion-accuracy score.

Typical usage example:

  takes = list(render_takes(config.WORKFLOW_PATH, settings, video=dance,
                            image=hero, seeds=[1, 2, 3]))
  best = pick_best(takes)
"""

import json
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from comfy_sdk import Comfy, ComfyError, JobFailed, Progress

from rigless import comfy_client, config, pose, video_prep, workflow
from rigless.score import TakeScore, score_take
from rigless.workflow import Graph, Settings


@dataclass(frozen=True)
class Take:
    """One rendered or failed take.

    Attributes:
        seed: The sampler's noise seed.
        dir: The folder holding the take's videos and run.json.
        error: Why the take failed, or None if it succeeded.
        score: How faithfully the take follows the dancer, or None if it failed.
    """

    seed: int
    dir: Path
    error: str | None = None
    score: TakeScore | None = None

    @property
    def result(self) -> Path:
        """The rendered video."""
        return self.dir / "result.mp4"


def render_takes(
    workflow_path: Path,
    settings: Settings,
    *,
    video: Path,
    image: Path,
    seeds: Sequence[int],
    seconds: float | None = None,
    on_progress: Callable[[int, Progress], None] | None = None,
) -> Iterator[Take]:
    """Renders and scores one take per seed, yielding each as soon as it's done.

    A take that fails inside ComfyUI is recorded with its error and yielded,
    and the sweep goes on to the next seed. Failing to reach ComfyUI at all
    ends the sweep.

    Args:
        workflow_path: The API-format workflow template.
        settings: Generation settings shared by every take.
        video: The driving dance video.
        image: The character reference image.
        seeds: The seeds to render, one take each, in order.
        seconds: Render only this many seconds from the start of the video.
        on_progress: Called with the take's index in seeds and each progress
            update while it renders.

    Yields:
        Each take, in seed order.

    Raises:
        ValueError: seeds contains a duplicate.
    """
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each seed may appear only once.")
    template = workflow.load(workflow_path)
    run_dir = config.RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)

    crop = video_prep.detect_crop(video)
    driving = run_dir / "driving.mp4"
    video_prep.prepare(
        video,
        driving,
        width=settings.width,
        height=settings.height,
        seconds=seconds,
        crop=crop,
    )
    dancer = pose.estimate(driving)
    shared = {
        "run_id": run_dir.name,
        "video": _repo_path(video),
        "image": _repo_path(image),
        "seconds": seconds,
        "crop": crop,
        "workflow": _repo_path(workflow_path),
        "settings": asdict(settings),
    }

    with comfy_client.connect() as client:
        video_asset = client.assets.from_file(driving)
        image_asset = client.assets.from_file(image)
        for index, seed in enumerate(seeds):
            take_dir = run_dir / f"seed-{seed}"
            take_dir.mkdir()
            graph = workflow.patch(
                template, settings, video=video_asset, image=image_asset, seed=seed
            )
            progress = partial(on_progress, index) if on_progress else None
            started = time.monotonic()
            job_id, error = _render(client, graph, take_dir, progress)
            render_seconds = round(time.monotonic() - started, 1)
            take_score = (
                None
                if error
                else score_take(dancer, pose.estimate(take_dir / "result.mp4"))
            )
            record = {
                **shared,
                "seed": seed,
                "status": "failed" if error else "done",
                "error": error,
                "job_id": job_id,
                "render_seconds": render_seconds,
                "score": take_score.as_record() if take_score else None,
            }
            (take_dir / "run.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            yield Take(seed, take_dir, error, take_score)


def pick_best(takes: Iterable[Take]) -> Take | None:
    """Returns the take that follows the dancer best, or None if none succeeded.

    Takes flagged frozen or mirrored are picked only when every take is.
    """
    scored = [(take, take.score) for take in takes if take.score is not None]
    if not scored:
        return None
    best, _ = max(
        scored,
        key=lambda pair: (not (pair[1].frozen or pair[1].mirrored), pair[1].score),
    )
    return best


def _render(
    client: Comfy,
    graph: Graph,
    take_dir: Path,
    on_progress: Callable[[Progress], None] | None,
) -> tuple[str | None, str | None]:
    """Runs one take and saves its videos into take_dir.

    Returns:
        A tuple (job_id, error), where error is None on success and job_id is
        None if the job failed.
    """
    outputs = {
        nid: take_dir / f"{role}.mp4"
        for nid, role in workflow.video_outputs(graph).items()
    }
    try:
        job = comfy_client.run(client, graph, on_progress)
    except JobFailed as exc:
        return None, (f"{exc}: {exc.error}" if exc.error else str(exc))
    except ComfyError as exc:
        return None, str(exc)
    comfy_client.download(job, outputs)
    return job.id, None


def _repo_path(path: Path) -> str:
    """Returns path relative to the repo when inside it, so run.json is portable."""
    resolved = path.resolve()
    if resolved.is_relative_to(config.REPO_ROOT):
        resolved = resolved.relative_to(config.REPO_ROOT)
    return resolved.as_posix()
