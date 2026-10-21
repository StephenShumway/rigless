"""Rendering takes: prepare the driving video once, then one scored take per seed.

Each run lives in `runs/<run id>/`, holding the prepared driving video and one
`seed-<seed>/` folder per take, with its videos and a `run.json` recording
everything needed to reproduce it, including its motion-accuracy score. The raw
render (`result.mp4`) is what gets scored; `finished.mp4` adds the camera finish.

`render_chunked` instead renders one take chunk by chunk, keeping the best
seed for each chunk, in `runs/<run id>/chunked/`.

Typical usage example:

  driving = prepare_driving(dance, new_run_dir(), settings)
  takes = list(render_takes(config.WORKFLOW_PATH, settings, driving=driving,
                            image=hero, seeds=[1, 2, 3]))
  best = pick_best(takes)
"""

import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from comfy_sdk import Comfy, ComfyError, JobFailed, Progress
from PIL import Image

from rigless import comfy_client, compose, config, framing, pose, video_prep, workflow
from rigless.pose import PoseTrack
from rigless.score import TakeScore, score_take
from rigless.video_prep import Crop
from rigless.workflow import Graph, Settings


@dataclass(frozen=True)
class Driving:
    """A dance video prepared for rendering, and what was found in it.

    Attributes:
        source: The dance video as given.
        seconds: How much of the source is used, from the start; None for all.
        video: The prepared copy ComfyUI renders from: trimmed, reframed, and
            scaled to the output size. Takes are saved in its folder.
        dancer: The dancer's poses in the prepared copy, to score takes against.
        window: The area of the source the copy shows.
        warnings: Plain-language problems with the footage, with fixes. None
            of them block a render.
    """

    source: Path
    seconds: float | None
    video: Path
    dancer: PoseTrack
    window: Crop
    warnings: list[str]


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
        """The rendered video, as scored."""
        return self.dir / "result.mp4"

    @property
    def finished(self) -> Path:
        """The rendered video with the camera finish, the one to show."""
        return self.dir / "finished.mp4"


def new_run_dir() -> Path:
    """Creates and returns an empty folder for a run, named by the time."""
    run_dir = config.RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    return run_dir


def prepare_driving(
    video: Path,
    run_dir: Path,
    settings: Settings,
    *,
    seconds: float | None = None,
    reframe: bool = True,
) -> Driving:
    """Checks a dance video and writes the copy to render from into run_dir.

    The dancer's poses are found in the source first, to choose the window and
    to check the footage. Black bars are always cropped away.

    Args:
        video: The dance video.
        run_dir: The run's folder; the copy is saved there as driving.mp4.
        settings: The generation settings, for the output size.
        seconds: Use only this many seconds from the start of the video.
        reframe: Frame the dancer in the output; otherwise keep the center of
            the frame, as the workflow does on its own.

    Raises:
        ValueError: video is missing or isn't a readable video.
    """
    info = video_prep.probe(video)
    frame = video_prep.detect_crop(video) or (info.width, info.height, 0, 0)
    source = pose.estimate(video, seconds)
    aspect = settings.width / settings.height
    window = (
        framing.reframe(source, frame, aspect)
        if reframe
        else framing.centered(frame, aspect)
    )
    length = min(info.duration, seconds) if seconds is not None else info.duration
    warnings = [warning] if (warning := video_prep.length_warning(length)) else []
    warnings += framing.check(source, window)
    driving = run_dir / "driving.mp4"
    video_prep.prepare(
        video,
        driving,
        width=settings.width,
        height=settings.height,
        seconds=seconds,
        crop=window,
    )
    return Driving(video, seconds, driving, pose.estimate(driving), window, warnings)


def compose_references(
    *,
    background: Path,
    front: Path,
    close_up: Path | None,
    description: str,
    out_dir: Path,
    seeds: Sequence[int],
    size: tuple[int, int],
) -> Iterator[Path]:
    """Composes the character into a background photo with Qwen-Image 2.1.

    Makes one reference image per seed, `seed-<seed>.png` in out_dir, beside
    the photo fitted to the output's aspect ratio.

    Args:
        background: A photo of the place.
        front: The character's full-body front view.
        close_up: A close-up of the character's face, or None.
        description: What the character looks like.
        out_dir: The folder for the images; created if missing.
        seeds: One seed per image.
        size: The output video's (width, height).

    Yields:
        Each reference image's path, as soon as it's done.

    Raises:
        comfy_sdk.JobFailed: ComfyUI couldn't make an image.
    """
    template = workflow.load(config.COMPOSE_QWEN21_WORKFLOW_PATH)
    save_node = workflow.find_node(template, "SaveImage")
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = out_dir / "background.png"
    with Image.open(background) as photo:
        compose.fit_background(photo, *size).save(fitted)
    instruction = compose.prompt(description, close_up is not None)
    with comfy_client.connect() as client:
        background_asset = client.assets.from_file(fitted)
        front_asset = client.assets.from_file(front)
        close_up_asset = client.assets.from_file(close_up) if close_up else None
        for seed in seeds:
            graph = compose.patch(
                template,
                background=background_asset,
                character=front_asset,
                close_up=close_up_asset,
                prompt=instruction,
                seed=seed,
            )
            dest = out_dir / f"seed-{seed}.png"
            comfy_client.download(comfy_client.run(client, graph), {save_node: dest})
            yield dest


def render_takes(
    workflow_path: Path,
    settings: Settings,
    *,
    driving: Driving,
    image: Path,
    seeds: Sequence[int],
    out_dir: Path | None = None,
    on_progress: Callable[[int, Progress], None] | None = None,
) -> Iterator[Take]:
    """Renders and scores one take per seed, yielding each as soon as it's done.

    A take that fails inside ComfyUI is recorded with its error and yielded,
    and the sweep goes on to the next seed. Failing to reach ComfyUI at all
    ends the sweep.

    Args:
        workflow_path: The API-format workflow template.
        settings: Generation settings shared by every take.
        driving: The prepared dance video.
        image: The character reference image.
        seeds: The seeds to render, one take each, in order.
        out_dir: Where the take folders go; None puts them beside the driving
            video.
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
    run_dir = out_dir or driving.video.parent
    shared = _shared_record(workflow_path, settings, driving, image)

    with comfy_client.connect() as client:
        video_asset = client.assets.from_file(driving.video)
        image_asset = client.assets.from_file(image)
        for index, seed in enumerate(seeds):
            take_dir = run_dir / f"seed-{seed}"
            take_dir.mkdir(parents=True)
            graph = workflow.patch(
                template, settings, video=video_asset, image=image_asset, seed=seed
            )
            outputs = {
                nid: take_dir / f"{role}.mp4"
                for nid, role in workflow.video_outputs(graph).items()
            }
            progress = partial(on_progress, index) if on_progress else None
            started = time.monotonic()
            job_id, error = _render(client, graph, outputs, progress)
            render_seconds = round(time.monotonic() - started, 1)
            take_score = None
            if not error:
                output = pose.estimate(take_dir / "result.mp4")
                take_score = score_take(driving.dancer, output)
                video_prep.finish(
                    take_dir / "result.mp4",
                    take_dir / "finished.mp4",
                    output,
                    settings.fragment_length,
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


def render_chunked(
    workflow_path: Path,
    settings: Settings,
    *,
    driving: Driving,
    image: Path,
    seeds: Sequence[int],
    out_dir: Path | None = None,
    on_progress: Callable[[int, int, Progress], None] | None = None,
) -> Take:
    """Renders one take chunk by chunk, keeping the best seed for each chunk.

    Every chunk is rendered once per seed, each continuing from the chunk kept
    before it, and the one that follows the dancer best is kept. That costs
    the same as one full take per seed, but a seed that goes wrong partway no
    longer spoils the whole take. The take is saved in a `chunked/` folder
    with every candidate chunk, and its run.json records how each candidate
    scored.

    Args:
        workflow_path: The API-format workflow template.
        settings: Generation settings for every chunk.
        driving: The prepared dance video.
        image: The character reference image.
        seeds: The seeds each chunk is tried with, in order.
        out_dir: Where the take's folder goes; None puts it beside the driving
            video.
        on_progress: Called with the chunk's index, the seed's index in seeds,
            and each progress update while a candidate renders.

    Returns:
        The take, under the seed its first chunk was kept from. It failed if
        every seed failed on one of the chunks.

    Raises:
        ValueError: seeds is empty or contains a duplicate.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Give at least one seed, each only once.")
    template = workflow.load(workflow_path)
    take_dir = (out_dir or driving.video.parent) / "chunked"
    take_dir.mkdir(parents=True)
    frames = len(driving.dancer.keypoints)
    kept: list[_Candidate] = []
    chunk_records: list[dict[str, object]] = []
    error = None
    started = time.monotonic()

    with comfy_client.connect() as client:
        video_asset = client.assets.from_file(driving.video)
        image_asset = client.assets.from_file(image)
        for index, chunk in enumerate(
            workflow.plan_chunks(frames, settings.fragment_length)
        ):
            continue_from = (
                client.assets.from_file(kept[-1].last_frame) if kept else None
            )
            dancer = _between(driving.dancer, chunk.start, chunk.start + chunk.length)
            candidates = []
            for seed_index, seed in enumerate(seeds):
                graph = workflow.chunk_graph(
                    workflow.patch(
                        template,
                        settings,
                        video=video_asset,
                        image=image_asset,
                        seed=seed,
                    ),
                    chunk,
                    continue_from,
                )
                [result] = workflow.video_outputs(graph)
                video = take_dir / f"chunk{index}-seed-{seed}.mp4"
                last_frame = video.with_suffix(".png")
                progress = (
                    partial(on_progress, index, seed_index) if on_progress else None
                )
                _, failure = _render(
                    client,
                    graph,
                    {result: video, workflow.LAST_FRAME: last_frame},
                    progress,
                )
                score = None if failure else score_take(dancer, pose.estimate(video))
                candidates.append(_Candidate(seed, video, last_frame, score, failure))
            chunk_records.append(
                {
                    "start": chunk.start,
                    "length": chunk.length,
                    "candidates": [candidate.as_record() for candidate in candidates],
                }
            )
            scored = [(c, c.score) for c in candidates if c.score is not None]
            if not scored:
                error = f"Every seed failed on chunk {index + 1}."
                break
            best, _ = max(scored, key=lambda pair: _rank(pair[1]))
            kept.append(best)
            chunk_records[-1]["kept_seed"] = kept[-1].seed

    take_score = None
    if error is None:
        video_prep.join(
            [candidate.video for candidate in kept],
            take_dir / "result.mp4",
            frames=frames,
            audio=driving.video,
        )
        output = pose.estimate(take_dir / "result.mp4")
        take_score = score_take(driving.dancer, output)
        video_prep.finish(
            take_dir / "result.mp4",
            take_dir / "finished.mp4",
            output,
            settings.fragment_length,
        )
    seed = kept[0].seed if kept else seeds[0]
    record = {
        **_shared_record(workflow_path, settings, driving, image),
        "seed": seed,
        "seeds": list(seeds),
        "status": "failed" if error else "done",
        "error": error,
        "render_seconds": round(time.monotonic() - started, 1),
        "score": take_score.as_record() if take_score else None,
        "chunks": chunk_records,
    }
    (take_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return Take(seed, take_dir, error, take_score)


def pick_best(takes: Iterable[Take]) -> Take | None:
    """Returns the take that follows the dancer best, or None if none succeeded.

    Takes flagged frozen or mirrored are picked only when every take is.
    """
    scored = [(take, take.score) for take in takes if take.score is not None]
    if not scored:
        return None
    best, _ = max(scored, key=lambda pair: _rank(pair[1]))
    return best


@dataclass(frozen=True)
class _Candidate:
    """One seed's render of a chunk."""

    seed: int
    video: Path
    last_frame: Path
    score: TakeScore | None
    error: str | None

    def as_record(self) -> dict[str, object]:
        """Returns the candidate's outcome as JSON-ready values."""
        score = self.score.as_record() if self.score else {}
        return {
            "seed": self.seed,
            "score": score.get("score"),
            "frozen": score.get("frozen"),
            "mirrored": score.get("mirrored"),
            "error": self.error,
        }


def _rank(score: TakeScore) -> tuple[bool, float]:
    """Orders scores so that the best is the largest, flagged ones last."""
    return not (score.frozen or score.mirrored), score.score


def _between(track: PoseTrack, start: int, stop: int) -> PoseTrack:
    """Returns the part of a track from frame start up to frame stop."""
    return PoseTrack(
        track.keypoints[start:stop],
        track.scores[start:stop],
        track.fps,
        track.others[start:stop],
    )


def _render(
    client: Comfy,
    graph: Graph,
    outputs: Mapping[str, Path],
    on_progress: Callable[[Progress], None] | None,
) -> tuple[str | None, str | None]:
    """Runs one job and saves each output node's file to its path in outputs.

    Returns:
        A tuple (job_id, error), where error is None on success and job_id is
        None if the job failed.
    """
    try:
        job = comfy_client.run(client, graph, on_progress)
    except JobFailed as exc:
        return None, (f"{exc}: {exc.error}" if exc.error else str(exc))
    except ComfyError as exc:
        return None, str(exc)
    comfy_client.download(job, outputs)
    return job.id, None


def _shared_record(
    workflow_path: Path, settings: Settings, driving: Driving, image: Path
) -> dict[str, object]:
    """Returns the run.json fields every take of a run shares."""
    return {
        "run_id": driving.video.parent.name,
        "video": _repo_path(driving.source),
        "image": _repo_path(image),
        "seconds": driving.seconds,
        "window": driving.window,
        "warnings": driving.warnings,
        "workflow": _repo_path(workflow_path),
        "settings": asdict(settings),
    }


def _repo_path(path: Path) -> str:
    """Returns path relative to the repo when inside it, so run.json is portable."""
    resolved = path.resolve()
    if resolved.is_relative_to(config.REPO_ROOT):
        resolved = resolved.relative_to(config.REPO_ROOT)
    return resolved.as_posix()
