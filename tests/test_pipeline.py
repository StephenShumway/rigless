import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from comfy_sdk import JobFailed

from rigless import comfy_client, config, pipeline, pose, video_prep, workflow
from rigless.score import TakeScore

MakeVideo = Callable[..., Path]
FAILING_SEED = 1


@contextmanager
def _fake_connect() -> Iterator[SimpleNamespace]:
    yield SimpleNamespace(assets=SimpleNamespace(from_file=lambda path: path.name))


def _fake_run(
    client: object, graph: workflow.Graph, on_progress: object
) -> SimpleNamespace:
    seed = graph[workflow.find_node(graph, "SamplerCustom")]["inputs"]["noise_seed"]
    if seed == FAILING_SEED:
        raise JobFailed("job ended failed")
    return SimpleNamespace(id=f"job-{seed}")


def _fake_download(job: object, destinations: Mapping[str, Path]) -> None:
    for dest in destinations.values():
        dest.write_bytes(b"video")


def _fake_estimate(video: Path, seconds: float | None = None) -> pose.PoseTrack:
    return pose.PoseTrack(np.zeros((30, 17, 2)), np.zeros((30, 17)), 30.0)


def _take(seed: int, score: float | None, frozen: bool = False) -> pipeline.Take:
    result = None
    if score is not None:
        result = TakeScore(score, np.array([score]), 1.0, frozen, mirrored=False)
    return pipeline.Take(seed, Path(f"seed-{seed}"), score=result)


def test_failed_take_is_recorded_and_sweep_continues(
    make_video: MakeVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(comfy_client, "connect", _fake_connect)
    monkeypatch.setattr(comfy_client, "run", _fake_run)
    monkeypatch.setattr(comfy_client, "download", _fake_download)
    monkeypatch.setattr(pose, "estimate", _fake_estimate)
    monkeypatch.setattr(video_prep, "finish", lambda src, dest: None)
    settings = workflow.read_settings(workflow.load(config.WORKFLOW_PATH))
    driving = pipeline.prepare_driving(
        make_video(1080, 1920), pipeline.new_run_dir(), settings
    )

    takes = list(
        pipeline.render_takes(
            config.WORKFLOW_PATH,
            settings,
            driving=driving,
            image=tmp_path / "hero.png",
            seeds=[FAILING_SEED, 2],
        )
    )

    records = [json.loads((take.dir / "run.json").read_text()) for take in takes]
    assert [record["status"] for record in records] == ["failed", "done"]
    assert takes[0].error is not None
    assert takes[1].error is None and takes[1].result.exists()


def test_best_take_moves_even_if_a_frozen_take_scores_higher() -> None:
    takes = [_take(1, 80, frozen=True), _take(2, 70), _take(3, 60), _take(4, None)]

    best = pipeline.pick_best(takes)

    assert best is not None and best.seed == 2


def test_no_best_take_when_every_take_failed() -> None:
    assert pipeline.pick_best([_take(1, None), _take(2, None)]) is None


def _standing(frames: int) -> pose.PoseTrack:
    keypoints = np.tile(np.arange(34, dtype=float).reshape(17, 2) * 10, (frames, 1, 1))
    return pose.PoseTrack(keypoints, np.ones((frames, 17)), 30.0)


def _driving(run_dir: Path) -> pipeline.Driving:
    run_dir.mkdir(parents=True)
    video = run_dir / "driving.mp4"
    return pipeline.Driving(video, None, video, _standing(150), (720, 1280, 0, 0), [])


@pytest.fixture
def fake_comfy(monkeypatch: pytest.MonkeyPatch) -> list[workflow.Graph]:
    """Fakes ComfyUI and the pose model; returns every graph submitted.

    Seed 1 fails, seed 3 renders nobody, and any other seed follows the dancer.
    """
    graphs: list[workflow.Graph] = []

    def run(client: object, graph: workflow.Graph, on_progress: object) -> object:
        graphs.append(graph)
        return _fake_run(client, graph, on_progress)

    def estimate(video: Path, seconds: float | None = None) -> pose.PoseTrack:
        if "seed-3" in video.name:
            return pose.PoseTrack(np.zeros((81, 17, 2)), np.zeros((81, 17)), 30.0)
        return _standing(81)

    monkeypatch.setattr(comfy_client, "connect", _fake_connect)
    monkeypatch.setattr(comfy_client, "run", run)
    monkeypatch.setattr(comfy_client, "download", _fake_download)
    monkeypatch.setattr(pose, "estimate", estimate)
    monkeypatch.setattr(video_prep, "finish", lambda src, dest: None)
    return graphs


def test_each_chunk_keeps_its_best_seed_and_the_next_continues_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_comfy: list[workflow.Graph]
) -> None:
    joined: list[Path] = []
    monkeypatch.setattr(
        video_prep, "join", lambda parts, dest, frames, audio: joined.extend(parts)
    )
    settings = workflow.read_settings(workflow.load(config.WORKFLOW_PATH))

    take = pipeline.render_chunked(
        config.WORKFLOW_PATH,
        settings,
        driving=_driving(tmp_path / "run"),
        image=tmp_path / "hero.png",
        seeds=[FAILING_SEED, 2, 3],
    )

    record = json.loads((take.dir / "run.json").read_text())
    assert take.error is None
    assert [chunk["kept_seed"] for chunk in record["chunks"]] == [2, 2]
    assert [part.name for part in joined] == ["chunk0-seed-2.mp4", "chunk1-seed-2.mp4"]
    second_chunk = fake_comfy[3:]
    assert {
        graph[workflow.CONTINUE_FROM]["inputs"]["image"] for graph in second_chunk
    } == {"chunk0-seed-2.png"}


def test_chunked_take_fails_when_every_seed_fails_a_chunk(
    tmp_path: Path, fake_comfy: list[workflow.Graph]
) -> None:
    settings = workflow.read_settings(workflow.load(config.WORKFLOW_PATH))

    take = pipeline.render_chunked(
        config.WORKFLOW_PATH,
        settings,
        driving=_driving(tmp_path / "run"),
        image=tmp_path / "hero.png",
        seeds=[FAILING_SEED],
    )

    assert take.error == "Every seed failed on chunk 1."
    assert json.loads((take.dir / "run.json").read_text())["status"] == "failed"


def test_takes_go_into_a_new_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_comfy: list[workflow.Graph]
) -> None:
    settings = workflow.read_settings(workflow.load(config.WORKFLOW_PATH))
    out_dir = tmp_path / "run" / "render-1"

    [take] = pipeline.render_takes(
        config.WORKFLOW_PATH,
        settings,
        driving=_driving(tmp_path / "run"),
        image=tmp_path / "hero.png",
        seeds=[2],
        out_dir=out_dir,
    )

    assert take.dir == out_dir / "seed-2"
    assert (take.dir / "run.json").exists()
