import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from comfy_sdk import JobFailed

from rigless import comfy_client, config, pipeline, pose, workflow
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


def _fake_estimate(video: Path) -> pose.PoseTrack:
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
    settings = workflow.read_settings(workflow.load(config.WORKFLOW_PATH))

    takes = list(
        pipeline.render_takes(
            config.WORKFLOW_PATH,
            settings,
            video=make_video(1080, 1920),
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
