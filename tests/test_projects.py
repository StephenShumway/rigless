import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from comfy_sdk import Progress
from PIL import Image

from rigless import describe, pose, projects, selection, swap, video_prep
from rigless.projects import Person, Project, Store, TakeState
from rigless.score import TakeScore


def _renderable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Store, str]:
    """A tracked shoot with a character, ready to render with a fake swap."""
    store = Store(tmp_path)
    project_id = store.create(trim=(0.0, 8.0))["id"]
    store.set(
        project_id,
        footage={"fps": 24.0, "frames": 720, "width": 720, "height": 1280},
        track={"people": [{"masks": "selection/person-a.mp4", "lost": [], "x": 0.5}]},
        inputs={"dance": "inputs/dance.mp4"},
    )

    def ready(state: Project) -> None:
        state["people"][0]["pictures"] = [
            {"id": "a", "file": "views/a.png", "role": "front"}
        ]
        state["people"][0]["anchors"] = [
            {"frame": 10, "positive": [[5, 5]], "negative": []}
        ]

    store.update(project_id, ready)
    folder = store.folder(project_id)
    (folder / "views").mkdir()
    Image.new("RGB", (8, 8)).save(folder / "views" / "a.png")
    (folder / "selection").mkdir()
    blank = np.zeros((3, 16, 16, 3), np.uint8)
    video_prep.write_video(blank, 24.0, folder / "selection" / "part.mp4")
    (folder / "inputs").mkdir(exist_ok=True)
    video_prep.write_video(blank, 24.0, folder / "inputs" / "dance.mp4")
    monkeypatch.setattr(describe, "unsafe", lambda *_: False)
    return store, project_id


def test_footage_is_ready_for_taps_even_when_nobody_is_found_on_the_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pose, "find_people", lambda image: [])  # nobody on the sheet
    store = Store(tmp_path)
    project_id = store.create(trim=(0.0, 1.0))["id"]
    folder = store.folder(project_id)
    blank = np.zeros((24, 16, 16, 3), np.uint8)
    video_prep.write_video(blank, 24.0, folder / "inputs" / "dance.mp4")
    Image.new("RGB", (32, 32), "white").save(folder / "inputs" / "sheet-0.png")

    def given(state: Project) -> None:
        state["inputs"]["dance"] = "inputs/dance.mp4"
        state["people"][0]["sheet"] = ["inputs/sheet-0.png"]

    store.update(project_id, given)

    projects.prepare(store, project_id)

    state = store.get(project_id)
    assert (state["status"], state["step"]) == ("waiting", "select")
    assert any("Nobody" in warning for warning in state["warnings"])


def test_render_times_a_take_by_its_sampler_steps_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, project_id = _renderable(tmp_path, monkeypatch)

    def render_takes(
        *args: Any, seeds: list[int], **kwargs: Any
    ) -> Iterator[swap.Take]:
        report = kwargs["on_progress"]
        report(0, Progress(0.5, current_node="encode", step=1, steps=3))
        for step in (1, 1, 2):  # ComfyUI reports a step more than once
            report(0, Progress(step / 8, current_node=swap.SAMPLER, step=step, steps=8))
        yield swap.Take(seed=seeds[0], dir=tmp_path, error="Stopped for the test.")

    monkeypatch.setattr(swap, "render_takes", render_takes)

    projects.render(store, project_id, {"takes": 1})

    sampling = store.get(project_id)["takes"][0]["sampling"]
    assert sampling is not None
    assert (sampling["steps"], sampling["first"][0], sampling["last"][0]) == (8, 1, 2)


def test_a_new_take_joins_the_done_ones_and_full_quality_shows_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, project_id = _renderable(tmp_path, monkeypatch)
    store.set(
        project_id,
        takes=[_take(1, "draft", "done", 95.0), _take(2, "full", "failed", None)],
    )
    take_dir = store.folder(project_id) / "render" / "seed-3"
    score = TakeScore(
        score=80.0,
        per_frame=np.array([80.0]),
        energy_ratio=None,
        frozen=False,
        mirrored=False,
    )

    def render_takes(
        *args: Any, seeds: list[int], **kwargs: Any
    ) -> Iterator[swap.Take]:
        yield swap.Take(seed=seeds[0], dir=take_dir, score=score)

    monkeypatch.setattr(swap, "render_takes", render_takes)

    projects.render(store, project_id, {"takes": 1, "seeds": [3], "quality": "full"})

    project = store.get(project_id)
    assert [(take["seed"], take["quality"]) for take in project["takes"]] == [
        (1, "draft"),
        (3, "full"),
    ]
    assert project["best"] == 3


def _take(seed: int, quality: str, status: str, score: float | None) -> TakeState:
    """A take from an earlier render."""
    return {
        "seed": seed,
        "quality": quality,
        "status": status,
        "score": score,
        "frozen": False,
        "mirrored": False,
        "per_frame": None,
        "video": f"render/seed-{seed}/finished.mp4" if status == "done" else None,
        "raw": None,
        "error": None,
        "started": None,
        "expected_seconds": None,
        "sampling": None,
    }


def test_shoot_saved_before_people_keeps_its_person(tmp_path: Path) -> None:
    store = Store(tmp_path)
    project_id = store.create()["id"]
    saved = tmp_path / project_id / "project.json"
    old = json.loads(saved.read_text("utf-8"))
    del old["people"]
    old["inputs"]["sheet"] = ["inputs/sheet-0.png"]
    old.update(
        anchors=[{"frame": 3, "positive": [[1, 1]], "negative": [], "mask": "m.png"}],
        pictures=[{"id": "a", "file": "views/a.png", "role": "front"}],
        removed_pictures=["b"],
        views={"front": "views/front.png"},
        track={"masks": "selection/masks.mp4", "lost": [[1.0, 2.0]]},
    )
    saved.write_text(json.dumps(old), "utf-8")

    project = store.get(project_id)

    (person,) = project["people"]
    assert person["sheet"] == ["inputs/sheet-0.png"]
    assert person["anchors"][0]["frame"] == 3
    assert person["pictures"][0]["id"] == "a"
    assert person["removed_pictures"] == ["b"]
    track = project["track"]
    assert track is not None
    assert track["people"][0]["masks"] == "selection/masks.mp4"


@pytest.mark.parametrize(
    ("middles", "expected"),
    [
        ([0.8, 0.2], ["on the right", "on the left"]),
        ([0.5, 0.9, 0.1], ["in the middle", "on the right", "on the left"]),
    ],
)
def test_places_name_people_left_to_right_whatever_order_they_were_added(
    middles: list[float], expected: list[str]
) -> None:
    assert projects.places(middles) == expected


def test_removed_person_hands_their_tint_to_the_next_one(tmp_path: Path) -> None:
    store = Store(tmp_path)
    project_id = store.create()["id"]
    store.update(
        project_id, lambda state: state["people"].append(_tapped(tmp_path, project_id))
    )

    project = projects.remove_person(store, project_id, 0)

    (person,) = project["people"]
    overlay = np.asarray(
        Image.open(store.folder(project_id) / person["anchors"][0]["mask"])
    )
    assert tuple(overlay[1, 1, :3]) == selection.TINTS[0]


@pytest.mark.parametrize(("removed", "kept"), [(1, True), (0, False)])
def test_removing_a_person_keeps_the_tracking_only_while_its_preview_fits(
    tmp_path: Path, removed: int, kept: bool
) -> None:
    store = Store(tmp_path)
    project_id = store.create()["id"]
    tracked = {"masks": "selection/person-a.mp4", "lost": [], "x": 0.5}
    store.set(
        project_id,
        track={"people": [tracked], "preview": "selection/preview-a.mp4", "lost": []},
    )
    projects.add_person(store, project_id)

    track = projects.remove_person(store, project_id, removed)["track"]

    assert track == (
        {"people": [tracked], "preview": "selection/preview-a.mp4", "lost": []}
        if kept
        else None
    )


def _tapped(root: Path, project_id: str) -> Person:
    """A second person, tapped on frame 0, with their overlay in the second tint."""
    folder = root / project_id / "selection"
    folder.mkdir(exist_ok=True)
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    selection.overlay(mask, selection.TINTS[1]).save(folder / "anchor.png")
    return {
        "anchors": [
            {
                "frame": 0,
                "positive": [[1, 1]],
                "negative": [],
                "mask": "selection/anchor.png",
            }
        ],
        "sheet": [],
        "views": None,
        "pictures": [],
        "removed_pictures": [],
    }


@pytest.mark.parametrize(
    ("counts", "expected"),
    [([3, 3], [3, 3]), ([9, 1], [8, 1]), ([9, 9], [5, 4]), ([2, 9, 9], [2, 4, 3])],
)
def test_pictures_are_dealt_out_in_turn_up_to_what_h3_takes(
    counts: list[int], expected: list[int]
) -> None:
    assert projects.allot(counts, 9) == expected


def test_an_unreadable_shoot_is_left_out_instead_of_breaking_the_list(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    good = store.create()["id"]
    (tmp_path / "20261001-120000").mkdir()
    (tmp_path / "20261001-120000" / "project.json").write_text("{ cut off", "utf-8")

    assert [project["id"] for project in store.all()] == [good]
