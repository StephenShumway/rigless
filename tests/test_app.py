import io
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rigless import config, projects, runpod, selection, sheet, terms
from rigless.app import create_app
from rigless.config import Connection
from rigless.projects import Person, Project, Store


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "CONNECTION_FILE", tmp_path / "connection.json")
    monkeypatch.setattr(config, "CUSTOM_URL", None)
    monkeypatch.setattr(config, "CUSTOM_KEY", None)
    return tmp_path / "projects"


@pytest.fixture
def client(root: Path) -> TestClient:
    return TestClient(create_app(root), base_url="http://127.0.0.1:8190")


def test_requests_naming_another_host_are_refused(client: TestClient) -> None:
    # What a page on another site gets after DNS rebinding its own name to 127.0.0.1.
    response = client.get("/api/projects", headers={"Host": "attacker.example:8190"})

    assert response.status_code == 400


def test_changes_sent_from_another_site_are_refused(
    root: Path, client: TestClient
) -> None:
    response = client.post("/api/terms", headers={"Origin": "https://attacker.example"})

    assert response.status_code == 403
    assert not terms.accepted(root)


def test_the_page_itself_can_make_changes(root: Path, client: TestClient) -> None:
    response = client.post("/api/terms", headers={"Origin": "http://localhost:8190"})

    assert response.status_code == 204
    assert terms.accepted(root)


def test_project_files_stay_inside_the_project(root: Path, client: TestClient) -> None:
    project_id = _swap_project(root)
    (root / "secret.txt").write_text("not yours")

    inside = client.get(f"/files/{project_id}/project.json")
    outside = client.get(f"/files/{project_id}/%2E%2E/secret.txt")

    assert inside.status_code == 200
    assert outside.status_code == 404


def test_upload_that_is_not_a_video_is_refused_and_cleaned_up(
    root: Path, client: TestClient
) -> None:
    response = client.post(
        "/api/projects",
        data={"description": "A man"},
        files=[
            ("dance", ("dance.mp4", b"not a video", "video/mp4")),
            ("sheet", ("sheet.png", b"not an image", "image/png")),
        ],
    )

    assert response.status_code == 400
    assert "readable video" in response.json()["detail"]
    assert client.get("/api/projects").json() == []


def _swap_project(root: Path) -> str:
    store = Store(root)
    project = store.create(trim=(0.0, 8.0))
    store.set(
        project["id"],
        status="waiting",
        step="select",
        footage={"fps": 24.0, "frames": 720, "width": 720, "height": 1280},
        track={"people": [{"masks": "selection/person-a.mp4", "lost": [], "x": 0.5}]},
    )
    return project["id"]


@pytest.mark.parametrize(
    "tap",
    [{"frame": 10, "positive": [[720, 100]]}, {"frame": 720, "positive": [[10, 10]]}],
    ids=["past the right edge", "past the last frame"],
)
def test_taps_outside_the_footage_are_refused(
    root: Path, client: TestClient, tap: dict[str, object]
) -> None:
    project_id = _swap_project(root)

    response = client.post(f"/api/projects/{project_id}/anchors", json=tap)

    assert response.status_code == 400


def test_deleted_shoot_is_gone_with_its_files(root: Path, client: TestClient) -> None:
    project_id = _swap_project(root)

    assert client.delete(f"/api/projects/{project_id}").status_code == 204

    assert client.get(f"/api/projects/{project_id}").status_code == 404
    assert not (root / project_id).exists()


def test_shoot_still_working_cannot_be_deleted(root: Path, client: TestClient) -> None:
    project_id = _swap_project(root)
    Store(root).set(project_id, status="working")

    assert client.delete(f"/api/projects/{project_id}").status_code == 409
    assert (root / project_id).exists()


def _ready(person: Person) -> None:
    person["pictures"] = [{"id": "a", "file": "views/a.png", "role": "front"}]
    person["anchors"] = [{"frame": 10, "positive": [[5, 5]], "negative": []}]


def test_swap_render_needs_the_character_first(root: Path, client: TestClient) -> None:
    project_id = _swap_project(root)

    def tap_only(state: Project) -> None:
        state["people"][0]["anchors"] = [{"frame": 10, "positive": [[5, 5]]}]

    Store(root).update(project_id, tap_only)

    response = client.post(f"/api/projects/{project_id}/render", json={"takes": 1})

    assert response.status_code == 400
    assert "character" in response.json()["detail"]


def _ready_project(root: Path) -> str:
    project_id = _swap_project(root)
    Store(root).update(project_id, lambda state: _ready(state["people"][0]))
    return project_id


def test_render_names_the_added_person_who_has_no_character(
    root: Path, client: TestClient
) -> None:
    project_id = _ready_project(root)
    assert client.post("/api/terms").status_code == 204
    assert client.post(f"/api/projects/{project_id}/people").status_code == 200

    response = client.post(f"/api/projects/{project_id}/render", json={"takes": 1})

    assert response.status_code == 400
    assert "person 2" in response.json()["detail"]


def test_people_past_the_limit_are_refused(root: Path, client: TestClient) -> None:
    project_id = _swap_project(root)
    for _ in range(projects.MAX_PEOPLE - 1):
        client.post(f"/api/projects/{project_id}/people")

    response = client.post(f"/api/projects/{project_id}/people")

    assert response.status_code == 400
    assert len(Store(root).get(project_id)["people"]) == projects.MAX_PEOPLE


def test_taps_for_a_person_who_isnt_there_are_refused(
    root: Path, client: TestClient
) -> None:
    project_id = _swap_project(root)
    tap = {"person": 1, "frame": 10, "positive": [[5, 5]]}

    response = client.post(f"/api/projects/{project_id}/anchors", json=tap)

    assert response.status_code == 404


def test_seeds_cannot_ask_for_more_takes_than_the_limit(
    root: Path, client: TestClient
) -> None:
    project_id = _ready_project(root)
    client.post("/api/terms")

    response = client.post(
        f"/api/projects/{project_id}/render", json={"takes": 1, "seeds": list(range(6))}
    )

    assert response.status_code == 422


def test_render_waits_for_the_h3_terms(root: Path, client: TestClient) -> None:
    project_id = _ready_project(root)

    response = client.post(f"/api/projects/{project_id}/render", json={"takes": 1})

    assert response.status_code == 409
    assert client.get("/api/terms").json()["accepted"] is False


def test_tracking_needs_taps_first(root: Path, client: TestClient) -> None:
    project_id = _swap_project(root)

    assert client.post(f"/api/projects/{project_id}/track").status_code == 400


def test_new_trim_drops_tracking_made_for_the_old_one(
    root: Path, client: TestClient
) -> None:
    project_id = _swap_project(root)

    response = client.patch(f"/api/projects/{project_id}", json={"trim": [2.0, 9.5]})

    assert response.json()["trim"] == [2.0, 9.5]
    assert response.json()["track"] is None


def test_trim_longer_than_a_render_allows_is_refused(
    root: Path, client: TestClient
) -> None:
    project_id = _swap_project(root)

    response = client.patch(f"/api/projects/{project_id}", json={"trim": [3.0, 25.0]})

    assert response.status_code == 400
    assert client.get(f"/api/projects/{project_id}").json()["trim"] == [0.0, 8.0]


def test_recent_characters_list_each_character_once(
    root: Path, client: TestClient
) -> None:
    store = Store(root)
    for picture in (b"monocle man", b"monocle man", b"curly hair"):
        project = store.create()
        (root / project["id"] / "inputs" / "sheet-0.png").write_bytes(picture)

        def give_character(state: Project) -> None:
            state["people"][0]["sheet"] = ["inputs/sheet-0.png"]
            state["people"][0]["views"] = {"front": "views/front.png"}

        store.update(project["id"], give_character)

    characters = client.get("/api/characters").json()

    assert len(characters) == 2


def test_recent_characters_skip_a_shoot_whose_pictures_are_gone(
    root: Path, client: TestClient
) -> None:
    store = Store(root)
    project = store.create()

    def give_character(state: Project) -> None:
        state["people"][0]["sheet"] = ["inputs/sheet-0.png"]  # never saved
        state["people"][0]["views"] = {"front": "views/front.png"}

    store.update(project["id"], give_character)

    assert client.get("/api/characters").json() == []


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def one_view_per_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stands in for the pose model: n images give a front view and n other
    views, the same ones each time, each a different flat color."""

    def extract(images: list[Image.Image]) -> sheet.Views:
        views = [
            Image.new("RGB", (8, 8), (40 * i, 0, 0)) for i in range(len(images) + 1)
        ]
        return sheet.Views(views[0], None, None, None, tuple(views[1:]))

    monkeypatch.setattr(sheet, "extract", extract)


@pytest.mark.usefixtures("one_view_per_image")
def test_removed_picture_stays_out_when_more_are_added(
    root: Path, client: TestClient
) -> None:
    project_id = _swap_project(root)
    url = f"/api/projects/{project_id}"
    pictures = client.post(
        f"{url}/character", files=[("sheet", ("a.png", _png(), "image/png"))]
    ).json()["people"][0]["pictures"]
    removed = pictures[1]["id"]

    client.delete(f"{url}/pictures/{removed}")
    after = client.post(
        f"{url}/character",
        data={"append": "true"},
        files=[("sheet", ("b.png", _png(), "image/png"))],
    ).json()["people"][0]["pictures"]

    assert len(after) == 2
    assert removed not in {picture["id"] for picture in after}


@pytest.mark.usefixtures("one_view_per_image")
def test_a_failed_character_replacement_keeps_the_old_pictures(
    root: Path, client: TestClient
) -> None:
    project_id = _swap_project(root)
    url = f"/api/projects/{project_id}/character"
    old = client.post(url, files=[("sheet", ("a.png", _png(), "image/png"))]).json()
    old_sheet = old["people"][0]["sheet"]

    response = client.post(url, files=[("sheet", ("b.gif", b"GIF89a", "image/gif"))])

    assert response.status_code == 400
    assert client.get(f"/api/projects/{project_id}").json()["people"] == old["people"]
    assert all((root / project_id / path).is_file() for path in old_sheet)


def test_a_damaged_picture_is_refused_and_the_shoot_cleaned_up(
    client: TestClient, make_video: Callable[..., Path]
) -> None:
    picture = bytearray(_png())
    picture[-13] ^= 0xFF  # the image data's checksum, just before the end chunk
    clip = make_video(64, 64).read_bytes()

    response = client.post(
        "/api/projects",
        files=[
            ("dance", ("dance.mp4", clip, "video/mp4")),
            ("sheet", ("sheet.png", bytes(picture), "image/png")),
        ],
    )

    assert response.status_code == 400
    assert client.get("/api/projects").json() == []


@pytest.mark.usefixtures("one_view_per_image")
def test_last_picture_cannot_be_removed(root: Path, client: TestClient) -> None:
    project_id = _swap_project(root)
    url = f"/api/projects/{project_id}"
    pictures = client.post(
        f"{url}/character", files=[("sheet", ("a.png", _png(), "image/png"))]
    ).json()["people"][0]["pictures"]

    client.delete(f"{url}/pictures/{pictures[0]['id']}")
    response = client.delete(f"{url}/pictures/{pictures[1]['id']}")

    assert response.status_code == 400


def test_shoot_saved_before_titles_and_posters_still_lists(
    root: Path, client: TestClient
) -> None:
    project_id = _swap_project(root)
    saved = root / project_id / "project.json"
    state = json.loads(saved.read_text())
    del state["title"], state["poster"]
    saved.write_text(json.dumps(state))

    assert [shoot["id"] for shoot in client.get("/api/projects").json()] == [project_id]


def test_runpod_needs_the_users_key(client: TestClient) -> None:
    assert client.get("/api/runpod").status_code == 400


def test_started_pod_becomes_the_connection_with_its_own_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: dict[str, str] = {}

    def start(key: str, gpu: str, env: dict[str, str]) -> dict[str, str]:
        sent.update(env)
        return {"id": "abc123"}

    monkeypatch.setattr(runpod, "start", start)
    monkeypatch.setattr(runpod, "pod", lambda key: None)
    monkeypatch.setattr(runpod, "gpus", lambda key: [])
    client.put("/api/connection", json={"target": "local"})

    response = client.post(
        "/api/runpod/start", json={"gpu": "NVIDIA A40"}, headers={"X-RunPod-Key": "rk"}
    )

    assert response.status_code == 200
    token = sent["PROXY_TOKEN"]
    assert token != "rk"
    assert config.connection() == Connection(
        "runpod", "https://abc123-8189.proxy.runpod.net", token
    )
    assert token not in client.get("/api/connection").text


def test_connection_test_fails_when_nothing_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "LOCAL_URL", "http://127.0.0.1:9")
    client.put("/api/connection", json={"target": "local"})

    result = client.post("/api/connection/test").json()

    assert (result["target"], result["ok"]) == ("local", False)


def test_runpod_without_a_pod_says_to_deploy_one(client: TestClient) -> None:
    result = client.post("/api/connection/test").json()

    assert (result["target"], result["ok"]) == ("runpod", False)
    assert "Deploy" in result["message"]


def test_choice_is_refused_while_comfy_base_url_sets_the_connection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "CUSTOM_URL", "http://gpu-box:8189")

    response = client.put("/api/connection", json={"target": "local"})

    assert response.status_code == 409


def test_runpods_refusal_reaches_the_page_in_its_own_words(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def start(key: str, gpu: str, env: dict[str, str]) -> None:
        raise runpod.RunPodError("No H100s are free right now.")

    monkeypatch.setattr(runpod, "start", start)

    response = client.post(
        "/api/runpod/start", json={"gpu": "NVIDIA H100"}, headers={"X-RunPod-Key": "rk"}
    )

    # Cloudflare replaces the body of a 502 or 504 with its own page.
    assert response.status_code not in (502, 504)
    assert response.json()["detail"] == "No H100s are free right now."


def test_connection_follows_a_pod_runpod_moved_to_another_machine(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.save_pod(runpod.proxy_url("old-id"), "token")
    moved = {"id": "new-id", "name": runpod.NAME, "desiredStatus": "RUNNING"}
    monkeypatch.setattr(runpod, "pod", lambda key: moved)
    monkeypatch.setattr(runpod, "setup_progress", lambda url: None)

    client.get("/api/runpod", headers={"X-RunPod-Key": "rk"})

    assert config.connection() == Connection(
        "runpod", runpod.proxy_url("new-id"), "token"
    )


def test_new_shoot_is_numbered_after_the_highest_so_far(
    root: Path,
    client: TestClient,
    make_video: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projects, "prepare", lambda store, project_id: None)
    store = Store(root)
    for title in ("Shoot 4", "Shoot 2", "My dance"):
        store.set(store.create()["id"], title=title)
    clip = make_video(64, 64)

    response = client.post(
        "/api/projects", files=[("dance", ("clip.mp4", clip.read_bytes(), "video/mp4"))]
    )

    assert client.get(f"/api/projects/{response.json()['id']}").json()["title"] == (
        "Shoot 5"
    )


def test_a_new_pick_clears_the_last_failure(
    root: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = _swap_project(root)
    Store(root).set(project_id, status="failed", error="Tap the person in the trim.")
    found = np.zeros((1280, 720), dtype=bool)
    found[600:700, 300:400] = True
    monkeypatch.setattr(selection, "detect", lambda footage, anchor, work: found)

    response = client.post(
        f"/api/projects/{project_id}/anchors",
        json={"frame": 100, "positive": [[350, 650]]},
    )

    assert (response.json()["status"], response.json()["error"]) == ("waiting", None)


def _tapped(root: Path, frames: list[int]) -> str:
    project_id = _swap_project(root)

    def tap(state: Project) -> None:
        state["people"][0]["anchors"] = [
            {"frame": frame, "positive": [[5, 5]], "negative": [], "mask": "m.png"}
            for frame in frames
        ]

    Store(root).update(project_id, tap)
    return project_id


def test_removing_one_tap_keeps_the_others_and_drops_the_tracking(
    root: Path, client: TestClient
) -> None:
    project_id = _tapped(root, [10, 100])

    project = client.delete(f"/api/projects/{project_id}/anchors/100").json()

    assert [anchor["frame"] for anchor in project["people"][0]["anchors"]] == [10]
    assert project["track"] is None


def test_clearing_a_persons_taps_removes_every_one(
    root: Path, client: TestClient
) -> None:
    project_id = _tapped(root, [10, 100])

    project = client.delete(f"/api/projects/{project_id}/anchors?person=0").json()

    assert project["people"][0]["anchors"] == []
    assert project["track"] is None
