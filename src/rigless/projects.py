"""Shoots in the app: footage and a character, from upload to scored takes.

A project lives in its own folder with its inputs, intermediate files, takes,
and `project.json`, which holds its state in the shape the app's API reports.
Steps run one at a time on a background worker, since they share one GPU, and
update the state as they go, so a page polling it shows live progress and a
refresh loses nothing.

Everything happens on one screen: `prepare` readies the footage, the user adds
the character (`set_character`) and the trim in any order, taps the person to
replace (`add_anchor`), `track` follows them through the trimmed part for the
user to check, and `render` describes the character, renders the swap with
MiniMax H3, and scores the takes. A shoot can replace several people at once
(`add_person`), each with their own character and taps, in one H3 render.

Typical usage example:

  store = Store(config.RUNS_DIR / "app")
  worker = Worker(store)
  project = store.create(trim=(2.0, 10.0))
  ...  # save the inputs into store.folder(project["id"]) and record them
  worker.submit(project["id"], prepare)
"""

import dataclasses
import hashlib
import json
import queue
import random
import re
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, TypedDict, cast

import httpx
import numpy as np
from comfy_sdk import ComfyError, JobFailed, Progress
from PIL import Image

from rigless import config, describe, pose, selection, sheet, swap, video_prep
from rigless.selection import Anchor

FPS = swap.FPS  # the footage is readied at H3's frame rate, so frames line up
MAX_SIDE = 1280  # enough for the taps and tracking; H3 renders smaller
OUTPUT_SIDE = 1920  # finished takes are pasted onto the footage at up to full HD
POSTER_DELAY = 12  # frames into the trim for the shoot's thumbnail, past any fade-in
RECENT_CHARACTERS = 8
MAX_PEOPLE = len(selection.TINTS)
QUALITIES = {"draft": "480", "full": "768"}  # the page's choice: H3's resolution
# Where each person is in the clip, left to right, by how many people there are.
PLACES = {
    2: ("on the left", "on the right"),
    3: ("on the left", "in the middle", "on the right"),
}
UNSAFE = (
    "Rigless won't render this: the pictures or the clip look like nudity or "
    "sexual content, which MiniMax H3's terms don't allow."
)
_ID = re.compile(r"\d{8}-\d{6}(-\d+)?")
UNREACHABLE = {
    "runpod": "Couldn't reach your RunPod pod. A new pod takes a while to set up",
    "local": "Couldn't reach ComfyUI on this computer. Check that ComfyUI and "
    "comfy-api-proxy are running",
    "custom": "Couldn't reach the ComfyUI set in COMFY_BASE_URL",
}


class Activity(TypedDict):
    """What a project is busy with right now."""

    label: str
    fraction: float | None
    started: str
    eta_seconds: int | None


class TakeState(TypedDict):
    """One take as the page shows it; paths are relative to the project."""

    seed: int
    quality: str  # one of QUALITIES
    status: str
    progress: float
    score: float | None
    frozen: bool
    mirrored: bool
    per_frame: list[float | None] | None
    video: str | None
    raw: str | None
    error: str | None
    started: str | None
    expected_seconds: float | None  # from earlier renders' speed; None until measured
    # The sampler's progress, for the page's estimate: {"steps": n, "first": [step,
    # time], "last": [step, time]}, times as ISO strings; None until it starts.
    sampling: dict[str, Any] | None


class Person(TypedDict):
    """One person to replace, and the character they become."""

    anchors: list[dict[str, Any]]  # the taps per frame, each with its mask overlay
    sheet: list[str]  # the character's images as uploaded
    views: dict[str, str | None] | None  # the front, close-up and back found on them
    # The character's views H3 gets, Picture 1 first: {"id", "file", "role"} each.
    pictures: list[dict[str, str]]
    removed_pictures: list[str]  # ids removed by the user, kept out if found again


class Project(TypedDict):
    """A project's state, as stored in project.json and served to the page.

    Projects saved before a field was added lack it, so read the newer fields
    (title and poster) with get(). Ones saved before people were added are
    brought up to date as they're read.
    """

    id: str
    created: str
    description: str  # the user's notes on the character, added to the prompt
    status: str  # working, waiting (for the user), done, or failed
    step: str  # footage, select, track, render, or results
    activity: Activity | None
    error: str | None
    warnings: list[str]
    inputs: dict[str, Any]  # the uploads: "dance", the footage
    driving: str | None  # the footage readied for taps, tracking, and the monitor
    people: list[Person]  # whom to replace, in the order they were added
    render: dict[str, Any] | None
    takes: list[TakeState]
    best: int | None
    mode: str  # always "swap"; Wan Animate projects from before Oct 25 say "scene"
    trim: list[float] | None  # the part of the footage used, [start, end] seconds
    footage: dict[str, Any] | None  # the readied footage's fps, frames, width, height
    # Per person, in people's order, their masks, lost spans and place across
    # the frame (None if they have no taps in the trim); the preview with
    # everyone tinted, and everyone's lost spans.
    track: dict[str, Any] | None
    title: str  # the footage's file name, to tell shoots apart
    poster: str | None  # a frame of the footage, for the list of shoots


Step = Callable[["Store", str], None]


class Store:
    """The app's projects on disk, safe to use from several threads."""

    def __init__(self, root: Path) -> None:
        """Opens the store, creating its folder if needed.

        Args:
            root: The folder holding one subfolder per project.
        """
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create(
        self, *, description: str = "", trim: tuple[float, float] | None = None
    ) -> Project:
        """Starts a new project, waiting for its first step."""
        with self._lock:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            project_id, count = stamp, 1
            while (self.root / project_id).exists():
                count += 1
                project_id = f"{stamp}-{count}"
            (self.root / project_id / "inputs").mkdir(parents=True)
            project: Project = {
                "id": project_id,
                "created": _now(),
                "description": description,
                "status": "working",
                "step": "footage",
                "activity": activity("Waiting for its turn"),
                "error": None,
                "warnings": [],
                "inputs": {"dance": None},
                "driving": None,
                "people": [_new_person()],
                "render": None,
                "takes": [],
                "best": None,
                "mode": "swap",
                "trim": list(trim) if trim else None,
                "footage": None,
                "track": None,
                "title": "",
                "poster": None,
            }
            self._write(project)
            return project

    def folder(self, project_id: str) -> Path:
        """The project's folder.

        Raises:
            KeyError: No project has that id.
        """
        if not _ID.fullmatch(project_id) or not (self.root / project_id).is_dir():
            raise KeyError(project_id)
        return self.root / project_id

    def get(self, project_id: str) -> Project:
        """Returns a project's current state.

        Raises:
            KeyError: No project has that id.
        """
        with self._lock:
            return self._read(project_id)

    def all(self) -> list[Project]:
        """Returns every project, newest first."""
        with self._lock:
            ids = sorted(
                (path.name for path in self.root.iterdir() if _ID.fullmatch(path.name)),
                reverse=True,
            )
            return [
                self._read(project_id)
                for project_id in ids
                if (self.root / project_id / "project.json").exists()
            ]

    def delete(self, project_id: str) -> None:
        """Deletes a project and all its files.

        Raises:
            KeyError: No project has that id.
        """
        with self._lock:
            shutil.rmtree(self.folder(project_id))

    def update(self, project_id: str, change: Callable[[Project], None]) -> Project:
        """Changes a project's state and saves it.

        Args:
            project_id: The project.
            change: Edits the state in place.

        Returns:
            The new state.
        """
        with self._lock:
            project = self._read(project_id)
            change(project)
            self._write(project)
            return project

    def set(self, project_id: str, **values: Any) -> Project:
        """Sets top-level fields of a project's state and saves it."""

        def change(project: Project) -> None:
            cast(dict[str, Any], project).update(values)

        return self.update(project_id, change)

    def render_speed(self, key: str) -> float | None:
        """Seconds a take took per second of footage, last time with key's setup."""
        with self._lock:
            return self._timing().get(key)

    def record_render_speed(self, key: str, seconds: float) -> None:
        """Remembers seconds per second of footage for a take with key's setup."""
        with self._lock:
            timing = self._timing()
            timing[key] = round(seconds, 1)
            (self.root / "timing.json").write_text(
                json.dumps(timing, indent=2), "utf-8"
            )

    def _timing(self) -> dict[str, float]:
        path = self.root / "timing.json"
        timing: dict[str, float] = (
            json.loads(path.read_text("utf-8")) if path.exists() else {}
        )
        return timing

    def recover(self) -> None:
        """Marks steps that were running when the app last stopped as failed."""
        for project in self.all():
            if project["status"] == "working":
                self.set(
                    project["id"],
                    status="failed",
                    activity=None,
                    error="Rigless was stopped while this was running.",
                )

    def _read(self, project_id: str) -> Project:
        path = self.folder(project_id) / "project.json"
        return _upgrade(json.loads(path.read_text("utf-8")))

    def _write(self, project: Project) -> None:
        path = self.root / project["id"] / "project.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(project, indent=2), "utf-8")
        temporary.replace(path)


class Worker:
    """Runs project steps one at a time, in order, on a background thread."""

    def __init__(self, store: Store) -> None:
        """Starts the worker's thread."""
        self._store = store
        self._queue: queue.Queue[tuple[str, Step]] = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, project_id: str, step: Step) -> None:
        """Queues a step for a project."""
        self._queue.put((project_id, step))

    def _run(self) -> None:
        while True:
            project_id, step = self._queue.get()
            try:
                step(self._store, project_id)
            except Exception as exc:  # a failed step ends the step, never the worker
                self._store.set(
                    project_id, status="failed", activity=None, error=explain(exc)
                )


def _new_person() -> Person:
    return {
        "anchors": [],
        "sheet": [],
        "views": None,
        "pictures": [],
        "removed_pictures": [],
    }


def _upgrade(project: dict[str, Any]) -> Project:
    """Brings a project saved by an earlier Rigless up to date.

    Projects saved before people were added get their one person, and takes
    from before the quality choice were all rendered at full quality.
    """
    for take in project.get("takes", []):
        take.setdefault("quality", "full")
    if "people" not in project:
        project["people"] = [
            {
                "anchors": project.pop("anchors", []),
                "sheet": project["inputs"].pop("sheet", []),
                "views": project.pop("views", None),
                "pictures": project.pop("pictures", []),
                "removed_pictures": project.pop("removed_pictures", []),
            }
        ]
        track = project.get("track")
        if track and "masks" in track:
            person = {"masks": track.pop("masks"), "lost": track["lost"], "x": 0.5}
            track["people"] = [person]
    return cast(Project, project)


def prepare(store: Store, project_id: str) -> None:
    """Readies the whole footage at H3's frame rate for taps, then waits.

    Finds the character on the sheet too, if one came with the footage.
    """
    project = store.get(project_id)
    folder = store.folder(project_id)
    if project["people"][0]["sheet"]:
        _extract_views(store, project_id, 0)
    store.set(project_id, activity=activity("Getting the footage ready"))
    dance = folder / project["inputs"]["dance"]
    footage = folder / "footage.mp4"
    video_prep.conform(
        dance,
        footage,
        start=0.0,
        end=video_prep.probe(dance).duration,
        fps=FPS,
        max_side=MAX_SIDE,
    )
    info = video_prep.probe(footage)
    store.set(
        project_id,
        driving=footage.name,
        footage={
            "fps": info.fps,
            "frames": info.frames,
            "width": info.width,
            "height": info.height,
        },
    )
    start, stop = _trimmed_frames(store.get(project_id))
    video_prep.extract_frame(
        footage, min(start + POSTER_DELAY, stop - 1), folder / "poster.jpg"
    )
    store.set(
        project_id, poster="poster.jpg", status="waiting", step="select", activity=None
    )


def set_character(store: Store, project_id: str, person: int) -> Project:
    """Finds a person's character on their sheet images, already saved in it.

    Raises:
        ValueError: No character was found on the images.
    """
    views = _extract_views(store, project_id, person)
    return store.set(
        project_id, activity=None, warnings=[views.warning] if views.warning else []
    )


def add_person(store: Store, project_id: str) -> Project:
    """Adds another person to replace, with no character or taps yet.

    Raises:
        ValueError: The project already has MAX_PEOPLE people.
    """

    def change(project: Project) -> None:
        if len(project["people"]) == MAX_PEOPLE:
            raise ValueError(f"Rigless replaces up to {MAX_PEOPLE} people at once.")
        project["people"].append(_new_person())
        if project["track"]:
            project["track"]["people"].append(None)

    return store.update(project_id, change)


def remove_person(store: Store, project_id: str, person: int) -> Project:
    """Takes a person out, with their character and taps.

    The people after them move up a place and take that place's tint. The
    tracking is kept while its preview still fits: when the person had no
    masks and nobody tinted moves up. Otherwise it's dropped, and tracking
    again reuses everyone's masks, so it only makes a new preview.

    Raises:
        ValueError: They are the only person.
    """
    folder = store.folder(project_id)

    def change(project: Project) -> None:
        if len(project["people"]) == 1:
            raise ValueError("Keep at least one person to replace.")
        del project["people"][person]
        for index in range(person, len(project["people"])):
            for anchor in project["people"][index]["anchors"]:
                with Image.open(folder / anchor["mask"]) as old:
                    mask = np.asarray(old.getchannel("A")) > 0
                anchor["mask"] = _save_overlay(folder, mask, index, anchor["frame"])
        track = project["track"]
        if track and not any(track["people"][person:]):
            del track["people"][person]
        else:
            project["track"] = None

    return store.update(project_id, change)


def add_anchor(store: Store, project_id: str, person: int, anchor: Anchor) -> Project:
    """Finds the person the anchor's taps point at, and keeps the taps.

    Taps on a frame where this person already has some replace them. Any
    earlier tracking is dropped, since it no longer matches the taps.

    Args:
        store: The projects.
        project_id: The project.
        person: Whose taps these are, by their place in the project's people.
        anchor: The taps.

    Raises:
        ValueError: Nobody was found where the user tapped.
        comfy_sdk.ComfyError: ComfyUI couldn't run the workflow.
    """
    folder = store.folder(project_id)
    work = folder / "selection"
    work.mkdir(exist_ok=True)
    mask = selection.detect(folder / "footage.mp4", anchor, work)
    if not mask.any():
        raise ValueError(
            "Nobody was found there. Tap on the person, not the background."
        )
    record = {
        "frame": anchor.frame,
        "positive": [list(point) for point in anchor.positive],
        "negative": [list(point) for point in anchor.negative],
        "mask": _save_overlay(folder, mask, person, anchor.frame),
    }

    def change(project: Project) -> None:
        taps = project["people"][person]
        kept = [a for a in taps["anchors"] if a["frame"] != anchor.frame]
        taps["anchors"] = sorted([*kept, record], key=lambda a: a["frame"])
        project["track"] = None
        # A new pick is a fresh start: the last step's failure no longer applies.
        if project["status"] == "failed":
            project["status"], project["error"] = "waiting", None

    return store.update(project_id, change)


def remove_anchor(store: Store, project_id: str, person: int, frame: int) -> Project:
    """Drops a person's taps on one frame, and any tracking made with them."""

    def change(project: Project) -> None:
        taps = project["people"][person]
        taps["anchors"] = [a for a in taps["anchors"] if a["frame"] != frame]
        project["track"] = None

    return store.update(project_id, change)


def _save_overlay(folder: Path, mask: np.ndarray, person: int, frame: int) -> str:
    """Saves a person's mask on a tapped frame, in their tint, for the page."""
    # A new name each time, so the page never shows a cached overlay.
    name = f"selection/anchor-{frame}-{time.time_ns()}.png"
    selection.overlay(mask, selection.TINTS[person]).save(folder / name)
    return name


def recent_characters(store: Store, current: str | None = None) -> list[dict[str, Any]]:
    """Returns the characters of recent projects, newest first, each once.

    Each is {"project": id, "front": the front view, "sheet": the images},
    with paths inside that project, so the page can offer them again.

    Args:
        store: The projects.
        current: A project whose own character is left out, since it's in use.
    """
    seen = set()
    if current:
        seen = {
            _character_digest(store.folder(current), person)
            for person in store.get(current)["people"]
        }
    recent: list[dict[str, Any]] = []
    for project in store.all():
        for person in project["people"]:
            views = person["views"] or {}
            digest = _character_digest(store.folder(project["id"]), person)
            if not views.get("front") or not digest or digest in seen:
                continue
            seen.add(digest)
            recent.append(
                {
                    "project": project["id"],
                    "front": views["front"],
                    "sheet": person["sheet"],
                }
            )
            if len(recent) == RECENT_CHARACTERS:
                return recent
    return recent


def _character_digest(folder: Path, person: Person) -> str:
    """A fingerprint of a person's character images; empty if any are missing."""
    files = [folder / path for path in person["sheet"]]
    if not files or not all(file.is_file() for file in files):
        return ""
    digest = hashlib.sha1()
    for file in files:
        digest.update(file.read_bytes())
    return digest.hexdigest()


def track(store: Store, project_id: str) -> None:
    """Follows the tapped people through the trimmed part, then waits for the user.

    Saves each person's masks, a preview of the part with everyone tinted, and
    the stretches where someone was lost, so the user can tap them again there.
    Taps outside the trimmed part are kept but not used.

    Raises:
        ValueError: No taps lie in the trimmed part.
    """
    _track(store, project_id)
    store.set(project_id, status="waiting", step="select", activity=None)


def _track(store: Store, project_id: str) -> None:
    """Tracks each tapped person and records the masks, preview, and lost spans.

    A person whose taps haven't changed since they were last tracked keeps
    their masks, and the preview is made again only if they all differ from
    when it was made.
    """
    project = store.get(project_id)
    folder = store.folder(project_id)
    work = folder / "selection"
    work.mkdir(exist_ok=True)
    start, stop = _trimmed_frames(project)
    taps = [
        [
            Anchor(
                frame=a["frame"] - start,
                positive=tuple((x, y) for x, y in a["positive"]),
                negative=tuple((x, y) for x, y in a["negative"]),
            )
            for a in person["anchors"]
            if start <= a["frame"] < stop
        ]
        for person in project["people"]
    ]
    if not any(taps):
        raise ValueError("Tap the person somewhere in the trimmed part.")
    paths = [
        _masks_file(work, start, stop, anchors) if anchors else None for anchors in taps
    ]
    # Named by what it shows, so the page never plays a stale or half-written one.
    preview = work / f"preview-{_digest([path and path.name for path in paths])}.mp4"
    following = any(path and not path.exists() for path in paths)
    store.set(
        project_id,
        activity=activity(
            "Following them through the clip" if following else "Making the preview"
        ),
    )
    part = work / "part.mp4"  # the render reads it too
    video_prep.cut_frames(folder / "footage.mp4", part, start, stop)
    rgb = video_prep.read_frames(part)[..., ::-1]
    found: list[np.ndarray | None] = []
    people: list[dict[str, Any] | None] = []
    for anchors, path in zip(taps, paths, strict=True):
        if path is None:
            found.append(None)
            people.append(None)
            continue
        if path.exists():
            masks = video_prep.read_frames(path)[..., 0] > 127
        else:
            masks = _follow(store, project_id, part, rgb, anchors)
            video_prep.write_video(masks.astype(np.uint8) * 255, FPS, path)
        found.append(masks)
        lost = [
            [round(start / FPS + a, 2), round(start / FPS + b, 2)]
            for a, b in selection.lost_spans(masks, FPS)
        ]
        people.append(
            {
                "masks": path.relative_to(folder).as_posix(),
                "lost": lost,
                "x": _middle(masks),
            }
        )
    if not preview.exists():
        if following:
            store.set(project_id, activity=activity("Making the preview"))
        unfinished = work / "preview-unfinished.mp4"
        video_prep.write_video(selection.tint(rgb, found), FPS, unfinished)
        for old in work.glob("preview*.mp4"):
            if old != unfinished:
                old.unlink()
        unfinished.replace(preview)
    store.set(
        project_id,
        track={
            "people": people,
            "preview": preview.relative_to(folder).as_posix(),
            "start": start / FPS,
            "lost": sorted(
                span for person in people if person for span in person["lost"]
            ),
        },
    )


def _follow(
    store: Store, project_id: str, part: Path, rgb: np.ndarray, anchors: list[Anchor]
) -> np.ndarray:
    """Tracks one person from their taps, and again with more if the mask misses them.

    Args:
        store: The projects.
        project_id: The project, whose activity says what's happening.
        part: The trimmed part.
        rgb: The part's frames, RGB.
        anchors: The person's taps, on the part's frames.

    Returns:
        Their (frames, height, width) bool masks.
    """
    work = store.folder(project_id) / "selection"
    masks = selection.track(part, anchors, work)
    store.set(project_id, activity=activity("Checking the mask"))
    retaps = selection.head_taps(pose.estimate_in(part, pose.mask_boxes(masks)), masks)
    for anchor in anchors:
        body = selection.body_taps(
            pose.find_people(rgb[anchor.frame]), masks[anchor.frame]
        )
        if body:
            retaps.append(Anchor(anchor.frame, body))
    if not retaps:
        return masks
    store.set(project_id, activity=activity("Finding all of them again"))
    retracked = selection.track(part, selection.merge(anchors, retaps), work)
    # Taps meant to add the head or body can still land SAM on something
    # smaller, so the new masks are kept only if they don't shrink from there on.
    first = min(tap.frame for tap in retaps)
    if np.median(retracked[first:].sum(axis=(1, 2))) >= np.median(
        masks[first:].sum(axis=(1, 2))
    ):
        return retracked
    return masks


def _masks_file(work: Path, start: int, stop: int, anchors: list[Anchor]) -> Path:
    """Where a person's masks from these taps on this part are kept."""
    key = [start, stop, [dataclasses.astuple(a) for a in anchors]]
    return work / f"person-{_digest(key)}.mp4"


def _digest(value: Any) -> str:
    """A short fingerprint of a JSON-able value, for file names."""
    return hashlib.sha1(json.dumps(value).encode()).hexdigest()[:12]


def _middle(masks: np.ndarray) -> float:
    """Where a person is across the frame on average, from 0 (left) to 1 (right)."""
    columns = masks.sum(axis=(0, 1), dtype=np.int64)
    return float(
        (columns * np.arange(len(columns))).sum() / columns.sum() / len(columns)
    )


def allot(counts: list[int], budget: int) -> list[int]:
    """Deals out H3's pictures among people, one each in turn, as app.js shows it.

    Args:
        counts: How many pictures each person has.
        budget: How many H3 takes in all.

    Returns:
        How many of each person's pictures, from their first, H3 gets.
    """
    given = [0] * len(counts)
    while budget and any(had < count for had, count in zip(given, counts, strict=True)):
        for index, count in enumerate(counts):
            if budget and given[index] < count:
                given[index] += 1
                budget -= 1
    return given


def places(middles: list[float]) -> list[str]:
    """Words for where each person is, such as "on the left", for H3's prompt.

    Args:
        middles: Each person's place across the frame, from 0 (left) to 1, for
            two to MAX_PEOPLE people.
    """
    words = PLACES[len(middles)]
    order = sorted(range(len(middles)), key=lambda index: middles[index])
    named = [""] * len(middles)
    for word, index in zip(words, order, strict=True):
        named[index] = word
    return named


def render(store: Store, project_id: str, options: Mapping[str, Any]) -> None:
    """Swaps the tapped people for their characters, scores the takes, picks the best.

    The people are tracked first if the latest taps haven't been. The new takes
    join the project's earlier finished ones (failed ones are dropped), and the
    best is picked from them all.

    Args:
        store: The projects.
        project_id: A project with footage, a character, and taps.
        options: The render request: "takes", and optionally "seeds" and
            "quality" (one of QUALITIES; "full" if not given).
    """
    if store.get(project_id)["track"] is None:
        _track(store, project_id)
    tracked = cast(dict[str, Any], store.get(project_id)["track"])
    for number, person in enumerate(tracked["people"], start=1):
        if person is None:
            raise ValueError(
                _who(number, tracked) + " has no taps in the trimmed part."
            )
    _refuse_unsafe(store, project_id)
    project = store.get(project_id)
    folder = store.folder(project_id)
    seeds = list(options.get("seeds") or [])
    seeds += [random.randrange(2**53) for _ in range(options["takes"] - len(seeds))]
    quality = options.get("quality", "full")
    resolution = QUALITIES[quality]
    start, stop = _trimmed_frames(project)
    seconds = (stop - start) / FPS
    # Render time grows with the clip, so the speed is kept per second of footage.
    speed_key = f"swap|{swap.MODEL}|{swap.SWAP_LORA}|{resolution}"
    earlier = [take for take in project["takes"] if take["status"] == "done"]
    first = len(earlier)  # where this render's takes start in the project's
    store.set(
        project_id,
        status="working",
        step="render",
        error=None,
        best=_best(earlier),
        render={"takes": len(seeds), "seeds": seeds, "quality": quality},
        takes=earlier + [_queued_take(seed, quality) for seed in seeds],
        activity=activity("Rendering", fraction=0.0),
    )
    started: dict[int, float] = {}
    steps_seen: dict[int, int] = {}

    def progress(index: int, event: Progress) -> None:
        if index not in started:
            started[index] = time.monotonic()
            per_second = store.render_speed(speed_key)
            expected = per_second * seconds if per_second else None
            store.update(
                project_id,
                partial(
                    _start_take,
                    index=first + index,
                    number=index + 1,
                    count=len(seeds),
                    expected=expected,
                ),
            )
        # ComfyUI reports each step more than once; only a new one is written.
        step = event.step or 0
        if (
            event.current_node == swap.SAMPLER
            and event.steps
            and step > steps_seen.get(index, 0)
        ):
            steps_seen[index] = step
            store.update(
                project_id,
                partial(_note_step, index=first + index, step=step, steps=event.steps),
            )

    people = project["people"]
    shares = allot([len(person["pictures"]) for person in people], swap.MAX_PICTURES)
    pictures = [
        _picture_order(folder, person["pictures"][:share], folder / found["masks"])
        for person, share, found in zip(people, shares, tracked["people"], strict=True)
    ]
    prompt = _prompt(store, project_id, pictures)
    store.set(project_id, activity=activity("Getting the full-size footage ready"))
    full = _full_size_part(folder, project, start, stop)
    store.set(project_id, activity=activity("Rendering", fraction=0.0))
    for index, take in enumerate(
        swap.render_takes(
            folder / "selection" / "part.mp4",
            [folder / person["masks"] for person in tracked["people"]],
            [folder / picture["file"] for each in pictures for picture in each],
            prompt=prompt,
            seeds=seeds,
            out_dir=folder / f"render-{datetime.now():%H%M%S}",
            sound=(folder / project["inputs"]["dance"], start / FPS),
            full=full,
            resolution=resolution,
            on_progress=progress,
        )
    ):
        if not take.error and index in started:
            store.record_render_speed(
                speed_key, (time.monotonic() - started[index]) / seconds
            )
        store.update(
            project_id,
            partial(_finish_take, index=first + index, take=take, folder=folder),
        )
    best = _best(store.get(project_id)["takes"])
    store.set(
        project_id,
        status="done" if best is not None else "failed",
        step="results",
        activity=None,
        best=best,
        error=None if best is not None else "Every take failed; see the takes for why.",
    )


def _best(takes: list[TakeState]) -> int | None:
    """The seed of the take to show first, or None if no take is done.

    Like swap.pick_best, takes flagged frozen or mirrored come last; then full
    quality comes before drafts, and the higher score before the lower.
    """
    done = [
        take for take in takes if take["status"] == "done" and take["score"] is not None
    ]
    if not done:
        return None
    return max(
        done,
        key=lambda take: (
            not (take["frozen"] or take["mirrored"]),
            take["quality"] == "full",
            take["score"],
        ),
    )["seed"]


def _full_size_part(folder: Path, project: Project, start: int, stop: int) -> Path:
    """The trimmed part at up to OUTPUT_SIDE, frame for frame with the tracked one.

    The footage at full size is made once per shoot, the same way as the
    footage itself, so its frames line up with the ones that were tracked.
    """
    footage = folder / "footage-full.mp4"
    if not footage.exists():
        dance = folder / project["inputs"]["dance"]
        making = folder / "footage-full.partial.mp4"
        video_prep.conform(
            dance,
            making,
            start=0.0,
            end=video_prep.probe(dance).duration,
            fps=FPS,
            max_side=OUTPUT_SIDE,
        )
        making.replace(footage)
    part = folder / "selection" / "part-full.mp4"
    video_prep.cut_frames(footage, part, start, stop)
    return part


def _refuse_unsafe(store: Store, project_id: str) -> None:
    """Stops the render if the pictures or the clip look unsafe for work.

    Raises:
        ValueError: They do.
    """
    project = store.get(project_id)
    folder = store.folder(project_id)
    store.set(project_id, activity=activity("Checking the pictures and the clip"))
    work = folder / "safety"
    work.mkdir(exist_ok=True)
    part = folder / "selection" / "part.mp4"
    last = video_prep.probe(part).frames - 1
    frames = []
    for index in (0, last // 2, last):
        video_prep.extract_frame(part, index, work / f"frame-{index}.png")
        frames.append(Image.open(work / f"frame-{index}.png"))
    pictures = [
        Image.open(folder / picture["file"])
        for person in project["people"]
        for picture in person["pictures"]
    ]
    if describe.unsafe(pictures, work) or describe.unsafe(frames, work):
        raise ValueError(UNSAFE)


def _picture_order(
    folder: Path, pictures: list[dict[str, str]], masks: Path
) -> list[dict[str, str]]:
    """The character's pictures in the order H3 gets them.

    H3 takes the person's opening look from <Picture 1>: given the front view
    for a clip that opens from behind, it drew them facing the camera. So the
    back view, if there is one, goes first when the clip opens that way.
    """
    if not any(picture["role"] == "back" for picture in pictures):
        return pictures
    opening = video_prep.read_frames(masks)[: round(FPS), ..., 0] > 127
    track = pose.estimate_in(
        folder / "selection" / "part.mp4", pose.mask_boxes(opening)
    )
    if not pose.starts_facing_away(track):
        return pictures
    return sorted(pictures, key=lambda picture: picture["role"] != "back")


def _prompt(store: Store, project_id: str, pictures: list[list[dict[str, str]]]) -> str:
    """Writes the swap prompt with each character described in words.

    The words also name what each tapped person has that their character
    doesn't, and, with several people, what each wears in the clip, to tell
    them apart. Words that can't be written are left out, with a warning on
    the project.

    Args:
        store: The projects.
        project_id: The project to render.
        pictures: Each person's pictures, in the order H3 gets them.
    """
    project = store.get(project_id)
    folder = store.folder(project_id)
    start, stop = _trimmed_frames(project)
    store.set(project_id, activity=activity("Describing the character"))
    castings = []
    for number, (person, theirs) in enumerate(
        zip(project["people"], pictures, strict=True), start=1
    ):
        paths = [folder / picture["file"] for picture in theirs]
        anchor = next(a for a in person["anchors"] if start <= a["frame"] < stop)
        work = folder / "describe" / f"person-{number}"
        try:
            performer = _tapped_person(folder, anchor)
            description = describe.character(paths, work)
            unlike = describe.differences(paths, performer, work)
            look = describe.look(performer, work) if len(pictures) > 1 else ""
        except Exception:  # without the words, the takes are still worth rendering
            description, unlike, look = "", [], ""
            warning = (
                "The character couldn't be described automatically, so small "
                "details may come out less exact."
            )

            def warn(state: Project, warning: str = warning) -> None:
                if warning not in state["warnings"]:
                    state["warnings"] = [*state["warnings"], warning]

            store.update(project_id, warn)
        castings.append(
            swap.Casting(
                where="",
                roles=tuple(picture["role"] for picture in theirs),
                look=look,
                description=description,
                unlike=tuple(unlike),
            )
        )
    if len(castings) == 1:
        (only,) = castings
        return swap.prompt(
            only.roles,
            project["description"],
            description=only.description,
            unlike=only.unlike,
        )
    track = cast(dict[str, Any], project["track"])
    where = places([person["x"] for person in track["people"]])
    return swap.group_prompt(
        [
            dataclasses.replace(casting, where=words)
            for casting, words in zip(castings, where, strict=True)
        ],
        project["description"],
    )


def _who(number: int, track: dict[str, Any]) -> str:
    """Names a person for a message: "Person 2", or "The person" if alone."""
    return f"Person {number}" if len(track["people"]) > 1 else "The person"


def _tapped_person(folder: Path, anchor: dict[str, Any]) -> Image.Image:
    """The tapped person on their anchor's frame, with a little room around them."""
    frame = folder / "selection" / f"frame-{anchor['frame']}.png"
    video_prep.extract_frame(folder / "footage.mp4", anchor["frame"], frame)
    with Image.open(folder / anchor["mask"]) as overlay:
        ys, xs = np.nonzero(np.asarray(overlay.getchannel("A")))
    margin_x, margin_y = (xs.max() - xs.min()) // 10, (ys.max() - ys.min()) // 10
    with Image.open(frame) as image:
        return image.convert("RGB").crop(
            (
                max(0, xs.min() - margin_x),
                max(0, ys.min() - margin_y),
                min(image.width, xs.max() + margin_x),
                min(image.height, ys.max() + margin_y),
            )
        )


def _extract_views(store: Store, project_id: str, person: int) -> sheet.Views:
    """Finds a person's character on their sheet and saves the views in the project.

    The pictures H3 gets are the front view, the face close-up, the back view,
    then any other views, at most swap.MAX_PICTURES, leaving out the ones the
    user removed.
    """
    project = store.get(project_id)
    folder = store.folder(project_id)
    theirs = project["people"][person]
    store.set(project_id, activity=activity("Finding the character on the sheet"))
    images = [Image.open(folder / path).convert("RGB") for path in theirs["sheet"]]
    views = sheet.extract(images)
    (folder / "views").mkdir(exist_ok=True)
    saved: dict[str, str | None] = {}
    for name, image in (
        ("front", views.front),
        ("close_up", views.close_up),
        ("back", views.back),
    ):
        saved[name] = None
        if image is not None:
            # Named by content, so several people's views never overwrite each other.
            path = f"views/{name}-{_image_id(image)}.png"
            image.save(folder / path)
            saved[name] = path
    removed = set(theirs["removed_pictures"])
    pictures: list[dict[str, str]] = []
    candidates = [
        ("front", views.front),
        ("face", views.close_up),
        ("back", views.back),
        *(("view", image) for image in views.others),
    ]
    for role, image in candidates:
        if image is None:
            continue
        # The same image always gets the same id, so a removed one stays removed.
        picture_id = _image_id(image)
        if picture_id in removed or len(pictures) == swap.MAX_PICTURES:
            continue
        name = f"views/picture-{picture_id}.png"
        image.save(folder / name)
        pictures.append({"id": picture_id, "file": name, "role": role})

    def change(project: Project) -> None:
        project["people"][person]["views"] = saved
        project["people"][person]["pictures"] = pictures

    store.update(project_id, change)
    return views


def _image_id(image: Image.Image) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()[:12]


def remove_picture(
    store: Store, project_id: str, person: int, picture_id: str
) -> Project:
    """Takes one picture out of the ones H3 gets for a person, for good.

    Raises:
        KeyError: The person has no such picture.
        ValueError: It is the last picture of their character.
    """

    def change(project: Project) -> None:
        theirs = project["people"][person]
        kept = [
            picture for picture in theirs["pictures"] if picture["id"] != picture_id
        ]
        if len(kept) == len(theirs["pictures"]):
            raise KeyError(picture_id)
        if not kept:
            raise ValueError("Keep at least one picture of the character.")
        theirs["pictures"] = kept
        theirs["removed_pictures"] = [*theirs["removed_pictures"], picture_id]

    return store.update(project_id, change)


def explain(exc: Exception) -> str:
    """Puts a failure into words for the page."""
    if isinstance(exc, JobFailed):
        return f"ComfyUI couldn't finish the job: {exc.error or exc}"
    if isinstance(exc, ComfyError | ConnectionError | httpx.HTTPError):
        reason = UNREACHABLE[config.connection().target]
        # A refused connection says nothing more than the reason itself.
        return (
            f"{reason}."
            if isinstance(exc, httpx.ConnectError)
            else f"{reason} ({exc})."
        )
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Something went wrong: {exc}"


def activity(
    label: str, fraction: float | None = None, started: str | None = None
) -> Activity:
    """Returns a new activity, started now unless started is given.

    Args:
        label: What is being done, for the page.
        fraction: How much of it is done, 0 to 1, or None if unknown.
        started: When it started, as an ISO time; None means now.
    """
    return {
        "label": label,
        "fraction": fraction,
        "started": started or _now(),
        "eta_seconds": None,
    }


def _start_take(
    project: Project, *, index: int, number: int, count: int, expected: float | None
) -> None:
    """Marks a take as rendering; the page estimates its progress from expected.

    Args:
        project: The project.
        index: The take's place in the project's takes.
        number: The take's place in this render's, from 1.
        count: How many takes this render makes.
        expected: How long the take should take, in seconds, if known.
    """
    take = project["takes"][index]
    take["status"] = "rendering"
    take["started"] = _now()
    take["expected_seconds"] = expected
    if project["activity"] is not None:
        project["activity"]["label"] = (
            f"Rendering take {number} of {count}" if count > 1 else "Rendering"
        )


def _note_step(project: Project, *, index: int, step: int, steps: int) -> None:
    """Records that a take's sampler finished a step, and when."""
    take = project["takes"][index]
    mark = [step, _now()]
    if take.get("sampling") is None:
        take["sampling"] = {"steps": steps, "first": mark, "last": mark}
    else:
        cast(dict[str, Any], take["sampling"])["last"] = mark


def _now() -> str:
    """The time now in UTC, as ISO 8601 with a Z, which browsers read correctly."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _queued_take(seed: int, quality: str) -> TakeState:
    return {
        "seed": seed,
        "quality": quality,
        "status": "queued",
        "progress": 0.0,
        "score": None,
        "frozen": False,
        "mirrored": False,
        "per_frame": None,
        "video": None,
        "raw": None,
        "error": None,
        "started": None,
        "expected_seconds": None,
        "sampling": None,
    }


def _finish_take(project: Project, index: int, take: swap.Take, folder: Path) -> None:
    """Records a finished take in the project's state."""
    state = project["takes"][index]
    state["seed"] = take.seed
    state["progress"] = 1.0
    state["status"] = "failed" if take.error else "done"
    state["error"] = take.error
    if take.score is not None:
        record = take.score.as_record()
        state["score"] = cast(float, record["score"])
        state["frozen"] = take.score.frozen
        state["mirrored"] = take.score.mirrored
        state["per_frame"] = cast(list[float | None], record["per_frame"])
        state["video"] = take.finished.relative_to(folder).as_posix()
        state["raw"] = take.result.relative_to(folder).as_posix()


def _trimmed_frames(project: Project) -> tuple[int, int]:
    """The trimmed part of the footage, as [start, stop) frames."""
    frames = cast(dict[str, Any], project["footage"])["frames"]
    start, end = project["trim"] or (0.0, frames / FPS)
    return round(start * FPS), min(frames, round(end * FPS))
