"""The Rigless web app: a FastAPI server for the page in `web/`.

`uv run rigless` starts it on this computer and opens the browser. The page
is one screen: it uploads the footage as soon as it's chosen, then the
character, the trim, and the user's taps on the person to replace, and
follows the project by polling its state; the work happens in
`rigless.projects`. A shoot can replace several people, each with their
own character and taps; requests about one of them name it by its place
(`person`, from 0). It only answers this computer, and only the page
itself may change anything.

Typical usage example:

  uv run rigless
  uv run rigless --port 8190 --no-browser
"""

import argparse
import logging
import re
import secrets
import shutil
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, FiniteFloat
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rigless import comfy_client, config, projects, runpod, terms, video_prep
from rigless.projects import Person, Project, Store, Worker
from rigless.selection import Anchor

logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"
DEFAULT_PORT = 8190  # next to ComfyUI's 8188 and comfy-api-proxy's 8189
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
SHOOT_TITLE = re.compile(r"Shoot (\d+)")
MAX_FOOTAGE_SECONDS = 120  # trimmed in the app to at most video_prep.MAX_TRIM_SECONDS
LOCAL_HOSTS = ["127.0.0.1", "localhost"]
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# The page runs only its own script, and no other site may frame it: a framed
# page's clicks would come from this origin and pass the Origin check.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' blob: data:; media-src 'self' blob:; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
}
# RunPod or ComfyUI refused a request. Not 502: Cloudflare, in front of quick
# tunnels and RunPod's proxy, swaps a 502's body for its own page, reason and all.
FAILED_DEPENDENCY = 424
CONNECTED = {
    "runpod": "Connected to your RunPod pod.",
    "local": "Connected to ComfyUI on this computer.",
    "custom": "Connected to the ComfyUI set in COMFY_BASE_URL.",
}


class ConnectionChoice(BaseModel):
    """Where to send Rigless's work: a RunPod pod or ComfyUI on this computer."""

    target: config.Choice


class ProjectChange(BaseModel):
    """Changes to a project; None leaves a setting as it is."""

    trim: tuple[FiniteFloat, FiniteFloat] | None = None
    description: str | None = None
    title: str | None = Field(None, max_length=80)


class AnchorRequest(BaseModel):
    """Taps on one frame of the footage, in its pixels, for one person."""

    person: int = Field(0, ge=0)
    frame: int = Field(ge=0)
    positive: list[tuple[int, int]] = Field(min_length=1, max_length=24)
    negative: list[tuple[int, int]] = Field(default_factory=list, max_length=24)


class PodStart(BaseModel):
    """A request to start Rigless's pod on a GPU type from RunPod's list."""

    gpu: str = Field(min_length=1, max_length=120)


class RenderRequest(BaseModel):
    """A request to render takes, optionally with some of their seeds.

    Attributes:
        takes: How many takes to render.
        seeds: Seeds for some or all of the takes.
        quality: "draft" (480p, faster) or "full" (768p).
    """

    takes: int = Field(1, ge=1, le=5)
    # The page matches takes by seed in JavaScript, which counts exactly to 2**53.
    seeds: list[Annotated[int, Field(ge=0, lt=2**53)]] | None = Field(
        None, max_length=5
    )
    quality: Literal["draft", "full"] = "full"


def create_app(root: Path) -> FastAPI:
    """Builds the app, with its projects in root.

    Projects left running when the app last stopped are marked failed.
    """
    store = Store(root)
    store.recover()
    worker = Worker(store)
    deploying = threading.Lock()
    app = FastAPI(title="Rigless", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)

    @app.middleware("http")
    async def page_only(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # A page on another site can still make the browser send requests here,
        # as a form post naming that site as its Origin. Reading the answers
        # takes DNS rebinding, which TrustedHostMiddleware refuses.
        path = request.url.path
        origin = request.headers.get("origin")
        if (
            request.method not in SAFE_METHODS
            and origin is not None
            and urlsplit(origin).hostname not in LOCAL_HOSTS
        ):
            return JSONResponse(
                {"detail": "Not from this app's page."}, status_code=403
            )
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        if not path.startswith(("/api/", "/files/")):
            # The page's own files change with every update; browsers check them
            # each time (a cheap 304 when unchanged) instead of running stale ones.
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/connection")
    def get_connection() -> dict[str, Any]:
        return _connection()

    @app.put("/api/connection")
    def choose_connection(body: ConnectionChoice) -> dict[str, Any]:
        if config.connection().target == "custom":
            raise HTTPException(409, "COMFY_BASE_URL in .env sets the connection.")
        config.choose(body.target)
        return _connection()

    @app.get("/api/connection/status")
    def connection_status() -> dict[str, Any]:
        found = _connection()
        try:
            with comfy_client.connect() as client:
                # list_jobs is lazy: only reading from it asks the server.
                next(client.list_jobs(limit=1), None)
        except Exception as exc:  # any failure is the answer the page shows
            return {**found, "ok": False, "message": projects.explain(exc)}
        return {**found, "ok": True, "message": CONNECTED[found["target"]]}

    @app.get("/api/terms")
    def get_terms() -> dict[str, Any]:
        return {
            "version": terms.VERSION,
            "notice": terms.NOTICE,
            "license_url": terms.LICENSE_URL,
            "authorization_url": terms.AUTHORIZATION_URL,
            "report_email": terms.REPORT_EMAIL,
            "accepted": terms.accepted(root),
        }

    @app.post("/api/terms", status_code=204)
    def accept_terms() -> None:
        terms.accept(root)

    @app.get("/api/characters")
    def recent_characters(project: str | None = None) -> list[dict[str, Any]]:
        if project is not None:
            _project(store, project)
        return projects.recent_characters(store, project)

    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return [_summary(project) for project in store.all()]

    @app.post("/api/projects", status_code=201)
    def create_project(
        dance: Annotated[UploadFile, File()],
        sheet: Annotated[list[UploadFile] | None, File()] = None,
        description: Annotated[str, Form()] = "",
        start: Annotated[float, Form(ge=0)] = 0.0,
        end: Annotated[float | None, Form(gt=0)] = None,
    ) -> dict[str, str]:
        """Creates a shoot from its footage and readies it in the background.

        Without an end, the trim runs MAX_TRIM_SECONDS from start. A file that
        isn't a readable video or picture is a 400, and the shoot is removed.
        """
        title = _next_title(store)
        project = store.create(description=description.strip())
        folder = store.folder(project["id"])
        try:
            dance_path, duration = _save_video(dance, folder, "dance")
            if duration > MAX_FOOTAGE_SECONDS:
                raise ValueError(
                    f"Use a clip up to {MAX_FOOTAGE_SECONDS // 60} minutes long. "
                    f"This one is {duration:.0f} seconds."
                )
            if end is None:
                end = start + video_prep.MAX_TRIM_SECONDS
            trim = video_prep.check_trim(start, end, duration)
            sheet_paths = [
                _save_image(image, folder, f"sheet-{index}")
                for index, image in enumerate(sheet or [])
            ]
        except ValueError as exc:
            shutil.rmtree(folder)
            raise HTTPException(400, str(exc)) from exc

        def change(state: Project) -> None:
            state["inputs"]["dance"] = dance_path
            state["people"][0]["sheet"] = sheet_paths
            state["trim"] = list(trim)
            state["title"] = title

        store.update(project["id"], change)
        worker.submit(project["id"], projects.prepare)
        return {"id": project["id"]}

    @app.delete("/api/projects/{project_id}", status_code=204)
    def delete_project(project_id: str) -> None:
        _idle_project(store, project_id)
        store.delete(project_id)

    @app.patch("/api/projects/{project_id}", response_model=None)
    def change_project(project_id: str, body: ProjectChange) -> Project:
        """Renames a shoot or changes its prompt or trim; new trims drop tracking."""
        project = _project(store, project_id)
        values: dict[str, Any] = {}
        if body.title is not None:
            if not body.title.strip():
                raise HTTPException(400, "Give the shoot a name.")
            values["title"] = body.title.strip()
        if body.description is not None:
            values["description"] = body.description.strip()
        if body.trim is not None and list(body.trim) != project.get("trim"):
            footage = project.get("footage")
            if project["status"] == "working" or not footage:
                raise HTTPException(409, "Wait for the footage to finish its step.")
            try:
                trim = video_prep.check_trim(
                    *body.trim, footage["frames"] / footage["fps"]
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            values.update(trim=list(trim), track=None)
        return store.set(project_id, **values)

    @app.post("/api/projects/{project_id}/people", response_model=None)
    def add_person(project_id: str) -> Project:
        _idle_project(store, project_id)
        try:
            return projects.add_person(store, project_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/projects/{project_id}/people/{person}", response_model=None)
    def remove_person(project_id: str, person: int) -> Project:
        _person(_idle_project(store, project_id), person)
        try:
            return projects.remove_person(store, project_id, person)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/projects/{project_id}/character", response_model=None)
    def set_character(
        project_id: str,
        sheet: Annotated[list[UploadFile], File()],
        # Text, not bool: FastAPI reads a missing bool form field as False.
        append: Annotated[str, Form()] = "false",
        person: Annotated[int, Form()] = 0,
    ) -> Project:
        """Gives a person new character pictures, or more of them when appending.

        If the new pictures can't be used, the person keeps the old ones.
        """
        theirs = _person(_project(store, project_id), person)
        folder = store.folder(project_id)
        adding = append.lower() == "true"
        new: list[str] = []

        def use(paths: list[str], removed: list[str]) -> None:
            def change(state: Project) -> None:
                state["people"][person]["sheet"] = paths
                state["people"][person]["removed_pictures"] = removed

            store.update(project_id, change)

        # The old pictures go only once the new ones have worked.
        try:
            for image in sheet:
                # Random names, so several people's uploads never overwrite each other.
                new.append(_save_image(image, folder, f"sheet-{secrets.token_hex(6)}"))
            kept = theirs["sheet"] if adding else []
            use(kept + new, theirs["removed_pictures"] if adding else [])
            project = projects.set_character(store, project_id, person, added=new)
        except ValueError as exc:
            for path in new:
                (folder / path).unlink(missing_ok=True)
            use(theirs["sheet"], theirs["removed_pictures"])
            raise HTTPException(400, str(exc)) from exc
        if not adding:
            for old in theirs["sheet"]:
                (folder / old).unlink(missing_ok=True)
        return project

    @app.delete("/api/projects/{project_id}/pictures/{picture_id}", response_model=None)
    def remove_picture(project_id: str, picture_id: str, person: int = 0) -> Project:
        _person(_project(store, project_id), person)
        try:
            return projects.remove_picture(store, project_id, person, picture_id)
        except KeyError as exc:
            raise HTTPException(404, "No such picture.") from exc

    # Projects are returned as stored, not re-validated: one saved before a field
    # was added would otherwise fail to load.
    @app.get("/api/projects/{project_id}", response_model=None)
    def get_project(project_id: str) -> Project:
        return _project(store, project_id)

    @app.post("/api/projects/{project_id}/anchors", response_model=None)
    def add_anchor(project_id: str, body: AnchorRequest) -> Project:
        """Picks a person on one frame; SAM finds them there before it answers."""
        project = _idle_project(store, project_id)
        _person(project, body.person)
        footage = project["footage"]
        if not footage:
            raise HTTPException(400, "This shoot has no footage to tap on yet.")
        if body.frame >= footage["frames"]:
            raise HTTPException(400, "That frame is past the end of the footage.")
        for x, y in body.positive + body.negative:
            if not (0 <= x < footage["width"] and 0 <= y < footage["height"]):
                raise HTTPException(400, "A tap lies outside the picture.")
        anchor = Anchor(
            frame=body.frame,
            positive=tuple(body.positive),
            negative=tuple(body.negative),
        )
        try:
            return projects.add_anchor(store, project_id, body.person, anchor)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # ComfyUI's failures, in words the page can show
            logger.warning("Tap failed: %r", exc)
            raise HTTPException(FAILED_DEPENDENCY, projects.explain(exc)) from exc

    @app.delete("/api/projects/{project_id}/anchors/{frame}", response_model=None)
    def remove_anchor(project_id: str, frame: int, person: int = 0) -> Project:
        _person(_idle_project(store, project_id), person)
        return projects.remove_anchors(store, project_id, person, frame)

    @app.delete("/api/projects/{project_id}/anchors", response_model=None)
    def clear_anchors(project_id: str, person: int = 0) -> Project:
        _person(_idle_project(store, project_id), person)
        return projects.remove_anchors(store, project_id, person)

    @app.post("/api/projects/{project_id}/track", status_code=202, response_model=None)
    def track(project_id: str) -> Project:
        project = _idle_project(store, project_id)
        if not any(person["anchors"] for person in project["people"]):
            raise HTTPException(400, "Tap the person to replace first.")
        # Marked working before it's queued, so a step that fails at once isn't
        # overwritten as working, stuck for good.
        working = store.set(
            project_id,
            status="working",
            step="track",
            error=None,
            activity=projects.activity("Waiting for its turn"),
        )
        worker.submit(project_id, projects.track)
        return working

    @app.post("/api/projects/{project_id}/render", status_code=202, response_model=None)
    def render(project_id: str, body: RenderRequest) -> Project:
        """Queues new takes once every person has a character and taps.

        The takes are added to the shoot's earlier ones; the terms must be agreed.
        """
        project = _idle_project(store, project_id)
        seeds = body.seeds or []
        if len(set(seeds)) != len(seeds):
            raise HTTPException(400, "Each seed may appear only once.")
        if set(seeds) & {take["seed"] for take in project["takes"]}:
            raise HTTPException(400, "This shoot already has a take with that seed.")
        if not project["footage"]:
            raise HTTPException(400, "Wait for the footage to be ready.")
        people = project["people"]
        for number, person in enumerate(people, start=1):
            whose = f" for person {number}" if len(people) > 1 else ""
            if not person["pictures"]:
                raise HTTPException(400, f"Add the character{whose} first.")
            if not (person["views"] or {}).get("front"):
                raise HTTPException(
                    400, f"Add a picture{whose} that shows the whole body, head to toe."
                )
            if not person["anchors"]:
                who = f"person {number}" if len(people) > 1 else "the person"
                raise HTTPException(400, f"Tap {who} to replace first.")
        if not terms.accepted(root):
            raise HTTPException(409, "Agree to the MiniMax H3 terms first.")
        options = body.model_dump()
        options["takes"] = max(body.takes, len(body.seeds or []))
        working = store.set(
            project_id,
            status="working",
            step="render",
            error=None,
            render={"takes": options["takes"], "quality": body.quality},
            activity=projects.activity("Waiting for its turn"),
        )
        worker.submit(
            project_id,
            lambda store, project_id: projects.render(store, project_id, options),
        )
        return working

    # The user's RunPod key lives in their browser and comes with each request.
    @app.get("/api/runpod")
    def runpod_status(request: Request) -> dict[str, Any]:
        return _runpod_state(_runpod_key(request))

    @app.post("/api/runpod/pod", status_code=201)
    def runpod_start(request: Request, body: PodStart) -> dict[str, Any]:
        key = _runpod_key(request)
        token = secrets.token_urlsafe(32)
        # One Deploy at a time: a second click would otherwise see no pod yet and
        # start another, billed and forgotten.
        with deploying:
            try:
                created = runpod.start(key, body.gpu, {"PROXY_TOKEN": token})
            except runpod.RunPodError as exc:
                raise _runpod_failed(exc) from exc
            config.save_pod(runpod.proxy_url(created["id"]), token)
        return _runpod_state(key)

    @app.post("/api/runpod/pod/pause")
    def runpod_pause(request: Request) -> dict[str, Any]:
        key = _runpod_key(request)
        try:
            runpod.pause(key)
        except runpod.RunPodError as exc:
            raise _runpod_failed(exc) from exc
        return _runpod_state(key)

    @app.post("/api/runpod/pod/resume")
    def runpod_resume(request: Request) -> dict[str, Any]:
        key = _runpod_key(request)
        try:
            runpod.resume(key)
        except runpod.RunPodError as exc:
            raise _runpod_failed(exc) from exc
        return _runpod_state(key)

    @app.delete("/api/runpod/pod")
    def runpod_delete(request: Request) -> dict[str, Any]:
        key = _runpod_key(request)
        try:
            runpod.delete(key)
        except runpod.RunPodError as exc:
            raise _runpod_failed(exc) from exc
        config.forget_pod()
        return _runpod_state(key)

    @app.get("/files/{project_id}/{path:path}")
    def project_file(project_id: str, path: str) -> FileResponse:
        try:
            folder = store.folder(project_id).resolve()
        except KeyError as exc:
            raise HTTPException(404) from exc
        file = (folder / path).resolve()
        if not file.is_relative_to(folder) or not file.is_file():
            raise HTTPException(404)
        return FileResponse(file)

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def main() -> None:
    """Starts the app and opens it in the browser."""
    parser = argparse.ArgumentParser(description="Run the Rigless web app.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="don't open it")
    args = parser.parse_args()
    app = create_app(config.RUNS_DIR / "app")
    if not args.no_browser:
        url = f"http://127.0.0.1:{args.port}"
        threading.Timer(1.5, webbrowser.open, [url]).start()
    # An open page keeps connections alive (a playing take), and uvicorn would
    # otherwise wait on them after Ctrl+C until the page is closed.
    uvicorn.run(app, host="127.0.0.1", port=args.port, timeout_graceful_shutdown=2)


def _connection() -> dict[str, Any]:
    """Returns the connection as the page sees it; the token itself stays here."""
    found = config.connection()
    return {"target": found.target, "url": found.url}


def _project(store: Store, project_id: str) -> Project:
    try:
        return store.get(project_id)
    except KeyError as exc:
        raise HTTPException(404, "No such project.") from exc


def _runpod_key(request: Request) -> str:
    key = request.headers.get("X-RunPod-Key", "").strip()
    if not key:
        raise HTTPException(400, "Add your RunPod API key first.")
    return key


def _runpod_state(key: str) -> dict[str, Any]:
    """Returns Rigless's pod, and the GPUs to start one on when there's none.

    A pod paused and resumed in RunPod's console may come back on another machine
    with a new id; the connection follows it, keeping its token.
    """
    try:
        pod = runpod.pod(key)
        gpus = runpod.gpus(key) if pod is None else []
    except runpod.RunPodError as exc:
        raise _runpod_failed(exc) from exc
    if pod is None:
        return {"pod": None, "gpus": gpus}
    url = runpod.proxy_url(pod["id"])
    current = config.connection()
    if current.target == "runpod" and current.key and current.url != url:
        config.save_pod(url, current.key)
    paused = pod["desiredStatus"] == "EXITED"
    progress = None if paused else runpod.setup_progress(url)
    if progress is not None:
        progress |= runpod.setup_estimate(progress, time.time())
    return {
        "pod": {
            "id": pod["id"],
            "paused": paused,
            "idle_cost_per_hour": runpod.idle_cost_per_hour(pod),
            "gpu": runpod.gpu_name(pod),
            "cost_per_hour": pod.get("costPerHr"),
            "started": pod.get("lastStartedAt"),
            "url": url,
            "progress": progress,
        },
        "gpus": gpus,
    }


def _runpod_failed(exc: runpod.RunPodError) -> HTTPException:
    """Returns the answer for a request RunPod refused, and logs it."""
    logger.warning("RunPod: %s", exc)
    return HTTPException(FAILED_DEPENDENCY, str(exc))


def _person(project: Project, person: int) -> Person:
    """Returns one of a project's people, or answers 404."""
    if not 0 <= person < len(project["people"]):
        raise HTTPException(404, "No such person.")
    return project["people"][person]


def _idle_project(store: Store, project_id: str) -> Project:
    """Returns a project that isn't busy, or answers 409."""
    project = _project(store, project_id)
    if project["status"] == "working":
        raise HTTPException(409, "This project is still working on its last step.")
    return project


def _summary(project: Project) -> dict[str, Any]:
    """Returns a project's line in the list of recent projects."""
    scores = [take["score"] for take in project["takes"] if take["score"] is not None]
    views = project["people"][0]["views"] or {}
    return {
        "id": project["id"],
        "mode": project.get("mode", "scene"),
        "created": project["created"],
        "title": project.get("title") or _shorten(project["description"], words=8),
        "status": project["status"],
        "step": project["step"],
        "thumbnail": project.get("poster") or views.get("front"),
        "best_score": max(scores, default=None),
        # For the queue: what it's doing, and enough to estimate how long it takes.
        "activity": project["activity"],
        "trim": project.get("trim"),
        "render": project["render"],
        "takes": [
            {
                key: take.get(key)
                for key in ("seed", "status", "started", "expected_seconds", "sampling")
            }
            for take in project["takes"]
        ],
    }


def _next_title(store: Store) -> str:
    """Names a new shoot "Shoot N", one past the highest N so far."""
    numbers = [
        int(found.group(1))
        for project in store.all()
        if (found := SHOOT_TITLE.fullmatch(project.get("title") or ""))
    ]
    return f"Shoot {max(numbers, default=0) + 1}"


def _shorten(text: str, words: int) -> str:
    """Returns the first words of text, with an ellipsis if any were cut."""
    parts = text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def _save_video(upload: UploadFile, folder: Path, name: str) -> tuple[str, float]:
    """Saves an uploaded video into the project's inputs.

    Returns:
        A tuple (path, duration), where path is the saved file's path in the
        project and duration is the video's length in seconds.

    Raises:
        ValueError: The file isn't one of VIDEO_TYPES or isn't a readable video.
    """
    path = _save(upload, folder, name, VIDEO_TYPES)
    try:
        duration = video_prep.probe(folder / path).duration
    except ValueError as exc:
        raise ValueError(f"{upload.filename} isn't a readable video.") from exc
    return path, duration


def _save_image(upload: UploadFile, folder: Path, name: str) -> str:
    """Saves an uploaded image into the project's inputs and returns its path.

    Raises:
        ValueError: The file isn't one of IMAGE_TYPES or isn't a readable image.
    """
    path = _save(upload, folder, name, IMAGE_TYPES)
    try:
        with Image.open(folder / path) as image:
            image.verify()
    # Pillow reports a damaged PNG as a SyntaxError, a huge image as a bomb.
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        (folder / path).unlink(missing_ok=True)
        raise ValueError(f"{upload.filename} isn't a readable image.") from exc
    return path


def _save(upload: UploadFile, folder: Path, name: str, types: set[str]) -> str:
    """Saves an upload as inputs/<name><suffix> and returns that path.

    Raises:
        ValueError: The file's type isn't one of types.
    """
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in types:
        allowed = ", ".join(sorted(types))
        raise ValueError(f"{upload.filename or 'A file'} isn't one of {allowed}.")
    path = f"inputs/{name}{suffix}"
    with (folder / path).open("wb") as dest:
        shutil.copyfileobj(upload.file, dest)
    return path
