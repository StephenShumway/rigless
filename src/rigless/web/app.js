"use strict";

// One screen: the character and Generate on the left, the footage on the monitor
// in the middle (trim it, tap the person to replace), and the takes on the right.
// The footage uploads as soon as it's chosen; everything else follows the
// project's state on the server.

const FPS = 24; // MiniMax H3's frame rate; the server readies the footage at it
const MIN_TRIM = 1;
const MAX_TRIM = 20;
const MAX_FOOTAGE = 120;
// Seconds per second of clip for one take on an H100, by quality (projects.QUALITIES).
const RENDER_SECONDS_PER_SECOND = { full: 52, draft: 26 }; // measured Oct 24 and Oct 27
const TRACK_DELAY_MS = 1200; // tracking starts this long after the last tap
const SYNC_DELAY_MS = 700;
const REVEAL_MS = 240; // the found person fades in over this long
const POLL_MS = 2000;
const RETRY_MS = 5000;
const THUMBS = 12;
const PREFS_KEY = "rigless:prefs";
const RUNPOD_KEY = "rigless:runpod-key"; // the user's RunPod API key, kept only in this browser
const POD_POLL_MS = 10000;
const SETUP_POLL_MS = 3000; // while a new pod reports its setup's progress
// Each person's mask colour, in the order they're added (--mask-1 to --mask-3);
// selection.TINTS matches.
const MASK_COLORS = ["#90E2DD", "#65955A", "#96A7E7"];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const $ = (selector, root = document) => root.querySelector(selector);

const state = {
  route: 0, // bumped on every navigation, so stale polls and uploads stop
  project: null,
  pollTimer: null,
  local: null, // footage picked on this page: { file, url }, until the server's is ready
  person: 0, // the person being cast, by place: taps and new pictures go to them
  pendingSheet: null, // character images picked before the project exists
  sendingSheet: null, // character images on their way to the server, or being looked at there
  sheetPerson: 0, // whose character sendingSheet is
  recent: [], // characters used before, for the Recent row
  uploading: false,
  trim: null, // [start, end] seconds on the trim bar
  trimDirty: false, // changed here and not yet saved on the server
  descriptionDirty: false,
  duration: 0,
  quality: "full", // what Generate renders: draft or full
  tapBusy: false,
  // What the status bar shows from the click until the server's own step starts:
  // { label, started }.
  localActivity: null,
  pendingTap: null, // { person, frame, point } drawn while the server finds them
  sheetProgress: null, // how much of the character's pictures has uploaded, 0-1
  trackTimer: null,
  syncTimer: null,
  reconnecting: false,
  revealed: new Set(), // takes whose score has already counted up
  viewing: null, // the take shown in the main viewer, by seed; null shows the footage
  lastStatus: null, // the project's status and step at the last poll, to notice a render ending
  lastStep: null,
  shoots: [], // every shoot, newest first: Shoots, Queue, and the top bar's count read it
  view: "set", // the page shown: set (a shoot), shoots, or queue
  queueTimer: null,
  podTimer: null,
  podStarted: null, // when the running pod started, for its uptime
  connection: null, // the last connection check: { target, url, ok, message }
  connectionChecks: 0,
  connectionTimer: null,
  addingPerson: false,
  tagMenu: null, // the open @ menu: { active } is the highlighted character
};

const QUEUE_POLL_MS = 2000;
const CONNECTION_RETRY_MS = 30000;
const WAITING = "Waiting for its turn"; // projects.Worker's label for a queued step
const VIEWS = ["set", "shoots", "queue"];
const NEW_TITLE = "New Video";

const monitor = {
  revealed: new Map(), // overlay path -> when it first showed, for the fade-in
  src: "",
  offset: 0, // the footage time at the monitor video's time 0 (the preview starts at the trim)
  masks: new Map(), // overlay path -> Image
  thumbsKey: "",
  key: "",
};

/* Helpers */

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key in node && !key.includes("-")) node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function svg(tag, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

function icon(id) {
  const node = svg("svg", { "aria-hidden": "true" });
  node.append(svg("use", { href: `#${id}` }));
  return node;
}

function fileUrl(projectId, path) {
  const parts = path.split("/").map(encodeURIComponent).join("/");
  return `/files/${encodeURIComponent(projectId)}/${parts}`;
}

function pad(number, width = 2) {
  return String(number).padStart(width, "0");
}

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const mm = pad(Math.floor((s % 3600) / 60));
  return h ? `${h}:${mm}:${pad(s % 60)}` : `${mm}:${pad(s % 60)}`;
}

function timecode(seconds, fps) {
  const frames = Math.floor(seconds * fps + 1e-6);
  const rate = Math.round(fps);
  return `${pad(Math.floor(frames / rate / 60))}:${pad(Math.floor(frames / rate) % 60)}:${pad(frames % rate)}`;
}

function minutesLeft(seconds) {
  if (seconds < 60) return "under a minute left";
  return `about ${Math.ceil(seconds / 60)} min left`;
}

function shortDate(iso) {
  return new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

// Wraps times like "2.7-3.1s" in a note so they read as timecodes.
function withTimes(text) {
  const parts = text.split(/(\d+(?:\.\d+)?-\d+(?:\.\d+)?s)/);
  return parts.map((part, index) => (index % 2 ? el("span", { class: "tc" }, part) : part));
}

const prefs = {
  load() {
    try {
      return JSON.parse(localStorage.getItem(PREFS_KEY)) || {};
    } catch {
      return {};
    }
  },
  save(change) {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify({ ...prefs.load(), ...change }));
    } catch {
      // Storage can be blocked (private windows); preferences are a nicety.
    }
  },
};

// History has its own line, for a rename or delete that failed there.
function showError(message) {
  const node = $(state.view === "shoots" ? "#view-shoots .form-error" : ".set-panel .form-error");
  node.textContent = message || "";
  node.hidden = !message;
}

function footageReady(project = state.project) {
  return Boolean(project && project.footage && project.driving);
}

function busy(project = state.project) {
  return Boolean(project && project.status === "working");
}

/* API */

function errorMessage(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0] && detail[0].msg) return detail[0].msg;
  return `Something went wrong (error ${status}).`;
}

async function api(path, { method = "GET", json, headers = {} } = {}) {
  const options = { method, headers: { ...headers } };
  if (json !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(json);
  }
  const response = await fetch(path, options);
  const body = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    const failure = new Error(errorMessage(body, response.status));
    failure.status = response.status;
    throw failure;
  }
  return body;
}

function upload(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.responseType = "json";
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () => {
      const body = request.response;
      if (request.status >= 200 && request.status < 300) {
        resolve(body);
      } else {
        reject(new Error(errorMessage(body, request.status)));
      }
    };
    request.onerror = () => reject(new Error("The upload failed. Check the connection and try again."));
    request.send(form);
  });
}

function projectPath(suffix = "") {
  return `/api/projects/${encodeURIComponent(state.project.id)}${suffix}`;
}

/* Routing */

function navigate(url) {
  history.pushState(null, "", url);
  route();
}

function route() {
  state.route += 1;
  clearTimeout(state.pollTimer);
  clearTimeout(state.queueTimer);
  clearTimeout(state.podTimer);
  const params = new URLSearchParams(location.search);
  const id = params.get("p");
  const asked = params.get("view");
  state.view = !id && VIEWS.includes(asked) ? asked : "set";
  show(`view-${state.view}`);
  if (state.view === "shoots") showError("");
  if (state.view === "set") {
    if (!id || !state.project || state.project.id !== id) resetSet();
    if (id) refresh(id, state.route);
    loadRecent();
  }
  // The ComfyUI settings are a dialog over the shoot; ?view=comfyui opens it.
  const settings = $("#comfyui-dialog");
  if (asked === "comfyui") openComfyUI();
  else if (settings.open) settings.close();
  renderShoots();
  loadShoots();
}

function show(viewId) {
  for (const view of document.querySelectorAll(".view")) view.hidden = view.id !== viewId;
}

// Held, not looked up: renaming swaps the heading for a text box for a while.
const pageTitle = $(".page-title");

function setTitle(text) {
  pageTitle.textContent = text;
}

/* Connection status in the top bar: where Rigless renders and whether it answers */

const TARGET_NAMES = { runpod: "RunPod", local: "This Computer", custom: "Custom" };

async function checkConnection() {
  // Only the latest check counts, so switching back and forth never shows an older answer.
  const check = ++state.connectionChecks;
  let result;
  try {
    result = await api("/api/connection/status");
  } catch (failure) {
    result = { ok: false, message: failure.message };
  }
  if (check !== state.connectionChecks) return;
  state.connection = result;
  showConnection(result);
  // Until it answers, keep asking, so the top bar notices a pod that's become ready.
  clearTimeout(state.connectionTimer);
  if (!result.ok) state.connectionTimer = setTimeout(checkConnection, CONNECTION_RETRY_MS);
  if (result.target === "local" || result.target === "custom") {
    setTargetState(result.target, result.ok ? "ok" : "bad", result.message);
  }
}

function showConnection({ target, url, ok, message }) {
  const indicator = $("#connection");
  indicator.dataset.state = ok ? "ok" : "bad";
  indicator.title = message;
  const status = ok ? "Connected" : target === "runpod" && !url ? "No pod" : "Not connected";
  $(".connection-label", indicator).textContent = target ? `${TARGET_NAMES[target]} · ${status}` : status;
}

// Shown at once on a switch, while the check takes its few seconds.
function showConnecting(target) {
  const indicator = $("#connection");
  indicator.dataset.state = "checking";
  indicator.title = "";
  $(".connection-label", indicator).textContent = `${TARGET_NAMES[target]} · Connecting…`;
  if (target !== "runpod") setTargetState(target, "checking");
}

// The status line on the This Computer and Custom tabs.
function setTargetState(target, tone, message = "") {
  const line = $(`.${target}-state`);
  line.className = `pod-state ${target}-state ${tone}`;
  if (tone === "checking") line.replaceChildren(spinner(), "Connecting…");
  else line.textContent = tone === "ok" ? "Connected" : message;
}

function spinner() {
  return el("span", { class: "spinner", "aria-hidden": "true" });
}

/* The set */

function resetSet() {
  clearTimeout(state.trackTimer);
  clearTimeout(state.syncTimer);
  showOriginal(false);
  if (state.local) URL.revokeObjectURL(state.local.url);
  Object.assign(state, {
    project: null, local: null, person: 0, pendingSheet: null, sendingSheet: null, sheetPerson: 0,
    uploading: false, trim: null,
    trimDirty: false, descriptionDirty: false, duration: 0, tapBusy: false, localActivity: null,
    pendingTap: null, sheetProgress: null, viewing: null, lastStatus: null, lastStep: null,
  });
  const video = $(".monitor video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  Object.assign(monitor, { src: "", offset: 0, thumbsKey: "", key: "" });
  monitor.masks.clear();
  $(".monitor").dataset.state = "empty";
  $(".set").style.removeProperty("--ratio");
  for (const part of [".status", ".notes", ".result", ".rename-title"]) $(part).hidden = true;
  const strip = $(".filmstrip");
  strip.getContext("2d").clearRect(0, 0, strip.width, strip.height);
  renderTrim();
  $("textarea[name=description]").value = "";
  renderPromptMarks();
  $(".people").replaceChildren();
  for (const url of fileUrls.values()) URL.revokeObjectURL(url);
  fileUrls.clear();
  renderCharacter();
  setMessage("");
  showError("");
  document.title = "Rigless";
  setTitle(NEW_TITLE);
  $(".compare-hold").hidden = true;
  renderChecklist();
  renderViewerTabs();
  renderClipInfo();
  renderResult();
  renderTakes();
}

async function refresh(id, token) {
  let project;
  try {
    project = await api(`/api/projects/${encodeURIComponent(id)}`);
  } catch (failure) {
    if (token !== state.route) return;
    if (failure.status === 404) resetSet(); // deleted, not a server restarting
    if (state.project && state.project.id === id) {
      // The server may be restarting; keep the page and try again.
      state.reconnecting = true;
      updateClock();
      state.pollTimer = setTimeout(() => refresh(id, token), RETRY_MS);
    } else {
      showError(failure.message);
    }
    return;
  }
  if (token !== state.route) return;
  state.reconnecting = false;
  applyProject(project);
  if (busy(project)) {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(() => refresh(id, token), POLL_MS);
  }
}

function poll() {
  clearTimeout(state.pollTimer);
  const id = state.project.id;
  const token = state.route;
  state.pollTimer = setTimeout(() => refresh(id, token), POLL_MS);
}

function applyProject(project) {
  // A slow answer about a shoot the page has since left is dropped.
  if (new URLSearchParams(location.search).get("p") !== project.id) return;
  if (project.mode !== "swap") {
    state.project = project;
    $(".monitor").dataset.state = "empty";
    showError("This shoot was made with an earlier version of Rigless and can't be opened here.");
    return;
  }
  const opening = !state.project || state.project.id !== project.id;
  const finished = state.lastStatus === "working" && state.lastStep === "render" && !busy(project);
  const statusChanged = state.lastStatus !== project.status;
  state.project = project;
  state.person = Math.min(state.person, project.people.length - 1);
  state.lastStatus = project.status;
  state.lastStep = project.step;
  if (statusChanged) loadShoots();
  document.title = busy(project) && project.activity ? `${project.activity.label} · Rigless` : "Rigless";
  setTitle(project.title || "Untitled shoot");
  $(".rename-title").hidden = false;
  // A finished render shows its new take; a shoot opened with takes, its best.
  const fresh = finished ? newTake(project) : null;
  if (fresh) {
    state.viewing = fresh.seed;
    notifyDone(fresh, project);
  } else if (opening && project.best != null) {
    state.viewing = project.best;
  }
  if (opening) loadRecent();
  if (state.viewing != null && !viewedTake(project)) state.viewing = null;
  if (!state.trimDirty && project.trim) state.trim = project.trim.slice();
  if (footageReady(project)) state.duration = project.footage.frames / project.footage.fps;
  const notes = $("textarea[name=description]");
  if (!state.descriptionDirty && document.activeElement !== notes) notes.value = project.description || "";
  if (busy(project)) state.localActivity = null; // the server's step has taken over
  else syncEdits();
  if (state.pendingSheet && !state.uploading) sendCharacter(state.pendingSheet, false);
  if (state.sendingSheet && !state.uploading && (pictures(project, state.sheetPerson).length || !busy(project))) {
    state.sendingSheet = null;
  }
  renderMonitor();
  renderTrim();
  renderCharacter();
  renderStatus();
  renderNotes();
  renderChecklist();
  renderViewerTabs();
  renderClipInfo();
  renderResult();
  renderTakes();
  if (project.status === "failed" && project.error) showError(project.error);
  // Tracking follows the taps on its own.
  if (hasTaps(project) && !project.track && !busy(project) && project.status !== "failed" && !state.tapBusy) {
    scheduleTrack();
  }
}

/* Footage: picked, shown at once, uploaded in the background */

// By name too: Windows gives some clips, such as .mkv, no type.
function isVideo(file) {
  return Boolean(file) && (file.type.startsWith("video/") || /\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name));
}

async function chooseFootage(file) {
  if (!isVideo(file)) return;
  if (state.project || state.local) navigate("/");
  const token = state.route;
  state.local = { file, url: URL.createObjectURL(file) };
  setSource(state.local.url, 0);
  buildFilmstrip(state.local.url);
  $(".monitor").dataset.state = "local";
  const video = $(".monitor video");
  // A clip this browser can't play (HEVC in some browsers, ProRes) still
  // uploads: the server converts it, then checks its length and trims it.
  const playable = await new Promise((resolve) => {
    video.addEventListener("loadedmetadata", () => resolve(true), { once: true });
    video.addEventListener("error", () => resolve(false), { once: true });
  });
  if (token !== state.route) return;
  const known = playable && Number.isFinite(video.duration);
  if (known) {
    state.duration = video.duration;
    if (state.duration > MAX_FOOTAGE) {
      showError(`Use a clip up to ${MAX_FOOTAGE / 60} minutes long. This one is ${Math.round(state.duration)} seconds.`);
      return;
    }
    state.trim = [0, Math.min(state.duration, MAX_TRIM)];
    renderTrim();
    renderChecklist();
  }

  const data = new FormData();
  data.append("dance", file);
  if (known) {
    data.append("start", String(state.trim[0]));
    data.append("end", String(state.trim[1]));
  }
  const description = $("textarea[name=description]").value.trim();
  if (description) data.append("description", description);
  const sheet = state.pendingSheet;
  for (const image of sheet || []) data.append("sheet", image);
  if (sheet) {
    // They ride with the footage; their tiles spin until the server has found the views.
    state.sendingSheet = sheet;
    state.sheetPerson = 0;
    state.pendingSheet = null;
    renderCharacter();
  }
  const sent = state.trim ? state.trim.slice() : null;
  state.uploading = true;
  try {
    const { id } = await upload("/api/projects", data, (fraction) => {
      if (token === state.route) setMessage(fraction < 1 ? `Uploading ${Math.round(fraction * 100)}%` : "Getting the footage ready");
    });
    if (token !== state.route) return;
    state.trimDirty = Boolean(sent) && (state.trim[0] !== sent[0] || state.trim[1] !== sent[1]);
    history.replaceState(null, "", `?p=${encodeURIComponent(id)}`);
    await refresh(id, token);
  } catch (failure) {
    if (token === state.route) {
      setMessage("");
      showError(failure.message);
      state.pendingSheet = state.sendingSheet;
      state.sendingSheet = null;
      renderCharacter();
    }
  } finally {
    state.uploading = false;
  }
}

// The monitor shows the server's footage once it's ready, and the tracking
// preview (the trimmed part, person tinted) once tracking is done.
function renderMonitor() {
  const project = state.project;
  const figure = $(".monitor");
  const take = viewedTake(project);
  $(".compare-hold").hidden = !take;
  if (footageReady(project)) buildFilmstrip(fileUrl(project.id, project.driving));
  if (take) {
    // A take covers the trimmed part, so its time 0 is the trim's start.
    setSource(fileUrl(project.id, take.video), project.trim ? project.trim[0] : 0);
    figure.dataset.state = "take";
  } else if (footageReady(project)) {
    const preview = project.track && project.track.preview;
    const src = preview ? fileUrl(project.id, preview) : fileUrl(project.id, project.driving);
    setSource(src, preview ? project.track.start || 0 : 0);
    if (state.local && !preview) {
      URL.revokeObjectURL(state.local.url);
      state.local = null;
    }
    figure.dataset.state = preview ? "tracked" : "ready";
  } else {
    figure.dataset.state = state.local ? "local" : project ? "pending" : "empty";
  }
  figure.classList.toggle("rolling", busy(project));
  if (busy(project) && project.activity) setMessage(project.activity.label);
  else if (needsPerson(project)) {
    const several = people(project).length > 1;
    setMessage(several ? `Click person ${state.person + 1} in the video` : "Click the person you want to replace", "prompt");
  }
  else if (!state.uploading) setMessage("");
}

function setSource(src, offset) {
  const video = $(".monitor video");
  if (monitor.src === src) return;
  const time = footageTime();
  const wasPlaying = !video.paused;
  monitor.src = src;
  monitor.offset = offset;
  video.src = src;
  video.addEventListener("loadedmetadata", () => {
    $(".set").style.setProperty("--ratio", String(video.videoWidth / video.videoHeight)); // the take cards' shape
    seekFootage(time || (state.trim ? state.trim[0] : 0));
    if (wasPlaying) video.play().catch(() => {});
  }, { once: true });
}

function footageTime() {
  const video = $(".monitor video");
  return (video.currentTime || 0) + monitor.offset;
}

function seekFootage(seconds) {
  const video = $(".monitor video");
  const local = Math.max(0, seconds - monitor.offset);
  video.currentTime = Math.min(local, Math.max(0, (video.duration || local) - 0.001));
}

function currentFrame() {
  return Math.floor(footageTime() * FPS + 1e-3);
}

function setMessage(text, tone = "") {
  const message = $(".monitor-message");
  message.textContent = text;
  message.dataset.tone = tone;
}

function setupMonitor() {
  const input = $(".monitor-drop input");
  input.addEventListener("change", () => chooseFootage(input.files[0]));

  const video = $(".monitor video");
  // The monitor loops the trimmed part.
  video.addEventListener("timeupdate", () => {
    if (!state.trim || video.paused) return;
    const time = footageTime();
    if (time >= state.trim[1] - 0.02 || time < state.trim[0] - 0.05) seekFootage(state.trim[0]);
  });
  video.addEventListener("play", () => $(".transport").classList.add("playing"));
  video.addEventListener("pause", () => $(".transport").classList.remove("playing"));
  $(".monitor-frame").addEventListener("click", tapMonitor);
  $(".remove-tap").addEventListener("click", (event) => {
    event.stopPropagation(); // not a tap on the footage
    removeTaps(state.person, currentFrame());
  });

  for (const button of document.querySelectorAll(".transport .icon-btn")) {
    button.addEventListener("click", () => {
      const action = button.dataset.action;
      if (action === "play") {
        if (video.paused) {
          if (state.trim && (footageTime() < state.trim[0] || footageTime() >= state.trim[1] - 0.05)) seekFootage(state.trim[0]);
          video.play().catch(() => {});
        } else {
          video.pause();
        }
        return;
      }
      video.pause();
      const step = action === "back" ? -1 : 1;
      seekFootage((currentFrame() + step + 0.5) / FPS);
    });
  }
  document.addEventListener("keydown", (event) => {
    if (state.view !== "set" || event.target.closest("input, textarea, select") || $(".monitor").dataset.state === "empty"
      || $("dialog[open]")) return;
    const key = event.key.toLowerCase();
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    } else if (key === " ") {
      if (event.target.closest("button, a, [tabindex]")) return; // Space presses what has focus
      event.preventDefault();
      $(".transport [data-action=play]").click();
    } else if (key === "arrowleft" || key === "arrowright") {
      event.preventDefault();
      $(`.transport [data-action=${key === "arrowleft" ? "back" : "forward"}]`).click();
    }
  });
  requestAnimationFrame(drawOverlay);
}

/* Trim: a filmstrip with in and out handles */

// With no clip yet, the transport stays as an empty, disabled timeline.
function renderTrim() {
  const shown = Boolean(state.trim && state.duration);
  const transport = $(".transport");
  transport.classList.toggle("empty", !shown);
  for (const button of transport.querySelectorAll("button")) button.disabled = !shown;
  if (!shown) return;
  const [start, end] = state.trim;
  const left = (start / state.duration) * 100;
  const right = (end / state.duration) * 100;
  const trim = $(".trim");
  trim.style.setProperty("--start", `${left}%`);
  trim.style.setProperty("--end", `${right}%`);
  $(".trim-readout").replaceChildren(el("b", {}, `${(end - start).toFixed(1)} s`), "  selected");
  renderTrimMarks();
  renderChecklist();
}

// Ticks for the frames with taps, and red stretches where tracking lost the person.
function renderTrimMarks() {
  const project = state.project;
  const marks = [];
  if (project && state.duration) {
    for (const [index, person] of people(project).entries()) {
      for (const anchor of person.anchors) {
        marks.push(el("i", {
          class: "tap-tick",
          style: `left: ${(anchor.frame / FPS / state.duration) * 100}%; background: ${MASK_COLORS[index]}`,
        }));
      }
    }
    for (const [start, end] of (project.track && project.track.lost) || []) {
      marks.push(el("i", {
        class: "lost-span",
        style: `left: ${(start / state.duration) * 100}%; width: ${((end - start) / state.duration) * 100}%`,
      }));
    }
  }
  $(".trim-marks").replaceChildren(...marks);
}

function setupTrim() {
  const trim = $(".trim");
  let dragging = null;
  const timeAt = (event) => {
    const box = $(".trim-track").getBoundingClientRect();
    return Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)) * state.duration;
  };
  for (const handle of trim.querySelectorAll(".trim-handle")) {
    handle.addEventListener("pointerdown", (event) => {
      if (!state.trim || busy()) return;
      event.preventDefault();
      event.stopPropagation();
      handle.setPointerCapture(event.pointerId);
      dragging = handle.dataset.edge;
      $(".monitor video").pause();
    });
    handle.addEventListener("pointermove", (event) => {
      if (dragging !== handle.dataset.edge) return;
      moveEdge(dragging, timeAt(event));
    });
    handle.addEventListener("pointerup", () => {
      if (!dragging) return;
      dragging = null;
      commitTrim();
    });
    handle.addEventListener("keydown", (event) => {
      if (!state.trim || busy() || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      event.stopPropagation();
      const edge = handle.dataset.edge === "start" ? 0 : 1;
      const step = (event.shiftKey ? 1 : 1 / FPS) * (event.key === "ArrowLeft" ? -1 : 1);
      moveEdge(handle.dataset.edge, state.trim[edge] + step);
      commitTrim();
    });
  }
  trim.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".trim-handle") || !state.trim) return;
    $(".monitor video").pause();
    const time = Math.min(state.trim[1] - 1 / FPS, Math.max(state.trim[0], timeAt(event)));
    seekFootage(time);
  });
}

function moveEdge(edge, time) {
  let [start, end] = state.trim;
  if (edge === "start") start = Math.min(Math.max(0, time, end - MAX_TRIM), end - MIN_TRIM);
  else end = Math.max(Math.min(state.duration, time, start + MAX_TRIM), start + MIN_TRIM);
  state.trim = [Math.round(start * FPS) / FPS, Math.round(end * FPS) / FPS];
  state.trimDirty = true;
  seekFootage(edge === "start" ? state.trim[0] : state.trim[1] - 1 / FPS);
  renderTrim();
}

function commitTrim() {
  clearTimeout(state.syncTimer);
  state.syncTimer = setTimeout(syncEdits, SYNC_DELAY_MS);
}

// Saves edits made here (the trim, the notes) once the server can take them.
async function syncEdits() {
  const project = state.project;
  if (!project || !footageReady(project) || busy(project)) return;
  const body = {};
  if (state.trimDirty) body.trim = state.trim;
  if (state.descriptionDirty) body.description = $("textarea[name=description]").value;
  if (!Object.keys(body).length) return;
  const sentTrim = state.trim.slice();
  try {
    const updated = await api(projectPath(), { method: "PATCH", json: body });
    // Only what was sent is saved; anything changed meanwhile goes next time.
    if (state.trim[0] === sentTrim[0] && state.trim[1] === sentTrim[1]) state.trimDirty = false;
    if ($("textarea[name=description]").value === body.description) state.descriptionDirty = false;
    applyProject(updated);
    if (body.trim && hasTaps(updated)) scheduleTrack();
  } catch (failure) {
    if (failure.status === 409) return; // retried when the current step ends
    state.trimDirty = false;
    if (state.project.trim) state.trim = state.project.trim.slice();
    renderTrim();
    showError(failure.message);
  }
}

// Draws a strip of frames under the trim handles from a second, hidden video.
async function buildFilmstrip(src) {
  if (monitor.thumbsKey === src) return;
  monitor.thumbsKey = src;
  const canvas = $(".filmstrip");
  const strip = document.createElement("video");
  strip.muted = true;
  strip.preload = "auto";
  strip.src = src;
  await new Promise((resolve) => strip.addEventListener("loadeddata", resolve, { once: true }));
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * ratio); // the track's width, not the bar's
  canvas.height = Math.round(canvas.clientHeight * ratio);
  const context = canvas.getContext("2d");
  const cell = canvas.width / THUMBS;
  for (let index = 0; index < THUMBS; index += 1) {
    if (monitor.thumbsKey !== src) return;
    strip.currentTime = ((index + 0.5) / THUMBS) * strip.duration;
    await new Promise((resolve) => strip.addEventListener("seeked", resolve, { once: true }));
    const scale = Math.max(cell / strip.videoWidth, canvas.height / strip.videoHeight);
    const width = strip.videoWidth * scale;
    const height = strip.videoHeight * scale;
    context.save();
    context.beginPath();
    context.rect(index * cell, 0, cell, canvas.height);
    context.clip();
    context.drawImage(strip, index * cell + (cell - width) / 2, (canvas.height - height) / 2, width, height);
    context.restore();
  }
  strip.removeAttribute("src");
  strip.load();
}

/* Taps: who gets replaced */

function canTap() {
  const project = state.project;
  return footageReady(project) && !busy(project) && !state.tapBusy && !viewedTake(project);
}

// The selected person isn't picked out yet, so the monitor says what to do.
function needsPerson(project) {
  const person = people(project)[state.person];
  return footageReady(project) && !viewedTake(project) && !busy(project) && !state.tapBusy
    && Boolean(person) && !person.anchors.length;
}

// Maps a click on the monitor to the video's own pixels.
function videoPoint(event) {
  const video = $(".monitor video");
  const box = video.getBoundingClientRect();
  const scale = Math.min(box.width / video.videoWidth, box.height / video.videoHeight);
  const left = box.left + (box.width - video.videoWidth * scale) / 2;
  const top = box.top + (box.height - video.videoHeight * scale) / 2;
  const x = Math.round((event.clientX - left) / scale);
  const y = Math.round((event.clientY - top) / scale);
  if (x < 0 || y < 0 || x >= video.videoWidth || y >= video.videoHeight) return null;
  return [x, y];
}

async function tapMonitor(event) {
  const video = $(".monitor video");
  if (viewedTake()) {
    if (video.paused) video.play().catch(() => {});
    else video.pause();
    return;
  }
  if (!canTap()) return;
  const point = videoPoint(event);
  if (!point) return;
  video.pause();
  const frame = currentFrame();
  ripple(event);
  setMessage("");
  // A click picks the selected person on this frame, replacing their earlier pick here.
  const person = state.person;
  state.pendingTap = { person, frame, point };
  startLocalActivity("Finding them");
  await sendAnchor({ person, frame, positive: [point], negative: [] });
}

// Shows a step in the status bar at once, before the server has it.
function startLocalActivity(label) {
  const started = state.localActivity ? state.localActivity.started : Date.now();
  state.localActivity = { label, started };
  renderStatus();
}

function endLocalActivity() {
  state.localActivity = null;
  renderStatus();
}

function ripple(event) {
  const frame = $(".monitor-frame");
  const box = frame.getBoundingClientRect();
  const mark = el("span", {
    class: "ripple",
    style: `left: ${event.clientX - box.left}px; top: ${event.clientY - box.top}px`,
  });
  mark.addEventListener("animationend", () => mark.remove());
  frame.append(mark);
}

async function sendAnchor(body) {
  state.tapBusy = true;
  clearTimeout(state.trackTimer);
  showError("");
  try {
    const project = await api(projectPath("/anchors"), { method: "POST", json: body });
    applyProject(project);
    scheduleTrack();
  } catch (failure) {
    endLocalActivity();
    showError(failure.message);
  } finally {
    state.tapBusy = false;
    state.pendingTap = null;
  }
}

// Clears a person's taps, or only those on one frame; what's left is tracked again.
async function removeTaps(person, frame = null) {
  clearTimeout(state.trackTimer);
  showError("");
  const path = frame === null ? `/anchors?person=${person}` : `/anchors/${frame}?person=${person}`;
  try {
    const project = await api(projectPath(path), { method: "DELETE" });
    applyProject(project);
    if (hasTaps(project)) scheduleTrack();
    else endLocalActivity();
  } catch (failure) {
    showError(failure.message);
  }
}

// Tracking waits a moment after a tap, in case another follows; the status bar
// says it's coming meanwhile.
function scheduleTrack(delay = TRACK_DELAY_MS) {
  clearTimeout(state.trackTimer);
  startLocalActivity("Following them through the clip");
  state.trackTimer = setTimeout(startTrack, delay);
}

async function startTrack() {
  const project = state.project;
  if (!project || !hasTaps(project) || project.track) {
    endLocalActivity();
    return;
  }
  if (busy(project) || state.tapBusy) {
    scheduleTrack();
    return;
  }
  try {
    applyProject(await api(projectPath("/track"), { method: "POST" }));
    poll();
  } catch (failure) {
    endLocalActivity();
    showError(failure.message);
  }
}

// Draws each person's taps and mask on the frame they belong to. Once
// tracking is done, the preview video carries the tint itself.
function drawOverlay() {
  requestAnimationFrame(drawOverlay);
  const canvas = $(".overlay");
  const video = $(".monitor video");
  const project = state.project;
  const empty = $(".monitor").dataset.state === "empty";
  // Written only when it changes: this runs every frame, even while nothing plays.
  const hud = $(".hud-tc");
  const tc = empty ? hud.textContent : timecode(footageTime(), FPS);
  if (hud.textContent !== tc) hud.textContent = tc;
  drawPlayhead();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * ratio);
  const height = Math.round(canvas.clientHeight * ratio);
  const frame = currentFrame();
  const anchors = people(project).map((person) => person.anchors.find((item) => item.frame === frame));
  const pending = state.pendingTap && state.pendingTap.frame === frame ? state.pendingTap : null;
  const tracked = Boolean(project && project.track);
  const showingTake = Boolean(viewedTake(project));
  $(".remove-tap").hidden = !anchors[state.person] || !video.paused || busy(project)
    || state.tapBusy || showingTake;
  const fades = anchors.map((anchor) => (anchor && !tracked ? revealAlpha(anchor.mask) : 1));
  const masks = anchors.map((anchor) => (anchor ? anchor.mask : "")).join(",");
  const key = `${frame}|${width}x${height}|${masks}|${pending ? pending.point : ""}|${tracked}|${monitor.src}|${fades.map((fade) => fade.toFixed(2))}|${showingTake}`;
  if (key === monitor.key) return;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  if (empty || showingTake || !video.videoWidth || (!anchors.some(Boolean) && !pending)) {
    monitor.key = key;
    return;
  }
  const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
  const left = (width - video.videoWidth * scale) / 2;
  const top = (height - video.videoHeight * scale) / 2;
  if (!tracked) {
    for (const [index, anchor] of anchors.entries()) {
      if (!anchor) continue;
      const image = maskImage(anchor.mask);
      if (!image.complete) return; // try again next frame
      context.globalAlpha = fades[index];
      context.drawImage(image, left, top, video.videoWidth * scale, video.videoHeight * scale);
    }
    context.globalAlpha = 1;
  }
  const at = ([x, y]) => [left + x * scale, top + y * scale];
  for (const [index, anchor] of anchors.entries()) {
    // While a new pick is on its way, the old one's dots step aside.
    if (!anchor || (pending && pending.person === index)) continue;
    for (const point of anchor.positive) drawTap(context, at(point), true, ratio, MASK_COLORS[index]);
    for (const point of anchor.negative) drawTap(context, at(point), false, ratio, MASK_COLORS[index]);
  }
  if (pending) drawTap(context, at(pending.point), true, ratio, MASK_COLORS[pending.person], true);
  monitor.key = key;
}

function revealAlpha(path) {
  const image = monitor.masks.get(path);
  if (!image || !image.complete) return 0;
  if (!monitor.revealed.has(path)) monitor.revealed.set(path, performance.now());
  return reducedMotion.matches ? 1 : Math.min(1, (performance.now() - monitor.revealed.get(path)) / REVEAL_MS);
}

function maskImage(path) {
  let image = monitor.masks.get(path);
  if (!image) {
    image = new Image();
    image.src = fileUrl(state.project.id, path);
    monitor.masks.set(path, image);
  }
  return image;
}

function drawTap(context, [x, y], positive, ratio, color, pending = false) {
  context.globalAlpha = pending ? 0.6 : 1;
  context.beginPath();
  context.arc(x, y, 5.5 * ratio, 0, Math.PI * 2);
  context.lineWidth = 2 * ratio;
  context.strokeStyle = "#E8E8E1"; // --text-primary: the spec has no pure white
  if (positive) {
    context.fillStyle = color;
    context.fill();
    context.stroke();
  } else {
    context.fillStyle = "rgba(12, 12, 11, 0.6)";
    context.fill();
    context.stroke();
    const arm = 3 * ratio;
    context.beginPath();
    context.moveTo(x - arm, y - arm);
    context.lineTo(x + arm, y + arm);
    context.moveTo(x + arm, y - arm);
    context.lineTo(x - arm, y + arm);
    context.stroke();
  }
  context.globalAlpha = 1;
}

function drawPlayhead() {
  if (!state.duration || $(".transport").classList.contains("empty")) return;
  $(".trim").style.setProperty("--head", `${(footageTime() / state.duration) * 100}%`);
}

/* The people to replace, each with their character */

const ROLES = { front: "Front", face: "Face", back: "Back", view: "View" };
const MAX_PICTURES = 9; // H3's limit, swap.MAX_PICTURES
const MAX_PEOPLE = MASK_COLORS.length; // projects.MAX_PEOPLE
const fileUrls = new Map(); // picked File -> object URL, for its thumbnail

function setupCharacter() {
  $(".add-person").addEventListener("click", addPerson);
}

function people(project = state.project) {
  return (project && project.people) || [];
}

function hasTaps(project = state.project) {
  return people(project).some((person) => person.anchors.length > 0);
}

function pictures(project = state.project, index = state.person) {
  const person = people(project)[index];
  return (person && person.pictures) || [];
}

// How many of each person's pictures H3 gets: one each in turn, up to its
// limit, first pictures first (projects.allot).
function allot(counts) {
  const given = counts.map(() => 0);
  let left = MAX_PICTURES;
  while (left > 0 && given.some((had, index) => had < counts[index])) {
    for (const index of counts.keys()) {
      if (left > 0 && given[index] < counts[index]) {
        given[index] += 1;
        left -= 1;
      }
    }
  }
  return given;
}

// Each card's pictures: the server's, then any still on their way there.
function cardFiles(index) {
  return index === state.sheetPerson ? state.sendingSheet || state.pendingSheet || [] : [];
}

function selectPerson(index) {
  if (index === state.person) return;
  state.person = index;
  renderCharacter();
  renderMonitor();
}

// New images join the selected person's character; a recent character replaces it.
function chooseCharacter(files, { replace = false } = {}) {
  if (!files.length) return;
  if (!replace && pictures().length >= MAX_PICTURES) {
    showError(`A character has up to ${MAX_PICTURES} pictures. Remove one to add another.`);
    return;
  }
  if (state.project) sendCharacter(files, !replace && pictures().length > 0);
  else state.pendingSheet = replace ? files : [...(state.pendingSheet || []), ...files];
  renderCharacter();
  renderChecklist();
}

async function sendCharacter(files, append) {
  state.pendingSheet = null;
  state.sendingSheet = files;
  state.sheetPerson = state.person;
  state.sheetProgress = 0;
  renderCharacter();
  renderChecklist();
  const data = new FormData();
  for (const file of files) data.append("sheet", file);
  data.append("append", String(append));
  data.append("person", String(state.sheetPerson));
  showError("");
  try {
    applyProject(await upload(projectPath("/character"), data, (fraction) => {
      state.sheetProgress = fraction;
      renderCharacterStatus();
    }));
    loadRecent();
  } catch (failure) {
    showError(failure.message);
  } finally {
    state.sendingSheet = null;
    state.sheetProgress = null;
    renderCharacter();
    renderChecklist();
  }
}

async function removePicture(index, id) {
  showError("");
  try {
    const path = `/pictures/${encodeURIComponent(id)}?person=${index}`;
    applyProject(await api(projectPath(path), { method: "DELETE" }));
  } catch (failure) {
    showError(failure.message);
  }
}

async function addPerson() {
  if (state.addingPerson) return; // a double-click adds one person, not two
  state.addingPerson = true;
  showError("");
  try {
    const project = await api(projectPath("/people"), { method: "POST" });
    state.person = project.people.length - 1;
    applyProject(project);
  } catch (failure) {
    showError(failure.message);
  } finally {
    state.addingPerson = false;
  }
}

// The first click asks, the second removes, like deleting a shoot.
function removePerson(index) {
  return async (event) => {
    event.stopPropagation();
    const button = event.currentTarget;
    if (!button.classList.contains("confirm")) {
      button.classList.add("confirm");
      button.textContent = "Remove";
      button.addEventListener("pointerleave", () => {
        button.classList.remove("confirm");
        button.textContent = "×";
      }, { once: true });
      return;
    }
    if (event.detail > 1) return; // a double-click's second click isn't a decision
    // The button stays, on the card that now shows the next person: unarmed.
    button.classList.remove("confirm");
    button.textContent = "×";
    showError("");
    // The card goes at once; the server's answer follows.
    const before = state.project;
    if (state.person >= index) state.person = Math.max(0, state.person - 1);
    applyProject(withoutPerson(before, index));
    try {
      const project = await api(projectPath(`/people/${index}`), { method: "DELETE" });
      applyProject(project);
      // The server drops the tracking only when the preview's tints change; the
      // masks are kept, so tracking again just makes a new preview.
      if (hasTaps(project) && !project.track) {
        startLocalActivity("Making the preview");
        startTrack();
      }
    } catch (failure) {
      applyProject(before);
      showError(failure.message);
    }
  };
}

// The project as it will be without a person: the tracking stays until the
// server says otherwise.
function withoutPerson(project, index) {
  const others = (list) => list.filter((_, at) => at !== index);
  const track = project.track && { ...project.track, people: others(project.track.people) };
  return { ...project, people: others(project.people), track };
}

function localUrl(file) {
  if (!fileUrls.has(file)) fileUrls.set(file, URL.createObjectURL(file));
  return fileUrls.get(file);
}

// One card per person, in their mask's colour and number; the selected one
// takes the taps and new pictures. Then "Add person".
function renderCharacter() {
  const project = state.project;
  const count = Math.max(1, people(project).length);
  const container = $(".people");
  while (container.children.length > count) container.lastElementChild.remove();
  while (container.children.length < count) container.append(personCard(container.children.length));
  const totals = [...container.children].map((_, index) => pictures(project, index).length + cardFiles(index).length);
  const used = allot(totals);
  [...container.children].forEach((card, index) => renderPerson(card, index, count > 1, used[index]));
  const total = used.reduce((sum, value) => sum + value, 0);
  $(".reference-count").textContent = `References ${total}/${MAX_PICTURES}`;
  $(".set").style.setProperty("--active", MASK_COLORS[state.person]);
  const add = $(".add-person");
  add.hidden = !project || count >= MAX_PEOPLE;
  add.disabled = busy(project);
  $("textarea[name=description]").placeholder = `Optional. Anything the pictures don't show. Type @ to tag ${
    count > 1 ? "a character" : "the character"}.`;
  renderPromptMarks();
  renderCharacterStatus();
}

// Highlights each @n in the prompt in person n's colour, on a copy of the text
// behind the box; tags for nobody stay plain.
function renderPromptMarks() {
  const notes = $("textarea[name=description]");
  const count = people().length || 1;
  const parts = notes.value.split(/(@\d)/);
  $(".prompt-marks").replaceChildren(...parts.map((part, index) => {
    const number = index % 2 ? Number(part.slice(1)) : 0;
    if (number < 1 || number > count) return part;
    return el("mark", { style: `--tag: ${MASK_COLORS[number - 1]}73` }, part); // 45% alpha
  }), "\n"); // a final newline still takes a line in the box
  $(".prompt-marks").scrollTop = notes.scrollTop;
}

function personCard(index) {
  const card = $("#person-card").content.firstElementChild.cloneNode(true);
  card.style.setProperty("--mask", MASK_COLORS[index]);
  $(".mask-badge", card).textContent = String(index + 1);
  card.addEventListener("click", () => selectPerson(index));
  card.addEventListener("keydown", (event) => {
    if (event.target === card && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      selectPerson(index);
    }
  });
  const input = $(".slot input", card);
  input.addEventListener("change", () => chooseCharacter([...input.files]));
  $(".person-remove", card).addEventListener("click", removePerson(index));
  $(".clear-taps", card).addEventListener("click", (event) => {
    event.stopPropagation(); // not a click that selects the person
    removeTaps(index);
  });
  return card;
}

// A person's pictures, Picture 1 first, each with an × on hover; then a +.
// Pictures past the person's part of H3's limit are dimmed.
function renderPerson(card, index, several, used) {
  const project = state.project;
  const theirs = cardFiles(index);
  card.classList.toggle("active", several && index === state.person);
  $(".card-title", card).textContent = several ? `Person ${index + 1}` : "Character";
  const remove = $(".person-remove", card);
  remove.hidden = !several;
  remove.disabled = busy(project);
  const taps = people(project)[index]?.anchors ?? [];
  $(".taps", card).hidden = taps.length === 0;
  $(".taps-label", card).textContent = taps.length === 1
    ? "Tapped on 1 frame. A tap on another frame fixes the tracking from there."
    : `Tapped on ${taps.length} frames, each followed to the next.`;
  $(".clear-taps", card).disabled = busy(project) || state.tapBusy;
  const tiles = [
    ...pictures(project, index).map((picture, position) => pictureTile({
      key: picture.id, src: fileUrl(project.id, picture.file), label: ROLES[picture.role] || "View",
      first: position === 0, unused: position >= used, onRemove: () => removePicture(index, picture.id),
    })),
    ...theirs.map((file) => pictureTile({
      key: file.name + file.size, src: localUrl(file), label: state.sendingSheet ? "Adding" : "New",
      busy: Boolean(state.sendingSheet),
      onRemove: state.sendingSheet ? null : () => {
        state.pendingSheet = state.pendingSheet.filter((item) => item !== file);
        if (!state.pendingSheet.length) state.pendingSheet = null;
        renderCharacter();
        renderChecklist();
      },
    })),
  ];
  const grid = $(".pictures", card);
  $(".slot", card).hidden = tiles.length > 0;
  grid.hidden = tiles.length === 0;
  const recent = state.recent.slice(0, 6);
  const showRecent = index === state.person && recent.length > 0;
  $(".recent", card).hidden = !showRecent;
  if (showRecent) {
    const recentKey = recent.map((item) => item.project + item.front).join("|");
    const list = $(".recent-list", card);
    if (list.dataset.key !== recentKey) {
      list.dataset.key = recentKey;
      list.replaceChildren(...recent.map((item) => el("button", {
        type: "button", class: "recent-item", title: "Use this character",
        onclick: (event) => {
          event.stopPropagation();
          state.person = index;
          useRecent(item);
        },
      }, el("img", { src: fileUrl(item.project, item.front), alt: "A recent character" }))));
    }
  }
  const full = tiles.length >= MAX_PICTURES || Boolean(state.sendingSheet);
  const key = `${tiles.map((tile) => tile.dataset.key).join("|")}|${full}|${used}`;
  if (grid.dataset.key === key) return;
  grid.dataset.key = key;
  grid.classList.toggle("many", tiles.length > 5);
  grid.replaceChildren(...tiles, ...(full ? [] : [addTile()]));
}

// While pictures upload, and then while the server finds the views on them.
function renderCharacterStatus() {
  [...$(".people").children].forEach((card, index) => {
    const box = $(".character-status", card);
    const sending = Boolean(state.sendingSheet) && index === state.sheetPerson;
    box.hidden = !sending;
    if (!sending) return;
    const fraction = state.sheetProgress;
    const uploading = fraction != null && fraction < 1;
    const count = state.sendingSheet.length;
    $(".character-status-label", box).textContent = uploading
      ? `Uploading ${count === 1 ? "the picture" : `${count} pictures`} · ${Math.round(fraction * 100)}%`
      : "Finding the views on the pictures…";
    const meter = $(".meter", box);
    meter.classList.toggle("indeterminate", !uploading);
    $("span", meter).style.width = uploading ? `${Math.round(fraction * 100)}%` : "";
  });
}

function pictureTile({ key, src, label, first = false, unused = false, busy = false, onRemove }) {
  const classes = `picture${first ? " first" : ""}${unused ? " unused" : ""}${busy ? " busy" : ""}`;
  const tile = el("figure", {
    class: classes, title: unused ? `H3 takes ${MAX_PICTURES} pictures in all, so this one isn't used` : null,
  },
    el("img", { src, alt: `${label} view of the character` }),
    first ? el("span", { class: "picture-index", title: "Picture 1" }, "1") : null,
    el("figcaption", {}, label),
    onRemove ? el("button", {
      type: "button", class: "picture-remove", "aria-label": `Remove this ${label.toLowerCase()} picture`,
      title: "Remove", onclick: onRemove,
    }, "×") : null);
  tile.dataset.key = key;
  return tile;
}

function addTile() {
  const input = el("input", { type: "file", accept: "image/*", multiple: true, "aria-label": "Add pictures of the character" });
  input.addEventListener("change", () => chooseCharacter([...input.files]));
  return el("label", { class: "picture picture-add", title: "Add pictures" }, input, el("span", { "aria-hidden": "true" }, "+"));
}

// Characters used before, a click away.
async function loadRecent() {
  const current = state.project ? `?project=${encodeURIComponent(state.project.id)}` : "";
  try {
    state.recent = await api(`/api/characters${current}`);
  } catch {
    return;
  }
  renderCharacter();
}

async function useRecent(item) {
  showError("");
  try {
    const files = await Promise.all(item.sheet.map(async (path) => {
      const response = await fetch(fileUrl(item.project, path));
      if (!response.ok) throw new Error("That character's pictures are no longer there.");
      const blob = await response.blob();
      return new File([blob], path.split("/").pop(), { type: blob.type });
    }));
    chooseCharacter(files, { replace: true });
  } catch (failure) {
    showError(failure.message);
  }
}

// Files dropped or pasted anywhere go where they belong: a clip is the
// footage, images are the character.
function takeFiles(files) {
  const video = files.find(isVideo);
  const images = files.filter((file) => file.type.startsWith("image/"));
  if (video) chooseFootage(video);
  if (images.length) chooseCharacter(images);
}

function setupDropAnywhere() {
  const carriesFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  let depth = 0;
  document.addEventListener("dragenter", (event) => {
    if (!carriesFiles(event)) return;
    depth += 1;
    document.body.classList.add("dragging");
  });
  document.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) document.body.classList.remove("dragging");
  });
  document.addEventListener("dragover", (event) => {
    if (carriesFiles(event)) event.preventDefault();
  });
  document.addEventListener("drop", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    depth = 0;
    document.body.classList.remove("dragging");
    takeFiles([...event.dataTransfer.files]);
  });
  document.addEventListener("paste", (event) => {
    const files = [...(event.clipboardData?.files || [])];
    if (!files.length || event.target.closest("input, textarea")) return;
    event.preventDefault();
    takeFiles(files);
  });
}

function setupNotes() {
  const notes = $("textarea[name=description]");
  notes.addEventListener("input", () => {
    state.descriptionDirty = true;
    renderPromptMarks();
    clearTimeout(state.syncTimer);
    state.syncTimer = setTimeout(syncEdits, SYNC_DELAY_MS);
  });
  notes.addEventListener("scroll", () => {
    $(".prompt-marks").scrollTop = notes.scrollTop;
  });
  setupTagMenu(notes);
}

/* The @ menu: typing @ in the prompt lists the characters to tag */

function setupTagMenu(notes) {
  notes.addEventListener("input", updateTagMenu);
  notes.addEventListener("click", updateTagMenu);
  notes.addEventListener("blur", closeTagMenu);
  notes.addEventListener("keydown", (event) => {
    if (!state.tagMenu) return;
    const count = $(".tag-menu").children.length;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      state.tagMenu.active = (state.tagMenu.active + (event.key === "ArrowDown" ? 1 : count - 1)) % count;
      updateTagMenu();
    } else if (event.key === "Enter" || event.key === "Tab") {
      insertTag(state.tagMenu.active);
    } else if (event.key === "Escape") {
      closeTagMenu();
    } else {
      return;
    }
    event.preventDefault();
  });
  // The menu is fixed to the window, so it closes when the page under it scrolls.
  document.addEventListener("scroll", (event) => {
    if (!event.target.closest?.(".prompt-box")) closeTagMenu();
  }, true);
  window.addEventListener("resize", closeTagMenu);
}

// Open while the caret is right after an @ that starts a word.
function updateTagMenu() {
  const notes = $("textarea[name=description]");
  const before = notes.value.slice(0, notes.selectionStart);
  if (notes.selectionStart !== notes.selectionEnd || !/(^|[\s(])@$/.test(before)) {
    closeTagMenu();
    return;
  }
  const count = Math.max(1, people().length);
  const active = Math.min(state.tagMenu?.active ?? 0, count - 1);
  state.tagMenu = { active };
  const menu = $(".tag-menu");
  menu.replaceChildren(...Array.from({ length: count }, (_, index) => tagOption(index, index === active, count > 1)));
  menu.hidden = false;
  const box = notes.getBoundingClientRect();
  const below = box.bottom + 6 + menu.offsetHeight <= innerHeight;
  menu.style.left = `${box.left}px`;
  menu.style.width = `${box.width}px`;
  menu.style.top = `${below ? box.bottom + 6 : box.top - 6 - menu.offsetHeight}px`;
}

function tagOption(index, active, several) {
  const picture = pictures(state.project, index)[0];
  const pending = cardFiles(index)[0];
  const src = picture ? fileUrl(state.project.id, picture.file) : pending ? localUrl(pending) : null;
  return el("li", {
    role: "option", class: "tag-option", "aria-selected": String(active),
    onpointerdown: (event) => event.preventDefault(), // keeps the caret in the prompt
    onclick: () => insertTag(index),
  },
  src ? el("img", { src, alt: "" }) : el("span", { class: "tag-thumb" }),
  el("span", { class: "mask-badge", style: `--mask: ${MASK_COLORS[index]}` }, String(index + 1)),
  el("span", {}, several ? `Person ${index + 1}` : "Character"),
  el("span", { class: "tag-name" }, `@${index + 1}`));
}

// Turns the typed @ into @n, with a space after it unless one follows already.
function insertTag(index) {
  const notes = $("textarea[name=description]");
  const caret = notes.selectionStart;
  const space = /\s/.test(notes.value[caret] || "") ? "" : " ";
  notes.setRangeText(`@${index + 1}${space}`, caret - 1, caret, "end");
  closeTagMenu();
  notes.dispatchEvent(new Event("input"));
}

function closeTagMenu() {
  state.tagMenu = null;
  $(".tag-menu").hidden = true;
}

/* Generate */

function everyone(project, has) {
  const all = people(project);
  return all.length > 0 && all.every(has);
}

function renderChecklist() {
  const project = state.project;
  const ready = everyone(project, (person) => person.pictures.length > 0 && person.anchors.length > 0);
  $("#generate").disabled = !ready || busy(project);
}

function setupGenerate() {
  const saved = prefs.load().quality;
  state.quality = saved in RENDER_SECONDS_PER_SECOND ? saved : "full";
  $(`.quality input[value=${state.quality}]`).checked = true;
  $(".quality").addEventListener("change", (event) => {
    state.quality = event.target.value;
    prefs.save({ quality: state.quality });
    renderChecklist();
  });
  $("#terms-agree").addEventListener("change", (event) => {
    $("#terms-accept").disabled = !event.target.checked;
  });
  $("#generate").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    showError("");
    // Asked here, on a click, so the browser shows the prompt.
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission().catch(() => {});
    }
    try {
      if (!(await termsAgreed())) return;
      // Edits still waiting out their delay go first, or the take would miss them.
      clearTimeout(state.syncTimer);
      await syncEdits();
      applyProject(await api(projectPath("/render"), { method: "POST", json: { takes: 1, quality: state.quality } }));
      poll();
    } catch (failure) {
      showError(failure.message);
    } finally {
      renderChecklist();
    }
  });
}

// MiniMax H3's terms, agreed to once before the first render; the server keeps
// the agreement and asks again when the terms change.
async function termsAgreed() {
  const terms = await api("/api/terms");
  if (terms.accepted) return true;
  const dialog = $("#terms-dialog");
  const paragraphs = terms.notice.map((text) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    return paragraph;
  });
  $(".terms-notice", dialog).replaceChildren(...paragraphs);
  $(".license-link", dialog).href = terms.license_url;
  $(".authorization-link", dialog).href = terms.authorization_url;
  $("#terms-agree").checked = false;
  $("#terms-accept").disabled = true;
  dialog.returnValue = ""; // Esc keeps the last value, which may be an earlier "agree"
  dialog.showModal();
  await new Promise((resolve) => dialog.addEventListener("close", resolve, { once: true }));
  if (dialog.returnValue !== "agree") return false;
  await api("/api/terms", { method: "POST" });
  return true;
}

// The report line under Generate, once the address is set.
async function showReportLink() {
  try {
    const terms = await api("/api/terms");
    if (!terms.report_email) return;
    $(".report-link").href = `mailto:${terms.report_email}`;
    $(".powered .report").hidden = false;
  } catch {
    // Without the terms the line just stays hidden.
  }
}

/* Status */

function renderStatus() {
  const project = state.project;
  const box = $(".status");
  const activity = busy(project) ? project.activity : state.localActivity;
  box.hidden = !activity;
  if (!activity) return;
  updateClock();
}

// "Rendering · step 3 of 8" once the sampler reports its steps.
function activityLabel(project) {
  const take = (project.takes || []).find((item) => item.status === "rendering");
  const sampling = take && take.sampling;
  return sampling ? `${project.activity.label} · step ${sampling.last[0]} of ${sampling.steps}` : project.activity.label;
}

// How much of a take is done: its sampler steps once they come in, with one more
// step's worth for decoding the video; before that, from how long earlier takes
// took. Null until there is a measure.
function takeFraction(take) {
  if (take.status === "done" || take.status === "failed") return 1;
  if (take.status !== "rendering") return 0;
  const sampling = take.sampling;
  if (sampling) return Math.min(0.97, sampling.last[0] / (sampling.steps + 1));
  if (!take.started || !take.expected_seconds) return null;
  return Math.min(0.97, (Date.now() - new Date(take.started)) / 1000 / take.expected_seconds);
}

// Seconds left on a rendering take: the steps to go at the pace measured so
// far, plus one for decoding; before two steps are in, from earlier takes.
function takeSecondsLeft(take) {
  const sampling = take.sampling;
  if (sampling && sampling.last[0] > sampling.first[0]) {
    const [firstStep, firstAt] = sampling.first;
    const [lastStep, lastAt] = sampling.last;
    const perStep = (new Date(lastAt) - new Date(firstAt)) / 1000 / (lastStep - firstStep);
    const sinceLast = (Date.now() - new Date(lastAt)) / 1000;
    return Math.max(0, (sampling.steps - lastStep + 1) * perStep - sinceLast);
  }
  if (!take.started || !take.expected_seconds) return null;
  return Math.max(0, take.expected_seconds - (Date.now() - new Date(take.started)) / 1000);
}

function overallProgress(project) {
  const activity = project.activity;
  if (project.step !== "render") {
    const known = typeof activity.fraction === "number" && activity.fraction > 0;
    return { fraction: known ? activity.fraction : null, eta: null };
  }
  // Only this render's takes count; earlier ones are done. Before its seeds are
  // picked, it's still tracking or checking the pictures.
  const seeds = project.render && project.render.seeds;
  if (!seeds) return { fraction: null, eta: null };
  const takes = (project.takes || []).filter((take) => seeds.includes(take.seed));
  const current = takes.find((take) => take.status === "rendering");
  const left = current ? takeSecondsLeft(current) : null;
  // A whole take: this render's own pace once it's measured, else earlier renders'.
  const whole = left != null
    ? (Date.now() - new Date(current.started)) / 1000 + left
    : takes.find((take) => take.expected_seconds)?.expected_seconds;
  const queued = takes.filter((take) => take.status === "queued").length;
  const fractions = takes.map(takeFraction);
  const eta = current ? (left == null || (queued && !whole) ? null : left + queued * (whole || 0))
    : (whole ? queued * whole : null);
  return {
    fraction: fractions.some((value) => value == null) ? null
      : fractions.reduce((sum, value) => sum + value, 0) / takes.length,
    eta,
  };
}

// Runs every second, and after each poll, so progress moves between polls.
function updateClock() {
  const project = state.project;
  if (!busy(project)) {
    updateLocalClock();
    return;
  }
  const activity = project.activity;
  if (!activity) return;
  $(".status-label").textContent = activityLabel(project);
  const { fraction, eta } = overallProgress(project);
  const meter = $(".status .meter");
  meter.classList.toggle("indeterminate", fraction == null);
  $("span", meter).style.width = fraction == null ? "" : `${Math.round(fraction * 100)}%`;
  const parts = [];
  if (activity.started) parts.push(clock((Date.now() - new Date(activity.started)) / 1000));
  if (eta != null) parts.push(minutesLeft(eta));
  if (state.reconnecting) parts.push("reconnecting");
  $(".status-time").textContent = parts.join("  ·  ");
}

function updateLocalClock() {
  const activity = state.localActivity;
  if (!activity) return;
  $(".status-label").textContent = activity.label;
  $(".status .meter").classList.add("indeterminate");
  $(".status-time").textContent = clock((Date.now() - activity.started) / 1000);
}

function renderNotes() {
  const project = state.project;
  const notes = [...((project && project.warnings) || [])];
  const tracked = (project && project.track && project.track.people) || [];
  for (const [index, person] of tracked.entries()) {
    const who = tracked.length > 1 ? `Person ${index + 1} is` : "They're";
    for (const [start, end] of (person && person.lost) || []) {
      notes.push(`${who} lost at ${start.toFixed(1)}-${end.toFixed(1)}s. Pause there and tap them again.`);
    }
  }
  $(".notes").hidden = notes.length === 0;
  $(".notes ul").replaceChildren(...notes.map((note) => el("li", {}, ...withTimes(note))));
}

/* The top bar, Shoots and Queue: all drawn from the list of shoots */

async function loadShoots() {
  try {
    state.shoots = (await api("/api/projects")).filter((project) => project.mode === "swap");
  } catch {
    // The server may be restarting; the queue keeps asking.
    if (state.view === "queue") {
      clearTimeout(state.queueTimer);
      state.queueTimer = setTimeout(loadShoots, QUEUE_POLL_MS);
    }
    return;
  }
  renderShoots();
}

function renderShoots() {
  for (const tab of document.querySelectorAll(".nav .tab")) {
    tab.toggleAttribute("aria-current", tab.dataset.view === state.view);
    if (tab.hasAttribute("aria-current")) tab.setAttribute("aria-current", "page");
  }
  const jobs = state.shoots.filter((project) => project.status === "working").length;
  $(".nav-count").textContent = String(jobs);
  $(".nav-count").hidden = jobs === 0;
  if (state.view === "shoots") renderShootGrid();
  if (state.view === "queue") renderQueue();
}

function renderShootGrid() {
  const cards = state.shoots.map((project) => el("li", { class: "shoot-card" },
    el("a", { href: `?p=${encodeURIComponent(project.id)}`, "data-link": true },
      project.thumbnail
        ? el("img", { class: "shoot-poster", src: fileUrl(project.id, project.thumbnail), alt: "", loading: "lazy" })
        : el("span", { class: "shoot-poster" }),
      el("span", { class: "shoot-body" },
        el("span", { class: "shoot-title" }, project.title || "Untitled shoot"),
        el("span", { class: "shoot-meta" }, shootStatus(project), shortDate(project.created)),
        el("span", { class: "shoot-stats" }, shootStats(project)))),
    el("div", { class: "shoot-actions" },
      renameButton("shoot-action", (event) => editName($(".shoot-title", event.currentTarget.closest(".shoot-card")), project)),
      project.status === "working"
        ? null
        : el("button", { class: "shoot-action", type: "button", title: "Delete", "aria-label": "Delete this shoot", onclick: deleteShoot(project) },
          icon("i-clear")))));
  $(".shoot-grid").replaceChildren(...cards);
  $(".shoots-count").textContent = cards.length ? String(cards.length) : "";
  $("#view-shoots .page-empty").hidden = cards.length > 0;
}

// "17.5 s · 3 takes · best 88.9", from what the list of shoots carries.
function shootStats(project) {
  const parts = [];
  if (project.trim) parts.push(`${(project.trim[1] - project.trim[0]).toFixed(1)} s`);
  const takes = (project.takes || []).length;
  if (takes) parts.push(`${takes} take${takes === 1 ? "" : "s"}`);
  if (project.best_score != null) parts.push(`best ${project.best_score.toFixed(1)}`);
  return parts.join(" · ");
}

function renameButton(className, onclick) {
  return el("button", { class: className, type: "button", title: "Rename", "aria-label": "Rename this shoot", onclick },
    icon("i-edit"));
}

// Swaps a shoot's name for a text box: Enter or clicking away saves, Esc cancels.
function editName(label, project) {
  const input = el("input", { class: "name-input", value: project.title || "", maxlength: 80, "aria-label": "Shoot name", spellcheck: false });
  let done = false;
  const finish = async (keep) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    input.replaceWith(label);
    if (keep && name && name !== project.title) {
      label.textContent = name;
      try {
        const renamed = await api(`/api/projects/${encodeURIComponent(project.id)}`, { method: "PATCH", json: { title: name } });
        if (state.project && state.project.id === project.id) applyProject(renamed);
      } catch (failure) {
        label.textContent = project.title || "Untitled shoot";
        showError(failure.message);
      }
    }
    loadShoots();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") finish(true);
    if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
  // The box sits inside the shoot's link, which a click would otherwise follow.
  input.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  label.replaceWith(input);
  input.focus();
  input.select();
}

function shootStatus(project) {
  const label = { working: "Working", waiting: "Draft", failed: "Stopped", done: "Done" }[project.status] || project.status;
  return el("span", { class: `shoot-status ${project.status}` },
    project.status === "working" ? el("span", { class: "live", "aria-hidden": "true" }) : null, label);
}

// The first click asks, the second deletes; leaving the button cancels.
function deleteShoot(project) {
  return async (event) => {
    const button = event.currentTarget;
    if (!button.classList.contains("confirm")) {
      button.classList.add("confirm");
      button.textContent = "Delete";
      button.addEventListener("pointerleave", renderShoots, { once: true });
      return;
    }
    if (event.detail > 1) return; // a double-click's second click isn't a decision
    try {
      await api(`/api/projects/${encodeURIComponent(project.id)}`, { method: "DELETE" });
    } catch (failure) {
      showError(failure.message);
      renderShoots();
      return;
    }
    state.shoots = state.shoots.filter((shoot) => shoot.id !== project.id);
    renderShoots();
  };
}

// The job running now comes first, then the waiting ones in the order they were
// asked for, each with how long it should take and when it should start.
function queueJobs() {
  return state.shoots
    .filter((project) => project.status === "working" && project.activity)
    .sort((a, b) => Number(a.activity.label === WAITING) - Number(b.activity.label === WAITING)
      || new Date(a.activity.started) - new Date(b.activity.started));
}

function renderQueue() {
  const rows = queueJobs().map((project) => {
    const waiting = project.activity.label === WAITING;
    return el("li", { class: `job${waiting ? " waiting" : ""}`, "data-id": project.id },
      el("a", { href: `?p=${encodeURIComponent(project.id)}`, "data-link": true },
        project.thumbnail
          ? el("img", { class: "job-thumb", src: fileUrl(project.id, project.thumbnail), alt: "" })
          : el("span", { class: "job-thumb" }),
        el("span", { class: "job-text" },
          el("span", { class: "job-title" }, project.title || "Untitled shoot"),
          el("span", { class: "job-step" }),
          waiting ? null : el("div", { class: "meter" }, el("span"))),
        el("span", { class: "job-time" })));
  });
  $(".job-list").replaceChildren(...rows);
  $(".queue-count").textContent = rows.length ? String(rows.length) : "";
  $("#view-queue .page-empty").hidden = rows.length > 0;
  tickQueue();
  clearTimeout(state.queueTimer);
  state.queueTimer = setTimeout(loadShoots, QUEUE_POLL_MS);
}

// Runs every second, like the shoot page's clock, so times move between polls.
function tickQueue() {
  let ahead = 0; // seconds until the GPU is free for the next waiting job
  for (const project of queueJobs()) {
    const row = document.querySelector(`.job[data-id="${CSS.escape(project.id)}"]`);
    if (!row) continue;
    const waiting = project.activity.label === WAITING;
    let time;
    if (waiting) {
      const estimate = waitingEstimate(project);
      time = estimate == null ? "Waiting" : `Starts ${ahead ? `in about ${Math.ceil(ahead / 60)} min` : "next"}`;
      ahead += estimate || 0;
    } else {
      const { fraction, eta } = overallProgress(project);
      const meter = $(".meter", row);
      meter.classList.toggle("indeterminate", fraction == null);
      $("span", meter).style.width = fraction == null ? "" : `${Math.round(fraction * 100)}%`;
      time = [clock((Date.now() - new Date(project.activity.started)) / 1000), eta != null ? minutesLeft(eta) : null]
        .filter(Boolean).join("  ·  ");
      ahead = eta || 0;
    }
    $(".job-step", row).textContent = waiting ? queuedLabel(project) : activityLabel(project);
    $(".job-time", row).textContent = time;
  }
}

function queuedLabel(project) {
  if (project.step === "render" && project.render) {
    const takes = project.render.takes;
    const quality = (project.render.quality || "full") === "draft" ? "draft" : "full";
    return `${takes} ${quality} take${takes === 1 ? "" : "s"}, waiting`;
  }
  return project.step === "track" ? "Tracking, waiting" : "Waiting";
}

// A waiting render: its takes at the measured H100 speed; other steps are short.
function waitingEstimate(project) {
  if (project.step !== "render" || !project.render || !project.trim) return null;
  const speed = RENDER_SECONDS_PER_SECOND[project.render.quality || "full"];
  return project.render.takes * (project.trim[1] - project.trim[0]) * speed;
}

/* Takes: cards in the right column, the chosen one in the main viewer */

function doneTakes(project = state.project) {
  return ((project && project.takes) || []).filter((take) => take.video);
}

function viewedTake(project = state.project) {
  if (state.viewing == null) return null;
  return doneTakes(project).find((take) => take.seed === state.viewing) || null;
}

// The take the last render made, once it's done.
function newTake(project) {
  const seeds = (project.render && project.render.seeds) || [];
  return doneTakes(project).find((take) => seeds.includes(take.seed)) || null;
}

// Marked only when there's more than one to choose from.
function isBest(project, take) {
  return take.seed === project.best && doneTakes(project).length > 1;
}

function takeName(project, take) {
  return `Take ${project.takes.indexOf(take) + 1}`;
}

function draftTag(take) {
  return take.quality === "draft" ? el("span", { class: "draft-tag", title: "Rendered at 480p" }, "Draft") : null;
}

function view(seed) {
  showOriginal(false);
  state.viewing = seed;
  renderMonitor();
  renderViewerTabs();
  renderResult();
  renderTakes();
  if (seed != null) $(".monitor video").play().catch(() => {});
}

// Footage first, then a tab per finished take.
function renderViewerTabs() {
  const project = state.project;
  const items = [{ seed: null, label: "Original", tags: [] }, ...doneTakes(project).map((take) => ({
    seed: take.seed,
    label: takeName(project, take),
    tags: [draftTag(take), isBest(project, take) ? el("span", { class: "tab-best" }, "Best") : null],
  }))];
  $(".viewer-tabs").replaceChildren(...items.map((item) => el("button", {
    type: "button", role: "tab", "aria-selected": String(state.viewing === item.seed), onclick: () => view(item.seed),
  }, item.label, ...item.tags)));
}

function renderClipInfo() {
  const footage = footageReady() ? state.project.footage : null;
  $(".clip-info").textContent = footage
    ? `${footage.width} × ${footage.height} · ${Math.round(footage.fps)} fps · ${clock(footage.frames / footage.fps)}`
    : "";
}

// The inspector's result: one download for the take in the viewer, or the best.
function renderResult() {
  const project = state.project;
  const takes = doneTakes(project);
  const generate = $("#generate");
  generate.textContent = takes.length ? "Generate again" : "Generate";
  generate.classList.toggle("btn-primary", !takes.length);
  $(".result").hidden = !takes.length;
  if (!takes.length) return;
  const take = viewedTake(project) || takes.find((item) => item.seed === project.best) || takes[0];
  const number = project.takes.indexOf(take) + 1;
  const best = isBest(project, take);
  const link = $("#download");
  link.href = fileUrl(project.id, take.video);
  link.download = `rigless-take-${number}.mp4`;
  link.textContent = best ? "Download the best take" : `Download take ${number}`;
  const parts = [`Take ${number}`];
  if (take.quality === "draft") parts.push("draft");
  if (take.score != null) parts.push(`${take.score.toFixed(1)} motion score`);
  if (best) parts.push(`best of ${takes.length}`);
  $(".result-note").textContent = parts.join(" · ");
}

function notifyDone(take, project) {
  const body = take.score != null ? `It scored ${take.score.toFixed(1)}.` : "It's on the screen.";
  if (!document.hidden) return;
  document.title = "Take ready · Rigless";
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  try {
    new Notification(`${takeName(project, take)} is ready`, { body, tag: project.id });
  } catch {
    // Some browsers allow notifications only from a service worker; the title still says it.
  }
}

// Hold the button to see the original footage at the same moment.
function showOriginal(on) {
  const project = state.project;
  const original = $(".monitor-original");
  const button = $(".compare-hold");
  if (!on || !viewedTake(project)) {
    original.pause();
    original.hidden = true;
    button.classList.remove("active");
    return;
  }
  const src = fileUrl(project.id, project.driving);
  if (original.getAttribute("src") !== src) original.src = src;
  original.currentTime = footageTime();
  original.hidden = false;
  button.classList.add("active");
  if (!$(".monitor video").paused) original.play().catch(() => {});
}

function setupCompare() {
  const button = $(".compare-hold");
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    showOriginal(true);
  });
  for (const type of ["pointerup", "pointerleave", "pointercancel"]) {
    button.addEventListener(type, () => showOriginal(false));
  }
  button.addEventListener("click", (event) => event.stopPropagation()); // not a play toggle
  const main = $(".monitor video");
  main.addEventListener("timeupdate", () => {
    const original = $(".monitor-original");
    if (!original.hidden && Math.abs(original.currentTime - footageTime()) > 0.12) original.currentTime = footageTime();
  });
  main.addEventListener("pause", () => $(".monitor-original").pause());
  main.addEventListener("play", () => {
    const original = $(".monitor-original");
    if (!original.hidden) original.play().catch(() => {});
  });
}

// A take joins the top of the column once it's finished (or failed); until the
// first, the column says so.
function renderTakes() {
  const project = state.project;
  const takes = (project && project.takes) || [];
  const finished = takes.filter((take) => take.status === "done" || take.status === "failed");
  $(".takes-count").textContent = finished.length ? String(finished.length) : "";
  $(".takes-empty").hidden = finished.length > 0;
  syncKeyed($(".take-grid"), finished.reverse(), (take) => take.seed, createTake,
    (card, take) => updateTake(card, take, project));
}

function createTake() {
  const parts = {
    number: el("span", { class: "slate-take" }),
    seed: el("span", { class: "slate-seed" }),
    frame: el("button", { class: "take-frame", type: "button" }),
    score: el("div", { class: "take-score" }),
    flags: el("div", { class: "flags" }),
    trace: el("div", { class: "trace-slot" }),
    error: el("p", { class: "take-error", hidden: true }),
  };
  const card = el("article", { class: "take" },
    el("header", { class: "slate" }, parts.number, parts.seed),
    parts.frame, parts.score, parts.flags, parts.trace, parts.error);
  card.parts = parts;
  return card;
}

// Updates a take in place: re-inserting a video element would stop it playing.
function updateTake(card, take, project) {
  const { parts } = card;
  const index = project.takes.indexOf(take);
  const best = isBest(project, take);
  card.classList.toggle("best", best);
  card.classList.toggle("viewing", state.viewing === take.seed);
  parts.number.replaceChildren(takeName(project, take), ...[draftTag(take),
    best ? el("span", { class: "best-badge" }, "Best") : null].filter(Boolean));
  parts.seed.textContent = `Seed ${take.seed}`;

  const video = take.video ? fileUrl(project.id, take.video) : "";
  parts.frame.hidden = !video;
  if (video && card.dataset.media !== video) {
    card.dataset.media = video;
    parts.frame.replaceChildren(
      el("video", { src: video, muted: true, loop: true, playsInline: true, autoplay: true, preload: "metadata" }));
    parts.frame.onclick = () => view(take.seed);
    parts.frame.setAttribute("aria-label", `Show take ${index + 1} in the viewer`);
  }

  renderScore(parts.score, take, best, project.id);
  parts.flags.replaceChildren(...[
    take.frozen && el("span", { class: "flag" }, icon("i-alert"), "Frozen"),
    take.mirrored && el("span", { class: "flag" }, icon("i-alert"), "Mirrored"),
  ].filter(Boolean));
  if (take.per_frame && !parts.trace.firstChild) parts.trace.append(...trace(take));
  parts.error.hidden = take.status !== "failed";
  parts.error.textContent = take.status === "failed" ? take.error || "This take failed." : "";
}

function renderScore(box, take, best, projectId) {
  if (take.score == null) {
    box.replaceChildren();
    return;
  }
  const key = `${projectId}:${take.seed}`;
  if (box.dataset.key === key && box.dataset.best === String(best)) return;
  box.dataset.key = key;
  box.dataset.best = String(best);
  const value = el("span", { class: "score-value" }, take.score.toFixed(1));
  box.replaceChildren(value, el("span", { class: "score-unit" }, "motion score"));
  if (!state.revealed.has(key) && !reducedMotion.matches) countUp(value, take.score);
  state.revealed.add(key);
}

function countUp(node, target) {
  const start = performance.now();
  const duration = 900;
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    node.textContent = (target * (1 - (1 - t) ** 3)).toFixed(1);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// The per-frame score as a trace.
function trace(take) {
  const values = take.per_frame;
  const scored = values.filter((value) => value != null);
  const lowest = Math.min(...scored, 100);
  const floor = Math.min(60, Math.floor(lowest / 10) * 10);
  const width = Math.max(values.length - 1, 1);
  const height = 40;
  const y = (value) => height - ((value - floor) / (100 - floor)) * height;

  const chart = svg("svg", { class: "trace", viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none", role: "img",
    "aria-label": scored.length ? `Motion score per frame, lowest ${lowest.toFixed(0)}` : "No frames scored" });
  chart.append(svg("line", { class: "guide", x1: 0, x2: width, y1: y(100), y2: y(100) }));
  chart.append(svg("line", { class: "guide", x1: 0, x2: width, y1: height, y2: height }));

  // Frames with no score break the line instead of bridging the gap.
  let run = [];
  const flush = () => {
    if (run.length) chart.append(svg("polyline", { class: "line", points: run.join(" ") }));
    run = [];
  };
  values.forEach((value, index) => {
    if (value == null) flush();
    else run.push(`${index},${y(value).toFixed(2)}`);
  });
  flush();

  const legend = el("div", { class: "trace-legend" },
    el("span", {}, "Per frame"),
    el("span", {}, scored.length ? `Low ${lowest.toFixed(1)}` : ""));
  return [chart, legend];
}

/* Keeps a container's children in step with a list, reusing nodes by key. */
function syncKeyed(container, items, keyOf, create, update) {
  const existing = new Map([...container.children].filter((child) => child.dataset.key).map((child) => [child.dataset.key, child]));
  items.forEach((item, index) => {
    const key = String(keyOf(item));
    let node = existing.get(key);
    if (node) existing.delete(key);
    else {
      node = create(item);
      node.dataset.key = key;
    }
    update(node, item);
    if (container.children[index] !== node) container.insertBefore(node, container.children[index] || null);
  });
  for (const node of existing.values()) node.remove();
}

/* ComfyUI: where Rigless renders, a RunPod pod, this computer or a custom address, in a dialog */

const TABS = ["runpod", "local", "custom"];

function openComfyUI() {
  const dialog = $("#comfyui-dialog");
  if (state.connection?.target) showTab(state.connection.target);
  loadComfyUI();
  if (!dialog.open) dialog.showModal();
}

async function loadComfyUI() {
  try {
    renderTarget(await api("/api/connection"));
  } catch (failure) {
    setRunPodMessage(failure.message, "bad");
  }
}

function showTab(tab) {
  for (const input of document.querySelectorAll(".target-choice input")) input.checked = input.value === tab;
  for (const name of TABS) $(`#${name}`).hidden = name !== tab;
}

// COMFY_BASE_URL in .env ("custom") overrides the choice, so the other tabs are
// off until it's removed. Without it, the Custom tab only says how to set it.
function renderTarget({ target, url }) {
  const custom = target === "custom";
  for (const input of document.querySelectorAll(".target-choice input")) {
    input.disabled = custom && input.value !== "custom";
  }
  $(".custom-status").hidden = !custom;
  $(".custom-on").hidden = !custom;
  $(".custom-off").hidden = custom;
  if (custom) $(".custom-url").textContent = url;
  showTab(target);
  clearTimeout(state.podTimer);
  if (target === "runpod") loadRunPod();
}

// RunPod: at most one pod, through the app's server with the key from here.
const runpodKey = {
  get() {
    try {
      return localStorage.getItem(RUNPOD_KEY) || "";
    } catch {
      return "";
    }
  },
  set(value) {
    try {
      if (value) localStorage.setItem(RUNPOD_KEY, value);
      else localStorage.removeItem(RUNPOD_KEY);
    } catch {
      // Blocked storage (a private window): the key lasts until the page closes.
    }
    runpodKey.memory = value;
  },
  memory: "",
};

function currentKey() {
  return runpodKey.get() || runpodKey.memory;
}

async function runpodCall(path, options = {}) {
  return api(`/api/runpod${path}`, { ...options, headers: { "X-RunPod-Key": currentKey() } });
}

async function loadRunPod() {
  clearTimeout(state.podTimer);
  const key = currentKey();
  $("#runpod-key").hidden = Boolean(key);
  $("#forget-key").hidden = !key;
  if (!key) {
    for (const part of ["#runpod .pod-status", "#runpod-start", ".pod-costs", "#deploy", "#pause-pod", "#resume-pod", "#delete-pod"]) {
      $(part).hidden = true;
    }
    return;
  }
  // Nothing shown yet: say so while RunPod answers.
  const blank = $("#runpod .pod-status").hidden && $("#runpod-start").hidden;
  if (blank) {
    $("#runpod .pod-status").replaceChildren(el("span", { class: "pod-state" }, spinner(), "Checking RunPod…"));
    $("#runpod .pod-status").hidden = false;
  }
  try {
    renderRunPod(await runpodCall(""));
  } catch (failure) {
    if (blank) $("#runpod .pod-status").hidden = true;
    setRunPodMessage(failure.message, "bad");
  }
}

function renderRunPod(runpod) {
  const { pod, gpus } = runpod;
  $("#runpod .pod-status").hidden = !pod;
  $(".pod-costs").hidden = !pod;
  $("#runpod-start").hidden = Boolean(pod);
  $("#deploy").hidden = Boolean(pod);
  $("#delete-pod").hidden = !pod;
  $("#pause-pod").hidden = !pod || pod.paused;
  $("#resume-pod").hidden = !pod || !pod.paused;
  if (pod) {
    renderPod(pod);
    state.podTimer = setTimeout(() => {
      checkConnection();
      if ($("#comfyui-dialog").open) loadRunPod();
    }, pod.progress ? SETUP_POLL_MS : POD_POLL_MS);
    return;
  }
  state.podStarted = null;
  const select = $("#runpod-start select");
  const chosen = select.value;
  select.replaceChildren(...gpus.map((gpu) => el("option", { value: gpu.id, selected: gpu.id === chosen || null },
    `${gpu.name.replace(/^NVIDIA /, "")} · ${gpu.memory} GB · $${gpu.price.toFixed(2)}/hr · ${gpu.stock ? `${gpu.stock} stock` : "none free"}`)));
  if (!chosen) {
    const first = gpus.find((gpu) => gpu.stock);
    if (first) select.value = first.id;
  }
}

// A pod is setting up, ready, or paused. Pausing keeps its disk, billed while idle.
// The card is updated in place, so its spinner keeps turning and its meter slides
// instead of starting over at every poll.
function renderPod(pod) {
  const card = $("#runpod .pod-status");
  if (!$(".pod-gpu", card)) {
    card.replaceChildren(
      el("strong", { class: "pod-gpu" }),
      el("span", { class: "muted" }, el("span", { class: "pod-price" }), el("span", { class: "pod-uptime" })),
      el("span", { class: "pod-state" }, el("span", { class: "state-text" })),
      el("div", { class: "meter", hidden: true }, el("span")),
      el("code", { class: "muted pod-ssh", title: "The setup log is /workspace/logs/setup.log" }),
    );
  }
  const idle = `$${pod.idle_cost_per_hour.toFixed(2)}/hr`;
  state.podStarted = !pod.paused && pod.started ? new Date(pod.started) : null;
  $(".pod-gpu", card).textContent = pod.gpu || "Your pod";
  $(".pod-price", card).textContent = pod.paused ? idle : `$${(pod.cost_per_hour || 0).toFixed(2)}/hr`;
  tickPod();

  const { tone, busy, text, fraction } = podState(pod);
  const line = $(".pod-state", card);
  line.className = `pod-state ${tone}`;
  const turning = $(".spinner", line);
  if (busy && !turning) line.prepend(spinner());
  if (!busy && turning) turning.remove();
  $(".state-text", line).textContent = text;
  const meter = $(".meter", card);
  meter.hidden = fraction === null;
  if (fraction !== null) $("span", meter).style.width = `${Math.min(100, fraction * 100)}%`;

  const ssh = $(".pod-ssh", card);
  ssh.textContent = pod.ssh || "";
  ssh.hidden = !pod.ssh || pod.paused;
  $(".pod-costs").textContent = pod.paused
    ? `Its disk keeps the models for ${idle} until you delete it. Resume brings it back in a few minutes if its machine's GPU is free.`
    : `Pause to stop paying for the GPU. The disk keeps the models for ${idle}, and the pod resumes in a few minutes. Delete removes everything.`;
}

// What a pod is doing, from the progress its setup reports while it sets up.
function podState(pod) {
  const progress = pod.progress;
  const shown = (text, { tone = "", busy = false, fraction = null } = {}) => ({ text, tone, busy, fraction });
  if (pod.paused) return shown("Paused");
  if (!progress && $("#connection").dataset.state === "ok") return shown("Ready", { tone: "ok" });
  if (progress?.stage === "failed") {
    return shown("The setup failed. Delete the pod and deploy a new one.", { tone: "bad" });
  }
  if (progress?.fraction != null) {
    const percent = Math.round(progress.fraction * 100);
    return shown(`Setting up · ${percent}% · ${minutesLeft(progress.seconds_left)}`,
      { busy: true, fraction: progress.fraction });
  }
  return shown(progress ? "Setting up" : "Starting the pod", { busy: true });
}

// Runs every second, so the uptime counts up between polls.
function tickPod() {
  const uptime = $(".pod-uptime");
  if (uptime) uptime.textContent = state.podStarted ? ` · up ${clock((Date.now() - state.podStarted) / 1000)}` : "";
}

function setRunPodMessage(text, tone) {
  const message = $(".runpod-message");
  message.textContent = text;
  message.className = `settings-message runpod-message ${tone}`;
}

function setupRunPod() {
  $("#runpod-key").addEventListener("submit", (event) => {
    event.preventDefault();
    runpodKey.set(event.currentTarget.elements.key.value.trim());
    event.currentTarget.reset();
    setRunPodMessage("", "");
    loadRunPod();
  });
  $("#forget-key").addEventListener("click", () => {
    runpodKey.set("");
    setRunPodMessage("Key forgotten.", "ok");
    loadRunPod();
  });
  $("#runpod-start").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#deploy");
    button.disabled = true;
    setRunPodMessage("Starting…", "");
    try {
      const gpu = event.currentTarget.elements.gpu.value;
      renderRunPod(await runpodCall("/pod", { method: "POST", json: { gpu } }));
      setRunPodMessage("", "");
      loadComfyUI();
      checkConnection();
    } catch (failure) {
      setRunPodMessage(failure.message, "bad");
    } finally {
      button.disabled = false;
    }
  });
  $("#pause-pod").addEventListener("click", (event) => podAction(event.currentTarget, "/pod/pause", "POST",
    "Paused. The GPU is released, so only the disk is billed now."));
  $("#resume-pod").addEventListener("click", (event) => podAction(event.currentTarget, "/pod/resume", "POST",
    "Resuming. It's back in a few minutes.",
    "If its machine's GPU is taken, delete the pod and deploy a fresh one."));
  // The first click asks. The second removes the pod and its disk.
  $("#delete-pod").addEventListener("click", (event) => {
    const button = event.currentTarget;
    if (!button.classList.contains("btn-danger")) {
      button.classList.add("btn-danger");
      button.textContent = "Delete the pod";
      button.addEventListener("pointerleave", () => {
        button.classList.remove("btn-danger");
        button.textContent = "Delete";
      }, { once: true });
      return;
    }
    if (event.detail > 1) return; // a double-click's second click isn't a decision
    button.classList.remove("btn-danger");
    button.textContent = "Delete";
    podAction(button, "/pod", "DELETE", "Deleted. The pod and its disk are gone, so nothing more is billed.");
  });
}

async function podAction(button, path, method, done, hint = "") {
  button.disabled = true;
  try {
    renderRunPod(await runpodCall(path, { method }));
    setRunPodMessage(done, "ok");
    checkConnection();
  } catch (failure) {
    setRunPodMessage(`${failure.message} ${hint}`.trim(), "bad");
  } finally {
    button.disabled = false;
  }
}

function setupComfyUI() {
  setupRunPod();
  $("#connection").addEventListener("click", openComfyUI);
  $("#comfyui-dialog").addEventListener("close", () => {
    clearTimeout(state.podTimer);
    // Closed, it shouldn't open again on a reload.
    const params = new URLSearchParams(location.search);
    if (params.get("view") !== "comfyui") return;
    params.delete("view");
    history.replaceState(null, "", params.size ? `?${params}` : "/");
  });
  $(".target-choice").addEventListener("change", async (event) => {
    const tab = event.target.value;
    showTab(tab);
    if (tab === "custom") return;
    showConnecting(tab);
    try {
      renderTarget(await api("/api/connection", { method: "PUT", json: { target: tab } }));
    } catch (failure) {
      setRunPodMessage(failure.message, "bad");
      loadComfyUI();
    }
    checkConnection();
  });
  $("#test-local").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    showConnecting("local");
    await checkConnection();
    button.disabled = false;
  });
}

/* Start */

function init() {
  setupDropAnywhere();
  setupMonitor();
  setupTrim();
  setupCharacter();
  setupNotes();
  setupGenerate();
  setupCompare();
  setupComfyUI();
  $(".rename-title").addEventListener("click", () => editName(pageTitle, state.project));
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-link]");
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
    event.preventDefault();
    navigate(link.getAttribute("href"));
  });
  window.addEventListener("popstate", route);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !busy()) document.title = "Rigless";
  });
  setInterval(() => {
    updateClock();
    tickPod();
    if (state.view === "queue") tickQueue();
  }, 1000);
  route();
  checkConnection();
  showReportLink();
}

init();
