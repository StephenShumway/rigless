# Changelog

## [Unreleased]

### Changed

- Wan Animate 2's distilled model is the default, run as its makers do: 10 Euler steps
  with the LightX2V LoRA off. On the full 13.9s test it scores 90.4 against the base
  model's 93.9 and takes ~1.5x as long, but keeps finer detail and looks more like real
  footage. The base model's recipe (6 LCM steps with LightX2V) is `render.py --standard`
  (`workflow.standard`), and `download_models.py --standard` fetches its file;
  `--bf16` now fetches the bf16 distilled model.
- The motion-transfer workflow no longer saves a side-by-side comparison video. It held
  a second full-length copy of every frame and ran a 48GB pod out of memory on a 14s
  clip.
- The `cloud-48gb` profile uses the int8 model, as verified on an A40; it named the
  bf16 model, which the pod didn't have.

### Fixed

- RunPod: ComfyUI starts with `--disable-pinned-memory`. It sizes its RAM caches from
  the host's memory, not the pod's limit, so pinned model copies piled up until the
  container was killed mid-render.

- Pose guidance now covers the whole sampling run. Inside our exported workflow's
  subgraph, `pose_end_percent` was wired to the `pose_start_percent` input, so the window
  was 0 to 0 and the model saw the driving video for at most the first step. Our workflow
  sets it to 1.0, ComfyUI's trained default, in both the API file (the value) and the UI
  file (the stray link removed), which turned loose, unrelated motion into close tracking.

### Added

- New setting `sampler` (`render.py --sampler`), the sampling method.
- Composing references in one call (`pipeline.compose_references`), with a generic
  instruction built from the character description (`compose.prompt`) that matched the
  hand-tuned prompt's quality; `sheet.extract` returns all the sheet's views at once.
- Input quality gate (`framing.check`): warns in plain language, with times and a fix,
  when the head, a hand, or the feet leave the frame, when someone else at least half
  the dancer's height is in it, when the dancer can't be seen, or when the pose is too
  blurry to read. Warnings never block a render.
- Auto-reframe (`framing.reframe`, on by default; `render.py --no-reframe`): one fixed
  window around everything the dancer reaches, at the output's aspect ratio, so a small
  or off-center dancer fills the frame. Fixed rather than panning, because the
  generated background stands still and a panning crop would turn steps into sliding.
- Chunk-wise best-of-N (`pipeline.render_chunked`, `render.py --chunked`): every ~2.7s
  chunk is rendered with each seed, continuing from the best version of the chunk
  before, and the best is kept; same cost as one take per seed. Chunks are planned
  exactly as the workflow's loop plans them (`workflow.plan_chunks`), and each runs as
  its own job (`workflow.chunk_graph`).
- Real-footage LoRA in the motion-transfer workflow ("Real-footage LoRA" node at
  strength 2.0, after LightX2V): trained on the developer's own iPhone clips with
  musubi-tuner on Wan 2.1 I2V 720p, the model Wan Animate 2 was fine-tuned from. It
  adds camera-like motion blur on fast moves and flatter light. New setting
  `realism_lora_strength` (`render.py --realism-lora-strength`); 0 removes the node, so
  installs without the LoRA file still run. Training setup in `deploy/lora/` and
  `scripts/prepare_lora_clips.py` (upright, tone-mapped SDR clips with captions).
- `scripts/sheet.py` takes several images (say, a full-body front, a face close-up, and
  a back view) and picks the best views across them.
- Paste-and-relight tooling (`compose.paste`, `sheet.cutout`, `scripts/relight.py`,
  `compose.patch(denoise=...)`, and `compose.py` without `--character` for background-
  only edits); parked, since the cut edge left a light line around the character.
- Qwen-Image 2.1 as the default editor for composing the character into a background
  photo (`workflows/rigless_compose_qwen_image_2_1_api.json`, `compose.py --editor`,
  `download_models.py --qwen-image-21`). It keeps the sheet's suit, tie, and monocle in
  one pass, where Qwen-Image-Edit 2511 needed full sampling and a face-detail pass. Its
  license allows research and evaluation only; 2511 (Apache-2.0) remains available.
- Camera finish: every take also gets `finished.mp4`, with optical-flow motion blur and
  a slightly brighter, less saturated grade so it reads like phone footage
  (`video_prep.finish`). Scoring still uses the raw `result.mp4`, since the blur would
  inflate the motion score.
- Character sheet splitting (`rigless.sheet`, `scripts/sheet.py`): finds every figure
  with the pose estimator's person detector, separates it from the plain backdrop by
  color, and picks the front full-body view, the most frontal face close-up, and the back
  view by their keypoints, cropped with a margin of the sheet's backdrop. Warns when the
  front view's hands rest on the body, the pose frozen takes fell back to.
  `pose.find_people` returns every person in a still image.
- Reference composition for real backgrounds: Qwen-Image-Edit 2511 edits the character
  into a background photo with the scene's lighting, since Wan Animate 2 keeps its
  reference image's background. `workflows/rigless_compose_reference_api.json` is
  ComfyUI's int8 template flattened, with inputs for the background, the character's
  full-body view, and an optional face close-up; `rigless.compose` fits the photo to the
  output's aspect (leaving a band for the floor so the character fits head to shoes) and
  patches the workflow; `scripts/compose.py` makes one reference per seed, with `--full`
  for slower sampling without the Lightning LoRA, whose glossy look the video inherits.
- Face-detail pass for composed references (`scripts/detail_face.py`): in a full-body
  shot the face is too small for details such as a monocle, so the head is cut out,
  upscaled, re-edited with the sheet's face close-up, aligned back by face keypoints
  (`compose.similarity_transform`), and blended over the inner face only
  (`compose.blend_face`). `pose.face_keypoints` finds the face in a still image.
- `download_models.py --qwen-edit` fetches the Qwen-Image-Edit 2511 models (int8, which
  pre-Ada GPUs run natively, unlike FP8); RunPod setup includes them.

- RunPod deployment: `deploy/runpod/setup.sh` installs ComfyUI, the models, and
  comfy-api-proxy on a pod's volume; `deploy/runpod/start.sh` starts ComfyUI (pod-local
  only) behind the token-protected proxy.

- Motion-accuracy scoring: CPU pose estimation (RTMPose + YOLOX via rtmlib, ONNX Runtime)
  and a 0-100 score from limb angles, with a per-frame curve, frozen-take and mirroring
  flags, and time-based frame matching across frame rates.
- Every take is scored right after it renders; `pick_best` chooses the best take, and
  `render.py` reports it.
- `scripts/bench.py`: compares scored takes across settings (mean, best, frozen rate).

- Seed sweeps: `scripts/render.py` renders several takes per run into
  `runs/<id>/seed-<seed>/`, each with its own `run.json`; a failed take is recorded and the
  sweep continues.
- Every workflow setting as a parameter (prompts, pose and reference strength, steps, LoRA
  strength, fragment length, context windows, model, cache device, size), read from and
  written to the workflow through one role-based lookup. CLI flags and `--settings` TOML
  files override them.
- GPU profiles in `profiles/`: `local-10gb` (verified), `cloud-24gb`, `cloud-48gb`.
- Black-bar detection and cropping, clip length warnings, and video probing.
- `scripts/download_models.py`: resumable download of every model from Hugging Face.
- `scripts/render.py`: renders motion-transfer takes through the Comfy SDK into
  `runs/<id>/` (driving clip, result, side-by-side comparison, `run.json`).
- Workflow patching by node role rather than node id; UI-format exports are rejected.
- Driving-video prep with bundled ffmpeg (imageio-ffmpeg): scales the clip to the output
  size and keeps its aspect, frame rate, and audio, with an optional trim. This keeps
  ComfyUI's frame decoding from exhausting system RAM.
- `config.py` and `.env.example`: settings from `.env`, all optional for local runs.
- The Wan Animate 2 motion-transfer workflow in UI and API formats.

[Unreleased]: https://github.com/StephenShumway/rigless/compare/...HEAD
