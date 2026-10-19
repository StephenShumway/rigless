# Changelog

## [Unreleased]

### Fixed

- Pose guidance now covers the whole sampling run. Inside our exported workflow's
  subgraph, `pose_end_percent` was wired to the `pose_start_percent` input, so the window
  was 0 to 0 and the model saw the driving video for at most the first step. Our workflow
  sets it to 1.0, ComfyUI's trained default, in both the API file (the value) and the UI
  file (the stray link removed), which turned loose, unrelated motion into close tracking.

### Added

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
