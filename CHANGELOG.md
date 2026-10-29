# Changelog

## [Unreleased]

### Added

- A web app (`uv run rigless`) that swaps people in a clip for characters, on one
  screen: drop a clip (up to 2 minutes) and trim it (up to 20 seconds), click each
  person to replace (up to three), add each character's pictures, and generate.
  SAM 3.1 tracks every person from the clicks, with a tinted preview that shows where
  it lost them; a tap on another frame fixes the tracking from there, and **Clear
  taps** starts a person over.
- Character pictures as you have them: a character sheet is split into its front,
  face and back views; a photo of one person is used whole. Picture 1 is the
  full-body view facing the camera, or the back view when the clip opens from behind.
- The swap: MiniMax H3 redraws only the tracked people, in overlapping windows, with
  up to 9 pictures as references, the 8-step Turbo LoRA and the Rigless LoRA. Outside
  the people H3 keeps the clip's own latents, so the scene comes back from the footage.
  Each take gets the clip's own sound back and says in its metadata that it's
  AI-generated.
- The Rigless LoRA (`rigless-lora-v1`), a fine-tune of akatz's H3 Relay Character Swap
  LoRA v1.5 on real footage, preferred over v1.5 in a blind test (3 clips, one take
  each, judged by its developer).
- No prompt needed: Qwen3-VL 8B describes each character from their pictures and lists
  what the performer has that the character doesn't ("no rings, no watch"), so H3
  doesn't carry it over. Notes of your own can name each character with **@**.
- **Draft** (480p) and **Full** (768p) takes that add up; each gets a motion score
  against the performance, frozen and mirrored takes are flagged, and the best take is
  marked. Hold to compare a take with the original.
- RunPod from inside the app: deploy a GPU with your own API key, follow its setup with
  a time estimate, then pause, resume or delete it. Or use your own NVIDIA GPU, or any
  ComfyUI behind comfy-api-proxy.
- MiniMax H3's terms before the first render, and an address for reporting misuse.
- The ComfyUI workflows Rigless runs (the H3 swap, SAM 3.1 detection and tracking, the
  Qwen3-VL describer) in `workflows/`, regenerated from the code by
  `scripts/export_workflows.py`.
- `deploy/runpod/setup.sh` sets up a pod: the models (about 67 GB) on its disk, the
  Python environments on its container disk, and ComfyUI behind a token-protected
  proxy. `scripts/download_models.py` fetches the models on any machine.

### Removed

- The Wan Animate 2 motion-transfer pipeline, its workflows, GPU profiles, the
  reference compose and relight steps, and their scripts. MiniMax H3 is the one model.
