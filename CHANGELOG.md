# Changelog

## [0.1.1] - 2026-10-30

### Changed

- With one person, the clip now goes into H3 led by the character's `<Picture 1>` as it
  is, for 8 frames (a third of a second) cut from the take afterward. Before, it led
  with the clip's own opening played backward. H3's first window has nothing earlier to
  go on. On a clip that opens with the head out of frame, it drew the character's head
  into the first 2 seconds and could copy the performer's costume. With the picture
  first, the head stayed out on all 3 seeds tested, and the costume came out the
  character's own on 2 of them. On the README's runway dance nothing changed: the same
  motion score and no turns toward the camera. Clips with several people keep the old
  lead-in.
- New pods set up from the `v0.1.1` tag. Nothing changed on the pod, so pods set up
  from `v0.1.0` keep working as they are.

## [0.1.0] - 2026-10-29

The first release.

### Added

- A web app (`uv run rigless`) that swaps people in a clip for characters, on one
  screen. Drop a clip (up to 2 minutes), trim it (up to 20 seconds), click each person
  to replace (up to three), add each character's pictures, and generate. SAM 3.1
  tracks every person from the clicks, with a tinted preview that shows where it lost
  them. A tap on another frame fixes the tracking from there, and **Clear taps**
  starts a person over.
- Clips are converted to a steady 24 fps, H3's frame rate. Clips with uneven frame
  times, such as phone clips and screen recordings, keep every frame.
- Character pictures as you have them. A character sheet is split into its front, face
  and back views, and a photo of one person is used whole. One picture must show the
  whole body: without one, the app says so and Generate asks for it. Picture 1 is the
  full-body view facing the camera, or the back view when the clip opens from behind.
- The swap. MiniMax H3 redraws only the tracked people, in overlapping windows, with
  up to 9 pictures as references, the 8-step Turbo LoRA and the Rigless LoRA. Outside
  the people H3 keeps the clip's own latents, so the scene comes back from the footage.
  H3 hears the clip's sound, so mouths move with the song or speech.
- The clip goes in with its first 8 and last 8 frames played backward around it, cut
  from the take afterward, so H3's pull toward the pictures at the edges of a render
  lands on frames nobody sees.
- The Rigless LoRA (`rigless-lora-v1`), a fine-tune of akatz's H3 Relay Character Swap
  LoRA v1.5 on real footage, preferred over v1.5 in a blind test (3 clips, one take
  each, judged by its developer).
- No prompt needed. Qwen3-VL 8B describes each character from their pictures and lists
  what the performer has that the character doesn't ("no rings, no watch"), so H3
  doesn't carry it over. A prompt of your own can name each character with **@**.
- **Draft** (480p) and **Full** (768p) takes that add up. Each gets a motion score
  against the performance, frame by frame, frozen and mirrored takes are flagged, and
  the best take is marked. Hold to compare a take with the original. Each take gets
  the clip's own sound back and says in its metadata that it's AI-generated.
- RunPod from inside the app. Deploy a GPU with your own API key, follow its setup with
  a time estimate, then pause, resume or delete it. New pods set up from this release's
  tag. Or use your own NVIDIA GPU, or any ComfyUI behind comfy-api-proxy.
- MiniMax H3's terms before the first render, with the address for reporting misuse.
- The ComfyUI workflows Rigless runs (the H3 swap, SAM 3.1 detection and tracking, the
  Qwen3-VL describer) in `workflows/`, regenerated from the code by
  `scripts/export_workflows.py`.
- `deploy/runpod/setup.sh` sets up a pod: the models (about 67 GB) on its disk, the
  Python environments on its container disk, and ComfyUI behind a token-protected
  proxy. `scripts/download_models.py` fetches the models on any machine.

[0.1.1]: https://github.com/StephenShumway/rigless/releases/tag/v0.1.1
[0.1.0]: https://github.com/StephenShumway/rigless/releases/tag/v0.1.0
