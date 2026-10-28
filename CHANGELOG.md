# Changelog

## [Unreleased]

### Added

- A web app (`uv run rigless`) that replaces a person in a clip with a character, on
  one screen: drop a clip, trim it, click the person to replace, add the character's
  pictures, and generate. SAM 3.1 finds the person from the clicks and tracks them
  through the trimmed part, with a tinted preview and the stretches where they were
  lost. Takes play in the main viewer, with a hold-to-compare against the original.
- The swap itself: MiniMax H3 redraws only the tracked person, in overlapping windows,
  with every one of the character's pictures (up to 9) as references, so the scene and
  everyone else stay as filmed. It runs the Rigless Swap LoRA, our fine-tune of H3
  Relay's Character Swap LoRA on real footage, and puts the clip's own sound back.
- No prompt needed: before each render Qwen3-VL-8B describes the character from their
  pictures and checks, item by item, what the replaced person wears that the character
  doesn't, so H3 neither turns a monocle into glasses nor carries over the performer's
  rings.
- Best of N: every take is scored for how closely it follows the performance, and the
  best is picked automatically; frozen or mirrored takes are flagged.
- A side panel of shoots to switch between, start, and delete.
- `deploy/runpod/setup.sh` sets up a RunPod pod on a network volume (about 60GB of
  models on the first start) and starts ComfyUI behind a token-protected proxy;
  `scripts/download_models.py` fetches the models on any machine.

### Removed

- The Wan Animate 2 motion-transfer pipeline, its workflows, GPU profiles, the
  reference compose and relight steps, and their scripts. MiniMax H3 is the one model.
