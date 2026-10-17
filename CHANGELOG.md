# Changelog

## [Unreleased]

### Added

- `scripts/render.py`: renders one motion-transfer take through the Comfy SDK into
  `runs/<id>/` (driving clip, result, side-by-side comparison, `run.json`).
- Workflow patching by node role rather than node id; UI-format exports are rejected.
- Driving-video prep with bundled ffmpeg (imageio-ffmpeg): scales the clip to the output
  size and keeps its aspect, frame rate, and audio, with an optional trim. This keeps
  ComfyUI's frame decoding from exhausting system RAM.
- `config.py` and `.env.example`: settings from `.env`, all optional for local runs.
- The Wan Animate 2 motion-transfer workflow in UI and API formats.

[Unreleased]: https://github.com/StephenShumway/rigless/compare/...HEAD
