"""Writes the ComfyUI workflows Rigless runs to workflows/, as API-format JSON.

Rigless builds each graph in code and sends it to ComfyUI through the Comfy
SDK. This writes one file per graph builder, with placeholder file names
where the app would upload the clip, masks and pictures, so each graph can be
queued or opened in ComfyUI on its own. The swap is the app's default render:
768p, three pictures (front, face and back) and the prompt for one character.
Rerun it after changing a graph builder; a test fails while the files are out
of date.

Typical usage example:

  uv run python scripts/export_workflows.py
"""

import argparse
import json
from pathlib import Path

from rigless import describe, selection, swap, video_prep
from rigless.comfy_client import Graph

FOLDER = Path(__file__).parents[1] / "workflows"
ROLES = ("front", "face", "back")  # the views the app takes from a character sheet
SEED = 1  # the app draws a new seed for each take
# One tap on the first frame, in the middle of a 1280x720 clip, the largest
# footage the app tracks in.
TAP = selection.Anchor(frame=0, positive=((640, 360),))
# The longest clip the app takes; ImageFromBatch trims the run to the clip.
FRAMES = round(video_prep.MAX_TRIM_SECONDS * swap.FPS)
DESCRIPTION_TOKENS = 256  # as describe.character asks


def workflows() -> dict[str, Graph]:
    """Returns each workflow by its file name, with placeholder inputs."""
    pictures = [f"picture_{number}.png" for number in range(1, len(ROLES) + 1)]
    return {
        "swap.json": swap.swap_graph(
            "input.mp4", "masks.mp4", pictures, prompt=swap.prompt(ROLES), seed=SEED
        ),
        "detect.json": selection.detect_graph("frame.png", TAP),
        "track.json": selection.track_graph(
            "input.mp4",
            None,
            selection.plan_segments([TAP], FRAMES),
            FRAMES,
            swap.FPS,
        ),
        "describe.json": describe.ask_graph(
            "collage.png", [describe.DESCRIBE], DESCRIPTION_TOKENS
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Write Rigless's ComfyUI workflows.")
    parser.add_argument(
        "--out", type=Path, default=FOLDER, help="the folder to write to (workflows/)"
    )
    return parser.parse_args()


def main() -> None:
    """Writes every workflow, replacing the files already there."""
    out = parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    for name, graph in workflows().items():
        (out / name).write_text(json.dumps(graph, indent=2) + "\n", "utf-8")
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
