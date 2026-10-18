"""Accuracy benchmark: compares rendered takes across settings variants.

Reads the takes in run folders made by render.py, scores any that have no
score yet (saving it to their run.json), and prints a Markdown table per clip
and variant: mean score (what a single random seed gets), best score (best of
the takes), and how many takes froze.

Typical usage example:

  uv run python scripts/bench.py runs/20261016-215249 runs/20261016-221352 \
      --by pose_end_percent pose_strength
"""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rigless import config, pose, workflow
from rigless.score import score_take


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Compare scored takes by setting.")
    parser.add_argument("runs", type=Path, nargs="+", help="run folders from render.py")
    parser.add_argument(
        "--by", nargs="+", default=["pose_end_percent"], help="settings to compare by"
    )
    return parser.parse_args()


def load_takes(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Returns every successful take's run.json, scoring takes that lack a score.

    Each record's settings are completed from the workflow it ran with, so
    runs from before a setting existed still group correctly.
    """
    records = []
    for run_dir in run_dirs:
        dancer = None
        for path in sorted(run_dir.glob("seed-*/run.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] != "done":
                continue
            if record.get("score") is None:
                dancer = dancer or pose.estimate(run_dir / "driving.mp4")
                result = score_take(dancer, pose.estimate(path.parent / "result.mp4"))
                record["score"] = result.as_record()
                path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            template = workflow.load(config.REPO_ROOT / record["workflow"])
            defaults = asdict(workflow.read_settings(template))
            record["settings"] = {**defaults, **record["settings"]}
            records.append(record)
    return records


def table(records: list[dict[str, Any]], by: list[str]) -> str:
    """Formats mean and best score and frozen count per clip and variant."""
    groups = defaultdict(list)
    for record in records:
        length = f"{record['seconds']:g}s" if record["seconds"] else "full"
        clip = f"{Path(record['video']).name} ({length})"
        variant = tuple(record["settings"][name] for name in by)
        groups[clip, variant].append(record["score"])
    lines = [
        f"| clip | {' | '.join(by)} | takes | mean score | best score | frozen |",
        "|---|" + "---|" * len(by) + "---|---|---|---|",
    ]
    for (clip, variant), scores in sorted(groups.items()):
        values = [s["score"] for s in scores]
        frozen = sum(s["frozen"] for s in scores)
        lines.append(
            f"| {clip} | {' | '.join(str(v) for v in variant)} | {len(values)} "
            f"| {sum(values) / len(values):.1f} | {max(values):.1f} "
            f"| {frozen}/{len(values)} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Scores the given runs and prints the comparison table."""
    args = parse_args()
    print(table(load_takes(args.runs), args.by))


if __name__ == "__main__":
    main()
