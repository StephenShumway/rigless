"""Splits a character sheet into the views the pipeline needs.

A dev tool until the app exists; it runs on the CPU and needs no ComfyUI. Each
run writes runs/sheet-<time>/ with front.png (compose.py's --character),
close_up.png (its --close-up) and back.png when the sheet has them, views.png
(the sheet with every figure boxed and the picks in red) to check by eye, and
views.json with the boxes, picks, and warnings.

Typical usage example:

  uv run python scripts/sheet.py --sheet local_fixtures/character_sheet.jpg
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from rigless import config, sheet

PICKED, OTHER = (220, 0, 0), (0, 160, 0)


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description="Split a character sheet into views.")
    parser.add_argument("--sheet", type=Path, required=True, help="character sheet")
    return parser.parse_args()


def main() -> None:
    """Saves the sheet's front, close-up, and back views and records the split."""
    args = parse_args()
    image = Image.open(args.sheet).convert("RGB")
    views = sheet.find_views(image)
    front = sheet.pick_front(views)
    picks = {
        "front": front,
        "close_up": sheet.pick_close_up(views),
        "back": sheet.pick_back(views),
    }

    run_dir = config.RUNS_DIR / datetime.now().strftime("sheet-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    records = []
    for view in views:
        name = next((name for name, pick in picks.items() if pick is view), None)
        draw.rectangle(view.box, outline=PICKED if name else OTHER, width=3)
        if name:
            draw.text((view.box[0] + 6, view.box[1] + 4), name, PICKED, font_size=20)
            sheet.crop(image, views, view).save(run_dir / f"{name}.png")
        records.append({"box": view.box, "full_body": view.full_body, "pick": name})
    overview.save(run_dir / "views.png")

    warning = sheet.pose_warning(front)
    record = {
        "sheet": args.sheet.as_posix(),
        "views": records,
        "warnings": [warning] if warning else [],
    }
    (run_dir / "views.json").write_text(json.dumps(record, indent=2), "utf-8")
    if warning:
        print(f"Warning: {warning}")
    print(run_dir)


if __name__ == "__main__":
    main()
