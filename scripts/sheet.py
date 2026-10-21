"""Splits a character sheet, or a set of character images, into the views we need.

A dev tool until the app exists; it runs on the CPU and needs no ComfyUI. Give one
sheet or several images (say, a full-body front, a face close-up, and a back view):
the views of all of them are pooled, and the best front, close-up, and back are picked
across the lot. Each run writes runs/sheet-<time>/ with front.png (compose.py's
--character), front_cutout.png (the front view on a transparent background, for
relight.py), close_up.png (its --close-up) and back.png when found, views-<name>.png
per input (every figure boxed, the picks in red) to check by eye, and views.json
with the boxes, picks, and warnings.

Typical usage example:

  uv run python scripts/sheet.py --sheet local_fixtures/character_sheet.jpg
  uv run python scripts/sheet.py --sheet views/front.jpg views/face.jpg views/back.jpg
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
    parser.add_argument(
        "--sheet", type=Path, nargs="+", required=True, help="sheet or images"
    )
    return parser.parse_args()


def main() -> None:
    """Saves the best front, close-up, and back views and records the split."""
    args = parse_args()
    images = {path: Image.open(path).convert("RGB") for path in args.sheet}
    found = {path: sheet.find_views(image) for path, image in images.items()}
    source = {id(view): path for path, views in found.items() for view in views}
    pooled = [view for views in found.values() for view in views]
    front = sheet.pick_front(pooled)
    picks = {
        "front": front,
        "close_up": sheet.pick_close_up(pooled),
        "back": sheet.pick_back(pooled),
    }

    run_dir = config.RUNS_DIR / datetime.now().strftime("sheet-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    records = []
    for path, views in found.items():
        image = images[path]
        overview = image.copy()
        draw = ImageDraw.Draw(overview)
        for view in views:
            name = next((name for name, pick in picks.items() if pick is view), None)
            draw.rectangle(view.box, outline=PICKED if name else OTHER, width=3)
            if name:
                draw.text(
                    (view.box[0] + 6, view.box[1] + 4), name, PICKED, font_size=20
                )
                sheet.crop(image, views, view).save(run_dir / f"{name}.png")
            records.append(
                {
                    "image": path.as_posix(),
                    "box": view.box,
                    "full_body": view.full_body,
                    "pick": name,
                }
            )
        overview.save(run_dir / f"views-{path.stem}.png")
    sheet.cutout(images[source[id(front)]], front).save(run_dir / "front_cutout.png")

    warning = sheet.pose_warning(front)
    record = {
        "sheets": [path.as_posix() for path in args.sheet],
        "views": records,
        "warnings": [warning] if warning else [],
    }
    (run_dir / "views.json").write_text(json.dumps(record, indent=2), "utf-8")
    if warning:
        print(f"Warning: {warning}")
    print(run_dir)


if __name__ == "__main__":
    main()
