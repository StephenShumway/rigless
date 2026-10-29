"""The MiniMax H3 terms each user agrees to before Rigless renders with H3.

H3's license (section V.2) asks anyone who offers H3 to others to bind each
user to terms at least as protective as its use restrictions, and to tell them
those restrictions apply. The app shows these terms before the first render,
keeps the agreement in the app's own folder, and asks again when VERSION
changes. Nothing leaves the user's computer.

Typical usage example:

  if not terms.accepted(root):
      terms.accept(root)  # after the user ticks the box
"""

import json
from datetime import UTC, datetime
from pathlib import Path

VERSION = "2026-10-27"
LICENSE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
AUTHORIZATION_URL = "https://platform.minimax.io/h3-license"
REPORT_EMAIL = "pickax-diamond0s@icloud.com"  # where misuse is reported
NOTICE = (
    "Rigless redraws people with MiniMax H3, licensed under the MiniMax H3 "
    "Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved. "
    "Rendering with it means agreeing to that license, its use restrictions and "
    "its Acceptable Use Policy.",
    "Among other things, you agree to use only footage and pictures you have the "
    "rights to, of people who consented or who aren't real. You agree never to "
    "impersonate, harass, deceive or harm anyone, or to involve minors in anything "
    "harmful. You agree to say clearly that a video is AI-generated wherever you "
    "share it, and never to use what H3 makes to train or improve another AI model.",
    "MiniMax H3 can't be used in the United States, the European Union, the United "
    "Kingdom or South Korea without your own authorization from MiniMax.",
)
FILE = "terms.json"


def accepted(root: Path) -> bool:
    """Returns whether the user agreed to this version of the terms."""
    try:
        record = json.loads((root / FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and record.get("version") == VERSION


def accept(root: Path) -> None:
    """Records that the user agreed to this version of the terms, and when."""
    root.mkdir(parents=True, exist_ok=True)
    record = {"version": VERSION, "accepted": datetime.now(UTC).isoformat()}
    (root / FILE).write_text(json.dumps(record, indent=2), "utf-8")
