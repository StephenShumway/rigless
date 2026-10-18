"""The scorer must tell real good takes from real frozen ones.

Uses the labeled takes in local_fixtures/takes/ (see HANDOFF.md): good_* takes
rendered with the full pose window and frozen_* takes rendered without it, plus
the driving.mp4 they share. Skips when they're missing, as on a fresh clone.
"""

import pytest

from rigless import config, pose
from rigless.score import TakeScore, score_take

TAKES = config.REPO_ROOT / "local_fixtures" / "takes"

pytestmark = pytest.mark.fixtures


@pytest.fixture(scope="module")
def scores() -> dict[str, TakeScore]:
    if not (TAKES / "driving.mp4").exists():
        pytest.skip("local_fixtures/takes/ is missing")
    source = pose.estimate(TAKES / "driving.mp4")
    return {
        video.stem: score_take(source, pose.estimate(video))
        for video in TAKES.glob("*.mp4")
        if video.stem != "driving"
    }


def test_good_takes_clearly_outscore_frozen_takes(scores: dict[str, TakeScore]) -> None:
    good = [result.score for name, result in scores.items() if name.startswith("good_")]
    frozen = [
        result.score for name, result in scores.items() if name.startswith("frozen_")
    ]

    assert good and frozen
    assert min(good) > max(frozen) + 15


def test_only_frozen_takes_are_flagged(scores: dict[str, TakeScore]) -> None:
    flagged = {name: result.frozen for name, result in scores.items()}

    assert flagged == {name: name.startswith("frozen_") for name in scores}
