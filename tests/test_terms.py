from pathlib import Path

import pytest

from rigless import terms


def test_agreement_holds_until_the_terms_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not terms.accepted(tmp_path)
    terms.accept(tmp_path)
    assert terms.accepted(tmp_path)

    monkeypatch.setattr(terms, "VERSION", "2027-01-01")

    assert not terms.accepted(tmp_path)
