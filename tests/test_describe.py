import pytest

from rigless import describe


@pytest.mark.parametrize("answer", ["Yes", "yes.", " **Yes**", "Yes, they do"])
def test_yes_in_any_form_is_yes(answer: str) -> None:
    assert describe.said_yes(answer)


@pytest.mark.parametrize("answer", ["No", "no.", "", "Not visible"])
def test_anything_but_yes_is_no(answer: str) -> None:
    assert not describe.said_yes(answer)
