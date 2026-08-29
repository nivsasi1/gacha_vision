"""The worker protocol, tested at the request level.

The I/O loop is a few lines; the part worth testing is what one request
produces. `handle` is pure for that reason -- a dict in, a dict out -- so
these tests need no subprocess.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from gacha_vision.worker import handle

DATA = Path(__file__).parent / "data"


def _req(path: Path, **kw) -> dict:
    return {"image": base64.b64encode(path.read_bytes()).decode(), **kw}


def test_ping_answers_pong():
    """A caller needs to know the process is alive without sending an image."""
    assert handle({"ping": True}) == {"pong": True}


def test_an_image_answers_with_the_slots_to_claim():
    reply = handle(_req(DATA / "card_print_1550.png", expected=1))
    assert reply["slots"] == [1]


def test_the_reply_carries_the_cards_it_already_read():
    """`analyze_cards` runs inside pick either way, so reporting what it saw
    costs nothing and saves the caller a second call."""
    reply = handle(_req(DATA / "card_print_1550.png", expected=1))
    assert "cards" in reply
    card = reply["cards"][0]
    assert card["slot"] == 1
    assert card["printNo"] == 1550
    assert card["frame"] == "normal"


def test_an_e_card_reports_a_null_print_number():
    reply = handle(_req(DATA / "card_uncatalogued_frame.png", expected=1))
    card = reply["cards"][0]
    assert card["slot"] == 1
    assert card["frame"] in {"normal", "e", "other", "unknown"}
    assert "printNo" in card, "the key must always be present, even when null"


def test_the_uncatalogued_frame_is_reported_as_other():
    """The caller wants to know *why* a bad print was still claimed."""
    reply = handle(_req(DATA / "card_uncatalogued_frame.png", expected=1))
    assert reply["cards"][0]["frame"] == "other"
    assert reply["slots"] == [1]


def test_cards_are_ordered_left_to_right_and_match_the_slots():
    reply = handle(_req(DATA / "card_print_1550.png", expected=1))
    got = [c["slot"] for c in reply["cards"]]
    assert got == sorted(got)
    assert set(reply["slots"]) <= set(got)


def test_a_broken_image_answers_with_an_error_not_an_empty_result():
    """"Nothing worth claiming" and "the download broke" must never look the
    same to the caller."""
    reply = handle({"image": base64.b64encode(b"not an image").decode()})
    assert "error" in reply
    assert "slots" not in reply


def test_a_request_with_no_image_is_an_error():
    assert "error" in handle({"expected": 2})


def test_an_unparseable_field_is_an_error_rather_than_a_crash():
    assert "error" in handle({"image": "!!!not base64!!!"})


@pytest.mark.parametrize("expected", [None, 1])
def test_expected_is_optional(expected):
    req = _req(DATA / "card_print_1550.png")
    if expected is not None:
        req["expected"] = expected
    assert handle(req)["slots"] == [1]
