"""Tests for the task grammars. SPEC §4.8, §3.6, §3.8."""

from __future__ import annotations

import re

import pytest

from diffgemma_fa.compile.tasks.grammars import (
    channel_header_regex,
    countdown_regex,
    gsm_symbolic_regex,
    refusal_regex,
    sudoku_regex,
)


def full(rx: str, s: str) -> bool:
    return re.fullmatch(rx, s) is not None


# --------------------------------------------------------------------------
# Sudoku — the prefilled-cell constraint is the substance
# --------------------------------------------------------------------------

PUZZLE = [
    [1, 0, 0, 0],
    [0, 0, 3, 0],
    [0, 4, 0, 0],
    [0, 0, 0, 2],
]


# A grid consistent with PUZZLE's givens: (0,0)=1, (1,2)=3, (2,1)=4, (3,3)=2.
# The grammar enforces format + givens, not the Latin-square property, so this
# need not be a *solved* Sudoku — only a well-formed completion.
CONSISTENT = "1234" "1131" "1411" "1112"


def test_sudoku_accepts_a_completion_that_preserves_givens():
    rx = sudoku_regex(PUZZLE)
    assert full(rx, CONSISTENT)
    assert full(rx, "1111" "1131" "1411" "1112")


def test_sudoku_rejects_overwriting_a_given():
    """A format-only grammar would let the model solve a different puzzle."""
    rx = sudoku_regex(PUZZLE)
    assert not full(rx, "2234" "1131" "1411" "1112"), "cell (0,0) given is 1"
    assert not full(rx, "1234" "1131" "1411" "1113"), "cell (3,3) given is 2"
    assert not full(rx, "1234" "1141" "1411" "1112"), "cell (1,2) given is 3"
    assert not full(rx, "1234" "1131" "1311" "1112"), "cell (2,1) given is 4"


def test_sudoku_rejects_out_of_range_digits():
    rx = sudoku_regex(PUZZLE)
    assert not full(rx, "1534" "1131" "1411" "1112")


def test_sudoku_rejects_wrong_length():
    rx = sudoku_regex(PUZZLE)
    assert not full(rx, "123" "1131" "1411" "1112")


def test_sudoku_9x9():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    rx = sudoku_regex(grid)
    good = "5" + "1" * 80
    bad = "6" + "1" * 80
    assert full(rx, good) and not full(rx, bad)


def test_sudoku_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        sudoku_regex([[1, 2, 3]])


def test_sudoku_separators():
    rx = sudoku_regex([[1, 0], [0, 2]], box=1, separator=",", row_separator="\n")
    assert full(rx, "1,2\n1,2")
    assert not full(rx, "12\n12")


# --------------------------------------------------------------------------
# Countdown
# --------------------------------------------------------------------------

def test_countdown_single_step():
    rx = countdown_regex(max_steps=3)
    assert full(rx, "3+4=7")
    assert full(rx, "12*3=36")


def test_countdown_multi_step():
    rx = countdown_regex(max_steps=3)
    assert full(rx, "3+4=7\n7*2=14")
    assert full(rx, "3+4=7\n7*2=14\n14-1=13")


def test_countdown_step_cap_is_enforced():
    rx = countdown_regex(max_steps=2)
    assert full(rx, "1+1=2\n2+2=4")
    assert not full(rx, "1+1=2\n2+2=4\n4+4=8")


def test_countdown_rejects_malformed():
    rx = countdown_regex()
    assert not full(rx, "3+4")        # no result
    assert not full(rx, "3 4=7")      # no operator
    assert not full(rx, "=7")


def test_countdown_value_width_bound():
    rx = countdown_regex(max_value=99)
    assert full(rx, "12+34=46")
    assert not full(rx, "123+4=127"), "3-digit exceeds the width bound"


def test_countdown_rejects_leading_zeros():
    rx = countdown_regex()
    assert not full(rx, "01+2=3")


# --------------------------------------------------------------------------
# GSM-Symbolic
# --------------------------------------------------------------------------

def test_gsm_plain_prose_is_allowed():
    assert full(gsm_symbolic_regex(), "He buys three apples and stops.")


def test_gsm_calculation_must_be_inside_guillemets():
    rx = gsm_symbolic_regex()
    assert full(rx, "He spends «3*4=12» dollars.")
    assert full(rx, "First «2+2=4» then «4*5=20» total.")


def test_gsm_rejects_malformed_calculation():
    rx = gsm_symbolic_regex()
    assert not full(rx, "He spends «3*4» dollars."), "must include the result"
    assert not full(rx, "He spends «=12» dollars.")


def test_gsm_expression_cap():
    rx = gsm_symbolic_regex(max_expressions=1)
    assert full(rx, "a «1+1=2» b")
    assert not full(rx, "a «1+1=2» b «2+2=4» c")


def test_gsm_free_text_excludes_guillemets():
    """Segmentation must be unambiguous, so a stray guillemet cannot appear in
    the free-text run."""
    assert not full(gsm_symbolic_regex(), "a » b")


# --------------------------------------------------------------------------
# Refusal branch — SPEC §3.8
# --------------------------------------------------------------------------

def test_refusal_accepts_prose():
    assert full(refusal_regex(), "I cannot help with that request.")


def test_refusal_cannot_start_with_a_call_delimiter():
    """This is what keeps `FA_grammar | FA_refusal` unambiguous at the first
    token."""
    rx = refusal_regex()
    assert not full(rx, '[{"name": "f"}]')
    assert not full(rx, '{"name": "f"}')


def test_refusal_cannot_start_with_whitespace():
    assert not full(refusal_regex(), "  no")


def test_refusal_may_contain_braces_later():
    assert full(refusal_regex(), "No, but {this} is fine.")


# --------------------------------------------------------------------------
# Channel header — SPEC §3.6
# --------------------------------------------------------------------------

def test_channel_header_matches_the_observed_prefix():
    """Phase 0 measured every long generation opening with
    `<|channel>thought\\n<channel|>`."""
    assert full(channel_header_regex(), "<|channel>thought\n<channel|>")


def test_channel_header_is_not_hardcoded_to_thought():
    rx = channel_header_regex()
    assert full(rx, "<|channel>final\n<channel|>")
    assert full(rx, "<|channel>answer\n<channel|>")
