"""Countdown and Sudoku task modules. SPEC §7.1.

These two datasets exist to test the regime BFCL cannot: tight combinatorial
grammars with small `|S|`, where the constraint eliminates almost the whole
output space. The scorers are the load-bearing part — a grammar guarantees the
*shape* of a solution and nothing about whether it is right, so both scorers
re-derive correctness rather than trusting the automaton.
"""

from __future__ import annotations

import pytest

from diffgemma_fa.compile.tasks import countdown as CD
from diffgemma_fa.compile.tasks import sudoku as SD
from diffgemma_fa.compile.tasks.grammars import countdown_regex, sudoku_regex


# --------------------------------------------------------------------------
# Countdown scoring
# --------------------------------------------------------------------------

def _rec(nums, target):
    return CD.CountdownRecord(id="t", numbers=tuple(nums), target=target)


def test_countdown_accepts_a_correct_solution():
    ok, why = CD.score_solution("3*4=12\n12+5=17", _rec([3, 4, 5], 17))
    assert ok, why


def test_countdown_rejects_false_arithmetic():
    """The grammar admits `1+1=3` — it constrains shape, not truth. Scoring
    format-only would report that as a solved puzzle."""
    ok, why = CD.score_solution("1+1=3", _rec([1, 1], 3))
    assert not ok and "arithmetic" in why


def test_countdown_rejects_reusing_a_number():
    """Each given may be used at most once; the pool is tracked explicitly."""
    ok, why = CD.score_solution("3*3=9", _rec([3, 4], 9))
    assert not ok and "not available" in why


def test_countdown_consumes_intermediates_and_allows_their_reuse():
    """`3*4=12` removes 3 and 4 and adds 12, so the next step may use 12 —
    but only once."""
    r = _rec([3, 4, 5], 17)
    assert CD.score_solution("3*4=12\n12+5=17", r)[0]
    ok, why = CD.score_solution("3*4=12\n12+5=17\n12+5=17", r)[:2]
    assert not ok and "not available" in why


def test_countdown_rejects_a_wrong_target():
    ok, why = CD.score_solution("3+4=7", _rec([3, 4], 99))
    assert not ok and "target" in why


def test_countdown_rejects_non_integer_division():
    ok, why = CD.score_solution("7/2=3", _rec([7, 2], 3))
    assert not ok and "division" in why


def test_countdown_grammar_admits_the_scored_format():
    import re
    rx = countdown_regex(max_steps=4, max_value=999, step_separator=r"\n")
    assert re.fullmatch(rx, "3*4=12\n12+5=17")
    assert not re.fullmatch(rx, "3*4")


# --------------------------------------------------------------------------
# Sudoku generation and scoring
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def puzzles():
    return SD.generate(12, seed=0, n_blanks=6)


def test_every_generated_puzzle_has_exactly_one_solution(puzzles):
    """A puzzle with two solutions measures nothing: a model that produced the
    other valid completion would be scored wrong for being right."""
    for r in puzzles:
        assert SD.solutions(r.puzzle) == 1, r.id


def test_generation_is_deterministic_in_the_seed():
    a = SD.generate(5, seed=7)
    b = SD.generate(5, seed=7)
    assert [x.puzzle for x in a] == [x.puzzle for x in b]
    assert [x.puzzle for x in SD.generate(5, seed=8)] != [x.puzzle for x in a]


def test_the_stored_solution_actually_solves_the_puzzle(puzzles):
    for r in puzzles:
        txt = "\n".join("".join(str(c) for c in row) for row in r.solution)
        ok, why = SD.score_solution(txt, r)
        assert ok, f"{r.id}: {why}"


def test_sudoku_rejects_overwriting_a_given(puzzles):
    """The grammar is supposed to pin the givens, but the scorer must not
    assume it: if that ever regressed, a model solving a DIFFERENT puzzle
    would score as correct."""
    r = next(x for x in puzzles if any(c for row in x.puzzle for c in row))
    grid = [list(row) for row in r.solution]
    (gr, gc) = next((i, j) for i in range(4) for j in range(4) if r.puzzle[i][j])
    grid[gr][gc] = 1 + (grid[gr][gc] % 4)
    txt = "\n".join("".join(str(c) for c in row) for row in grid)
    ok, why = SD.score_solution(txt, r)
    assert not ok and "given" in why


def test_sudoku_rejects_a_duplicate_in_a_row(puzzles):
    r = puzzles[0]
    grid = [list(row) for row in r.solution]
    grid[0] = [grid[0][0]] * 4
    txt = "\n".join("".join(str(c) for c in row) for row in grid)
    assert not SD.score_solution(txt, r)[0]


def test_sudoku_grammar_pins_the_givens_and_frees_the_blanks(puzzles):
    import re
    r = puzzles[0]
    rx = sudoku_regex(r.puzzle, row_separator="\n")
    good = "\n".join("".join(str(c) for c in row) for row in r.solution)
    assert re.fullmatch(rx, good)
    # flip one given: must now be rejected by the GRAMMAR, not just the scorer
    grid = [list(row) for row in r.solution]
    (gr, gc) = next((i, j) for i in range(4) for j in range(4) if r.puzzle[i][j])
    grid[gr][gc] = 1 + (grid[gr][gc] % 4)
    bad = "\n".join("".join(str(c) for c in row) for row in grid)
    assert not re.fullmatch(rx, bad), (
        "the grammar let a prefilled cell be overwritten — a model could then "
        "solve a different puzzle and be scored correct"
    )
