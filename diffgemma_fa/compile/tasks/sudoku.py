"""Sudoku 4x4, generated synthetically. SPEC §7.1.

SPEC records that there is **no canonical source** — Sakana's `Sudoku-Bench` is
401/discontinued — so the puzzles are generated here: all completed 4x4 grids,
cells masked, and **uniqueness verified by exhaustive solve**. Fully
reproducible from a seed, which is better than an unavailable download.

**Why uniqueness matters.** A puzzle with two solutions cannot distinguish a
model that solved it from one that guessed a different valid completion, so
accuracy on such a puzzle measures nothing. Every generated puzzle is solved
exhaustively and kept only if the solution count is exactly 1.

**Why this dataset earns its GPU time.** It is the opposite regime from BFCL:
`sudoku_regex` pins every prefilled cell as a literal and every blank to a
4-way digit choice, so `|S|` is tiny and the constraint eliminates almost the
entire output space. SPEC §2.8 cites the paper measuring unconstrained
constraint satisfaction at **7.6%** here. If constrained decoding is worth
anything, it is worth the most on a grammar like this one.

The grammar preserving the givens is the subtle half: a format-only grammar
lets the model overwrite the prefilled cells and "solve" a different puzzle,
which would score as correct. `grammars.sudoku_regex` handles that; the scorer
below re-checks it rather than trusting it.
"""

from __future__ import annotations

import dataclasses
import random
import re
from typing import Iterator, Sequence

from diffgemma_fa.compile.tasks import answer_region

__all__ = ["SudokuRecord", "generate", "solutions", "score_solution",
           "build_prompt", "locate_grid"]

N = 4
BOX = 2

#: One row of the answer: exactly `N` digits on a line of its own, modulo
#: surrounding whitespace. Anchored on both ends **on purpose** — `1234` is a
#: row, `Check: 1234 in every row.` is not, and `4x4` is not.
_ROW = re.compile(rf"^\s*([1-{N}]{{{N}}})\s*$")

#: The degenerate rendering: the whole grid on one line, which is what
#: `sudoku_regex(..., row_separator="")` compiles to.
_FLAT = re.compile(rf"^\s*([1-{N}]{{{N * N}}})\s*$")


@dataclasses.dataclass(frozen=True)
class SudokuRecord:
    id: str
    puzzle: tuple[tuple[int, ...], ...]      # 0 = blank
    solution: tuple[tuple[int, ...], ...]


def _ok(grid: list[list[int]], r: int, c: int, v: int) -> bool:
    if any(grid[r][j] == v for j in range(N)):
        return False
    if any(grid[i][c] == v for i in range(N)):
        return False
    br, bc = (r // BOX) * BOX, (c // BOX) * BOX
    return all(grid[i][j] != v
               for i in range(br, br + BOX) for j in range(bc, bc + BOX))


def solutions(grid: Sequence[Sequence[int]], cap: int = 2) -> int:
    """Exhaustive solution count, stopping at `cap`.

    `cap=2` is all uniqueness needs and keeps generation fast: the moment a
    second solution appears the puzzle is rejected.
    """
    g = [list(r) for r in grid]

    def rec() -> int:
        for r in range(N):
            for c in range(N):
                if g[r][c] == 0:
                    total = 0
                    for v in range(1, N + 1):
                        if _ok(g, r, c, v):
                            g[r][c] = v
                            total += rec()
                            g[r][c] = 0
                            if total >= cap:
                                return total
                    return total
        return 1

    return rec()


def _complete(rng: random.Random) -> list[list[int]]:
    g = [[0] * N for _ in range(N)]

    def fill(pos: int) -> bool:
        if pos == N * N:
            return True
        r, c = divmod(pos, N)
        vals = list(range(1, N + 1))
        rng.shuffle(vals)
        for v in vals:
            if _ok(g, r, c, v):
                g[r][c] = v
                if fill(pos + 1):
                    return True
                g[r][c] = 0
        return False

    fill(0)
    return g


def generate(n: int, *, seed: int = 0, n_blanks: int = 6) -> list[SudokuRecord]:
    """`n` puzzles with a **unique** solution, deterministic in `seed`.

    Args:
      n_blanks: cells removed. More blanks means a harder puzzle and a larger
        automaton; 6 of 16 keeps uniqueness common enough that generation is
        fast while leaving real work to do.
    """
    rng = random.Random(seed)
    out: list[SudokuRecord] = []
    seen: set[tuple] = set()
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 200 * n:
            raise RuntimeError(
                f"only generated {len(out)}/{n} unique-solution puzzles; "
                f"n_blanks={n_blanks} is probably too high"
            )
        sol = _complete(rng)
        cells = [(r, c) for r in range(N) for c in range(N)]
        rng.shuffle(cells)
        puz = [row[:] for row in sol]
        for r, c in cells[:n_blanks]:
            puz[r][c] = 0
        key = tuple(tuple(r) for r in puz)
        if key in seen or solutions(puz) != 1:
            continue
        seen.add(key)
        out.append(SudokuRecord(
            id=f"sudoku4_{len(out)}",
            puzzle=key,
            solution=tuple(tuple(r) for r in sol)))
    return out


def build_prompt(rec: SudokuRecord) -> str:
    rows = "\n".join("".join(str(c) if c else "." for c in r)
                     for r in rec.puzzle)
    return (
        f"Solve this 4x4 Sudoku. Digits 1-4; each row, each column and each "
        f"2x2 box must contain every digit exactly once. `.` marks a blank.\n\n"
        f"{rows}\n\n"
        f"Write only the completed grid, four digits per line, no separators."
    )


def locate_grid(text: str) -> list[list[int]] | None:
    """Find the completed grid in an emission, **structurally**.

    `N` consecutive lines of exactly `N` digits each, in the answer region — the
    shape `build_prompt` asks for and the shape `grammars.sudoku_regex`
    compiles. Falls back to a single line of `N*N` digits, which is what
    `row_separator=""` produces.

    **[AUDIT-D] Why not "the first N*N digits".** That rule is right only when
    nothing precedes the grid. The earlier fix stripped the channel header,
    which is the only source of stray digits on the *constrained* arms — but
    `unconstrained` and `mask` have no header, they emit prose, and they are the
    baseline the whole cross-task comparison is measured against. `"Here is the
    completed 4x4 grid:\\n"` shifts the scan by one cell and the scorer reports
    `overwrote the given at (0,0)` for a grid that is intact. `4x4` is not a
    contrived example: it is the phrase the prompt itself uses.

    Returns:
      The grid, or `None` if the emission does not contain one.
    """
    lines = answer_region(text).splitlines()
    rows = [_ROW.match(ln) for ln in lines]
    for i in range(len(rows) - N + 1):
        window = rows[i:i + N]
        if all(m is not None for m in window):
            return [[int(ch) for ch in m.group(1)] for m in window]
    for ln in lines:
        m = _FLAT.match(ln)
        if m is not None:
            return [[int(m.group(1)[r * N + c]) for c in range(N)]
                    for r in range(N)]
    return None


def score_solution(text: str, rec: SudokuRecord) -> tuple[bool, str]:
    """`(correct, reason)`, re-checking the givens rather than trusting them.

    The grammar is supposed to pin the prefilled cells, but scoring is the
    wrong place to assume that: if the grammar ever regressed, a model that
    overwrote a given and solved a *different* puzzle would silently score as
    correct.

    The reason for an emission with no grid at all is `"empty"` — the same word
    Countdown uses, so `eval/run_tasks.py`'s parse column means the same thing
    for both tasks (`tasks.produced_an_answer`). It used to be
    `"only 0 digits"`, which made that column 100% for Sudoku by construction.
    """
    if not answer_region(text):
        return False, "empty"
    g = locate_grid(text)
    if g is None:
        # Non-empty but no grid: a real model failure, and a *different*
        # diagnosis from "emitted nothing". Both mean "no answer to score", so
        # both are in `tasks.NO_ANSWER_REASONS`.
        return False, "no grid"

    for r in range(N):
        for c in range(N):
            if rec.puzzle[r][c] and g[r][c] != rec.puzzle[r][c]:
                return False, f"overwrote the given at ({r},{c})"

    want = set(range(1, N + 1))
    for i in range(N):
        if {g[i][j] for j in range(N)} != want:
            return False, f"row {i}"
        if {g[j][i] for j in range(N)} != want:
            return False, f"col {i}"
    for br in range(0, N, BOX):
        for bc in range(0, N, BOX):
            box = {g[r][c] for r in range(br, br + BOX)
                   for c in range(bc, bc + BOX)}
            if box != want:
                return False, f"box ({br},{bc})"
    return True, "ok"
