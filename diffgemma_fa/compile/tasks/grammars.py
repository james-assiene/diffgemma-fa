"""Task grammars other than the BFCL/xLAM function-call formats. SPEC §4.8.

| Task | Constraint | FA | Paper's state count |
|---|---|---|---|
| Sudoku | format **+ preserves prefilled cells** | DFA | 21 (4x4, fixed) |
| Countdown | each step matches `A op B=C` | DFA | 47-77 |
| GSM-Symbolic | symbolic expressions only inside `«…»` | DFA | 56 |
| refusal (§3.8) | free text that cannot start with `[` or `{` | DFA | small |

Each builder returns an **anchored** regex, ready for
`pipeline.compile_regex`.
"""

from __future__ import annotations

import re
from typing import Sequence

__all__ = [
    "sudoku_regex",
    "countdown_regex",
    "gsm_symbolic_regex",
    "refusal_regex",
    "channel_header_regex",
]


def channel_header_regex(*, channel_name: str = r"[a-z]+") -> str:
    """The `<|channel>NAME\\n<channel|>` prefix. SPEC §3.6.

    Phase 0 measured every long generation opening with exactly
    `[100, 45518, 107, 101]`. The delimiters are single dedicated token ids, so
    this is a literal prefix rather than the Aho-Corasick construction the
    original spec feared. The channel *name* is left open — only `thought` was
    observed, but `final`/`answer` tokenize fine and hardcoding would be
    fragile.
    """
    return rf"<\|channel>{channel_name}\n<channel\|>"


def sudoku_regex(
    puzzle: Sequence[Sequence[int]],
    *,
    box: int | None = None,
    separator: str = "",
    row_separator: str = "",
) -> str:
    """Grammar for a Sudoku solution that **preserves the prefilled cells**.

    That second half is the whole point and is easy to miss: a format-only
    grammar lets the model overwrite the givens and "solve" a different puzzle.
    Each prefilled cell becomes a literal; each blank becomes the digit
    alternation.

    Args:
      puzzle: `n x n` grid, 0 for a blank.
      box: box width, for validating `n`. Defaults to `isqrt(n)`.
      separator: between cells.
      row_separator: between rows.

    Returns:
      An anchored regex matching exactly the completions of `puzzle`.
    """
    n = len(puzzle)
    if any(len(r) != n for r in puzzle):
        raise ValueError("puzzle must be square")
    if not 1 <= n < 10:
        raise ValueError("only 1 <= n < 10 is supported (single-character digits)")
    if box is None:
        import math

        b = math.isqrt(n)
        if b * b != n:
            raise ValueError(
                f"n={n} is not a perfect square; pass box= explicitly"
            )

    digits = "".join(str(d) for d in range(1, n + 1))
    blank = f"[{digits}]"

    rows = []
    for r in puzzle:
        cells = [re.escape(str(c)) if c else blank for c in r]
        rows.append(re.escape(separator).join(cells))
    return re.escape(row_separator).join(rows)


def countdown_regex(
    *,
    max_steps: int = 4,
    max_value: int = 999,
    ops: str = r"[+\-*/]",
    step_separator: str = r"\n",
) -> str:
    """Grammar for Countdown: a sequence of `A op B=C` steps. SPEC §4.8.

    Args:
      max_steps: bound on the number of steps. A reportable cap.
      max_value: largest integer permitted (digit-width bound, not a numeric
        bound — outlines-style numeric ranges are exactly what SPEC §4.2 says
        is silently dropped, so this is expressed structurally).
      ops: the operator class.
      step_separator: between steps.
    """
    if max_value < 1:
        raise ValueError("max_value must be >= 1")
    width = len(str(max_value))
    num = rf"(?:0|[1-9][0-9]{{0,{width - 1}}})"
    step = rf"{num}\s?{ops}\s?{num}={num}"
    return rf"{step}(?:{step_separator}{step}){{0,{max_steps - 1}}}"


def gsm_symbolic_regex(
    *,
    max_expressions: int = 8,
    max_free_chars: int = 200,
    max_value: int = 99999,
) -> str:
    """GSM-Symbolic: free reasoning text, but arithmetic only inside `«…»`.

    The constraint is scoped: prose is unconstrained, and every calculation
    must appear as a well-formed symbolic expression between the guillemets, so
    the answer can be checked mechanically.

    Args:
      max_expressions: bound on `«…»` groups. Reportable cap.
      max_free_chars: bound on each run of free text. Reportable cap.
      max_value: digit-width bound on the numbers.
    """
    width = len(str(max_value))
    num = rf"(?:0|[1-9][0-9]{{0,{width - 1}}})(?:\.[0-9]{{1,4}})?"
    expr = rf"{num}(?:\s?[+\-*/]\s?{num})*"
    calc = rf"«{expr}={num}»"
    # Free text explicitly excludes the guillemets so the segmentation is
    # unambiguous.
    free = rf"[^«»]{{0,{max_free_chars}}}"
    return rf"{free}(?:{calc}{free}){{0,{max_expressions}}}"


def refusal_regex(*, max_chars: int = 400) -> str:
    """The `FA_refusal` escape hatch. SPEC §3.8.

    BFCL scores `irrelevance` + `live_irrelevance` — measured at 240 + 884 =
    **1,124 of the single-turn entries** — by treating a decode *exception* as
    "no call". A grammar that forces a well-formed call scores **0** on all of
    them. So the answer region is `FA_grammar | FA_refusal`, where the refusal
    branch is free text that **cannot start with `[` or `{`** (which is what
    makes the union unambiguous at the first token).

    > **State this caveat next to every `CS = 100%` claim.** With a refusal
    > branch available, the model can spend its whole budget outside
    > `FA_grammar`, making constraint satisfaction trivially 100% while
    > measuring nothing. Report **CS conditional on the grammar branch being
    > taken**, plus the branch-taken rate.
    """
    first = r"[^\[\{\s]"
    rest = rf"[\s\S]{{0,{max_chars}}}"
    return rf"{first}{rest}"
