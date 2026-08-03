"""Countdown-Tasks-3to4: reach a target from given numbers. SPEC §7.1.

TinyZero's actual source, test slice `ds.select(range(327680, 328704))` (1,024
rows); the paper reports 993.

**Why this dataset earns its GPU time.** Every accuracy result in this project
so far comes from BFCL, whose grammars are large permissive JSON schemas —
`|S|` in the hundreds to 1,024, where §7.3 measures the tree at up to 82.6% of
a model forward and where the automaton barely narrows the output space.
Countdown is the opposite regime: a tight combinatorial grammar over a handful
of digits and operators, small `|S|`, where the constraint eliminates almost
everything. If the whitespace finding was a BFCL-specific artifact of JSON
rendering rather than something general about grammar-tokenizer alignment,
this is where that shows up.

**Scoring is mechanical and needs no model.** A solution is correct iff every
step is arithmetically true, every intermediate is used at most once, only the
given numbers are consumed, and the final result equals the target. That is
checked here rather than trusted, because a grammar that admits `1+1=3` would
otherwise score as a valid call.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Iterator, Sequence

__all__ = ["CountdownRecord", "iter_records", "score_solution", "build_prompt"]

#: The test slice TinyZero uses. Stated rather than sampled, so the split is
#: reproducible without shipping the data.
TEST_SLICE = (327_680, 328_704)

_STEP = re.compile(r"^\s*(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)\s*$")


@dataclasses.dataclass(frozen=True)
class CountdownRecord:
    id: str
    numbers: tuple[int, ...]
    target: int


def iter_records(path: str, limit: int = 0) -> Iterator[CountdownRecord]:
    """Stream records from a local JSONL dump of the HF dataset.

    Kept file-based rather than calling `datasets.load_dataset` at eval time:
    the runs are long, and a network fetch inside a 20-hour job is a failure
    mode we have already paid for elsewhere.
    """
    import json

    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                return
            d = json.loads(line)
            nums = d.get("nums") or d.get("numbers")
            yield CountdownRecord(id=f"countdown_{i}",
                                  numbers=tuple(int(x) for x in nums),
                                  target=int(d["target"]))


def build_prompt(rec: CountdownRecord) -> str:
    nums = ", ".join(str(n) for n in rec.numbers)
    return (
        f"Using the numbers {nums}, each at most once, write arithmetic steps "
        f"that reach {rec.target}.\n"
        f"Write one step per line in the form `A op B=C`, and let the last "
        f"step's result be {rec.target}."
    )


def score_solution(text: str, rec: CountdownRecord) -> tuple[bool, str]:
    """`(correct, reason)`.

    Checks arithmetic, resource use, and the target — not just the format. The
    grammar guarantees the *shape* `A op B=C`; it cannot guarantee that the
    arithmetic is true or that the model used only the numbers it was given,
    and scoring format-only would report a solver that writes `1+1=3` as
    correct.
    """
    # Strip SPEC §3.6's channel header first -- see the note in
    # `sudoku.score_solution`. Here it made the FIRST step unparsable on every
    # record, because the header shares its line with the opening step.
    body = text.split("<channel|>", 1)[1] if "<channel|>" in text else text
    # With --think the answer follows a literal `ANSWER:` marker; everything
    # before it is the model's scratchpad and must not be scored.
    if "ANSWER:" in body:
        body = body.split("ANSWER:", 1)[1]
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    if not lines:
        return False, "empty"

    pool: list[int] = list(rec.numbers)
    last = None
    for ln in lines:
        m = _STEP.match(ln)
        if not m:
            return False, f"unparsable step: {ln.strip()[:40]!r}"
        a, op, b, c = int(m[1]), m[2], int(m[3]), int(m[4])

        for operand in (a, b):
            if operand in pool:
                pool.remove(operand)
            else:
                return False, f"operand {operand} not available"

        if op == "+":
            got = a + b
        elif op == "-":
            got = a - b
        elif op == "*":
            got = a * b
        else:
            if b == 0 or a % b != 0:
                return False, f"non-integer division {a}/{b}"
            got = a // b
        if got != c:
            return False, f"arithmetic: {a}{op}{b} = {got}, claimed {c}"

        pool.append(c)
        last = c

    if last != rec.target:
        return False, f"final {last} != target {rec.target}"
    return True, "ok"
