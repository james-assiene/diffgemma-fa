"""Task datasets, grammars and scorers.

What lives here is the small amount of structure the *scorers* share, because
the two places they have been wrong were both about reading the wrong span of
the emission rather than about the task:

- **[AUDIT-D]** the Sudoku scorer read the first 16 digits of the raw string, so
  a channel-header name containing digits shifted the grid and it reported
  *"overwrote the given"* on 250/250 records **while CS was 1.000** — while the
  automaton had already proved the givens intact;
- **[AUDIT-C]** SPEC §3.5 makes the post-stop tail unscored (`ACC --Σ--> ACC`),
  so the decoded text always carries a trailing `<turn|>` or `<|channel>`. The
  Countdown step pattern is anchored `\\s*$`, so that marker landed inside the
  last step's line and the last step was always "unparsable step" — merging a
  format diagnosis with an arithmetic one, which `eval/run_tasks.py`'s own
  docstring says must not happen.

Both are the same mistake: scoring the *sampler's* framing instead of the
model's answer. `answer_region` is the one place that framing is removed.
"""

from __future__ import annotations

import re

__all__ = ["answer_region", "NO_ANSWER_REASONS", "produced_an_answer"]

#: SPEC §3.6's header closer. Everything before it is the channel **name**,
#: which is free text over the whole vocabulary and is not part of the answer.
_HEADER_CLOSE = "<channel|>"

#: With `--think` the answer follows this literal marker; everything before it
#: is the model's scratchpad (`eval/run_tasks.py::THINK_TOKENS`).
_THINK_MARKER = "ANSWER:"

#: Gemma's channel/turn sentinels, in their decoded form. Verified against the
#: tokenizer: 100 -> `<|channel>`, 101 -> `<channel|>`, 105 -> `<|turn>`,
#: 106 -> `<turn|>`, 50 -> `<|tool_response>`; 0/1 are SentencePiece control
#: pieces and normally decode to nothing, but are matched here so a harness that
#: renders them cannot reintroduce the bug.
_TAIL_MARKER = re.compile(
    r"<\|(?:channel|turn|tool_response)>|<(?:channel|turn)\|>|<eos>|<pad>"
)


def answer_region(text: str) -> str:
    """The span of a decoded emission a task scorer is entitled to read.

    Strips, in order:

    1. SPEC §3.6's `<|channel>NAME` header, up to and including `<channel|>`;
    2. the `--think` scratchpad, up to and including `ANSWER:`;
    3. SPEC §3.5's **unscored tail** — everything from the first channel/turn
       sentinel onwards. The tail is a property of the sampler (the automaton
       accepts `ACC --Σ--> ACC` after the answer, and early stopping does not
       prevent a final emission), not of the model's answer.

    A sentinel at position 0 is **not** a tail — there is nothing before it for
    it to terminate. The `mask` arm emits `<|channel>124\\n4321\\n…`: it opens a
    channel and never closes it, so its whole answer sits in what would have
    been the header *name*. Cutting at the first sentinel unconditionally
    deletes that arm's entire output (measured: 245/250 Sudoku and 228/250
    Countdown records scored `"empty"`), which is a worse version of the bug
    this function exists to fix. A leading sentinel is dropped and the scan
    continues.

    Note what this does *not* do: it does not go looking for well-formed
    content. Whatever survives is scored as-is, so a genuinely malformed answer
    is still reported malformed rather than quietly skipped over.

    Args:
      text: the decoded emission.

    Returns:
      The answer region, stripped of surrounding whitespace.
    """
    body = text.split(_HEADER_CLOSE, 1)[1] if _HEADER_CLOSE in text else text
    if _THINK_MARKER in body:
        body = body.split(_THINK_MARKER, 1)[1]
    while True:
        body = body.lstrip()
        m = _TAIL_MARKER.search(body)
        if m is None:
            break
        if m.start() == 0:            # an opener, not a tail -- see above
            body = body[m.end():]
            continue
        body = body[: m.start()]
        break
    return body.strip()


#: Reasons that mean *the scorer found no answer at all*, as opposed to an
#: answer that is wrong.
#:
#: **[AUDIT-E]** `eval/run_tasks.py` counted parsed records with
#: `n_parsed += int(why != "empty")`. Countdown returns `"empty"`; Sudoku
#: returned `"only 0 digits"`, so the predicate was true for **every** Sudoku
#: record including the ones that emitted nothing — a parse column that is 100%
#: by construction, the same shape of vacuity as `Σ_v q_i(v) == 1`. The two
#: scorers now agree on `"empty"`, and the predicate is named rather than
#: spelled out at the call site so a third task cannot drift from it.
NO_ANSWER_REASONS: frozenset[str] = frozenset({"empty", "no grid"})


def produced_an_answer(why: str) -> bool:
    """Did the emission contain something the scorer could score at all?

    True for a *wrong* answer — that is the point. `"arithmetic: 9+17 = 26"`
    is a model failure and counts as parsed; `"empty"` is not.
    """
    return why not in NO_ANSWER_REASONS
