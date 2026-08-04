"""E2/E4 experiment flags: the prompt contract and the grammar bundle.

The measurement that motivates all of this: the production grammar accepts
**0 of the 130 outputs the unconstrained 0.641-accuracy arm actually
produced**. These flags are the two candidate fixes, and each one has to be
exactly what it claims before any GPU time is spent on it.
"""

from __future__ import annotations

import pytest

from diffgemma_fa.compile import pipeline
from diffgemma_fa.compile.validate import Simulator
from diffgemma_fa.compile.vocab import END_TOKENS
from diffgemma_fa.eval.run import (
    PROMPT_STYLES,
    PRETTY_WS,
    _case_insensitive_enums,
    build_prompt,
)


def _accepts(automaton, text: str) -> bool:
    """Simulate `text` **plus a stop token**.

    The compiled automaton is `[HEADER ·] GRAMMAR · STOP · Σ*` (SPEC §3.5), so
    a bare grammar string ends in a live-but-not-accepting state and is *not*
    accepted on its own. Omitting the stop token makes every acceptance check
    below silently False — which it did on the first draft of this file, and
    would have made a grammar-replay experiment report 0/130 for every variant.
    Same trap CLAUDE.md flags for `test_guarantee.py`'s per-block canvases.
    """
    from gemma import gm
    tok = gm.text.Gemma3Tokenizer()
    return Simulator(automaton).accepts(tok.encode(text) + [END_TOKENS[0]])


class _Rec:
    id = "t"
    question = [[{"content": "What is the weather in Paris?"}]]
    functions = [{"name": "get_weather",
                  "parameters": {"type": "dict",
                                 "properties": {"city": {"type": "string"}},
                                 "required": ["city"]}}]


def test_the_compact_prompt_names_every_habit_the_grammar_forbids():
    """E2 is a hypothesis test, not a nudge: each clause must correspond to a
    measured mismatch — newlines (0/130 accepted), fences (126/130 present),
    and omitted keys (the shortest-member collapse)."""
    p = build_prompt(_Rec(), _Rec.functions[0], "compact")
    low = p.lower()
    assert "single-line" in low and "no newlines" in low
    assert "no code fences" in low
    assert "every listed key" in low


def test_the_stock_prompt_is_unchanged_so_the_arms_stay_comparable():
    stock = build_prompt(_Rec(), _Rec.functions[0], "stock")
    assert stock.endswith("keys in this order: ['city'].")
    assert PROMPT_STYLES["stock"] == ""


def test_case_insensitive_enums_cannot_manufacture_a_wrong_answer():
    """Every added variant normalises to the same scored value under BFCL's
    lowercase-and-strip rule, so this can only ever *admit* what the scorer
    would already accept."""
    from diffgemma_fa.eval.metrics import normalise
    out = _case_insensitive_enums(
        {"type": "dict", "properties": {"x": {"type": "string",
                                              "enum": ["PIZZA", "salad"]}}})
    variants = out["properties"]["x"]["enum"]
    assert "pizza" in variants and "Pizza" in variants
    assert {normalise(v) for v in variants} == {"pizza", "salad"}, (
        "a variant normalises to something the original enum did not — that "
        "would be a new wrong answer, not a relaxation"
    )


def test_case_insensitive_enums_leaves_non_string_enums_alone():
    out = _case_insensitive_enums({"enum": [1, 2, None]})
    assert out["enum"] == [1, 2, None]


def test_pretty_whitespace_admits_indented_json_and_the_compact_form():
    """P1. The pattern is structured rather than `[ \\n\\t]{0,6}` so it cannot
    admit blank lines; it must still accept the compact and `": "` styles the
    current grammar relies on."""
    import re
    rx = re.compile(PRETTY_WS)
    for ok in ("", " ", "\n  ", "\n    ", "\n      "):
        assert rx.fullmatch(ok), f"must admit {ok!r}"
    for bad in ("\n\n", "      ", "\t"):
        assert not rx.fullmatch(bad), f"must not admit {bad!r}"


@pytest.fixture(scope="module")
def fenced():
    return pipeline.compile_json_schema(
        _Rec.functions[0]["parameters"], name="get_weather", from_bfcl=True,
        whitespace_pattern=PRETTY_WS, fence=True, channel_header=False,
    ).automaton


def test_the_fence_is_optional_and_both_branches_are_accepted(fenced):
    """P2. Optional, because the fence is a habit and not a guarantee; if it
    were required, a model that skipped it would be unable to emit anything.

    This test was a strict xfail for one commit. The regex-level wrap
    (`r"(```json\n)?" + regex + r"(\n```)?"`) is correct under
    `re.fullmatch` but its **closing** branch is lost in `lift_regex` — the
    lifted DFA walks the opening fence and then has no outgoing newline edge
    from the grammar-final state. `automaton.wrap_with_fence` does the same
    thing as DFA surgery instead, the way `prepend_channel_header` already
    handles literal token prefixes, and the branch survives.

    Worth keeping in mind: whitespace tolerance alone accepts 0/130 of the
    model's own outputs, whitespace + fence accepts 74/130. A silently
    no-op fence would have made experiment E4 look like a null result."""
    assert _accepts(fenced, '{"city": "Paris"}'), "compact must survive"
    assert _accepts(fenced, '```json\n{"city": "Paris"}\n```'), (
        "the fenced rendering 126/130 unconstrained outputs use must be "
        "accepted — that is the whole point of the flag"
    )
    assert _accepts(fenced, '{\n  "city": "Paris"\n}'), (
        "indented JSON must survive: this is P1's target"
    )


def test_the_historical_over_constraint_and_the_fixed_default():
    """This test used to assert that the *default* grammar rejects indented
    JSON. That was true and was the finding the whole experiment rested on —
    it is no longer true, because `schema.JSON_WS` (RFC 8259's own whitespace
    definition) is now the default rather than outlines' `[ ]?`.

    So the control moves: the historical pattern must still reject, proving the
    experiment measured something real, and the new default must accept, proving
    it is fixed. A test that only checked the old behaviour would now be
    protecting the bug.
    """
    params = _Rec.functions[0]["parameters"]
    old = pipeline.compile_json_schema(
        params, name="get_weather", from_bfcl=True, channel_header=False,
        whitespace_pattern="").automaton          # the pre-fix pattern
    assert _accepts(old, '{"city":"Paris"}'), "compact was always fine"
    assert not _accepts(old, '{\n  "city": "Paris"\n}'), (
        "if the old pattern now accepts indented JSON, the historical finding "
        "was misattributed"
    )

    new = pipeline.compile_json_schema(
        params, name="get_weather", from_bfcl=True,
        channel_header=False).automaton           # JSON_WS by default
    assert _accepts(new, '{"city": "Paris"}')
    assert _accepts(new, '{\n  "city": "Paris"\n}'), (
        "the default must admit the model's own rendering; that is the whole "
        "point of making JSON_WS the default"
    )
    # The fence remains a separate opt-in -- whitespace does not imply it.
    assert not _accepts(new, '```json\n{"city": "Paris"}\n```')


def test_the_bundle_is_a_superset_not_a_replacement(fenced):
    """Relaxation only: anything the stock grammar accepted must still be
    accepted, or the flags would be trading one forced rendering for another."""
    from gemma import gm
    tok = gm.text.Gemma3Tokenizer()
    stock = pipeline.compile_json_schema(
        _Rec.functions[0]["parameters"], name="get_weather", from_bfcl=True,
        channel_header=False).automaton
    for s in ('{"city": "Paris"}', '{"city":"Paris"}', '{ "city": "x" }'):
        ids = tok.encode(s)
        if Simulator(stock).accepts(ids):
            assert Simulator(fenced).accepts(ids), f"bundle lost {s!r}"


def test_the_fence_wrap_keeps_the_bare_rendering_reachable():
    """`wrap_with_fence` makes the fence chain the automaton's start, so the
    unfenced path has to be re-wired from state 0 explicitly. If that wiring is
    dropped, the fence stops being optional and becomes mandatory — which would
    make every model that omits it unable to emit anything at all."""
    from diffgemma_fa.compile.automaton import wrap_with_fence
    from diffgemma_fa.compile.minimize import Dfa
    # `a b` accepted; fence tokens are 2717/3723/107.
    g = Dfa(n_states=3, transitions=((0, 500, 1), (1, 501, 2)), start=0,
            finals=frozenset({2}))
    w = wrap_with_fence(g)
    T = {}
    for s, a, d in w.transitions:
        T.setdefault((s, a), d)

    def run(seq):
        cur = w.start
        for t in seq:
            if (cur, t) not in T:
                return False
            cur = T[(cur, t)]
        return cur in w.finals

    assert run([500, 501]), "the bare rendering must stay accepted"
    assert run([2717, 3723, 107, 500, 501]), "opening fence only"
    assert run([2717, 3723, 107, 500, 501, 107, 2717]), "both fences"
    assert run([500, 501, 107, 2717]), "closing fence only"
    assert not run([2717, 500, 501]), "a partial opening fence must not pass"


def test_the_eval_runner_does_not_shadow_the_model_params():
    """A regression guard for a bug that cost a full overnight queue.

    `eval/run.py` binds `params = gm.ckpts.load_params(CKPT)` once, then loops
    over records. The `--ci-enums` change introduced a second `params = ...`
    inside that loop holding the *schema's* parameters, so every subsequent
    `_prefill.prefill(params=params, ...)` received a dict of schema strings
    and died inside gemma's own `_dtype()` with "'str' object has no attribute
    'dtype'". Six queued arms failed identically and the checkpoint looked
    corrupt; it was fine.

    Checked by source inspection because reproducing it needs the 51 GB model.
    """
    import inspect
    import re
    from diffgemma_fa.eval import run as R

    src = inspect.getsource(R.main)
    # `\s*=` would also catch the keyword argument `params=params`; require
    # a space before `=` so only real assignments match.
    binds = re.findall(r"^\s*params\s+=\s*(.+)$", src, re.M)
    assert len(binds) == 1, (
        f"`params` is bound {len(binds)} times in eval.run.main: {binds}. "
        "It must name the model weights and nothing else."
    )
    assert "load_params" in binds[0]
