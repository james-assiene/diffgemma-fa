"""Audit of the measurement and scoring layer. SPEC §7.2, §4.8, §3.5, §3.6.

**Why this file exists, separately from `test_metrics.py` and
`test_tasks_new.py`.** Those two pin the bugs that were *already found*. This
one is written from the outside — from BFCL's own checker source, from the
compiled automaton, and from emissions recorded verbatim in `artifacts/` — to
find the next member of the same family. The family is:

    a number is computed from the raw emission (or over a denominator that
    depends on the emission), something upstream shifts what is being read
    (or which records are counted), and the number is wrong in a direction
    nobody notices because it still looks plausible.

Four instances have already been paid for:

1. the Sudoku scorer reading digits out of the raw emission, so a digit-bearing
   channel-header name shifted the grid — *"overwrote the given"* on 250/250
   records while CS was 1.000, i.e. while the automaton had already **proved**
   the givens intact;
2. the same header bug in Countdown, where the header shares a line with the
   first step;
3. `schema_valid` validating against the un-expanded schema while the grammar
   was compiled from the ci-enum-expanded one;
4. `Sum_v q_i(v) == 1` asserted for weeks while being mathematically incapable
   of failing.

Rules this file follows, because (4) is what happens when they are not:

- **No test may re-derive its expectation from the code under test.** The
  oracles here are BFCL's vendored `standardize_string` (a different codebase),
  the compiled automaton (a different module, and the thing whose proof the
  scorer contradicted), and literal emissions copied out of the run artifacts.
- **Every test must be capable of failing.** Where a test is a regression guard
  for an already-fixed bug it says so, and the adversarial variant beside it is
  the one that can still go red.
"""

from __future__ import annotations

import ast
import copy
import json
import pathlib
import re
import types

import pytest

from diffgemma_fa.compile.tasks import countdown as CD
from diffgemma_fa.compile.tasks import sudoku as SD
from diffgemma_fa.eval.metrics import Scores, extract_json, normalise, score_arguments


# ==========================================================================
# Oracle 1 — BFCL's own value normalisation
# ==========================================================================
#
# `metrics.normalise` claims to implement SPEC §4.8: "scoring lowercases and
# strips `",./-_*^`". That sentence is a *quotation* of BFCL's docstring, and
# the outer `"` is the quote delimiter, not a member of the strip set. BFCL's
# actual rule, `bfcl_eval/eval_checker/ast_eval/ast_checker.py:174`:
#
#     regex_string = r"[ \,\.\/\-\_\*\^]"
#     return re.sub(regex_string, "", input_string).lower().replace("'", '"')
#
# — i.e. it strips **space** and does **not** strip the double quote, and it
# folds single quotes to double quotes. Transcribed here rather than imported,
# because `ast_checker` pulls in the rest of the BFCL package and the checkout
# lives under the gitignored `artifacts/data/`. `test_the_bfcl_oracle_is_a
# _faithful_transcription` re-reads the vendored source when it is present so
# the transcription cannot rot silently.

_BFCL_STRIP_RE = r"[ \,\.\/\-\_\*\^]"

_VENDORED_CHECKER = pathlib.Path(
    "/home/ubuntu/diffgemma_fa/artifacts/data/gorilla/"
    "berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/"
    "ast_checker.py"
)


def bfcl_standardize(s: str) -> str:
    """BFCL's `standardize_string`, transcribed from its source."""
    return re.sub(_BFCL_STRIP_RE, "", s).lower().replace("'", '"')


#: Values taken from the recorded runs and from BFCL ground truth, not invented:
#: the emissions in `artifacts/eval_bfcl_live_simple_unconstrained_sample.json`
#: and the `possible_answer` entries quoted in `compile/bfcl_data.py`.
_REAL_VALUES = [
    "Tel Aviv",
    "San Francisco",
    "Divinópolis, MG",
    "New York",
    "2020 Addison Street, Berkeley, CA, USA",
    "221B Baker Street, Berkeley, CA, USA",
    "Comfort",
    "Plus",
    "fahrenheit",
    "ShishirPatil/gorilla",
    "gorilla-llm/gorilla-cli",
    "AIR_CLEAN",
    "N/A",
    "April 1, 2024",
    "April 1 2024",
    "black",
    '"black"',
    "it's",
]


def test_the_bfcl_oracle_is_a_faithful_transcription():
    """Guard on the oracle itself, so a wrong oracle cannot bless a wrong metric.

    Skips rather than fails when the BFCL checkout is absent: `artifacts/data/`
    is gitignored, so a fresh clone has no vendored source to compare against.
    """
    if not _VENDORED_CHECKER.exists():
        pytest.skip("BFCL checkout not present (artifacts/data is gitignored)")
    src = _VENDORED_CHECKER.read_text()
    m = re.search(r'regex_string = r"(\[[^"]*\])"', src)
    assert m, "BFCL's standardize_string no longer has the expected shape"
    assert m.group(1) == _BFCL_STRIP_RE, (
        f"BFCL changed its strip set to {m.group(1)!r}; this file's oracle "
        f"({_BFCL_STRIP_RE!r}) is stale"
    )
    assert 'replace("\'", \'"\')' in src, (
        "BFCL's single-quote folding is gone; the oracle is stale"
    )


def test_normalise_agrees_with_bfcls_own_standardize_string():
    """The whole point of `normalise` is to be BFCL's rule and not our own.

    `metrics.py`: "Scoring more strictly than the benchmark understates the
    result and is not comparable to the leaderboard." That cuts both ways — a
    scorer that is *more lenient* than BFCL manufactures accuracy the
    leaderboard would not award.
    """
    disagreements = [
        (v, normalise(v), bfcl_standardize(v))
        for v in _REAL_VALUES
        if normalise(v) != bfcl_standardize(v)
    ]
    assert not disagreements, (
        "our normalisation is not BFCL's:\n"
        + "\n".join(f"  {v!r}: ours={a!r} BFCL={b!r}"
                    for v, a, b in disagreements)
    )


def test_normalise_strips_the_space_character():
    """The consequential half of the divergence, isolated.

    BFCL's strip set **begins with a space** — that is what lets `April 1, 2024`
    match `April 1,2024` and `April 1 2024`, which its docstring says in so many
    words. A `normalise` that keeps spaces scores `NewYork` against a ground
    truth of `New York` as WRONG, while BFCL scores it correct. That is the
    understatement `metrics.py`'s own docstring forbids.
    """
    assert normalise("New York") == normalise("newyork")
    assert normalise("April 1, 2024") == normalise("April 1 2024")


def test_normalise_does_not_invent_a_strip_of_the_double_quote():
    """The other half: `"` is the quote delimiter in SPEC §4.8's citation, not
    a member of the set. Stripping it makes us **more lenient** than BFCL — a
    model that emits the literal string `"black"` (quotes inside the JSON
    string) is scored as having emitted `black`, which BFCL rejects.

    NOTE FOR THE REVIEWER: this contradicts
    `test_metrics.py::test_normalise_matches_bfcls_own_rule`, which asserts
    `normalise('"black"') == normalise("black")`. That existing assertion
    encodes the misreading; one of the two has to go, and BFCL's source is the
    tiebreaker.
    """
    assert normalise('"black"') != normalise("black")


def test_non_string_values_are_compared_exactly_as_bfcl_compares_them():
    """BFCL normalises **only** strings.

    `ast_checker.simple_function_checker` routes `str` through
    `string_checker` (which standardises) and everything else through a bare
    `value not in possible_answer[param]` — exact equality. Applying the string
    rule to numbers deletes the decimal point and the minus sign, so `1.0`
    becomes `10` and `-5` becomes `5`. Those are not near-misses being
    forgiven; they are different answers being counted correct.
    """
    assert score_arguments({"n": 1.0}, {"n": 10})[0] == 0, "1.0 is not 10"
    assert score_arguments({"n": 1.5}, {"n": 15})[0] == 0, "1.5 is not 15"
    assert score_arguments({"t": -5}, {"t": 5})[0] == 0, "-5 is not 5"


def test_a_string_is_not_credited_for_matching_a_boolean():
    """`{"aligned": "true"}` against a ground truth of `true`.

    BFCL's `type_checker` rejects it before any value comparison (`str` is not
    `bool`); `str()`-then-lowercase makes them identical. The `aligned`
    parameter is real — it appears in
    `artifacts/eval_bfcl_live_simple_unconstrained_sample.json`.
    """
    assert score_arguments({"aligned": "true"}, {"aligned": True})[0] == 0


# ==========================================================================
# Oracle 2 — denominators that must not depend on the emission
# ==========================================================================
#
# Measured, in the project's own artifacts, over the SAME 130 records:
#
#   arm             parsed   arg_total   arg_accuracy   exact_call_rate
#   mask              67       120          0.5083          0.2308
#   j1_sample        128       287          0.3798          0.2846
#   unconstrained    126       276          0.6413          0.4692
#
# `mask` has the *highest* `arg_accuracy` of every arm and the *lowest*
# `exact_call_rate`, because 63 records that produced nothing at all were
# dropped from `arg_total` and kept in `n`. A reader of that table concludes
# the naive baseline is the most accurate arm.
#
# Its comparable accuracy is at most **61/291 = 0.2096** — 291 being the
# ground-truth key count over all 130 records, which is the denominator every
# arm must share; the 287 quoted in an earlier draft of this file was `j1`'s
# parsed-only figure and understated the denominator by four. (The rebuilt
# `mask` arm reports 0.2165 rather than 0.2096 because Finding A independently
# turns two previously-wrong arguments correct; the bound applies to the old
# `arg_correct` of 61, not the new one.)


def test_an_arm_that_emits_nothing_cannot_outscore_one_that_answers():
    """The `mask` row above, reduced to ten records.

    Arm A answers every record and gets 5 of 10 right. Arm B answers one
    record, gets it right, and emits garbage on the other nine. No defensible
    accuracy metric ranks B above A.
    """
    a = Scores()
    for i in range(10):
        a.add(accepted=True, parsed_obj={"k": "right" if i < 5 else "wrong"},
              want={"k": "right"})
    b = Scores()
    b.add(accepted=True, parsed_obj={"k": "right"}, want={"k": "right"})
    for _ in range(9):
        b.add(accepted=True, parsed_obj=None, want={"k": "right"})

    assert b.as_dict()["arg_accuracy"] < a.as_dict()["arg_accuracy"], (
        f"the silent arm scores {b.as_dict()['arg_accuracy']} against "
        f"{a.as_dict()['arg_accuracy']} for the arm that actually answered"
    )


def test_arg_total_counts_every_ground_truth_key_regardless_of_parsing():
    """`arg_total` is a property of the *benchmark*, not of the model.

    Three records, two ground-truth keys each: the denominator is six. Whether
    the model's output happened to parse cannot change how many arguments the
    benchmark asked for, and if it does then two arms are quoting the same
    column over different populations.
    """
    s = Scores()
    want = {"a": "x", "b": "y"}
    s.add(accepted=True, parsed_obj={"a": "x", "b": "y"}, want=want)
    s.add(accepted=True, parsed_obj={"a": "x", "b": "NO"}, want=want)
    s.add(accepted=False, parsed_obj=None, want=want)      # did not parse
    assert s.as_dict()["arg_total"] == 6


def test_arg_accuracy_and_exact_call_rate_are_over_the_same_records():
    """Two headline columns, printed side by side, must mean the same thing.

    One perfect record and one unparsable record, both with ground truth: the
    honest answer is 0.5 for both columns. Today `exact_call_rate` divides by
    `n` (2) and `arg_accuracy` divides by the parsed-only `arg_total` (1), so
    the same run reports 0.5 and 1.0 for "how much of this did it get right".
    """
    s = Scores()
    s.add(accepted=True, parsed_obj={"a": "x"}, want={"a": "x"})
    s.add(accepted=False, parsed_obj=None, want={"a": "x"})
    d = s.as_dict()
    assert d["exact_call_rate"] == 0.5
    assert d["arg_accuracy"] == 0.5, (
        f"arg_accuracy={d['arg_accuracy']} over arg_total={d['arg_total']} "
        f"while exact_call_rate={d['exact_call_rate']} over n={d['n']}"
    )


def test_a_perfect_arg_accuracy_requires_having_answered():
    """The degenerate case the table cannot distinguish today.

    An arm that parses exactly one record out of a hundred and gets it right
    reports `arg_accuracy = 1.000`. That number belongs beside `parsed`, and
    the two are printed in different columns of the same row — nothing stops a
    reader (or a later report) from quoting it alone.
    """
    s = Scores()
    s.add(accepted=True, parsed_obj={"a": "x"}, want={"a": "x"})
    for _ in range(99):
        s.add(accepted=False, parsed_obj=None, want={"a": "x"})
    assert s.as_dict()["arg_accuracy"] < 0.5


# --------------------------------------------------------------------------
# The same hole through the other door: records that never generated
# --------------------------------------------------------------------------
#
# `Scores.add` treats `want=None` as "this record has no ground truth, score
# nothing". `eval/run.py` reuses that sentinel for records that *do* have
# ground truth but failed before producing an emission — OOM and
# `ZeroPartitionError` — so those records sit in `n` (and therefore in
# `cs_rate`, `schema_valid_rate` and `exact_call_rate`) while being absent from
# `arg_total`. It is Finding B exactly, entering through the harness rather
# than through `add`'s early return, and the Finding-B fix does not close it.
#
# Magnitude, from `artifacts/exp_e5_grammar130_j0.json`: **70 zero-partition
# records out of 130 (54%)**. Its 60 surviving records carry 112 of the 291
# ground-truth arguments, so with the Finding-A comparison it reports
# `arg_accuracy = 82/112 = 0.7321` — the best in the table — for an arm that
# failed to generate on more than half its records. The honest figure over the
# denominator every other arm uses is `82/291 = 0.2818`.

_RUN_PY = pathlib.Path("/home/ubuntu/diffgemma_fa/diffgemma_fa/eval/run.py")


def _scores_add_calls_with_want(source: str | None = None):
    """Every `.add(...)` call in `eval/run.py` that passes a `want=` keyword.

    Returns `[(lineno, want_node, enclosing_except_or_None)]`. Located by AST
    rather than by grep so that a reformat, a renamed accumulator or an extra
    keyword cannot make this test quietly stop looking.

    `source` overrides the file, for the mutation self-test below.
    """
    import ast

    tree = ast.parse(_RUN_PY.read_text() if source is None else source)
    handler_lines: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            label = ast.unparse(node.type) if node.type else "bare except"
            for sub in ast.walk(node):
                if hasattr(sub, "lineno"):
                    handler_lines.setdefault(sub.lineno, label)

    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add"):
            continue
        for kw in node.keywords:
            if kw.arg == "want":
                out.append((node.lineno, kw.value,
                            handler_lines.get(node.lineno)))
    return out


#: `eval/run.py` as it stood when this hole was found, reduced to the shape the
#: detector has to recognise. Both handlers pass `want=None`; the success path
#: passes the real `want`.
_PRE_FIX_HARNESS = '''
def main():
    for rec in records:
        try:
            state = sampler.sample_constrained()
        except jax.errors.JaxRuntimeError:
            oom.append(rec.id)
            sc.add(accepted=False, parsed_obj=None, want=None, schema_ok=False)
            continue
        except ZeroPartitionError:
            zero_partition.append(rec.id)
            sc.add(accepted=False, parsed_obj=None, want=None, schema_ok=False)
            continue
        sc.add(accepted=accepted, parsed_obj=parsed, want=want, schema_ok=ok)
'''


def test_the_want_none_detector_actually_detects_want_none():
    """Mutation self-test for the two source-level tests below.

    Both of them now pass, because the harness was fixed. A guard that passes
    is worth exactly as much as its ability to fail, and this one's subject is
    a file it does not control — so the detector is run against the pre-fix
    source and required to find the defect it was written for. Without this,
    a helper that silently stopped matching `.add(...)` would turn both guards
    green and nobody would know.
    """
    import ast

    calls = _scores_add_calls_with_want(_PRE_FIX_HARNESS)
    assert len(calls) == 3, f"detector found {len(calls)} call sites, expected 3"

    offenders = [(ln, ctx) for ln, node, ctx in calls
                 if isinstance(node, ast.Constant) and node.value is None]
    assert len(offenders) == 2, (
        f"detector found {len(offenders)} want=None sites in the pre-fix "
        f"harness, expected the OOM and zero-partition handlers"
    )
    assert all(ctx is not None for _, ctx in offenders), (
        "both offenders must be attributed to their enclosing `except`, which "
        "is what makes the failure message actionable"
    )
    assert {ctx for _, ctx in offenders} == {
        "jax.errors.JaxRuntimeError", "ZeroPartitionError"}


def test_the_harness_never_scores_a_record_as_having_no_ground_truth():
    """`want=None` means "the benchmark asked nothing of this record".

    It must never be used to mean "the model failed to answer this one", and
    `eval/run.py` has the ground truth in scope at both failure sites — `truth`
    is loaded before the loop and keyed by `rec.id`, so the OOM and
    zero-partition handlers can pass the real `want` and let the record score
    zero out of its full argument count, which is what happened.

    Written against the source rather than against a run because the two
    failure paths need a device to reach: a 51 GB checkpoint for OOM and a
    grammar with no in-budget completion for zero-partition. The invariant is
    lexical and does not need either.
    """
    import ast

    calls = _scores_add_calls_with_want()
    # Non-vacuity: if a refactor moved these calls somewhere this helper cannot
    # see, the test must fail rather than pass by finding nothing to check.
    assert len(calls) >= 3, (
        f"only found {len(calls)} `.add(want=...)` call sites in {_RUN_PY.name}; "
        f"this test is no longer looking at the harness it was written for"
    )

    literal_none = [
        (lineno, ctx) for lineno, node, ctx in calls
        if isinstance(node, ast.Constant) and node.value is None
    ]
    assert not literal_none, (
        "these call sites tell Scores the benchmark asked nothing of the "
        "record, which removes it from arg_total while leaving it in n:\n"
        + "\n".join(f"  {_RUN_PY.name}:{ln}  (inside `except {ctx}`)"
                    for ln, ctx in literal_none)
    )


def test_the_denominator_policy_string_is_true_of_the_argument_denominator():
    """The artifact ships a `denominator_policy` field. Make it falsifiable.

    A prose field that no test can contradict documents whatever the code
    happens to do, which is worse than no field — a reader trusts it. Today it
    promises "oom and zero_partition records stay in n as failures", and that
    is true of `cs_rate`, `schema_valid_rate` and `exact_call_rate` and false
    of `arg_accuracy`, which is the column the promise most affects.

    So: read the claim out of the source, and hold the code to it.
    """
    import ast

    tree = ast.parse(_RUN_PY.read_text())
    policy = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant)
                        and key.value == "denominator_policy"):
                    policy = ast.literal_eval(value)
    assert policy, "eval/run.py no longer states a denominator_policy"

    # Case-folded: the field is prose, and "Oom" at the start of a sentence is
    # the same claim as "oom".
    claim = policy.lower()
    claims_failures_are_kept = (
        "oom" in claim and "zero_partition" in claim and "stay in n" in claim
    )
    if not claims_failures_are_kept:
        pytest.fail(
            "the denominator_policy no longer makes the claim this test "
            f"checks; re-read it and re-point the test:\n  {policy!r}"
        )

    offenders = [
        lineno for lineno, node, ctx in _scores_add_calls_with_want()
        if ctx is not None
        and isinstance(node, ast.Constant) and node.value is None
    ]
    assert not offenders, (
        f"denominator_policy says {policy!r}\n"
        f"but the failure handlers at lines {offenders} pass want=None, which "
        f"keeps those records in `n` and drops them from `arg_total`. The "
        f"policy is true of three of the four rates it covers."
    )


def test_a_generation_failure_scores_zero_rather_than_disappearing():
    """The behavioural half: what the harness must produce once it passes
    `want`.

    `exp_e5_grammar130_j0`'s shape in miniature — three records answered
    perfectly, seven that never generated, two ground-truth arguments each. The
    honest accuracy is 6/20; dropping the seven gives 6/6. This passes today
    (the `parsed_obj is None` path was fixed), which is exactly what makes the
    two source-level tests above worth failing: the correct call already
    produces the correct number, and the harness simply is not making it.
    """
    want = {"a": "x", "b": "y"}
    s = Scores()
    for _ in range(3):
        s.add(accepted=True, parsed_obj={"a": "x", "b": "y"}, want=want)
    for _ in range(7):
        s.add(accepted=False, parsed_obj=None, want=want)
    d = s.as_dict()
    assert d["arg_total"] == 20
    assert d["arg_accuracy"] == 0.3
    assert d["exact_call_rate"] == 0.3, (
        "the two accuracy columns must still be over the same records"
    )


# ==========================================================================
# Oracle 3 — the compiled automaton, against the scorers it contradicted
# ==========================================================================
#
# The 250-record Sudoku arm was lost because the scorer said "overwrote the
# given" while the automaton said the string was in the language — and the
# language pins the givens. Whenever those two can disagree, the automaton is
# the proof and the scorer is the suspect. These tests wire the two together
# so the disagreement is caught by CI rather than by a wasted GPU-day.

pipeline = pytest.importorskip("diffgemma_fa.compile.pipeline")


@pytest.fixture(scope="module")
def tokenizer():
    from diffgemma_fa.compile import vocab as V
    return V.gemma_tokenizer()


@pytest.fixture(scope="module")
def end_of_turn():
    from diffgemma_fa.compile import vocab as V
    return V.END_TOKENS[1]          # 106, decodes to `<turn|>`


@pytest.fixture(scope="module")
def sudoku_record():
    return SD.generate(1, seed=0)[0]


@pytest.fixture(scope="module")
def sudoku_fa(sudoku_record):
    from diffgemma_fa.compile.tasks.grammars import sudoku_regex
    from diffgemma_fa.compile.validate import Simulator
    a = pipeline.compile_regex(
        sudoku_regex(sudoku_record.puzzle, row_separator="\n"),
        name="audit_sudoku").automaton
    return Simulator(a)


@pytest.fixture(scope="module")
def countdown_fa():
    from diffgemma_fa.compile.tasks.grammars import countdown_regex
    from diffgemma_fa.compile.validate import Simulator
    a = pipeline.compile_regex(
        countdown_regex(max_steps=4, max_value=999, step_separator=r"\n"),
        name="audit_countdown").automaton
    return Simulator(a)


def _grid(rows) -> str:
    return "\n".join("".join(str(c) for c in row) for row in rows)


def test_an_empty_emission_is_never_constraint_satisfied(countdown_fa):
    """100 of the 250 unconstrained Countdown records emitted the empty string.

    If the start state were final, every one of them would have been counted as
    CS-satisfied and the headline column would be meaningless. Cheap to check,
    catastrophic to get wrong.
    """
    assert not countdown_fa.accepts([])


def test_sudoku_scorer_never_contradicts_the_automaton_on_the_givens(
        sudoku_record, sudoku_fa, tokenizer, end_of_turn):
    """The 250-record bug, stated as the invariant that would have caught it.

    Every string in the language has the prefilled cells pinned — that is what
    `sudoku_regex` is for, and `test_tasks_new.py` checks it at the regex level.
    So for any emission the *automaton accepts*, "overwrote the given" is a
    statement the scorer is not entitled to make, whatever the header contains.

    The header names below are verbatim from
    `artifacts/task_sudoku_j0map.json` and `artifacts/task_sudoku_think.json`,
    including the two that are pure digits and the one that contains a whole
    grid.
    """
    grid = _grid(sudoku_record.solution)
    n_accepted = 0
    for name in ("312", "2", "43", "21431", "214314321", "thought", " de", "中"):
        text = f"<|channel>{name}\n<channel|>{grid}<turn|>"
        toks = tokenizer.encode(text)
        if not sudoku_fa.accepts(toks):
            continue          # not in the language; the scorer owes nothing
        n_accepted += 1
        ok, why = SD.score_solution(text, sudoku_record)
        assert "given" not in why, (
            f"scorer says {why!r} for an emission the automaton ACCEPTS, and "
            f"the language pins the givens (header name {name!r})"
        )
        assert "digits" not in why, f"header {name!r}: {why!r}"
        assert ok, f"header {name!r}: {why!r}"

    # Without this the test passes by asserting nothing the moment the header
    # grammar changes shape -- the `Sum_v q_i(v) == 1` failure mode.
    assert n_accepted >= 3, (
        f"only {n_accepted} of the 8 header names produced a string in the "
        f"language; this test is no longer exercising the invariant"
    )


def test_countdown_scorer_never_reports_a_format_failure_on_an_accepted_string(
        countdown_fa, tokenizer):
    """The same invariant for Countdown, and the one that is still open.

    `countdown_regex` guarantees the shape `A op B=C` per line; the scorer is
    entitled to reject the *arithmetic*, the *operand pool* and the *target*,
    and nothing else. "unparsable step" on a string the automaton accepts means
    the scorer and the grammar disagree about what a step looks like.

    The emission shape is verbatim from `artifacts/task_countdown_j0map.json`:
    a header whose free-text name happens to be an arithmetic step, and a
    `<turn|>` end-of-turn marker welded onto the last step's line. (One step,
    because one step is all the compiled automaton admits — see
    `test_the_compiled_countdown_grammar_admits_a_multi_step_solution`.)
    """
    rec = CD.CountdownRecord(id="t", numbers=(3, 4), target=12)
    text = "<|channel>82-80=2\n<channel|>3*4=12<turn|>"
    toks = tokenizer.encode(text)
    assert countdown_fa.accepts(toks), (
        "test construction error: this emission is not in the language"
    )
    ok, why = CD.score_solution(text, rec)
    assert "unparsable" not in why, (
        f"scorer says {why!r} for an emission the automaton ACCEPTS"
    )
    assert ok, why


# --------------------------------------------------------------------------
# The extreme case: a grammar that cannot express the thing being scored
# --------------------------------------------------------------------------
#
# SCOPE NOTE FOR THE CODER. The two tests below fail in
# `compile/pipeline.py` (regex lifting), not in the measurement layer. They are
# here because they are the reason a measurement is meaningless: every
# constrained Countdown arm reports `cs_rate = 1.000` and `solve_rate = 0.000`,
# and all 250 emissions in `artifacts/task_countdown_j0map.json` are a single
# step, because a second step is not in the compiled language. A run whose
# answer is unreachable by construction is not a measurement of the model.


def test_the_compiled_countdown_grammar_admits_a_multi_step_solution(
        countdown_fa, tokenizer):
    """`build_prompt` asks for "one step per line"; `score_solution` walks a
    chain of steps and consumes intermediates; `countdown_regex(max_steps=4)`
    says four are allowed; and `test_tasks_new.py` checks the *regex* matches
    `3*4=12\\n12+5=17`. The compiled automaton admits only the first step, so
    no record needing two operations can ever be solved.
    """
    text = "<|channel>thought\n<channel|>3*4=12\n12+5=17<turn|>"
    assert countdown_fa.accepts(tokenizer.encode(text)), (
        "the compiled Countdown automaton rejects a two-step solution that "
        "countdown_regex matches; every constrained Countdown arm was scored "
        "against a grammar that cannot express the answer"
    )


@pytest.mark.parametrize("regex,string", [
    (r"1(?:x1){0,3}", "1x1"),
    (r"1(?:x2)*", "1x2x2"),
    (r"a(?:ba)*", "ababa"),
    (r"ab(?:cab)*", "abcab"),
])
def test_lifting_preserves_repeated_groups(regex, string, tokenizer):
    """Minimal reproducer, with `re` as the independent oracle.

    A repeated *group* is lifted as though it repeated its minimum number of
    times: `(?:x1){0,3}` becomes zero copies, `(?:\\n1){1,3}` becomes exactly
    one. Character-level repetition (`x*`, `[0-9]{0,3}`, `[^"]*`) is unaffected,
    which is why it has stayed invisible — and also why the Countdown step
    separator, the only group repetition in a shipped grammar, is the place it
    surfaced.
    """
    from diffgemma_fa.compile.validate import Simulator
    assert re.fullmatch(regex, string), "test construction error"
    a = pipeline.compile_regex(regex, name="rep", channel_header=False).automaton
    toks = tokenizer.encode(string) + [106]
    assert Simulator(a).accepts(toks), (
        f"compiled automaton for {regex!r} rejects {string!r}, which the "
        f"source regex matches"
    )


# --------------------------------------------------------------------------
# The end-of-turn marker, isolated from the automaton fixtures
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tail", ["<turn|>", "<|channel>", "\n\n", ""])
def test_countdown_scoring_survives_the_post_stop_tail(tail):
    """SPEC §3.5 makes the post-stop tail **unscored** (`ACC --Sigma--> ACC`),
    so the automaton accepts whatever follows the answer and the decoded text
    keeps it. Every recorded Countdown emission carries one:

        '<|channel>thought\\n<channel|>51-4=49<turn|>'          (j0, map)
        '<channel|>79+17=96\\n96-60=36<|channel>'               (unconstrained)

    `_STEP` is anchored with `\\s*$`, so the marker lands inside the last step's
    line and that step is "unparsable". The tail is a property of the sampler,
    not of the model's answer, and it must not be scored.
    """
    rec = CD.CountdownRecord(id="t", numbers=(3, 4, 5), target=17)
    ok, why = CD.score_solution(f"<|channel>x\n<channel|>3*4=12\n12+5=17{tail}",
                                rec)
    assert ok, f"tail {tail!r}: {why!r}"


def test_a_wrong_answer_is_not_reported_as_a_format_failure():
    """`run_tasks.py`: "a format failure and a wrong answer are different
    diagnoses and must not be merged."

    Verbatim from `artifacts/task_countdown_think.json`, whose
    `failure_reasons` attributes 111 of 250 records to "unparsable step". This
    emission's step is perfectly well formed and simply false — `9+17` is 26,
    not 36 — so the diagnosis should be "arithmetic". Merging the two makes the
    `failure_reasons` histogram, which is the only diagnostic that arm
    produces, point at the grammar instead of at the model.
    """
    rec = CD.CountdownRecord(id="t", numbers=(9, 17), target=36)
    ok, why = CD.score_solution("<|channel>79-60=19\n<channel|>9+17=36<turn|>",
                                rec)
    assert not ok
    assert "arithmetic" in why, f"diagnosed as {why!r}"


# --------------------------------------------------------------------------
# Sudoku: locating the grid, on arms that have no channel header
# --------------------------------------------------------------------------

def test_sudoku_scorer_handles_the_headerless_unconstrained_emission(
        sudoku_record):
    """Regression guard. Verbatim shapes from
    `artifacts/task_sudoku_unconstrained.json`: sometimes the header is there,
    sometimes only its closing marker, sometimes neither, and there is a
    trailing blank line.
    """
    grid = _grid(sudoku_record.solution)
    for text in (grid, f"{grid}\n\n", f"<channel|>{grid}<turn|>",
                 f"<|channel>214314321\n<channel|>{grid}"):
        ok, why = SD.score_solution(text, sudoku_record)
        assert ok, f"{text!r}: {why}"


def test_sudoku_scorer_locates_the_grid_structurally_not_by_first_16_digits(
        sudoku_record):
    """The unfixed half of the 250-record bug.

    The fix stripped the *channel header*, which is the only source of stray
    digits on the constrained arms. The `unconstrained` and `mask` arms have no
    header — they are the comparison baseline, and they emit prose. Any digit
    in that prose shifts the grid by one cell exactly as the header name did,
    and the scorer will again report "overwrote the given" for a grid that is
    intact.

    "4x4" is not a contrived example: it is the phrase the prompt itself uses
    ("Solve this 4x4 Sudoku"), so it is the single most likely token to appear
    in a restatement of the question.
    """
    grid = _grid(sudoku_record.solution)
    for preamble in ("Here is the completed 4x4 grid:\n",
                     "Sure! The solution to the 4x4 puzzle is:\n",
                     "Answer (row 1 first):\n"):
        ok, why = SD.score_solution(preamble + grid, sudoku_record)
        assert ok, f"{preamble!r} shifted the grid: {why}"


def test_sudoku_scorer_ignores_a_restatement_of_the_puzzle_after_the_answer(
        sudoku_record):
    """The tail direction of the same question.

    Reading the *first* 16 digits is right only if nothing precedes the grid;
    it is silently right here and silently wrong above, which is why the rule
    has to be structural. Kept as a guard so a structural fix does not break
    the case the current rule gets right.
    """
    grid = _grid(sudoku_record.solution)
    ok, why = SD.score_solution(f"{grid}\n\nCheck: 1234 in every row.",
                                sudoku_record)
    assert ok, why


def test_both_task_scorers_agree_on_how_to_say_no_output(sudoku_record):
    """`run_tasks.py:211` counts parsed records with a rule shared by both
    tasks:

        n_parsed += int(why != "empty")

    Countdown returns `"empty"`; Sudoku returns `"only 0 digits"`. The rule is
    therefore true for every Sudoku record, including the ones that emitted
    nothing at all, so the parse column for Sudoku is 100% by construction —
    the same shape of vacuity as `Sum_v q_i(v) == 1`. (It is currently unsaved,
    which is the only reason no published number depends on it.)
    """
    rec = CD.CountdownRecord(id="t", numbers=(3, 4), target=7)
    assert CD.score_solution("", rec)[1] == "empty"
    assert SD.score_solution("", sudoku_record)[1] == "empty"


# ==========================================================================
# Oracle 4 — the schema column and the ci-enum expansion
# ==========================================================================
#
# "5 of 7 schema-invalid records" were an emitted `"pizza"` against an enum of
# `PIZZA`: the grammar was compiled from the expanded schema and the validator
# was handed the un-expanded one. `run.py` now derives both from
# `schema_params`. These tests pin that they cannot drift apart again.


def _norm(schema):
    from diffgemma_fa.compile.schema import normalize_bfcl_schema
    return normalize_bfcl_schema(schema)


def test_the_validator_accepts_the_case_variant_the_grammar_admits():
    """Regression guard for the 5-of-7 bug, and a statement of the invariant:
    whatever schema the grammar was compiled from is the schema the validator
    must see.
    """
    from diffgemma_fa.eval.metrics import schema_valid
    from diffgemma_fa.eval.run import _case_insensitive_enums

    raw = {"type": "dict",
           "properties": {"topping": {"type": "string", "enum": ["PIZZA"]}},
           "required": ["topping"]}
    expanded = _case_insensitive_enums(raw)

    assert schema_valid({"topping": "pizza"}, _norm(expanded)), (
        "the grammar admits `pizza`; the validator must too"
    )
    assert not schema_valid({"topping": "pizza"}, _norm(raw)), (
        "the un-expanded schema is what produced the false invalids — if this "
        "passes, the two schemas are no longer distinguishable and the guard "
        "above proves nothing"
    )


def test_case_expansion_reaches_enums_nested_under_items_and_anyof():
    """A partial recursion would silently leave some enums un-expanded, which
    reintroduces the bug on exactly the schemas nobody spot-checks.
    """
    from diffgemma_fa.eval.run import _case_insensitive_enums

    out = _case_insensitive_enums({
        "type": "dict",
        "properties": {
            "tags": {"type": "array",
                     "items": {"type": "string", "enum": ["RED"]}},
            "mode": {"anyOf": [{"type": "string", "enum": ["FAST"]}]},
        },
    })
    assert "red" in out["properties"]["tags"]["items"]["enum"]
    assert "fast" in out["properties"]["mode"]["anyOf"][0]["enum"]


def test_case_expansion_leaves_non_string_enums_and_structure_alone():
    """It must expand values, never types or required lists — and a property
    that happens to be *named* `enum` is a property, not an enum.
    """
    from diffgemma_fa.eval.run import _case_insensitive_enums

    src = {"type": "dict",
           "properties": {"n": {"type": "integer", "enum": [1, 2]},
                          "enum": {"type": "string"}},
           "required": ["n"]}
    out = _case_insensitive_enums(src)
    assert out["properties"]["n"]["enum"] == [1, 2]
    assert out["properties"]["enum"] == {"type": "string"}
    assert out["required"] == ["n"]


def test_case_expansion_cannot_change_which_answer_is_correct():
    """`run.py`'s justification for the flag: "BFCL's scorer lowercases and
    strips, so every variant scores identically — this cannot manufacture a
    wrong answer."

    Checked against BFCL's rule rather than against ours, so it stays true if
    `normalise` is corrected.
    """
    from diffgemma_fa.eval.run import _case_insensitive_enums

    for literal in ("PIZZA", "New York", "AIR_CLEAN", "u.s.a", "e-mail"):
        expanded = _case_insensitive_enums(
            {"enum": [literal]})["enum"]
        # Non-vacuity: if the expansion silently stopped happening, "every
        # variant is safe" would be true of a one-element list and prove
        # nothing.
        assert len(expanded) > 1, f"{literal!r} was not expanded at all"
        assert all(bfcl_standardize(v) == bfcl_standardize(literal)
                   for v in expanded), (literal, expanded)


# ==========================================================================
# Oracle 5 — extraction from a channel-tagged emission
# ==========================================================================


def test_extract_json_ignores_an_object_inside_the_channel_name():
    """The BFCL-side form of the header bug. The channel *name* is free text
    over the whole vocabulary — `artifacts/task_countdown_j0map.json` has names
    like `82-80=2` and `56-49=` — so it can contain a brace. Taking the first
    `{` in the raw emission would then score the header.
    """
    text = '<|channel>{"city": "WRONG"}\n<channel|>{"city": "Paris"}<turn|>'
    assert extract_json(text) == {"city": "Paris"}


def test_extract_json_on_the_real_unconstrained_shapes():
    """Verbatim from `artifacts/eval_bfcl_live_simple_unconstrained_sample.json`.
    The unconstrained arm emits a fenced, pretty-printed object, sometimes
    behind a bare `<channel|>` and sometimes behind nothing but newlines — and
    it is the arm every constrained number is compared against.
    """
    a = ('<channel|>```json\n{\n  "user_id": 7890,\n  "special": "black"\n}\n'
         '```<turn|>')
    b = ('\n\n```json\n{\n  "repos": [\n    "ShishirPatil/gorilla"\n  ],\n'
         '  "aligned": true\n}\n```<turn|>')
    assert extract_json(a) == {"user_id": 7890, "special": "black"}
    assert extract_json(b) == {"repos": ["ShishirPatil/gorilla"],
                              "aligned": True}


def test_extract_json_does_not_report_an_empty_object_as_content():
    """`{}` parses. It is not an answer, and the content columns are the only
    thing standing between CS = 1.000 and a run that measured nothing
    (SPEC §3.8).
    """
    s = Scores()
    s.add(accepted=True, parsed_obj=extract_json("<channel|>{}"),
          want={"a": "x"})
    d = s.as_dict()
    assert d["parsed"] == 1
    assert d["nonempty_rate"] == 0.0
    assert d["arg_accuracy"] == 0.0


# ==========================================================================
# Oracle 6 — the records that left the denominator, by name
# ==========================================================================
#
# `n` is "the records the model was actually asked", so a record whose grammar
# fails to compile leaves `n` and is reported in `skipped_by_reason`. That rule
# is right and is not what this section is about. What this section is about is
# that `skipped_by_reason` is a **histogram**:
#
#     skipped[k] = skipped.get(k, 0) + 1        # k = f"compile:{type(e).__name__}"
#
# so the artifact records *how many* records left and *what kind of exception*
# took them, and nothing whatsoever about **which**. Four lines away, the OOM
# and zero-partition handlers already append `{"id": ..., "fn": ...}` and the
# artifact ships `oom_records` / `zero_partition_records`. The skip path is the
# one failure mode whose records cannot be named.
#
# This is not hypothetical and it is the launch gate on the next arm. The build
# gate wired by [AUDIT-D3] legitimately refuses `live_simple_117-73-0` and
# `live_simple_122-78-0` (indices 117 and 122 of the 258 single-function
# records), so the next `bfcl_live_simple` run reports **n = 128** against the
# **n = 130** of every row in `docs/RESULTS.md`. Without the ids in the
# artifact, a reader holding the two files cannot compute the difference set,
# and therefore cannot say whether the new number is comparable to the old one
# — which is the entire purpose of `records_available`, `denominator_policy`
# and the coverage section of `docs/RESULTS.md`.
#
# What the artifact must satisfy, stated before any implementation exists:
#
#   R1  a reader with only the JSON can recover exactly WHICH records were
#       compile-skipped, and for each of them WHY;
#   R2  nothing else moves: the skipped record does not reach `Scores`, does
#       not appear in `rows`, and touches no accumulator that is published
#       under a field a reader compares — this is additive reporting, not a
#       denominator change;
#   R3  `skipped_by_reason` keeps its current reason->count shape, because it
#       has live consumers (below). The fix is an ADDITIONAL field;
#   R4  whatever is recorded survives `json.dump` — no sets, no exception
#       objects, no tuples-as-keys.
#
# HOW THESE ARE TESTED. Reaching the handler through `main()` needs a 51 GB
# checkpoint, and `import diffgemma_fa.eval.run` alone costs ~4.4 GB. So the
# handler is **replayed**: its own AST, lifted verbatim out of the shipped
# source and executed against fake records, with the accumulators initialised
# from the same `main()` and `Scores` replaced by a recorder. Nothing is
# re-implemented and nothing is read back from the implementation — the
# expectations below are R1-R4.
#
# TWO PROPERTIES KEEP THE REPLAY HONEST, because a replay that silently stops
# finding the handler would turn every guard here green:
#
#   * **the R3 interlock.** R3 asserts, against the *real* source, that the
#     replay produced `{"compile:ValueError": 2, "compile:KeyError": 1}`. A
#     zero-iteration or misdirected replay fails R3, so a green R3 is proof the
#     shipped handler was located and executed three times.
#   * **scope fidelity.** Only the names actually in scope at the handler are
#     seeded (`eval/run.py` has `fn`, `eval/run_tasks.py` does not — it has no
#     `fn` anywhere in the file). Seeding a name the harness cannot see would
#     bless a fix that raises `NameError` after the GPU time is spent.

_RUN_TASKS_PY = pathlib.Path(
    "/home/ubuntu/diffgemma_fa/diffgemma_fa/eval/run_tasks.py")
_PHASE5_REPORT = pathlib.Path("/home/ubuntu/diffgemma_fa/scripts/phase5_report.py")

#: The two harnesses compile a grammar per record through one of these.
_COMPILE_CALLS = ("compile_json_schema", "compile_regex")

#: Keys a per-record entry may use for the record identifier. Deliberately a
#: family rather than one spelling: the tester does not get to dictate the
#: coder's field names, only that the id and the reason are both there and are
#: attached to each other.
_ID_KEYS = ("id", "record_id", "rec_id")

#: The three fake failures every replay is driven with. Two share an exception
#: type on purpose — with only a histogram, `{"compile:ValueError": 2}` cannot
#: distinguish them, and that is precisely the information being lost. The ids
#: are the real ones the build gate refuses.
_FAKE_SKIPS = (
    ("live_simple_117-73-0", "get_movie_rating", ValueError("grammar rejects")),
    ("live_simple_122-78-0", "predict", ValueError("grammar rejects")),
    ("live_simple_9-9-9", "sink", KeyError("properties")),
)
_FAKE_IDS = {rid for rid, _, _ in _FAKE_SKIPS}

#: R2's oracle: the top-level fields of an artifact that has already been
#: **published**, i.e. the things a reader compares between two runs. An
#: accumulator reached by the skip handler and published under one of these has
#: moved a number, whatever the change was called. Transcribed from
#: `artifacts/eval_bfcl_live_simple_j0_map.json` and
#: `artifacts/task_countdown_j0map.json`; `test_the_published_field_oracle_is_a
#: _faithful_transcription` re-reads both so it cannot rot. Fields added *by*
#: the fix are deliberately absent — a new field is exactly what R3 asks for
#: and cannot invalidate an old comparison.
_RUN_PUBLISHED = frozenset({
    "arg_accuracy", "arg_correct", "arg_total", "cs", "cs_rate",
    "elapsed_seconds", "emission", "entropy_bound", "exact_call_rate",
    "exact_calls", "n", "nonempty", "nonempty_rate", "nonempty_strings",
    "parsed", "per_key", "records_available", "rows", "schema_ok",
    "schema_valid_rate", "seed", "skipped_by_reason", "task", "variant",
    "zero_partition", "zero_partition_records", "oom", "oom_records"})
_TASKS_PUBLISHED = frozenset({
    "confidence", "cs", "cs_rate", "elapsed_seconds", "emission",
    "entropy_bound", "failure_reasons", "n", "oom", "rows", "seed",
    "skipped_by_reason", "solve_rate", "solved", "task", "variant",
    "zero_partition", "parsed", "parse_rate"})

_HARNESSES = [
    pytest.param(_RUN_PY, _RUN_PUBLISHED, id="eval/run.py"),
    pytest.param(_RUN_TASKS_PY, _TASKS_PUBLISHED, id="eval/run_tasks.py"),
]


def test_the_published_field_oracle_is_a_faithful_transcription():
    """Guard on R2's oracle, so a stale field list cannot bless a moved number.

    Skips per artifact rather than failing when one is absent: `artifacts/` is
    only partly tracked, and a fresh clone may have neither.
    """
    seen = 0
    for path, transcribed in (
            ("artifacts/eval_bfcl_live_simple_j0_map.json", _RUN_PUBLISHED),
            ("artifacts/task_countdown_j0map.json", _TASKS_PUBLISHED)):
        p = pathlib.Path("/home/ubuntu/diffgemma_fa") / path
        if not p.exists():
            continue
        seen += 1
        shipped = set(json.load(open(p)))
        assert shipped <= set(transcribed), (
            f"{path} publishes {sorted(shipped - set(transcribed))}, which R2 "
            f"does not know about and would let the skip handler move")
    if not seen:
        pytest.skip("no published arm artifact on disk to check against")


class _RecordingScores:
    """Stands in for `metrics.Scores` and records rather than accumulates.

    R2: a compile-skipped record must never reach `Scores.add`. If it did it
    would enter `n`, `cs_rate`, `schema_valid_rate` and `exact_call_rate`, and
    the published `denominator_policy` would become false.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, **kw):
        self.calls.append(kw)

    def as_dict(self):
        return {}


class _AnyArgs:
    """`argparse` namespace stand-in: every flag exists and is falsy."""

    def __getattr__(self, name):
        return None


def _main_of(source: str) -> ast.FunctionDef:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("no `main()` in this harness")


def _called_names(nodes) -> set[str]:
    out = set()
    for n in nodes:
        for c in ast.walk(n):
            if isinstance(c, ast.Call):
                out.add(getattr(c.func, "attr", None)
                        or getattr(c.func, "id", ""))
    return out


def _compile_try(main_fn: ast.FunctionDef):
    """`(the record loop, the compile Try, its handler)`.

    Located by the grammar-compilation call in the `try` body rather than by
    line number or by the `"compile:"` string, so a rename of the reason key
    cannot make this stop finding the handler it is written about.
    """
    for stmt in main_fn.body:
        if not isinstance(stmt, ast.For):
            continue
        for node in stmt.body:
            if not isinstance(node, ast.Try):
                continue
            if _called_names(node.body) & set(_COMPILE_CALLS):
                assert len(node.handlers) == 1, (
                    "the compile `try` grew a second handler; this replay "
                    "assumes one")
                return stmt, node, node.handlers[0]
    raise AssertionError(
        f"no `for` loop with a `try` around {_COMPILE_CALLS} directly in "
        f"main(); the replay no longer models this harness and every guard "
        f"below would be vacuous")


def _out_dict_items(main_fn: ast.FunctionDef):
    """`[(key, value_node)]` of the artifact dict `out = {...}`."""
    for stmt in main_fn.body:
        if (isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "out"
                        for t in stmt.targets)
                and isinstance(stmt.value, ast.Dict)):
            return [(k.value, v) for k, v in zip(stmt.value.keys,
                                                 stmt.value.values)
                    if isinstance(k, ast.Constant)]
    raise AssertionError("main() no longer assembles an `out` dict")


def _assign_targets(stmt) -> list[str]:
    if isinstance(stmt, ast.AnnAssign):
        return [stmt.target.id] if isinstance(stmt.target, ast.Name) else []
    out = []
    for t in stmt.targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                out.append(n.id)
    return out


def _names_in_scope(main_fn, loop, try_stmt, handler) -> set[str]:
    """Every local name bound *before* control can reach the skip handler.

    Anything else the handler touches is a runtime `NameError` (a name that
    exists nowhere — `fn` in `eval/run_tasks.py`) or an `UnboundLocalError` (a
    name bound only on the success path — `want`, `state`, `a`). Both surface
    130 records into an 80-minute run, so the replay must not paper over them
    by seeding a generous namespace.
    """
    scope: set[str] = {handler.name} if handler.name else set()
    for stmt in main_fn.body:                      # before the record loop
        if stmt is loop:
            break
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            scope |= set(_assign_targets(stmt))
    scope |= {n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)}
    for stmt in loop.body:                         # before the compile `try`
        if stmt is try_stmt:
            break
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            scope |= set(_assign_targets(stmt))
    return scope


def _replay_compile_skips(source: str, failures=_FAKE_SKIPS):
    """Execute the harness's own compile-failure handler against fake records.

    Returns `(before, after, scores, main_fn)` where `before`/`after` are the
    accumulator namespace either side of the replay. The accumulators are the
    ones `main()` itself initialises before the record loop — executed from the
    shipped source, so a new list added by the fix is picked up without this
    helper being told about it.
    """
    main_fn = _main_of(source)
    loop, try_stmt, handler = _compile_try(main_fn)
    scope = _names_in_scope(main_fn, loop, try_stmt, handler)

    ns: dict = {"__builtins__": __builtins__}
    accum: list[str] = []
    for stmt in main_fn.body:
        if stmt is loop:
            break
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        mod = ast.Module(body=[stmt], type_ignores=[])
        ast.fix_missing_locations(mod)
        try:
            exec(compile(mod, "<harness-init>", "exec"), ns)   # noqa: S102
        except Exception:      # noqa: BLE001 - `sc = Scores()`, model loads...
            continue
        accum += [t for t in _assign_targets(stmt) if t in ns]

    scores = _RecordingScores()
    before = {k: copy.deepcopy(ns[k]) for k in accum}

    # `continue` is a syntax error outside a loop, so the handler body is
    # wrapped in a one-iteration `for`. Everything else is the shipped AST.
    wrapped = ast.Module(body=[ast.For(
        target=ast.Name(id="_replay_i", ctx=ast.Store()),
        iter=ast.List(elts=[ast.Constant(0)], ctx=ast.Load()),
        body=list(handler.body), orelse=[])], type_ignores=[])
    ast.fix_missing_locations(wrapped)
    code = compile(wrapped, "<compile-skip-handler>", "exec")

    for i, (rid, fname, exc) in enumerate(failures):
        fn = {"name": fname, "parameters": {"type": "dict", "properties": {}}}
        pool = {
            handler.name or "e": exc,
            "rec": types.SimpleNamespace(id=rid, functions=(fn,), split="s"),
            "fn": fn, "idx": i, "sc": scores, "args": _AnyArgs(),
            "prompt": "", "regex": "", "scorer": None,
        }
        # SCOPE FIDELITY: seed only what this harness can actually see, and
        # remove anything a previous iteration or the init sweep left behind.
        for name, value in pool.items():
            if name in scope:
                ns[name] = value
            else:
                ns.pop(name, None)
        try:
            exec(code, ns)                                     # noqa: S102
        except NameError as exc_:
            missing = getattr(exc_, "name", None) or str(exc_)
            if missing in scope:
                pytest.fail(
                    f"the compile-skip handler uses `{missing}`, which is in "
                    f"scope in the harness but not modelled by this replay. "
                    f"Add it to `pool` above.")
            pytest.fail(
                f"the compile-skip handler uses `{missing}`, which is NOT in "
                f"scope where it is used: the names available there are "
                f"{sorted(scope)}. This raises NameError on the first "
                f"compile failure — i.e. after the arm has been running for "
                f"an hour. (`eval/run_tasks.py` has no `fn`; its loop unpacks "
                f"`(rec, prompt, regex, scorer)`.)")

    after = {k: ns[k] for k in accum}
    return before, after, scores, main_fn


def reconstruct_skipped(published: dict) -> dict[str, set[str]]:
    """`{record_id: every string recorded beside it}`, from JSON alone.

    The oracle for R1, and deliberately tolerant about shape — a reader of the
    artifact does not care whether the fix ships
    `[{"id": ..., "reason": ...}, ...]` or `{reason: [id, ...]}`, only that
    both halves are there and are attached to each other. It is *not* tolerant
    about content: an id with nothing beside it comes back with an empty set
    and fails the "why" half of R1 rather than the "which" half, so the
    failure message points at what is actually missing.
    """
    out: dict[str, set[str]] = {}
    for value in published.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    out.setdefault(item, set())
                elif isinstance(item, dict):
                    ids = [v for k, v in item.items()
                           if k in _ID_KEYS and isinstance(v, str)]
                    if not ids:
                        continue
                    out.setdefault(ids[0], set()).update(
                        v for k, v in item.items()
                        if k not in _ID_KEYS and isinstance(v, str))
        elif isinstance(value, dict):
            for reason, ids in value.items():
                if not isinstance(ids, list):
                    continue
                for item in ids:
                    if isinstance(item, str):
                        out.setdefault(item, set()).add(str(reason))
                    elif isinstance(item, dict):
                        got = [v for k, v in item.items()
                               if k in _ID_KEYS and isinstance(v, str)]
                        if got:
                            out.setdefault(got[0], set()).add(str(reason))
    return out


def _publishes_whole(value: ast.expr, carriers: set[str]) -> set[str]:
    """The carriers this `out` value expression actually puts in the file.

    `"skipped_records": skipped_records` publishes them; `list(...)` and
    `sorted(...)` of one still do; `len(skipped_records)` mentions the name and
    publishes a **number**, from which no id can be recovered. The distinction
    is the whole of R1's third half, so it is made structurally rather than by
    "the name appears somewhere in the expression".
    """
    if isinstance(value, ast.Name):
        return {value.id} & carriers
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id in ("list", "sorted", "tuple") and value.args):
        return _publishes_whole(value.args[0], carriers)
    return set()


def _skip_report(source: str, published_fields: frozenset) -> dict:
    """Everything R1-R4 needs to be decided, for one harness source."""
    before, after, scores, main_fn = _replay_compile_skips(source)
    out_items = _out_dict_items(main_fn)

    # R4 first: a set or an exception object never reaches a reader at all, so
    # everything downstream is judged on what actually survives to the file.
    json_error = None
    survived: dict = {}
    try:
        survived = json.loads(json.dumps(after))
    except TypeError as exc:
        json_error = str(exc)

    # `rows` is the measured population: `scripts/phase5_report.py::per_record`
    # turns it into `{id: bool}` and pairs arms on the shared ids, so an entry
    # there is a scored record whatever it is labelled. Excluded from the
    # reconstruction, and checked separately by R2.
    rows_name = next((v.id for k, v in out_items
                      if k == "rows" and isinstance(v, ast.Name)), "rows")
    reconstructable = {k: v for k, v in survived.items() if k != rows_name}

    carriers = {name for name, value in reconstructable.items()
                if reconstruct_skipped({name: value})}
    published_carriers: set[str] = set()
    out_keys_of: dict[str, set[str]] = {}
    for key, value in out_items:
        published_carriers |= _publishes_whole(value, carriers)
        for node in ast.walk(value):
            if isinstance(node, ast.Name):
                out_keys_of.setdefault(node.id, set()).add(key)

    counts_name = next((v.id for k, v in out_items
                        if k == "skipped_by_reason" and isinstance(v, ast.Name)),
                       None)
    # EVERY accumulator that moved, not only the integer ones: `oom`,
    # `zero_partition` and `reasons` are lists and dicts, and they are
    # published as `len(oom)`, `len(zero_partition)` and `failure_reasons`.
    changed = {k: (before[k], after[k]) for k in after if before[k] != after[k]}
    return {
        "recovered": reconstruct_skipped(reconstructable),
        "json_error": json_error,
        "carriers": carriers,
        "published_carriers": published_carriers,
        "counts_name": counts_name,
        "counts": after.get(counts_name) if counts_name else None,
        "rows": after.get(rows_name),
        "scored": scores.calls,
        "changed": changed,
        "moved_published_fields": {
            name: sorted(out_keys_of.get(name, set()) & published_fields)
            for name in changed
            if name != counts_name
            and out_keys_of.get(name, set()) & published_fields},
    }


def _assert_r2(r: dict, label: str) -> None:
    """R2, as one reusable assertion so the adversarial cases can require it
    to *fire* rather than merely observing the mechanism."""
    assert r["scored"] == [], (
        f"{label}: the compile-skip handler called Scores.add{r['scored']} — "
        f"the record enters `n` and every rate over it, and the artifact's "
        f"denominator_policy stops being true")
    assert not r["rows"], (
        f"{label}: a compile-skipped record was appended to `rows`: "
        f"{r['rows']}. `rows` is the scored population — phase5_report pairs "
        f"arms on the ids it finds there.")
    assert not r["moved_published_fields"], (
        f"{label}: the skip handler moved accumulators that are published "
        f"under fields a reader compares: {r['moved_published_fields']}. "
        f"Values now {({k: v[1] for k, v in r['changed'].items() if k in r['moved_published_fields']})}. "
        f"This commit is supposed to be additive reporting.")


# --------------------------------------------------------------------------
# R1/R2/R3/R4 against the shipped harnesses
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path,published", _HARNESSES)
def test_a_compile_skipped_record_can_be_named_from_the_artifact(path, published):
    """R1, the "which" half — and the launch gate.

    Three records leave the run, two of them for the same reason. A histogram
    says `{"compile:ValueError": 2, "compile:KeyError": 1}`, from which the
    difference set against a published `n = 130` cannot be computed: you know
    three records are missing and not one of their ids. The OOM handler eleven
    lines below already does this correctly (`oom_records`), so the shape is
    not in question.
    """
    r = _skip_report(path.read_text(), published)
    assert r["json_error"] is None, (
        f"{path.name}: the compile-skip bookkeeping does not survive "
        f"json.dump: {r['json_error']}")
    assert set(r["recovered"]) == _FAKE_IDS, (
        f"{path.name}: replayed three compile failures and the artifact can "
        f"name {sorted(r['recovered']) or 'none'} of them.\n"
        f"  counts kept: {r['counts']}\n"
        f"A reader holding this artifact and a published n=130 row cannot "
        f"compute the difference set, so the two n's are not comparable.")


@pytest.mark.parametrize("path,published", _HARNESSES)
def test_each_named_record_carries_the_reason_it_was_skipped(path, published):
    """R1, the "why" half.

    An id list alone is only enough while exactly one exception type occurred;
    the moment two do — and `compile:ValueError` from the build gate will not
    be the only one forever — the reader is back to guessing. The reason must
    be attached to the record, not merely present in the same file.
    """
    r = _skip_report(path.read_text(), published)
    unnamed = _FAKE_IDS - set(r["recovered"])
    missing = {rid: sorted(r["recovered"].get(rid, ()))
               for rid, _fname, exc in _FAKE_SKIPS
               if rid not in unnamed
               and not any(type(exc).__name__ in s
                           for s in r["recovered"][rid])}
    assert not unnamed, (
        f"{path.name}: {sorted(unnamed)} are not named at all, so the `why` "
        f"half of R1 cannot even be asked — see the test above")
    assert not missing, (
        f"{path.name}: these records are named without saying why they left: "
        f"{missing}. `skipped_by_reason` has the reasons and the record list "
        f"has the ids, and nothing joins them.")


@pytest.mark.parametrize("path,published", _HARNESSES)
def test_the_skipped_records_reach_the_artifact_and_not_just_a_local(
        path, published):
    """R1, third half: recorded is not reported.

    `oom_records` and `zero_partition_records` are values of the `out` dict. A
    list that is appended to and never written out is a variable, not an
    artifact — and a list published as `len(...)` is a fourth count, which is
    the failure this test is most likely to have to catch.
    """
    r = _skip_report(path.read_text(), published)
    assert r["carriers"], "nothing carries the ids; see the R1 test above"
    assert r["published_carriers"], (
        f"{path.name}: {sorted(r['carriers'])} carries the skipped record ids "
        f"and no `out` value **is** that list, so no id reaches the artifact "
        f"file. (A `len(...)` of it does not count: it is another number.)")


@pytest.mark.parametrize("path,published", _HARNESSES)
def test_reporting_a_skip_does_not_move_a_single_measured_number(path, published):
    """R2. Additive reporting only.

    Passes today, and must still pass afterwards — it is the whole reason this
    fix is safe to land before an arm. A compile-skipped record was never put
    to the model, so it must not reach `Scores` (which would put it in `n`,
    `cs_rate` and `exact_call_rate`), must not reach `rows` (which
    `phase5_report.per_record` turns into the paired McNemar population, where
    a fabricated `accepted=False` becomes a discordant pair against every arm
    measured before the fix), and must not move any other accumulator that is
    published — `oom`, `zero_partition` and `failure_reasons` are lists and
    dicts, and are just as published as `n`.
    """
    _assert_r2(_skip_report(path.read_text(), published), path.name)


@pytest.mark.parametrize("path,published", _HARNESSES)
def test_skipped_by_reason_stays_a_reason_to_count_histogram(path, published):
    """R3, and the interlock that makes the R1 guards non-vacuous.

    `skipped_by_reason` has two live consumers that this test exists to
    protect, both checked by `test_the_consumers_that_make_skipped_by_reason_
    load_bearing_still_exist` below:

      * `eval/run_tasks.py` computes the published denominator from it —
        `n = len(items) - sum(skipped.values())`. Values that are not numbers
        make `n` a `TypeError` at best and silently wrong at worst;
      * `scripts/phase5_report.py` prints it verbatim into the coverage
        section of `docs/RESULTS.md` ("skipped at compile time: none"), which
        is a published string.

    So the histogram is not free to be repurposed into `{reason: [ids]}`, and
    a fix that does so is not additive however good the ids look.

    It doubles as the anti-vacuity interlock: the counts asserted here are
    produced by executing the **shipped** handler three times, so a replay that
    quietly stopped finding or running it turns this red rather than turning
    the R1 guards green.
    """
    r = _skip_report(path.read_text(), published)
    assert r["counts_name"], (
        f"{path.name}: `skipped_by_reason` is no longer a plain name in the "
        f"`out` dict; its two consumers are pinned to a reason->count mapping")
    counts = r["counts"]
    assert isinstance(counts, dict) and counts, "skipped_by_reason is empty"
    assert all(isinstance(v, int) and not isinstance(v, bool)
               for v in counts.values()), (
        f"{path.name}: skipped_by_reason became {counts!r}; "
        f"`n = len(items) - sum(skipped.values())` needs counts")
    assert sum(counts.values()) == len(_FAKE_SKIPS), (
        f"{path.name}: three records were skipped and the histogram totals "
        f"{sum(counts.values())} — `n` is now wrong by "
        f"{len(_FAKE_SKIPS) - sum(counts.values())}")
    assert set(counts) == {"compile:ValueError", "compile:KeyError"}, (
        f"{path.name}: reason keys are {sorted(counts)}")
    # The two fields must agree, or the reader has to pick one to believe.
    assert sum(counts.values()) == len(r["recovered"]) or not r["recovered"], (
        f"{path.name}: skipped_by_reason totals {sum(counts.values())} but "
        f"{len(r['recovered'])} records are named; one of them is wrong")


def test_the_consumers_that_make_skipped_by_reason_load_bearing_still_exist():
    """The evidence for R3, kept falsifiable.

    R3 is an empirical claim about two other files, and a claim like that rots.
    The published `n` must still be *derived from* the histogram — checked by
    expanding `n`'s expression one level through the local assignments, so
    `n = len(items) - n_skipped` counts only while `n_skipped` is itself
    `sum(skipped.values())`. Rewriting it to `len(skipped)` (which counts
    reasons, not records) fails here, as it should: the compatibility argument
    would then be about a different expression.
    """
    tasks_src = _RUN_TASKS_PY.read_text()
    assigns: dict[str, str] = {}
    for node in ast.walk(ast.parse(tasks_src)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            assigns[node.targets[0].id] = ast.unparse(node.value)
    assert "n" in assigns, "eval/run_tasks.py no longer assigns `n`"
    expanded = assigns["n"]
    for name, expr in assigns.items():
        if name != "n":
            expanded = re.sub(rf"\b{re.escape(name)}\b", f"({expr})", expanded)
    assert "sum(" in expanded and "values()" in expanded, (
        f"eval/run_tasks.py's `n` no longer sums the skipped histogram — it "
        f"expands to `{expanded}`. Re-derive R3 before changing "
        f"skipped_by_reason's shape.")

    assert _PHASE5_REPORT.exists()
    assert "skipped_by_reason" in _PHASE5_REPORT.read_text(), (
        "scripts/phase5_report.py no longer reads skipped_by_reason; it is "
        "what writes `skipped at compile time: ...` into docs/RESULTS.md")


# --------------------------------------------------------------------------
# Self-tests: the replay must be able to tell right from wrong
# --------------------------------------------------------------------------
#
# The section above is a guard over files it does not control. Its R1 tests are
# kept honest by the R3 interlock; these synthetic harnesses cover the rest —
# one wrong in the way the shipped code is wrong today, two right in different
# shapes, and one for each way the fix is most likely to be wrong.

#: `eval/run.py`'s shape: `fn` is bound in the loop before the `try`.
#: `eval/run_tasks.py`'s: the loop unpacks four names and there is no `fn`
#: anywhere in the file.
_LOOP_BFCL = ("    for idx, rec in enumerate(records):\n"
              "        fn = rec.functions[0]\n"
              "        try:\n"
              "            a = pipeline.compile_json_schema("
              "fn[\"parameters\"]).automaton")
_LOOP_TASKS = ("    for idx, (rec, prompt, regex, scorer) in enumerate(items):\n"
               "        try:\n"
               "            a = pipeline.compile_regex(regex).automaton")


def _harness(init: str = "", handler: str = "", out: str = "",
             loop: str = _LOOP_BFCL) -> str:
    """A minimal `main()` with the two harnesses' shape, including the
    accumulators R2 cares about and the `out` keys they are published under."""
    return f'''
def main():
    sc = metrics.Scores()
    rows, skipped, zero_partition, oom = [], {{}}, [], []
    reasons = {{}}
    n_ok = n_cs = n_parsed = 0
{init}
{loop}
        except Exception as e:
            k = f"compile:{{type(e).__name__}}"
{handler}
            continue
        sc.add(accepted=True, parsed_obj=None, want=None)
    out = {{"skipped_by_reason": skipped, "rows": rows, "oom": len(oom),
           "zero_partition": len(zero_partition), "failure_reasons": reasons,
           "solved": n_ok, {out}}}
'''


#: The synthetic `out` keys that stand for "already published".
_SYNTHETIC_PUBLISHED = frozenset({
    "skipped_by_reason", "rows", "oom", "zero_partition", "failure_reasons",
    "solved"})

#: What the code does today: a histogram and nothing else.
_SKIP_HISTOGRAM_ONLY = _harness(
    handler="            skipped[k] = skipped.get(k, 0) + 1")

#: One correct fix, in the shape of the OOM handler four lines away.
_SKIP_FIXED = _harness(
    init="    skipped_records = []",
    handler="            skipped[k] = skipped.get(k, 0) + 1\n"
            "            skipped_records.append({'id': rec.id, "
            "'fn': fn.get('name'),\n"
            "                                    'reason': k, "
            "'detail': str(e)[:200]})",
    out='"skipped_records": skipped_records')

#: A second correct fix in a different shape, so the tests above are not
#: secretly demanding one field layout.
_SKIP_FIXED_BY_REASON = _harness(
    init="    skipped_ids = {}",
    handler="            skipped[k] = skipped.get(k, 0) + 1\n"
            "            skipped_ids.setdefault(k, []).append(rec.id)",
    out='"skipped_record_ids": skipped_ids')


@pytest.mark.parametrize("source", [_SKIP_FIXED, _SKIP_FIXED_BY_REASON],
                         ids=["per-record-list", "reason-to-id-lists"])
def test_the_replay_passes_a_correct_fix(source):
    """Satisfiability. A checker nothing can satisfy is not a specification.

    Both layouts recover all three ids with their reasons, publish them whole,
    keep the histogram, and move no published field.
    """
    r = _skip_report(source, _SYNTHETIC_PUBLISHED)
    assert r["json_error"] is None
    assert set(r["recovered"]) == _FAKE_IDS
    assert all(any("ValueError" in s for s in r["recovered"][rid])
               for rid, _f, e in _FAKE_SKIPS if isinstance(e, ValueError))
    assert r["published_carriers"]
    assert r["counts"] == {"compile:ValueError": 2, "compile:KeyError": 1}
    _assert_r2(r, "correct fix")


def test_the_replay_detects_todays_histogram_only_handler():
    """Mutation self-test, on the synthetic pre-fix shape.

    Complements the R3 interlock: this shows the machinery reports "nothing
    recoverable" for a handler that keeps only a count, while R3 on the real
    files shows the handler being executed is the shipped one.
    """
    r = _skip_report(_SKIP_HISTOGRAM_ONLY, _SYNTHETIC_PUBLISHED)
    assert r["counts"] == {"compile:ValueError": 2, "compile:KeyError": 1}, (
        "the replay did not execute the handler; every guard above is vacuous")
    assert r["recovered"] == {}, (
        "the pre-fix harness records only a histogram, so nothing should be "
        f"recoverable — got {r['recovered']}")
    assert r["carriers"] == set()


# --- the adversarial cases: plausible fixes that are still wrong -----------

def test_a_handler_that_reaches_for_a_name_not_in_its_scope_is_rejected():
    """The `fn` hole, which is specific to `eval/run_tasks.py`.

    That file has no `fn` anywhere: its loop unpacks
    `(rec, prompt, regex, scorer)`. Copying `eval/run.py`'s
    `{"id": rec.id, "fn": fn.get("name")}` across — the obvious way to keep
    the two artifacts consistent, and the reviewer's first instinct — compiles,
    reads correctly, and raises `NameError` on the first compile failure, an
    hour into an 80-minute arm. Nothing else in this file would notice, so the
    replay refuses to seed names the harness cannot see.
    """
    src = _harness(
        init="    skipped_records = []",
        handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                "            skipped_records.append({'id': rec.id, "
                "'fn': fn.get('name'), 'reason': k})",
        out='"skipped_records": skipped_records',
        loop=_LOOP_TASKS)
    with pytest.raises(pytest.fail.Exception, match="NOT in scope"):
        _skip_report(src, _SYNTHETIC_PUBLISHED)

    # ...and the same handler under `eval/run.py`'s loop, where `fn` IS bound,
    # is fine. Without this the test would pass on a replay that rejected
    # everything.
    ok = _skip_report(src.replace(_LOOP_TASKS, _LOOP_BFCL),
                      _SYNTHETIC_PUBLISHED)
    assert set(ok["recovered"]) == _FAKE_IDS


def test_a_handler_that_reaches_for_a_success_path_name_is_rejected():
    """The same hole through the other door: `want` and `state` exist in
    `eval/run.py`, but are bound *after* the compile `try`. Referencing one
    from the skip handler is an `UnboundLocalError` at run time, and a replay
    that seeded every name in the file would bless it.
    """
    src = _harness(
        init="    skipped_records = []",
        handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                "            skipped_records.append({'id': rec.id, "
                "'reason': k, 'want': str(want)})",
        out='"skipped_records": skipped_records')
    with pytest.raises(pytest.fail.Exception, match="NOT in scope"):
        _skip_report(src, _SYNTHETIC_PUBLISHED)


def test_an_id_without_a_reason_is_rejected():
    """Most likely near-miss: `skipped_ids.append(rec.id)`.

    It answers "which" and not "why", and it reads as complete. With one
    exception type in play it even *is* complete, which is how it survives
    review — and the build gate's `ValueError` will not be the only one for
    long. The "which" test must pass and the "why" test must fail, so the
    failure message points at the missing half.
    """
    src = _harness(init="    skipped_ids = []",
                   handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                           "            skipped_ids.append(rec.id)",
                   out='"skipped_ids": skipped_ids')
    r = _skip_report(src, _SYNTHETIC_PUBLISHED)
    assert set(r["recovered"]) == _FAKE_IDS, "the ids are there"
    assert all(r["recovered"][rid] == set() for rid in _FAKE_IDS), (
        "and no reason is attached to any of them — if this passes, the "
        "`why` guard cannot fail and is decorative")


def test_a_set_of_ids_is_rejected_because_it_never_reaches_the_file():
    """R4. `skipped_ids = set()` is the natural type for "which records",
    and `json.dump` raises `TypeError: Object of type set is not JSON
    serializable` — at the END of the run, after the GPU time is spent.
    """
    src = _harness(init="    skipped_ids = set()",
                   handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                           "            skipped_ids.add(rec.id)",
                   out='"skipped_ids": skipped_ids')
    r = _skip_report(src, _SYNTHETIC_PUBLISHED)
    assert r["json_error"] is not None, (
        "a set survived json.dumps; the R4 guard cannot fail")
    assert r["recovered"] == {}


def test_recording_the_exception_object_is_rejected():
    """R4, the other way: `{"id": rec.id, "error": e}` is the obvious thing to
    write and is not serialisable either. Same failure, same timing.
    """
    src = _harness(init="    skipped_records = []",
                   handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                           "            skipped_records.append({'id': rec.id, "
                           "'error': e})",
                   out='"skipped_records": skipped_records')
    assert _skip_report(src, _SYNTHETIC_PUBLISHED)["json_error"] is not None


def test_publishing_only_a_count_of_the_skipped_records_is_rejected():
    """R1's third half, sharpened: `"n_skipped_records": len(skipped_records)`.

    The list is built correctly and the artifact gets a fourth number instead
    of the ids — which is the defect this whole section is about, reintroduced
    by the fix for it.
    """
    r = _skip_report(_harness(
        init="    skipped_records = []",
        handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                "            skipped_records.append({'id': rec.id, "
                "'reason': k})",
        out='"n_skipped_records": len(skipped_records)'),
        _SYNTHETIC_PUBLISHED)
    assert r["carriers"], "the list itself is fine"
    assert not r["published_carriers"], (
        "a `len(...)` was accepted as publishing the records")


def test_a_fix_that_forgets_to_publish_is_rejected():
    """R1's third half: appended, never written out."""
    r = _skip_report(_harness(
        init="    skipped_records = []",
        handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                "            skipped_records.append({'id': rec.id, "
                "'reason': k})"), _SYNTHETIC_PUBLISHED)
    assert set(r["recovered"]) == _FAKE_IDS
    assert r["carriers"] and not r["published_carriers"], (
        "the unpublished-list case is not being detected")


def test_a_skipped_record_smuggled_into_rows_is_rejected():
    """R2's sharp edge, and the most dangerous plausible fix.

    "Put it in `rows` with `skipped: True`" names the record, survives JSON,
    and keeps `n` intact — and `scripts/phase5_report.py::per_record` builds
    `{id: bool}` from `rows` and pairs arms on the shared ids, so the record
    reappears as `accepted=False` in every McNemar comparison against an arm
    that measured it. That manufactures discordant pairs out of a record
    nobody ran. R1 is satisfied and R2 must refuse it.
    """
    src = _harness(handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                           "            rows.append({'id': rec.id, "
                           "'accepted': False,\n"
                           "                         'reason': k, "
                           "'skipped': True})")
    r = _skip_report(src, _SYNTHETIC_PUBLISHED)
    assert set(reconstruct_skipped({"rows": r["rows"]})) == _FAKE_IDS, (
        "it does name the records, which is exactly why it is tempting")
    with pytest.raises(AssertionError, match="appended to `rows`"):
        _assert_r2(r, "rows-smuggling fix")


def test_scoring_the_skipped_record_as_a_failure_is_rejected():
    """R2's other edge. `sc.add(..., want=want)` for a compile-skipped record
    looks like [AUDIT-B2]'s fix and is the opposite of it: B2's records
    *reached the model and failed*, these were never asked. It moves `n` from
    128 to 130 and drops `cs_rate` — a denominator change wearing a bug fix's
    clothes, on the same commit that was supposed to be additive.
    """
    src = _harness(init="    skipped_records = []",
                   handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                           "            skipped_records.append({'id': rec.id, "
                           "'reason': k})\n"
                           "            sc.add(accepted=False, "
                           "parsed_obj=None, want={'a': 'x'}, schema_ok=False)",
                   out='"skipped_records": skipped_records')
    with pytest.raises(AssertionError, match="called Scores.add"):
        _assert_r2(_skip_report(src, _SYNTHETIC_PUBLISHED), "scoring fix")


@pytest.mark.parametrize("acc,field,mutation", [
    ("oom", "oom", "            oom.append(rec.id)"),
    ("zero_partition", "zero_partition",
     "            zero_partition.append(rec.id)"),
    ("reasons", "failure_reasons",
     "            reasons[k] = reasons.get(k, 0) + 1"),
    ("n_ok", "solved", "            n_ok += 0 * len(rec.id) + 1"),
])
def test_charging_the_skip_to_another_published_column_is_rejected(
        acc, field, mutation):
    """R2 over the accumulators that are not integers.

    `oom` and `zero_partition` are lists published as `len(...)`; `reasons` is
    a dict published as `failure_reasons`. Filing a compile skip under any of
    them moves a column in `docs/RESULTS.md` while `n` stays put — and a
    reviewer reading the diff sees a record being recorded, which is what the
    commit is *supposed* to do. R2 must fire on every one of them, not only on
    the `int` counters.
    """
    src = _harness(init="    skipped_records = []",
                   handler="            skipped[k] = skipped.get(k, 0) + 1\n"
                           "            skipped_records.append({'id': rec.id, "
                           "'reason': k})\n" + mutation,
                   out='"skipped_records": skipped_records')
    r = _skip_report(src, _SYNTHETIC_PUBLISHED)
    assert set(r["recovered"]) == _FAKE_IDS, "R1 is satisfied by this shape"
    assert acc in r["changed"], f"{acc} was not touched; the case is not live"
    with pytest.raises(AssertionError, match="published"):
        _assert_r2(r, f"{acc}-charging fix")
    assert r["moved_published_fields"] == {acc: [field]}


def test_repurposing_the_histogram_into_id_lists_is_rejected():
    """R3. The tidiest-looking fix of all: one field instead of two.

    `skipped_by_reason = {"compile:ValueError": [id, id]}` answers R1
    perfectly. It also feeds `sum(skipped.values())` two lists, so
    `eval/run_tasks.py`'s `n` raises `TypeError`, and it prints record ids into
    the `docs/RESULTS.md` coverage line. R1 must pass here and R3 must fail.
    """
    r = _skip_report(
        _harness(handler="            skipped.setdefault(k, []).append(rec.id)"),
        _SYNTHETIC_PUBLISHED)
    assert set(r["recovered"]) == _FAKE_IDS, "R1 is satisfied by this shape"
    assert not all(isinstance(v, int) for v in r["counts"].values()), (
        "the histogram survived; the R3 guard cannot fail")
    with pytest.raises(TypeError):
        sum(r["counts"].values())          # what run_tasks.py does to it


# --------------------------------------------------------------------------
# The records this is actually about
# --------------------------------------------------------------------------

def test_the_next_arm_really_does_skip_the_two_records_the_audit_named():
    """Grounding. R1 is worth a commit only if the skip path is live.

    `docs/RESULTS.md` and `docs/PHASE1_FINDINGS.md` name
    `live_simple_117-73-0` and `live_simple_122-78-0` (a BFCL `any`-typed
    property, whose unparenthesised top-level alternation can never close its
    brace) as the two records the [AUDIT-D3] build gate refuses. This asserts
    that they are still at indices 117 and 122 of the 258 single-function
    records, that the shipped compile call raises, and that the reason key the
    handler will build for them is `compile:ValueError` — so the artifact the
    next arm writes must name exactly these two.

    The control record is not decoration: if the gate refused everything, the
    two ids above would be trivia rather than the difference set.
    """
    from diffgemma_fa.compile import bfcl_data, schema as _schema
    from diffgemma_fa.eval.run import ALLOW, WHITESPACE_PATTERNS

    split = pathlib.Path(bfcl_data.DATA_DIR) / "BFCL_v4_live_simple.json"
    if not split.exists():
        pytest.skip("BFCL checkout not present (artifacts/data is gitignored)")

    records = [r for r in bfcl_data.iter_split("BFCL_v4_live_simple.json")
               if len(r.functions) == 1]
    assert len(records) == 258, f"the cut changed shape: {len(records)}"

    def compile_as_the_arm_does(rec):
        fn = rec.functions[0]
        norm = _schema.normalize_bfcl_schema(fn["parameters"])
        return pipeline.compile_json_schema(
            fn["parameters"], name=fn.get("name", ""), from_bfcl=True,
            allow=ALLOW, allow_wildcard=True,
            whitespace_pattern=WHITESPACE_PATTERNS["json"], fence=False,
            verify_renderings=_schema.synthesize_instance(norm),
            verify_strict=True, nonempty_required_strings=True)

    refused = {}
    for idx, want_id in ((117, "live_simple_117-73-0"),
                         (122, "live_simple_122-78-0")):
        assert records[idx].id == want_id, (
            f"index {idx} is {records[idx].id}, not {want_id}; the "
            f"difference set against the n=130 rows has moved")
        with pytest.raises(ValueError) as exc:
            compile_as_the_arm_does(records[idx])
        refused[want_id] = f"compile:{type(exc.value).__name__}"

    assert set(refused.values()) == {"compile:ValueError"}, refused
    # Non-vacuity: the gate is selective, so `n = 128` and not `n = 0`.
    assert compile_as_the_arm_does(records[3]).automaton is not None, (
        "a control record no longer compiles; the gate is refusing more than "
        "the two records this test is about and the arm is not runnable")
