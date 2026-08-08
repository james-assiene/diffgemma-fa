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

import pathlib
import re

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
