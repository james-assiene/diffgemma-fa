"""AUDIT SUITE — the grammar compiler and the guarantee.

Written by the *tester* agent of CLAUDE.md's tester -> coder -> reviewer loop,
from SPEC §3.1b / §3.5 / §4.2–§4.5 and from the failure record, **not** from the
implementation. It duplicates a little of `test_schema.py`, `test_automaton.py`
and `test_minimize.py` on purpose: those tests were written alongside the code
they check, and the failures this file targets all survived a green suite.

The failure family, in the project's own words:

* the compiled grammar accepted **0 of 130** outputs the unconstrained model
  actually produced, because it forbade the model's newline+indent separators.
  It did not fail loudly: the renormalised draw EXTENDS a value when its
  preferred separator is inadmissible, so `600` silently became `6000`.
  30 accuracy points, invisible for weeks.
* `normalize_bfcl_schema` DELETED any parameter literally named
  `description`/`default`/`optional`.
* `minimize()` silently changed the language on a duplicate transition.
* `--variant=j2` was never implemented and silently ran J0's emission.

Every one of those is the same shape: **plausible code, no independent
expectation**. So this file asserts the expectation directly.

The central property, stated once:

    A compiled grammar must accept what the model would naturally write.

It is asserted *structurally* — against the format's own definition (RFC 8259
whitespace, JSON's five standard renderings) — never by fitting observed
outputs. Fitting the model's habits is what produced the hand-tuned pattern
that still missed tabs.

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_audit_grammar.py -q
"""

from __future__ import annotations

import ast
import collections
import itertools
import json
import pathlib
import random
import re

import numpy as np
import pytest

from diffgemma_fa.compile import pipeline
from diffgemma_fa.compile import schema as S
from diffgemma_fa.compile.automaton import (
    INF_DISTANCE,
    CompiledAutomaton,
    _assert_structural_invariants,
    compile_automaton,
    wrap_with_fence,
)
from diffgemma_fa.compile.classes import build_tables
from diffgemma_fa.compile.minimize import Dfa, minimize
from diffgemma_fa.compile.validate import Simulator
from diffgemma_fa.compile.vocab import END_TOKENS, PAD_TOKEN, gemma_tokenizer

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The five standard renderings of a JSON value. Every one of them is a
#: `json.dumps` default somewhere in the wild, and RFC 8259 §2 makes all five
#: the *same* value: whitespace between structural tokens is insignificant.
#: A grammar that admits fewer than five is narrower than the format.
RENDERINGS = {
    "compact": dict(separators=(",", ":")),
    "spaced": {},
    "indent2": dict(indent=2),
    "indent4": dict(indent=4),
    "tabs": dict(indent="\t"),
}


def render(instance) -> dict[str, str]:
    return {name: json.dumps(instance, **kw) for name, kw in RENDERINGS.items()}


def accepted_renderings(regex: str, instance) -> set[str]:
    return {n for n, text in render(instance).items() if re.fullmatch(regex, text)}


# --------------------------------------------------------------------------
# Schemas. Deliberately BFCL-shaped: BFCL v4 carries 959 `array` and 9,464
# `dict` occurrences, so neither an array property nor a nested object is an
# exotic case to test against.
# --------------------------------------------------------------------------

FLAT_SCALAR = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["name", "count"],
}
FLAT_SCALAR_INSTANCE = {"name": "ab", "count": 12}

WITH_ARRAY = {
    "type": "object",
    "properties": {"name": {"type": "string"},
                   "tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["name", "tags"],
}
WITH_ARRAY_INSTANCE = {"name": "ab", "tags": ["x", "y"]}

NESTED = {
    "type": "object",
    "properties": {"cfg": {
        "type": "object",
        "properties": {"host": {"type": "string"}},
        "required": ["host"]}},
    "required": ["cfg"],
}
NESTED_INSTANCE = {"cfg": {"host": "h"}}


@pytest.fixture(scope="module")
def flat_automaton():
    """One real end-to-end compile. ~7 s (tokenizer + 262k-piece vocabulary)
    plus ~3 s of lift/minimize/interning, so the whole file uses exactly one.
    """
    return pipeline.compile_json_schema(FLAT_SCALAR, name="audit_flat").automaton


@pytest.fixture(scope="module")
def flat_sim(flat_automaton):
    return Simulator(flat_automaton)


# ==========================================================================
# A. THE CENTRAL PROPERTY
#    A compiled grammar must accept what the model would naturally write.
# ==========================================================================

@pytest.mark.parametrize("name,schema,instance", [
    ("flat-scalars", FLAT_SCALAR, FLAT_SCALAR_INSTANCE),
    ("with-array", WITH_ARRAY, WITH_ARRAY_INSTANCE),
    ("nested-object", NESTED, NESTED_INSTANCE),
])
def test_the_default_whitespace_admits_all_five_renderings_of_every_shape(
        name, schema, instance):
    """`schema.JSON_WS` must accept every standard rendering — of an object
    with an **array**, and of a **nested** object, not only of a flat one.

    This is the fixed bug's own test, generalised past the shape it was fixed
    on. `JSON_WS` is documented as "the whitespace pattern to use for JSON, not
    a tuning knob", and it was measured 5/5 on a flat 3-key schema. The measured
    property must hold for the shapes the benchmark actually contains: pretty-
    printing indents by nesting depth, so a bound on the whitespace *run* is a
    bound on nesting depth, and the schema does not know about it.

    A miss here is not a crash. It is `600` -> `6000` at the value boundary.
    """
    regex = S.build_regex(schema, whitespace_pattern=S.JSON_WS)
    ok = accepted_renderings(regex, instance)
    assert ok == set(RENDERINGS), (
        f"{name}: grammar rejects {sorted(set(RENDERINGS) - ok)}; a rendering "
        f"the model may choose is inadmissible, and the renormalised draw "
        f"silently extends the value rather than failing"
    )


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_the_whitespace_bound_is_not_secretly_a_nesting_depth_bound(depth):
    """Same property, isolated to the one variable that drives it.

    `JSON_WS = [ \\t\\n\\r]{0,8}` bounds a whitespace *run* at 8 characters. A
    pretty-printer emits `newline + indent*depth`, so the bound binds at
    `indent*depth >= 8`. Nothing in the compiler relates that constant to the
    schema's nesting depth, so if it binds it binds silently.
    """
    schema = {"type": "string"}
    instance = "x"
    for i in range(depth):
        schema = {"type": "object", "properties": {f"k{i}": schema},
                  "required": [f"k{i}"]}
        instance = {f"k{i}": instance}
    regex = S.build_regex(schema, whitespace_pattern=S.JSON_WS)
    ok = accepted_renderings(regex, instance)
    assert ok == set(RENDERINGS), (
        f"at nesting depth {depth} the grammar rejects "
        f"{sorted(set(RENDERINGS) - ok)}"
    )


def _cli_whitespace_arg() -> dict[str, object]:
    """The `--whitespace` argparse keywords `eval/run.py` declares, by AST.

    Read rather than hardcoded so a new choice — or a changed default — cannot
    be introduced without this test seeing it.
    """
    tree = ast.parse((REPO / "diffgemma_fa/eval/run.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and getattr(node.args[0], "value", None) == "--whitespace"):
            return {kw.arg: ast.literal_eval(kw.value)
                    for kw in node.keywords
                    if kw.arg in ("choices", "default")}
    raise AssertionError("no --whitespace argument found in eval/run.py")


def _cli_whitespace_choices() -> list[str]:
    return list(_cli_whitespace_arg()["choices"])


def test_the_whitespace_setting_the_eval_cli_defaults_to_accepts_all_renderings():
    """**The fix must be reachable from the thing that runs the experiment,
    and it must be what you get when you do not ask.**

    AMENDED BY THE REVIEWER, 2026-08-08. Justification, recorded because a
    reviewer editing a tester's assertion is exactly the move that must never
    be silent:

    The original form hardcoded `{"stock": None, "pretty": PRETTY_WS}` and
    demanded 5/5 from **both**. That is unsatisfiable against two assertions
    this project deliberately keeps, and which are the right ones:

      * `test_schema.py::test_json_ws_accepts_every_standard_rendering` pins
        `whitespace_pattern=None` (outlines' own) at 2/5 as the negative
        control — "the gate must fail on the default pattern, or it is not
        testing anything";
      * `test_the_shipped_pretty_pattern_does_not_survive_the_gate`, in this
        very file, pins `PRETTY_WS` at 4/5 (it misses tabs) because a gate that
        cannot catch a *near* miss is not a gate.

    So no implementation can make `stock` and `pretty` accept 5/5 while those
    hold: the demand was not "the CLI must offer a correct grammar" but "the
    CLI must not be able to name a historical one" — and naming them is the
    only way to reproduce the arms already in `docs/RESULTS.md`.

    The satisfiable statement of the same property, which is what the D2 defect
    actually was, is asserted instead, and it is **stronger** in the direction
    that matters:

      1. the mapping is read from the shipped `WHITESPACE_PATTERNS`, so a new
         choice cannot be added unmodelled (the original guard, kept);
      2. the CLI **default** — what every arm gets that does not opt out —
         accepts all five renderings, on a flat schema, on one with an array,
         and on a nested one;
      3. every non-default choice is one of the declared-historical patterns;
      4. and none of them is silently selectable: each still fails the strict
         build gate, so choosing one is a deliberate, loud act.

    (3) and (4) are what stop this from degrading into "the default is fine,
    who cares about the rest" — the D2 defect was precisely that the *default*
    was the broken one.
    """
    from diffgemma_fa.eval.run import WHITESPACE_PATTERNS

    arg = _cli_whitespace_arg()
    choices, default = set(arg["choices"]), arg["default"]
    assert choices == set(WHITESPACE_PATTERNS), (
        f"eval/run.py's --whitespace choices and its WHITESPACE_PATTERNS map "
        f"disagree: {sorted(choices ^ set(WHITESPACE_PATTERNS))}"
    )
    assert default in WHITESPACE_PATTERNS, f"unknown default {default!r}"

    # 1. The default is the repaired grammar, at every shape — not only on the
    #    flat schema the original 5/5 claim was measured on.
    #
    #    `deep` is not decoration. D1 was a *nesting-depth* bound wearing a
    #    whitespace-run bound's clothes, and flat/array/nested all sit at depth
    #    <= 2. A future default of `[ \t\n\r]{0,24}` would pass the first three
    #    shapes and still fail at depth 6, which is the same bug with a bigger
    #    constant. `test_the_whitespace_bound_is_not_secretly_a_nesting_depth_
    #    bound` pins `S.JSON_WS` directly; this pins whatever the CLI hands out,
    #    which is not required to be the same object.
    deep_schema, deep_instance = {"type": "string"}, "x"
    for i in range(5):
        deep_schema = {"type": "object", "properties": {f"k{i}": deep_schema},
                       "required": [f"k{i}"]}
        deep_instance = {f"k{i}": deep_instance}

    bad = {}
    for name, schema, instance in [("flat", FLAT_SCALAR, FLAT_SCALAR_INSTANCE),
                                   ("array", WITH_ARRAY, WITH_ARRAY_INSTANCE),
                                   ("nested", NESTED, NESTED_INSTANCE),
                                   ("deep(5)", deep_schema, deep_instance)]:
        regex = S.build_regex(schema, whitespace_pattern=WHITESPACE_PATTERNS[default])
        missing = set(RENDERINGS) - accepted_renderings(regex, instance)
        if missing:
            bad[name] = sorted(missing)
    assert not bad, (
        f"--whitespace={default} is the CLI default and rejects {bad}. An arm "
        f"run without an explicit --whitespace would carry the 0/130 defect."
    )

    # 2. Everything else is a declared-historical pattern, kept only so the
    #    arms already in docs/RESULTS.md can be reproduced.
    assert choices - {default} <= {"stock", "pretty"}, (
        f"undeclared non-default --whitespace choice(s): "
        f"{sorted(choices - {default} - {'stock', 'pretty'})}"
    )

    # 3. ...and each of them still FAILS the strict build gate, so it cannot be
    #    selected by accident. If one ever stops failing, either it was widened
    #    (fine — promote it) or the gate stopped checking (not fine).
    for choice in sorted(choices - {default}):
        with pytest.raises(ValueError, match="rejects"):
            pipeline.compile_json_schema(
                FLAT_SCALAR, name=f"audit_ws_{choice}",
                whitespace_pattern=WHITESPACE_PATTERNS[choice],
                verify_renderings=FLAT_SCALAR_INSTANCE,
                do_minimize=False, channel_header=False)


def test_the_compiled_token_automaton_accepts_the_models_own_renderings(
        flat_automaton, flat_sim):
    """The property at the level that actually matters: **tokens**.

    Regex acceptance is not the claim. The lift (SPEC §4.3) drops
    `RESERVED_TOKENS`, the channel header (§3.6) is prepended as literal token
    ids, and the stop augmentation (§3.5) is bolted on afterwards. Any of those
    can narrow the language without touching the regex.

    The sequence built here is exactly what Phase 0 measured the released model
    emitting: `[100, <name>, 107, 101]`, the JSON, then a stop token.
    """
    tok = gemma_tokenizer()
    header = [100, *tok.encode("thought"), 107, 101]
    rejected = []
    for name, text in render(FLAT_SCALAR_INSTANCE).items():
        tokens = header + [int(t) for t in tok.encode(text)] + [END_TOKENS[0]]
        if not flat_sim.accepts(tokens):
            rejected.append(name)
    assert not rejected, (
        f"the compiled automaton rejects the {rejected} rendering(s) of an "
        f"instance its schema permits"
    )


def test_the_grammar_is_not_narrower_than_the_schema_on_scalar_values():
    """Over-constraint in the *value* grammar, checked against the schema.

    Every instance below is legal under the schema, so a grammar narrower than
    the schema shows up as a rejection. The negative case at the end is the
    control: a test that only asserts acceptance passes on `.*`.
    """
    schema = {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "f": {"type": "number"},
            "b": {"type": "boolean"},
            "e": {"type": "string", "enum": ["fast", "slow"]},
        },
        "required": ["s", "i", "f", "b", "e"],
    }
    regex = S.build_regex(schema, whitespace_pattern=S.JSON_WS)
    legal = [
        {"s": "a b", "i": 0, "f": 0.0, "b": True, "e": "fast"},
        {"s": "", "i": -3, "f": -0.25, "b": False, "e": "slow"},
        # An integer literal is a legal JSON `number`; a model writing `3` for a
        # float-typed argument must not be forced off its plan.
        {"s": "x", "i": 1000, "f": 3, "b": True, "e": "fast"},
    ]
    for inst in legal:
        compact = json.dumps(inst, separators=(",", ":"))
        assert re.fullmatch(regex, compact), (
            f"grammar is narrower than its schema: rejects {compact}")
    illegal = json.dumps({"s": "x", "i": 1, "f": 1, "b": True, "e": "nope"},
                         separators=(",", ":"))
    assert not re.fullmatch(regex, illegal), (
        "grammar accepts a non-enum value — the acceptance assertions above "
        "would then be vacuous")


# ==========================================================================
# B. WHERE THE COMPILER CHOOSES TO BE STRICTER THAN THE SCHEMA,
#    THE OVER-CONSTRAINT MUST BE DETECTABLE.
# ==========================================================================

def test_require_nonempty_strings_is_detectable_at_build_time():
    """`--nonempty` is a *deliberate* over-constraint (schema.py says so). A
    deliberate over-constraint is fine; a silent one is not.

    The detectable signal is the build gate: compiling with the flag and an
    instance the **schema** permits must fail, naming the loss. Two records of
    BFCL-Live become unanswerable under it, and the results table is required to
    say so — which is only possible if something reports it.
    """
    empty_string_instance = {"name": "", "count": 1}

    # Without the flag the schema permits it and the gate is silent.
    pipeline.compile_json_schema(
        FLAT_SCALAR, name="audit_ok",
        verify_renderings=empty_string_instance,
        # stop before the expensive lift: the gate runs first, and if it does
        # not fire we want the failure to be "no raise", not a 10 s compile.
        do_minimize=False, channel_header=False)

    with pytest.raises(ValueError, match="rejects"):
        pipeline.compile_json_schema(
            FLAT_SCALAR, name="audit_nonempty",
            nonempty_required_strings=True,
            verify_renderings=empty_string_instance,
            do_minimize=False, channel_header=False)


def test_the_shipped_pretty_pattern_does_not_survive_the_gate():
    """The gate must fail the hand-fitted pattern, not just the empty one.

    `PRETTY_WS` was reverse-engineered from 130 observed outputs and accepts
    4/5 renderings — it misses tabs. That is precisely the near-miss the gate
    exists to catch, and it is a stronger test of the gate than
    `whitespace_pattern=""`, which fails 4 of 5 and would be caught by anything.
    """
    from diffgemma_fa.eval.run import PRETTY_WS

    ok, bad = S.accepts_all_renderings(
        S.build_regex(FLAT_SCALAR, whitespace_pattern=PRETTY_WS),
        FLAT_SCALAR_INSTANCE)
    assert not ok and "tabs" in bad, (
        "PRETTY_WS was measured to miss the tab rendering; if it no longer "
        "does, the checker stopped checking")

    with pytest.raises(ValueError, match="rejects"):
        pipeline.compile_json_schema(
            FLAT_SCALAR, name="audit_pretty", whitespace_pattern=PRETTY_WS,
            verify_renderings=FLAT_SCALAR_INSTANCE,
            do_minimize=False, channel_header=False)


# --- verify_strict: what a "declared narrow" policy may and may not excuse ---
#
# The gate exists to make an over-constraint loud. A flag that turns it off is
# therefore the one place where the original defect can be re-introduced *with
# a code comment saying it is fine*, so the flag needs its own boundary, and
# the boundary has to come from the format rather than from the flag's name.
#
# It does. `json.dumps(separators=(",", ":"))` emits **no whitespace between
# structural tokens at all**, so the compact rendering is admitted by every
# whitespace policy that admits anything — measured, including the narrowest
# one expressible:
#
#     whitespace_pattern=""      compact accepted, other four rejected
#     outlines' default          compact accepted, three rejected
#     PRETTY_WS                  compact accepted, tabs rejected
#
# So `compact` in the rejected set is a *proof* that the failure is not about
# whitespace: it is the grammar and the schema disagreeing about structure or
# values. A whitespace flag has no standing to excuse it.

_TYPE_MISMATCH_INSTANCE = {"name": "ab", "count": "12"}


@pytest.mark.parametrize("case,kwargs,instance", [
    # A compiler-side over-constraint of VALUES, not of whitespace.
    # `require_nonempty_strings` is documented as deliberate and opt-in, and its
    # cost is two BFCL-Live records that become unanswerable — a fact the
    # results table is required to state, which it can only do if something
    # reports it. `verify_strict` is documented for "a whitespace policy that is
    # declared narrow"; this is not one.
    ("value over-constraint", dict(nonempty_required_strings=True),
     {"name": "", "count": 1}),
    # The grammar and the schema disagree about a value's TYPE, so the grammar
    # does not accept an instance of its own schema in any rendering. This is
    # the shape of the 11 BFCL-Live schemas `synthesize_instance` exposed, of
    # which `live_simple_117-73-0` (a grammar accepting a bare `null` as a
    # complete answer) is the reported example: the gate is the only thing in
    # the build that can see it, and it compiled clean under
    # `--whitespace pretty`.
    ("structural mismatch", {}, _TYPE_MISMATCH_INSTANCE),
])
@pytest.mark.parametrize("strict", [True, False])
def test_a_non_whitespace_gate_failure_raises_whatever_verify_strict_says(
        case, kwargs, instance, strict):
    """**`verify_strict=False` may narrow the gate to whitespace. It may not
    turn the gate off.**

    Both cases below have the *compact* rendering in the rejected set, which no
    whitespace policy can explain (see the module comment above). Downgrading
    them to a printed line is not "a declared-narrow whitespace policy" — it is
    the build proceeding with a grammar that provably rejects an instance of
    its own schema, which is precisely the over-constraint `validate.py` calls
    the far more dangerous direction because the output stays well-formed and
    can never score.
    """
    # Mirror what `compile_json_schema` will build, so the precondition below
    # is about the same regex the gate will see.
    prepared = (S.require_nonempty_strings(FLAT_SCALAR)
                if kwargs.get("nonempty_required_strings") else FLAT_SCALAR)
    _ok, rejected = S.accepts_all_renderings(
        S.build_regex(prepared, whitespace_pattern=S.JSON_WS), instance)
    assert "compact" in rejected, (
        f"{case}: fixture no longer rejects the whitespace-free rendering, so "
        f"this test would no longer be about non-whitespace failures")

    with pytest.raises(ValueError, match="rejects"):
        pipeline.compile_json_schema(
            FLAT_SCALAR, name=f"audit_strict_{strict}",
            verify_renderings=instance, verify_strict=strict,
            do_minimize=False, channel_header=False, **kwargs)


def test_verify_strict_false_still_downgrades_a_whitespace_only_failure(capsys):
    """The other half, without which the test above is satisfiable by ignoring
    `verify_strict` entirely.

    A *whitespace-only* failure — compact accepted, a pretty rendering rejected
    — is what the flag is for: `eval/run.py` keeps `--whitespace=stock|pretty`
    so the arms already in `docs/RESULTS.md` can be reproduced, and those
    grammars are narrow **by declaration**. That must compile.

    It must also be **reported**. CLAUDE.md: never silently cap coverage. A
    downgrade that leaves no trace in the log is indistinguishable from a gate
    that passed, and the whole point of the flag is that the narrowing was a
    choice someone can later find.
    """
    from diffgemma_fa.eval.run import PRETTY_WS

    _ok, rejected = S.accepts_all_renderings(
        S.build_regex(FLAT_SCALAR, whitespace_pattern=PRETTY_WS),
        FLAT_SCALAR_INSTANCE)
    assert "compact" not in rejected and "tabs" in rejected, (
        "fixture must be a whitespace-ONLY failure for this to test the "
        "downgrade rather than the raise")

    pipeline.compile_json_schema(
        FLAT_SCALAR, name="audit_declared_narrow",
        whitespace_pattern=PRETTY_WS, verify_renderings=FLAT_SCALAR_INSTANCE,
        verify_strict=False, do_minimize=False, channel_header=False)

    out = capsys.readouterr().out
    assert "audit_declared_narrow" in out and "tabs" in out, (
        f"the downgrade left no usable trace; got {out!r}")


def _compile_json_schema_callsites() -> list[tuple[str, int, dict[str, object]]]:
    """Every `compile_json_schema(...)` call under `diffgemma_fa/`, by AST.

    Returns the keyword names mapped to a crude description of the argument,
    enough to tell "absent" from "present but None".
    """
    out = []
    for path in sorted((REPO / "diffgemma_fa").rglob("*.py")):
        if path.name == "pipeline.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if nm == "compile_json_schema":
                    kws = {}
                    for k in node.keywords:
                        if k.arg is None:
                            continue
                        kws[k.arg] = ("literal-None"
                                      if isinstance(k.value, ast.Constant)
                                      and k.value.value is None
                                      else "given")
                    out.append((str(path.relative_to(REPO)), node.lineno, kws))
    return out


def test_the_production_compile_path_uses_the_build_gate():
    """**An unwired gate is not a mitigation.**

    `accepts_all_renderings` + `verify_renderings` is documented as the thing
    that "turns 'the grammar must accept how the model writes' from advice into
    a build failure". It costs milliseconds and needs no model. It is only
    reached when a caller passes an instance, so a caller that does not pass one
    is back to the pre-fix behaviour with the fix sitting unused beside it —
    which is exactly how the original defect survived: the round-trip check
    existed and was run on the wrong split.
    """
    sites = _compile_json_schema_callsites()
    assert sites, "no call sites found — this test has gone vacuous"
    # `verify_renderings=None` is the same as not passing it, so presence of the
    # keyword is not enough: the gate has to be handed something to check.
    ungated = [(p, ln) for p, ln, kws in sites
               if kws.get("verify_renderings") in (None, "literal-None")]
    assert not ungated, (
        f"compile_json_schema call sites that skip the rendering gate: "
        f"{ungated}. Every production grammar should be proved to accept a "
        f"rendering of an instance its schema permits before it is used."
    )


# ==========================================================================
# C. COMPILER INVARIANTS THAT ARE ONLY ESTABLISHED AT BUILD TIME
# ==========================================================================

def _class_members(a: CompiledAutomaton) -> list[np.ndarray]:
    """Decode each interned class back to a `[V] bool` membership mask."""
    t = a.tables
    out = []
    for c in range(t.n_classes):
        m = np.zeros(t.vocab_size, dtype=bool)
        m[t.sum_indices[t.sum_indptr[c]:t.sum_indptr[c + 1]]] = True
        out.append(~m if t.sum_is_neg[c] else m)
    return out


def test_a_freshly_compiled_automaton_satisfies_the_load_time_invariants(
        flat_automaton):
    """`_assert_structural_invariants` proves unit `(src, dst)` multiplicity on
    the way in from disk. Nothing proves it on the way *out* of the compiler.

    `_group_edges` establishes it by construction, and a construction argument
    is exactly what stopped being true in every failure this file targets.
    """
    _assert_structural_invariants(flat_automaton, "freshly compiled")
    pairs = list(zip(flat_automaton.edge_src.tolist(),
                     flat_automaton.edge_dst.tolist()))
    assert len(pairs) == len(set(pairs)), (
        "duplicate (src, dst): eq (8)'s multiplicity stops being 0/1 and the "
        "sampling DISTRIBUTION changes, which no acceptance check can see")


def test_is_dfa_true_implies_pairwise_disjoint_outgoing_label_sets(
        flat_automaton):
    """`is_dfa` honesty, checked on **tokens** rather than on class ids.

    SPEC §2.6: eq (8)'s cheap `∃` token draw is valid only on a DFA; on an NFA
    it is off by ~1.7e-2 against the exact posterior. Determinism is the
    statement that no token leaves a state by two edges — a statement about the
    label *sets*, not about the interned class ids they happen to carry.
    """
    a = flat_automaton
    if not a.is_dfa:
        pytest.skip("automaton is an NFA; nothing to over-claim")
    members = _class_members(a)
    out = collections.defaultdict(list)
    for e in range(a.n_edges):
        out[int(a.edge_src[e])].append((int(a.edge_dst[e]), int(a.edge_class[e])))
    for src, edges in out.items():
        for (d1, c1), (d2, c2) in itertools.combinations(edges, 2):
            overlap = int((members[c1] & members[c2]).sum())
            assert overlap == 0, (
                f"is_dfa=True but state {src} sends {overlap} token(s) to both "
                f"{d1} and {d2}")


def test_the_is_dfa_check_catches_overlap_across_distinct_classes():
    """The adversarial case for the same invariant.

    `_assert_structural_invariants` keys its determinism check on
    `(src, edge_class)`, so it can only see a conflict when the two edges were
    interned to the *same* class. Two edges out of one state whose label sets
    overlap but are not identical get different class ids — and a stale or
    hand-made artifact claiming `is_dfa=True` then loads clean, silently
    licensing eq (8)'s `∃` fast path on an NFA.

    The same-class case is asserted first as the control, so a failure here is
    unambiguously the cross-class hole and not a broken fixture.

    **All three storage polarities are exercised, and that is not padding.**
    `classes.py` stores a class as its complement whenever `|S_c| > V/2`, and
    measured on real BFCL every large class is complement-stored (`|N_c|` in
    the 988–1,064 range against a 262,144-token vocabulary). A fix that
    compares the *stored* index lists would pass a pos x pos test and be wrong
    on every grammar this project actually compiles: `~{1,2}` and `~{1,3}`
    share 12 of 16 tokens while their stored lists share none.
    """
    V, bucket = 16, 16
    ALL = frozenset(range(V))

    def artifact(labels, class_id, is_dfa):
        tables = build_tables(labels, vocab_size=V)
        return CompiledAutomaton(
            name="adversarial", n_states=3, n_states_bucket=bucket,
            n_edges=2, vocab_size=V, is_dfa=is_dfa,
            edge_src=np.array([0, 0], np.int32),
            edge_dst=np.array([1, 2], np.int32),
            edge_class=np.asarray(class_id, np.int32),
            d=np.zeros(bucket, np.int32), is_final=np.zeros(bucket, bool),
            start_vector=np.zeros(bucket, bool),
            tables=tables, schema_hash="", tokenizer_hash="",
            compiler_version="audit")

    # Control: identical labels intern to ONE class, which the current check
    # does see. If this stops raising, the fixture broke, not the code.
    same = [frozenset({3, 4}), frozenset({3, 4})]
    with pytest.raises(ValueError, match="two destinations"):
        _assert_structural_invariants(artifact(same, [0, 0], True))

    # The hole, at each polarity. `build_tables` negates a class iff
    # `|S_c| > V // 2 == 8`, so the sizes below pick the branch deliberately.
    cases = {
        # pos x pos: both stored as members. |S| = 2 each.
        "pos x pos": [frozenset({3, 4}), frozenset({4, 5})],
        # neg x neg: both stored as COMPLEMENTS. |S| = 14 each, overlap 12,
        # stored lists {1,2} and {1,3} — which share a token too, so a fix that
        # compares stored lists gets the right answer here for the wrong
        # reason; the pos x neg case below is what separates them.
        "neg x neg": [ALL - {1, 2}, ALL - {1, 3}],
        # pos x neg: mixed storage, and the stored lists are DISJOINT
        # ({5,6} vs {1,2}) while the true sets overlap on exactly {5,6}. Any
        # fix that forgets to decode polarity reports "no conflict" here.
        "pos x neg": [frozenset({5, 6}), ALL - {1, 2}],
    }
    for name, labels in cases.items():
        assert labels[0] & labels[1], f"{name}: fixture labels do not overlap"
        with pytest.raises(ValueError):
            _assert_structural_invariants(artifact(labels, [0, 1], True))

    # ...and the invariant must not fire when the sets are genuinely disjoint,
    # or the "fix" is just `raise` and every real DFA stops loading.
    disjoint = [frozenset({5, 6}), ALL - {1, 2, 5, 6}]
    assert not (disjoint[0] & disjoint[1])
    _assert_structural_invariants(artifact(disjoint, [0, 1], True))


def test_d_agrees_with_an_independent_forward_bfs(flat_automaton):
    """`b_L(s) = 1[d(s) <= R]` is the whole budget closure (SPEC §3.1b), so a
    `d` that is wrong by one silently either forbids a completable string or
    admits an uncompletable one.

    `distance_to_final` is a reverse BFS from `F`. This is a forward BFS from
    each state — a different algorithm over the same edge list, which is what
    makes it a check rather than a restatement.
    """
    a = flat_automaton
    adj = collections.defaultdict(list)
    for e in range(a.n_edges):
        adj[int(a.edge_src[e])].append(int(a.edge_dst[e]))
    finals = set(np.nonzero(np.asarray(a.is_final))[0].tolist())

    def forward_distance(s: int):
        seen, frontier, depth = {s}, [s], 0
        while frontier:
            if any(q in finals for q in frontier):
                return depth
            nxt = []
            for q in frontier:
                for r in adj[q]:
                    if r not in seen:
                        seen.add(r)
                        nxt.append(r)
            frontier, depth = nxt, depth + 1
        return None

    d = np.asarray(a.d)
    for s in range(a.n_states):
        got = int(d[s])
        want = forward_distance(s)
        want = INF_DISTANCE if want is None else want
        assert got == want, f"d({s}) = {got}, forward BFS says {want}"

    assert (d[a.n_states:] == INF_DISTANCE).all(), (
        "bucket padding must be an absorbing dead state (SPEC §5.5)")


def test_d_is_computed_after_the_stop_augmentation(flat_automaton):
    """SPEC §3.1b: `d` must be measured on the augmented automaton, "or it is
    wrong". The signature of getting the order backwards is visible without
    recompiling: only `ACC` may be at distance 0.

    Computed before augmentation, every grammar-final state would read 0 and
    `ACC` would be unreachable; `b_L` would then let a canvas terminate one
    token before its stop token.
    """
    a = flat_automaton
    zeros = np.nonzero(np.asarray(a.d)[:a.n_states] == 0)[0]
    finals = np.nonzero(np.asarray(a.is_final))[0]
    assert zeros.tolist() == finals.tolist() == [a.n_states - 1], (
        f"d == 0 at {zeros.tolist()}, is_final at {finals.tolist()}; "
        f"exactly ACC = {a.n_states - 1} is expected")

    acc = a.n_states - 1
    into_acc = {int(a.edge_src[e]) for e in range(a.n_edges)
                if int(a.edge_dst[e]) == acc and int(a.edge_src[e]) != acc}
    assert into_acc, "no grammar-final state has a stop-token edge into ACC"
    for s in into_acc:
        assert int(a.d[s]) == 1, (
            f"state {s} reaches ACC in one token but d({s}) = {int(a.d[s])}")


def test_the_post_stop_tail_spans_the_full_vocabulary(flat_automaton):
    """`ACC --Σ--> ACC`, over the **whole** vocabulary (SPEC §3.5 trap 4).

    Pinned to PAD instead, a joint decode pays `(255-j)·log p(PAD)` to
    terminate at position `j` and puts the stop token at 255 or never. Checked
    on the real 262,144-token vocabulary rather than on a toy, because the
    construction loops `range(vocab_size)` and a toy cannot distinguish
    "the full alphabet" from "the alphabet that happened to occur".
    """
    a = flat_automaton
    acc = a.n_states - 1
    loops = [e for e in range(a.n_edges)
             if int(a.edge_src[e]) == acc and int(a.edge_dst[e]) == acc]
    assert len(loops) == 1, f"expected one ACC self-loop, found {len(loops)}"
    cls = int(a.edge_class[loops[0]])
    assert int(a.tables.class_size[cls]) == a.vocab_size, (
        f"the unscored tail carries {int(a.tables.class_size[cls])} of "
        f"{a.vocab_size} tokens; its emission mass is then < 1 and stopping is "
        f"no longer free")
    assert _class_members(a)[cls].all(), "tail membership is not all of Σ"

    assert [int(a.edge_dst[e]) for e in range(a.n_edges)
            if int(a.edge_src[e]) == acc] == [acc], (
        "ACC must be absorbing")


def test_no_sigma_self_loop_anywhere_except_the_unscored_tail(flat_automaton):
    """A Σ self-loop has emission mass 1 — staying in it is free, so a joint
    MAP sits there for the whole canvas rather than pay a token's probability
    to leave. That is a language-changing bug that looks like a decode bug, and
    it has already bitten this project once (the channel-header name loop:
    "the MAP consumed all 64 positions inside the name loop and never closed
    the header").

    Exactly one such loop is legitimate: the post-stop tail.

    A near-Σ self-loop is also asserted about, more weakly but for a concrete
    reason: an unbounded JSON string body is one, and it must at least be
    unable to emit a stop token or PAD, or the canvas gets truncated mid-string
    and `A_{k+1}` goes empty (§3.1b failure mode 3).
    """
    a = flat_automaton
    acc = a.n_states - 1
    members = _class_members(a)
    reserved = set(END_TOKENS) | {PAD_TOKEN}
    for e in range(a.n_edges):
        src, dst = int(a.edge_src[e]), int(a.edge_dst[e])
        if src != dst:
            continue
        size = int(a.tables.class_size[int(a.edge_class[e])])
        if src == acc:
            continue
        assert size < a.vocab_size, (
            f"state {src} has a Σ self-loop outside the unscored tail — a "
            f"free attractor for a joint decode")
        emitted = {v for v in reserved if members[int(a.edge_class[e])][v]}
        assert not emitted, (
            f"self-loop at {src} can emit {sorted(emitted)}; the canvas would "
            f"be truncated inside the grammar")


# ==========================================================================
# D. REGRESSIONS
# ==========================================================================

def test_a_parameter_named_like_a_dropped_keyword_survives_to_the_grammar():
    """Regression: the `properties`-MAP keyword drop.

    `normalize_bfcl_schema` dropped `description`/`default`/`optional` as
    keywords, and applied that to the *keys* of the `properties` map — deleting
    41 real BFCL parameters, 13 of them `required`. The resulting grammar
    **rejects** the required argument and **accepts** its omission: an
    over-constraint, which `validate.py` calls the far more dangerous
    direction, because the output stays well-formed and can never score.

    Asserted at the grammar level, not just on the dict, because the dict
    surviving is not the property that was lost.
    """
    bfcl = {
        "type": "dict",
        "properties": {
            "description": {"type": "String", "description": "prose"},
            "default": {"type": "integer", "default": 3},
            "optional": {"type": "boolean", "optional": True},
        },
        "required": ["description", "default", "optional"],
    }
    norm = S.normalize_bfcl_schema(bfcl)
    assert set(norm["properties"]) == {"description", "default", "optional"}
    # ...and the annotation of the same name, one level down, is still dropped.
    assert norm["properties"]["description"] == {"type": "string"}
    assert norm["properties"]["default"] == {"type": "integer"}

    regex = S.build_regex(bfcl, from_bfcl=True, whitespace_pattern=S.JSON_WS)
    full = json.dumps({"description": "d", "default": 1, "optional": True},
                      separators=(",", ":"))
    assert re.fullmatch(regex, full), (
        "the grammar rejects a required parameter named `description`")
    without = json.dumps({"default": 1, "optional": True}, separators=(",", ":"))
    assert not re.fullmatch(regex, without), (
        "the grammar accepts the omission of a required parameter — the "
        "acceptance assertion above would then prove nothing")


def test_name_keyed_maps_are_respected_at_every_nesting_level():
    """The asymmetry that hid the bug was that `check_supported` recursed
    correctly into name-keyed maps while `normalize_bfcl_schema` did not. Both
    must treat `properties`, `$defs` and `definitions` as maps of *names* at
    every depth, not only at the root.
    """
    bfcl = {
        "type": "dict",
        "$defs": {"default": {"type": "String"}},
        "properties": {
            # a parameter named like a keyword, whose OWN properties contain
            # another one.
            "optional": {
                "type": "dict",
                "properties": {"description": {"type": "float"}},
                "required": ["description"],
            },
            # a parameter named like a structural keyword.
            "type": {"type": "String"},
        },
        "required": ["optional", "type"],
    }
    norm = S.normalize_bfcl_schema(bfcl)
    assert set(norm["properties"]) == {"optional", "type"}
    assert norm["$defs"] == {"default": {"type": "string"}}
    inner = norm["properties"]["optional"]["properties"]
    assert set(inner) == {"description"}
    assert inner["description"]["type"] == "number", (
        "a nested name-keyed value was not normalised — the BFCL type "
        "vocabulary would leak into outlines")
    assert norm["properties"]["type"]["type"] == "string"


# --- minimize --------------------------------------------------------------

def _language(dfa: Dfa, alphabet, max_len: int) -> set[tuple[int, ...]]:
    """Every accepted word up to `max_len`, by explicit set simulation.

    A set simulation, so it is also correct on the NFAs the raise-cases build.
    """
    delta = collections.defaultdict(set)
    for s, a, d in dfa.transitions:
        delta[(s, a)].add(d)
    out = set()
    if dfa.is_empty:
        return out
    for n in range(max_len + 1):
        for word in itertools.product(alphabet, repeat=n):
            cur = {dfa.start}
            for sym in word:
                cur = set().union(*(delta[(s, sym)] for s in cur)) if cur else set()
                if not cur:
                    break
            if cur & dfa.finals:
                out.add(word)
    return out


@pytest.mark.parametrize("seed", range(12))
def test_minimize_preserves_the_language_on_random_partial_dfas(seed: int):
    """The property, not the state count. A minimizer that returns *some*
    smaller automaton is worthless if the language moved.

    Partial DFAs specifically: SPEC §4.5 forbids completing with a sink state
    (`m = 170k` becomes `5.24e9`), so every automaton this project minimizes has
    missing `(state, label)` pairs, and that is the case the textbook algorithm
    is least often tested on.
    """
    rng = random.Random(seed)
    n, alphabet = 6, (0, 1, 2)
    trans = []
    for s in range(n):
        for a in alphabet:
            if rng.random() < 0.7:
                trans.append((s, a, rng.randrange(n)))
    # At least one final, drawn from the states reachable at depth <= 2, so the
    # language is never trivially empty — half the seeds were, and a test that
    # compares two empty sets passes on any minimizer at all.
    finals = frozenset(rng.sample(range(1, n), rng.randint(1, 3)) + [
        d for s, a, d in trans if s == 0][:1])
    src = Dfa(n_states=n, transitions=tuple(trans), start=0, finals=finals)

    out = minimize(src).dfa
    before = _language(src, alphabet, 5)
    after = _language(out, alphabet, 5) if not out.is_empty else set()
    assert before, "degenerate fixture: the source language is empty"
    assert before == after, (
        f"minimize changed the language: only-before={sorted(before - after)[:4]} "
        f"only-after={sorted(after - before)[:4]}")


def test_minimize_raises_on_a_duplicate_and_on_a_nondeterministic_pair():
    """Valmari's `mark()` has no re-mark guard, so marking one element twice
    pushes `marked[s]` past the set size and corrupts the partition. The port is
    faithful; the *precondition* was unchecked, and an exact duplicate silently
    CHANGED THE LANGUAGE.

    Both inputs are realistic: concatenating two transition tuples is exactly
    how SPEC §3.8's `FA_grammar | FA_refusal` union would be built.
    """
    duplicate = Dfa(n_states=3,
                    transitions=((0, 0, 1), (0, 0, 1), (1, 1, 2), (0, 1, 2)),
                    start=0, finals=frozenset({2}))
    with pytest.raises(ValueError, match="duplicate"):
        minimize(duplicate)
    # ...and the reason it must raise: the language would otherwise move.
    assert _language(duplicate, (0, 1), 4) == {(1,), (0, 1)}

    nfa = Dfa(n_states=3, transitions=((0, 0, 1), (0, 0, 2)), start=0,
              finals=frozenset({1}))
    with pytest.raises(ValueError, match="nondeterministic"):
        minimize(nfa)


def test_the_fence_wrap_reports_the_nondeterminism_it_can_introduce():
    """`wrap_with_fence` copies the grammar start's outgoing edges onto a new
    state 0 that also carries the fence's first token. When the grammar itself
    can start with that token the two collide, and the result is an NFA.

    Two things must then be true, and neither is automatic:
      * `compile_automaton` must report `is_dfa=False` — over-claiming would put
        eq (8) on its `∃` fast path, wrong by ~1.7e-2 (SPEC §2.6);
      * `minimize` must **raise** rather than quietly minimize an NFA.
    """
    collide = 2717  # the fence's first token, "```"
    grammar = Dfa(n_states=2, transitions=((0, collide, 1),), start=0,
                  finals=frozenset({1}))
    wrapped = wrap_with_fence(grammar)

    starts = [(s, l, d) for s, l, d in wrapped.transitions
              if s == wrapped.start and l == collide]
    assert len(starts) > 1, "fixture no longer produces the collision"

    with pytest.raises(ValueError):
        minimize(wrapped)

    a = compile_automaton(wrapped, name="fenced", end_tokens=(1,),
                          vocab_size=4096)
    assert not a.is_dfa, (
        "the fence wrap made the automaton nondeterministic and the artifact "
        "still claims is_dfa=True")


def test_the_fence_keeps_both_the_fenced_and_bare_renderings():
    """The fence is a habit (126/130 outputs), not a guarantee, so it is
    documented as optional in both directions. An optional wrapper that turns
    out to be mandatory is an over-constraint of exactly the kind this file
    exists for — and it is only visible by simulating both branches.
    """
    body = 500
    grammar = Dfa(n_states=2, transitions=((0, body, 1),), start=0,
                  finals=frozenset({1}))
    a = compile_automaton(wrap_with_fence(grammar), name="fence_opt",
                          end_tokens=(1,), vocab_size=4096)
    sim = Simulator(a)
    assert sim.accepts([body, 1]), "the bare rendering became unreachable"
    assert sim.accepts([2717, 3723, 107, body, 107, 2717, 1]), (
        "the fenced rendering is not accepted")
