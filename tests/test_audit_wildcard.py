"""AUDIT SUITE — BFCL's `any` wildcard, and the unparenthesised alternation.

Written by the **tester** agent of CLAUDE.md's tester -> coder -> reviewer loop,
from the schema semantics and from the measured failure, **not** from the
implementation. Most of this file is RED today, on purpose: the feature it
specifies does not exist yet.

--------------------------------------------------------------------------
The defect
--------------------------------------------------------------------------

`compile/schema.py:normalize_bfcl_schema` translates BFCL's `"type": "any"` by
*omitting* `type` and setting a `__wildcard__` marker. `outlines_core` expands
an omitted type into a seven-way alternation over the JSON types **and does not
wrap it in a group**, so composing it as a property value yields

    \\{ws"input_value"ws:ws((true|false))|(null)|(number)|(integer)|(string)|(array)|(object)ws\\}

`|` binds loosest. Only the *first* branch carries the opening `\\{"key":`, and
only the *last* carries the closing `\\}`. Measured on `reverse_input`
(`live_simple_117-73-0`), whitespace `schema.JSON_WS`, `allow_wildcard=True`:

    rejected  {"input_value": "say hi"}      <- the ground-truth answer
    rejected  all five standard renderings   (accepts_all_renderings -> 0/5)
    ACCEPTED  {"input_value": "say hi"}}     <- extra brace; the observed emission
    ACCEPTED  {"input_value": true           <- unterminated object
    ACCEPTED  1   null   "say hi"   1.5   [1]   []   {"a": 1}}

This is not hypothetical. `artifacts/exp_e5_grammar130_map.json`, record
`live_simple_122-78-0`, emitted the single character `1` with `accepted=True`
(CS 1.000) and `schema_ok=False`, while the unconstrained arm on the same record
emitted the correct complete object. `live_simple_117-73-0` emitted
`{"input_value": "hi say reverse"}}` — valid under the grammar, invalid JSON.
The automaton was doing its job perfectly; the *language* was wrong.

--------------------------------------------------------------------------
What these tests pin
--------------------------------------------------------------------------

Two properties, stated separately so the *width* of the expansion can be
changed later without rewriting anything:

  P1  no over-constraint — the grammar must not reject a value the schema
      permits, and must accept every standard rendering of it;
  P2  no under-constraint — the grammar must not accept a document the schema
      forbids, in particular anything that is not a JSON object with the
      declared keys.

P1's floor is the five JSON *scalars*, which every candidate width admits and
which `reverse_input`'s own description names ("Can be a string, boolean, or
number (integer or float)"). Beyond the floor, §2b asserts that the grammar's
accepted type set equals the implementation's **declared** set exactly — that
is the enforceable form of CLAUDE.md's "never silently cap coverage": you may
narrow, but the narrowing has to be written down where a test can read it.

--------------------------------------------------------------------------
The API contract (tests are written first, so this file defines it)
--------------------------------------------------------------------------

  schema.WILDCARD_ANYOF_TYPES : tuple[str, ...]
      The JSON type names `any` expands to, in `__all__`. The declaration the
      "no silent narrowing" test reads.

  schema.normalize_bfcl_schema(node, *, expand_wildcard: bool = True)
  schema.build_regex(..., expand_wildcard: bool = True)
  pipeline.compile_json_schema(..., expand_wildcard: bool = True)
      Default `True` per the user's design directive: the safe behaviour is
      what you get when you do not ask.

  eval/run.py: `--expand-wildcard` / `--no-expand-wildcard`, default True, and
  the key `"expand_wildcard"` in the artifact's top-level dict.

`WILDCARD_FLAG` below holds the parameter name. If the coder wants a different
name they must say so to the reviewer rather than edit this file.

Run:
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
        python -m pytest tests/test_audit_wildcard.py -x -q

Mutation harness (CLAUDE.md's `DGFA_MUT` convention, see the bottom of the
file):
    DGFA_MUT=fix7 python -m pytest tests/test_audit_wildcard.py -q
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import pathlib
import re

import pytest

from diffgemma_fa.compile import pipeline
from diffgemma_fa.compile import schema as S

REPO = pathlib.Path("/home/ubuntu/diffgemma_fa")
RUN_PY = REPO / "diffgemma_fa/eval/run.py"

#: The name of the opt-out parameter. Part of the contract — see the module
#: docstring. Changing it is a conversation with the reviewer, not an edit.
WILDCARD_FLAG = "expand_wildcard"

#: The whitespace regime **every** measurement and assertion in this file uses,
#: stated once because CLAUDE.md requires the whole regime beside every number.
#: `schema.JSON_WS` is RFC 8259's `*( %x20 / %x09 / %x0A / %x0D )` and is the
#: production default (`compile_json_schema`, and `eval/run.py --whitespace=json`).
WS = S.JSON_WS

#: The other half of the regime: BFCL `any` reaches outlines only when the
#: caller has opted into the wildcard, which every production call site does
#: (`tasks/bfcl.py:52`, `eval/run.py:298`).
ALLOW_WILDCARD = True


# ===========================================================================
# The real schemas. Verbatim from BFCL v4 Live; `test_the_fixtures_are_the
# _real_schemas` re-reads them from the dataset so they cannot drift.
# ===========================================================================

#: `live_simple_117-73-0`. One property, and it is the wildcard. The minimal
#: reproduction, and the record whose constrained emission carried a stray `}`.
REVERSE_INPUT = {
    "type": "dict",
    "required": ["input_value"],
    "properties": {
        "input_value": {
            "type": "any",
            "description": "The value to be reversed. Can be a string, "
                           "boolean, or number (integer or float).",
        }
    },
}

#: `live_simple_122-78-0`. Three properties, the *last* of which is the
#: wildcard — so the leak has two typed properties in front of it and still
#: fires. This is the record that emitted the bare `1`.
PROCESS_DATA = {
    "type": "dict",
    "required": ["image_path", "question", "model"],
    "properties": {
        "image_path": {
            "type": "string",
            "description": "The file path to the image on which the question "
                           "is based, in the format of "
                           "'folder/subfolder/image.png'.",
        },
        "question": {
            "type": "string",
            "description": "The question to be answered, related to the "
                           "content of the provided image.",
        },
        "model": {
            "type": "any",
            "description": "The pre-loaded question-answering model from the "
                           "Hugging Face Transformers library used to answer "
                           "the question.",
        },
    },
}

#: `live_multiple_83-38-0`. The wildcard sits *between* two typed properties,
#: and the schema also carries a `dict` with an empty `properties` map — a
#: second wildcard-ish shape in the same schema.
ADD_DEFAULT_VALUE = {
    "type": "dict",
    "required": ["dict", "key", "default_value"],
    "properties": {
        "dict": {
            "type": "dict",
            "description": "The dictionary to which the default value should "
                           "be added.",
            "properties": {},
        },
        "key": {
            "type": "string",
            "description": "The key for which the default value is to be set.",
        },
        "default_value": {
            "type": "any",
            "description": "The value to set for the key if it does not exist "
                           "in the dictionary. The type of this value can be "
                           "any valid Python data type.",
        },
    },
}

#: `live_multiple_182-77-0` / `live_parallel_multiple_13-11-0`. Wildcard first,
#: typed properties after — the mirror image of PROCESS_DATA.
ESTIMATE_DERIVATIVE = {
    "type": "dict",
    "required": ["function", "x"],
    "properties": {
        "function": {
            "type": "any",
            "description": "The function of which to calculate the "
                           "derivative. It should be a single-variable "
                           "function.",
        },
        "x": {
            "type": "float",
            "description": "The point at which to calculate the derivative, "
                           "expressed as a floating-point number.",
        },
        "delta": {
            "type": "float",
            "description": "The small change in the input of the function "
                           "used for calculating the derivative. It "
                           "determines the accuracy of the approximation.",
            "default": 0.001,
        },
    },
}

#: schema fixture -> (record id, function name, the wildcard property's name).
REAL_SCHEMAS = {
    "reverse_input": (REVERSE_INPUT, "live_simple_117-73-0", "input_value"),
    "process_data": (PROCESS_DATA, "live_simple_122-78-0", "model"),
    "add_default_value": (ADD_DEFAULT_VALUE, "live_multiple_83-38-0",
                          "default_value"),
    "estimate_derivative": (ESTIMATE_DERIVATIVE, "live_multiple_182-77-0",
                            "function"),
}

#: The eleven BFCL-Live schemas carrying `"type": "any"`, measured by walking
#: every `type` in every `function[].parameters` across the four Live splits
#: (`test_exactly_eleven_bfcl_live_schemas_carry_the_wildcard` re-measures it).
WILDCARD_RECORD_IDS = (
    "live_simple_117-73-0",
    "live_simple_122-78-0",
    "live_multiple_83-38-0",
    "live_multiple_84-38-1",
    "live_multiple_85-38-2",
    "live_multiple_86-38-3",
    "live_multiple_87-38-4",
    "live_multiple_88-38-5",
    "live_multiple_182-77-0",
    "live_parallel_multiple_13-11-0",
    "live_parallel_multiple_14-12-0",
)

#: The two of those eleven that are in `live_simple`, i.e. inside the 130-record
#: cut every published BFCL arm reports on. These are the records the build gate
#: currently refuses, which is why the recent arms say `n = 128`.
GATE_REFUSED_IDS = ("live_simple_117-73-0", "live_simple_122-78-0")


# ===========================================================================
# Probe values. One representative per JSON type, and their JSON type name.
# ===========================================================================

#: JSON type name -> a Python value that `json.dumps` renders as that type.
#: `"say hi"` is `live_simple_117-73-0`'s literal ground-truth answer.
TYPE_PROBES: dict[str, object] = {
    "string": "say hi",
    "integer": 7,
    "number": 1.5,
    "boolean": True,
    "null": None,
    "array": [1, "a"],
    "object": {"a": 1},
}

#: The five JSON scalars. **P1's floor**: no candidate width may reject these.
#: Anchored outside the implementation — `reverse_input`'s own description says
#: "Can be a string, boolean, or number (integer or float)", the ground truth
#: at every wildcard position across all eleven records is a `string`, and
#: `default.add_default_value`'s says "any valid Python data type", which
#: includes `None`.
SCALAR_FLOOR = ("string", "integer", "number", "boolean", "null")

#: Every JSON type. What `any` / a typeless JSON Schema means literally.
ALL_JSON_TYPES = tuple(TYPE_PROBES)

#: The five standard renderings, as `schema.accepts_all_renderings` defines
#: them. Duplicated deliberately: this file must be able to say *which*
#: rendering failed without depending on that helper's return shape.
def renderings(instance: dict) -> dict[str, str]:
    """The five renderings of `instance` that all mean the same document."""
    def d(**kw) -> str:
        return json.dumps(instance, ensure_ascii=False, **kw)

    return {
        "compact": d(separators=(",", ":")),
        "spaced": d(),
        "indent2": d(indent=2),
        "indent4": d(indent=4),
        "tabs": d(indent="\t"),
    }


# ===========================================================================
# Helpers. None of these re-implements the thing under test: they build
# documents from the *schema's* meaning and ask the *production* regex.
# ===========================================================================

def production_regex(bfcl_schema: dict, **kwargs) -> str:
    """The regex a production call site gets. `tasks/bfcl.py` / `eval/run.py`.

    Both call `compile_json_schema` with `allow_wildcard=True`,
    `whitespace_pattern=JSON_WS` and `from_bfcl=True`; that lands here.
    """
    return S.build_regex(
        bfcl_schema, from_bfcl=True, allow=ALLOW,
        allow_wildcard=ALLOW_WILDCARD, whitespace_pattern=WS, **kwargs)


#: The keyword allow-list the production call sites pass. Read from
#: `tasks/bfcl.py` so a change there cannot make this file test a different
#: grammar than the one that ships.
def _production_allow() -> tuple[str, ...]:
    from diffgemma_fa.compile.tasks import bfcl as _bfcl
    return tuple(getattr(_bfcl, "ALLOW", ()))


ALLOW = _production_allow()


def accepts(regex: str, document: str) -> bool:
    """Is `document` in the language of the **anchored** `regex`?

    `re.fullmatch` is anchored at both ends, which is what `outlines_core`'s
    `regex-automata` DFA is (SPEC §4.3: "matching is anchored"). Alternation
    and grouping have identical semantics in the two dialects, which is the
    only construct this file's assertions turn on.
    """
    return re.fullmatch(regex, document) is not None


def instance_with(bfcl_schema: dict, prop: str, value: object) -> dict:
    """A full instance of `bfcl_schema` whose wildcard property holds `value`.

    Every *other* required property gets an obviously-valid value of its
    declared type, so a rejection can only be about the wildcard. Key order is
    the declared order, because the grammar is ordered (SPEC §4.2) — and that
    restriction is measured elsewhere, not here.
    """
    filler = {"string": "x", "integer": 1, "number": 1.0, "float": 1.0,
              "boolean": True, "null": None, "array": [1], "object": {},
              "dict": {}}
    out: dict[str, object] = {}
    props = bfcl_schema["properties"]
    required = set(bfcl_schema.get("required") or ())
    for key, sub in props.items():
        if key == prop:
            out[key] = value
        elif key in required:
            t = sub.get("type")
            assert t in filler, f"no filler for declared type {t!r} at {key!r}"
            out[key] = filler[t]
    return out


def top_level_alternation(regex: str) -> bool:
    """Does `regex` contain a `|` at group depth 0, outside a character class?

    A generic property of a regex string. It does not model outlines' schema
    translation, JSON, or anything else under test — it answers one syntactic
    question about a string, and §4 pairs it with a behavioural assertion so a
    bug in this scanner cannot by itself make a test pass or fail.
    """
    depth = 0
    in_class = False
    i = 0
    while i < len(regex):
        c = regex[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
        elif c == "[":
            in_class = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            return True
        i += 1
    return False


def accepted_type_set(regex: str, bfcl_schema: dict, prop: str) -> set[str]:
    """Which JSON types the grammar admits at `prop`, one probe per type."""
    out = set()
    for name, value in TYPE_PROBES.items():
        doc = json.dumps(instance_with(bfcl_schema, prop, value))
        if accepts(regex, doc):
            out.add(name)
    return out


# ===========================================================================
# §0  Fixtures and population. Guards against this file testing a schema BFCL
#     does not contain, or a population that has moved.
# ===========================================================================

def _bfcl_available() -> bool:
    from diffgemma_fa.compile import bfcl_data
    return (bfcl_data.DATA_DIR / bfcl_data.LIVE_SPLITS[0]).exists()


requires_bfcl = pytest.mark.skipif(
    not _bfcl_available(),
    reason="BFCL v4 data not present under artifacts/data/gorilla; the four "
           "schema fixtures in this file are then unverified copies")


@requires_bfcl
@pytest.mark.parametrize("fixture", sorted(REAL_SCHEMAS))
def test_the_fixtures_are_the_real_schemas(fixture):
    """The fixtures must be byte-identical to what BFCL ships.

    A tester who paraphrases the schema tests a schema nobody runs. This is the
    guard for the whole file: everything below asserts against these four
    dicts.
    """
    from diffgemma_fa.compile import bfcl_data

    want, rec_id, prop = REAL_SCHEMAS[fixture]
    for split in bfcl_data.LIVE_SPLITS:
        for rec in bfcl_data.iter_split(split):
            if rec.id != rec_id:
                continue
            got = [f["parameters"] for f in rec.functions
                   if json.dumps(f["parameters"]).find('"any"') >= 0]
            assert got, f"{rec_id} carries no `any` schema any more"
            assert want in got, (
                f"{fixture} fixture has drifted from {rec_id}:\n"
                f"  fixture = {json.dumps(want, sort_keys=True)}\n"
                f"  dataset = {json.dumps(got[0], sort_keys=True)}")
            assert prop in want["properties"]
            assert want["properties"][prop]["type"] == "any"
            return
    pytest.fail(f"record {rec_id} not found in the Live splits")


@requires_bfcl
def test_exactly_eleven_bfcl_live_schemas_carry_the_wildcard():
    """The population, re-measured rather than quoted.

    SPEC §4.7 says 11 of 4,549. If that number moves, every coverage statement
    in the report moves with it — including "the build gate refuses 11" and
    "`n` goes 128 -> 130".
    """
    from diffgemma_fa.compile import bfcl_data

    def types_in(node, acc):
        if isinstance(node, list):
            for v in node:
                types_in(v, acc)
        elif isinstance(node, dict):
            t = node.get("type")
            if isinstance(t, str):
                acc.append(t)
            elif isinstance(t, list):
                acc.extend(t)
            for k, v in node.items():
                if k != "description":
                    types_in(v, acc)

    total = 0
    hits = []
    for split in bfcl_data.LIVE_SPLITS:
        for rec in bfcl_data.iter_split(split):
            for fn in rec.functions:
                total += 1
                acc: list[str] = []
                types_in(fn.get("parameters"), acc)
                if "any" in acc:
                    hits.append(rec.id)

    assert total == 4549, f"BFCL-Live schema count moved: {total}"
    assert sorted(set(hits)) == sorted(WILDCARD_RECORD_IDS), (
        f"the wildcard population moved: {sorted(set(hits))}")


# ===========================================================================
# §1  THE DEFECT. RED today; every assertion here is a measured fact about the
#     grammar `eval/run.py` compiles right now.
# ===========================================================================

@pytest.mark.parametrize("fixture", sorted(REAL_SCHEMAS))
def test_grammar_accepts_the_object_the_schema_describes(fixture):
    """**P1, the headline.** A one-line summary of the whole bug.

    `{"input_value": "say hi"}` is `live_simple_117-73-0`'s ground-truth answer.
    A grammar that cannot express the correct answer has already lost the
    record, and no amount of sampler correctness recovers it — which is exactly
    what happened: CS was 1.000 while `schema_valid` was False.
    """
    schema, rec_id, prop = REAL_SCHEMAS[fixture]
    rx = production_regex(schema)
    doc = json.dumps(instance_with(schema, prop, "say hi"))
    assert accepts(rx, doc), (
        f"{rec_id}: the grammar REJECTS {doc}, a string at a `type: any` "
        f"property. The correct answer is not in the language.")


@pytest.mark.parametrize("fixture", sorted(REAL_SCHEMAS))
def test_grammar_accepts_all_five_renderings_of_a_permitted_instance(fixture):
    """P1 under every whitespace the model may choose.

    Same property as `accepts_all_renderings`, asserted here on an instance
    whose wildcard slot is deliberately filled — `synthesize_instance` picks
    `1` for a wildcard, and a test that only ever probes one JSON type cannot
    see a width regression.
    """
    schema, rec_id, prop = REAL_SCHEMAS[fixture]
    rx = production_regex(schema)
    inst = instance_with(schema, prop, "say hi")
    bad = [k for k, doc in renderings(inst).items() if not accepts(rx, doc)]
    assert not bad, (
        f"{rec_id}: grammar rejects rendering(s) {bad} of {inst!r}")


@pytest.mark.parametrize(
    "leak", ["1", "null", '"say hi"', "1.5", "[1]", "[]", "true"])
def test_grammar_rejects_a_bare_scalar_as_a_whole_answer(leak):
    """**P2, the headline.** A bare scalar is not an argument object.

    Measured today on `reverse_input`: `1`, `null`, `"say hi"`, `1.5`, `[1]`
    and `[]` are all ACCEPTED as complete answers, because the alternation is
    unparenthesised and every branch but the first is a free-standing JSON
    value. `live_simple_122-78-0` emitted exactly this — the single character
    `1`, with `accepted=True`.

    `true` is in the list as a **control**: it is correctly rejected today,
    because the boolean branch is the *first* one and so still carries the
    `\\{"key":` prefix. If a future implementation starts accepting it, the
    parametrisation says so without a second test.
    """
    rx = production_regex(REVERSE_INPUT)
    assert not accepts(rx, leak), (
        f"the grammar accepts the bare document {leak!r} as a complete answer "
        f"to reverse_input; the schema requires an object with the required "
        f"key `input_value`")


@pytest.mark.parametrize("fixture", sorted(REAL_SCHEMAS))
def test_grammar_rejects_the_double_brace_emission(fixture):
    """P2 on the shape the model *actually emitted*.

    `artifacts/exp_e5_grammar130_map.json` and `exp_e5_grammar130_s1.json` both
    contain `{"input_value": "hi say reverse"}}` and
    `{"image_path": ..., "model": "vikhyatk/moondream2"}}` — a complete correct
    object followed by a stray `}`. That is not a model error: the last branch
    of the alternation is `(object)ws\\}`, so a generic object followed by one
    closing brace is *in the language*, and the constrained sampler emitted a
    member of the language it was given. Invalid JSON; the record cannot score.
    """
    schema, rec_id, prop = REAL_SCHEMAS[fixture]
    rx = production_regex(schema)
    doc = json.dumps(instance_with(schema, prop, "say hi")) + "}"
    assert not accepts(rx, doc), (
        f"{rec_id}: the grammar accepts {doc} — a valid object plus a stray "
        f"closing brace. This is the emission recorded in "
        f"artifacts/exp_e5_grammar130_*.json")


def test_grammar_rejects_an_unterminated_object():
    """P2 at the other end: the *first* branch has no closing brace.

    Measured today: `{"input_value": true` (no `}`) is accepted, because branch
    one is `\\{ws"input_value"ws:ws((true|false))` and the `ws\\}` lives on
    branch seven. Under a block-structured decoder an unterminated object is
    the worst possible outcome — it is a live, accepting state at a canvas
    boundary.
    """
    rx = production_regex(REVERSE_INPUT)
    for doc in ('{"input_value": true', '{"input_value": false'):
        assert not accepts(rx, doc), (
            f"the grammar accepts the unterminated object {doc!r}")


def test_the_gate_can_see_a_wildcard_defect_at_all():
    """The gate is not the bug — it is the only thing that caught it.

    GREEN before and after the fix, and deliberately so: it asserts a property
    of the **gate**, not of the current grammar. Feed it the grammar this
    module produces today (which rejects all five renderings of its own
    instance) and it must say so.

    Rewritten from a first draft that read `if hard: assert not ok else: assert
    ok` — which is `assert True` spelled at length, and would have been the
    seventh vacuous test in this codebase. The regex is now pinned rather than
    recomputed, so the assertion cannot follow the implementation around.
    """
    # A literal instance, not `synthesize_instance`'s: what that helper picks
    # for a wildcard is itself a function of the normaliser under test, and
    # pinning it made this test fail under a *correct* candidate fix (it
    # returns `1` today and `"a"` once `any` expands to an `anyOf` whose first
    # arm is `string`). The gate's behaviour must not depend on that.
    inst = {"input_value": 1}

    # The literal shape of the defect: the object's braces attached to the
    # first and last branch of an unparenthesised alternation. Written out, not
    # generated, so a repaired `build_regex` cannot make this test evaporate.
    broken = (r'\{' + WS + r'"input_value"' + WS + ':' + WS
              + r'(true|false)|(null)|((-)?(0|[1-9][0-9]*))|'
              + r'("([^"\\]|\\["\\/bfnrt])*")' + WS + r'\}')
    ok, bad = S.accepts_all_renderings(broken, inst)
    hard = [b for b in bad if b != "reordered-keys"]
    assert not ok and set(hard) == {"compact", "spaced", "indent2", "indent4",
                                    "tabs"}, (
        f"the build gate does not detect a grammar that rejects every "
        f"rendering of its own instance: ok={ok} bad={bad}")

    # ...and it must pass a grammar that is right, or it is just a tripwire.
    good = production_regex({"type": "dict", "required": ["input_value"],
                             "properties": {"input_value": {"type": "string"}}})
    assert S.accepts_all_renderings(good, {"input_value": "a"})[0]


# ===========================================================================
# §2  THE PROPERTIES, WIDTH-AGNOSTIC. These survive a change of mind about
#     5-way vs 7-way; §3 is where the choice itself is asserted.
# ===========================================================================

@pytest.mark.parametrize("json_type", SCALAR_FLOOR)
@pytest.mark.parametrize("fixture", sorted(REAL_SCHEMAS))
def test_p1_floor_every_json_scalar_is_admitted_at_a_wildcard_property(
        fixture, json_type):
    """P1's floor. Independent of which width is chosen.

    The floor is *not* read from the implementation. It comes from the schemas'
    own text — `reverse_input`: "Can be a string, boolean, or number (integer
    or float)"; `default.add_default_value`: "can be any valid Python data
    type" — and from the ground truth, where every wildcard-position answer
    across all eleven records is a string.
    """
    schema, rec_id, prop = REAL_SCHEMAS[fixture]
    rx = production_regex(schema)
    doc = json.dumps(instance_with(schema, prop, TYPE_PROBES[json_type]))
    assert accepts(rx, doc), (
        f"{rec_id}: the grammar rejects a {json_type} at the `any` property "
        f"`{prop}`: {doc}")


@pytest.mark.parametrize("fixture", sorted(REAL_SCHEMAS))
def test_p2_nothing_outside_an_argument_object_is_admitted(fixture):
    """P2, stated once and for all widths.

    A wildcard *value* may be anything; the *document* may not. Two families:
    every bare JSON value, and every object missing a required key.
    """
    schema, rec_id, prop = REAL_SCHEMAS[fixture]
    rx = production_regex(schema)

    bare = [json.dumps(v) for v in TYPE_PROBES.values()]
    accepted = [d for d in bare if accepts(rx, d)]
    assert not accepted, (
        f"{rec_id}: accepted as complete answers, though none is an argument "
        f"object with the required keys: {accepted}")

    required = list(schema.get("required") or ())
    if len(required) > 1:
        for drop in required:
            inst = instance_with(schema, prop, "say hi")
            inst.pop(drop, None)
            doc = json.dumps(inst)
            assert not accepts(rx, doc), (
                f"{rec_id}: accepted {doc}, which omits the required key "
                f"{drop!r}")


@pytest.mark.parametrize("fixture", sorted(REAL_SCHEMAS))
def test_the_admitted_type_set_equals_the_declared_one(fixture):
    """**Never silently cap coverage** — the enforceable form.

    Narrowing `any` is a legitimate engineering choice. Narrowing it *silently*
    is the thing CLAUDE.md forbids, and a grammar is exactly where a silent
    narrowing hides: it produces well-formed JSON forever and only ever fails
    on the argument nobody thought to try.

    So the implementation must publish `WILDCARD_ANYOF_TYPES`, and this test
    holds it to it in **both** directions: every declared type is admitted (no
    undeclared narrowing) and no undeclared type is admitted (no accidental
    widening, which would put the 38 KB shape back without anyone choosing it).
    """
    declared = getattr(S, "WILDCARD_ANYOF_TYPES", None)
    assert declared is not None, (
        "compile/schema.py must export WILDCARD_ANYOF_TYPES: tuple[str, ...], "
        "the JSON types `any` expands to. Without a written-down declaration "
        "there is nothing to hold the grammar to, and a narrowing is silent "
        "by construction.")
    assert isinstance(declared, tuple), "WILDCARD_ANYOF_TYPES must be a tuple"
    assert set(declared) <= set(ALL_JSON_TYPES), (
        f"unknown JSON type names in WILDCARD_ANYOF_TYPES: "
        f"{sorted(set(declared) - set(ALL_JSON_TYPES))}")
    assert set(SCALAR_FLOOR) <= set(declared), (
        f"WILDCARD_ANYOF_TYPES omits scalar(s) "
        f"{sorted(set(SCALAR_FLOOR) - set(declared))}; the schemas' own "
        f"descriptions name them")

    schema, rec_id, prop = REAL_SCHEMAS[fixture]
    admitted = accepted_type_set(production_regex(schema), schema, prop)
    assert admitted == set(declared), (
        f"{rec_id}: grammar admits {sorted(admitted)} at `{prop}` but "
        f"WILDCARD_ANYOF_TYPES declares {sorted(declared)}. "
        f"missing={sorted(set(declared) - admitted)} "
        f"extra={sorted(admitted - set(declared))}")


def test_the_expansion_is_declared_in_the_module_api():
    """A declaration nobody can import is not a declaration."""
    assert "WILDCARD_ANYOF_TYPES" in getattr(S, "__all__", ()), (
        "WILDCARD_ANYOF_TYPES must be in compile/schema.py's __all__")


# ===========================================================================
# §3  THE CHOICE. Separate from the properties, and asserted explicitly, so
#     changing your mind about the width is a one-line edit *here* and a
#     visible one.
# ===========================================================================

def test_the_default_width_is_the_full_seven_way():
    """`any` means any JSON value, and the faithful reading is not the
    expensive one.

    The 212x figure in the brief is an artefact of one particular *spelling*.
    Measured, `whitespace_pattern=JSON_WS`, property name `v`, regex length:

        current (omitted `type`, broken)                  15,265
        anyOf over 7 bare types                           15,253   <- this
        anyOf, array as {items:{}}, object as
            {additionalProperties:true}                   38,369
        anyOf over the 5 scalars                             133

    The middle row is 12 characters *shorter* than the shape that ships today,
    and accepts the identical language: `(?:<current wildcard regex>)` and the
    7-way `anyOf` agreed on 10,000/10,000 randomly generated JSON documents
    rendered two ways each. The 38 KB spelling differs only by admitting arrays
    one nesting level deeper (depth 4 vs 3) — outlines bounds the recursion
    either way, so *every* candidate is a bounded approximation of `any` and
    the honest question is which bound, not whether.

    Against that, the 5-way rejects 2,710 of those same 10,000 documents. It
    would cost nothing on BFCL-Live today — all four ground-truth values at
    wildcard positions are strings — but "the benchmark does not currently
    exercise it" is the argument that retired the Countdown grammar.

    **Regex length is not what decides this; the automaton is.** Full pipeline
    on `reverse_input`, `JSON_WS`, `channel_header=True`, Gemma's 262,144-token
    vocabulary, CPU:

        reverse_input        |S|  bucket  edges  classes  tree GB   wall s
        current (broken)     568    1024   2810      208    2.143    361.6
        anyOf 7 bare types   568    1024   2594      182    2.143    246.9
        anyOf 5 scalars       57      64    141       94    0.008      4.0

        process_data         |S|  bucket  edges  classes  tree GB   wall s
        current (broken)     610    1024   3274      268    2.143    376.7
        anyOf 7 bare types   610    1024   2728      223    2.143    255.5
        anyOf 5 scalars       99     128    275      142    0.033      9.2

    The seven-way is the *same automaton size* as what the wildcard already
    compiles to — bucket 1024 is already in SPEC §4.7's histogram (1024:14) and
    2.14 GB is already its largest tree, both of them these very schemas. The
    five-way would be genuinely cheaper: one bucket instead of four levels up,
    and ~60x faster to compile (pure-Python Valmari is 200 s of the 247 s).

    That is a real saving and it is the reason this is a `[D]`ecision rather
    than an obvious call. It is resolved for **fidelity** because the saving
    buys nothing measurable: the eleven schemas are 0.24% of BFCL-Live, they
    add no new shape bucket, `needs_chain_path` stays False, and the tree fits
    with ~18 GB to spare. If a §7.3 timing later shows those eleven dominating
    the overhead table, flip this test and say so in `docs/RESULTS.md` — that
    is a declared narrowing, which is allowed. A silent one is not.
    """
    assert tuple(getattr(S, "WILDCARD_ANYOF_TYPES", ())) == ALL_JSON_TYPES, (
        f"expected the full seven-way expansion {ALL_JSON_TYPES}, got "
        f"{getattr(S, 'WILDCARD_ANYOF_TYPES', None)}")


def test_the_seven_way_does_not_cost_more_regex_than_todays_broken_shape():
    """The cost claim behind §3's choice, re-measured rather than quoted.

    Regime: `whitespace_pattern=schema.JSON_WS`, the wildcard as a required
    property of a one-key object. Regex length is *not* the number that decides
    the design — `|S|` is — but it is the number the "too large a regex OOM-ed
    the host" warning in SPEC §4.2 is about, so it gets its own assertion.
    """
    from outlines_core.json_schema import build_regex_from_schema

    def obj(sub: dict) -> str:
        return build_regex_from_schema(
            json.dumps({"type": "object", "properties": {"v": sub},
                        "required": ["v"]}),
            whitespace_pattern=WS)

    todays = obj({})
    seven = obj({"anyOf": [{"type": t} for t in ALL_JSON_TYPES]})
    verbose = obj({"anyOf": [{"type": t} for t in SCALAR_FLOOR]
                   + [{"type": "array", "items": {}},
                      {"type": "object", "additionalProperties": True}]})

    assert len(seven) <= len(todays), (
        f"the parenthesised seven-way ({len(seven)}) is longer than the "
        f"unparenthesised shape that ships today ({len(todays)})")
    assert len(verbose) > 2 * len(seven), (
        f"the verbose spelling is supposed to be the expensive one: "
        f"{len(verbose)} vs {len(seven)}")


def test_the_seven_way_and_todays_wildcard_are_the_same_language():
    """The fix is a **parenthesisation**, not a change of language.

    If this fails, the fix has quietly redefined what `any` means, and the
    right response is to say so in the results table, not to adjust the test.

    Regime: `JSON_WS`, sub-schema regex compared *alone* (not embedded), so
    that the only difference under test is the grouping.
    """
    import random
    from outlines_core.json_schema import build_regex_from_schema

    wild = build_regex_from_schema(json.dumps({}), whitespace_pattern=WS)
    seven = build_regex_from_schema(
        json.dumps({"anyOf": [{"type": t} for t in ALL_JSON_TYPES]}),
        whitespace_pattern=WS)
    wrapped = f"(?:{wild})"

    rng = random.Random(20260812)

    def rnd(depth=0):
        if depth > 4:
            return rng.choice([1, "a", True, None, 1.5])
        k = rng.randrange(7)
        if k == 0:
            return rng.randint(-99, 99)
        if k == 1:
            return rng.choice(["a", "", "xy", "a b"])
        if k == 2:
            return rng.choice([True, False])
        if k == 3:
            return None
        if k == 4:
            return round(rng.uniform(-9, 9), 2)
        if k == 5:
            return [rnd(depth + 1) for _ in range(rng.randrange(4))]
        return {f"k{i}": rnd(depth + 1) for i in range(rng.randrange(3))}

    disagree = []
    n_accepted = 0
    for _ in range(2000):
        for sep in ((",", ":"), (", ", ": ")):
            doc = json.dumps(rnd(), separators=sep)
            a = accepts(wrapped, doc)
            if a:
                n_accepted += 1
            if a != accepts(seven, doc):
                disagree.append(doc[:80])
    # Guard against a vacuous pass: if neither regex accepted anything, the
    # zero disagreement count would mean nothing.
    assert n_accepted > 1000, (
        f"the probe corpus is not exercising the grammar: only {n_accepted} "
        f"of 4000 documents were accepted")
    assert not disagree, (
        f"{len(disagree)} disagreements between the parenthesised wildcard and "
        f"the seven-way anyOf, e.g. {disagree[:5]}")


#: What the expanded wildcard **does not** admit, though `any` in the abstract
#: does. Each is `outlines_core`'s own bound, present identically before and
#: after this fix — but CLAUDE.md's "never silently cap coverage" applies to
#: inherited caps too, so they are named here rather than left to be discovered.
KNOWN_WILDCARD_NARROWINGS: dict[str, str] = {
    "nesting depth >= 4": json.dumps([[[[1]]]]),
    "nested object depth >= 4": '{"a": {"b": {"c": {"d": 1}}}}',
    "unsigned exponent (1e5)": "1e5",
    # The `\uXXXX` **escape sequence**, not a non-ASCII character: outlines'
    # string body permits only `\["\\/bfnrt]`, so `"é"` is refused while
    # the literal `"é"` is accepted. Writing the latter here (as a first draft
    # did) tests the opposite of what it claims.
    r"\uXXXX string escape": '"\\u00e9"',
}


def test_the_wildcard_narrowings_are_inherited_not_introduced():
    """**The caps this grammar carries, named.**

    `any` means any JSON value; no finite automaton can mean that, because
    arbitrarily nested JSON is not regular. Every candidate is therefore a
    *bounded* approximation, and the honest question was never "faithful or
    narrowed" but "which bound, and is it declared".

    Measured at the value position, `JSON_WS`, outlines-core 0.2.14:
    `(?:<pre-fix wildcard regex>)` and the seven-way `anyOf` agree on **all**
    of these, and disagree on none. So the bounds are outlines', inherited
    unchanged — this fix moved the parentheses, not the language.

    Two things that are **not** narrowings, checked because they were suggested
    to be: whitespace inside arrays and objects is fine (`[1, 2]`, `[ 1,2 ]`,
    newline-and-indent all accepted), and a signed exponent (`1e+5`) is
    accepted. Only the *unsigned* exponent is refused, which is SPEC §4.2's
    known `NUMBER` quirk and not specific to the wildcard.
    """
    from outlines_core.json_schema import build_regex_from_schema

    pre = f"(?:{build_regex_from_schema(json.dumps({}), whitespace_pattern=WS)})"
    post = build_regex_from_schema(
        json.dumps({"anyOf": [{"type": t}
                              for t in S.WILDCARD_ANYOF_TYPES]}),
        whitespace_pattern=WS)

    for name, doc in KNOWN_WILDCARD_NARROWINGS.items():
        assert not accepts(post, doc), (
            f"{name!r} is now ACCEPTED ({doc}); the documented narrowing list "
            f"is stale and docs/RESULTS.md overstates the cap")
        assert not accepts(pre, doc), (
            f"{name!r} was accepted before the fix and is not now — this fix "
            f"introduced a narrowing rather than inheriting one")

    # Non-vacuity, and the counter-claims: things that ARE admitted.
    for doc in ("[[[1]]]", "1e+5", "-0", r'"a\nb"', "[1, 2]", "[ 1,2 ]",
                '{\n  "a": 1\n}', '{"a": {"b": {"c": 1}}}'):
        assert accepts(post, doc), f"unexpectedly rejected: {doc}"
        assert accepts(pre, doc), f"pre-fix rejected {doc}; the pair is not "\
                                  f"comparable"


def test_the_five_way_would_be_a_real_narrowing():
    """The counterfactual, measured — so "5-way is fine" cannot be asserted
    without a number beside it.

    GREEN whichever width is chosen; it is evidence, not a preference.
    """
    from outlines_core.json_schema import build_regex_from_schema

    def sub(types):
        return build_regex_from_schema(
            json.dumps({"anyOf": [{"type": t} for t in types]}),
            whitespace_pattern=WS)

    five, seven = sub(SCALAR_FLOOR), sub(ALL_JSON_TYPES)
    narrowed = [json.dumps(TYPE_PROBES[t]) for t in ("array", "object")]
    for doc in narrowed:
        assert accepts(seven, doc), f"seven-way rejects {doc}"
        assert not accepts(five, doc), (
            f"five-way accepts {doc}; then it is not a narrowing and this "
            f"test's premise is wrong")


# ===========================================================================
# §4  THE FLAG. Default True, opt-out loud, and visible in the artifact.
# ===========================================================================

def _flag_default(fn) -> object:
    params = inspect.signature(fn).parameters
    assert WILDCARD_FLAG in params, (
        f"{fn.__module__}.{fn.__qualname__} has no `{WILDCARD_FLAG}` "
        f"parameter")
    return params[WILDCARD_FLAG].default


@pytest.mark.parametrize("fn", [
    pytest.param(S.normalize_bfcl_schema, id="schema.normalize_bfcl_schema"),
    pytest.param(S.build_regex, id="schema.build_regex"),
    pytest.param(pipeline.compile_json_schema, id="pipeline.compile_json_schema"),
])
def test_the_wildcard_flag_defaults_to_the_safe_behaviour(fn):
    """The user's directive: `True` means apply the normalisation, and `True`
    is what you get when you do not ask.

    A safe behaviour reachable only by opting in is the shape of the `--fence`
    and `--whitespace` near-misses in this repo: the flag existed, the default
    did not use it, and every published arm ran the unsafe path.
    """
    assert _flag_default(fn) is True, (
        f"`{WILDCARD_FLAG}` must default to True; got "
        f"{_flag_default(fn)!r}")


def test_the_default_path_needs_no_keyword_to_be_correct():
    """Behavioural twin of the signature check.

    A default that is `True` in the signature but shadowed by a call-site
    argument is not a default. This asserts the *language* you get from a plain
    call, which is the only thing a user experiences.
    """
    rx = production_regex(REVERSE_INPUT)          # no flag passed at all
    assert accepts(rx, '{"input_value": "say hi"}')
    assert not accepts(rx, "1")


class _Reached(Exception):
    """Raised by the `compile_regex` stub below. Not an error."""

    def __init__(self, kwargs: dict) -> None:
        self.kwargs = kwargs
        super().__init__("compile_regex reached")


def _stop_at_compile_regex(monkeypatch):
    """Let `compile_json_schema` run regex + gate, then stop before the lift.

    The lift for a wildcard grammar is ~4 minutes and 25 GB (measured), and
    none of the assertions below is about the automaton. Stubbing the boundary
    keeps these tests CPU-cheap **and** lets them read the `schema_hash` that
    `compile_json_schema` computes, which is the thing actually under test.
    """
    def stub(regex, **kwargs):
        raise _Reached(kwargs)

    monkeypatch.setattr(pipeline, "compile_regex", stub)


def test_opting_out_is_loud_the_build_gate_still_refuses():
    """**The opt-out reproduces a known-broken grammar, so it must not be
    silent.**

    Direct precedent, from `pipeline.compile_json_schema`'s own comments: the
    `verify_strict=False` whitespace hatch was found in review to swallow
    `live_simple_117-73-0` — *this very schema* — because it suppressed every
    gate failure rather than only the whitespace ones. The fix was to scope the
    hatch to the narrowness the caller declared.

    The same trap is here. `{WILDCARD_FLAG}=False` declares "give me the
    unexpanded wildcard"; it does **not** declare "let a grammar through that
    rejects its own instance and accepts a bare `1`". So the gate must still
    refuse, and the reason it gives must not be about wildcards.
    """
    inst = S.synthesize_instance(REVERSE_INPUT, from_bfcl=True)
    with pytest.raises(ValueError) as exc:
        pipeline.compile_json_schema(
            REVERSE_INPUT, name="reverse_input", from_bfcl=True,
            allow=ALLOW, allow_wildcard=ALLOW_WILDCARD,
            whitespace_pattern=WS, verify_renderings=inst,
            **{WILDCARD_FLAG: False})
    assert "rendering" in str(exc.value).lower()
    assert not isinstance(exc.value, TypeError), (
        f"`{WILDCARD_FLAG}` is not a real parameter")


def test_the_default_path_passes_the_gate_that_the_opt_out_fails(monkeypatch):
    """The other half of `test_opting_out_is_loud_...`, and the half that makes
    it mean something.

    "Opting out raises" is also true of an implementation where *everything*
    raises — which is today. So the same call with the flag left at its default
    must get **past** the gate and into the lift.
    """
    _stop_at_compile_regex(monkeypatch)
    inst = S.synthesize_instance(REVERSE_INPUT, from_bfcl=True)
    with pytest.raises(_Reached):
        pipeline.compile_json_schema(
            REVERSE_INPUT, name="reverse_input", from_bfcl=True,
            allow=ALLOW, allow_wildcard=ALLOW_WILDCARD,
            whitespace_pattern=WS, verify_renderings=inst,
            verify_strict=True)


def test_opting_out_is_not_swallowed_by_the_whitespace_hatch():
    """`verify_strict=False` is scoped to whitespace. It must stay scoped.

    Under `--whitespace=pretty|stock` the gate downgrades to a report *only*
    when the declared-narrow whitespace pattern is the whole cause. A wildcard
    defect is not, so the raise must survive `verify_strict=False`.
    """
    inst = S.synthesize_instance(REVERSE_INPUT, from_bfcl=True)
    with pytest.raises(ValueError):
        pipeline.compile_json_schema(
            REVERSE_INPUT, name="reverse_input", from_bfcl=True,
            allow=ALLOW, allow_wildcard=ALLOW_WILDCARD,
            whitespace_pattern=WS, verify_renderings=inst,
            verify_strict=False, **{WILDCARD_FLAG: False})


def test_the_flag_changes_the_compilation_cache_key(monkeypatch):
    """**Measurement-affecting, therefore in the key.**

    `schema_fingerprint`'s own docstring: every option that can change the
    compiled language goes into the key, because a collision returns an
    internally-consistent automaton for the wrong language and nothing
    downstream can tell. This flag changes the language outright.

    Asserted on the hash `compile_json_schema` actually *computes*, captured at
    the `compile_regex` boundary. Calling `schema_fingerprint` directly proves
    nothing — it takes `**options` and hashes whatever it is handed, so any two
    distinct keyword values differ by construction. That first draft was green
    against today's code, in which the flag does not exist at all.
    """
    _stop_at_compile_regex(monkeypatch)

    hashes = {}
    for value in (True, False):
        try:
            pipeline.compile_json_schema(
                REVERSE_INPUT, name="reverse_input", from_bfcl=True,
                allow=ALLOW, allow_wildcard=ALLOW_WILDCARD,
                whitespace_pattern=WS, **{WILDCARD_FLAG: value})
        except _Reached as reached:
            hashes[value] = reached.kwargs.get("schema_hash")
        else:                                              # pragma: no cover
            pytest.fail("compile_json_schema never reached compile_regex")

    assert all(hashes.values()), f"no schema_hash was passed: {hashes}"
    assert hashes[True] != hashes[False], (
        f"`{WILDCARD_FLAG}` does not reach schema_fingerprint; two different "
        f"languages share the cache key {hashes[True]}")


def _run_py_tree() -> ast.Module:
    return ast.parse(RUN_PY.read_text())


def test_the_flag_is_reachable_from_the_eval_cli():
    """A grammar-shaping option that the experiment runner cannot set is not
    an option.

    `--whitespace` was exactly this: the repaired default was unreachable from
    the CLI because `eval/run.py` passed `None` through, and the near-miss cost
    is recorded in `compile_json_schema`'s docstring.
    """
    cli = f"--{WILDCARD_FLAG.replace('_', '-')}"
    found = {}
    for node in ast.walk(_run_py_tree()):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(getattr(node.args[0], "value", None), str)
                and node.args[0].value.lstrip("-").replace("-", "_")
                    == WILDCARD_FLAG):
            found = {kw.arg: kw.value for kw in node.keywords}
            break
    assert found, f"eval/run.py declares no {cli} argument"
    assert "default" in found, f"{cli} declares no explicit default"
    assert ast.literal_eval(found["default"]) is True, (
        f"{cli} must default to True")


def test_the_flag_is_recorded_in_the_run_artifact():
    """**Every arm's config is read back out of its artifact.**

    That is how a re-run queue gets validated, and a grammar-shaping boolean
    that never lands in the file is invisible in exactly the way `--whitespace`
    was — a default no artifact had ever used, which nearly cost 13.5 GPU-hours.

    Checked statically: reaching the `out` dict at run time needs the
    checkpoint and a GPU, and this file is CPU-only by construction.
    """
    tree = _run_py_tree()
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    keys: set[str] = set()
    for node in ast.walk(main):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    assert WILDCARD_FLAG in keys, (
        f"eval/run.py's output dict has no {WILDCARD_FLAG!r} key; the flag "
        f"would not be recoverable from the artifact. Present keys include "
        f"{sorted(k for k in keys if k in ('whitespace', 'fence', 'ci_enums', 'dtype'))}")


# ===========================================================================
# §5  IS THE BUG WILDCARD-SPECIFIC, OR GENERAL? The answer changes the size of
#     the problem: general would make 11 a floor rather than a ceiling.
# ===========================================================================

#: Sub-schema shapes that *could* produce a top-level alternation, with the
#: JSON values each one permits. Composed as a required property of an object,
#: which is how every BFCL parameter appears.
ALTERNATION_SHAPES: dict[str, tuple[dict, tuple[object, ...]]] = {
    "anyOf-2": ({"anyOf": [{"type": "string"}, {"type": "integer"}]},
                ("a", 1)),
    "anyOf-5-scalars": ({"anyOf": [{"type": t} for t in SCALAR_FLOOR]},
                        ("a", 1, 1.5, True, None)),
    "anyOf-7-full": ({"anyOf": [{"type": t} for t in ALL_JSON_TYPES]},
                     ("a", 1, 1.5, True, None, [1], {"a": 1})),
    "oneOf-2": ({"oneOf": [{"type": "string"}, {"type": "integer"}]},
                ("a", 1)),
    "nullable-type-list": ({"type": ["string", "null"]}, ("a", None)),
    "enum-3": ({"enum": ["a", "b", "c"]}, ("a", "c")),
    "enum-mixed-types": ({"enum": ["a", 1, True]}, (1, True)),
    "const": ({"const": "a"}, ("a",)),
    "items-union": ({"type": "array",
                     "items": {"anyOf": [{"type": "string"},
                                         {"type": "integer"}]}},
                    ([1, "a"],)),
    "additionalProperties-true": ({"type": "object",
                                   "additionalProperties": True},
                                  ({"a": 1},)),
    # The literal empty schema. **The only** typeless spelling outlines accepts
    # at all: a node carrying just an annotation
    # (`{"description": "a value of any type"}`) makes
    # `build_regex_from_schema` raise `ValueError: Unsupported JSON Schema
    # structure`, so "omitted type" reaches outlines only after
    # `normalize_bfcl_schema` has stripped the annotations down to `{}`. That
    # project-side route is covered by
    # `test_the_project_never_hands_outlines_a_typeless_node`; here there is
    # exactly one upstream shape to survey, not two.
    "empty-schema-WILDCARD": ({}, ("a", 1, True, None, [1], {"a": 1})),
}

#: The shapes `outlines_core` 0.2.14 composes defectively. Naming them is the
#: finding: if a shape outside this set starts failing, the bug is more general
#: than the wildcard and 11 is a floor, not a ceiling.
#:
#: These two are **upstream**, in `outlines_core`, and no change in this repo
#: can make `build_regex_from_schema` parenthesise them. The project's fix is
#: to never hand outlines a typeless node — which is what
#: `test_the_project_never_hands_outlines_a_typeless_node` requires, and that
#: one *is* fixable here.
KNOWN_DEFECTIVE_SHAPES = frozenset({"empty-schema-WILDCARD"})

#: `strict=True`: if outlines ever fixes this, these go XPASS and the suite
#: goes red, which is the signal to delete the workaround. An `xfail` with a
#: named upstream cause is a declared limitation, not a silent cap.
_upstream = pytest.mark.xfail(
    strict=True,
    reason="outlines_core 0.2.14 emits the typeless-schema expansion as a "
           "bare seven-way alternation with no enclosing group; unfixable "
           "from this repo. See test_the_project_never_hands_outlines_a_"
           "typeless_node for the property that IS this project's to keep.")


def _shape_param(name: str):
    mark = [_upstream] if name in KNOWN_DEFECTIVE_SHAPES else []
    return pytest.param(name, marks=mark, id=name)


def _shape_object_regex(sub: dict) -> str:
    from outlines_core.json_schema import build_regex_from_schema
    return build_regex_from_schema(
        json.dumps({"type": "object", "properties": {"v": sub},
                    "required": ["v"]}),
        whitespace_pattern=WS)


@pytest.mark.parametrize(
    "shape", [_shape_param(n) for n in sorted(ALTERNATION_SHAPES)])
def test_composed_subschema_keeps_the_object_intact(shape):
    """**The general property, behavioural.**

    Whatever a property's sub-schema is, embedding it must not let one of its
    branches escape the surrounding object. Two assertions, one per direction:
    the object with a permitted value is accepted, and that value on its own is
    not.

    Deliberately *not* restricted to the wildcard: if `anyOf`, `oneOf`, a
    nullable type union, an `enum` or an `items` union had the same composition
    bug, the eleven refused schemas would be the tip of it — BFCL-Live carries
    5,786 `enum` and many `anyOf`-shaped parameters.
    """
    sub, values = ALTERNATION_SHAPES[shape]
    rx = _shape_object_regex(sub)
    for value in values:
        doc = json.dumps({"v": value})
        assert accepts(rx, doc), f"{shape}: object regex rejects {doc}"
        bare = json.dumps(value)
        assert not accepts(rx, bare), (
            f"{shape}: object regex accepts the bare value {bare} as a whole "
            f"document — a branch has escaped the object")


@pytest.mark.parametrize(
    "shape", [_shape_param(n) for n in sorted(ALTERNATION_SHAPES)])
def test_composed_subschema_has_no_top_level_alternation(shape):
    """The same property, structurally — the cause rather than the symptom.

    Paired with the behavioural test above on purpose: a bug in
    `top_level_alternation` can make this test wrong, but it cannot make the
    behavioural one wrong, and the two must agree.
    """
    sub, _ = ALTERNATION_SHAPES[shape]
    rx = _shape_object_regex(sub)
    assert not top_level_alternation(rx), (
        f"{shape}: the composed object regex has a `|` at group depth 0, so "
        f"the object's braces are attached to single branches:\n"
        f"  {rx[:220]}...")


#: Routes to a typeless node that the expansion **must** cover, as
#: `(sub-schema, from_bfcl)`.
#:
#: `from_bfcl` is load-bearing and was wrong in the first draft of this file:
#: `explicit-typeless` was written as `{"description": ...}` with
#: `from_bfcl=True`, and `normalize_bfcl_schema` **drops `description`**, so
#: that case was byte-identical to `empty-schema` and tested nothing. It looked
#: like it covered the annotation-only node — which is exactly where the
#: over-firing predicate lives — and did not. Caught in review.
TYPELESS_ROUTES: dict[str, tuple[dict, bool]] = {
    # BFCL's own spelling, through the dialect translation.
    "bfcl-any": ({"type": "any", "description": "any value"}, True),
    # The literal empty schema, straight through.
    "empty-schema": ({}, False),
    # **Annotation-only.** `from_bfcl=False`, so `description` survives to the
    # predicate. Upstream this raises `ValueError: Unsupported JSON Schema
    # structure` rather than expanding, so the expansion must handle it here.
    "annotation-only-description": ({"description": "no type here"}, False),
    "annotation-only-title-default": ({"title": "T", "default": 1}, False),
    # Nested one level down, to catch a fix applied only at the top level.
    "nested-under-properties": (
        {"type": "object", "properties": {"w": {}}, "required": ["w"]}, False),
    # **The fifth route.** `prefixItems` *is* a precedence keyword, so the
    # outer node is not typeless — but its element is, and outlines composes
    # that element the same broken way (measured: depth-0 `|`, bare `1`
    # accepted, 15,337 chars).
    "prefixItems-inner-empty": ({"prefixItems": [{}]}, False),
}


@pytest.mark.parametrize("spelling", sorted(TYPELESS_ROUTES))
def test_the_project_never_hands_outlines_a_typeless_node(spelling):
    """**The fix must cover every route to a typeless node, not just `any`.**

    BFCL's `"type": "any"` is one way to reach outlines' broken expansion.
    A literal `{}` sub-schema is another, and `check_supported` lets it through
    unchanged the moment `allow_wildcard=True` — which every production call
    site passes. A fix scoped to the string `"any"` would leave the same defect
    reachable by a schema that simply omits `type`, and BFCL is not the only
    source of schemas this compiler will ever see.

    So the property is stated over `schema.build_regex`, the project's own
    boundary: whatever the spelling, what comes out must keep the object
    intact.
    """
    sub, from_bfcl = TYPELESS_ROUTES[spelling]
    schema = {"type": "object", "properties": {"v": sub}, "required": ["v"]}
    rx = S.build_regex(schema, from_bfcl=from_bfcl, allow=ALLOW,
                       allow_wildcard=ALLOW_WILDCARD, whitespace_pattern=WS)
    assert not top_level_alternation(rx), (
        f"{spelling}: depth-0 `|` survives into the project's own regex")
    assert not accepts(rx, "1"), (
        f"{spelling}: the grammar accepts the bare document `1`")
    if spelling == "prefixItems-inner-empty":
        # A tuple-typed array: the permitted document is `{"v": [<any>]}`.
        assert accepts(rx, '{"v": ["say hi"]}'), (
            f"{spelling}: the grammar rejects a string in the tuple slot")
    elif spelling == "nested-under-properties":
        assert accepts(rx, '{"v": {"w": "say hi"}}'), (
            f"{spelling}: the grammar rejects a string at the nested typeless "
            f"property")
    else:
        assert accepts(rx, '{"v": "say hi"}'), (
            f"{spelling}: the grammar rejects a string at a typeless property")


#: Nodes that carry a **real constraint** and no precedence keyword. Every one
#: of these makes `build_regex_from_schema` raise today — measured against
#: outlines-core 0.2.14 — so treating them as typeless does not "fix" anything:
#: it converts a loud `ValueError` into a silent any-JSON grammar with the
#: constraint dropped.
CONSTRAINT_BEARING_TYPELESS: dict[str, dict] = {
    "pattern": {"pattern": "^a$"},
    "minLength": {"minLength": 3},
    "items": {"items": {"type": "string"}},
    "format": {"format": "date-time"},
    "additionalProperties-schema": {"additionalProperties": {"type": "string"}},
    "maximum": {"maximum": 5},
}


@pytest.mark.parametrize("node", sorted(CONSTRAINT_BEARING_TYPELESS))
def test_a_constraint_bearing_typeless_node_raises_rather_than_widening(node):
    """**The expansion must not over-fire, and this is the decision test.**

    "No precedence keyword" is *not* the same as "means any JSON value".
    `{"pattern": "^a$"}` carries a genuine constraint that outlines refuses to
    compile — it raises `ValueError: Unsupported JSON Schema structure`. A
    predicate that only checks the precedence list treats it as a wildcard, and
    the schema's `pattern` disappears into a grammar that accepts every JSON
    document there is.

    That is a *widening*, and it is the exact failure family this project
    already paid for twice: `additionalProperties: false` "not honoured" (the
    claim was inverted), and the `enum`-at-array-level rewrite. Both were
    silent, both produced well-formed output, both were only ever visible as a
    score. Its reach on BFCL-Live is **0 of 4,549** — a latent defect, not a
    live one — which is precisely why it needs a test rather than a measurement.

    The decision, pinned: **raise `UnsupportedSchemaError`.** The fail-loud
    pre-pass is this module's whole contract (`schema.py`'s docstring: "never
    to silence a surprise"), and a caller who genuinely means "any value here"
    can write `{}` or `anyOf`. Widening to any-JSON is the one response that
    cannot be reviewed, because nothing downstream can tell it happened.
    """
    schema = {"type": "object",
              "properties": {"v": CONSTRAINT_BEARING_TYPELESS[node]},
              "required": ["v"]}

    # Non-vacuity: this is only a real hazard because outlines itself refuses
    # the node. If it ever starts compiling one, the right answer may change.
    from outlines_core.json_schema import build_regex_from_schema
    with pytest.raises(ValueError):
        build_regex_from_schema(json.dumps(schema), whitespace_pattern=WS)

    with pytest.raises(S.UnsupportedSchemaError):
        rx = S.build_regex(schema, allow=ALLOW,
                           allow_wildcard=ALLOW_WILDCARD, whitespace_pattern=WS)
        if accepts(rx, '{"v": {"a": [1]}}'):                # pragma: no cover
            pytest.fail(
                f"{node}: silently widened to any-JSON — the constraint was "
                f"dropped and outlines' own ValueError was swallowed")


def test_the_defect_is_confined_to_the_wildcard_shapes():
    """A single assertion the report can quote.

    Measured today: of the eleven composed shapes above, exactly one — the
    empty schema `{}` — carries a depth-0 `|`. `anyOf`, `oneOf`, nullable type
    unions, `enum`, `const`, `items` unions and `additionalProperties: true`
    are all correctly parenthesised by outlines-core 0.2.14. So the defect is
    **wildcard-specific**, and the eleven affected BFCL-Live schemas are a
    ceiling, not a floor.

    This is asserted rather than written down so that an outlines upgrade that
    generalises the bug cannot land quietly.
    """
    defective = {name for name in ALTERNATION_SHAPES
                 if top_level_alternation(
                     _shape_object_regex(ALTERNATION_SHAPES[name][0]))}
    assert defective <= KNOWN_DEFECTIVE_SHAPES, (
        f"the unparenthesised alternation now affects shapes beyond the "
        f"wildcard: {sorted(defective - KNOWN_DEFECTIVE_SHAPES)}. That makes "
        f"this a general composition bug and the 11-schema figure a floor.")


def test_check_supported_does_not_crash_on_a_type_union():
    """A live bug found while measuring the alternatives, kept because the
    `type: [...]` spelling of the fix trips over it.

    `check_supported` does `if t in WILDCARD_TYPES` where `WILDCARD_TYPES` is a
    `frozenset` and `t` may be a **list** (`{"type": ["string", "null"]}` is
    legal JSON Schema, and `normalize_bfcl_schema` explicitly maps type
    unions). That raises `TypeError: unhashable type: 'list'` — an unhandled
    crash where the module's whole contract is to raise
    `UnsupportedSchemaError` or pass. It fires only on the `allow_wildcard=False`
    path, which is why `eval/run.py` has never hit it.
    """
    node = {"type": "object",
            "properties": {"v": {"type": ["string", "null"]}},
            "required": ["v"]}
    try:
        S.check_supported(node)
    except S.UnsupportedSchemaError:
        pass
    except TypeError as e:                                  # pragma: no cover
        pytest.fail(f"check_supported raised TypeError on a legal type union: "
                    f"{e}")


# ===========================================================================
# §6  THE DENOMINATOR. The fix changes `n`, and a changed `n` is not
#     comparable to the published rows unless somebody says so.
# ===========================================================================

#: **The regime the blocked GPU queue actually runs**, read off `eval/run.py`'s
#: defaults: `--whitespace=json`, `--nonempty` (default `True`), `--fence`
#: absent, `verify_strict=True` because the whitespace is not declared-narrow.
#:
#: Every other gate assertion in this file runs at
#: `nonempty_required_strings=False`, which is *not* what any arm uses. That
#: gap was invisible until review: it was covered only by
#: `test_audit_measurement.py`'s arm test, so rewriting that test would have
#: silently taken the coverage with it.
ARM_REGIME: dict[str, object] = {
    "allow": None,                    # filled below; ALLOW is defined later
    "allow_wildcard": True,
    "whitespace_pattern": WS,
    "fence": False,
    "verify_strict": True,
    "nonempty_required_strings": True,
}


def compile_as_the_arm_does(bfcl_schema: dict, name: str, monkeypatch,
                            **overrides):
    """Run `eval/run.py`'s exact `compile_json_schema` call, minus the lift.

    Returns the kwargs `compile_regex` would have received; raises whatever the
    gate raises. The lift is stubbed because it is ~4 minutes and ~25 GB on a
    wildcard grammar and none of these assertions is about the automaton.
    """
    _stop_at_compile_regex(monkeypatch)
    kwargs = {k: v for k, v in ARM_REGIME.items() if k != "allow"}
    kwargs.update(overrides)
    norm = S.normalize_bfcl_schema(bfcl_schema)
    try:
        pipeline.compile_json_schema(
            bfcl_schema, name=name, from_bfcl=True, allow=ALLOW,
            verify_renderings=S.synthesize_instance(norm), **kwargs)
    except _Reached as reached:
        return reached.kwargs
    raise AssertionError("compile_json_schema never reached compile_regex")


@pytest.mark.parametrize("rec_id", GATE_REFUSED_IDS)
def test_the_two_records_compile_under_the_arms_own_regime(rec_id, monkeypatch):
    """The gate, under `--nonempty --whitespace=json`, which is what runs.

    `nonempty_required_strings=True` rewrites required string properties to
    `minLength: 1` **and** re-enters `normalize_bfcl_schema` before
    `build_regex` — a second path through the wildcard expansion, on a schema
    that has already been normalised once. `reverse_input`'s wildcard property
    is *required*, so this is not a hypothetical interaction.

    Split from the plain-regime test above deliberately: if only one of the two
    regimes breaks, the failure should say which.
    """
    schema = {"live_simple_117-73-0": REVERSE_INPUT,
              "live_simple_122-78-0": PROCESS_DATA}[rec_id]
    got = compile_as_the_arm_does(schema, rec_id, monkeypatch)
    assert got.get("schema_hash"), f"{rec_id}: no schema_hash reached the lift"


def test_the_arm_regime_is_what_the_eval_cli_actually_defaults_to():
    """`ARM_REGIME` must be the CLI's defaults, not a guess about them.

    Read by AST, so a changed default cannot leave the regime tests asserting
    against a regime nobody runs — the `--whitespace` near-miss in this repo was
    exactly a repaired default that no call site reached.
    """
    defaults = {}
    for node in ast.walk(_run_py_tree()):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(getattr(node.args[0], "value", None), str)):
            flag = node.args[0].value.lstrip("-").replace("-", "_")
            for kw in node.keywords:
                if kw.arg == "default":
                    try:
                        defaults[flag] = ast.literal_eval(kw.value)
                    except ValueError:
                        pass
    assert defaults.get("whitespace") == "json", (
        f"--whitespace default moved to {defaults.get('whitespace')!r}; "
        f"ARM_REGIME still asserts JSON_WS")
    assert defaults.get("nonempty") is True, (
        f"--nonempty default moved to {defaults.get('nonempty')!r}; "
        f"ARM_REGIME still asserts nonempty_required_strings=True")
    assert defaults.get("fence") in (False, None), (
        f"--fence default moved to {defaults.get('fence')!r}")


@pytest.mark.parametrize("rec_id", GATE_REFUSED_IDS)
def test_the_two_gate_refused_records_pass_the_gate_after_the_fix(rec_id):
    """`n = 128` becomes `n = 130`.

    Every recent BFCL-Live arm reports `n = 128` with
    `skipped_by_reason={'compile:ValueError': 2}` and
    `skipped_records=[live_simple_117-73-0, live_simple_122-78-0]` — measured
    in `artifacts/eval_bfcl_live_simple_j*.json` and `artifacts/exp_marmap_*.json`.
    Both are refused by the build gate for this defect.

    If the fix repairs them, the next arm reports 130, and **130 is not
    comparable to those 128 rows without accounting for the two records**. A
    queue is running on that assumption right now. This test exists so the
    change of denominator is a deliberate, dated event rather than a surprise
    in a results table.

    Gate-level rather than full-pipeline: the gate is what refuses the record,
    it is regex-only, and the lift for a wildcard schema is minutes
    (`test_the_full_pipeline_admits_a_wildcard_schema` does that one, opt-in).
    """
    schema = {"live_simple_117-73-0": REVERSE_INPUT,
              "live_simple_122-78-0": PROCESS_DATA}[rec_id]
    inst = S.synthesize_instance(schema, from_bfcl=True)
    ok, bad = S.accepts_all_renderings(production_regex(schema), inst)
    hard = [b for b in bad if b != "reordered-keys"]
    assert not hard, (
        f"{rec_id} still fails the build gate on {hard}; it stays out of `n` "
        f"and the arm remains n=128")


def test_the_published_arms_that_report_n_128_name_the_two_records():
    """The difference set must be computable from the artifacts alone.

    Not a test of the fix — a test of whether the comparison the fix forces is
    *possible*. If an n=128 artifact does not name which two records left, no
    later n=130 row can be reconciled with it.
    """
    import glob

    checked = 0
    for path in glob.glob(str(REPO / "artifacts" / "*.json")):
        try:
            data = json.loads(pathlib.Path(path).read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("skipped_by_reason", {}).get("compile:ValueError"):
            named = {r.get("id") for r in data.get("skipped_records") or []}
            assert set(GATE_REFUSED_IDS) <= named, (
                f"{pathlib.Path(path).name} reports n={data.get('n')} with "
                f"compile skips but does not name "
                f"{sorted(set(GATE_REFUSED_IDS) - named)}")
            checked += 1
    assert checked >= 3, (
        f"expected several published n=128 artifacts to check; found "
        f"{checked}. If they have moved, the denominator claim in the report "
        f"needs re-measuring.")


@requires_bfcl
def test_the_whole_population_passes_the_gate_after_the_fix():
    """`n` for a *full* BFCL-Live compile, measured rather than argued.

    Regime: `whitespace_pattern=schema.JSON_WS`, `allow_wildcard=True`,
    `allow=tasks.bfcl.ALLOW`, instance from `synthesize_instance`, key order
    ignored — i.e. exactly `tasks/bfcl.py`'s gate. Measured today:

        current normalisation   4,538 / 4,549 compile, 11 refused
        anyOf over 5 scalars    4,549 / 4,549
        anyOf over 7 types      4,549 / 4,549

    and the 11 are precisely `WILDCARD_RECORD_IDS`, each failing **all five**
    renderings. So the fix is worth 11 schemas whichever width is chosen, and
    no other schema in the benchmark changes state in either direction.

    SPEC §4.7's "4,549 / 4,549, zero failures" predates the build gate
    (`b58fd84`, 2026-08-08) and is stale: re-run today it is 4,538.

    Whole-population and still ~1 s, because this is the *regex* gate — the
    minutes-long part is the token lift, which nothing here performs.
    """
    from diffgemma_fa.compile import bfcl_data

    refused = []
    total = 0
    for split in bfcl_data.LIVE_SPLITS:
        for rec in bfcl_data.iter_split(split):
            for fn in rec.functions:
                total += 1
                params = fn.get("parameters")
                try:
                    norm = S.normalize_bfcl_schema(params)
                    inst = S.synthesize_instance(norm)
                    rx = S.build_regex(norm, allow=ALLOW,
                                       allow_wildcard=ALLOW_WILDCARD,
                                       whitespace_pattern=WS)
                    _, bad = S.accepts_all_renderings(rx, inst)
                    if [b for b in bad if b != "reordered-keys"]:
                        refused.append((rec.id, fn.get("name"), bad))
                except Exception as e:                     # noqa: BLE001
                    refused.append((rec.id, fn.get("name"),
                                    f"{type(e).__name__}: {e}"))
    assert total == 4549, f"population moved: {total}"
    assert not refused, (
        f"{len(refused)} of {total} BFCL-Live schemas still fail the build "
        f"gate: {refused[:5]}")


@pytest.mark.skipif(
    not os.environ.get("DGFA_SLOW"),
    reason="full lift + Valmari on a wildcard grammar; ~minutes and ~25 GB "
           "RSS. Set DGFA_SLOW=1 to run. NOT a silent cap: the gate-level "
           "test above covers the acceptance property, this one covers only "
           "whether the automaton is buildable at all.")
def test_the_full_pipeline_admits_a_wildcard_schema():
    """End to end, with `eval/run.py`'s exact keywords.

    The gate is a regex check; this is the only assertion that the *automaton*
    for a repaired wildcard schema can actually be built — the shape SPEC §4.7
    measured at 3,261 raw states and 36.2M transitions, 26-130x every other
    BFCL shape.
    """
    inst = S.synthesize_instance(REVERSE_INPUT, from_bfcl=True)
    report = pipeline.compile_json_schema(
        REVERSE_INPUT, name="reverse_input", from_bfcl=True,
        allow=ALLOW, allow_wildcard=ALLOW_WILDCARD, whitespace_pattern=WS,
        verify_renderings=inst, verify_strict=True,
        nonempty_required_strings=True)
    a = report.automaton
    assert a.n_states_bucket <= 2048, (
        f"|S|={a.n_states} lands in bucket {a.n_states_bucket}, above the "
        f"ladder's top; needs_chain_path={a.needs_chain_path}")
    assert not a.needs_chain_path


# ===========================================================================
# §7  Vacuity guards. CLAUDE.md: six vacuous tests in this codebase in a week,
#     all asserting an implementation back to itself. These check that the
#     probes above can *distinguish* things.
# ===========================================================================

def test_the_probe_documents_are_what_the_schema_permits():
    """Independent check of this file's own expectations, via `jsonschema`.

    Every document `test_p1_floor_*` demands be accepted is validated against
    the *translated* schema by a third-party validator. If jsonschema says a
    probe is invalid, the test above is demanding something wrong and the bug
    is here, not in the grammar.
    """
    jsonschema = pytest.importorskip(
        "jsonschema", reason="third-party check of this file's expectations")

    for fixture, (schema, rec_id, prop) in REAL_SCHEMAS.items():
        norm = S.normalize_bfcl_schema(schema)
        # The wildcard property is `any`: every JSON value satisfies it, so
        # drop the marker and validate the *rest* of the object.
        norm = json.loads(json.dumps(norm).replace('"__wildcard__": true',
                                                   '"__any__": true'))
        norm.pop("__any__", None)
        stripped = {k: v for k, v in norm.items() if k != "__wildcard__"}
        stripped["properties"] = {
            k: {kk: vv for kk, vv in v.items()
                if kk not in ("__wildcard__", "description")}
            for k, v in stripped["properties"].items()}
        for json_type in SCALAR_FLOOR:
            inst = instance_with(schema, prop, TYPE_PROBES[json_type])
            jsonschema.validate(inst, stripped)   # raises if this file is wrong


def test_the_leak_probes_are_not_trivially_rejected_by_every_grammar():
    """A `reject` assertion is worthless if nothing could ever accept it.

    `test_grammar_rejects_a_bare_scalar_as_a_whole_answer` asserts that `1` is
    rejected. That would also hold for a grammar accepting the empty language.
    So: the *same* documents must be accepted by a grammar that genuinely
    permits them, proving the probes are well-formed and the regex dialect
    behaves as this file assumes.
    """
    from outlines_core.json_schema import build_regex_from_schema

    bare_any = build_regex_from_schema(
        json.dumps({"anyOf": [{"type": t} for t in ALL_JSON_TYPES]}),
        whitespace_pattern=WS)
    for leak in ("1", "null", '"say hi"', "1.5", "[1]", "[]", "true"):
        assert accepts(bare_any, leak), (
            f"{leak!r} is not accepted even by a grammar for a bare JSON "
            f"value; the probe is malformed, not the grammar under test")


def test_the_accept_assertions_are_not_trivially_satisfied():
    """The mirror image: an `accept` assertion is worthless against `.*`.

    A grammar that accepted everything would pass every P1 test in this file.
    So each P1 probe document must be *rejected* by a grammar that genuinely
    forbids it — here, the same schema with the wildcard pinned to `boolean`.
    """
    from outlines_core.json_schema import build_regex_from_schema

    narrow = build_regex_from_schema(
        json.dumps({"type": "object",
                    "properties": {"input_value": {"type": "boolean"}},
                    "required": ["input_value"]}),
        whitespace_pattern=WS)
    assert accepts(narrow, '{"input_value": true}')
    for json_type in ("string", "integer", "number", "null"):
        doc = json.dumps({"input_value": TYPE_PROBES[json_type]})
        assert not accepts(narrow, doc), (
            f"a boolean-only grammar accepts {doc}; the P1 probes cannot "
            f"distinguish widths")


def test_top_level_alternation_scanner_is_correct():
    """The one helper in this file with any logic, checked against cases whose
    answers are obvious by inspection."""
    yes = ["a|b", r"(x)|(y)", r"[abc]|d", r"\(|\)", r"(a(b|c))|d"]
    no = [r"(a|b)", r"[a|b]", r"\|", r"(?:a|b)c", r"a[|]b", r"((a|b)|c)",
          r"\\|"[:-1] + r"\\"]
    for r in yes:
        assert top_level_alternation(r), f"missed a depth-0 `|` in {r!r}"
    for r in no:
        assert not top_level_alternation(r), f"false positive on {r!r}"


# ===========================================================================
# Mutation harness. CLAUDE.md's `DGFA_MUT` convention (see
# tests/test_audit_partition.py:350). Each mutation breaks the **shipped**
# implementation in a specific, plausible way; the suite must go red, and it
# must go red on the tests that are *about* that property rather than on
# everything at once.
#
#   DGFA_MUT=revert              stop expanding -> the pre-fix language
#   DGFA_MUT=narrow5             declared five-way (a legitimate alternative)
#   DGFA_MUT=undeclared          seven-way grammar, declaration says five
#   DGFA_MUT=overfire            the reviewer's finding: `{"pattern": ...}`
#                                widens to any-JSON instead of raising
#   DGFA_MUT=flag_default_false  the safe behaviour becomes opt-in
#
# Caveat on the last one: it reproduces the *behaviour* of a flipped default,
# and the two behavioural tests catch it, but a `functools.partial` cannot
# convincingly fake a changed signature, so
# `test_the_wildcard_flag_defaults_to_the_safe_behaviour` stays green under it.
# That test reads the real default off `inspect.signature`, so an actual
# `expand_wildcard: bool = False` in the source does trip it — the gap is in
# the harness, not the assertion, which is why the behavioural twins exist.
#
# An earlier version of this block installed *candidate* implementations,
# because it was written before the feature existed. It is now obsolete twice
# over — it patched a signature that has since gained `expand_wildcard`, which
# made the whole suite red for the wrong reason and hid what the mutations were
# supposed to show.
# ===========================================================================

def _apply_mutation(name: str) -> None:      # pragma: no cover - tooling only
    import functools

    if name == "revert":
        # The pre-fix language: stop expanding, keep everything else.
        for mod, fn in ((S, "normalize_bfcl_schema"), (S, "build_regex"),
                        (pipeline, "compile_json_schema")):
            original = getattr(mod, fn)
            setattr(mod, fn, functools.partial(original, expand_wildcard=False))

    elif name == "narrow5":
        # A *declared* narrowing to the five scalars. Only the width test and
        # the cost/equivalence evidence should notice.
        S.WILDCARD_ANYOF_TYPES = tuple(SCALAR_FLOOR)

    elif name == "undeclared":
        # The grammar stays seven-way; the published declaration says five.
        # The exact case CLAUDE.md's "never silently cap coverage" is about,
        # inverted: a declaration that understates what ships.
        real = tuple(S.WILDCARD_ANYOF_TYPES)
        original = S._expand_wildcards

        def patched(node, *, path=""):
            S.WILDCARD_ANYOF_TYPES = real
            try:
                return original(node, path=path)
            finally:
                S.WILDCARD_ANYOF_TYPES = tuple(SCALAR_FLOOR)

        S._expand_wildcards = patched
        S.WILDCARD_ANYOF_TYPES = tuple(SCALAR_FLOOR)

    elif name == "overfire":
        # The reviewer's blocking finding, reinstated: treat *any* node with no
        # precedence keyword as a wildcard, so `{"pattern": "^a$"}` silently
        # becomes any-JSON instead of raising.
        original = S._typeless_kind
        S._typeless_kind = lambda node: (original(node)[0], [])

    elif name == "flag_default_false":
        for mod, fn in ((S, "normalize_bfcl_schema"), (S, "build_regex"),
                        (pipeline, "compile_json_schema")):
            original = getattr(mod, fn)
            wrapper = functools.partial(original, expand_wildcard=False)
            wrapper.__signature__ = inspect.Signature(
                [p.replace(default=False) if p.name == WILDCARD_FLAG else p
                 for p in inspect.signature(original).parameters.values()])
            setattr(mod, fn, wrapper)

    else:
        raise SystemExit(
            f"unknown DGFA_MUT={name!r}; expected one of revert, narrow5, "
            f"undeclared, overfire, flag_default_false")


if os.environ.get("DGFA_MUT"):
    _apply_mutation(os.environ["DGFA_MUT"])
