"""JSON Schema -> regex, with a fail-loud pre-pass.  SPEC §4.2.

`outlines_core.json_schema.build_regex_from_schema` is the fastest path to a
working grammar, but its limitations are **severe and silent**: it drops
sibling keywords, ignores every numeric bound, and honours only the first
keyword in a fixed precedence order. A schema that is silently under-constrained
produces a grammar that accepts strings the benchmark will mark wrong, and
nothing downstream will notice.

So: everything outlines would drop raises here instead (SPEC §4.2's "fail
loud"), and the caller decides explicitly whether to relax it.

This module also carries the **BFCL type translation** that SPEC §4.8 does not
mention. BFCL's `function[].parameters` are *not* JSON Schema — they are a
Python-flavoured dialect (`dict`, `float`, `any`, `tuple`) with Java/JavaScript
types (`String`, `HashMap`, `ArrayList`, `long`, ...) leaking in from the
`simple_java` / `simple_javascript` splits. Measured over BFCL v4: 9,464 `dict`
and 1,690 `float` occurrences against 0 `object` / 0 `number`.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

__all__ = [
    "UnsupportedSchemaError",
    "BFCL_TYPE_MAP",
    "normalize_bfcl_schema",
    "check_supported",
    "build_regex",
    "JSON_WS",
    "accepts_all_renderings",
    "synthesize_instance",
    "unordered_object_regex",
]


class UnsupportedSchemaError(ValueError):
    """Raised for anything `outlines_core` would silently drop or mishandle."""

    def __init__(self, path: str, keyword: str, detail: str) -> None:
        self.path = path
        self.keyword = keyword
        self.detail = detail
        super().__init__(f"{path or '<root>'}: {keyword!r} — {detail}")


# --- BFCL dialect -> JSON Schema ------------------------------------------
#
# Measured type vocabulary across all BFCL v4 splits (occurrences):
#   string 21854 · dict 9464 · integer 4757 · boolean 3115 · float 1690
#   array 959 · any 199 · String 115 · tuple 66 · Array 13 · HashMap 7
#   long 7 · ArrayList 6 · Boolean 4 · double 1 · char 1
BFCL_TYPE_MAP: dict[str, str] = {
    # Python dialect
    "dict": "object",
    "float": "number",
    "tuple": "array",
    # already JSON Schema
    "string": "string",
    "integer": "integer",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "number": "number",
    "null": "null",
    # Java / JavaScript leakage (simple_java, simple_javascript splits)
    "String": "string",
    "char": "string",
    "long": "integer",
    "double": "number",
    "Boolean": "boolean",
    "Array": "array",
    "ArrayList": "array",
    "HashMap": "object",
}

#: BFCL's wildcard. Expands to a 7-way alternation over every JSON type and is
#: the realistic route to "regex too large" — Phase 0 measured the equivalent
#: `{}` schema at 16.7 s / 3,261 states / 36.2M transitions, i.e. 26-130x every
#: other shape. Callers must opt in.
WILDCARD_TYPES = frozenset({"any"})

#: Keys whose value is a **map of user-chosen names to schemas**, not a schema.
#: The keyword-drop in `normalize_bfcl_schema` must never be applied to these
#: keys' *keys*. Kept in sync with `check_supported`, which already recursed
#: correctly here -- the asymmetry is what hid the bug.
_NAME_KEYED = frozenset({"properties", "$defs", "definitions"})

#: **The whitespace pattern to use for JSON. Not a tuning knob.**
#:
#: RFC 8259 defines JSON whitespace as any run of space, tab, LF or CR between
#: structural tokens. Accepting exactly that means the grammar admits *every*
#: rendering of a given object rather than one chosen spelling — which is the
#: property that matters, because a grammar that rejects the model's own
#: rendering forces it off its plan at every value boundary.
#:
#: RFC 8259 §2: `ws = *( %x20 / %x09 / %x0A / %x0D )`. **Unbounded**, and that
#: is deliberate — see below.
#:
#: **[AUDIT-D1] This was `[ \t\n\r]{0,8}`, and the `{0,8}` was silently a
#: nesting-depth bound.** A pretty-printer emits `newline + indent*depth`, so a
#: bound on the whitespace *run* binds at `indent*depth >= 8` — and nothing in
#: the compiler ever related that constant to the schema's depth, so when it
#: bound, it bound silently. Measured against `{0,8}`:
#:
#:     flat 2-key object        5/5 renderings
#:     object with one array    4/5   (rejects indent-4: elements sit at depth 2)
#:     nesting depth 2          4/5   (rejects indent-4)
#:     nesting depth 4          3/5   (rejects indent-2 AND indent-4)
#:
#: BFCL v4 carries 959 `array` and 9,464 `dict` occurrences, so neither shape is
#: exotic: the original 5/5 measurement was taken on a *flat* schema and did not
#: generalise. This is the same failure family as the 0/130 whitespace bug (the
#: grammar rejects a rendering the model may choose, and the renormalised draw
#: then silently EXTENDS the value instead of failing: `600` -> `6000`).
#:
#: **The bound was not buying finiteness.** A Kleene star over a 4-character
#: class is one DFA state; `{0,8}` is a counter and costs eight. Measured
#: end-to-end through the token lift + Valmari minimisation (`|S|` after
#: minimisation, `channel_header=False`):
#:
#:     pattern        flat        one array     3-key BFCL     tree @L=256
#:     [ \t\n\r]{0,8}  98 (b128)  132 (b256)    155 (b256)     0.033-0.134 GB
#:     [ \t\n\r]{0,24} 226 (b256) 308 (b512)    347 (b512)     0.134-0.536 GB
#:     [ \t\n\r]*      34 (b64)    45 (b64)      59 (b64)      0.008 GB
#:
#: So the correct pattern is also the cheapest by a factor of ~3 in `|S|` and
#: ~4-16x in tree bytes — the widened *constant* (`{0,24}`, which would have
#: covered indent-4 to depth 5 and still failed at depth 6) is the one that
#: walks toward SPEC §7.3's `|S| ~ 512` cliff. Deriving the bound from the
#: schema's depth was considered and rejected for the same reason: it is more
#: machinery, it still has an edge, and it is strictly larger than `*`.
#:
#: For the record, on the flat 3-key schema this docstring was originally
#: measured on:
#:
#:     outlines' default   53 states, accepts 2/5
#:     hand-fitted        125 states, accepts 4/5   (misses tabs)
#:     {0,8}              137 states, accepts 5/5 FLAT ONLY (4/5 with an array)
#:     THIS (`*`)          <- fewer states than any of them, 5/5 at every depth
JSON_WS = r"[ \t\n\r]*"

#: Keywords `outlines_core` genuinely ignores.
#:
#: **[V-P5] SPEC §4.2's list is wrong on four entries.** It groups
#: `minLength`, `maxLength`, `minItems` and `maxItems` with the silently-dropped
#: keywords; measured against outlines-core 0.2.14, **all four are enforced** —
#: the generated regex rejects a too-short string and a too-long array. Only the
#: *numeric* bounds are actually dropped. Keeping the wrong four in this list is
#: not harmless: it forces callers to `allow`-list constraints that do work,
#: which desensitises the very signal the fail-loud pre-pass exists to give.
_SILENTLY_DROPPED = {
    "minimum": "numeric bounds are ignored; port llguidance's rx_int_range/rx_float_range",
    "maximum": "numeric bounds are ignored; port llguidance's rx_int_range/rx_float_range",
    "exclusiveMinimum": "numeric bounds are ignored",
    "exclusiveMaximum": "numeric bounds are ignored",
    "multipleOf": "numeric bounds are ignored",
    "patternProperties": "ignored entirely",
    "propertyNames": "ignored entirely",
    "uniqueItems": "ignored entirely",
    "not": "ignored entirely",
    "if": "ignored entirely",
    "then": "ignored entirely",
    "else": "ignored entirely",
}

#: Verified **enforced** by outlines-core 0.2.14, contra SPEC §4.2.
ENFORCED_LENGTH_KEYWORDS = frozenset(
    {"minLength", "maxLength", "minItems", "maxItems"})

# First-match-wins precedence. If a schema carries one of these AND a
# later-precedence sibling, the sibling is silently discarded.
_PRECEDENCE = (
    "properties",
    "allOf",
    "anyOf",
    "oneOf",
    "prefixItems",
    "enum",
    "const",
    "$ref",
    "type",
)


def require_nonempty_strings(schema: dict[str, Any]) -> dict[str, Any]:
    """Give every **required** string property `minLength: 1`.

    Phase 5 measured that free-form string arguments collapse to `""` while
    *enum* arguments come out correct — the empty string is a self-consistent
    fixed point that the constrained sampler is right to allow and the model is
    happy to confirm. BFCL schemas essentially never carry `minLength`, so the
    grammar faithfully permits it.

    This is a **compiler-side** decision, not a schema fidelity one. It is a
    deliberate, opt-in over-constraint, and the cost is measured rather than
    assumed: across BFCL-Live, **exactly two** ground-truth answers are a
    required string that the benchmark accepts as empty —
    `live_multiple_507-149-4` (`origin_airport = ['']`) and
    `live_multiple_834-178-9` (`track = ['', 'Borbena']`). Those two records
    become unanswerable under `--nonempty`, so any results table using it must
    say so (CLAUDE.md: never silently cap coverage). Since `minLength` is
    genuinely enforced by outlines (contra SPEC §4.2), it is the right lever.

    Applied only to `required` properties, and only at the top level, so it
    cannot silently over-constrain optional or nested fields.
    """
    out = dict(schema)
    if "properties" not in schema:
        # Do NOT create one. `properties` outranks `type` in outlines'
        # precedence, so writing an empty map onto `{"type": "object"}` turns a
        # free-form object regex into `\{()?[ ]?\}` -- a grammar accepting only
        # `{}`. Unreachable on BFCL (0/6,226 functions lack `properties`), but
        # it is a language change, not a no-op.
        return out
    props = dict(out.get("properties") or {})
    required = set(out.get("required") or [])
    for key in list(props):
        if key not in required:
            continue
        sub = props[key]
        if isinstance(sub, dict) and sub.get("type") == "string" \
                and "enum" not in sub and "minLength" not in sub:
            props[key] = {**sub, "minLength": 1}
    out["properties"] = props
    return out


def normalize_bfcl_schema(node: Any, *, path: str = "") -> Any:
    """Rewrite a BFCL parameter schema into JSON Schema, recursively.

    Translates the type vocabulary via `BFCL_TYPE_MAP`, drops BFCL's `default`
    and `optional` annotations (which are documentation, not constraints), and
    leaves everything else untouched.

    Raises:
      UnsupportedSchemaError: on an unrecognised type name, so a new dialect
        leaking into the data set is a loud failure rather than a silently
        mistyped grammar.
    """
    if isinstance(node, list):
        return [normalize_bfcl_schema(v, path=f"{path}[{i}]") for i, v in enumerate(node)]
    if not isinstance(node, dict):
        return node

    # BFCL routinely writes an *element* enum at the array level:
    #
    #   {"type": "array", "items": {"type": "string"}, "enum": ["view", ...]}
    #
    # Read literally that says the whole array must equal one of those strings,
    # which is unsatisfiable — and because outlines' precedence puts `enum`
    # above `type`, it compiles to exactly that, silently requiring
    # `"metrics":"view"` where the reference answer is `"metrics":["view"]`.
    # Measured on `live_simple`, this single pattern caused 21 of 23 remaining
    # ground-truth rejections. The intent is unambiguous, so push the enum down
    # into `items` where it belongs.
    if (isinstance(node.get("type"), str)
            and node["type"] in ("array", "tuple")
            and isinstance(node.get("enum"), list)):
        node = dict(node)
        enum_values = node.pop("enum")
        items = dict(node.get("items") or {})
        items.setdefault("enum", enum_values)
        node["items"] = items

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _NAME_KEYED and isinstance(value, dict):
            # A **map of names to schemas**. Recurse into the values only: at
            # this level the dict's keys are user-chosen parameter names, and
            # applying the keyword-drop below to them silently DELETES any
            # parameter literally named `description`, `default` or `optional`.
            #
            # Measured on BFCL v4: 41 such parameters, 13 of them in `required`.
            # `live_multiple_998-229-0` (`create_website_alert_config`) compiled
            # to a grammar that *rejects* the required `description` argument and
            # *accepts* its omission -- an over-constraint, which validate.py's
            # docstring rightly calls the far more dangerous direction, producing
            # well-formed JSON that can never score. It survived Phase 1 because
            # the round-trip was measured on `live_simple` only, which has zero
            # occurrences; `check_supported` recurses correctly here, so the
            # fail-loud pre-pass could not see the asymmetry either.
            out[key] = {
                name: normalize_bfcl_schema(sub, path=f"{path}.{key}.{name}")
                for name, sub in value.items()
            }
            continue
        if key in ("default", "optional", "description"):
            # `default` and `optional` are not JSON Schema constraints and
            # outlines ignores `description`; dropping them keeps the regex
            # smaller and the behaviour identical.
            continue
        if key == "type" and isinstance(value, str):
            if value in WILDCARD_TYPES:
                # BFCL's `any` has no JSON Schema spelling — the equivalent is
                # *omitting* `type`, which outlines expands to its 7-way
                # alternation over all JSON types. Passing the literal string
                # through instead makes outlines raise
                # `ValueError: Unsupported type: any`; measured on BFCL-Live,
                # that killed 11 of 4,549 schemas. `check_supported` still sees
                # the wildcard via the marker below and can refuse it.
                out["__wildcard__"] = True
                continue
            if value not in BFCL_TYPE_MAP:
                raise UnsupportedSchemaError(
                    path, "type", f"unrecognised BFCL type {value!r}"
                )
            out["type"] = BFCL_TYPE_MAP[value]
            continue
        if key == "type" and isinstance(value, list):
            mapped = []
            for v in value:
                if v not in BFCL_TYPE_MAP:
                    raise UnsupportedSchemaError(
                        path, "type", f"unrecognised BFCL type {v!r} in union"
                    )
                mapped.append(BFCL_TYPE_MAP[v])
            out["type"] = mapped
            continue
        out[key] = normalize_bfcl_schema(value, path=f"{path}.{key}" if path else key)
    return out


def check_supported(
    node: Any,
    *,
    path: str = "",
    allow: Iterable[str] = (),
    allow_wildcard: bool = False,
) -> None:
    """Raise on anything `outlines_core` would silently drop.

    Args:
      node: a (already normalized) JSON Schema fragment.
      path: dotted path, for error messages.
      allow: keywords to tolerate despite being dropped. Use this to accept a
        known, *reported* loss of constraint — never to silence a surprise.
      allow_wildcard: permit `{}` / missing `type` / `additionalProperties:
        true` / BFCL `any`, which expand to a 7-way alternation over all JSON
        types. Phase 0 measured this shape at 16.7 s and 36.2M transitions
        against 0.13-0.65 s for every other shape, so it is opt-in.

    Raises:
      UnsupportedSchemaError
    """
    allow = set(allow)

    if isinstance(node, list):
        for i, v in enumerate(node):
            check_supported(v, path=f"{path}[{i}]", allow=allow,
                            allow_wildcard=allow_wildcard)
        return
    if not isinstance(node, dict):
        return

    for kw, why in _SILENTLY_DROPPED.items():
        if kw in node and kw not in allow:
            raise UnsupportedSchemaError(path, kw, why)

    # `additionalProperties: false` is deliberately NOT flagged.
    #
    # This used to raise with "`false` is not honoured; the grammar will accept
    # extra properties" — which is **inverted**. Measured against
    # outlines-core 0.2.14, a schema with `properties` and no
    # `additionalProperties` already rejects `{"a":"x","b":1}` and accepts
    # `{"a":"x"}`: closedness is the default, so `false` is redundant, not
    # dropped. Raising on it forced callers to put `additionalProperties` on the
    # `allow` list, which then also silenced the `true` case below — exactly the
    # desensitisation this module's docstring warns about ("never to silence a
    # surprise").

    # First-match-wins: flag a schema whose later-precedence siblings will be
    # thrown away.
    #
    # Dropping `type` is benign when the winning keyword already pins the value
    # set down completely — `properties`, `enum` and `const` all do. Both are
    # ubiquitous: BFCL v4 carries 9,405 `properties` and 5,786 `enum`, nearly
    # all of them alongside a redundant `type`, so treating this as an error
    # would reject essentially the whole benchmark for no gain in constraint.
    #
    # `enum`/`const` beating `type` is benign only for **scalar** types, where
    # the enum already pins the value set. For `array`/`object` it is not: an
    # array-level enum listing element values compiles to "the whole array
    # equals this string". `normalize_bfcl_schema` rewrites the BFCL spelling
    # of that; anything left here is genuinely ambiguous and must be flagged.
    present = [k for k in _PRECEDENCE if k in node]
    if len(present) > 1:
        winner, losers = present[0], present[1:]
        scalar = node.get("type") in (
            "string", "integer", "number", "boolean", "null", None
        )
        benign = losers == ["type"] and (
            winner == "properties" or (winner in ("enum", "const") and scalar)
        )
        if not benign and "first_match_wins" not in allow:
            raise UnsupportedSchemaError(
                path, winner,
                f"first-match-wins: {losers} will be silently dropped in favour "
                f"of {winner!r}",
            )

    if not allow_wildcard:
        t = node.get("type")
        if t in WILDCARD_TYPES or node.get("__wildcard__"):
            raise UnsupportedSchemaError(
                path, "type",
                "a wildcard over all JSON types; pass allow_wildcard=True "
                "to accept the ~26-130x compile cost (SPEC §4.2, §4.7)",
            )
        if node.get("additionalProperties") is True:
            raise UnsupportedSchemaError(
                path, "additionalProperties",
                "`true` expands to a 7-way alternation; pass allow_wildcard=True",
            )
        if not node and path:
            raise UnsupportedSchemaError(
                path, "{}", "empty schema is a wildcard; pass allow_wildcard=True"
            )

    for key, value in node.items():
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            for k, v in value.items():
                check_supported(v, path=f"{path}.{key}.{k}" if path else f"{key}.{k}",
                                allow=allow, allow_wildcard=allow_wildcard)
        elif key in ("items", "additionalProperties", "contains"):
            check_supported(value, path=f"{path}.{key}" if path else key,
                            allow=allow, allow_wildcard=allow_wildcard)
        elif key in ("allOf", "anyOf", "oneOf", "prefixItems") and isinstance(value, list):
            check_supported(value, path=f"{path}.{key}" if path else key,
                            allow=allow, allow_wildcard=allow_wildcard)


def _strip_markers(node: Any) -> Any:
    """Remove the internal `__wildcard__` marker before handing to outlines."""
    if isinstance(node, list):
        return [_strip_markers(v) for v in node]
    if not isinstance(node, dict):
        return node
    return {k: _strip_markers(v) for k, v in node.items() if k != "__wildcard__"}


def accepts_all_renderings(regex: str, instance: dict) -> tuple[bool, list[str]]:
    """Does `regex` accept every standard rendering of `instance`?

    **This is the check that turns "the grammar must accept the way the model
    writes" from advice into something enforceable.** It needs no model and no
    GPU: it renders one instance five ways that all mean the same thing and
    asserts the grammar takes all of them.

    Whitespace failures are reported as a hard problem; `reordered-keys` is
    reported too but is expected to fail on an ordered grammar — see
    `unordered_object_regex` for when that matters.

    It exists because the failure it catches cost 30 accuracy points and was
    invisible for weeks. The production grammar accepted **0 of 130** outputs
    the unconstrained model actually produced — it forbade newline-and-indent
    separators — and the symptom was not a crash but quietly corrupted values
    (`600` -> `6000`), because the renormalised draw extends a value when the
    separator it wants is inadmissible.

    Returns:
      `(ok, rejected)` — `rejected` names the renderings that failed, so the
      caller can report which one rather than just that something did.
    """
    import json as _json
    import re as _re

    # `ensure_ascii=False`: the grammar is a byte-level regex over the text the
    # model emits, and a tokenizer emits `실행`, not `실행`. Python's
    # default would spell a non-ASCII enum literal as an escape the regex has
    # never heard of and report a grammar defect that is really a renderer
    # artifact (it fired on 17 of BFCL-Live's 4,549 schemas). Identical output
    # for any ASCII instance, which is every instance in the test suite.
    def _dumps(**kw) -> str:
        return _json.dumps(instance, ensure_ascii=False, **kw)

    renderings = {
        "compact": _dumps(separators=(",", ":")),
        "spaced": _dumps(),
        "indent2": _dumps(indent=2),
        "indent4": _dumps(indent=4),
        "tabs": _dumps(indent="\t"),
    }
    # Key ORDER is checked too, and reported separately, because it is a
    # different kind of risk from whitespace. Whitespace variation is free to
    # admit; key-order independence costs 2^k states (measured: |S| 150 / 346 /
    # 738 / 1522 at k = 2/3/4/5, and 579 s to compile at k=5), which is past
    # SPEC §7.3's performance cliff. So this reports rather than fails: the
    # caller needs to know whether the restriction binds on THEIR model and
    # data before paying for `unordered_object_regex`.
    if len(instance) > 1:
        rev = {k: instance[k] for k in reversed(list(instance))}
        renderings["reordered-keys"] = _json.dumps(rev, ensure_ascii=False)
    bad = [k for k, v in renderings.items() if not _re.fullmatch(regex, v)]
    return (not bad), bad


#: Ceiling on `synthesize_instance` recursion. Bounded because a `$ref` cycle
#: would otherwise be an infinite descent; `$ref` is refused outright, so this is
#: belt-and-braces.
_SYNTH_MAX_DEPTH = 16

#: Keywords `synthesize_instance` cannot honour. It must produce an instance the
#: **grammar** accepts, so guessing past any of these would make the build gate
#: fire on a correct grammar — the one failure mode a gate must not have.
_SYNTH_UNSUPPORTED = ("$ref", "pattern", "format", "prefixItems", "allOf")


#: JSON type name -> the Python types `json.dumps` renders as it. `bool` is
#: excluded from the numeric rows deliberately: it is a subclass of `int` in
#: Python but `true` in JSON.
_JSON_PY_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _declared_type(node: dict[str, Any]) -> str | None:
    """The node's declared JSON type, first entry of a union, `None` if absent."""
    t = node.get("type")
    if isinstance(t, list):
        return t[0] if t else None
    return t if isinstance(t, str) else None


def _json_type_matches(value: Any, type_name: str | None) -> bool:
    """Would `value` satisfy a bare `{"type": type_name}`? `True` if unknown."""
    if type_name is None or type_name not in _JSON_PY_TYPES:
        return True
    if type_name in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _JSON_PY_TYPES[type_name])


def _synth(node: Any, *, path: str, depth: int) -> Any:
    if depth > _SYNTH_MAX_DEPTH:
        raise UnsupportedSchemaError(path, "$depth", "schema nests too deeply "
                                     "to synthesize a gate instance")
    if not isinstance(node, dict):
        raise UnsupportedSchemaError(path, "<node>", f"not a schema: {node!r}")

    for kw in _SYNTH_UNSUPPORTED:
        if kw in node:
            raise UnsupportedSchemaError(
                path, kw, "cannot synthesize an instance the grammar is known "
                          "to accept; the rendering gate would fire on a "
                          "correct grammar")

    if "const" in node:
        return node["const"]
    if isinstance(node.get("enum"), list) and node["enum"]:
        # `enum` outranks `type` in outlines' precedence, so the enum is what
        # the grammar admits and the enum is what the gate must render. Two
        # preferences, in order, neither of them a requirement:
        #
        #  * **type-consistent**, where the schema declares a `type` its own
        #    enum contradicts. Measured in review: 16 of BFCL-Live's 4,549
        #    schemas do (`{"type": "boolean", "enum": ["True", "False"]}`), and
        #    for those NO member is consistent -- the enum wins anyway, which is
        #    correct against the grammar and is stated in `synthesize_instance`.
        #  * **ASCII**. `json.dumps` defaults to `ensure_ascii=True`, which
        #    spells a non-ASCII literal as `\uXXXX` while outlines' regex
        #    carries the character itself. `accepts_all_renderings` renders with
        #    `ensure_ascii=False` for exactly that reason, but staying ASCII
        #    keeps the gate independent of that choice where it can. 17 schemas
        #    have an enum with no ASCII member at all (Korean `command`).
        members = node["enum"]
        typed = [c for c in members if _json_type_matches(c, _declared_type(node))]
        for pool in (typed, members):
            for cand in pool:
                if not isinstance(cand, str) or cand.isascii():
                    return cand
        return (typed or members)[0]
    for key in ("anyOf", "oneOf"):
        if isinstance(node.get(key), list) and node[key]:
            return _synth(node[key][0], path=f"{path}.{key}[0]", depth=depth + 1)

    t = _declared_type(node)

    # `properties` outranks `type` in outlines' precedence, so an object is an
    # object whatever `type` says.
    props = node.get("properties")
    if (isinstance(props, dict) and props) or t == "object":
        props = props if isinstance(props, dict) else {}
        required = {k for k in (node.get("required") or []) if k in props}
        keep = set(required)
        # Pad to two keys where possible: with a single key the rendering never
        # contains a `,` separator, and the separator is exactly where the
        # whitespace defect bites.
        for k in props:
            if len(keep) >= 2:
                break
            keep.add(k)
        return {k: _synth(props[k], path=f"{path}.{k}", depth=depth + 1)
                for k in props if k in keep}          # declared order: the
                                                      # grammar is ordered.
    if t == "array":
        items = node.get("items")
        n = max(int(node.get("minItems") or 0), 1)
        if node.get("maxItems") is not None:
            n = min(n, int(node["maxItems"]))
        if items is None:
            return [1] * n
        return [_synth(items, path=f"{path}[]", depth=depth + 1) for _ in range(n)]
    if t == "string":
        n = max(int(node.get("minLength") or 0), 1)
        if node.get("maxLength") is not None:
            n = min(n, int(node["maxLength"]))
        return "a" * n
    if t == "integer":
        return 1
    if t == "number":
        return 1
    if t == "boolean":
        return True
    if t == "null":
        return None
    if t is None or node.get("__wildcard__"):
        # The 7-way alternation over every JSON type; a number is in it.
        return 1
    raise UnsupportedSchemaError(path, "type", f"cannot synthesize {t!r}")


def synthesize_instance(
    schema: dict[str, Any], *, from_bfcl: bool = False
) -> dict:
    """A minimal instance the **grammar** should accept, for the build gate.

    **Not, in general, an instance `jsonschema` would validate.** The gate's job
    is to prove the compiled grammar admits a value the schema means to allow,
    and the grammar is outlines' first-match-wins reading of the schema, not the
    schema itself. Where the two disagree this follows outlines — otherwise the
    gate would report a defect the grammar does not have. Measured in review
    against `jsonschema`: **16 of BFCL-Live's 4,549** synthesized instances fail
    strict validation, every one of them a schema whose own `enum` contradicts
    its own `type` (`{"type": "boolean", "enum": ["True", "False"]}` yields
    `"True"`). `enum` outranks `type` in outlines' precedence, so `"True"` is
    exactly what that grammar accepts, and none of the 16 produced a gate
    failure. A type-consistent member is preferred where one exists; in these 16
    none does.

    `accepts_all_renderings` needs one concrete instance to render five ways.
    Producing it from the schema — rather than from an observed output — is what
    lets every production call site gate unconditionally (SPEC §4.2's fail-loud,
    CLAUDE.md's "an unwired gate is not a mitigation"). It needs no model, no
    data set and no GPU.

    Every required property is present, plus optionals up to two keys total so
    that the rendering contains a `,` separator; strings honour `minLength` /
    `maxLength`, arrays `minItems` / `maxItems`, and an `enum` contributes its
    first member that is type-consistent and ASCII, relaxing each of those two
    preferences in turn if no member satisfies them.

    Measured over all 4,549 BFCL-Live schemas: 0 failures to synthesize, and 11
    schemas whose grammar then rejects **all five** renderings of their own
    instance — a real defect the gate had never been given the chance to see.

    Raises:
      UnsupportedSchemaError: when no instance can be produced that the grammar
        is *known* to accept (`$ref`, `pattern`, `format`, `prefixItems`,
        `allOf`). Raising is deliberate: silently returning `None` would turn
        the gate off for exactly the schemas whose grammars are hardest to
        reason about, which is how the original defect survived.
    """
    if from_bfcl:
        schema = normalize_bfcl_schema(schema)
    value = _synth(schema, path="", depth=0)
    if not isinstance(value, dict):
        raise UnsupportedSchemaError(
            "", "type", "the gate needs an object instance; this schema's root "
                        f"synthesizes to {type(value).__name__}")
    return value


#: Hard ceiling on generated branches for `unordered_object_regex`. The number
#: of valid key sequences is `sum over subsets S containing required of |S|!`,
#: which grows fast enough that a cap is mandatory rather than tidy. 5,000
#: branches is a few hundred KB of regex, which outlines compiles in seconds.
MAX_PERMUTATION_BRANCHES = 5000


def unordered_object_regex(
    schema: dict,
    *,
    whitespace_pattern: str = JSON_WS,
    max_branches: int = MAX_PERMUTATION_BRANCHES,
) -> str | None:
    """An object regex accepting the keys in **any order**. SPEC §4.2 gap.

    **Why this exists.** `outlines_core` emits `properties` in map order only,
    so `{"b":1,"a":"x"}` is rejected for a schema declaring `a` first — even
    though JSON objects are unordered by definition (RFC 8259 §4: "An object is
    an unordered collection"). That is the same class of over-constraint as the
    whitespace bug, which cost 30 accuracy points by silently corrupting values
    rather than failing: when the separator or key the model wants is
    inadmissible, the renormalised draw extends whatever it is already writing.

    Measured on BFCL v4, the risk had not fired — 98/98 unconstrained outputs
    used the declared order — but the prompt *told* the model that order, so
    that number measures the prompt, not the model. A grammar reused without
    that sentence inherits the hazard. This removes it structurally.

    **Construction.** Enumerate every valid key sequence: each subset of
    properties that contains all `required` keys, in every order. Emit one
    alternation branch per sequence. The resulting regex is large but the
    *minimised* automaton is not — the DFA is the subset construction, `2^k`
    states, and `minimize.py` finds it.

    **Coverage.** 98.6% of BFCL v4's 6,226 schemas have <= 8 keys and 96.5%
    have <= 6. Beyond `max_branches` this returns `None` rather than emitting
    something quietly wrong, and the caller must decide — falling back to the
    ordered grammar is legitimate, doing so *silently* is not.

    Returns:
      The regex, or `None` if the schema is not a plain object of scalar-ish
      properties or the branch count exceeds `max_branches`.
    """
    import itertools
    from outlines_core.json_schema import build_regex_from_schema

    if schema.get("type") != "object":
        return None
    props = schema.get("properties") or {}
    if not props or any(k in schema for k in ("allOf", "anyOf", "oneOf", "$ref")):
        return None
    required = [k for k in schema.get("required") or [] if k in props]

    # Count first: building 40k branches to then discard them is the slow way
    # to discover a schema is too wide.
    n_opt = len(props) - len(required)
    total = 0
    for extra in range(n_opt + 1):
        import math
        total += (math.comb(n_opt, extra)
                  * math.factorial(len(required) + extra))
        if total > max_branches:
            return None
    if total == 0:
        return None

    ws = whitespace_pattern
    pair: dict[str, str] = {}
    for key, sub in props.items():
        try:
            val = build_regex_from_schema(json.dumps(sub),
                                          whitespace_pattern=ws)
        except Exception:  # noqa: BLE001
            return None
        pair[key] = f'{json.dumps(key)}{ws}:{ws}(?:{val})'

    branches: list[str] = []
    keys = list(props)
    for size in range(len(required), len(keys) + 1):
        for subset in itertools.combinations(keys, size):
            if not set(required) <= set(subset):
                continue
            for order in itertools.permutations(subset):
                branches.append(f"{ws},{ws}".join(pair[k] for k in order))
    if not branches:
        return None
    body = "|".join(f"(?:{b})" for b in branches)
    return rf"\{{{ws}(?:{body}){ws}\}}"


def build_regex(
    schema: dict[str, Any],
    *,
    from_bfcl: bool = False,
    allow: Iterable[str] = (),
    allow_wildcard: bool = False,
    whitespace_pattern: str | None = None,
) -> str:
    """Compile a schema to an anchored byte-level regex.

    Args:
      schema: JSON Schema, or a BFCL parameter block if `from_bfcl`.
      from_bfcl: run `normalize_bfcl_schema` first.
      allow: keywords to tolerate despite outlines dropping them.
      allow_wildcard: permit the 7-way-alternation shapes.
      whitespace_pattern: passed through to outlines; `""` forbids whitespace
        between tokens, which shrinks the automaton considerably.

    Returns:
      The regex string.
    """
    import outlines_core as oc  # local: keeps import cost off module load

    if from_bfcl:
        schema = normalize_bfcl_schema(schema)
    check_supported(schema, allow=allow, allow_wildcard=allow_wildcard)
    schema = _strip_markers(schema)

    kwargs = {}
    if whitespace_pattern is not None:
        kwargs["whitespace_pattern"] = whitespace_pattern
    return oc.json_schema.build_regex_from_schema(json.dumps(schema), **kwargs)
