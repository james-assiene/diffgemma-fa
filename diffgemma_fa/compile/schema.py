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

# Keywords outlines-core silently ignores (SPEC §4.2, maintainer-confirmed).
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
    "minItems": "ignored; cap at build time instead (SPEC §4.7)",
    "maxItems": "ignored; cap at build time instead (SPEC §4.7)",
    "minLength": "ignored; cap at build time instead (SPEC §4.7)",
    "maxLength": "ignored; cap at build time instead (SPEC §4.7)",
}

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

    if node.get("additionalProperties") is False and "additionalProperties" not in allow:
        raise UnsupportedSchemaError(
            path, "additionalProperties",
            "`false` is not honoured; the grammar will accept extra properties",
        )

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
