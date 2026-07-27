"""Grammar for BFCL's Python call format. SPEC §4.8.

**No canonical grammar exists — this authors one**, and the choices are forced
by how BFCL actually scores:

- Parsing is Python's `ast`, and the scorer reads **only `elem.keywords`**.
  Positional arguments are silently discarded, which guarantees a
  `missing_required` failure. **So the grammar forbids positional args
  entirely.** This is the single most important constraint here.
- Dotted function names are supported (`module.func`).
- Brackets are auto-added if missing, so `[...]` around the call list is
  optional.
- Scoring lowercases and strips `",./-_*^"` from string values.

The format is **not regular**: `[] {} ()` nest arbitrarily, and `BinOp`/`Lambda`
reach the scorer through `eval`. So this constrains to a **bounded-depth**
subset — dotted name, keyword-only args, values drawn from
{string, number, bool, null, list, dict}, nesting depth ≤ `d` (default 2).
`coverage_report` measures what that bound costs against the ground-truth
answers, which SPEC requires be stated rather than silently capped.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Sequence

__all__ = ["PythonGrammarConfig", "build_call_regex", "build_calls_regex",
           "value_regex", "measure_depth"]

# A JSON-ish string body: no raw quote, no backslash, no control characters.
_STR_BODY = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt])*'
_INT = r"-?(?:0|[1-9][0-9]*)"
_NUM = _INT + r"(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_BOOL = r"(?:True|False)"
_NONE = r"None"
#: Python identifiers, optionally dotted, for the function name.
_DOTTED_NAME = r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*"


@dataclasses.dataclass(frozen=True)
class PythonGrammarConfig:
    """Knobs, all of which are reportable over-constraints.

    Attributes:
      max_depth: nesting bound for list/dict values. `d=2` per SPEC.
      allow_missing_brackets: accept a bare call as well as `[call]`, matching
        BFCL's auto-bracketing.
      space_after_comma: whether `, ` is required, optional, or forbidden.
      fixed_key_order: emit keyword arguments in signature order only. `k!`
        permutations is not tractable beyond ~5 params, so this is an
        over-constraint — **state the key order in the prompt** (the same
        caveat SPEC §4.2 records for outlines' `properties` map order).
    """

    max_depth: int = 2
    allow_missing_brackets: bool = True
    space_after_comma: bool = True
    fixed_key_order: bool = True


def _string_regex(enum: Sequence[Any] | None = None) -> str:
    if enum:
        alts = "|".join(re.escape(str(v)) for v in enum)
        return f'"(?:{alts})"'
    return f'"{_STR_BODY}"'


def value_regex(
    schema: dict[str, Any],
    *,
    depth: int,
    max_depth: int,
) -> str:
    """Regex for one argument value, at bounded nesting depth.

    Args:
      schema: a **normalized** (JSON Schema) parameter schema.
      depth: current nesting depth.
      max_depth: bound; containers stop recursing at this depth.
    """
    enum = schema.get("enum")
    if enum is not None:
        # Enum members may be of any scalar type; render each literally.
        alts = []
        for v in enum:
            if isinstance(v, bool):
                alts.append("True" if v else "False")
            elif v is None:
                alts.append("None")
            elif isinstance(v, (int, float)):
                alts.append(re.escape(str(v)))
            else:
                alts.append(f'"{re.escape(str(v))}"')
        return "(?:" + "|".join(alts) + ")"

    t = schema.get("type")
    if isinstance(t, list):
        return "(?:" + "|".join(
            value_regex({**schema, "type": x}, depth=depth, max_depth=max_depth)
            for x in t
        ) + ")"

    if t == "string":
        return _string_regex()
    if t == "integer":
        return _INT
    if t == "number":
        return _NUM
    if t == "boolean":
        return _BOOL
    if t == "null":
        return _NONE

    if t == "array":
        if depth >= max_depth:
            return r"\[\]"
        item = schema.get("items") or {}
        inner = value_regex(item, depth=depth + 1, max_depth=max_depth) if item \
            else _scalar_union()
        sep = r",[ ]?"
        return rf"\[(?:{inner}(?:{sep}{inner})*)?\]"

    if t == "object":
        if depth >= max_depth:
            return r"\{\}"
        props = schema.get("properties") or {}
        if props:
            entries = []
            for k, sub in props.items():
                v = value_regex(sub, depth=depth + 1, max_depth=max_depth)
                entries.append(rf'"{re.escape(k)}":[ ]?{v}')
            sep = r",[ ]?"
            # Fixed key order, same over-constraint as elsewhere.
            body = f"(?:{sep})?".join(f"(?:{e})?" for e in entries)
            return rf"\{{{body}\}}"
        v = _scalar_union()
        sep = r",[ ]?"
        entry = rf'"{_STR_BODY}":[ ]?{v}'
        return rf"\{{(?:{entry}(?:{sep}{entry})*)?\}}"

    # No type / `any`: a scalar union. Deliberately NOT the 7-way alternation
    # including containers - Phase 0 measured that shape at 26-130x the cost of
    # every other, and it is the realistic route to "regex too large".
    return _scalar_union()


def _scalar_union() -> str:
    return f'(?:{_string_regex()}|{_NUM}|{_BOOL}|{_NONE})'


def build_call_regex(
    function: dict[str, Any],
    *,
    config: PythonGrammarConfig | None = None,
    normalized: bool = False,
) -> str:
    """Regex for a single `name(key=value, ...)` call.

    Args:
      function: a BFCL function entry (`{name, description, parameters}`).
      config: grammar knobs.
      normalized: set if `parameters` is already JSON Schema (see
        `schema.normalize_bfcl_schema`); otherwise it is normalized here.

    Returns:
      An unanchored regex fragment.
    """
    from diffgemma_fa.compile.schema import normalize_bfcl_schema

    cfg = config or PythonGrammarConfig()
    params = function.get("parameters") or {}
    if not normalized:
        params = normalize_bfcl_schema(params)

    props: dict[str, Any] = params.get("properties") or {}
    required = list(params.get("required") or [])

    name = re.escape(function.get("name", ""))

    # Signature order: required first, then optional, both in declaration order.
    ordered = [k for k in props if k in required] + \
              [k for k in props if k not in required]

    sep = r",[ ]" if cfg.space_after_comma else r",[ ]?"
    pieces: list[str] = []
    for i, key in enumerate(ordered):
        v = value_regex(props[key], depth=0, max_depth=cfg.max_depth)
        kw = rf"{re.escape(key)}={v}"
        if key in required:
            pieces.append(rf"(?:{sep})?{kw}" if i else kw)
        else:
            # Optional args may be omitted entirely, separator and all.
            pieces.append(rf"(?:(?:{sep})?{kw})?" if i else rf"(?:{kw})?")

    args = "".join(pieces)
    return rf"{name}\({args}\)"


def build_calls_regex(
    functions: Sequence[dict[str, Any]],
    *,
    config: PythonGrammarConfig | None = None,
    max_calls: int = 4,
    normalized: bool = False,
) -> str:
    """Regex for BFCL's `[call, call, ...]` answer format.

    Args:
      functions: candidate functions. A `multiple` record offers several and
        the model picks; measured on BFCL v4, records carry 1-8+ functions.
      max_calls: bound on the call list for the `parallel` splits. Another
        reportable cap.
      normalized: see `build_call_regex`.

    Returns:
      An **anchored** regex.
    """
    cfg = config or PythonGrammarConfig()
    alts = "|".join(
        f"(?:{build_call_regex(f, config=cfg, normalized=normalized)})"
        for f in functions
    )
    one = f"(?:{alts})"
    sep = r",[ ]" if cfg.space_after_comma else r",[ ]?"
    listed = rf"\[{one}(?:{sep}{one}){{0,{max_calls - 1}}}\]"
    if cfg.allow_missing_brackets:
        return f"(?:{listed}|{one})"
    return listed


def measure_depth(value: Any) -> int:
    """Nesting depth of a decoded Python value (scalars are 0)."""
    if isinstance(value, dict):
        return 1 + max((measure_depth(v) for v in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((measure_depth(v) for v in value), default=0)
    return 0
