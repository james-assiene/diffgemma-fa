"""Scoring. SPEC §7.2, §3.8, §4.8.

Three things are reported together and none substitutes for another:

- **CS** — constraint satisfaction, checked by an *independent* simulator over
  the compiled automaton. This is the paper's headline column.
- **accuracy** — BFCL's own notion, using **BFCL's own normalisation**
  (lowercase, strip `",./-_*^`, per SPEC §4.8). Scoring more strictly than the
  benchmark understates the result and is not comparable to the leaderboard.
- **content** — parse rate and non-empty-argument rate.

That last one exists because of SPEC §3.8's warning, sharpened by Phase 4:

> with a free-text branch … the model can spend its whole budget outside
> `FA_grammar`, making CS trivially 100% while measuring nothing.

A run that emits `{"location":""}` every time scores CS = 100% and is useless.
**Never report CS without the content columns beside it.**
"""

from __future__ import annotations

import dataclasses
import json
import re

__all__ = ["Scores", "normalise", "extract_json", "score_arguments",
           "schema_valid", "values_match"]

#: BFCL's value normalisation, transcribed from
#: `bfcl_eval/eval_checker/ast_eval/ast_checker.py::standardize_string`:
#:
#: ```python
#: regex_string = r"[ \,\.\/\-\_\*\^]"
#: return re.sub(regex_string, "", input_string).lower().replace("'", '"')
#: ```
#:
#: **[AUDIT-A] This used to strip `"` and keep the space, which is neither half
#: of BFCL's rule.** SPEC §4.8 quotes the docstring as "lowercases and strips
#: `",./-_*^`" — the outer `"` there is the quotation delimiter, not a member of
#: the set, and the set's *first* character is a space. The two errors point in
#: opposite directions and neither is safe:
#:
#: - keeping the space scores `NewYork` against `New York` as **wrong** where
#:   BFCL scores it right (an understatement `metrics.py` forbids);
#: - stripping the quote scores a literal `"black"` — quotes inside the JSON
#:   string value — as **right** where BFCL rejects it (manufactured accuracy).
#:
#: The single-quote fold (`'` -> `"`) was missing entirely.
_BFCL_STRIP = re.compile(r"[ \,\.\/\-\_\*\^]")


def normalise(value) -> str:
    """BFCL's `standardize_string`. **Strings only** — see `values_match`."""
    return _BFCL_STRIP.sub("", str(value)).lower().replace("'", '"')


def values_match(got, want) -> bool:
    """Is `got` the same argument value as `want`, the way BFCL decides it?

    **[AUDIT-A] BFCL normalises only strings.**
    `simple_function_checker` routes a parameter whose declared type is `str`
    through `string_checker` (which standardises), and everything else through a
    bare `value not in possible_answer[param]` — exact equality — after a
    `type_checker` that rejects a type mismatch outright.

    Applying the string rule to everything, as this module used to, deletes the
    decimal point and the minus sign from `str(value)`: `1.0` and `10` become
    the same answer, as do `-5` and `5`. And `str(True).lower() == "true"` makes
    a model that emitted the *string* `"true"` indistinguishable from one that
    emitted the boolean — which BFCL's type check rejects before it ever looks
    at the value.

    `bool` is handled before `int`/`float` deliberately: `True == 1` in Python,
    so an `isinstance(x, int)` branch would silently accept a boolean for an
    integer parameter.
    """
    if isinstance(want, str) and isinstance(got, str):
        return normalise(got) == normalise(want)
    if isinstance(want, bool) or isinstance(got, bool):
        # `bool` only ever matches `bool`.
        return isinstance(want, bool) and isinstance(got, bool) and want == got
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        # BFCL's own "allow python auto conversion from int to float".
        return want == got
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(
            values_match(g, w) for g, w in zip(got, want))
    if isinstance(want, dict) and isinstance(got, dict):
        return (set(want) == set(got)
                and all(values_match(got[k], want[k]) for k in want))
    if want is None or got is None:
        return want is None and got is None
    return type(want) is type(got) and want == got


def extract_json(text: str) -> dict | None:
    """Pull the argument object out of a channel-tagged completion.

    The grammar carries SPEC §3.6's `<|channel>NAME\\n<channel|>` header, so the
    emission legitimately *begins* with it — naive splitting on `<` yields the
    empty string and scores everything unparsable.
    """
    body = text
    if "<channel|>" in body:
        body = body.split("<channel|>", 1)[1]
    start = body.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(body)):
        c = body[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(body[start:i + 1])
                except Exception:  # noqa: BLE001
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def schema_valid(obj: dict | None, normalised_schema: dict) -> bool:
    """Does the parsed object validate against the schema? SPEC §7.2.

    **Why this exists beside CS.** `cs_rate` asks whether the *emitted token
    sequence* is accepted by the compiled automaton, and that automaton carries
    SPEC §3.6's `<|channel>NAME\n<channel|>` header. The header is **our**
    addition — the stock model has no reason to emit it — so `cs_rate` scores
    the `unconstrained` and `mask` baselines at ~0 partly for a convention they
    were never asked to follow. Reporting a 0% -> 100% CS jump on that basis
    alone would overstate the result.

    This column asks the question the benchmark actually cares about, in a form
    every arm can be asked fairly: *is the argument object schema-conformant?*
    It is tokenizer-independent (so a constrained arm cannot be penalised for
    re-tokenisation ambiguity) and header-independent.

    Both are reported. `cs_rate` is the guarantee the sampler enforces;
    `schema_valid_rate` is the cross-arm comparison.

    Args:
      obj: the parsed object, or None if the output did not parse.
      normalised_schema: the schema **after** `normalize_bfcl_schema` — the raw
        BFCL dialect uses `dict`/`float`/`any`, which no JSON Schema validator
        understands.
    """
    if obj is None:
        return False
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - eval-only dependency
        raise
    schema = dict(normalised_schema)
    # `__wildcard__` is our own marker for BFCL's `any`; JSON Schema spells that
    # as an absent `type`, which is what `normalize_bfcl_schema` already did.
    schema.pop("__wildcard__", None)
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        # A schema our compiler accepts but the validator rejects is a fact
        # about the corpus, not about the sample. Surfaced, never swallowed.
        raise
    return True


@dataclasses.dataclass
class Scores:
    """Running totals for one configuration."""

    n: int = 0
    cs: int = 0                 # accepted by the independent simulator
    schema_ok: int = 0          # parsed object validates against the schema
    parsed: int = 0
    nonempty: int = 0
    arg_correct: int = 0
    arg_total: int = 0
    exact_calls: int = 0        # every argument correct
    detail: list = dataclasses.field(default_factory=list)

    def add(self, *, accepted: bool, parsed_obj: dict | None,
            want: dict | None, schema_ok: bool = False) -> None:
        """Record one benchmark record.

        **[AUDIT-B] `arg_total` must not depend on the emission.** This method
        used to `return` before touching `arg_total` when `parsed_obj is None`,
        so records that produced nothing were dropped from the denominator while
        staying in `n`. Measured over the same 130 `bfcl_live_simple` records:

            arm             parsed  arg_total  arg_accuracy  exact_call_rate
            mask               67       120       0.5083         0.2308
            j1_sample         128       287       0.3798         0.2846
            unconstrained     126       276       0.6413         0.4692

        `mask` — the arm that failed to produce a parsable object 63 times out
        of 130 — reported the **highest** accuracy and the lowest exact-call
        rate in the same row, because its 63 silent records were removed from
        its denominator and from nobody else's. Two arms were quoting the same
        column over different populations.

        The denominator is now `sum(len(want))` over every record that has a
        ground truth, whatever the model did, so `arg_accuracy` and
        `exact_call_rate` are over the same records. Records with `want=None`
        (no ground truth available — including the compile-skip, OOM and
        zero-partition rows the harnesses feed in) contribute to neither.
        """
        self.n += 1
        self.cs += int(accepted)
        self.schema_ok += int(schema_ok)
        if parsed_obj is not None:
            self.parsed += 1
            if any(v not in ("", None, [], {}) for v in parsed_obj.values()):
                self.nonempty += 1
        if want is None:
            return
        n_ok, n_tot, det = score_arguments(parsed_obj or {}, want)
        self.arg_correct += n_ok
        self.arg_total += n_tot
        self.exact_calls += int(n_tot > 0 and n_ok == n_tot)
        self.detail.extend(det)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "cs": self.cs,
            "cs_rate": round(self.cs / max(1, self.n), 4),
            "schema_ok": self.schema_ok,
            "schema_valid_rate": round(self.schema_ok / max(1, self.n), 4),
            "parsed": self.parsed,
            "nonempty": self.nonempty,
            "nonempty_rate": round(self.nonempty / max(1, self.n), 4),
            "arg_correct": self.arg_correct,
            "arg_total": self.arg_total,
            "arg_accuracy": round(self.arg_correct / max(1, self.arg_total), 4),
            "exact_calls": self.exact_calls,
            "exact_call_rate": round(self.exact_calls / max(1, self.n), 4),
        }


def score_arguments(pred: dict, want: dict) -> tuple[int, int, list]:
    """`(n_correct, n_expected, per_key)` under BFCL's own comparison.

    `n_expected` is `len(want)` — a property of the **benchmark**, never of the
    emission. `pred` may legitimately be `{}` (the record produced nothing);
    every key is then a miss, which is the honest answer and not an exclusion.
    """
    detail, n_ok = [], 0
    for key, value in want.items():
        got = pred.get(key)
        ok = key in pred and values_match(got, value)
        n_ok += int(ok)
        detail.append({"key": key, "want": str(value)[:80],
                       "got": str(got)[:80], "ok": ok})
    return n_ok, len(want), detail
