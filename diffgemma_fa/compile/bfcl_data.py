"""Loader for BFCL v4 records. SPEC §7.1.

The data lives in git (`ShishirPatil/gorilla`), **not** on HF — the HF mirror is
v3 only. v4 renamed `simple` -> `simple_python`.

Records are JSONL with `{id, question, function: [...]}`. `function[].parameters`
is a BFCL-dialect schema (see `schema.normalize_bfcl_schema`), and a record can
carry **1-8+ functions**, so a grammar for the `multiple` splits is a union over
them.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Iterator

DATA_DIR = pathlib.Path(
    "/home/ubuntu/diffgemma_fa/artifacts/data/gorilla/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

#: Splits scored as "Live" (SPEC §7.1). `live_irrelevance` and `live_relevance`
#: are scored separately — see SPEC §3.8's irrelevance trap.
LIVE_SPLITS = (
    "BFCL_v4_live_simple.json",
    "BFCL_v4_live_multiple.json",
    "BFCL_v4_live_parallel.json",
    "BFCL_v4_live_parallel_multiple.json",
)

NON_LIVE_SPLITS = (
    "BFCL_v4_simple_python.json",
    "BFCL_v4_multiple.json",
    "BFCL_v4_parallel.json",
    "BFCL_v4_parallel_multiple.json",
)

IRRELEVANCE_SPLITS = (
    "BFCL_v4_irrelevance.json",
    "BFCL_v4_live_irrelevance.json",
)


@dataclasses.dataclass(frozen=True)
class BfclRecord:
    """One BFCL entry."""

    id: str
    split: str
    question: list
    functions: tuple[dict, ...]

    @property
    def n_functions(self) -> int:
        return len(self.functions)


def iter_split(name: str, *, data_dir: pathlib.Path = DATA_DIR) -> Iterator[BfclRecord]:
    """Yield the records of one split file.

    Skips lines that are not valid JSON. `BFCL_v4_format_sensitivity.json` is
    not JSONL and yields nothing; every other v4 split parses cleanly.
    """
    path = data_dir / name
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield BfclRecord(
                id=raw.get("id", ""),
                split=name,
                question=raw.get("question", []),
                functions=tuple(raw.get("function", []) or ()),
            )


def load(splits: tuple[str, ...], *, data_dir: pathlib.Path = DATA_DIR) -> list[BfclRecord]:
    """Load several splits in order."""
    return [r for s in splits for r in iter_split(s, data_dir=data_dir)]


def materialize_ground_truth(
    args: dict,
    schema: dict | None = None,
) -> dict:
    """Turn one BFCL ground-truth argument block into a concrete value dict.

    BFCL's `possible_answer` format wraps **every leaf, at every nesting level**,
    in a list of acceptable values:

        {"body": {"airConJobMode": ["AIR_CLEAN"], "enabled": [true]}}
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ still wrapped, one level down

    Unwrapping only the top level leaves `["AIR_CLEAN"]` as the value of a
    parameter the schema declares `string`, which then looks like a grammar
    over-constraint when it is nothing of the sort. Measured on
    `live_simple`, that mistake accounted for most apparent rejections.

    `null` is BFCL's "this parameter was omitted", not a value — a schema
    declaring `type: string` is not violated by it, so those keys are dropped
    rather than emitted as `null`.

    Args:
      args: the per-function argument block from `possible_answer`.
      schema: the (BFCL-dialect) parameter schema, used to decide whether a
        list is a genuine `array` value or an acceptable-values wrapper.

    Returns:
      A concrete argument dict, taking the first acceptable value throughout.
    """
    props = (schema or {}).get("properties") or {}

    def unwrap(value, sub: dict | None):
        sub = sub or {}
        declared = sub.get("type")
        if isinstance(value, list):
            # A genuine array value keeps its list; an acceptable-values
            # wrapper is unwrapped. When the schema says `array`, a list of
            # lists is the wrapper and a flat list is the value.
            if declared in ("array", "tuple"):
                if value and all(isinstance(v, list) for v in value):
                    value = value[0]
                items = sub.get("items") or {}
                # `null` elements are BFCL's "omitted" marker one level down,
                # not a value the schema's `items` type is expected to admit.
                return [u for u in (unwrap(v, items) for v in value)
                        if u is not None]
            if not value:
                return None
            # `""` among the acceptable values means "this parameter may be
            # omitted", and omission is then *always* an acceptable answer — so
            # prefer it. Skipping past `""` to the next entry is wrong and not
            # merely suboptimal: `{"unit": ["", "N/A"]}` against an enum of
            # ["seconds", "milliseconds"] has omission as its ONLY valid
            # reading, because "N/A" is not in the enum. Measured on
            # live_simple, taking the second value there produced 12 of 24
            # apparent grammar rejections.
            if any(v == "" for v in value):
                return None
            return unwrap(value[0], sub)
        if isinstance(value, dict):
            inner = (sub.get("properties") or {}) if sub else {}
            out = {}
            for k, v in value.items():
                got = unwrap(v, inner.get(k))
                if got is not None:
                    out[k] = got
            return out
        if value == "":
            return None
        return value

    out: dict = {}
    for key, value in args.items():
        got = unwrap(value, props.get(key))
        if got is not None:
            out[key] = got
    return out


def load_possible_answers(name: str, *, data_dir: pathlib.Path = DATA_DIR) -> dict[str, list]:
    """Ground-truth answers for a split, keyed by record id.

    These are what Phase 1's round-trip test must show the compiled FA accepts
    (SPEC §8) — a grammar that rejects the reference answer is wrong regardless
    of how well-formed its output looks.
    """
    path = data_dir / "possible_answer" / name
    out: dict[str, list] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in raw:
                out[raw["id"]] = raw.get("ground_truth", [])
    return out
