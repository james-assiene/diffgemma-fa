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
