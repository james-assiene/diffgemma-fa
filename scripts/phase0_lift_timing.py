"""Phase 0 — is SPEC §4.7's "4-8 minutes per schema" extrapolation right?

§4.7 extrapolates one published measurement (132-261 s per schema at a 151k
vocab) to Gemma's 262k vocab and concludes BFCL-Live is a 4-7 day serial job,
making the vocabulary pre-filter the single highest-leverage optimisation in
Phase 1. That number decides the Phase 1 schedule, so measure it.

Schemas here are hand-written to span the shape of BFCL function signatures:
trivial, typical, nested, enum-heavy, array-heavy, and the "regex too large"
wildcard case of §4.2.
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import outlines_core as oc
from gemma import gm

SCHEMAS: dict[str, dict] = {
    "trivial_1_string": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    "typical_weather": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer"},
        },
        "required": ["location", "unit"],
    },
    "many_props_8": {
        "type": "object",
        "properties": {
            f"field_{i}": {"type": t}
            for i, t in enumerate(
                ["string", "integer", "number", "boolean",
                 "string", "integer", "number", "boolean"]
            )
        },
        "required": [f"field_{i}" for i in range(8)],
    },
    "nested_depth2": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "filters": {
                "type": "object",
                "properties": {
                    "min_price": {"type": "number"},
                    "brand": {"type": "string"},
                },
                "required": ["min_price", "brand"],
            },
        },
        "required": ["query", "filters"],
    },
    "array_of_strings": {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
        },
        "required": ["ids", "limit"],
    },
    "enum_heavy_20": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [f"action_{i}" for i in range(20)]},
            "target": {"type": "string"},
        },
        "required": ["action", "target"],
    },
    "array_of_objects": {
        "type": "object",
        "properties": {
            "orders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string"},
                        "qty": {"type": "integer"},
                    },
                    "required": ["sku", "qty"],
                },
            }
        },
        "required": ["orders"],
    },
    "wildcard_any": {  # SPEC §4.2: expands to a 7-way alternation over all types
        "type": "object",
        "properties": {"payload": {}},
        "required": ["payload"],
    },
}


def build_vocabulary(tok) -> oc.Vocabulary:
    sp = tok._sp  # noqa: SLF001
    vocab = oc.Vocabulary(int(tok.special_tokens.EOS), {})
    for i in range(sp.GetPieceSize()):
        if sp.IsControl(i) or sp.IsUnknown(i):
            continue
        piece = sp.IdToPiece(i)
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            b = bytes([int(piece[3:5], 16)])
        else:
            b = piece.replace("▁", " ").encode("utf-8")
            if not b:
                continue
        vocab.insert(b, i)
    return vocab


def main() -> None:
    tok = gm.text.Gemma4Tokenizer()
    t0 = time.perf_counter()
    vocab = build_vocabulary(tok)
    build_s = time.perf_counter() - t0

    rows = []
    for name, schema in SCHEMAS.items():
        row: dict = {"schema": name}
        try:
            t0 = time.perf_counter()
            regex = oc.json_schema.build_regex_from_schema(json.dumps(schema))
            row["regex_seconds"] = round(time.perf_counter() - t0, 4)
            row["regex_len"] = len(regex)

            t0 = time.perf_counter()
            index = oc.Index(regex, vocab)
            row["index_seconds"] = round(time.perf_counter() - t0, 3)

            t0 = time.perf_counter()
            trans = index.get_transitions()
            row["get_transitions_seconds"] = round(time.perf_counter() - t0, 3)

            row["num_states"] = len(trans)
            row["nnz"] = int(sum(len(v) for v in trans.values()))
            row["total_seconds"] = round(
                row["regex_seconds"] + row["index_seconds"]
                + row["get_transitions_seconds"], 3
            )

            # SPEC §4.4 Layer 1: how many DISTINCT label sets are there really?
            # The class-dedup ratio is SPEC open question 9 - unpublished.
            classes = {frozenset(toks.keys()) for toks in trans.values()}
            row["distinct_label_sets"] = len(classes)
            row["dedup_ratio"] = round(len(trans) / max(1, len(classes)), 2)
            row["max_label_set_size"] = max(len(c) for c in classes)
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)[:200]
        rows.append(row)
        print(json.dumps(row), flush=True)

    totals = [r["total_seconds"] for r in rows if "total_seconds" in r]
    summary = {
        "vocabulary_build_seconds": round(build_s, 2),
        "vocab_len": len(vocab),
        "n_schemas": len(rows),
        "total_seconds_mean": round(statistics.mean(totals), 3),
        "total_seconds_max": round(max(totals), 3),
        "total_seconds_min": round(min(totals), 3),
        "projected_bfcl_live_1351_serial_seconds": round(
            statistics.mean(totals) * 1351, 1
        ),
        "projected_bfcl_live_1351_serial_hours": round(
            statistics.mean(totals) * 1351 / 3600, 3
        ),
        "spec_4_7_claim_seconds_per_schema": "240-480 (4-8 min)",
        "rows": rows,
    }
    print()
    json.dump(summary, sys.stdout, indent=2)
    print()
    with open("/home/ubuntu/diffgemma_fa/artifacts/phase0_lift_timing.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
