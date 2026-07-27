"""Phase 1.1 — re-measure SPEC §4.7 on REAL BFCL schemas.

Phase 0 measured 0.13-0.65 s per schema on hand-written shapes and projected
BFCL-Live at ~0.9 h serially, with the explicit caveat that the schemas were
not drawn from BFCL. This replaces that projection with the real thing.

Reports the full distribution, not a mean: §4.7's decision (build the
vocabulary pre-filter or not) turns on the tail, and Phase 0 already showed one
schema shape that is 130x the median.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, "/home/ubuntu/diffgemma_fa")

from diffgemma_fa.compile import bfcl_data, schema as schema_mod, vocab as vocab_mod


def function_schema(fn: dict) -> dict:
    """The JSON Schema for one BFCL function's arguments."""
    return fn.get("parameters") or {"type": "dict", "properties": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--splits", default="live")
    ap.add_argument("--out", default="/home/ubuntu/diffgemma_fa/artifacts/phase1_bfcl_timing.json")
    args = ap.parse_args()

    splits = {
        "live": bfcl_data.LIVE_SPLITS,
        "nonlive": bfcl_data.NON_LIVE_SPLITS,
    }[args.splits]

    records = bfcl_data.load(splits)
    if args.limit:
        records = records[: args.limit]
    print(f"[data] {len(records)} records from {len(splits)} splits", flush=True)

    t0 = time.perf_counter()
    vocab = vocab_mod.build_vocabulary()
    print(f"[vocab] built in {time.perf_counter() - t0:.2f}s, {len(vocab)} tokens", flush=True)

    import outlines_core as oc

    rows = []
    fails: dict[str, int] = {}
    fail_examples: dict[str, str] = {}

    for n, rec in enumerate(records):
        for fi, fn in enumerate(rec.functions):
            raw = function_schema(fn)
            row = {"id": rec.id, "split": rec.split, "fn_index": fi,
                   "fn_name": fn.get("name", "")}
            try:
                t0 = time.perf_counter()
                # allow_wildcard: BFCL really does contain `any`; refusing it
                # would drop records rather than measure them. `allow` covers
                # the bounds outlines drops - reported below, never silent.
                rx = schema_mod.build_regex(
                    raw, from_bfcl=True, allow_wildcard=True,
                    allow=("minimum", "maximum", "exclusiveMinimum",
                           "exclusiveMaximum", "multipleOf", "minItems",
                           "maxItems", "minLength", "maxLength",
                           "first_match_wins", "additionalProperties"),
                )
                row["regex_seconds"] = time.perf_counter() - t0
                row["regex_len"] = len(rx)

                t0 = time.perf_counter()
                index = oc.Index(rx, vocab)
                row["index_seconds"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                trans = index.get_transitions()
                row["get_transitions_seconds"] = time.perf_counter() - t0

                row["num_states"] = len(trans)
                row["nnz"] = int(sum(len(v) for v in trans.values()))
                row["total_seconds"] = (row["regex_seconds"] + row["index_seconds"]
                                        + row["get_transitions_seconds"])
                classes = {frozenset(t.keys()) for t in trans.values()}
                row["distinct_label_sets"] = len(classes)
                row["max_label_set_size"] = max(len(c) for c in classes) if classes else 0
            except Exception as e:  # noqa: BLE001
                kind = type(e).__name__
                msg = str(e)[:120]
                key = f"{kind}: {msg.split('—')[0].strip()[:80]}"
                fails[key] = fails.get(key, 0) + 1
                fail_examples.setdefault(key, f"{rec.id}/{fn.get('name','')}")
                row["error"] = f"{kind}: {msg}"
            rows.append(row)

        if (n + 1) % 100 == 0:
            ok = [r for r in rows if "total_seconds" in r]
            el = sum(r["total_seconds"] for r in ok)
            print(f"  [{n+1}/{len(records)}] {len(rows)} schemas, "
                  f"{len(ok)} ok, {el:.1f}s compile so far", flush=True)

    ok = [r for r in rows if "total_seconds" in r]
    tot = sorted(r["total_seconds"] for r in ok)

    def pct(p: float) -> float:
        if not tot:
            return float("nan")
        return round(tot[min(len(tot) - 1, int(p / 100 * len(tot)))], 4)

    summary = {
        "n_records": len(records),
        "n_schemas": len(rows),
        "n_compiled": len(ok),
        "n_failed": len(rows) - len(ok),
        "failure_kinds": fails,
        "failure_examples": fail_examples,
        "seconds_total_serial": round(sum(tot), 1),
        "seconds_mean": round(statistics.mean(tot), 4) if tot else None,
        "seconds_median": round(statistics.median(tot), 4) if tot else None,
        "seconds_p50": pct(50), "seconds_p90": pct(90),
        "seconds_p99": pct(99), "seconds_max": round(tot[-1], 4) if tot else None,
        "states_median": statistics.median(r["num_states"] for r in ok) if ok else None,
        "states_max": max(r["num_states"] for r in ok) if ok else None,
        "states_p90": sorted(r["num_states"] for r in ok)[int(0.9 * len(ok))] if ok else None,
        "nnz_median": statistics.median(r["nnz"] for r in ok) if ok else None,
        "max_label_set_size_max": max(r["max_label_set_size"] for r in ok) if ok else None,
        "min_complement_size": (
            len(vocab) - max(r["max_label_set_size"] for r in ok) if ok else None
        ),
        "rows": rows,
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        if k not in ("rows", "failure_examples"):
            print(f"{k}: {v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
