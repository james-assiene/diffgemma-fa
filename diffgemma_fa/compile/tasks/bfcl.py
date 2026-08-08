"""Compile the BFCL grammars. SPEC §4.7, §4.8.

    python -m diffgemma_fa.compile.tasks.bfcl --out artifacts/fa/bfcl/ --jobs $(nproc)

Embarrassingly parallel across schemas. Phase 1 measured ~0.45 s per schema
serially, so the whole of BFCL-Live is minutes on this box — the multi-day job
SPEC §4.7 originally projected does not exist, and the vocabulary pre-filter it
proposed is unnecessary.

Compilation is cached on `(schema_hash, tokenizer_hash, compiler_version)`.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import pathlib
import time
from typing import Any

from diffgemma_fa.compile import automaton as automaton_mod
from diffgemma_fa.compile import bfcl_data, pipeline
from diffgemma_fa.compile import schema as schema_mod

#: Keywords outlines silently drops. Tolerated here so the benchmark compiles,
#: but counted and reported — never silently.
ALLOW = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minItems", "maxItems", "minLength", "maxLength", "first_match_wins",
    "additionalProperties",
)

COMPILER_VERSION = "phase1"


def _job(task: tuple[str, str, int, dict, str]) -> dict[str, Any]:
    rec_id, fn_name, fn_index, params, out_dir = task
    t0 = time.perf_counter()
    try:
        # [AUDIT-D3] THE BUILD GATE, wired. It was written for the failure that
        # cost 30 accuracy points (a grammar accepting 0/130 real outputs) and
        # neither call site ever passed it an instance. The instance is
        # synthesized from the schema, so gating is unconditional: no model, no
        # data, milliseconds. A schema whose grammar rejects a standard
        # rendering of its own instance now shows up in this report's error
        # counts instead of being written to `--out` as if it were fine.
        rep = pipeline.compile_json_schema(
            params, name=fn_name, from_bfcl=True,
            allow=ALLOW, allow_wildcard=True,
            verify_renderings=schema_mod.synthesize_instance(
                schema_mod.normalize_bfcl_schema(params)),
        )
    except Exception as e:  # noqa: BLE001
        return {"id": rec_id, "fn": fn_name, "fn_index": fn_index,
                "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "seconds": time.perf_counter() - t0}

    a = rep.automaton
    if out_dir:
        path = pathlib.Path(out_dir) / f"{rec_id}__{fn_index}.npz"
        automaton_mod.save(a, str(path))

    return {
        "id": rec_id, "fn": fn_name, "fn_index": fn_index, "ok": True,
        "states": a.n_states, "bucket": a.n_states_bucket, "edges": a.n_edges,
        "classes": a.tables.n_classes,
        "dedup_ratio": round(a.tables.dedup_ratio, 4),
        "min_state_ratio": round(rep.minimization_state_ratio, 4),
        "is_dfa": bool(a.is_dfa),
        "k_max": a.tables.k_max,
        "max_neg_size": a.tables.max_neg_size,
        "nnz_sum": a.tables.nnz_sum, "nnz_max": a.tables.nnz_max,
        "tables_bytes": a.tables.nbytes,
        "tree_gb": round(a.tree_bytes / 1e9, 4),
        "seconds": round(rep.seconds_total, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="", help="directory for .npz artifacts; empty = do not write")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--splits", default="live", choices=["live", "nonlive", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default="artifacts/bfcl_compile_report.json")
    args = ap.parse_args()

    splits = {
        "live": bfcl_data.LIVE_SPLITS,
        "nonlive": bfcl_data.NON_LIVE_SPLITS,
        "both": bfcl_data.LIVE_SPLITS + bfcl_data.NON_LIVE_SPLITS,
    }[args.splits]

    records = bfcl_data.load(splits)
    if args.limit:
        records = records[: args.limit]

    if args.out:
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    tasks = [
        (r.id, fn.get("name", ""), i, fn.get("parameters") or {}, args.out)
        for r in records for i, fn in enumerate(r.functions)
    ]
    print(f"[bfcl] {len(records)} records -> {len(tasks)} schemas, jobs={args.jobs}",
          flush=True)

    t0 = time.perf_counter()
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            rows = []
            for n, row in enumerate(pool.imap_unordered(_job, tasks, chunksize=4)):
                rows.append(row)
                if (n + 1) % 200 == 0:
                    print(f"  {n+1}/{len(tasks)} "
                          f"({time.perf_counter() - t0:.0f}s wall)", flush=True)
    else:
        rows = [_job(t) for t in tasks]
    wall = time.perf_counter() - t0

    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    errs = collections.Counter(r["error"].split(":")[0] for r in bad)

    states = sorted(r["states"] for r in ok)
    buckets = collections.Counter(r["bucket"] for r in ok)
    summary = {
        "splits": list(splits),
        "n_records": len(records),
        "n_schemas": len(tasks),
        "n_ok": len(ok),
        "n_failed": len(bad),
        "error_kinds": dict(errs),
        "wall_seconds": round(wall, 1),
        "cpu_seconds_serial": round(sum(r["seconds"] for r in rows), 1),
        "jobs": args.jobs,
        "states_median": states[len(states) // 2] if states else None,
        "states_p90": states[int(0.9 * len(states))] if states else None,
        "states_max": states[-1] if states else None,
        "bucket_histogram": dict(sorted(buckets.items())),
        "n_dfa": sum(1 for r in ok if r["is_dfa"]),
        "n_nfa": sum(1 for r in ok if not r["is_dfa"]),
        "dedup_ratio_mean": round(sum(r["dedup_ratio"] for r in ok) / max(1, len(ok)), 4),
        "min_state_ratio_mean": round(
            sum(r["min_state_ratio"] for r in ok) / max(1, len(ok)), 4),
        "min_state_ratio_max": max((r["min_state_ratio"] for r in ok), default=0),
        "k_max_max": max((r["k_max"] for r in ok), default=0),
        "max_neg_size_max": max((r["max_neg_size"] for r in ok), default=0),
        "tree_gb_max": max((r["tree_gb"] for r in ok), default=0),
        "rows": rows,
    }

    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        if k != "rows":
            print(f"{k}: {v}")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
