"""Phase 1 exit criterion — round-trip validation. SPEC §8.

Two directions:

  FA -> validator : random-walk each automaton, decode, and check with
                    `json.loads` (an independent parser).
  ground truth -> FA : run BFCL's own reference answers through the automaton.

The second is the one that matters. A grammar that is *over*-constrained still
emits well-formed JSON — it just scores zero, because the one string the
benchmark wanted is outside the language. Nothing but this test finds that.

Never silently caps coverage: every skip is counted and reported by reason.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time

sys.path.insert(0, "/home/ubuntu/diffgemma_fa")

from diffgemma_fa.compile import bfcl_data, pipeline, validate
from diffgemma_fa.compile.vocab import END_TOKENS, gemma_tokenizer

ALLOW = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minItems", "maxItems", "minLength", "maxLength", "first_match_wins",
    "additionalProperties",
)


def ground_truth_objects(gt_entry: dict, fn_name: str, schema: dict) -> list[dict]:
    """Materialise concrete argument dicts from a BFCL ground-truth entry.

    Delegates to `bfcl_data.materialize_ground_truth`, which unwraps BFCL's
    acceptable-values lists **recursively** — see its docstring; unwrapping only
    the top level makes the grammar look over-constrained when it is not.
    """
    args = gt_entry.get(fn_name)
    if args is None:
        return []
    return [bfcl_data.materialize_ground_truth(args, schema)]


def ordered_dump(obj: dict, prop_order: list[str]) -> str:
    """Serialise in the schema's property order, no whitespace.

    outlines emits `properties` in **map order only** (SPEC §4.2) — an
    over-constraint that rejects otherwise-valid documents with reordered keys.
    So the reference answer must be serialised in that same order to be a fair
    test of anything else. State the key order in the prompt at eval time.
    """
    ordered = {k: obj[k] for k in prop_order if k in obj}
    ordered.update({k: v for k, v in obj.items() if k not in ordered})
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="BFCL_v4_live_simple.json")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--samples-per-fa", type=int, default=20)
    ap.add_argument("--out", default="/home/ubuntu/diffgemma_fa/artifacts/phase1_roundtrip.json")
    args = ap.parse_args()

    tok = gemma_tokenizer()
    records = list(bfcl_data.iter_split(args.split))[: args.limit]
    truth = bfcl_data.load_possible_answers(args.split)
    print(f"[data] {len(records)} records, {len(truth)} ground-truth entries", flush=True)

    stats = collections.Counter()
    skips: collections.Counter = collections.Counter()
    failures: list[dict] = []
    per_fa = []

    t_start = time.perf_counter()
    for n, rec in enumerate(records):
        if len(rec.functions) != 1:
            skips["not_single_function"] += 1
            continue
        fn = rec.functions[0]
        params = fn.get("parameters") or {}
        try:
            rep = pipeline.compile_json_schema(
                params, name=fn.get("name", ""), from_bfcl=True,
                allow=ALLOW, allow_wildcard=True,
            )
        except Exception as e:  # noqa: BLE001
            skips[f"compile_failed:{type(e).__name__}"] += 1
            continue

        a = rep.automaton
        sim = validate.Simulator(a)
        stats["fa_compiled"] += 1

        # --- direction 1: FA -> json.loads -------------------------------
        samples = validate.sample_strings(
            a, n=args.samples_per_fa, max_len=180, seed=n
        )
        sample_bad = 0
        for s in samples:
            stats["samples"] += 1
            if not s.accepted:
                stats["sample_not_accepted"] += 1
                sample_bad += 1
                continue
            toks = list(s.tokens)
            while toks and toks[-1] in END_TOKENS:
                toks.pop()
            try:
                json.loads(tok.decode(toks))
                stats["sample_valid_json"] += 1
            except Exception:  # noqa: BLE001
                stats["sample_invalid_json"] += 1
                sample_bad += 1

        # --- direction 2: ground truth -> FA -----------------------------
        gt = truth.get(rec.id)
        gt_ok = gt_total = 0
        if not gt:
            skips["no_ground_truth"] += 1
        else:
            prop_order = list((params.get("properties") or {}).keys())
            for entry in gt:
                for obj in ground_truth_objects(entry, fn.get("name", ""), params):
                    text = ordered_dump(obj, prop_order)
                    ids = [int(x) for x in tok.encode(text)]
                    gt_total += 1
                    stats["gt_total"] += 1
                    # A viable prefix is the right check: the arguments object
                    # is a *prefix* of the full string, which still owes a stop
                    # token to reach ACC (SPEC §6.4 - acceptance and viability
                    # are two different assertions).
                    reached = sim.run(ids)
                    if reached:
                        stats["gt_prefix_viable"] += 1
                        if sim.accepts(ids + [END_TOKENS[0]]):
                            stats["gt_accepted"] += 1
                            gt_ok += 1
                        else:
                            stats["gt_prefix_but_not_accepted"] += 1
                            failures.append({"id": rec.id, "fn": fn.get("name"),
                                             "text": text[:200],
                                             "why": "viable prefix, not accepted"})
                    else:
                        stats["gt_rejected"] += 1
                        failures.append({"id": rec.id, "fn": fn.get("name"),
                                         "text": text[:200], "why": "rejected"})

        per_fa.append({
            "id": rec.id, "fn": fn.get("name"),
            "states": a.n_states, "edges": a.n_edges,
            "classes": a.tables.n_classes,
            "dedup_ratio": round(a.tables.dedup_ratio, 3),
            "min_ratio": round(rep.minimization_state_ratio, 3),
            "is_dfa": bool(a.is_dfa),
            "k_max": a.tables.k_max,
            "seconds": round(rep.seconds_total, 3),
            "sample_bad": sample_bad,
            "gt_ok": gt_ok, "gt_total": gt_total,
        })

        if (n + 1) % 25 == 0:
            print(f"  [{n+1}/{len(records)}] fa={stats['fa_compiled']} "
                  f"gt {stats['gt_accepted']}/{stats['gt_total']} "
                  f"samples {stats['sample_valid_json']}/{stats['samples']}",
                  flush=True)

    elapsed = time.perf_counter() - t_start
    ok = [r for r in per_fa if r["is_dfa"]]
    summary = {
        "split": args.split,
        "records_considered": len(records),
        "elapsed_seconds": round(elapsed, 1),
        "stats": dict(stats),
        "skipped_by_reason": dict(skips),
        "n_dfa": len(ok),
        "n_nfa": len(per_fa) - len(ok),
        "states_max": max((r["states"] for r in per_fa), default=0),
        "states_mean": round(sum(r["states"] for r in per_fa) / max(1, len(per_fa)), 1),
        "dedup_ratio_mean": round(
            sum(r["dedup_ratio"] for r in per_fa) / max(1, len(per_fa)), 3),
        "min_ratio_mean": round(
            sum(r["min_ratio"] for r in per_fa) / max(1, len(per_fa)), 4),
        "min_ratio_max": max((r["min_ratio"] for r in per_fa), default=0),
        "k_max_max": max((r["k_max"] for r in per_fa), default=0),
        "failures_head": failures[:25],
        "n_failures": len(failures),
        "per_fa": per_fa,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        if k not in ("per_fa", "failures_head"):
            print(f"{k}: {v}")
    if failures:
        print("\nfirst failures:")
        for f_ in failures[:8]:
            print(f"  {f_['id']} {f_['fn']}: {f_['why']}\n     {f_['text'][:140]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
