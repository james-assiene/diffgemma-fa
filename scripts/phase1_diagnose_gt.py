"""Why does the compiled FA reject some BFCL ground-truth answers?

The round-trip test reports the rate; this classifies the cause. Over-constraint
is the dangerous direction — the model's output stays well-formed while scoring
zero — so the categories here decide what Phase 1 must fix versus what is an
inherent property of the benchmark.
"""

from __future__ import annotations

import collections
import json
import sys

sys.path.insert(0, "/home/ubuntu/diffgemma_fa")

from diffgemma_fa.compile import bfcl_data, pipeline, schema as sm, validate
from diffgemma_fa.compile.vocab import END_TOKENS, gemma_tokenizer

ALLOW = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
         "multipleOf", "minItems", "maxItems", "minLength", "maxLength",
         "first_match_wins", "additionalProperties")


def classify(obj: dict, params: dict, text: str, sim, tok) -> str:
    """Attribute a rejection to a cause, by narrowing the input."""
    props = sm.normalize_bfcl_schema(params).get("properties") or {}
    required = set((params.get("required") or []))

    extra = [k for k in obj if k not in props]
    if extra:
        return f"ground_truth_key_not_in_schema:{extra[0]}"

    missing = [k for k in required if k not in obj]
    if missing:
        return f"ground_truth_omits_required:{missing[0]}"

    # Does each property value match its own sub-schema in isolation?
    for k, v in obj.items():
        sub = props.get(k, {})
        t = sub.get("type")
        py = {"string": str, "integer": int, "number": (int, float),
              "boolean": bool, "array": list, "object": dict}.get(t)
        if py and not isinstance(v, py):
            if t == "integer" and isinstance(v, bool):
                return f"bool_where_integer_expected:{k}"
            return f"type_mismatch:{k}:{t}_vs_{type(v).__name__}"
        if sub.get("enum") is not None and v not in sub["enum"]:
            return f"value_not_in_enum:{k}"

    # Narrow to the longest accepted prefix to find the offending token.
    ids = [int(x) for x in tok.encode(text)]
    states = sim.start_states()
    for i, tid in enumerate(ids):
        nxt = sim.step(states, tid)
        if not nxt:
            piece = tok.decode([tid])
            ctx = tok.decode(ids[max(0, i - 4):i + 1])
            return f"dies_at_token:{piece!r}:ctx={ctx!r}"
        states = nxt
    return "prefix_ok_but_not_accepted"


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    tok = gemma_tokenizer()
    records = list(bfcl_data.iter_split("BFCL_v4_live_simple.json"))[:limit]
    truth = bfcl_data.load_possible_answers("BFCL_v4_live_simple.json")

    causes: collections.Counter = collections.Counter()
    examples: dict[str, dict] = {}
    n_gt = n_ok = 0

    for rec in records:
        if len(rec.functions) != 1:
            continue
        fn = rec.functions[0]
        params = fn.get("parameters") or {}
        try:
            rep = pipeline.compile_json_schema(
                params, name=fn.get("name", ""), from_bfcl=True,
                allow=ALLOW, allow_wildcard=True)
        except Exception:  # noqa: BLE001
            continue
        sim = validate.Simulator(rep.automaton)
        prop_order = list((params.get("properties") or {}).keys())

        for entry in truth.get(rec.id, []):
            args = entry.get(fn.get("name", ""))
            if args is None:
                continue
            obj = bfcl_data.materialize_ground_truth(args, params)
            ordered = {k: obj[k] for k in prop_order if k in obj}
            ordered.update({k: v for k, v in obj.items() if k not in ordered})
            text = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)
            ids = [int(x) for x in tok.encode(text)]
            n_gt += 1
            if sim.accepts(ids + [END_TOKENS[0]]):
                n_ok += 1
                continue
            cause = classify(ordered, params, text, sim, tok)
            key = cause.split(":")[0]
            causes[key] += 1
            examples.setdefault(key, {"id": rec.id, "fn": fn.get("name"),
                                      "detail": cause, "text": text[:180]})

    print(f"ground truth: {n_ok}/{n_gt} accepted ({100*n_ok/max(1,n_gt):.1f}%)\n")
    print("REJECTION CAUSES")
    for k, c in causes.most_common():
        print(f"  {c:4}  {k}")
    print("\nEXAMPLES")
    for k, ex in examples.items():
        print(f"\n[{k}] {ex['id']} {ex['fn']}")
        print(f"   {ex['detail'][:160]}")
        print(f"   {ex['text'][:160]}")


if __name__ == "__main__":
    main()
