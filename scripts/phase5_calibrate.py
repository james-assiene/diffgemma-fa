"""SPEC §3.4 — recalibrate the entropy bound, and diagnose the MAP minimality.

> `SampleFromPredictions` accepts the largest ascending-entropy prefix with
> `Σ_{i≤k} Hᵢ − max_{i≤k} Hᵢ ≤ 0.1` nats … **The defaults are calibrated for
> unconstrained entropies and will silently produce garbage.**
>
> Required: a calibration sweep … over
> `entropy_bound ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}` …, selecting for accuracy
> at fixed step budget. Report chosen values, the baseline's, and
> accepted-count-per-step curves for both.

The sweep also serves as the diagnosis Phase 4 left open: joint MAP emits
schema-valid but *empty* arguments (`{"location":""}`), and the entropy bound is
one of three live candidates (the others being MAP's length bias over a
variable-length language, and the prompt). `--diagnose` dumps the per-step
emission so the convergence — or lack of it — is visible directly.

Accuracy here is BFCL's own notion, softened: a predicted argument counts if its
value appears in that parameter's ground-truth acceptable list. Reported
alongside constraint satisfaction, never instead of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/diffgemma_fa")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402
from gemma import diffusion, gm  # noqa: E402
from gemma.diffusion import _sampler as ds  # noqa: E402
from gemma.gm.text import _prefill  # noqa: E402

from diffgemma_fa.compile import bfcl_data, pipeline  # noqa: E402
from diffgemma_fa.compile.validate import Simulator  # noqa: E402
from diffgemma_fa.compile.vocab import END_TOKENS  # noqa: E402
from diffgemma_fa.model.sampler import (  # noqa: E402
    DIAGNOSTICS, ConstrainedDiffusionSampler,
)
from diffgemma_fa.model.state import Automaton  # noqa: E402

CKPT = "/home/ubuntu/diffgemma_fa/artifacts/ckpt/diffusiongemma-26B-A4B-it"
ALLOW = ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength",
         "first_match_wins", "additionalProperties")

#: SPEC §3.4's grid. 0.1 is the stock default.
ENTROPY_BOUNDS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def to_traced(a, batch: int) -> Automaton:
    return Automaton(
        edge_src=jnp.asarray(a.edge_src), edge_dst=jnp.asarray(a.edge_dst),
        edge_class=jnp.asarray(a.edge_class),
        edge_valid=jnp.ones(a.n_edges, bool),
        csr_indices=jnp.asarray(a.tables.sum_indices),
        csr_indptr=jnp.asarray(a.tables.sum_indptr),
        is_neg=jnp.asarray(a.tables.sum_is_neg),
        d=jnp.asarray(a.d), is_final=jnp.asarray(a.is_final),
        active=jnp.broadcast_to(jnp.asarray(a.start_vector),
                                (batch, a.n_states_bucket)),
    )


#: BFCL's own value normalisation, per SPEC §4.8: "scoring lowercases and
#: strips `",./-_*^`". Using a stricter comparison understates accuracy — e.g.
#: a predicted `":black"` against a ground-truth `black` is a BFCL *match*.
_STRIP = '",./-_*^'


def normalise(v) -> str:
    s = str(v).strip().lower()
    for ch in _STRIP:
        s = s.replace(ch, "")
    return s


def score(pred: dict, truth_args: dict, schema: dict) -> tuple[int, int, list]:
    """`(n_correct, n_expected, per_key)` against BFCL's acceptable-value lists.

    Comparison uses BFCL's own normalisation so the number is comparable to the
    leaderboard's rather than to a stricter invention of ours.
    """
    want = bfcl_data.materialize_ground_truth(truth_args, schema)
    if not want:
        return 0, 0, []
    detail = []
    n_ok = 0
    for k, v in want.items():
        got = pred.get(k)
        ok = k in pred and normalise(got) == normalise(v)
        n_ok += int(ok)
        detail.append({"key": k, "want": str(v)[:60], "got": str(got)[:60],
                       "ok": ok})
    return n_ok, len(want), detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--bounds", default=",".join(str(b) for b in ENTROPY_BOUNDS))
    ap.add_argument("--emission", default="map", choices=["map", "sample"])
    ap.add_argument("--variant", default="j0", choices=["j0", "j1", "j2"])
    ap.add_argument("--nonempty", action="store_true",
                    help="give required string args minLength:1 (SPEC 4.2 says "
                         "minLength is dropped; measured, it is enforced)")
    ap.add_argument("--diagnose", action="store_true",
                    help="dump the per-step emission for the first record")
    ap.add_argument("--out", default="/home/ubuntu/diffgemma_fa/artifacts/phase5_calibrate.json")
    args = ap.parse_args()

    bounds = [float(x) for x in args.bounds.split(",")]

    print("[load] model + params", flush=True)
    model = diffusion.DiffusionGemma_26B_A4B()
    params = gm.ckpts.load_params(CKPT)
    base = diffusion.Sampler(model=model, params=params)
    tok = base.tokenizer
    print("[load] done", flush=True)

    records = [r for r in bfcl_data.iter_split("BFCL_v4_live_simple.json")
               if len(r.functions) == 1][: args.n]
    truth = bfcl_data.load_possible_answers("BFCL_v4_live_simple.json")

    rows = []
    for bound in bounds:
        agg = {"bound": bound, "n": 0, "accepted": 0, "parsed": 0,
               "nonempty_args": 0, "arg_correct": 0, "arg_total": 0,
               "steps": [], "examples": [], "predictions": [], "per_key": []}
        for idx, rec in enumerate(records):
            fn = rec.functions[0]
            try:
                a = pipeline.compile_json_schema(
                    fn["parameters"], name=fn.get("name", ""), from_bfcl=True,
                    allow=ALLOW, allow_wildcard=True,
                    nonempty_required_strings=args.nonempty).automaton
            except Exception:  # noqa: BLE001
                continue

            question = (rec.question[0][0]["content"] if rec.question
                        else "Call the function.")
            props = list((fn.get("parameters") or {}).get("properties", {}))
            prompt = (f"{question}\n\nRespond with a JSON object of arguments "
                      f"for `{fn.get('name')}`, keys in this order: {props}.")

            diag = args.diagnose and idx == 0
            sampler = ConstrainedDiffusionSampler(
                model=model,
                end_tokens=(tok.special_tokens.EOS,
                            tok.special_tokens.END_OF_TURN,
                            tok.special_tokens.BEGIN_OF_TOOL_RESPONSE),
                forbidden_tokens=None, sampling=base.sampling,
                cache_length=base.cache_length,
                special_tokens=tok.special_tokens,
                canvas_length=256, max_denoising_steps=48,
                text_vocab_size=tok.vocab_size,
                sliding_window_size=getattr(model.config, "sliding_window_size",
                                            None),
                n_states_bucket=a.n_states_bucket,
                n_classes=a.tables.n_classes,
                variant=args.variant, emission=args.emission,
                sample_from_predictions=ds.SampleFromPredictions(
                    entropy_bound=bound, text_vocab_size=tok.vocab_size),
                diagnose=diag,
            )

            inputs = base._get_inputs(prompt=prompt, images=None, add_bos=True,  # noqa: SLF001
                                      has_batch_dim=False, sharding=None)
            init_state = _prefill.prefill(
                model=model, params=params, input=inputs, last_state=None,
                cache_length=base.cache_length, pad_length=base.pad_length,
                rng=jax.random.PRNGKey(idx), sharding=None,
                max_out_length=base.max_out_length)

            if diag:
                DIAGNOSTICS.reset()
            aut = to_traced(a, batch=init_state.predicted_tokens.shape[0])
            state = sampler.sample_constrained(
                params=params, init_state=init_state, max_new_tokens=256,
                automaton=aut)
            jax.block_until_ready(state.predicted_tokens)

            toks = [int(x) for x in np.asarray(state.predicted_tokens)[0][:256]]
            while toks and toks[-1] == 0:
                toks.pop()
            text = tok.decode(toks)
            accepted = Simulator(a).accepts(toks)

            body = text.split("<")[0]
            parsed = None
            try:
                parsed = json.loads(body)
            except Exception:  # noqa: BLE001
                pass

            agg["n"] += 1
            agg["accepted"] += int(accepted)
            if isinstance(parsed, dict):
                agg["parsed"] += 1
                nonempty = any(v not in ("", None, [], {}) for v in parsed.values())
                agg["nonempty_args"] += int(nonempty)
                gt = truth.get(rec.id, [])
                if gt:
                    c, t, detail = score(parsed, gt[0].get(fn.get("name"), {}),
                                         fn["parameters"])
                    agg["arg_correct"] += c
                    agg["arg_total"] += t
                    agg["per_key"].extend(detail)
            # Every prediction is kept, so the metric can be re-derived offline
            # without another GPU run.
            agg["predictions"].append({"id": rec.id, "fn": fn.get("name"),
                                       "text": text[:200], "parsed": parsed,
                                       "accepted": bool(accepted)})
            if len(agg["examples"]) < 4:
                agg["examples"].append(text[:90])

            if diag:
                print(f"\n--- per-step trace, bound={bound}, {fn.get('name')} ---",
                      flush=True)
                for r in DIAGNOSTICS.rows:
                    e = r["emitted"]
                    cut = next((i for i, t in enumerate(e) if t in END_TOKENS),
                               len(e))
                    print(f"  step {r['step']:2} acc={r['n_accepted']:3} "
                          f"H={r['mean_entropy']:8.4f} "
                          f"emit={tok.decode(e[:cut])[:70]!r}", flush=True)
                agg["trace"] = [
                    {"step": r["step"], "n_accepted": r["n_accepted"],
                     "mean_entropy": r["mean_entropy"],
                     "emit": tok.decode(
                         r["emitted"][:next((i for i, t in enumerate(r["emitted"])
                                             if t in END_TOKENS),
                                            len(r["emitted"]))])[:120]}
                    for r in DIAGNOSTICS.rows]

        rows.append(agg)
        acc_rate = agg["arg_correct"] / max(1, agg["arg_total"])
        print(f"[bound={bound:<6}] n={agg['n']:3} CS={agg['accepted']}/{agg['n']} "
              f"parsed={agg['parsed']} nonempty={agg['nonempty_args']} "
              f"arg_acc={agg['arg_correct']}/{agg['arg_total']}={acc_rate:.3f}",
              flush=True)

    with open(args.out, "w") as f:
        json.dump({"variant": args.variant, "emission": args.emission,
                   "nonempty": args.nonempty, "rows": rows}, f, indent=2)

    print("\n===== SPEC §3.4 CALIBRATION =====")
    print(f"{'bound':>8} {'CS':>8} {'parsed':>7} {'nonempty':>9} {'arg_acc':>9}")
    for r in rows:
        print(f"{r['bound']:>8} {r['accepted']}/{r['n']:<6} {r['parsed']:>7} "
              f"{r['nonempty_args']:>9} "
              f"{r['arg_correct']}/{r['arg_total']:<5} "
              f"{r['arg_correct']/max(1,r['arg_total']):.3f}")
    print(f"\nstock default is 0.1\nwrote {args.out}")


if __name__ == "__main__":
    main()
