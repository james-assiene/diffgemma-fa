"""Evaluation runner. SPEC §7, CLAUDE.md's documented CLI.

    python -m diffgemma_fa.eval.run --task bfcl_live --variant j0 --emission map

SPEC §7.2's baselines, all reachable through `--variant`:

| `--variant` | what it is |
|---|---|
| `unconstrained` | stock DiffusionGemma. Accuracy **and CS** |
| `mask` | **naive per-position masking** — the wrong-but-obvious baseline §2.8 targets; its CS is the headline evidence |
| `j2` | constrained draw at accepted positions, uniform random elsewhere |
| `j0` | decoupled trajectory/emission (`--emission=map` or `sample`) |
| `j1` | single joint draw with flattened marginals at non-accepted |

**Every run reports CS beside the content columns**, never alone: SPEC §3.8
warns that CS can be trivially 100% while measuring nothing, and Phase 4
measured exactly that (`{"location":""}` for every prompt).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

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
from diffgemma_fa.eval import metrics  # noqa: E402
from diffgemma_fa.model.sampler import ConstrainedDiffusionSampler  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402

CKPT = "/home/ubuntu/diffgemma_fa/artifacts/ckpt/diffusiongemma-26B-A4B-it"
ALLOW = ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength",
         "first_match_wins", "additionalProperties")

TASKS = {
    "bfcl_live": bfcl_data.LIVE_SPLITS,
    "bfcl_live_simple": ("BFCL_v4_live_simple.json",),
    "bfcl_nonlive": bfcl_data.NON_LIVE_SPLITS,
}


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


def build_prompt(rec, fn) -> str:
    question = (rec.question[0][0]["content"] if rec.question
                else "Call the function.")
    props = list((fn.get("parameters") or {}).get("properties", {}))
    return (f"{question}\n\nRespond with a JSON object of arguments for "
            f"`{fn.get('name')}`, keys in this order: {props}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="bfcl_live_simple", choices=sorted(TASKS))
    ap.add_argument("--variant", default="j1",
                    choices=["unconstrained", "mask", "j0", "j1", "j2"])
    ap.add_argument("--emission", default="sample", choices=["map", "sample"])
    ap.add_argument("--entropy-bound", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--nonempty", action="store_true", default=True)
    ap.add_argument("--no-nonempty", dest="nonempty", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model = diffusion.DiffusionGemma_26B_A4B()
    params = gm.ckpts.load_params(CKPT)
    base = diffusion.Sampler(model=model, params=params)
    tok = base.tokenizer
    print(f"[eval] task={args.task} variant={args.variant} "
          f"emission={args.emission} bound={args.entropy_bound}", flush=True)

    splits = TASKS[args.task]
    records = [r for s in splits for r in bfcl_data.iter_split(s)
               if len(r.functions) == 1]
    if args.n:
        records = records[: args.n]
    truth = {}
    for s in splits:
        truth.update(bfcl_data.load_possible_answers(s))

    sc = metrics.Scores()
    rows, skipped = [], {}
    t_start = time.perf_counter()

    for idx, rec in enumerate(records):
        fn = rec.functions[0]
        try:
            a = pipeline.compile_json_schema(
                fn["parameters"], name=fn.get("name", ""), from_bfcl=True,
                allow=ALLOW, allow_wildcard=True,
                nonempty_required_strings=args.nonempty).automaton
        except Exception as e:  # noqa: BLE001
            k = f"compile:{type(e).__name__}"
            skipped[k] = skipped.get(k, 0) + 1
            continue

        sampler = ConstrainedDiffusionSampler(
            model=model,
            end_tokens=(tok.special_tokens.EOS, tok.special_tokens.END_OF_TURN,
                        tok.special_tokens.BEGIN_OF_TOOL_RESPONSE),
            forbidden_tokens=None, sampling=base.sampling,
            cache_length=base.cache_length, special_tokens=tok.special_tokens,
            canvas_length=256, max_denoising_steps=48,
            text_vocab_size=tok.vocab_size,
            sliding_window_size=getattr(model.config, "sliding_window_size", None),
            n_states_bucket=a.n_states_bucket, n_classes=a.tables.n_classes,
            variant=args.variant, emission=args.emission,
            sample_from_predictions=ds.SampleFromPredictions(
                entropy_bound=args.entropy_bound,
                text_vocab_size=tok.vocab_size),
        )

        inputs = base._get_inputs(prompt=build_prompt(rec, fn), images=None,  # noqa: SLF001
                                  add_bos=True, has_batch_dim=False,
                                  sharding=None)
        init = _prefill.prefill(
            model=model, params=params, input=inputs, last_state=None,
            cache_length=base.cache_length, pad_length=base.pad_length,
            rng=jax.random.PRNGKey(args.seed * 10_000 + idx), sharding=None,
            max_out_length=base.max_out_length)

        state = sampler.sample_constrained(
            params=params, init_state=init,
            max_new_tokens=args.max_new_tokens,
            automaton=to_traced(a, batch=init.predicted_tokens.shape[0]))
        jax.block_until_ready(state.predicted_tokens)

        toks = [int(x) for x in np.asarray(state.predicted_tokens)[0][
            : args.max_new_tokens]]
        while toks and toks[-1] == 0:
            toks.pop()
        text = tok.decode(toks)

        accepted = Simulator(a).accepts(toks)
        parsed = metrics.extract_json(text)
        gt = truth.get(rec.id, [])
        want = (bfcl_data.materialize_ground_truth(
            gt[0].get(fn.get("name"), {}), fn["parameters"]) if gt else None)

        sc.add(accepted=accepted, parsed_obj=parsed, want=want)
        rows.append({"id": rec.id, "fn": fn.get("name"), "text": text[:220],
                     "parsed": parsed, "accepted": accepted})

        if (idx + 1) % 10 == 0:
            d = sc.as_dict()
            print(f"  [{idx+1}/{len(records)}] CS={d['cs_rate']:.3f} "
                  f"acc={d['arg_accuracy']:.3f} "
                  f"nonempty={d['nonempty_rate']:.3f}", flush=True)

    out = {
        "task": args.task, "variant": args.variant, "emission": args.emission,
        "entropy_bound": args.entropy_bound, "nonempty_strings": args.nonempty,
        "seed": args.seed,
        "records_available": len(records),
        "skipped_by_reason": skipped,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        **sc.as_dict(),
        "per_key": sc.detail,
        "rows": rows,
    }
    path = args.out or (
        f"/home/ubuntu/diffgemma_fa/artifacts/eval_{args.task}_"
        f"{args.variant}_{args.emission}_s{args.seed}.json")
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n===== RESULT =====")
    for k in ("task", "variant", "emission", "n", "cs", "cs_rate", "parsed",
              "nonempty", "nonempty_rate", "arg_correct", "arg_total",
              "arg_accuracy", "exact_calls", "exact_call_rate",
              "skipped_by_reason", "elapsed_seconds"):
        print(f"{k}: {out[k]}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
