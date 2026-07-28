"""Phase 4 exit criterion — end-to-end constrained generation. SPEC §8.

> End-to-end constrained generation on 10 prompts. Diagnostics: **stop-token
> position per block**, marker position per block, non-accepted count on the
> final step.

Drives the real 26B-A4B checkpoint with a real compiled BFCL-Live grammar and
checks the emitted canvas against an independent simulator — the guarantee is
worthless if only the sampler believes it.
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
from gemma.gm.text import _prefill  # noqa: E402

from diffgemma_fa.compile import bfcl_data, pipeline  # noqa: E402
from diffgemma_fa.compile.validate import Simulator  # noqa: E402
from diffgemma_fa.compile.vocab import END_TOKENS  # noqa: E402
from diffgemma_fa.model.sampler import ConstrainedDiffusionSampler  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402

CKPT = "/home/ubuntu/diffgemma_fa/artifacts/ckpt/diffusiongemma-26B-A4B-it"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--emission", default="map", choices=["map", "sample"])
    ap.add_argument("--variant", default="j1", choices=["j1", "j2"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="/home/ubuntu/diffgemma_fa/artifacts/phase4_e2e.json")
    args = ap.parse_args()

    print("[load] model + params", flush=True)
    model = diffusion.DiffusionGemma_26B_A4B()
    params = gm.ckpts.load_params(CKPT)
    base = diffusion.Sampler(model=model, params=params)
    tok = base.tokenizer
    print(f"[load] done; cache_length={base.cache_length} "
          f"max_out_length={base.max_out_length}", flush=True)

    records = list(bfcl_data.iter_split("BFCL_v4_live_simple.json"))[: args.n]
    results = []

    for idx, rec in enumerate(records):
        fn = rec.functions[0]
        try:
            rep = pipeline.compile_json_schema(
                fn["parameters"], name=fn.get("name", ""), from_bfcl=True,
                allow=("minimum", "maximum", "minItems", "maxItems",
                       "minLength", "maxLength", "first_match_wins",
                       "additionalProperties"),
                allow_wildcard=True)
        except Exception as e:  # noqa: BLE001
            results.append({"id": rec.id, "error": f"compile: {e}"})
            continue
        a = rep.automaton
        sim = Simulator(a)

        question = rec.question[0][0]["content"] if rec.question else "Call the function."
        prompt = (f"{question}\n\nRespond with a JSON object of arguments for "
                  f"`{fn.get('name')}`, keys in this order: "
                  f"{list((fn.get('parameters') or {}).get('properties', {}))}.")

        sampler = ConstrainedDiffusionSampler(
            model=model,
            end_tokens=(tok.special_tokens.EOS, tok.special_tokens.END_OF_TURN,
                        tok.special_tokens.BEGIN_OF_TOOL_RESPONSE),
            forbidden_tokens=None, sampling=base.sampling,
            cache_length=base.cache_length,
            special_tokens=tok.special_tokens,
            canvas_length=256, max_denoising_steps=48,
            text_vocab_size=tok.vocab_size,
            sliding_window_size=getattr(model.config, "sliding_window_size", None),
            n_states_bucket=a.n_states_bucket, n_classes=a.tables.n_classes,
            variant=args.variant, emission=args.emission,
        )

        inputs = base._get_inputs(prompt=prompt, images=None, add_bos=True,  # noqa: SLF001
                                  has_batch_dim=False, sharding=None)
        init_state = _prefill.prefill(
            model=model, params=params, input=inputs, last_state=None,
            cache_length=base.cache_length, pad_length=base.pad_length,
            rng=jax.random.PRNGKey(idx), sharding=None,
            max_out_length=base.max_out_length)

        aut = to_traced(a, batch=init_state.predicted_tokens.shape[0])
        t0 = time.perf_counter()
        state = sampler.sample_constrained(
            params=params, init_state=init_state,
            max_new_tokens=args.max_new_tokens, automaton=aut)
        jax.block_until_ready(state.predicted_tokens)
        wall = time.perf_counter() - t0

        toks = [int(x) for x in np.asarray(state.predicted_tokens)[0]]
        emitted = [t for t in toks[: args.max_new_tokens]]
        # trim the trailing PAD the post-loop mask leaves behind
        trimmed = list(emitted)
        while trimmed and trimmed[-1] == 0:
            trimmed.pop()

        stop_pos = next((i for i, t in enumerate(emitted) if t in END_TOKENS), None)
        text = tok.decode(trimmed)
        accepted = sim.accepts(trimmed)
        viable = bool(sim.run(trimmed))

        rec_out = {
            "idx": idx, "id": rec.id, "fn": fn.get("name"),
            "states": a.n_states, "bucket": a.n_states_bucket,
            "wall_seconds": round(wall, 2),
            "stop_token_position": stop_pos,
            "n_emitted": len(trimmed),
            "simulator_accepts": accepted,
            "prefix_viable": viable,
            "text": text[:300],
        }
        results.append(rec_out)
        print(f"[{idx:02d}] {fn.get('name')[:24]:26} S={a.n_states:4} "
              f"{wall:6.1f}s stop@{stop_pos} accepted={accepted} "
              f"-> {text[:70]!r}", flush=True)

    ok = [r for r in results if "error" not in r]
    summary = {
        "n": len(results),
        "compiled": len(ok),
        "accepted": sum(1 for r in ok if r["simulator_accepts"]),
        "prefix_viable": sum(1 for r in ok if r["prefix_viable"]),
        "stop_positions": [r["stop_token_position"] for r in ok],
        "emission": args.emission, "variant": args.variant,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        if k != "results":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
