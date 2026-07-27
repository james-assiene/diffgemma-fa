"""Phase 0 — multi-block behaviour and the channel format.

The 20-prompt §1.4 baseline used max_new_tokens=256, which is exactly one
canvas, so it says nothing about the cross-block path that SPEC §3.5 and §5.7
warn "first shows up on block 2". This forces several blocks and additionally
dumps raw token ids, because the emitted format turned out to be channel-tagged
(`<|channel>` = id 100, `<channel|>` = id 101) rather than the single
end-of-thought marker §3.6 assumes.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

from gemma import diffusion, gm  # noqa: E402
from gemma.diffusion import _early_stopping, _sampler  # noqa: E402

CKPT = "/home/ubuntu/diffgemma_fa/artifacts/ckpt/diffusiongemma-26B-A4B-it"
STEP_LOG: list[dict[str, Any]] = []


def _record(step, n_accepted, mean_h):
    STEP_LOG.append(
        {"step": int(step), "n_accepted": int(n_accepted), "mean_h": float(mean_h)}
    )


@dataclasses.dataclass(frozen=True)
class Instrumented(_sampler.SampleFromPredictions):
    def __call__(self, *, rng, denoiser_logits, canvas, current_noise_proportion,
                 target_noise_proportion):
        logits32 = denoiser_logits.astype(jnp.float32)
        lp = jax.nn.log_softmax(logits32)
        p = jnp.exp(lp)
        h = -jnp.sum(jnp.where(p == 0, 0.0, lp) * p, axis=-1)
        si = jnp.argsort(h, axis=-1)
        se = jnp.take_along_axis(h, si, axis=-1)
        n_acc = jnp.sum((jnp.cumsum(se, axis=-1) - se) <= self.entropy_bound, axis=-1)[0]
        out = super().__call__(
            rng=rng, denoiser_logits=denoiser_logits, canvas=canvas,
            current_noise_proportion=current_noise_proportion,
            target_noise_proportion=target_noise_proportion,
        )
        jax.debug.callback(
            _record,
            jnp.round((1.0 - current_noise_proportion[0]) * 48.0),
            n_acc,
            jnp.mean(h),
        )
        return out


def split_blocks(log):
    blocks, cur = [], []
    for r in log:
        if cur and r["step"] <= cur[-1]["step"]:
            blocks.append(cur)
            cur = []
        cur.append(r)
    if cur:
        blocks.append(cur)
    return blocks


def main() -> None:
    model = diffusion.DiffusionGemma_26B_A4B()
    params = gm.ckpts.load_params(CKPT)
    tok = gm.text.Gemma4Tokenizer()
    print("[load] done", flush=True)

    prompts = [
        "Write a detailed 600-word essay on the history of the printing press.",
        "Explain, at length and with worked examples, how quicksort works, "
        "then how mergesort works, then compare them.",
        "List 40 distinct countries, each with its capital city and a one-sentence fact.",
    ]

    results = []
    for idx, prompt in enumerate(prompts):
        STEP_LOG.clear()
        sampler = diffusion.ChatSampler(
            model=model,
            params=params,
            sample_from_predictions=Instrumented(entropy_bound=0.1),
            max_out_length=2048,
        )
        t0 = time.perf_counter()
        out = sampler.chat(prompt, max_new_tokens=1024)
        wall = time.perf_counter() - t0

        blocks = split_blocks(list(STEP_LOG))
        ids = [int(x) for x in tok.encode(out)]
        rec = {
            "idx": idx,
            "prompt": prompt,
            "wall_seconds": round(wall, 2),
            "n_blocks_observed": len(blocks),
            "steps_per_block": [len(b) for b in blocks],
            "exit_per_block": [
                "budget" if len(b) >= 48 else "early_stop" for b in blocks
            ],
            "final_accepted_per_block": [b[-1]["n_accepted"] for b in blocks],
            "non_accepted_final_per_block": [256 - b[-1]["n_accepted"] for b in blocks],
            "output_chars": len(out),
            "output_token_ids_len": len(ids),
            "first_40_token_ids": ids[:40],
            "channel_open_positions": [i for i, t in enumerate(ids) if t == 100],
            "channel_close_positions": [i for i, t in enumerate(ids) if t == 101],
            "output_head": out[:300],
            "output_tail": out[-200:],
        }
        results.append(rec)
        print(
            f"[{idx}] {wall:.1f}s blocks={len(blocks)} "
            f"steps={rec['steps_per_block']} exits={rec['exit_per_block']} "
            f"non_acc={rec['non_accepted_final_per_block']} chars={len(out)}",
            flush=True,
        )
        print(f"    channel_open={rec['channel_open_positions'][:8]} "
              f"channel_close={rec['channel_close_positions'][:8]}", flush=True)
        print(f"    head={out[:160]!r}", flush=True)

    with open("/home/ubuntu/diffgemma_fa/artifacts/phase0_multiblock.json", "w") as f:
        json.dump({"results": results}, f, indent=2)

    tot = [b for r in results for b in r["exit_per_block"]]
    print("\n===== SUMMARY =====")
    print(f"blocks_total: {len(tot)}")
    print(f"via_budget: {tot.count('budget')}  via_early_stop: {tot.count('early_stop')}")
    print(f"max_non_accepted_on_final_step: "
          f"{max(x for r in results for x in r['non_accepted_final_per_block'])}")


if __name__ == "__main__":
    main()
