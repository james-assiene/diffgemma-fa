"""Phase 0 / SPEC §1.4 — smoke test and baseline instrumentation.

Loads the real checkpoint and generates unconstrained completions, capturing
the statistics §3.4 will later recalibrate against:

  per denoising step : n_accepted, mean/max entropy, sum(H) - max(H)
  per block          : non-accepted count on the FINAL EXECUTED step, and
                       whether the block exited via early stop or via the
                       48-step budget  (SPEC §1.2 invariant / open question 7)

The instrumentation rides on `jax.debug.callback`, which fires from inside the
`lax.while_loop` under `jit` — there is no Python between denoising steps
(SPEC §5.3), so nothing else would see these.
"""

from __future__ import annotations

import argparse
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

# Collected by the debug callbacks. Plain module state: the callbacks run on the
# host, in order, one per denoising step.
STEP_LOG: list[dict[str, Any]] = []
STOP_LOG: list[dict[str, Any]] = []


def _record_step(step, n_accepted, mean_h, max_h, sum_minus_max, n_changed):
    STEP_LOG.append({
        "step": int(step),
        "n_accepted": int(n_accepted),
        "mean_entropy": float(mean_h),
        "max_entropy": float(max_h),
        "sum_minus_max": float(sum_minus_max),
        "n_changed_vs_prev_canvas": int(n_changed),
    })


def _record_stop(step, should_stop, stability, entropy_mean):
    STOP_LOG.append({
        "step": int(step),
        "should_stop": bool(np.asarray(should_stop).ravel()[0]),
        "token_stability": bool(np.asarray(stability).ravel()[0]),
        "entropy_mean": float(entropy_mean),
    })


@dataclasses.dataclass(frozen=True)
class InstrumentedSampleFromPredictions(_sampler.SampleFromPredictions):
    """Stock behaviour, plus a host-side record of the accept statistics.

    Recomputes the accept mask exactly as the parent does (SPEC §1.2) rather
    than reaching into it, so this cannot drift from the stock rule.
    """

    def __call__(self, *, rng, denoiser_logits, canvas, current_noise_proportion,
                 target_noise_proportion):
        logits32 = denoiser_logits.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits32)
        probs = jnp.exp(log_probs)
        safe_log_probs = jnp.where(probs == 0, 0.0, log_probs)
        token_entropy = -jnp.sum(safe_log_probs * probs, axis=-1)  # [B, L]

        sorted_index = jnp.argsort(token_entropy, axis=-1)
        sorted_entropy = jnp.take_along_axis(token_entropy, sorted_index, axis=-1)
        accumulated = jnp.cumsum(sorted_entropy, axis=-1)
        sorted_mask = (accumulated - sorted_entropy) <= self.entropy_bound
        n_accepted = jnp.sum(sorted_mask, axis=-1)[0]

        out = super().__call__(
            rng=rng,
            denoiser_logits=denoiser_logits,
            canvas=canvas,
            current_noise_proportion=current_noise_proportion,
            target_noise_proportion=target_noise_proportion,
        )

        jax.debug.callback(
            _record_step,
            # `step` is not in scope here; noise_proportion encodes it exactly:
            # noise_proportions[step] = 1 - step/48.
            jnp.round((1.0 - current_noise_proportion[0]) * 48.0),
            n_accepted,
            jnp.mean(token_entropy),
            jnp.max(token_entropy),
            jnp.sum(token_entropy) - jnp.max(token_entropy),
            jnp.sum(out != canvas),
        )
        return out


@dataclasses.dataclass(frozen=True)
class InstrumentedEarlyStop(_early_stopping.EarlyStopFn):
    """The stock ChainedEarlyStop(TokenStability, Entropy), with a host record."""

    entropy_threshold: float = 0.005

    def should_stop(self, *, step, canvas, previous_canvas, logits):
        most_likely = jnp.argmax(logits, axis=-1)
        stability = jnp.all(most_likely == previous_canvas, axis=-1)

        log_probs = jax.nn.log_softmax(logits)
        probs = jnp.exp(log_probs)
        log_probs = jnp.where(probs == 0, 0.0, log_probs)
        entropy = jnp.mean(-jnp.sum(log_probs * probs, axis=-1), axis=-1)
        low_entropy = entropy <= self.entropy_threshold

        out = stability & low_entropy  # ChainedEarlyStop is AND (SPEC §1.2)
        jax.debug.callback(_record_stop, step, out, stability, entropy[0])
        return out


def summarise_blocks(step_log: list[dict[str, Any]], max_steps: int) -> list[dict]:
    """Split the flat step log into blocks and classify each block's exit."""
    blocks, cur = [], []
    for rec in step_log:
        if cur and rec["step"] <= cur[-1]["step"]:
            blocks.append(cur)
            cur = []
        cur.append(rec)
    if cur:
        blocks.append(cur)

    out = []
    for i, blk in enumerate(blocks):
        final = blk[-1]
        n_exec = len(blk)
        out.append({
            "block": i,
            "steps_executed": n_exec,
            "final_step_index": final["step"],
            "non_accepted_on_final_step": 256 - final["n_accepted"],
            "exit": "budget" if n_exec >= max_steps else "early_stop",
            "final_mean_entropy": round(final["mean_entropy"], 6),
            "accepted_curve": [r["n_accepted"] for r in blk],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="/home/ubuntu/diffgemma_fa/artifacts/phase0_baseline.json")
    args = ap.parse_args()

    prompts = [
        "What is the capital of France?",
        "Write a Python function that reverses a string.",
        "Explain what a finite automaton is in two sentences.",
        "List three prime numbers greater than 100.",
        "Convert 25 degrees Celsius to Fahrenheit.",
        "What is 17 * 23?",
        "Name the largest planet in the solar system.",
        "Write a JSON object with keys 'name' and 'age'.",
        "Summarise the plot of Hamlet in one sentence.",
        "What does the acronym GPU stand for?",
        "Give me a bash command to count lines in a file.",
        "What year did the Apollo 11 mission land on the moon?",
        "Translate 'good morning' into Spanish.",
        "What is the derivative of x^3?",
        "Describe the difference between TCP and UDP.",
        "Write a regular expression matching an email address.",
        "What is the boiling point of water at sea level?",
        "Give the first five Fibonacci numbers.",
        "What is the chemical symbol for gold?",
        "Explain recursion to a five-year-old.",
    ][: args.n_prompts]

    print(f"[load] devices={jax.devices()}", flush=True)
    t0 = time.perf_counter()
    model = diffusion.DiffusionGemma_26B_A4B()
    params = gm.ckpts.load_params(CKPT)
    load_s = time.perf_counter() - t0
    print(f"[load] params loaded in {load_s:.1f}s", flush=True)

    def hbm() -> dict[str, float]:
        st = jax.devices()[0].memory_stats() or {}
        return {
            "bytes_in_use_gb": round(st.get("bytes_in_use", 0) / 1e9, 2),
            "peak_bytes_in_use_gb": round(st.get("peak_bytes_in_use", 0) / 1e9, 2),
            "bytes_limit_gb": round(st.get("bytes_limit", 0) / 1e9, 2),
        }

    print(f"[mem] after load: {hbm()}", flush=True)

    results: list[dict[str, Any]] = []
    max_steps = 48

    for idx, prompt in enumerate(prompts):
        STEP_LOG.clear()
        STOP_LOG.clear()

        # A fresh sampler per prompt keeps the block log unambiguous. `self` is
        # a static argname (SPEC §5.3b) so this recompiles once and then hits
        # the compilation cache.
        sampler = diffusion.ChatSampler(
            model=model,
            params=params,
            sample_from_predictions=InstrumentedSampleFromPredictions(
                entropy_bound=0.1
            ),
            early_stop_fn=InstrumentedEarlyStop(entropy_threshold=0.005),
        )

        t0 = time.perf_counter()
        text = sampler.chat(prompt, max_new_tokens=args.max_new_tokens)
        wall = time.perf_counter() - t0

        blocks = summarise_blocks(list(STEP_LOG), max_steps)
        n_steps = len(STEP_LOG)
        rec = {
            "idx": idx,
            "prompt": prompt,
            "output": text,
            "output_chars": len(text),
            "wall_seconds": round(wall, 2),
            "total_denoising_steps": n_steps,
            "seconds_per_denoising_step": round(wall / max(1, n_steps), 4),
            "blocks": blocks,
            "n_blocks": len(blocks),
            "any_early_stop": any(b["exit"] == "early_stop" for b in blocks),
            "stop_fired_steps": [r["step"] for r in STOP_LOG if r["should_stop"]],
            "token_stability_ever": any(r["token_stability"] for r in STOP_LOG),
            "min_entropy_seen": round(
                min((r["entropy_mean"] for r in STOP_LOG), default=float("nan")), 6
            ),
        }
        results.append(rec)
        print(
            f"[{idx:02d}] {wall:6.1f}s  blocks={len(blocks)} "
            f"steps={n_steps} exits={[b['exit'] for b in blocks]} "
            f"non_accepted_final={[b['non_accepted_on_final_step'] for b in blocks]}",
            flush=True,
        )
        print(f"       -> {text[:110]!r}", flush=True)

    summary = {
        "checkpoint": CKPT,
        "load_seconds": round(load_s, 1),
        "memory": hbm(),
        "jax_version": jax.__version__,
        "n_prompts": len(results),
        "blocks_total": sum(r["n_blocks"] for r in results),
        "blocks_exiting_via_budget": sum(
            1 for r in results for b in r["blocks"] if b["exit"] == "budget"
        ),
        "blocks_exiting_via_early_stop": sum(
            1 for r in results for b in r["blocks"] if b["exit"] == "early_stop"
        ),
        "mean_non_accepted_on_final_step": round(
            float(np.mean([
                b["non_accepted_on_final_step"] for r in results for b in r["blocks"]
            ])), 2
        ),
        "max_non_accepted_on_final_step": int(max(
            b["non_accepted_on_final_step"] for r in results for b in r["blocks"]
        )),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        if k != "results":
            print(f"{k}: {v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
