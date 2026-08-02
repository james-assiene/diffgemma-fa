#!/usr/bin/env python
"""SPEC §7.3's overhead table — the paper's headline performance claim.

    python scripts/phase7_overhead.py --out artifacts/overhead.json

The claim under test is that constrained decoding costs only a few percent of
wall-clock. Measuring it honestly here requires stating three things the design
assumed and that are no longer true:

1. **The tree is not a GEMM any more.** `log_matmul` shifted from a single
   cuBLAS matmul to two fused reductions over the broadcast `A + B`, because
   the GEMM-shaped forms underflowed on real grammars (see its docstring —
   three wrong versions, each caught by the `Z == 0` detector). SPEC §0's
   kernel-count property survives; its GEMM-throughput property does not. Any
   number here must be attributed to *this* kernel, not the one §7.3 assumed.

2. **`|S|` is bucketed**, and the whitespace-tolerant grammar roughly doubles
   it, so overhead is reported per bucket rather than as one figure. SPEC §5.6
   puts the compute crossover near `|S| ≈ 512`: below it the tree is a few
   percent of a forward, above it the ratio grows fast.

3. **Overhead is per denoising step, but the model early-stops** (Phase 0:
   31/31 blocks, median 12 of 48 steps executed). A per-step ratio times 48
   overstates the real cost; the end-to-end column is measured, not derived.

Reported: model forward per step, tree per step, their ratio, and end-to-end
seconds per record from the real eval logs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.infer import scans  # noqa: E402

#: Buckets the ladder actually dispatches to on BFCL-Live.
BUCKETS = (64, 128, 256, 512, 1024)
CANVAS = 256


def time_call(fn, *args, n_warmup: int = 2, n_rep: int = 7) -> float:
    """Minimum of `n_rep` timed calls, in seconds. Minimum, not mean: we want
    the kernel's cost, not the scheduler's worst moment."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts)


def bench_tree(bucket: int, dtype) -> dict:
    """One up-sweep over `[L, S, S]` — the per-denoising-step constrained cost."""
    rng = np.random.default_rng(0)
    M = jnp.asarray(rng.standard_normal((CANVAS, bucket, bucket)) * 3 - 5.0,
                    dtype=dtype)
    f_sum = jax.jit(lambda m: scans.up_sweep_log(m).root)
    f_map = jax.jit(lambda m: scans.up_sweep_maxplus(m).root)
    out = {"bucket": bucket, "dtype": np.dtype(dtype).name}
    try:
        out["tree_sumproduct_s"] = round(time_call(f_sum, M), 5)
    except Exception as e:  # noqa: BLE001
        out["tree_sumproduct_s"] = None
        out["tree_sumproduct_error"] = type(e).__name__
    try:
        out["tree_maxplus_s"] = round(time_call(f_map, M), 5)
    except Exception as e:  # noqa: BLE001
        out["tree_maxplus_s"] = None
        out["tree_maxplus_error"] = type(e).__name__
    st = jax.local_devices()[0].memory_stats()
    out["peak_gpu_gb"] = round(st.get("peak_bytes_in_use", 0) / 1e9, 2)
    return out


def end_to_end_from_logs(logdir: pathlib.Path) -> dict:
    """Seconds per record, measured, from the artifacts the eval already wrote.

    This is the number that matters: a per-step ratio times 48 would overstate
    the cost, because the model early-stops at a median of 12 steps.
    """
    out = {}
    for f in sorted(logdir.glob("*.json")):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if "elapsed_seconds" in d and d.get("n"):
            out[f.stem] = round(d["elapsed_seconds"] / d["n"], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/overhead.json")
    ap.add_argument("--forward-s", type=float, default=0.21,
                    help="measured model forward per denoising step (Phase 0)")
    args = ap.parse_args()

    rows = []
    for b in BUCKETS:
        for dt in (jnp.float32, jnp.float64):
            print(f"[bench] bucket={b} dtype={np.dtype(dt).name}", flush=True)
            rows.append(bench_tree(b, dt))

    e2e = end_to_end_from_logs(pathlib.Path("artifacts"))

    out = {
        "canvas_length": CANVAS,
        "model_forward_per_step_s": args.forward_s,
        "per_step": rows,
        "end_to_end_seconds_per_record": e2e,
        "caveats": [
            "log_matmul is two fused reductions, not a GEMM; the GEMM-shaped "
            "forms underflowed on real grammars. SPEC §7.3's projection "
            "assumed the GEMM.",
            "Overhead is per denoising step, but the model early-stops at a "
            "median of 12 of 48 steps (Phase 0), so per-step x 48 overstates "
            "the real cost. Use the end-to-end column.",
            "The whitespace-tolerant grammar roughly doubles |S|, moving most "
            "records up one bucket.",
        ],
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n===== SPEC §7.3 OVERHEAD =====")
    print(f"model forward / denoising step: {args.forward_s:.3f} s (Phase 0)")
    print(f"{'bucket':>7} {'dtype':>8} {'sum-product':>12} {'max-plus':>10} "
          f"{'vs forward':>11} {'peak GB':>8}")
    for r in rows:
        sp = r.get("tree_sumproduct_s")
        mp = r.get("tree_maxplus_s")
        ratio = f"{sp / args.forward_s:.1%}" if sp else "n/a"
        print(f"{r['bucket']:>7} {r['dtype']:>8} "
              f"{(f'{sp:.4f} s' if sp else r.get('tree_sumproduct_error','n/a')):>12} "
              f"{(f'{mp:.4f} s' if mp else r.get('tree_maxplus_error','n/a')):>10} "
              f"{ratio:>11} {r['peak_gpu_gb']:>8}")
    print("\nend-to-end seconds per record (from completed arms):")
    for k, v in sorted(e2e.items(), key=lambda x: x[1]):
        print(f"  {k:38} {v:>6.1f} s")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
