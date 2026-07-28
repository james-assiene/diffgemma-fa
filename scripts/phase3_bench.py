"""Phase 3 microbenchmarks. SPEC §0, §5.5, §5.6, §7.3, §8.

Four things SPEC's Phase 3 exit criterion asks for:

  1. **Kernel launches per step must be `O(log L)`, not `O(L)`** — and SPEC is
     explicit that this is detected *by counting*, not by reading code. The
     failure mode is a `lax.scan` compiling to a device `while`, which is
     **absent** from XLA:GPU's default `xla_gpu_enable_command_buffer` set and
     so escapes CUDA-graph capture. That is the mechanism behind the paper's
     +114%.
  2. Latency vs `|S|`.
  3. Compile time per bucket.
  4. The **compute/launch crossover**, alongside the memory one.

Run: `python scripts/phase3_bench.py --out artifacts/phase3_bench.json`
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/diffgemma_fa")

import jax  # noqa: E402

jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from diffgemma_fa.infer import scans  # noqa: E402

#: One DiffusionGemma forward: 4B active params over 256 tokens.
MODEL_FLOPS_PER_STEP = 2 * 4e9 * 256


def hlo_stats(fn, *args) -> dict:
    """Instruction counts from the **optimized** HLO.

    Deterministic and CI-able, unlike a wall-clock profile. `while` is the one
    that matters: SPEC §0's table is `lax.scan` → 1 while loop, tree → 0.
    """
    lowered = jax.jit(fn).lower(*args)
    text = lowered.compile().as_text()
    lines = text.splitlines()
    return {
        "hlo_lines": len(lines),
        "while": sum(1 for l in lines if " while(" in l),
        "fusion": sum(1 for l in lines if " fusion(" in l),
        "dot_or_custom": sum(1 for l in lines
                             if " custom-call(" in l or " dot(" in l),
        "all_reduce": sum(1 for l in lines if " all-reduce(" in l),
    }


def sequential_chain(M):
    """The `O(L)`-depth form, for contrast. Compiles to a device `while`."""
    def step(carry, m):
        return carry @ m, None
    out, _ = lax.scan(step, jnp.eye(M.shape[1], dtype=M.dtype), M)
    return out


def tree_root(M):
    """The Blelloch up-sweep's root — unrolled straight-line code."""
    return scans.up_sweep(M, normalize=True).root


def bench(fn, *args, reps=20):
    f = jax.jit(fn)
    out = f(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ubuntu/diffgemma_fa/artifacts/phase3_bench.json")
    ap.add_argument("--sizes", default="64,128,256,385,512,1024,2048")
    args = ap.parse_args()

    device = jax.devices()[0]
    report: dict = {
        "device": f"{device.device_kind} ({device.platform})",
        "jax": jax.__version__,
        "model_flops_per_step": MODEL_FLOPS_PER_STEP,
    }
    print(f"[bench] {report['device']}", flush=True)

    # --- 1. kernel launches / HLO shape, tree vs sequential chain ---------
    launch = []
    for L in (16, 32, 64, 128, 256):
        S = 64
        M = jnp.asarray(np.abs(np.random.default_rng(0).random((L, S, S))) * 0.5,
                        dtype=jnp.float32)
        t = hlo_stats(tree_root, M)
        c = hlo_stats(sequential_chain, M)
        row = {
            "L": L, "S": S,
            "tree_while": t["while"], "chain_while": c["while"],
            "tree_dot": t["dot_or_custom"], "chain_dot": c["dot_or_custom"],
            "tree_fusion": t["fusion"], "chain_fusion": c["fusion"],
            "tree_hlo_lines": t["hlo_lines"], "chain_hlo_lines": c["hlo_lines"],
            "log2L": int(np.log2(L)),
        }
        launch.append(row)
        print(f"  L={L:4} tree: while={row['tree_while']} dot={row['tree_dot']} | "
              f"chain: while={row['chain_while']} dot={row['chain_dot']}", flush=True)
    report["launch_counts"] = launch

    # --- 2/3. latency and compile time vs |S| -----------------------------
    L = 256
    rows = []
    for S in [int(x) for x in args.sizes.split(",")]:
        tree_bytes = (2 * L - 1) * S * S * 4
        if tree_bytes > 12e9:
            print(f"  S={S}: skipped, tree would need {tree_bytes/1e9:.1f} GB",
                  flush=True)
            rows.append({"S": S, "skipped": True,
                         "tree_gb": round(tree_bytes / 1e9, 3)})
            continue
        M = jnp.asarray(
            np.abs(np.random.default_rng(1).random((L, S, S))).astype(np.float32)
            * 0.5)
        t0 = time.perf_counter()
        compiled = jax.jit(tree_root).lower(M).compile()
        compile_s = time.perf_counter() - t0

        out = compiled(M)
        jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(10):
            out = compiled(M)
        jax.block_until_ready(out)
        latency = (time.perf_counter() - t0) / 10

        # SPEC §0: the tree's arithmetic is "2L * 2|S|^3 FLOPs per denoising
        # step" — 2L combines, each a 2S^3 GEMM. (An earlier version of this
        # script double-counted and read 2x high against SPEC's table.)
        flops = 2 * L * 2 * S ** 3
        rows.append({
            "S": S,
            "tree_gb": round(tree_bytes / 1e9, 4),
            "compile_seconds": round(compile_s, 2),
            "latency_ms": round(latency * 1e3, 3),
            "tree_flops": flops,
            "pct_of_model_forward": round(100 * flops / MODEL_FLOPS_PER_STEP, 2),
            "hlo_lines": len(compiled.as_text().splitlines()),
        })
        print(f"  S={S:5} tree={tree_bytes/1e9:6.3f} GB  compile={compile_s:5.2f}s  "
              f"latency={latency*1e3:8.3f} ms  {rows[-1]['pct_of_model_forward']:7.2f}% "
              f"of a model forward", flush=True)
    report["scaling"] = rows

    # --- 4. crossovers ----------------------------------------------------
    ok = [r for r in rows if not r.get("skipped")]
    compute_cross = next((r["S"] for r in ok if r["pct_of_model_forward"] > 100),
                         None)
    # Wall-clock overhead against the measured per-denoising-step time of the
    # real model (Phase 0 §1.4: ~0.21 s per denoising step at B=1 on this H100).
    # This is the number that actually matters: the FLOP ratio assumes both run
    # at the same efficiency, but at B=1 the model forward is memory-bound while
    # the tree is a dense GEMM, so the FLOP ratio is pessimistic.
    MEASURED_STEP_SECONDS = 0.21
    for r in ok:
        r["pct_of_measured_step_walltime"] = round(
            100 * (r["latency_ms"] / 1e3) / MEASURED_STEP_SECONDS, 3)
    walltime_cross = next(
        (r["S"] for r in ok if r["pct_of_measured_step_walltime"] > 10), None)

    report["crossovers"] = {
        "measured_model_step_seconds": MEASURED_STEP_SECONDS,
        "S_where_tree_exceeds_10pct_of_measured_step": walltime_cross,
        "compute_S_where_tree_exceeds_one_model_forward": compute_cross,
        "memory_S_at_8GB": int((8e9 / ((2 * L - 1) * 4)) ** 0.5),
        "memory_S_at_measured_20GB_headroom": int((20e9 / ((2 * L - 1) * 4)) ** 0.5),
        "note": "SPEC §5.6: set the dispatch threshold from compute AND memory; "
                "compute binds far below the memory wall.",
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("\n===== CROSSOVERS =====")
    for k, v in report["crossovers"].items():
        print(f"{k}: {v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
