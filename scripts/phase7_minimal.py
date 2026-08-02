"""SPEC §7.3's overhead measurement — minimal, and only what is known to run.

`phase7_overhead.py` wedged twice at `bucket=128`: 15 s and 55 s of CPU over 19
and 39 minutes respectively, GPU idle, process state `Ssl` — asleep, not
compiling. The difference from an earlier probe that DID complete every bucket
is `up_sweep_maxplus`, whose `maxplus_combine` materialises `[L/2, n, k, m]`
(68 GB at `|S| = 512`). So this measures only the **sum-product** tree, which
is the one on the `emission=sample` hot path, and takes the end-to-end column
from the eval artifacts — which is the number that actually answers the
paper's claim.

Three things make this different from what SPEC §7.3 projected, and they must
travel with any number reported from it:

1. `log_matmul` is **no longer a GEMM**. It is two fused reductions over the
   broadcast `A + B`, because every GEMM-shaped form underflowed on real
   grammars. SPEC §0's kernel-count property survives; its GEMM-throughput
   property does not.
2. The whitespace-tolerant grammar roughly **doubles `|S|`**, moving most
   records up one bucket — so the per-bucket row that matters shifted.
3. Overhead is per denoising step, but the model **early-stops at a median of
   12 of 48 steps** (Phase 0), so per-step × 48 overstates the real cost.
"""

import json
import pathlib
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from diffgemma_fa.infer import scans  # noqa: E402

FORWARD_S = 0.21  # Phase 0, measured on this checkpoint
rows = []

for S in (64, 128, 256, 512):
    # FLOAT32 IS EXCLUDED, and not for tidiness. The benchmark wedged three
    # times, always at `bucket=128`, always asleep rather than compiling
    # (15-55 s of CPU over 16-39 minutes, GPU 0%). Isolated: `bucket=64
    # float32` completes in 0.08 s, `bucket=128 float32` never returns, and
    # every probe that ever completed the larger buckets was float64
    # (`jax_enable_x64` makes `standard_normal` float64 by default, so float32
    # at |S| >= 128 had never actually been exercised). Some XLA:GPU pathology
    # in the fused reduction at that shape/dtype. float64 is what production
    # uses -- the float32 default was retracted after it drove spurious Z==0
    # on >50% of records -- so the column is not needed, but the hang is real
    # and should not be quietly forgotten.
    for dt in (jnp.float64,):
        M = jnp.asarray(
            np.random.default_rng(0).standard_normal((256, S, S)) * 3 - 5.0,
            dtype=dt)
        f = jax.jit(lambda m: scans.up_sweep_log(m).root)
        t0 = time.perf_counter()
        jax.block_until_ready(f(M))
        compile_s = time.perf_counter() - t0
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(f(M))
            ts.append(time.perf_counter() - t0)
        r = {"bucket": S, "dtype": np.dtype(dt).name,
             "compile_s": round(compile_s, 2),
             "tree_s": round(min(ts), 5),
             "vs_forward": round(min(ts) / FORWARD_S, 4)}
        rows.append(r)
        print(r, flush=True)
        del M

e2e = {}
for path in sorted(pathlib.Path("artifacts").glob("*.json")):
    try:
        d = json.load(open(path))
    except Exception:  # noqa: BLE001
        continue
    if d.get("elapsed_seconds") and d.get("n"):
        e2e[path.stem] = round(d["elapsed_seconds"] / d["n"], 1)

out = {"model_forward_per_step_s": FORWARD_S, "canvas_length": 256,
       "per_step": rows, "end_to_end_seconds_per_record": e2e}
pathlib.Path("artifacts/overhead.json").write_text(json.dumps(out, indent=2))

print("\n=== SPEC 7.3 ===", flush=True)
for r in rows:
    print(f"  |S|={r['bucket']:>4} {r['dtype']:>8}  tree {r['tree_s'] * 1000:>9.1f} ms"
          f"  = {r['vs_forward']:>8.1%} of a forward   (compile {r['compile_s']:.1f} s)")
print("\nend-to-end seconds per record:")
for k, v in sorted(e2e.items(), key=lambda x: x[1]):
    print(f"  {k:42} {v:>6.1f}")
print("\nwrote artifacts/overhead.json")
