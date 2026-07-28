# Phase 3 findings — the JAX inference kernels

Date: **2026-07-28**. Deliverable: `diffgemma_fa/infer/{scans,marginals,tree}.py`.
**651 tests green.** Hardware: H100 80GB, JAX 0.11.0.

| module | what it is |
|---|---|
| `infer/scans.py` | Blelloch up-sweep retaining every aligned dyadic product, down-sweep for `a`/`b`, max-plus variant |
| `infer/marginals.py` | class-table SpMM (`gather + segment_sum`, static `num_segments`), complement-aware `q_i`, clamped entropy |
| `infer/tree.py` | eq (7) state sampling, eq (8) multiplicity-weighted token draw, max-plus MAP with `(s,s')→class` token recovery |
| `tests/test_jax_differential.py` | 88 cases, all against the Phase 2 float64 reference |
| `tests/test_kernel_shape.py` | 21 cases on the **optimized HLO** — the `O(log L)` exit criterion |
| `scripts/phase3_bench.py` | latency, compile time, crossovers |

---

## 1. SPEC §0's central claim, reproduced on an H100

SPEC's whole argument for JAX is that the tree compiles to straight-line code whose every op type
is in XLA:GPU's default command-buffer (CUDA-graph) capture set, while a sequential `lax.scan`
becomes a device `while` — which is **absent** from that set and so pays `L` per-kernel launches.

Counted on the optimized HLO:

| `L` | tree `while` | tree matmuls | chain `while` | chain matmuls |
|---|---|---|---|---|
| 16 | **0** | **4** | 1 | 1 |
| 32 | **0** | **5** | 1 | 1 |
| 64 | **0** | **6** | 1 | 1 |
| 128 | **0** | **7** | 1 | 1 |
| 256 | **0** | **8** | 1 | 1 |

**The tree's matmul count is exactly `log₂ L`** — not merely `O(log L)` — and it never emits a
`while`. The chain emits exactly one `while` with one matmul inside it, hiding all `L` products in
a loop body that escapes capture. That is SPEC §0's table, reproduced.

HLO size makes the same point structurally: the tree's program **grows** with `L` (243 → 403 lines
from `L=16` to `L=256`) because it is unrolled, while the chain's is **constant at 94 lines**
because the work is inside the loop. Unrolling is what buys the capture.

Pinned permanently in `tests/test_kernel_shape.py`, which asserts on the compiled HLO rather than
a profiler trace — deterministic, fast (5.5 s), and green on CPU too.

---

## 2. SPEC §0's FLOP table, confirmed to two significant figures

`L = 256`, tree arithmetic `2L · 2|S|³`, against one 4B-active forward over 256 tokens
(`2·4e9·256 = 2.05e12`):

| `\|S\|` | tree GB | measured latency | **% of a model forward (FLOPs)** | SPEC §0 predicted |
|---|---|---|---|---|
| 64 | 0.008 | 0.057 ms | 0.01% | — |
| 128 | 0.033 | 0.091 ms | 0.10% | — |
| 256 | 0.134 | 0.261 ms | 0.84% | — |
| 385 | 0.303 | 1.055 ms | **2.85%** | **2.9%** |
| 512 | 0.536 | 0.950 ms | **6.71%** | **6.7%** |
| 1024 | 2.143 | 3.874 ms | **53.69%** | **53.7%** |
| 2048 | 8.573 | 20.898 ms | 429.50% | 386% at `\|S\|=1976` |

Every predicted cell matches. SPEC §0 is correct as written.

---

## 3. The finding that changes the dispatch rule

**The FLOP ratio is drastically pessimistic as a latency proxy on this hardware — by ~43× at the
top of the range.**

Against the measured per-denoising-step wall time of the real model (Phase 0 §1.4: ~0.21 s at
`B=1`):

| `\|S\|` | FLOP ratio says | **measured wall-clock share** |
|---|---|---|
| 385 | 2.85% | **0.50%** |
| 512 | 6.71% | **0.45%** |
| 1024 | 53.69% | **1.84%** |
| 2048 | **429.50%** | **9.95%** |

Even at `|S| = 2048` — where the FLOP ratio says the tree costs 4.3 model forwards — it actually
costs **under 10% of a denoising step**.

The reason is that the two workloads are bound by different resources. At `B = 1` the model forward
is **memory-bound**, streaming ~50 GB of weights per step; the tree is a dense, compute-bound GEMM
of exactly the shape an H100 is built for. Comparing them by FLOPs assumes equal efficiency, and
they are nowhere near equal.

**Consequence for SPEC §5.6.** The dispatch threshold should be set from **measured wall-clock**,
not from the FLOP ratio, and on this hardware the tree path is viable across the entire usable
`|S|` range. Combined with Phase 1's measurement that BFCL-Live's largest automaton is **595
states** (0.3% of a step here), the chain-path fallback is not needed for BFCL at all — it remains
open only for Spider.

**Caveat, stated because it matters.** The 0.21 s/step baseline is total generation wall-time
divided by executed denoising steps, so it includes the KV-cache append forward and per-block
overhead. It is therefore an *upper* bound on the pure denoising forward, which makes these
percentages a *lower* bound. The direction and the order of magnitude are safe; the exact figures
should be re-derived in Phase 5 against an isolated forward, which is what §7.3 measures properly.

---

## 4. Compile time and bucketing (SPEC §5.5)

- **Cold compile: 1.16–2.88 s per bucket. Warm (compilation cache): 0.04–0.15 s.** The cache is
  doing real work; keep `jax_compilation_cache_dir` set.
- **HLO size is flat in `|S|`** — 403–485 lines across `|S| = 64…2048`. SPEC §5.5 predicted this
  ("2,878 lines in all three cases… you pay per **bucket**, not per size"); the absolute number
  differs because this is the up-sweep alone rather than the full program, but the invariance —
  which is the load-bearing part — holds exactly. Asserted in `test_hlo_size_is_flat_in_S`.

---

## 5. Differential testing (SPEC §6.2)

88 cases, every one against `infer/reference.py`, on **both DFAs and NFAs**, with x64 enabled so
the comparison measures the algorithm rather than fp32 noise:

| path | agreement with the float64 reference |
|---|---|
| class SpMM → `W`, `M_i` | 2.2e-16 |
| complement-aware `q_i` | 1.1e-16, rows sum to 1 |
| retained dyadic block products | 3.3e-16, all `2L−1` nodes |
| `a`/`b` down-sweep, `log Z` `i`-invariance | 4.4e-16 |
| max-plus MAP score and tokens | exact |
| tree sampler vs brute-force posterior | max deviation < 0.02 over 20k draws, NFAs included |

**Zero skips.** Two coverage gaps were closed rather than tolerated, both of the kind CLAUDE.md
forbids:

- the complement-aware tests skipped when a random instance happened to contain no negated class.
  The DFA generator was assigning destinations uniformly, which scattered the `V` tokens too thinly
  for any state pair to accumulate `> V/2` labels — so on DFAs the complement path was **never**
  exercised. The generator now uses a small per-state fan-out and the tests assert `is_neg.any()`.
- the tree-vs-posterior test drew 20,000 samples one at a time, i.e. 40,000 separate JAX
  dispatches, and took **19 minutes on its own**. `vmap` over the key batch makes it seconds. For a
  suite CLAUDE.md says to run constantly, that is not a nicety.

Full suite: **651 tests in 2m33s.**

---

## 6. Not done

- **The dynamax `O(S)` backward-sampling operator is not prototyped.** SPEC §2.6 lists it as worth
  trying and §7.4 as a third sampler variant. It does not change the wall (forward–backward stays
  `S×S`), so it is an optimisation, not a capability — but it is a Phase 3 deliverable and is
  outstanding.
- **No bucket dispatcher / AOT warm-up.** `compile/automaton.py` computes the bucket and flags
  `needs_chain_path`; nothing yet dispatches on it or pre-warms buckets at startup (SPEC §5.5).
- **No `donate_argnums` work.** SPEC §5.5 warns donation is silently ignored on the tree builder
  (`[L,S,S] → [2L−1,S,S]`) and that donating a buffer fed to the AOT warm-up deletes it. Neither
  has been exercised.
- **The chain sampler has no JAX implementation.** Only the reference has one. Since §3 above shows
  the tree covers the whole usable `|S|` range on this hardware, the chain path is currently
  unreachable in practice — but SPEC §7.4 wants chain-vs-tree as an ablation, so it will be needed
  for the eval.
- **`B = 1` only.** SPEC §5.6's dispatch rule is per *batch* (`B·S_max²`); batched behaviour is
  untested.
- **fp32 end-to-end is not differential-tested.** The differential suite runs x64 by design;
  `test_numerics.py` covers fp32 underflow on the reference, but the JAX path's fp32 accumulation
  has not been compared against x64 at `L = 256`.
