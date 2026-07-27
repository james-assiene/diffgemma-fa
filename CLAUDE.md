# CLAUDE.md — diffgemma_fa

Constrained decoding for DiffusionGemma via finite automata, in **JAX**. Implementation of
[arXiv:2607.07026](https://arxiv.org/abs/2607.07026) against `google-deepmind/gemma`'s
`gemma/diffusion/` and the Orbax checkpoint at
`gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it`.

**Read `SPEC.md` before doing anything.** It is the design document. This file is the operating
manual: how to work in this repo, not what to build.

---

## Hard invariant

**Inference only. No training, no fine-tuning, no weight modification, no backward pass.** This is
a sampler modification against the stock released checkpoint. The method consumes exactly one thing
from the model — the `[B, 256, V]` logits already produced at every denoising step. The Flax module
and the params are never touched. If any step of the plan starts to need a training run, it is out
of scope: write it up in `docs/LOG.md` and stop.

Minimum fork surface is **`DiffusionSampler.sample_next_canvas`, a widened `SamplingState`, *and*
`_sample_step`**. SPEC §5.3 derives this from the library's real loop structure; read it before
designing anything.

> **Corrected in Phase 0 (2026-07-27).** This paragraph previously said `_sample_step` does *not*
> need forking. That is false against `gemma` 4.1.0: `sample_next_canvas` never receives `state`
> (its signature is `canvas_length, max_denoising_steps, batch_size, cache, params, rng,
> full_attention_mask`), and `max_new_tokens` is not a field of `SamplingState` at all — it lives
> only in `_sample_loop`'s `cond_fn` closure. So neither `A_k` nor the remaining budget `R` can
> reach the constrained sampler without forking `_sample_step`. The *within-block* claim survives:
> the denoising `while_loop` is fully encapsulated in `sample_next_canvas`, so widening
> `_WhileLoopCarry` and replacing `body_fn` needs no changes elsewhere. See
> `docs/PHASE0_FINDINGS.md` §2.2.

The tempting drift is SPEC §3.3 (renoising R1/R2) and §3.7 (constrained self-conditioning), which
shift the model's *inputs* off its training distribution. Flags to ablate and possibly reject,
never a reason to train. If they hurt, ship R0 / unconstrained self-conditioning.

## Where this project actually needs a GPU

Most of the build does not. Check `echo $DGFA_DEDICATED` (set by `env.sh`) to know which box you
are on, and plan accordingly.

| Phase | Resource | Notes |
|---|---|---|
| 0 — verification | **CPU**, except one smoke test | Reading source and ticking §1.2 needs nothing. The model load + smoke test needs ~50 GB HBM for ~an hour. |
| 1 — automaton compiler | **CPU, many cores** | The longest job in the project (hours to days, §4.7) and embarrassingly parallel. A GPU is idle the whole time. |
| 2 — float64 reference | **CPU** | numpy. |
| 3 — JAX inference kernels | **CPU is enough to be correct** | CPU JAX validates every exactness test. A GPU is needed only for the latency/kernel-count microbenchmarks. |
| 4 — model integration | **GPU** | Interactive, modest. |
| 5 — eval + §7.3 timings | **GPU, uncontended** | See below. |

**§7.3's overhead table is the paper's headline claim being reproduced, and it cannot be measured
on a shared box.** With preallocation off, JAX runs on the slower on-demand allocator and fragments
against other processes; a contended GPU adds noise on top. Numbers from that setup are pessimistic
and not publishable. If `DGFA_DEDICATED=0`, still run the timings — but label them "shared box,
indicative only" in `docs/RESULTS.md` and re-run on a dedicated device before drawing conclusions.

## Running alongside other experiments — read this before touching a GPU

*(Applies when `DGFA_DEDICATED=0`.)* **This box is shared. Other experiments are running on it and
must not be disturbed.** `bootstrap.sh` set up the isolation; your job is not to undo it.

- **Always `source env.sh` first.** It activates the project venv, pins
  `CUDA_VISIBLE_DEVICES` to one device, caps JAX's memory, and points the compilation cache
  somewhere private.
- **Never re-enable JAX memory preallocation.** JAX grabs ~75% of VRAM on first use by default;
  on this box that OOMs whatever else is running. `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `XLA_PYTHON_CLIENT_MEM_FRACTION=.60` stay as they are. **This overrides SPEC §1.1**, which
  assumes a dedicated machine — note the override in `docs/ENV.md` and remember that any latency
  number you measure under the on-demand allocator is pessimistic.
- **Never `pip install` outside the venv**, and never `sudo pip`. Upgrading jax, flax or CUDA
  libraries system-wide would break a running job mid-flight. If a package seems to need a system
  install, write it in `docs/LOG.md` and stop.
- **Check before you allocate.** `nvidia-smi` before any long run. If your assigned GPU has become
  busy, stop and log it rather than racing for memory.
- **Do not pull the 47.4 GB checkpoint until Phase 0 explicitly needs it**, and check `df -h`
  first.
- **Stay in `$PROJECT_DIR`.** No writes to shared paths — not `~/.cache/jax`, not `/tmp` for
  anything large, not another user's directories.
- **Long runs go in the background** with logs under `logs/`, and the PID recorded in
  `docs/LOG.md`. Never hold the tmux session hostage to a foreground job.

## Before you start

1. **You are in Phase 0 until `docs/PHASE0_FINDINGS.md` exists.** SPEC.md was written by reading
   `gemma/diffusion/` source over the web plus numerical verification on CPU — nothing has been run
   against the real model. §1.2 is a checklist to verify locally. Verify it, **edit SPEC.md in
   place** where it is wrong, record what changed. No implementation code first.
2. **`pip install git+https://github.com/google-deepmind/gemma.git`** — PyPI is 4.0.1, released
   before DiffusionGemma shipped, and does **not** contain `gemma/diffusion/`. Python ≥3.12.
3. **Check `docs/` for prior sessions.** `docs/LOG.md` is the running record; read it before
   assuming anything is unbuilt.

Set on day one:

```python
jax.config.update("jax_compilation_cache_dir", "~/.cache/jax")   # highest-leverage single line
```

## The one thing that must not break

`tests/test_guarantee.py`. It asserts **two different things**, and conflating them is a known trap:

```
per block k:  δ*(A_k, canvas_k) ≠ ∅  and  ⊆ {s : d(s) ≤ R}     # viable prefix
at end:       simulator.accepts(concat(canvas_0..canvas_K))     # this is CS
```

A non-final block's canvas ends in a live-but-not-accepting state and is **not** accepted on its
own. Asserting per-canvas acceptance makes this red on every multi-block generation, and the "fix"
is to weaken it — which must not happen.

The guarantee comes from making the **emitted** canvas a constrained object (SPEC §3.1), **plus**
the three closures in §3.1b: a budget-aware terminal factor `b_L = 1[d(s) ≤ R]`, an automaton-aware
stopping conjunct fed `emit_canvas` (not the trajectory canvas), and a `FREE` region excluding every
`end_token` and `PAD`. Constraining the sampler alone is necessary but not sufficient.

If you find yourself weakening this test to make something pass, stop and write up why instead.

## Working rules

- **Correctness before speed, always.** `infer/reference.py` is plain float64 numpy with no
  cleverness. Every optimized path is differential-tested against it. If the fast path and the
  reference disagree, the fast path is wrong.
- **Never let `tests/test_exactness.py` go red.** It brute-force-enumerates the constrained
  posterior on tiny alphabets (`V=4, L≤8`) and is the only thing standing between you and a
  plausible-looking sampler that draws from the wrong distribution. A subtly wrong sampler still
  produces *valid strings*. Use fixed seeds and Bonferroni-corrected χ² thresholds — an uncorrected
  suite of dozens of statistical tests goes red on its own.
- **Test NFAs, not just DFAs.** Several bugs (eq (8)'s edge multiplicity, start vectors) are
  invisible on DFAs.
- **One phase at a time.** SPEC §8.
- **Build J1 before J0.** J1 needs no change to the denoising carry and is the fastest path to
  end-to-end constrained generation; it validates the whole inference stack first. SPEC §5.4. Note
  "no fork" refers only to the carry — J1 still needs §5.3's traced automaton threading.
- **Log every `[?]` you resolve** — by measurement, not by reasoning. Record it in `docs/LOG.md` and
  update SPEC.md.
- **Never silently cap coverage.** If you subsample a benchmark, skip a split, or bound a grammar's
  depth, say so in the results table.

## Commands

```bash
pytest tests/ -x -q
pytest tests/test_exactness.py -x            # the bedrock — run constantly
pytest tests/test_guarantee.py -x

python -m diffgemma_fa.compile.tasks.bfcl --out artifacts/fa/bfcl/ --jobs $(nproc)
python -m diffgemma_fa.eval.run    --task bfcl_live --variant j0 --emission map --trajectory R0
python -m diffgemma_fa.eval.ablate --sweep entropy_bound --task bfcl_live --n 100
```

Long runs: `nohup ... > logs/<name>.log 2>&1 &`, and note the PID and log path in `docs/LOG.md`.
Automaton compilation for BFCL-Live is measured in hours (SPEC §4.7); never launch it in the
foreground.

## Landmines

All documented with evidence in SPEC.md. These cost a day each.

**JAX**

- **`lax.associative_scan(fn, x, reverse=True)` reverses the operand order.** The suffix scan needs
  `lambda a, b: b @ a`. No shape error, just wrong probabilities. (§2.6(c))
- **`self` is a `static_argname` on both `_sample_loop` and `_sample_step`.** Anything stored on your
  custom `sample_from_predictions` / `logit_shaper` is baked into the trace — per-request automaton
  arrays as attributes means a full recompile per schema (~2 h across BFCL). **Automaton values must
  be traced, shapes static.** There is a test for this. (§5.3)
- **There is no Python between blocks.** The outer block loop is `lax.while_loop` under `jit` with a
  *traced* `max_new_tokens`. The per-block automaton protocol must be traced ops on fixed-shape
  padded arrays; `A_k` is a `[S_bucket]` array, not a Python set. (§5.3)
- **Never use `jax.experimental.sparse`** — unmaintained by its own docstring. Use
  `gather + jax.ops.segment_sum` with a **static `num_segments`**. (§4.4)
- **No `jax.ops.segment_argmax` exists, and the obvious two-pass workaround breaks the tie-break
  rule** — `segment_max` over indices picks the *largest*. Use the `segment_min` variant with an
  `N` sentinel; empty segments give `-inf`, not `-1`. (§2.7)
- **`donate_argnums` needs matching input/output shape** — it is silently ignored on the tree
  builder (`[L,S,S] → [2L−1,S,S]`), and donating a buffer you then feed to the AOT warm-up deletes
  it. (§5.5)
- **Shape polymorphism does not avoid recompilation.** Use power-of-two `|S|` bucketing, with the
  ladder **derived from the memory budget** — do not warm a 4096 bucket you can never dispatch to.
  (§5.5)
- **Do not reach for Pallas before profiling** — its Triton kernels are not captured into command
  buffers by default, so it can *cost* you the CUDA-graph capture that is the whole reason for
  choosing JAX. (§4.4)
- **Do not build a PyTorch/JAX hybrid.** A framework boundary forces a sync per denoising step.
  (§5.8)

**Model**

- **The emitted canvas is the *sample*, and non-accepted positions are emitted as uniform random
  tokens over the full 262k vocab — and early stopping does NOT prevent this.** `should_stop` is
  computed from `previous_canvas` and `logits`, so it certifies the step's *input*; the emitted
  canvas is gated on the *old* `carry.done`, so the step on which early stop fires still emits its
  own fresh sample including that step's unaccepted positions. **Measured in Phase 0: 31/31 blocks
  early-stopped, 0/31 hit the budget, and 1/31 still emitted a random token.** (The earlier claim
  here — that this reaches the output "only via the budget path" — was false; see
  `docs/PHASE0_FINDINGS.md` §2.1.) A directly-constructed `DiffusionSampler` additionally defaults
  to `NoEarlyStop` and so always runs all 48 steps, which will bite you in tests. (§3.1)
- **The constraint mask must go in `logit_shaper`, not `sample_from_predictions`,** if you want it
  to reach self-conditioning — the SC tap is strictly upstream of the sampler. Use a finite
  sentinel (`-1e30`), never `-inf`. (§3.7, §2.4)
- **`q` has exact zeros. Never compute `log q` directly** — the naive entropy is `NaN` and `-inf`
  logits poison the self-conditioning matmul. `jnp.log(jnp.maximum(q, 1e-30))`. (§2.4)
- **`selection_mask` is rebuilt from zeros every step.** No monotone commit set — and that is *why*
  automaton state can be recomputed statelessly each step. (§3.2)
- **`forbidden_tokens` and `sampling` are inert on the diffusion path.** They will silently do
  nothing. (§1.2)
- **`_MIN_TEMP = 1e-12`, so a near-greedy path is reachable by configuration alone**
  (`min_temperature = max_temperature = 1e-12`). That is the closest analogue of the paper's `T=0`.
  (§3.9)
- **`SampleFromPredictions.text_vocab_size` defaults to 0** and is repaired by a
  `dataclasses.replace` keyed on `== 0`. A custom class without that field raises. Your subclass
  must also be frozen, kw-only, and hashable — tuples not lists, no `jnp` arrays. (§5.1)
- **`b_L` is budget-aware: `1[d(s) ≤ R]`** — not `1[s ∈ F]`, and *no* final-block special case
  (`d(s) ≤ 0 ⟺ s ∈ F` handles it). `d(s)` is a BFS on the reversed automaton computed **after** the
  stop-token augmentation and the refusal-branch union, or it is wrong. `R = max_new_tokens − step`.
  (§3.1b, §3.5)
- **The post-stop tail must be unscored (`ACC --Σ--> ACC`).** Otherwise a joint decode pays
  `(255−j)·log p(PAD)` to terminate at position `j` and never emits a stop token. Diagnostic: log
  stop-token position per block; clustering at 255 means this is still open. (§3.5)
- **Handle all of `end_tokens`** — `(EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE, *stop_tokens)`. The
  most likely cause of an empty state set at a block boundary. (§3.5)
- **Start vectors are vectors.** `a_start = 1[s=s₀]` only holds for a DFA on block 0. First shows up
  on block 2 or on Spider. (§5.7)

**Algorithm**

- **eq (8)'s token draw must be edge-multiplicity-weighted, not an `∃` indicator.** On an NFA with
  parallel overlapping edges the `∃` form is off by ~1.7e-2 against the exact posterior; the
  weighted form by 2e-17. On a DFA the two coincide — gate the fast path on `is_dfa`. (§2.6)
- **`a` and `b` must come from a down-sweep over the tree**, never a sequential loop. A `lax.scan`
  compiles to a device `while`, which is *not* in XLA's default command-buffer set — that is
  precisely the paper's +114%. Detect by *counting kernel launches per step*, not by reading code.
  (§0, §2.4)
- **Build the Blelloch/Brent–Kung shape, not Kogge–Stone.** Kogge–Stone does 3.6× the work, does not
  produce the aligned dyadic node set eq (7) needs, and materializing its levels is 4× the memory —
  which invalidates the dispatch threshold. (§2.6(b))
- **`max` has no complement trick**, and the max table needs its **own** polarity rule
  (`|N_c| ≤ K_max`, default 256) or `K` is unbounded at 131,073. Two class tables with **independent
  Pos/Neg partitions** — do not share the flag array. (§4.4 Layer 2b)
- **Normalize every tree node's matrix**, or `L=256` products underflow fp32 to zero. Scales cancel
  in both the midpoint conditional and the root draw; the log-scale sum runs over all `2L−1` nodes.
  Do the MAP path in log space (`max, +`) where the question does not arise. (§2.4, §2.6, §2.7)
- **Never complete the partial DFA with a sink state** — 170k transitions become 5.24e9. Never use
  Hopcroft at `|Σ|=262,144`; use Valmari (pure Python is fine). (§4.5)
- **Recalibrate `entropy_bound` and `entropy_threshold`** before believing any constrained result,
  and note `--confidence=mar` changes *stopping* as well as acceptance. (§3.4)
- **The tree is compute-bound above `|S| ≈ 512`** — 2.9% of a model forward at `|S|=385` but 54% at
  1024 and 386% at 1976. The paper's "+4%" is a small-automaton result, and "high overhead means
  launch-bound" only applies below the crossover. Set the dispatch threshold from compute *and*
  memory; at 8 GB the memory threshold alone is `|S| = 1,978`. (§0, §5.6, §7.3)
- **Spider (19,509 states) is structurally infeasible dense** — 778 GB tree, 389 GB for the `M_i`
  leaves alone. The dynamax operator does **not** fix this; it only makes backward *sampling*
  `O(S)`, while forward–backward stays `S×S`. (§2.6, §5.6)

## Style

- Type-annotate public functions; docstrings state shapes and dtypes. Prefer `jaxtyping`-style
  annotations to match the gemma library's `Float['*B L V']` convention.
- Reference the spec section in comments for anything non-obvious: `# SPEC §3.1`.
- Do not add dependencies without noting why in `docs/LOG.md`. Intended stack: `jax`, `flax`,
  `gemma` (git main), `orbax-checkpoint`, `outlines-core>=0.2.14`, `datasets`, `numpy`, `pytest`,
  plus a minimizer — the ~130-line Valmari port of SPEC §4.5 (**pure Python is fine**; `|Σ|` does not
  appear in its bound), or `pynini` if you happen to be on manylinux x86-64.
- No `try/except` around numerical code to make tests pass. **`Z == 0` has three distinct causes**:
  (a) the automaton is genuinely empty — a real bug; (b) no live continuation of `L` tokens from
  `A_k` within budget — a grammar/budget condition (SPEC §3.1b/§3.5); (c) fp32 underflow on an
  unnormalized path — expected, and SPEC §6.3 deliberately tests for it. Raise on (a) and (b);
  (c) means your scaling is missing.

## When you are stuck

Write it in `docs/LOG.md` and stop, rather than working around it. Specifically: an empty state set
at a block boundary; `Z == 0` you cannot classify; exactness tests failing at `L > 32` but passing
below; exactness failing on NFAs but passing on DFAs (that is eq (8)); tree overhead far above the
`|S|`-appropriate expectation (below `|S|≈512` that is launch count — profile kernels per step;
above it you are compute-bound and it may be correct); or XLA recompiling on every request (SPEC
§5.3 — something per-request landed on a static argument).
