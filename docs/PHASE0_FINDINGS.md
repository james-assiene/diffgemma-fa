# Phase 0 findings

Date: **2026-07-27**. Stack: `gemma` 4.1.0 (git main), JAX 0.11.0, H100 80GB, checkpoint
`diffusiongemma-26B-A4B-it` loaded from local disk. Machine details in `docs/ENV.md`.

Phase 0's job was to verify SPEC.md against the real library and the real model, and to correct
SPEC.md in place. **Every box in §1.2 and §1.3 is now ticked or corrected.** SPEC.md carries the
corrections inline, marked `[V-P0]`; this file is the summary and the evidence index.

Scripts, all re-runnable:

| script | what it establishes |
|---|---|
| `scripts/phase0_tokenizer.py` | §1.3 — vocab padding, mask token, channel markers, special ids |
| `scripts/phase0_outlines.py` | §1.3(3), §4.3 — building an outlines-core `Vocabulary` from Gemma |
| `scripts/phase0_lift_timing.py` | §4.7 — schema→regex→DFA→token-lift wall clock |
| `scripts/phase0_smoke.py` | §1.4 — 20-prompt baseline, per-step and per-block instrumentation |
| `scripts/phase0_multiblock.py` | §3.5, §5.7 — multi-block behaviour and the emitted format |

Artifacts: `artifacts/phase0_baseline.json`, `artifacts/phase0_multiblock.json`,
`artifacts/phase0_lift_timing.json`.

---

## 1. Confirmed as written

SPEC.md was accurate on essentially all of the mechanical detail. Confirmed verbatim against source:

- `SampleFromPredictions.__call__` is fully keyword-only and returns a bare `[B, L]` token array.
- Emission is `jnp.where(selection_mask, denoiser_tokens, random_tokens)`; `random_tokens` is
  `jax.random.randint(0, text_vocab_size)` over the whole vocab.
- `selection_mask` is rebuilt from `jnp.zeros_like` every call — non-monotone, nothing mask-shaped
  in the carry. §3.2's "recompute automaton state statelessly each step" stands.
- Accept rule: ascending entropy `argsort`, `cumsum − sorted ≤ entropy_bound`, default `0.1`,
  always accepts ≥1.
- Entropy in nats, from `log_softmax(logits.astype(float32))`, on the **shaped** logits.
- `_WhileLoopCarry` is exactly `(step, canvas, sc_embeddings, rng, done)`.
- `sample_next_canvas` returns `final_carry.canvas`. **The only `argmax` in `_sampler.py` is
  `first_stop_idx`** — verified by grep, one hit. No argmax in the emission path.
- `shaped_prediction` is the single tensor feeding both `sample_from_predictions` and
  `embedder.encode_logits`. The self-conditioning tap is strictly upstream of the sampler, so
  §3.7's "the mask must go in `logit_shaper`" is correct.
- Softcap is applied at the end of `call_with_self_conditioning`, before the shaper.
- `_truncate_canvas_at_stop_tokens` keeps the first stop token and PADs after; `& ~done` makes a
  finished sequence emit an all-PAD block; `PAD_TOKEN = 0`.
- `_sample_step` advances by a fixed `canvas_length`; the last block does not shrink.
- `canvas_length = 256`, `max_denoising_steps = 48` (defaults on the diffusion `Sampler`/
  `ChatSampler`; **required fields with no default on `DiffusionSampler` itself**).
- `ChainedEarlyStop` is AND. `TokenStabilityEarlyStop` has no fields and no patience.
- `DiffusionSampler`'s own `early_stop_fn` default is `NoEarlyStop`.
- `_MIN_TEMP = 1e-12`; `min_temperature = max_temperature = 1e-12` is legal, so §3.9's near-greedy
  path is reachable by configuration alone.
- `forbidden_tokens` and `sampling` are inert on the diffusion path.
- The block loop is `lax.while_loop` inside `jit` with a **traced** `max_new_tokens`. There is no
  Python between blocks. §5.3 rests on this and it is correct.
- `is_zero_sc` is a global reduction across the batch.
- `MASK = 4` exists and is not the diffusion mechanism (corruption is uniform random tokens).

`SamplingState.cache_info`, flagged in §1.2 as "referenced but not retrieved", is **a `@property`**
returning `_cache_helper.Cache(self.cache)` — derived, not a field. Widening `SamplingState` does
not have to supply it.

---

## 2. Corrections — things SPEC.md got wrong

### 2.1 §3.1 — the emission invariant is false, and it matters

**This is the most important correction in Phase 0.** SPEC claimed:

> ~~Random tokens reach the output **only when the 48-step budget is exhausted** without reaching a
> stability fixed point.~~

That is wrong. In `body_fn`:

```python
out      = self.sample_step(...)
new_done = carry.done | self.early_stop_fn.should_stop(
    step=step, canvas=out.sampled_tokens, previous_canvas=carry.canvas, logits=out.logits)
canvas   = jnp.where(carry.done[:, None], carry.canvas, out.sampled_tokens)   # OLD done
```

`should_stop` certifies convergence of the step's **input** (`previous_canvas`, `logits`), while
the emitted canvas is gated on `carry.done` — the **old** flag. So on the step where early stop
first fires, the canvas still becomes `out.sampled_tokens`, including *that step's* unaccepted
positions, and the loop exits. **A block can early-stop and emit uniform random tokens in the same
step.**

Measured over 31 blocks: **31/31 exited via early stop, 0/31 via the budget**, and **one block
still emitted a random token** (1 unaccepted position on its final step). Rare — ~1 block in 31,
~1 position in 7,936 — but unconditional-guarantee-breaking, which is exactly what J0's decoupled
`emit_canvas` closes. The J2-vs-J0 gap this project wants to report is real but will be *small* on
this model; say so rather than overselling it.

### 2.2 §5.3(c) / CLAUDE.md — `_sample_step` does need forking

SPEC §5.3(c) and CLAUDE.md's "hard invariant" both state that the minimum fork surface is
`sample_next_canvas` plus a widened `SamplingState`, and that `_sample_step` need **not** be
forked. Two facts from the installed source make that impossible:

1. **`sample_next_canvas` never receives `state`.** Its real signature is
   `(*, canvas_length, max_denoising_steps, batch_size, cache, params, rng, full_attention_mask)`.
   `A_k`, `state.step` and `predicted_tokens` are all unreachable from the one place the
   constrained sampler must live. The only per-block quantity visible there is
   `cache_layer['end_index']`.
2. **`max_new_tokens` is not a field of `SamplingState`.** It is a parameter of `_sample_loop`,
   captured only in that function's `cond_fn` closure — so `R = max_new_tokens − state.step` is not
   computable inside `_sample_step` either, contrary to §5.3's bullet and §3.5's per-block sketch.

`b_L(s) = 1[d(s) ≤ R]` needs `R` and `a_start = 1[s ∈ A_k]` needs `A_k`, so both must be threaded
explicitly: widen `SamplingState`, **fork `_sample_step`** (~40 lines) to read them off the state
and pass them into a widened `sample_next_canvas`, and widen the state at the entry point (it is
built by `_prefill.prefill` inside `gm.text.Sampler.sample`). SPEC §5.3 now carries the revised
structure. **CLAUDE.md's hard-invariant paragraph should be amended to match** — it is the one
place the operating manual now disagrees with verified fact.

Note the *within-block* half of §5.3(c) survives intact: the denoising `while_loop` is fully
encapsulated in `sample_next_canvas`, so widening `_WhileLoopCarry` and replacing `body_fn` needs
no changes elsewhere.

### 2.3 §3.1b — there is a fourth way to terminate mid-grammar

`_sample_loop`'s `cond_fn` is
`(state.step < max_new_tokens) & ~all(state.done) & ~state.cache_info.is_full`.
**Cache exhaustion** is a third loop exit alongside budget and early stop, and it breaks the
guarantee the same way. Bound the remaining budget by cache capacity too:
`R = min(max_new_tokens − step, cache_length − used_cache_length)`.

### 2.4 §1.3(1) — there is no lm_head vocabulary padding

SPEC assumed the 262,144 vocab exceeds the real token count and that pad slots decoding to `""`
must be stripped from every automaton alphabet. Measured: the SentencePiece model has **262,144
real pieces**, the highest non-empty id is **262,143**, and the contiguous empty tail has **length
0**. Exactly three ids decode to `""` — `PAD=0`, `EOS=1`, `BOS=2` — and those are control tokens.
There is no padding to exclude.

### 2.5 §1.3(4) / §3.6 — the thought marker is real, but it is not what SPEC assumed

There is **no** `END_OF_THOUGHT` token and nothing thought-related in `special_tokens`. The model
emits a **channel-tagged** format whose delimiters *are* single dedicated token ids:

| literal | ids |
|---|---|
| `<\|channel>` | **100** |
| `<channel\|>` | **101** |
| `<\|channel>thought\n<channel\|>` | `[100, 45518, 107, 101]` |

All 11 long generations opened with exactly that 4-token prefix at positions 0–3 and contained no
further channel marker — the header is a prefix, not a thought/answer split. So §3.6's two-state
construction **holds**, keyed on ids 100/101; no Aho–Corasick and no BPE-segmentation enumeration
is needed, and `FREE*` can be dropped in favour of anchoring the grammar immediately after token
101. Note 100/101 are **not** in the `special_tokens` enum despite being dedicated ids — do not
enumerate the special set from that enum alone.

### 2.6 §1.3(3) — `Vocabulary.from_pretrained` is the wrong path, for a boring reason

It fails on `google/gemma-3-4b-it` with **HTTP 401**: the repo is gated. That is an auth failure,
not the `UnsupportedByTokenProcessor` rejection SPEC anticipated. Building the `Vocabulary`
directly from Gemma's own SentencePiece model works, needs no HF access, takes 2.9 s and maps
262,140 pieces (256 byte-fallback, 4 control/unknown skipped). `outlines_core.Index(regex, vocab)`
and `.get_transitions()` both work on the result. Code is in SPEC §1.3 and
`scripts/phase0_outlines.py`.

### 2.7 §4.3 — state id stride is 64, not 8

Observed ids run 256…2,816 with a **uniform stride of 64**. The trap SPEC describes (raw
`regex-automata` ids, not dense, renumber immediately) is real; only the number was wrong.

### 2.8 §1.1 / §5.6 — the memory figures

Checkpoint is **37.63 GiB (40.4 GB) on disk**, not ~47.4 GB. But it is **51.65 GB resident in
HBM**, peaking at **56.5 GB** during generation against a 76.52 GB limit — so ~**20 GB** is free
for the tree, not §5.6's assumed 8 GB. That lifts the memory-side dispatch threshold from
`|S| = 1,978` to `|S| ≈ 3,127`. Compute remains the binding constraint far below that (§0), so this
does not change the plan — but the ladder must be derived from measured free HBM, and it means the
paper's largest BFCL DFA (2,459 states) is not obviously a chain-path fallback here.

---

## 3. §4.7 was wrong by ~1,000× — the Phase 1 schedule changes

SPEC extrapolated one published number to **4–8 minutes per schema** and **4–7 days serially** for
BFCL-Live, making the vocabulary pre-filter "the largest single payoff" in Phase 1.

Measured, full `schema → regex → DFA → token lift → get_transitions` on the 262k Gemma vocab:

| schema shape | total | states | nnz |
|---|---|---|---|
| 1 string property | **0.126 s** | 25 | 268,397 |
| typical 3-property call | **0.271 s** | 75 | 268,750 |
| 8 properties, mixed types | **0.652 s** | 182 | 537,953 |
| nested object, depth 2 | **0.364 s** | 91 | 537,205 |
| array of strings | **0.262 s** | 56 | 537,077 |
| 20-way enum | **0.250 s** | 70 | 268,614 |
| array of objects | **0.390 s** | 95 | 537,315 |
| **`{}` wildcard** | **16.7 s** | 3,261 | 36,210,539 |

Realistic function-call schemas cost **0.13–0.65 s**. BFCL-Live's 1,351 instances project to
**~0.9 hours serially**, minutes across 26 cores.

- **Do not build the vocabulary pre-filter.** There is nothing left to win.
- **Open question 5 is closed.** Compilation throughput is not a project risk.
- **The real watch item is the `{}` / missing-`type` / `additionalProperties: true` shape**, which
  is 26–130× slower than anything else and produces 3,261 states and 36.2M transitions. §4.2's
  fail-loud pre-pass should flag it explicitly.

**Caveat on coverage.** These eight schemas are hand-written to span BFCL's shapes; they are not
drawn from BFCL itself, which has not been downloaded yet. The projection is indicative. Re-measure
on the real schema set at the start of Phase 1 before quoting the 0.9 h figure anywhere.

---

## 4. New quantitative facts for §4.4

The largest label class in **every** schema tested covers **261,077–261,153 of 262,141 tokens**
(~99.6% of the vocab). That is emphatic confirmation that Layer 2's complement trick is
load-bearing — SPEC's "without this the scheme fails" is correct.

It also breaks a default: those classes have `|N_c| = 988–1,064`, well above **Layer 2b's
`K_max = 256`**. With `K_max = 256` they would all fall back to positive storage in the max table
and Layer 3's `nnz ≈ 25,000` budget would be violated by ~four orders of magnitude. The fix is
free — `K_max = 1100` gives a topk buffer of `1101 × 256 × 4 B ≈ 1.1 MB`. **Set `K_max` from the
compiled grammar's measured `max_c |N_c|`, default 1,100, and assert `K > max_c |N_c|` at build
time.** SPEC §4.4 Layer 2b updated.

The class-dedup ratio (§4.4 Layer 1, open question 9) is **not** yet measured properly: outlines
exposes `state → {token → state}`, so what was counted here is distinct label sets per *state*
(1.14–19.6×), not per *edge*. The edge-level ratio is a Phase 1 deliverable.

---

## 5. Baseline measurements (§1.4)

`ChatSampler` defaults, H100, `entropy_bound = 0.1`, `entropy_threshold = 0.005`.

| Quantity | Measured |
|---|---|
| Blocks observed | 31 (20 single-block runs + 11 blocks across 3 multi-block runs, up to 4 deep) |
| Exit via early stop / budget | **31 / 0** |
| Denoising steps executed per block | **3 – 31** of 48, median ≈ 12 |
| Non-accepted positions on final step | 0 in 30 blocks, **1** in one block |
| Params resident / peak HBM | 51.65 GB / 56.5 GB of 76.52 GB |
| Checkpoint load | 35–36 s |
| First generation (cold compile) | 47.3 s → **11.9 s** warm cache |
| Steady state, 256 tokens | **1.2 – 4.2 s** (≈ 0.21 s per denoising step) |

Acceptance curves are smooth and monotone, e.g. `[7, 15, 15, 18, 34, 72, 169, 200, 228, 254, 256,
256]` for a code-generation prompt. Final-step mean entropies land at 5e-5 – 4e-4, well under the
0.005 stopping threshold.

**Implication for §7.4.** The model already converges in a median of ~12 of 48 steps, so the
`48 → 24 → 12` rungs of the denoising-steps ablation are largely no-ops on the unconstrained
baseline. Report *executed* steps, not the cap, or the ablation will look flat for the wrong
reason.

**Implication for §3.4.** The recalibration sweep is still required, but note the starting point:
the unconstrained model is already accepting all 256 positions within ~12 steps at
`entropy_bound = 0.1`. Constrained marginals will be sharper still, so the interesting direction is
*downward* — the sweep's `0.003` / `0.01` rungs matter more than `0.3` / `1.0`.

---

## 6. Open questions after Phase 0

| # | Status |
|---|---|
| 0a — how does the paper handle `\|S\| ≈ 20,000`? | **Still open.** Untouched by Phase 0; still the highest-value question. |
| 0b — compute/launch crossover on this hardware | **Still open**, Phase 3. Memory side now measured (~20 GB free ⇒ `\|S\| ≈ 3,127`). |
| 0c — joint decode termination bias | **Still open**, needs the first end-to-end constrained run. |
| 1 — entropy recalibration | Still open; §5 above sharpens the expected direction. |
| 2 — J0 vs J1 | Still open. §2.1 above shows the J2→J0 gap on this model is real but small. |
| 3 — renoising / self-conditioning | Still open. |
| 4 — carried `A_k` vs stateless recompute | **Partly resolved:** the stateless route does *not* avoid the `_sample_step` fork, only the extra field. Still measure. |
| 5 — compilation throughput | **CLOSED.** ~0.9 h for BFCL-Live; pre-filter unnecessary. |
| 6 — greedy semantics | Still open, Phase 2/3. |
| 7 — budget-path frequency | **CLOSED, and the question was mis-framed** — 0/31 budget exits, but random tokens still reached the emission once. See §2.1. |
| 8 — BFCL irrelevance | Still open. |
| 9 — dedup / minimization ratios | Still open; §4 above explains why the number measured in Phase 0 is not the right one. |

---

## 7. What Phase 0 did not do

Stated plainly so nobody assumes coverage that does not exist:

- **No benchmark data was downloaded.** BFCL, xLAM, Spider, GSM-Symbolic, Countdown and Sudoku are
  all untouched. The §4.7 projection uses hand-written schemas, not real BFCL ones.
- **No batching was exercised.** Everything ran at `B = 1`. §5.6's per-batch dispatch rule and the
  `is_zero_sc` batching quirk (§3.7) are unverified.
- **`--temp-schedule=near-greedy` was not exercised.** `_MIN_TEMP` was confirmed by source read and
  the config's `__post_init__` validation, not by running at `1e-12`.
- **Only 3 multi-block prompts** (11 blocks). Enough to confirm blocks thread and to catch the
  random-token emission, not enough for a frequency estimate with a real confidence interval.
- **No constrained anything.** No automaton was compiled for a real task, no tree was built. That
  is Phases 1–3.
- **`outlines-core`'s JSON-schema limitations (§4.2) were not re-verified** — those remain `[V]`
  from the original source reading, and Phase 1 must test them.
