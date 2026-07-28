# docs/LOG.md — running record

Newest last. One entry per working session.

---

## 2026-07-27 — Phase 0

Started from a bare repo: `SPEC.md` + `CLAUDE.md` + `env.sh`, empty `docs/`, empty venv.

### Environment

- Machine is **dedicated** (`DGFA_DEDICATED=1`, `DGFA_MAX_PHASE=6`): H100 80GB HBM3, idle at start,
  26 threads, 221 GB RAM, 2.7 TB free. CLAUDE.md's shared-box section does not apply here and
  `env.sh` correctly sets preallocation **on** — recorded in `docs/ENV.md`.
- Installed into the project venv: `jax[cuda12]` 0.11.0, `gemma` 4.1.0 from git main,
  `outlines-core` 0.2.14. SPEC §1.1 confirmed — the git install ships `gemma/diffusion/`, PyPI
  4.0.1 does not.
- **New dependency note (per CLAUDE.md's rule):** no dependencies were added beyond the intended
  stack. `gemma` drags in a very large transitive tree including two TensorFlow distributions
  (`tensorflow` 2.20.0 and `tensorflow-cpu` 2.21.0) side by side. Nothing in the diffusion path
  imports TF and nothing has broken; noted in `docs/ENV.md` in case it does.
- Checkpoint pulled to `artifacts/ckpt/` — **37.63 GiB**, not SPEC's ~47.4 GB. ~2.5 min at
  415 MiB/s, `commit_success.txt` present, no errors. `logs/ckpt_download.log`, PID 6522 (finished).

### Work done

Read `gemma/diffusion/{_sampler,_early_stopping,_transformer,_chat_sampler,_models,_paths}.py`,
`gemma/gm/text/{_sampler_loop,_sampler,_template}.py` in the installed tree. Ran five scripts under
`scripts/` (see `docs/PHASE0_FINDINGS.md` for the index). Background runs:

| PID | job | log |
|---|---|---|
| 6522 | checkpoint download | `logs/ckpt_download.log` |
| 11699 | 2-prompt smoke probe | `logs/phase0_smoke_probe.log` |
| 13610 | 20-prompt §1.4 baseline | `logs/phase0_smoke.log` |
| 14391 | 3-prompt multi-block probe | `logs/phase0_multiblock.log` |

All finished; none left running.

### `[?]` resolved by measurement

- **Open question 7 (budget-path frequency) — CLOSED, and mis-framed.** 31/31 blocks exit via early
  stop, 0/31 via the 48-step budget; median ~12 executed steps. But random tokens still reached the
  emission in 1/31 blocks, because early stop does not gate the emitted canvas. See below.
- **Open question 5 (compilation throughput) — CLOSED.** 0.13–0.65 s per realistic schema, not
  4–8 min. BFCL-Live projects to ~0.9 h serially. The vocabulary pre-filter is unnecessary; do not
  build it.
- **§1.3(4) thought marker — RESOLVED.** `<|channel>` = 100 and `<channel|>` = 101 *are* single
  dedicated token ids. §3.6's two-state construction survives on different tokens than assumed.
- **§1.2 `cache_info` — RESOLVED.** It is a `@property`, not a field.
- **§1.3(1) vocab padding — RESOLVED, and it does not exist.** 262,144 real pieces, no dead tail.

### Corrections made to SPEC.md (in place, marked `[V-P0]`)

§1.1 checkpoint size and HBM residency · §1.2 whole checklist ticked, plus four new findings ·
§1.3 all four items (two were wrong) · §1.4 measured baseline table · §3.1 **the emission invariant
was false** · §3.1b a fourth termination path (cache exhaustion) · §3.6 rewritten against the real
channel format · §4.3 state-id stride 64 not 8 · §4.4 `K_max` 256 → ~1,100 · §4.7 **retracted, wrong
by ~1,000×** · §5.3(c) **`_sample_step` does need forking** · §5.6 measured memory headroom ·
§9 open questions 5 and 7 closed.

### The two that change design decisions

1. **§3.1's invariant is false.** `should_stop` is computed from `previous_canvas` and `logits`, so
   it certifies the step's *input*; the emitted canvas is gated on the *old* `carry.done`, so the
   step where early stop fires still emits its own fresh sample including unaccepted positions.
   Measured: 1 of 31 blocks emitted a uniform random token despite early-stopping. J0 is still the
   right answer, but the J2→J0 gap on this model will be **small** — do not oversell it.
2. **`_sample_step` must be forked.** `sample_next_canvas` never receives `state`, and
   `max_new_tokens` is not a field of `SamplingState` at all — it lives only in `_sample_loop`'s
   closure. So neither `A_k` nor `R` can reach the constrained sampler without forking
   `_sample_step` and widening the state at the entry point.

**`CLAUDE.md` was amended for (2)** — its "Hard invariant" paragraph asserted the opposite, and a
future session following it would design into a dead end. The inference-only invariant itself is
untouched; only the fork-surface sentence changed, marked with a Phase-0 note.

### Not done — do not assume coverage

No benchmark data downloaded (BFCL/xLAM/Spider/GSM-Symbolic/Countdown/Sudoku all untouched; the
§4.7 timings use hand-written schemas). Everything ran at `B=1`. `near-greedy` temperature never
executed. Only 11 multi-block blocks observed. No automaton compiled, no tree built, no constrained
generation. §4.2's outlines limitations were not re-verified.

### Next

Phase 1 — `compile/`. First task should be downloading real BFCL schemas and re-measuring §4.7's
projection against them before quoting the 0.9 h figure.

---

## 2026-07-27 — Phase 1 (same session)

Built `diffgemma_fa/compile/`: `schema.py`, `vocab.py`, `lift.py`, `minimize.py` (Valmari),
`classes.py`, `automaton.py`, `pipeline.py`, `validate.py`, `bfcl_data.py`, `tasks/`. 256 tests.
Full write-up in `docs/PHASE1_FINDINGS.md`.

### Package layout — a deliberate deviation from SPEC's tree

SPEC §8 draws `compile/`, `infer/`, `model/`, `eval/` directly at the repo root, but CLAUDE.md's
commands are `python -m diffgemma_fa.compile.tasks.bfcl`. Those are inconsistent: the module path
requires `diffgemma_fa` to be an importable package. Resolved in favour of the **commands**, which
are what actually get run: the package lives at `diffgemma_fa/compile/…` inside the repo, so the
documented commands work from the repo root with no install step, and `tests/` and `docs/` stay
outside the package.

### Data

`ShishirPatil/gorilla` sparse-cloned to `artifacts/data/gorilla` (16 MB, gitignored). SPEC §7.1
confirmed: data is in git not HF, and v4 renamed `simple` → `simple_python`. **BFCL-Live is exactly
1,351** (257+1,052+15+23 across the four live splits) — SPEC's figure is right; `live_relevance`
(16) and `live_irrelevance` (884) are scored separately.

### Three correctness bugs, none self-announcing

1. **Reserved tokens leak into the grammar alphabet.** `<turn|>` (106) and `<|tool_response>` (50)
   are *not* SentencePiece control tokens, so a JSON string body can match their plain-ASCII bytes.
   The grammar could then emit a stop token mid-value → canvas truncated mid-grammar → empty
   `A_{k+1}`. Fixed via `vocab.RESERVED_TOKENS`.
2. **`T[final][eos]` had to be stripped, not just ignored.** SPEC §4.3 warns the edge exists;
   leaving it makes every augmented automaton spuriously nondeterministic and silently drops eq (8)
   off its fast path. Fixed in `lift.py`. Now 256/256 real BFCL schemas are genuine DFAs.
3. **BFCL's `any` cannot pass through to outlines** (`ValueError: Unsupported type: any`, 11 of
   4,549 schemas). Maps to a typeless schema now.

### `[?]` resolved by measurement

- **§4.7 on real data: 4,549 schemas, median 0.42 s, total 37.9 min serial.** Confirms Phase 0's
  ~0.9 h order of magnitude against SPEC's original 4–7 days. Pre-filter still unnecessary.
- **Open question 9, both halves.** Edge-level class dedup ratio **1.68**; post-lift minimization
  **1.11× mean, 9.29× max** — the second pass is usually marginal, occasionally dramatic.
- **`K_max`: measured `|N_c|` = 987–1,585**, so SPEC's 256 default is ~6× too small. Default is now
  1,100 and is auto-derived per grammar.
- **`states_max = 3,573` on BFCL-Live**, above the paper's quoted 2,459 and above any usable tree
  bucket → SPEC §5.6's chain fallback is real, flagged via `needs_chain_path`.

### A measurement trap worth remembering

Ground-truth acceptance first read 82.5%, which looks exactly like grammar over-constraint. It was
not: BFCL wraps **every leaf at every nesting level** in a list of acceptable values, and I was
unwrapping only the top level. Recursive unwrapping → 93.8%. Do not report an over-constraint rate
without first checking the harness.

### Final Phase 1 exit numbers

| | |
|---|---|
| BFCL-Live full compile | **4,549 / 4,549, zero failures, all DFAs**, 42.7 min on 12 cores |
| FA → random walk → `json.loads` | **5,157 / 5,160** (the 3 are a sampler UTF-8 artifact, not the grammar) |
| ground truth → FA | **255 / 258 = 98.8%** |
| remaining 3 rejections | 2 × `type: any` wildcard, 1 × nested array-of-objects — both shapes SPEC already flags |
| tests | **270 green** |

### Not done

Round-trip covers `live_simple` only; the `multiple`/`parallel` union and call-list grammars are
built and unit-tested but not round-tripped. The Python-format grammar is not round-tripped against
ground truth, and its `d=2` depth-bound coverage is unmeasured. Spider is not implemented (NFA,
and §5.6 says dense is infeasible). xLAM/GSM-Symbolic/Countdown/Sudoku grammars exist but no
dataset is downloaded. All listed in `docs/PHASE1_FINDINGS.md` §5.2.


---

## 2026-07-27/28 — Phases 2 and 3 (same session, autonomous)

Full write-ups in `docs/PHASE2_FINDINGS.md` and `docs/PHASE3_FINDINGS.md`.
**651 tests green in 2m33s.**

### Phase 2 — `infer/reference.py` + the exactness harness

SPEC §6.1's eleven tests, all green on DFAs **and** NFAs, **zero skips** (254 exactness cases + 18
numerics). Fixed seeds, Bonferroni-corrected χ² (α = 0.001/200), documented re-run protocol.

**One real bug, and it is the landmine SPEC warns about.** `is_dfa` in the conventional sense is
the *wrong* gate for eq (8)'s cheap `∃` token draw. Two **parallel** edges to the **same**
destination with overlapping labels are deterministic by every ordinary definition, yet give that
token multiplicity 2 — so the `∃` form draws from the wrong distribution on an automaton every
conventional check calls a DFA. `reference.py` now separates `is_deterministic` from
`has_unit_multiplicity` and gates on the latter. Compiled artifacts are safe regardless, because
`_group_edges` collapses to one edge per `(src,dst)` with the union of labels — but that is a
property of the *grouping*, not of determinism.

### Phase 3 — the JAX kernels

**SPEC §0's central claim reproduced on an H100.** Counted on the optimized HLO: the tree emits
**zero** `while` loops and **exactly `log₂ L`** matmuls (4,5,6,7,8 for L=16…256); the sequential
`lax.scan` emits **one** `while` hiding all L products. Pinned in `tests/test_kernel_shape.py`,
which asserts on compiled HLO rather than a profiler — deterministic and 5.5 s.

**SPEC §0's FLOP table confirmed to two significant figures**: 385→2.85% (predicted 2.9%),
512→6.71% (6.7%), 1024→53.69% (53.7%).

**The finding that changes a design decision: the FLOP ratio is ~43× pessimistic as a latency
proxy here.** Against the measured ~0.21 s denoising step, the tree costs 0.50% at |S|=385, 1.84%
at 1024, and **9.95% at |S|=2048 where the FLOP ratio says 429%**. At B=1 the model forward is
memory-bound (~50 GB of weights) while the tree is a compute-bound GEMM. **Set the §5.6 dispatch
threshold from measured wall-clock, not FLOPs** — the tree path is viable across the whole usable
|S| range on this hardware, and with BFCL's max |S|=595 the chain fallback is unreachable in
practice. Caveat recorded: the 0.21 s baseline includes cache-append overhead, so these are lower
bounds; Phase 5 §7.3 must re-derive them against an isolated forward.

### Two coverage gaps closed rather than tolerated

The complement-aware tests were skipping whenever a random instance had no negated class — and on
DFAs that was *always*, because the generator scattered tokens too thinly for any state pair to
reach `> V/2` labels. The complement path had zero DFA coverage. Generator fixed, skips replaced by
assertions. Separately, the tree-vs-posterior test made 40,000 individual JAX dispatches and took
19 minutes alone; `vmap` over the key batch makes it seconds.

### Next

Phase 4 — `model/`. J1 before J0 (SPEC §5.4). Note Phase 0's correction: `_sample_step` **must**
be forked, because `sample_next_canvas` never receives `state` and `max_new_tokens` is not a field
of `SamplingState`.


---

## 2026-07-28 — Phase 4 (autonomous)

`diffgemma_fa/model/`: widened `SamplingState`, forked `_sample_step` and denoising loop,
constrained emission. Full write-up in `docs/PHASE4_FINDINGS.md`. **670 tests green.**

### Exit criterion met

10/10 end-to-end constrained generations on the real 26B-A4B checkpoint with real BFCL-Live
grammars, **all accepted by an independent simulator**. Stop-token positions **4–13, never near
255**, so SPEC §3.5 trap 4's own diagnostic passes and the unscored `ACC --Σ--> ACC` tail works.
Repeat generations on the same `|S|` bucket take 9.6 s against ~38 s cold — the
no-recompile-per-request property of §5.3(b) showing up in the timings.

### Two findings

1. **Per-node normalization is not enough, and our own §3.5 fix is why.** The unscored
   `ACC --Σ--> ACC` edge has emission mass exactly 1.0, so it **pins the root's max at 1.0** and
   normalizing by it is a no-op — while genuine grammar paths sit at ~1e-49 (smallest positive
   entry measured: **1.2e-288**). fp32 underflows to exactly zero and the draw degenerates
   *silently*, returning plausible multilingual text. float64 fixes `L=64` but **not `L=256`**:
   sampling degenerates on 2 of 6 real prompts there. **The sum-product tree needs log space**, as
   §2.7 already chose for MAP — which is why MAP is unaffected and is currently the only working
   emission. `sample_tokens` now returns a `valid` flag so this can never be silent again.
2. **Joint MAP prefers the minimal completion.** Emissions are schema-valid but empty
   (`{"location":""}`), because every extra token multiplies in a probability < 1 and the grammar
   admits `""`. Not a bug — but a `CS = 100%` number from these runs would be true and nearly
   meaningless, which sharpens §3.8's existing warning.

### Next

Phase 5 needs the log-space tree first if `--emission=sample` is to be evaluated at all. J0, the
widened `EarlyStopFn` (§3.1b closure 2 is **not** yet enforced), and multi-block end-to-end runs
are all outstanding — see `docs/PHASE4_FINDINGS.md` §5.


---

## 2026-07-28 — `tests/test_guarantee.py`, and the bug it caught

CLAUDE.md names this file "the one thing that must not break"; it did not exist until now.
**700 tests green.**

It asserts the two *different* things CLAUDE.md warns against conflating — per-block viable prefix
(`δ*(A_k, canvas_k) ≠ ∅` **and** `⊆ {s : d(s) ≤ R}`) and, separately, acceptance of the
**concatenation** — plus a test asserting from the other side that a non-final canvas is *not*
accepted on its own, so anyone who adds a per-canvas acceptance assertion finds a test explaining
why it is wrong.

### It immediately caught a real bug: `b_L`'s budget was off by one canvas

SPEC §3.5 writes `R ← max_new_tokens − state.step` and applies `b_L(s) = 1[d(s) ≤ R]`. But
`state.step` counts tokens committed **before** the block, while `b_L` is evaluated at the state
reached **after** its `L` tokens. The correct terminal budget is
`max_new_tokens − state.step − canvas_length`.

Unadjusted it is too permissive by exactly `L`, so it admits states that cannot finish — and
generation then **never terminates**, because nothing forces completion as the budget runs down.
The test failed with "did not terminate in 8 blocks" and with reached states whose `d(s)` exceeded
the real remainder. Fixed in `model/state.py: terminal_budget`; SPEC §3.5 trap 1 corrected.

This is exactly the class of bug the file exists for: every single-block run passed happily before
and after, and only a multi-block assertion on the *budget half* of the viable-prefix property
exposed it.

### Honest gap

§3.1b **closure 2** (the automaton-aware stopping conjunct fed `emit_canvas`) is **not** enforced in
the sampler — the stock `early_stop_fn` is used unchanged. `test_closure_2_is_not_yet_enforced`
marks this deliberately rather than letting the file imply coverage it lacks, and the CS test
asserts termination-in-an-accepting-state explicitly instead of assuming it.


---

## 2026-07-28 — J0, and a hypothesis I disproved

Implemented **J0** (SPEC §5.4's `_ConstrainedCarry` with a separate `emit_canvas`): the trajectory
keeps stock uniform renoising so the model's inputs stay on its training distribution, while the
emission is the constrained MAP/draw over the model's **real** marginals. 6/6 accepted end to end.
`variant=j1/j2` with `emission=map` is now rejected on SPEC §3.9's grounds — those variants define
the emission to *be* a draw.

**A hypothesis I had and disproved, recorded so nobody re-runs it.** I attributed the minimal
outputs (`{"location":""}`) to J1's flattening: every non-accepted position becomes near-uniform at
1/262144, which looked like it would make any content ruinously expensive for a joint MAP. **That
is wrong** — J0-map, whose emission uses the real marginals, gives the same minimal outputs. The
remaining candidates are the genuine length bias of a joint MAP over a variable-length language,
the prompt, or the uncalibrated entropy bound (SPEC §3.4, still outstanding). **Phase 5 must not
quote an accuracy number before this is understood**; the docs and the code comment have been
corrected rather than left with the tidy but false story.

700 tests green.


---

## 2026-07-28 — SPEC open question 2 resolved: J0 vs J1

Added per-denoising-step instrumentation to the constrained sampler (`DIAGNOSTICS`, a module-level
sink fed by `jax.debug.callback` — there is no Python between steps, so nothing else can see it).

**J0 is the variant that fails, inverting SPEC's prior of "J0 — ship this".** Same prompt, same
grammar, same entropy bound:

| step | J0-map | J1-sample |
|---|---|---|
| 12 | `{"user_id":0}` | `{"user_id":17890}` |
| 14 | `{"user_id":1}` | `{"user_id":77890,"special":":black"}` |
| 21–47 | flips 0/1/6/8 **at H = 0.0000** | stable |

Ground truth `7890` / `black`. The tell is the J0 column at H = 0.0000: a maximally confident model
whose constrained MAP is arbitrary means the confidence is about something else — and it is. Under
J0 the trajectory is stock uniform renoising, so the model never sees JSON, converges on a
natural-language answer, and leaves its marginals peaked on tokens the grammar forbids. The
emission then has no signal but length. **J0 buys an unconditional guarantee at the cost of all
feedback** — exactly what §3.7 means by "its entire value is the self-conditioning feedback".

This also **retires the Phase 4 hypothesis** that MAP's length bias was the cause: J1 uses the same
joint machinery on the same grammar and does not collapse.

Ship J1, or J0 + `--self-cond=constrained` (§3.7's `logit_shaper` mask, still unimplemented).

### Process note

Two self-inflicted errors worth recording. I launched two 51 GB model instances concurrently on an
80 GB GPU and both died on `Failed to initialize BLASLT support`; and I twice wrapped a long GPU
run in a foreground wait that my own tool timeout then SIGTERM'd. Long GPU runs now go out under
`setsid` with a separate watcher.
