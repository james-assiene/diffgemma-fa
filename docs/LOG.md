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


---

## 2026-07-28 — SPEC §3.4 swept; a negative result, and the real cause chain

Full write-up in `docs/PHASE5_FINDINGS.md`. **703 tests green.**

**The entropy bound does not explain the accuracy.** Over the full §3.4 grid on 12 `live_simple`
records: **CS = 12/12 at every bound** (the guarantee is structural), and argument accuracy is 1–2
of 19 everywhere — statistically indistinguishable rows. Stock `0.1` is as good as anything. SPEC
§3.4's premise that the defaults "will silently produce garbage" is **not confirmed** on this slice.
Coverage stated: 12 records, one split, and `entropy_threshold` not swept.

**SPEC §4.2 is wrong about four keywords.** `minLength`, `maxLength`, `minItems`, `maxItems` are
**enforced** by outlines-core 0.2.14, not dropped; only the numeric bounds are dropped. Corrected in
`_SILENTLY_DROPPED` — keeping them there forces callers to `allow`-list working constraints and
desensitises the fail-loud signal.

**The cause chain, three steps, two fixed:**

1. **J0 starves the emission** (open question 2, resolved — see the previous entry).
2. **The empty string is a self-consistent fixed point.** Under J1 the model sees `{"location":""}`
   in its own canvas and confirms it; BFCL schemas never carry `minLength` so the grammar permits
   it. `require_nonempty_strings` → non-empty args **7/12 → 12/12**.
3. **The content is misplaced, not missing — unfixed.** Accuracy did not improve, but one output
   contains the ground truth verbatim inside junk:
   `{"loc":":{{    loc_\":221B Baker Street, Berkeley, CA, USA1  ",...}`. So the model *has* the
   answer and the decode places it badly. Hypothesis (untested): marginals are positional over a
   fixed 256-canvas while the grammar's fields start wherever the previous tokens end, and J1's
   draw changes field lengths between steps, so alignment never settles. That would be a real
   tension between variable-length grammars and fixed-canvas diffusion, not an implementation bug.

**No BFCL number should be quoted from this work until (3) is resolved.** Format is solved (12/12
CS, parsed, non-empty); content is 5–10% and the reason is characterised, not fixed.

## 2026-07-28 — review-driven fixes, then the Phase 5 baseline sweep at scale

Three review agents (infer/, compile/, test-teeth) reported against the tree at
721 tests. Everything below was verified by injecting the mutant and watching a
new test go red, not by reading code.

**Correctness-critical, fixed:**

1. `Z == 0` was undetectable end to end. `categorical`/`argmax` are
   shift-invariant, so an all-sentinel root gives a confident-looking draw from
   a provably empty language (measured: `valid == True` on 200/200). Root-mass
   predicate added, threaded out through `ConstrainedSamplingState.feasible`,
   raised as `ZeroPartitionError` outside the jit.
2. `advance_states` substituted a stale carry for an empty state set, making
   SPEC §3.1b closure 2 **unsound**. Fixed together with (1) per the
   researcher's sequencing ruling.
3. `normalize_bfcl_schema` deleted any BFCL parameter named `description`,
   `default` or `optional` — the keyword-drop was applied to the `properties`
   map's *keys*. 11 top-level occurrences, 5 required.
4. `minimize()` silently changed the language on a duplicate transition
   (Valmari's `mark()` has no re-mark guard).
5. `--variant=j2` was never implemented — it ran J0's emission, so the §7.2
   baseline table had a duplicated row.
6. Cache bound off by one: gemma's `is_full` is `end_index >= total - 1`.

**Test gaps closed:** eq (8)'s multiplicity (both NFA generators had zero
power), the vacuous `Σ_v q_i == 1` (replaced by the per-position `log Z`
invariant), `terminal_budget` (called by no test), automaton-stays-traced
(measured by XLA's compile counter).

**Running:** `scripts/phase5_baselines.sh 130 bfcl_live_simple`, PID 277316,
log `logs/phase5_baselines_130.log`. Six arms x 130 records; ~34 s/record
measured, so ~7.5 h. Stale artifacts moved to `artifacts/stale_pre_e559e31/` —
they predate the schema fix, the J2 implementation and the cache bound, so they
are not comparable and must not be merged into the table.

**Sweep restarted (2026-07-28, ~18:15).** Killed 20 records into arm 1 and
relaunched at commit `e5a0150`. Reason: the `unconstrained` arm reported
`CS = 0.000`, which is partly an artifact — `cs_rate` scores the emitted token
sequence against an automaton carrying SPEC §3.6's channel header, and the
stock model has no reason to emit that header. Added `schema_valid_rate`
(header- and tokenizer-independent) as the fair cross-arm column, and stopped
truncating `rows[].text` at 220 chars. Better to lose 20 minutes than to spend
7 hours producing a table with a known measurement artifact.

Stock model on 6 records: CS 0.000, schema valid 0.500, arg accuracy 0.833. The
claim under test is therefore "50% -> 100% schema validity at some accuracy
cost", not "0% -> 100%".

## 2026-07-29 — kernel underflow found via the Z==0 detector; constrained arms re-running

`live_simple_106-63-0`'s Z==0 was neither cause (a) nor (b): the automaton is
sound (d0=73, simulator accepts shortest+PAD*183) and the budget ample. Cause:
`log_matmul`'s single row/col shift underflows when the unscored tail pins the
shift at 0 while genuine paths sit ~850 nats below — contributions fall under
float64's subnormal floor and the (start, ACC) entry becomes sentinel. Fixed
with a two-band shift (4 GEMMs/combine), commit fe01dbb. All constrained n=130
arms were measured with the buggy kernel → re-running j0/j1/j2 + j0-map from
scratchpad/sweep2.sh, log logs/phase5_rerun_fixed_kernel.log. Old artifacts
moved to artifacts/stale_underflow_pre_fe01dbb/. unconstrained + mask arms do
not touch this kernel and stand.

Researcher panel (3 agents) reports delivered; synthesis pending the grammar
lens. Key adjudications so far: only 130 of 258 live_simple records ever
evaluated (must be stated in every table); the "entropy_bound ruled out" claim
retracted as underpowered (19 keys, <10% power); the 0.641→0.385 gap
contradicts the paper's direction in all 40 of its cells → port-configuration
effect until proven otherwise, prime suspects Mar (unwired) and the
prompt/rendering contract.

## 2026-08-03 — Countdown and Sudoku: the tight-grammar regime

Added `compile/tasks/{countdown,sudoku}.py` and `eval/run_tasks.py`. The
grammars already existed in `tasks/grammars.py`; what was missing was data,
scorers and eval integration.

**Why these two.** Every accuracy result so far is from BFCL, whose grammars
are large permissive JSON schemas — |S| 128-1024, where §7.3 puts the tree at
up to 82.6% of a model forward and the automaton barely narrows anything.
Countdown and Sudoku compile to **|S| bucket 64**, where the tree is 0.2% of a
forward and the constraint eliminates almost the whole output space. If the
whitespace finding (0.328 -> 0.628 on BFCL) was an artifact of JSON rendering
rather than something general about grammar-tokenizer alignment, it should
fail to appear here.

**Dependency added:** `datasets` 5.0.1, which CLAUDE.md already lists in the
intended stack. Installed in the venv only. Countdown's test slice
(TinyZero's `range(327680, 328704)`, 1,024 rows) is materialised to
`data/countdown_test.jsonl` so a long run never depends on the network.
Sudoku needs no download at all — SPEC records there is no canonical source,
so puzzles are generated synthetically with uniqueness verified by exhaustive
solve.

**The scorers are not format checks.** A grammar guarantees an answer's shape
and nothing about whether it is right: `countdown_regex` admits `1+1=3`, and a
format-only Sudoku grammar would let a model overwrite the givens and solve a
different puzzle. Both are re-derived and tested against exactly those two
failure modes.

Queued behind the seed-variance sweep: 4 arms (unconstrained + j0-map on each
task) at n=250.

## 2026-08-08 — grammar audit D1–D4: the whitespace bound was a nesting-depth bound

`tests/test_audit_grammar.py` (tester agent) against `compile/schema.py`,
`compile/automaton.py`, `compile/pipeline.py`, `compile/tasks/bfcl.py` and
`eval/run.py`. Four defects, all the same shape as the 0/130 whitespace failure:
plausible code, no independent expectation.

**D1. `JSON_WS = [ \t\n\r]{0,8}` was silently a nesting-depth bound.** It bounds
a whitespace *run*, but a pretty-printer emits `newline + indent*depth`, so it
binds at `indent*depth >= 8`. Measured: a flat 2-key object 5/5 renderings, an
object with **one array property** 4/5, nesting depth 4 **3/5**. The original
5/5 measurement in the docstring was taken on a flat 3-key schema — a claim
validated only on a toy shape. BFCL v4 carries 959 `array` and 9,464 `dict`
occurrences.

Fixed to `[ \t\n\r]*`, RFC 8259's own definition. The bound was **not** buying
finiteness — a Kleene star over a 4-character class is one DFA state and `{0,8}`
is an eight-state counter — so the correct pattern is also the cheapest. `|S|`
after minimisation, `channel_header=False`, three shapes:

| whitespace | flat | +array | 3-key BFCL | renderings |
|---|---|---|---|---|
| outlines' `[ ]?` | 42 (b64) | 55 (b64) | 71 (b128) | 2/5 |
| `PRETTY_WS` | 90 (b128) | 121 (b128) | 143 (b256) | 3–4/5 |
| `{0,8}` (was) | 98 (b128) | 132 (b256) | 155 (b256) | 4–5/5 |
| `[ \t\n\r]*` (now) | **34 (b64)** | **45 (b64)** | **59 (b64)** | **5/5** |

Tree at `L=256` drops 0.033–0.134 GB → 0.008 GB. A widened *constant* was the
obvious alternative and is the wrong one: `{0,24}` covers indent-4 to depth 5,
still fails at depth 6, and costs 226/308/347 states — walking toward §7.3's
`|S| ≈ 512` cliff. **This reverses `docs/RESULTS.md`'s "the whitespace-tolerant
grammar roughly doubles |S|, moving most records up one bucket":** with `*` the
tolerant grammar is *smaller* than the stock one. The §7.3 bucket mix must be
re-measured before that sentence is repeated.

**D2. The fix was unreachable from the eval CLI.** `eval/run.py` passed
`whitespace_pattern=(PRETTY_WS if args.whitespace == "pretty" else None)`, and
`None` does **not** mean "the pipeline default" — `build_regex` omits the kwarg
and outlines' own `[ ]?` is used. So `--whitespace stock` (the default) gave
2/5 and the repaired default was not selectable: **every eval arm to date ran a
grammar with a known over-constraint.** Replaced by an explicit
`WHITESPACE_PATTERNS` map with a new default `--whitespace json` (`JSON_WS`);
`stock` and `pretty` stay selectable to reproduce the arms already in
`RESULTS.md`. `build_regex(None)` deliberately still means "outlines' own" —
`test_schema.py` pins it as the negative control.

**D3. The build gate was unwired.** `verify_renderings` was never passed at
either call site, so the check that turns "the grammar must accept how the model
writes" into a build failure sat unused beside the defect it was written for.
Added `schema.synthesize_instance`, which builds a minimal instance from the
schema itself (no model, no data, ms), and wired both `tasks/bfcl.py` and
`eval/run.py` unconditionally. Measured over all **4,549** BFCL-Live schemas:
0 synthesis failures, and **11 schemas whose grammar rejects all five renderings
of their own instance**. Those 11 are a genuine, previously invisible defect,
not a gate artifact: a property whose schema is BFCL `any` compiles to an
**unparenthesised top-level alternation**, e.g.
`\{ws"input_value"ws:ws((true|false))|(null)|(...)`, so the object regex is one
branch among seven and the grammar accepts a bare `null` as a whole answer.
`live_simple_117-73-0` is in the default split. They now fail the compile loudly
and are counted in `skipped_by_reason` instead of being used. **Not fixed here —
it is outlines' emission and out of this change's scope. [?] open.**

`accepts_all_renderings` now renders with `ensure_ascii=False`: the grammar is a
byte-level regex over what the tokenizer emits (`실행`, not `실행`), and
Python's default spelled 17 Korean-enum schemas as escapes the regex has never
seen and reported a renderer artifact as a grammar defect.

**D4. `is_dfa` honesty was blind across distinct classes.**
`_assert_structural_invariants` keyed determinism on `(src, edge_class)`, so two
edges out of one state whose label sets **overlap without being identical**
intern to different class ids and loaded clean claiming `is_dfa=True` —
licensing eq (8)'s `∃` fast path on an NFA, where it is off by ~1.7e-2 (§2.6).
Now checked on tokens, using the stored (small) side of each class: pos×pos by
duplicate detection, pos×neg by `P \ N`, neg×neg by `V − |N1 ∪ N2|`. 18 ms on a
213-state / 684-edge BFCL grammar, and no false positive on any freshly compiled
automaton (`_group_edges` already derives `is_dfa` token-level; the hole was only
on the load path).

**One test in the suite is unsatisfiable and is left red for adjudication:**
`test_every_whitespace_setting_the_eval_cli_offers_accepts_all_renderings`
hardcodes `{"stock": None, "pretty": PRETTY_WS}` and requires both to accept 5/5,
while `test_schema.py::test_json_ws_accepts_every_standard_rendering` requires
`None` to reject indent2+tabs and
`test_audit_grammar.py::test_the_shipped_pretty_pattern_does_not_survive_the_gate`
requires `PRETTY_WS` to reject tabs. No implementation satisfies all three. It
failed on its own "grew a --whitespace choice this test does not model" guard.

> **Resolved, 2026-08-08.** Adjudicated in review and amended by the tester to
> `test_the_whitespace_setting_the_eval_cli_defaults_to_accepts_all_renderings`,
> which reads the mapping from the shipped `WHITESPACE_PATTERNS` and asserts the
> satisfiable form of the same property: the **default** accepts 5/5 on a flat,
> an array-bearing and a nested schema, every non-default choice is one of the
> declared-historical patterns, and none of them is silently selectable. The
> amended file is green.

### 2026-08-08 — D1–D4, reviewer round 1

Reviewer reproduced every `|S|` number, all 11 `any`-type gate failures with
zero false positives, and the D4 set algebra including the three polarity
branches the tester never exercised. One substantive objection sustained, plus
four corrections. All addressed:

**The `verify_strict=False` hatch was unscoped.** It suppressed *every* gate
failure, not just whitespace narrowness, so under `--whitespace pretty` the
`live_simple_117-73-0` defect — the very thing the gate had just been credited
with catching — walked straight through, gated on a flag with nothing to do
with its failure. Now scoped: before downgrading, `pipeline` recompiles the same
schema under `JSON_WS` and re-runs `accepts_all_renderings`; it downgrades only
if the **wide** grammar passes, and otherwise raises with the *wide* failure
list so the raise cannot be mistaken for the caller's chosen policy. Verified:

```
live_simple_117-73-0  --whitespace json   REFUSED     (5/5 renderings rejected)
live_simple_117-73-0  --whitespace pretty REFUSED     <- was COMPILED
live_simple_48-21-0   --whitespace json   COMPILED
live_simple_48-21-0   --whitespace pretty COMPILED + [gate] report on ['tabs']
```

The principle: the gate may be lenient about a narrowness the caller
**declared**; it may not be lenient about one nobody declared.

**Two records inside every published cut.** `live_simple_117-73-0` and
`live_simple_122-78-0` sit at index **117** and **122** of the 258 single-function
records of `BFCL_v4_live_simple.json` — inside the first-130 prefix behind every
arm in `docs/RESULTS.md`. Two records per arm (≤1.5%) were scored against a
grammar that cannot produce a well-formed object at all. Small, but it is not
zero and it was invisible until the gate was wired.

Note the interaction with the `denominator_policy` landed in the same file by
the measurement work: compile-skipped records are excluded from `n`. So a future
`bfcl_live_simple` arm reports **n = 128, not 130** — those two records move from
"scored, and could never score" into `skipped_by_reason`. That is the right
direction, but it means the new `n` is not the old `n` and the two must not be
compared without saying so.

**The string regex has no `\uXXXX` escape production.** Verified directly:
`{"s": "café"}` (raw UTF-8) is accepted, `{"s": "caf\u00e9"}` (the escape
`json.dumps` emits by default) is rejected; the body is
`([^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt])*` and `\\u` simply is not in the
alternation. So the grammar is narrower than RFC 8259 for **every** string, not
only for the 17 Korean-enum schemas — the `ensure_ascii=False` change to
`accepts_all_renderings` is therefore load-bearing rather than cosmetic.
Pre-existing outlines limitation, **[?] open, not fixed here.**

**`synthesize_instance` does not promise a schema-valid instance.** Validated by
the reviewer against `jsonschema`: **16 of 4,549** fail strict validation, all of
them a schema whose own `enum` contradicts its own `type`
(`{"type": "boolean", "enum": ["True", "False"]}` -> `"True"`). `enum` outranks
`type` in outlines' precedence, so `"True"` is exactly what that grammar
accepts and none of the 16 caused a gate failure — the gate must model the
grammar, not an idealised schema. A type-consistent member is now preferred
where one exists; in these 16 none does, so the count is unchanged at 16 and the
docstring says so instead of claiming otherwise.

Also corrected: the `--whitespace` help text referred to a `--no-verify-renderings`
flag that does not exist, and the `eval/run.py` OOM comment still said "the
whitespace-tolerant grammar roughly doubles |S|", which this change reverses.

## 2026-08-08 — measurement audit A–F: the numbers were computed over the wrong records

Tester→coder→reviewer, measurement/scoring slice. `tests/test_audit_measurement.py`
(34 tests) was written from BFCL's own vendored checker, the compiled automaton,
and emissions copied verbatim out of `artifacts/` — never from the code under
test. 20 of 34 failed on arrival. All 34 pass.

Six findings. Four are the same defect wearing different clothes: **a number
computed over a denominator that depends on what the model emitted.**

**[A] `metrics.normalise` was not BFCL's rule, in four ways at once.** SPEC §4.8
quotes BFCL's docstring as "lowercases and strips `",./-_*^`" and the outer `"`
there is the *quotation delimiter*, not a member of the set — whose first
character is a **space**. We stripped the quote (BFCL does not), kept the space
(BFCL strips it), had no single-quote fold, and applied the string rule to every
type. BFCL normalises **only** strings (`string_checker`); everything else goes
through a bare `value not in possible_answer[param]` after a `type_checker` that
rejects a type mismatch outright. Fixed against
`artifacts/data/gorilla/.../ast_eval/ast_checker.py:174`, which is vendored in
this repo — the docstring was never the authority and should not have been used
as one.

**[B] `Scores.add` returned before incrementing `arg_total` when the output did
not parse.** Records that produced nothing left the denominator while staying in
`n`. `mask-sample` therefore reported the **highest** `arg_accuracy` of any arm
(0.508) and nearly the lowest exact-call rate, in the same row, because 63 of its
130 records were removed from its denominator and from nobody else's.

**[B2] The same defect one level up, and larger — found only after [B] was
fixed.** `eval/run.py`'s OOM and `ZeroPartitionError` handlers passed
`want=None`, dropping those records from `arg_total` too. `exp_e5_grammar130_j0`
has **70 of 130 records** fail to generate (`Z == 0` on the float32 path); with
[B] alone its `arg_accuracy` *rose* to 0.732 and it stayed best-in-table while
having failed on 54% of its records. Both handlers now pass the record's real
`want` (`materialize_ground_truth` hoisted above the generation `try`). Their own
comments already said this was the intent — *"the record scores as a failure"*,
*"it is never swallowed"* — and the code did the opposite. Compile-skipped
records are a different case and were already right: the model was never asked,
so they leave `n` and are reported in `skipped_by_reason`. The
`denominator_policy` string in both harnesses now states both clauses, because
the first version of it asserted a policy the code did not implement.

**[C] The Countdown scorer charged every record with a format failure.** SPEC
§3.5's post-stop tail is unscored (`ACC --Σ--> ACC`), so decoded text always
carries a trailing `<turn|>` or `<|channel>`; `_STEP` is anchored `\s*$`, so the
marker landed inside the last step's line. `<|channel>82-80=2\n<channel|>3*4=12<turn|>`
is **accepted by the simulator** and was scored `unparsable step: '3*4=12<turn|>'`
— a format diagnosis on a string the grammar proves is well formed.
`task_countdown_j0map`'s `failure_reasons` was `{'unparsable step': 250}`; it is
actually `{'operand N not available': ~220, 'arithmetic': 7, 'unparsable step': 3}`.
The histogram pointed at the grammar; the model was doing arithmetic on numbers
it was not given. `tasks.answer_region` now strips header, `--think` scratchpad
and tail in one place.

**[D] The Sudoku scorer read the first 16 digits found anywhere.** The earlier
fix stripped the channel header, which covers only the arms that *have* one;
`unconstrained` and `mask` have none and emit prose, and `"Here is the completed
4x4 grid:\n"` shifts the scan by one cell. `sudoku.locate_grid` now finds four
consecutive lines of exactly four digits.

**[E] `run_tasks.py`'s parse column was vacuous** — `n_parsed += int(why !=
"empty")`, while Sudoku returned `"only 0 digits"`, so it was true for every
Sudoku record including the empty ones. Same shape as `Σ_v q_i(v) == 1`. Both
scorers now say `"empty"`, the predicate is named (`tasks.produced_an_answer`),
and `parsed`/`parse_rate` are actually written to the artifact — they were
computed and discarded, which is the only reason no published number depended on
it.

### [F] `[?] RESOLVED` — why repeated groups lifted as their minimum count

The Countdown retraction (`1c0aa77`) blamed `pipeline.compile_regex`. It is not
our code. **`outlines_core` 0.2.14, `src/index.rs`, in the token loop:**

```rust
let is_intermediate_state = !dfa.is_match_state(next_state);
let is_full_match_state = dfa.is_match_state(dfa.next_eoi_state(next_state));
if is_intermediate_state || is_full_match_state { ...record the edge... }
```

`regex-automata` reports matches one byte late, so `is_match_state(s)` is true of
a state entered by reading a byte *after* a match ended. A token that leaves an
accepting state and lands somewhere not itself accepting satisfies neither
disjunct and **the edge is discarded** — while the destination is still pushed
onto the BFS frontier (that push is outside the `if`), so it appears in the table
as a source with no incoming edge and nothing looks broken. `get_next_state` and
`Guide.accepts_tokens` are broken identically, so it is the index, not our
reader. Character repetition (`x*`, `[0-9]{0,3}`, `[^"]*`) is unaffected because
the state after the repeated character is accepting too — which is why this
survived for weeks.

**Fix: `lift.anchor_at_eoi` wraps every regex as `(?:R)\z`.** With an
end-of-haystack anchor no byte-reached state is ever flagged, so
`is_intermediate_state` is universally true and every edge is recorded.
Acceptance is unchanged because the finality `outlines_core` reports is already
`is_match_state(next_eoi_state(s))` — "would match if the input ended here" —
which is exactly what `\z` asks. Eight lines, in `lift.py`; `pipeline.py` needed
no edit.

- `countdown_regex(max_steps=4, max_value=999)`: `|S|` **40 → 124** (bucket 128).
  The anchor **grows** `|S|` — the discarded edges were the only thing making the
  mid-repetition states unreachable, so the minimizer had been collapsing a
  language the grammar was supposed to have. (A first draft of this note claimed
  `|S|` was "equal or smaller"; that was a raw `Index` count for one regex with
  no repeated group, generalised without checking, and it is wrong in direction.
  Corrected in the source comment.)
- Differential-tested regex-vs-DFA on the **real compiled Countdown automaton**:
  4,516 tokenizer-round-tripping strings, 1 disagreement — the SPEC §3.5
  stop-token augmentation behaving as designed. Reviewer's independent harness
  (7 regexes, 2,074 strings) recovered 104 lost strings with **0 false
  positives**, so acceptance is measured unchanged rather than argued.
- **BFCL is unaffected**: 15/15 `live_simple` schemas compile to an identical
  `|S|`. No BFCL result moves because of F.
- Same bug explains the "closing fence lost in the lift" recorded in
  `automaton.wrap_with_fence` and `tests/test_pipeline_flags.py`. The
  regex-level wrap now lifts correctly on its own; the DFA surgery is kept for
  independent reasons (5 fixed states, no second trip through `outlines_core`)
  and both docstrings now say so instead of asserting a cause that is fixed.
- **Countdown is still withdrawn.** Fixing the compiler does not retro-fit the
  emissions: all 250 recorded generations were produced under the broken grammar
  and are single-step by construction. The re-run needs a GPU.

### The Sudoku adjudication, because it moves a published number

Strict structural matching drops `task_sudoku_mask` from **0.204 to 0.012**. The
51 disputed records are strings like `<|channel>4312\n2134\n3241\n1423331` —
three clean rows and a fourth run of junk digits, from which the old rule
recovered a valid grid by taking the first 16 digits. Adjudicated on the
independent oracle this whole audit is built on: **the compiled automaton
rejects 50 of the 51** (`cs_rate = 0.004`), so strict agrees with the grammar and
lenient contradicts it on 45 records. Strict also leaves the clean arms alone
(`unconstrained` 0.864 → 0.864), so it is not merely punishing headerless
output. A prefix-tolerant variant measures 0.192 and was rejected as "a search,
not a rule" — it has no principled stopping point. Consequence, recorded in
`docs/RESULTS.md`: the Sudoku gradient *unconstrained 0.864 → masking 0.204 →
joint 0.000* **loses its middle term**. Masking is 0.012 against `j1`'s 0.012;
nothing in that cut implicates the joint machinery over masking any more.

### The A/B decomposition, and a reporting failure worth recording

Isolating a change to understand it is right; **quoting the isolated number as
the result is not**, and I did it three times in one report — quoting
`unconstrained 0.641 → 0.6739` (A-only) when the code ships **0.6392**, saying
"every delta is positive" of an A-only decomposition without labelling it as one,
and sizing [B2] at "1–5 records across 12 artifacts" from a scan keyed on a
summary field that the two worst artifacts (70 and 55) do not have. Each is
individually defensible; together, on a slice that edits published numbers, they
all lean the same way. Caught in review. **Every number in `docs/RESULTS.md` is
now the shipped column — all fixes applied together — and the isolation figures
appear nowhere in a table.**

Shipped column, 39 BFCL artifacts: `arg_accuracy` moves **up on 16, down on 13,
unchanged on 10**. Largest: `exp_e5_grammar130_j0` 0.696 → **0.282** (70
flagged), `exp_greedy_unconstrained` 0.500 → **0.076**, `exp_e5_grammar130_s1`
0.691 → **0.392** (55 flagged), `mask-sample` 0.508 → **0.216**. Headline row:
unconstrained 0.641 → **0.639**, whitespace-tolerant constrained 0.628 →
**0.650**. The conclusion — the guarantee is free on BFCL — is unchanged.

**It is NOT "slightly stronger", which is what this entry said until review
caught it.** The rescore puts the constrained arm ahead by +0.010 `arg acc` and
+0.023 `exact call`, and I reported that ordering without asking whether it was
earned. It is not. Decomposed: [A] is nearly symmetric across the two arms
(+0.033 vs +0.028), so the whole of the movement is [B], and [B] acts unequally
only because the *exposure* differs — it restores 4 records to the unconstrained
denominator against 1 to the constrained one. Restricted to the 125 records
where both arms produced a call, **the `arg acc` ordering reverses**
(unconstrained 0.6703, constrained 0.6667). The exact-call gap survives the
control (+0.024) but is 7 discordant records against 4, exact McNemar
**p = 0.55**.

Read the five records rather than assuming: unconstrained's four are two empty
emissions, one refusal, and one budget-truncated JSON object — so `extract_json`
is right to return `None` on all four and there is no parser bias — and
constrained's one is `live_simple_122-78-0`, one of the two records compiled
from the defective `any`-type grammar, i.e. our own bug counting against us. So
the fix does not quietly discount the baseline; the lead is a **coverage**
effect, and coverage is not what a column headed `arg acc` claims.

`docs/RESULTS.md` now says so in three places, and gained a
`Paired comparisons — exact call` section: the document ran paired tests for
schema validity and CS and for **neither of the two columns that moved our way**,
which is the asymmetry a reader would rightly have quoted.

**The generalisable lesson, which is not about this slice.** Reporting the
shipped column honestly is necessary and was not sufficient. When a scoring
change *we authored* moves the headline in our favour, the control — restrict to
the records both arms answered — is owed before the table ships, not after
someone asks for it.

[A] in isolation is net *favourable* and the reason is worth knowing: 128
comparisons went wrong→right on pairs like `133` against `133.0`, because the old
rule deleted the decimal point and compared `"1330"` to `"133"` — penalising
BFCL's own documented int→float acceptance. Six went right→wrong on pairs like
`1000` against `100.0`, which the old rule scored **correct** for the same
reason. That second class is exactly the failure the tester predicted, found in
real data.

**Still open [?]**: `metrics.values_match` recurses into nested lists and dicts,
where BFCL's `list_checker` standardises only top-level strings and compares the
rest with `==` (so `True == 1` matches for it and not for us). No record in this
corpus distinguishes the two. Not fixed; noted so it is not rediscovered.

## 2026-08-08 — Countdown re-run on the fixed grammar (in flight)

First GPU run since the four-slice audit and the launch-gate fix (`012660c`).

- **PID 2251286**, log `logs/countdown_refixed.log`, out
  `artifacts/task_countdown_j0map_refixed.json`.
- `--task countdown --variant j0 --emission map --n 250`. Prior run of the same
  shape took 46.2 min.

**Why it is being re-run rather than re-scored.** The Countdown rows were
retracted (`1c0aa77`, and the block in `docs/IMPROVEMENT_PLAN.md`) because
`pipeline.compile_regex` lifted a repeated *group* as its minimum count, so the
compiled automaton admitted only **single-step** solutions while
`countdown_regex` matched multi-step ones. Root-caused in `e2be58c` to
`outlines_core` 0.2.14 `src/index.rs` and fixed by anchoring at `\z`;
`|S|` 40 -> 124, and a differential test over 2,074 strings now shows 0 false
negatives and 0 false positives against `re.fullmatch`.

The emissions on disk were generated under the broken grammar, so re-scoring
them proves nothing — this needs real decodes. Note the *scorer* fixes moved
the unconstrained baseline independently (0.048 -> 0.236, `[C]` in `e2be58c`),
so the old comparison was wrong on both sides.

**What would falsify what.** The retraction says the grammar could not contain
the answer, not that constrained decoding fails at reasoning. If the solve rate
stays at ~0.00 with CS 1.000 *and* the emissions are now multi-step, the
grammar excuse is exhausted and the original finding stands on better evidence.
If it rises toward the rescored unconstrained 0.236, the retraction was right
and the earlier conclusion was an artifact throughout.

Do **not** compare against the withdrawn rows; compare against the rescored
unconstrained arm only.

### Result (PID 2251286): the grammar fix works; the residual is length economics

`CS 1.000, solved 0.036, n=250, 0 zero-partition, 0 OOM, 46.2 min.`
Artifact `artifacts/task_countdown_j0map_refixed.json`.

**The retraction is confirmed on its own terms and then superseded.** Before the
lift fix every one of 250 emissions was single-step, because the automaton could
not express anything else. Now:

| steps emitted | n | solved | rate |
|---|---|---|---|
| 1 | 189 | 2 | 0.011 |
| 2 | 50 | 5 | 0.100 |
| 3 | 10 | 2 | 0.200 |
| 4 | 1 | 0 | — |

So multi-step solutions are expressible and, when emitted, they solve at
10-20%. The grammar excuse is genuinely exhausted: `96-19=77\n77-1=76` is a real
constrained multi-step decode.

**But 0.036 is still far below the rescored unconstrained 0.236**, and the
mechanism is visible rather than inferred. **84 of 250 emissions (33.6%) are
degenerate** — `1 -0=1`, `1 -1=1` — and **none of them solve**. The
`failure_reasons` histogram reads `operand 1 not available: 101`, which looks
like the model using numbers it was not given and is nothing of the kind: it is
the shortest string the grammar admits, chosen by the MAP argmax.

That is **not** a reasoning failure and it is **not** new. It is the same
length-economics mode already measured as 39% of the BFCL gap — grammar-optional
structure collapsing to its shortest member (`{"body": {}}`, `[]`, wildcard
`any` -> literal `1`) — because closing early is admissible and, with the
post-stop tail unscored (SPEC §3.5), free.

**Hypothesis, and the test now running.** If the residual is length economics
rather than capability, `--emission sample` — which draws from the constrained
posterior instead of maximising it — should not collapse to the argmax short
string, and the degenerate fraction should fall. PID 2278246, log
`logs/countdown_sample.log`, out `artifacts/task_countdown_j0sample_refixed.json`.

If the degenerate fraction stays near 34% under sampling, length economics is
the wrong diagnosis and the emission rule is not the lever — in which case the
next candidate is SPEC's tail-weight prior γ (S2), which is declared-exact for
the modified target and is currently unimplemented.

**Do not restate the old Countdown rows.** They are withdrawn, and the scorer
fixes moved the unconstrained baseline independently (0.048 -> 0.236), so the
original comparison was wrong on both sides.

### Result (PID 2278246): hypothesis refuted, and a new defect found

`--emission sample`, same grammar and records:
`CS 0.708, solved 0.032, zero_partition 73/250, 46 min.`

**The length-economics hypothesis is REFUTED.** The prediction was that sampling
would reduce degeneracy and the solve rate would rise with it. Half held:

| | MAP | sample |
|---|---|---|
| degenerate emissions | 84/250 (33.6%) | **29/250 (11.6%)** |
| single-step | 189 | 118 |
| **solved** | **0.036** | **0.032** |

Degeneracy fell by two thirds, exactly as predicted, and the solve rate did not
move — 8/177 = 0.045 scored only over records that emitted, against MAP's 0.036.
So length economics genuinely drives the degenerate emissions and is **not** what
holds the accuracy gap open. The gap survives its removal. The tail-weight prior
γ (S2) targets the same mechanism, so its expected value on this task should be
revised down before anyone builds it.

**`Z == 0` on 73/250 under sample, 0/250 under MAP — a real defect, classified.**
CLAUDE.md gives three causes: (a) empty automaton, (b) no live continuation
within budget, (c) missing scaling. **MAP emitted non-empty output for all 73 of
the affected records**, on the identical grammar, records and budget — a
constructive witness that the language is non-empty and reachable within budget.
That eliminates (a) and (b) and leaves (c): the joint-draw path loses mass the
log-space `joint_map` path does not, in **float64**, at `L=256`, `|S|=124`.

This is not the fp32 case SPEC §6.3 tests for, and it is not a grammar or budget
condition. It is a normalization gap on the sample path only. It also means the
CS column of any sampled arm is measuring two different things at once: the
guarantee held on all 177 records that emitted, and 0.708 is the rate over an `n`
that includes 73 silent non-emissions.

Handing to tester -> coder -> reviewer rather than patching: SPEC §2.4/§2.6
require every tree node's matrix to be normalized with the log-scale sum over all
`2L-1` nodes, and "no try/except around numerical code to make tests pass".

## 2026-08-10 — `Z == 0` root-caused: catastrophic cancellation in `class_weights`

Tester deliverable `tests/test_audit_partition.py` — 33 tests, 18 failing.
Coder in flight.

**Not underflow. Cancellation.** SPEC §2.4 computes a negated class's mass as
`total − Σ_{v∈N_c} p_i(v)`. When the model is confident about a token *inside*
`N_c`, both operands agree to all 53 bits and the difference is `0.0`;
`joint_draw`'s `where(M > 0, log M, NEG)` then erases a structurally present
transition. Measured on the real grammar: true mass `exp(-47.5) = 2.3e-21`,
**661 nats above float64's smallest normal**, computed as zero. A wider float
recovers nothing a subtraction already discarded; `logsumexp` over the members
returns `-47.5` exactly. `joint_map` is immune — §2.7's `class_max` is a `max`,
never a difference.

Countdown's `|S|=124` grammar has two negated classes; one stores the reserved
ids `{0,1,50,100,101,106,107}` — the §3.6 channel-header class, on **8 edges
every accepted string must traverse**. That is the 73.

**The minimal reproducer is `L=8`, `|S|=4`, `V=64`, `N={0}`**, and it behaves
identically at `L=256`. **This inverts the project's own standing lesson.**
CLAUDE.md says toy shapes validate nothing, earned from float32 looking fine at
`L=4`. Here a toy fixture *would* have caught it — the suite's only negated
class is "the odd tokens", mass ≈ 0.5, where the subtraction is exact to 1e-16
relative. The rule should be "fixtures must span the *regime*", of which scale
is one axis and `|N_c|` is another.

Controlling variable is class mass, not `L`, `|S|` or grammar size: the kernel
resolves ~16 orders of magnitude (boundary bisected at 1.09e-16 = float64 eps)
against ~66 the model produces. Survivor error arrives first — at `ε=1e-14` the
weight is positive and **17% wrong**, which trips no detector.

**Two further defects.**

- **`joint_map`'s `1e-30` floor is above the production range** (smallest
  marginal `3.7e-58`). 1162 of 6144 (class, position) pairs have their whole
  class below it, so `class_max` is the clamp for every member and `argmax`
  ties to the **lowest token id**. This is visible in the shipped artifacts and
  **corrects yesterday's diagnosis**: the `1 -1=1` emissions I attributed to
  length economics are substantially this clamp — MAP 69/250 against sample
  17/177, digit "1" 475× against 253×. The refutation of length economics as
  the *gap-closer* stands (the solve rate did not move); my account of what the
  degeneracy *was* does not. SPEC §2.7's "MAP == exhaustive argmax, error
  0.000e+00" is currently false. §3.1b's guarantee is unaffected.

- **The `mask` baseline is destroyed by defect 1.** Same `p` and grammar: 255 of
  256 positions lose their entire support; the mask branch is deliberately
  unflagged, so it goes all-sentinel and `categorical`, being shift-invariant,
  draws **uniformly over 262k tokens** and reports nothing wrong. `mask` is the
  published comparison baseline for SPEC §2.8 / §7.2. **Treat its column in
  `docs/RESULTS.md` as unsafe until this is fixed and the arm re-run.**

**Why 1051 green tests missed it:** no fixture anywhere has a negated class whose
true mass can fall below 1e-16. Also, SPEC §2.4's own verification ("1.1e-16 with
mixed polarity") is *absolute*, which says nothing about a class whose true mass
is 1e-20 — the same absolute error is then 1e4 relative. And nothing compared the
two emissions' feasibility on the same input: **MAP-feasible ⟹ draw-feasible** is
a theorem, needs no ground truth, and would have caught this on day one.

The tree is exonerated: `up_sweep` recovers the unnormalised product exactly from
`root + root_log_scale` over all `2L−1` nodes, and `up_sweep_log` matches a
sequential float64 fold on the real leaves. The gap is entirely in **leaf
construction**, which is linear-space and outside the tree's normalization.

## 2026-08-10 — `Z == 0` fixed; a41bf74's float32 verdict survives; a second site

Coder deliverable against `tests/test_audit_partition.py` (33 tests / 18 failing
→ 43 / 43 green after the tester extended it). Two files: `infer/marginals.py`,
`model/constrained.py`, plus a prose fix in `compile/classes.py`.

**The fix is algebraic, not numerical, and stays in linear space.** A negated
class's mass is now a sum over its *real members*, split at the class tables'
own stored support `U`:

    Σ_{v ∉ N_c} p_i(v)  =  Σ_{v ∉ U} p_i(v)  +  Σ_{v ∈ U \ N_c} p_i(v)

Both terms are sums of non-negatives, so nothing cancels. `logsumexp` would also
have worked, but the log is not what buys the accuracy — *not subtracting* is;
log space still needs the same `U`-split for the complement, costs extra kernels
on the launch-bound side of §5.6's crossover, and would ripple into
`transition_matrices`, `up_sweep` and the `mask` baseline. Linear float64 covers
the required range with ~660 nats to spare (the softcap bounds a class mass at
`~e^-150`; float64 normals reach `e^-708`). The naive `O(C·V·L)` dense repair the
tester used as a diagnostic is **3.6e10 MACs** at BFCL's worst shape; the
`U`-split replaces `V = 262,144` with `nnz` (110 Countdown, median 9,125, max
51,200).

Verified by the reviewer against `math.fsum` over each class's true members —
sharing no code with the implementation or the tester's `logsumexp` oracle — on
real Countdown and three real BFCL grammars across 7–8 `p` regimes: **worst
relative error 1.75e-12, 0 entries lost**, against the old kernel's **relative
error 1.0 with entries lost**. On an idle H100 the worst real shape costs
**+0.679 ms = +0.323% of a 210 ms model step**. No `scan`, no `while` introduced.
Peak temp is *not* unchanged and is budgeted in the docstring (0.21 → 3.47 MB
Countdown, 100 → 205 MB at BFCL's worst `nnz`); §5.6 sets the `|S|` dispatch
threshold from that headroom.

**The `mask` baseline is repaired.** Same grammar and `p`: empty-support
positions **255/256 → 0/256**, dead `(position, edge)` leaf slots 8/68,608 → 0,
`joint_draw` feasible False → True. `docs/RESULTS.md`'s `mask` column still needs
its arm re-run; the kernel that produced it is fixed, the numbers in the file are
not.

### a41bf74 survives — do not re-open it

`test_audit_numerics.py::test_float32_at_production_scale_is_either_shown_safe_or_refused`
went red on the fix, and the honest reading is **the fixture broke, not the
guard**. Measured on its `_cycle_automaton(64, 64, 4096)` fixture: lost 32 → **0**
live root entries, survivor error 7.58e-3 → **5.5e-5** nats. The float32 loss
there was *entirely* the `total − partial` cancellation (float32 cancels at
~1e-7), never underflow.

The reviewer then measured the real thing on three real BFCL grammars at
`softcap 30·tanh, T = 0.4`: all four cells (f32-arith old/new × f32-`p` upcast
old/new) are **identical at every regime**, and `W_lost = 1315` on
`get_user_info` reproduces `require_x64`'s docstring row exactly. **a41bf74's
explanation is intact**: the production loss is the float32 `softmax` flushing
7.6M of 67.1M `p` entries to zero *upstream* of `class_weights`, which this
change does not touch. `require_x64` stays. The test needs re-pointing at a real
grammar; its guard half (that `require_x64()` refuses with x64 off) is currently
unreached and that is the only thing wrong with it.

Why the fixture stopped working is itself a41bf74's own pathology **running in
reverse**: its two classes are 2,048 tokens wide, so no `p` flush can kill a
class and cancellation was the only float32 loss it ever had. a41bf74 wrote "the
toy is where the patch looks like a fix"; here the toy is where the *fix* looks
like a regression.

### Second site: `scatter_edge_mass_to_tokens`, same subtraction one layer down

`r_i(v) = neg_total − Σ_{c ∈ Neg, v ∈ N_c} U_i(c)` is the identical shape, on a
path feeding `q_i` (eq 6), `--confidence=mar`, the §2.8 `mask` support and
`tree.sample_tokens`' eq (8) multiplicity — where a wrong `r` is a silently wrong
*distribution*, with no `Z == 0` to announce it. Constructed and measured (two
negated classes, token stored by the large-`U` one):

    U ratio 1e-8    old 1.000000e-08  rel err 6.1e-09   new exact
    U ratio 1e-14   old 9.992007e-15  rel err 8.0e-04   new exact
    U ratio 1e-16   old 0.0           rel err 1.0       new exact

Same `U`-split applied. A mixed-polarity `math.fsum` cross-check goes 2.64e-12 →
2.22e-16.

**Countdown's immunity is positional, and the first two explanations of it were
wrong.** Not "the scatter is off the emission path" (it is on it, via
`sample_tokens`), and not "one negated class has `U` exactly 0" (both are live).
Measured: class 1 — the §3.6 channel header — is live at positions **1..8**;
class 23 — the unscored `ACC --Σ--> ACC` tail (§3.5) — at **10..255**;
**0 of 256 positions carry both**, so `neg_total` never has two terms to cancel.
Front-pinned and back-pinned in *this* grammar only. BFCL carries two non-empty
negated classes (stored sizes 7 and 1,585) with no such separation. Record the
reason, not just the conclusion: a wrong reason is what makes the next reader
assume a new grammar inherits the immunity.

### `joint_map`'s `1e-30` floor — corrected, and the attribution retracted

`map_log_floor(dtype) = finfo(dtype).tiny` replaces the hardcoded `1e-30`, which
sat *above* the production range (smallest real marginal `3.7e-58`). SPEC §2.4's
`1e-30` is a rule about `q`, which has exact zeros where the clamp only ever
multiplies a zero; `p` is a softmax and the clamp was a floor on **live** values,
flattening every token under it so `argmax` tied to the lowest id.

**Retracting yesterday's entry.** It attributed the MAP Countdown arm's repeated
`1 -1=1` bodies (69/250, digit "1" 475× vs 253×) to this clamp. That does not
measure: across 6 seeds on real Countdown, with 1,162–1,213 fully-clamped
`(class, position)` pairs each time, swapping the floor changed **0 of 256 tokens
and 0.00 nats**, every time — the clamped classes do not lie on the winning
max-plus path. The fix is still correct and necessary (a mutant kills a test on
the minimal fixture, where the clamped class *is* on the path). **The cause of
the `1 -1=1` repetition is open**, and it is now open for the second time after
being closed twice on unmeasured reasoning.

### Prose fix

`compile/classes.py`'s module docstring stated `W_c = total − Σ_{v∈N_c} p` as the
**requirement**. That is why the defect survived review as documentation: an
implementation matching the spec looks correct by inspection, so the identity was
never the thing under test. It now states the quantity (`W_c = Σ_{v ∈ S_c} p_i(v)`)
and leaves the recovery to `marginals.py`. `tests/test_classes.py` had the same
formula and never called `class_weights` — fixed by the tester.

### Suite

`tests/test_audit_partition.py` 43/43 (7.6 GB peak RSS, 67 s with `-x` — run it
with `-x`, 22 retained tracebacks cost 136 GB and OOM-killed a process).
`test_classes.py` 43/43. Everything else on CPU: 1,062 passed, 1 failed — the
a41bf74 fixture above. 10/10 mutants still kill their named victim.

## 2026-08-10 — Re-running the arms invalidated by `da1294a` (in flight)

**PID 2513436**, log `logs/rerun_affected.log`, launched from an immutable copy
(`scratchpad/rerun_LOCKED.sh`, mode 444) because editing a live bash script
corrupts the running instance — a mistake already paid for once.

Seven arms, sequential, ~80-95 min each: `mask` first (its column is meaningless,
not noisy), then the three sampled arms, then the three `marmap` bounds.

**Not re-run, on measured evidence rather than caution:** `--emission=map` with
`--confidence=mf`. `joint_map` calls neither `class_weights` nor the scatter — it
builds its own dense `member` and takes a max — and J0's trajectory is stock
uniform renoising. The `map_log_floor` change is a measured null there (0/256
tokens, 0.00 nats across 6 seeds). That exemption covers
`eval_bfcl_live_simple_j0_map`, the four `exp_abl_*`, `exp_e5_grammar130_map`,
`exp_h2_mfmap`, `exp_seed{1,2,3}_map`, the Sudoku and Countdown `j0map` arms,
and every `unconstrained` arm.

**The `mar` arms ARE re-run despite being `--emission=map`.** The reviewer
withdrew its own blanket MAP exemption on tracing the code: `mar`'s acceptance
and stopping route through `class_weights` **and** `constrained_entropy_streamed`
→ the scatter, even though the emission does not. On their grammars the scatter
was losing thousands of entries.

**Expect `n = 128`, not 130.** The build gate refuses `live_simple_117-73-0` and
`live_simple_122-78-0`; `012660c` now names them in the artifact, the console log
and the generated table. Any comparison against the published `n = 130` rows must
account for the two-record difference — which is exactly why the launch gate was
closed before this ran.

### Correction: the v1 queue (PID 2513436) was stopped, unreviewed, and never ran

The entry above is superseded. That queue was launched **without a launch
review**, stopped 4 minutes in with **0 artifacts written** and the GPU
released, and its script must not be used. `CLAUDE.md` now requires a reviewing
agent to read every GPU- or artifact-writing command **before** it runs
(`3eb5b20`); this is the first application of that rule, and it paid for itself
immediately.

**Two independent launch reviews found six defects, none of which any test
would have caught:**

| defect | what it would have cost |
|---|---|
| v1 passed no grammar flags, inheriting `--whitespace json` | **No artifact in this repo has ever been measured at `json`.** The four table arms ran at `stock`, the `marmap` arms at `pretty --fence --ci-enums`. Whitespace alone is worth 0.327 -> 0.643 arg acc — ~20x the effect being measured. Seven numbers differing from the ones they replace for two reasons at once, and the `mar`-vs-`mf` contrast with **no control anywhere in `artifacts/`**. |
| v1 wrote `artifacts/rerun_*.json` | `phase5_report.py` globs `eval_<task>_<variant>_<emission>.json`, so regenerating `RESULTS.md` would have silently rebuilt it from the **old defective** artifacts. |
| `exp_h2_marmap` and the Countdown sample arm missing | 3 of 4 affected `mar` arms; and the arm that *found* the defect, whose 0/250 prediction converts 12 h of hope into a 46-minute proof. |
| `run()` opens `logs/$tag.log` with `>` | Eight of the nine canonical logs already existed from the published runs. `logs/` is gitignored: the queue archived each artifact and then **destroyed the console log of the arm it replaced**, unrecoverably. |
| the Countdown stop condition was **prose** | Under `nohup`, a falsified prediction would have burned the remaining **~12.5 GPU-hours** anyway. Now an enforced `exit 1`, tested against all five artifact states including the real pre-fix artifact (zp=73 -> STOP). |
| the smoke missed `constrained_entropy_streamed` | A different scatter call site — the acceptance/stopping path of the four `mar` arms, ~5 h and running **last**. First failure would have surfaced ~8.5 h in. |

**Two measurements that corrected claims of mine.** The two gate-refused records
are *not* whitespace-dependent — the gate refuses `live_simple_117-73-0` and
`live_simple_122-78-0` under `stock`, `pretty` **and** `json`, because their
defect is an unparenthesised top-level alternation. And `n = 128` means **two
different things**: "130 minus 2" for the four table arms and the three offset-0
`marmap` arms, but "all 128 ran" for `exp_h2_marmap` (records 130..257 refuse 0),
whose `mf` control `exp_h2_mfmap` is also 128 — that pair is denominator-clean
and needs no rescore. Flattening those was called "the same species of error v1
was condemned for".

A third axis neither review was asked about: `--dtype` did not exist until
`19a3eca` (Aug 1), so all four table arms predate it and ran float64 —
`--dtype float64` is a faithful reproduction, not a second variable. The float32
window contains exactly `exp_e5_grammar130_{j0,s1}`, which are **not
reproducible** (`--dtype float32 --emission sample` now raises) and should be
withdrawn rather than re-run.

## The approved queue (v3) — PID 2528151

`scratchpad/rerun_v3_LOCKED.sh`, mode 444, md5 `e28cdb66edc2fd800676cdf9c6359c47`.
Log `logs/rerun_v3.log`. Two smokes, then Countdown behind a hard stop, then the
four table arms at `stock`, then the four `mar` arms at `pretty --fence
--ci-enums`. ~13.5 h. Every measurement-affecting flag passed **explicitly**,
including those equal to today's default — `eval/run.py`'s defaults (`j1`,
`sample`) already contradict SPEC §3.9 and `eval/run_tasks.py`, and nothing
asserts either. Each arm prints its config **read back out of the artifact**,
which is the only record of what actually ran.

## 2026-08-12 — `prefix_suffix` was not implementing SPEC §2.4's scaling rule

`infer/scans.prefix_suffix` returned linear `a`, `b` normalised **per vector**,
and SPEC §2.4 eq (5) makes `u_i(e) = a_{i−1}(src e)·b_i(dst e)` the object of the
theory. Per-vector max normalisation bounds neither factor's within-vector range
nor their product, so the quantity the `mask` baseline and `--confidence=mar`
actually read was losing structurally live edges. Measured on the compiled
grammars at the sharpness Phase 0 saw on the checkpoint (mean `H = 3.0e-4` nats,
`T = 0.4`):

| grammar | live edges lost | legal (position, token) pairs lost | positions with EMPTY support | `log Z_i` spread |
|---|---|---|---|---|
| Sudoku (`\|S\|` = 64) | 108 | 2,097,192 | **16 of 256 — positions 0…15** | ∞ (some `Z_i = 0`) |
| Countdown (`\|S\|` = 128) | 5,432 | 0 | 0 | **480 nats** |
| BFCL `get_user_info` | 16,089 | 5 | 0 | 654 nats |
| BFCL `github_star` | 7,134 | 0 | 0 | 419 nats |
| BFCL `uber.ride` | 10,208 | **2,360,164** | **7** | ∞ |

At a flat `p` every one of these is clean: the defect is switched on by
sharpness, which is why no smoke test at `T = 1` ever saw it. `docs/RESULTS.md`'s
Sudoku `mask` row (`CS 0.004 / parsed 0.040 / solved 0.012`) was measured through
it, and the dead positions are exactly the front of the canvas — the §3.6 channel
header and the first grid row — which is where its emissions are malformed
(`'<|channel>124\n4321\n…'`, first row a digit short). **That row must be re-run
before it is quoted again.** The `mar` sweep must also be re-run: an arm was
launched from a dirty tree mid-edit on 2026-08-12 and its three artifacts are
quarantined in `artifacts/unattributable_dirty_tree/`.

**The fix** (`infer/scans.py`): the forward–backward runs in log space with
`log_matmul`'s pairwise-max anchor — the linear recursion was itself lossy, the
viable within-vector range of `a` being 3,420 nats on Countdown — and the two
returned linear factors are anchored on a **pair of per-boundary scales whose sum
is `log Z` exactly**, so `u` is the edge posterior and `Σ_v p_i(v) r_i(v) = 1` at
every position. Anchoring on the unrestricted vector maxima does not work: SPEC
§3.5's unscored `ACC --Σ--> ACC` tail pins `max_s b_i(s)` at 1.0 while the grammar
sits at `Z`, so the misalignment is exactly `−log Z` = 1,316 nats on Sudoku;
restricted to **viable** states (`a > 0 ∧ b > 0`) it is ≤ 474 nats across the 8
grammars measured.

**`[?]` resolved by measurement: how far a two-factor `u` can be pushed.** With
`γ_i = α'_i + β'_{i+1} − log Z`, any pair of scales with a fixed sum has
`max(log a) + max(log b) − (scale_a + scale_b) = γ_i` — invariant under the
split — and a floored factor multiplies a partner of up to `exp(max log)`. So
`Σ_e u_i` is undisturbed only while `γ_i ≤ |F_a| + |F_b| ≤ |log tiny| ≈ 708`
nats, and **no anchor, split or floor placement evades that** while `u` must
factor as `a(src)·b(dst)`. Sudoku measures `γ = 163 / 404 / 808 / 1,616` nats at
`T = 1.0 / 0.4 / 0.2 / 0.1`: the shipped `T = 0.4` is 304 nats inside the bound,
one halving of `T` away, and `_MIN_TEMP = 1e-12` plus the shipped `--temp greedy`
make that reachable by configuration alone. Above the bound the floor now widens
instead of clipping, so mass stays exact and **support** degrades — 16 live edges
at `T = 0.1`, no position emptied — and `prefix_suffix(..., feasible_out=True)`
reports every affected position out through `ConstrainedSamplingState.feasible`.
With a fixed floor the same sweep gave `max|Z_i − 1|` = 1.63 at `T = 0.2` and
1.9e+109 at `T = 0.1` **with nothing raising**, because `H(q_i)` stays inside
`[0, log V]` while being silently reweighted.

Two options were measured and rejected. Anchoring per position on
`c_i = max_e (log a(src) + log b(dst))` is bounded by construction but is not
available at this signature: `c_i − log Z` varies by 94.7 nats across positions
on Sudoku at `T = 0.4`, and production reads `u` through `a` and `b` alone, with
no channel to add a per-position scale back. Reassociating
`constrained_entropy_streamed` to `−(1/Z)Σ w(log w − log Z)` is better
conditioned and makes `H ≥ 0` structural, but it **deletes the suite's only
detector for this defect** — fed the pre-fix `u` the shipped form returns
`H = −708.4` at 17 Sudoku and 10 BFCL positions, which is exactly what
`test_mar_confidence_is_a_real_entropy` catches. Reverted, with the reason
recorded in its docstring.

**Method note, for the next numerical audit: use `decimal`.** The review that
cleared this change verified it against a 45-digit unbounded-exponent
`decimal`-module edge-list forward–backward sharing no code with the
implementation, over 8 real grammars × 4 `p` regimes and 38 randomised NFAs —
`scale_a[i] + scale_b[i+1] == log Z` to 0 or 1 ulp, `q` to 1.4e-15, floor
inflation exactly 0 by independent `fsum`. Unbounded exponent is what makes the
oracle able to answer questions float64 cannot represent, which is the whole
subject matter here; it is cheap at these shapes and it is now the recommended
reference for anything in §2.4.

**Also changed.** `_accept_from_entropy` snaps `|H| < 1e-12` to zero: the rule is
an exact comparison on a *running* total, so a single positive ulp at the
sharpest position gated every position behind it once `Z_i` became 1 (measured
`[2.5e-32, 2.5e-32, 0.0, 2.5e-32]` → 2 of 4 positions accepted at
`entropy_bound = 0`, against the 4 SPEC §3.4 requires). Large negatives are
deliberately left alone. And `float32` is now refused for `--variant mask` and
`--confidence=mar` regardless of `--emission`: the existing refusal is gated on
`emission == "sample"`, but both of those read the linear forward–backward on
every emission, and its bound is ~87 nats in float32 against the 404–474
measured.

## 2026-08-12 — Re-running the arms invalidated by `2c9f9d9` — PID 3267670

`scratchpad/rerun_2c9f9d9_LOCKED.sh`, mode 444, md5
`de27ca6b85bad51b27ee8178edecd07c`. Log `logs/rerun_2c9f9d9.log`. **16 arms,
~19.3 h** from the arms' own `elapsed_seconds`. Launched from a clean tree at
`2c9f9d9`, fingerprint `228fc705dc0f9c695cb03111cfe42087`, re-checked **before
every arm** — the guard that the previous queue lacked, which is how three
artifacts ended up unattributable.

**The exemption was re-derived from source, not inherited from the commit
message.** `model/sampler.py` has exactly two `prefix_suffix` call sites: the
`mar` branch (→ `constrained_entropy_streamed`) and the `mask` branch (→
`scatter_edge_mass_to_tokens`). `_accept_mask` takes `softmax(out.logits)` —
unconstrained shaped logits — so every `mf` arm is clean, and `joint_map` /
`joint_draw` never reach it. The one `2c9f9d9` change on the `mf` path, the
`|H| < 1e-12` snap, is a measured null at every bound ≥ 0.003, and every arm
here is at 0.003 or above.

**Configuration reproduced field-by-field.** All 14 existing arms match their
artifact exactly across `variant, emission, confidence, entropy_bound,
whitespace, fence, ci_enums, dtype, seed, offset, prompt_style, temp,
nonempty_strings, think`. Where an artifact stores no value, the review
established from git that the flag **did not exist yet** — `--confidence`
landed `66042d9`, `--temp` `e4e7144`, `--offset` `8c3be4c`, all 2026-08-02, all
after the four artifacts' mtimes — with the control that artifacts written
after each landing do carry the key.

**Two confounds that cannot be flagged away**, recorded rather than hidden: the
Aug-8 build gate moves `exp_f64_ws_only_j1` / `exp_oomfix_j0` from n=130 to
n=128, and the Aug-8 scorer (`e2be58c`) postdates every Sudoku/Countdown
artifact — `task_sudoku_mask` *stores* `solved = 0.204` while `RESULTS.md`
carries the rescored 0.012. **Compare new task arms against `RESULTS.md`, never
against the artifact they replace.** Every gate is therefore written on `cs`,
which is Simulator-based and scorer-independent.

**The falsifiable prediction, enforced as code (gate C):** `task_sudoku_mask`
must report `cs ≥ 2` against a published **1 of 250**. 249 of 250 pre-fix
emissions have the entire header-name region missing
(`'<|channel>124\n4321\n…'`); the single accepted one does not. One mechanical
cause — positions 0–15 empty ⇒ token 0 — which the fix removes. A false stop
would require the fix to have changed nothing.

Gate B is deliberately **graded** rather than strict on the long arms: `Z==0`
of 0 is `[ok]`, under 10% prints `[!!]` and continues, ≥10% stops. The commit's
γ ≤ 474 nats was measured on 8 grammars, not 130 records, and one tail record is
a finding in its own column — not a reason to discard 127 records and 14 hours.

The `mar` sweep is ranked high on a specific argument: pre-fix, a dead position
returned `H = −708`, sorted first and always accepted, making acceptance nearly
**bound-independent**. The published flatness (0.6655 / 0.6899 / 0.6725 across
1.7 decades) is exactly what a corrupted rule produces, so it is the most likely
artifact left in `RESULTS.md`, and 0.003 / 1.0 are where signal would appear.

`exp_f64_ws_only_j1` is judged **not worth its 105 min** — it published at n=130
pre-gate, returns n=128 on a scorer that also changed, and its only citation is
one s/record latency figure. It is placed **last**, so killing it costs nothing.

Still missing from SPEC §3.4 after this: bounds 0.03 and 0.3, and the whole
`entropy_threshold` axis, which is structurally unswept because `early_stop_fn`
is passed at zero production call sites.

---

## 2026-08-12 — The `any` wildcard compiled to an unparenthesised alternation (SPEC §4.2)

**Tester → coder → reviewer.** Tests first (`tests/test_audit_wildcard.py`, 105
tests, **63 red on the unfixed tree**), then this implementation, then review.
The tests were not editable by the implementer and are not edited here.

### The defect

`compile/schema.py` translated BFCL's `"type": "any"` by **omitting** `type`.
outlines-core 0.2.14 expands an omitted type into a seven-way alternation and
**does not group it**, so as a property value it composes to

```
\{ws"input_value"ws:ws((true|false))|(null)|(number)|(string)|(array)|(object)ws\}
```

`|` binds loosest: only the *first* branch carries the opening `\{ws"key"ws:ws`,
only the *last* carries the closing `ws\}`. Measured (`JSON_WS`,
`allow_wildcard=True`, `live_simple_117-73-0`): the grammar **rejects**
`{"input_value": "say hi"}` — the ground-truth answer — and **accepts** `1`,
`null`, `"say hi"`, `1.5`, `[1]`, `{"input_value": true` (unterminated) and
`{"input_value": "say hi"}}` (stray brace).

Both leaks are in published artifacts. `exp_e5_grammar130_map.json` /
`_s1.json`: `live_simple_122-78-0` emitted the single character `1` at
`accepted=True`, CS 1.000, `schema_ok=False`, while the *unconstrained* arm on
the same record emitted a correct complete object; `live_simple_117-73-0`
emitted `{"input_value": "hi say reverse"}}`. **The automaton was exact and the
language was wrong** — the same shape of failure as the whitespace bug, and the
reason CLAUDE.md's "a number that contradicts a proof means the measurement is
wrong" now has a companion: a *guarantee* that holds against the wrong language
guarantees nothing.

### The fix

`schema.WILDCARD_ANYOF_TYPES` (exported) + `_expand_wildcards`: every **typeless**
sub-schema is rewritten as `{"anyOf": [{"type": t} for t in
WILDCARD_ANYOF_TYPES]}`, which outlines *does* group. Typeless is defined as
"carries none of outlines' nine precedence keywords" — the exact condition for
the defect, not a proxy — so it covers all three routes: BFCL `"type": "any"`, a
literal `{}` sub-schema, and a node whose only key was an annotation (which
`normalize_bfcl_schema` strips to `{}`, and on which outlines *raises*). The
`__wildcard__` marker is kept so `check_supported(allow_wildcard=False)` still
refuses the shape; `_strip_markers` removes it before outlines sees it.
Recursion is schema-position aware — a JSON object literal inside an `enum` is
"typeless" by this predicate and must not be rewritten.

`expand_wildcard: bool = True` on `normalize_bfcl_schema`, `build_regex`,
`synthesize_instance`, `pipeline.compile_json_schema`, and
`eval/run.py --expand-wildcard` / `--no-expand-wildcard`, recorded in the run
artifact and **in `schema_fingerprint`** (it changes the language outright).

**The opt-out is loud, and deliberately not covered by the whitespace hatch.**
`verify_strict=False` re-checks under `JSON_WS` while keeping the caller's
`expand_wildcard`, so an unexpanded wildcard still raises. Precedent: the same
hatch was found in review to swallow `live_simple_117-73-0` — this very defect —
when it was scoped to "any failure" rather than "the narrowness you declared".

Also fixed, a live crash the tester found: `check_supported` did `t in
WILDCARD_TYPES` with `t` possibly a **list** (`{"type": ["string","null"]}` is
legal, and `normalize_bfcl_schema` maps type unions), raising `TypeError:
unhashable type: 'list'` in a module whose contract is to raise
`UnsupportedSchemaError` or pass. Reachable only on the `allow_wildcard=False`
path, which is why no production call site hit it.

### Measured (CPU only; `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=""`)

`expand_wildcard=False` reproduces the pre-fix language exactly, so both arms
come from one working tree and one process.

**Gate survey, all 4,549 BFCL-Live schemas** (`JSON_WS`, `allow_wildcard=True`,
`allow=tasks.bfcl.ALLOW`, instance from `synthesize_instance`, key order ignored
— `tasks/bfcl.py`'s own gate):

| | pass | refused |
|---|---|---|
| pre-fix | 4,538 / 4,549 | **11**, each failing all five renderings |
| fixed | **4,549 / 4,549** | 0 |

The 11 are exactly the `any` schemas. Nothing else changes state in either
direction. **SPEC §4.7's "4,549 / 4,549, zero failures" was stale** — it predates
the build gate (`b58fd84`) — and is corrected in place.

**Full pipeline** (`compile_json_schema`, `from_bfcl=True`, `allow=bfcl.ALLOW`,
`allow_wildcard=True`, `whitespace_pattern=JSON_WS`, `channel_header=True`,
`fence=False`, `nonempty_required_strings=False`, 262,144-token vocabulary,
gate not wired because the pre-fix arm is the grammar the gate refuses):

| schema | expand | regex | `\|S\|` | bucket | edges | classes | tree GB | wall s |
|---|---|---|---|---|---|---|---|---|
| `reverse_input` | False | 15,323 | 568 | 1024 | 2,810 | 208 | 2.1433 | 331.5 |
| `reverse_input` | **True** | 15,311 | 568 | 1024 | **2,594** | **182** | 2.1433 | **248.8** |
| `process_data` | False | 15,509 | 610 | 1024 | 3,274 | 268 | 2.1433 | 370.1 |
| `process_data` | **True** | 15,497 | 610 | 1024 | **2,728** | **223** | 2.1433 | **280.5** |

Every structural figure reproduces the tester's independent measurement exactly;
only wall-clock differs (this box, two single-threaded compiles overlapping for
the first two rows — `logs/wildcard_pipeline_measure.log`). Same `|S|`, same
bucket, same tree, fewer edges and classes, faster to compile.
`is_dfa` stays True and `needs_chain_path` stays False. **There is no cost
tradeoff**: the fix is a parenthesisation, 12 regex characters *shorter*, and
strictly cheaper downstream. (The `{items:{}}` / `{additionalProperties:true}`
spelling of the same seven-way is 38 KB and was killed at 54 GB RSS — not used.)

Width is **seven, declared, not five**. The five-scalar narrowing is genuinely
cheaper (`|S|` 57, bucket 64, ~60x faster) and would cost nothing on BFCL today
— every ground-truth value at a wildcard position is a string — but "the
benchmark does not exercise it" is the argument that retired the Countdown
grammar. It stays reachable as a one-line *declared* narrowing of
`WILDCARD_ANYOF_TYPES`, which `test_the_admitted_type_set_equals_the_declared_one`
holds the grammar to in both directions.

### THE DENOMINATOR MOVES: n = 128 → 130, for some arms only

Two of the 11 (`live_simple_117-73-0`, `live_simple_122-78-0`) are inside the
`live_simple` cut. An arm that reported `n = 128` **with
`records_available = 130` and `skipped_by_reason = {"compile:ValueError": 2}`**
now reports **130**, and 130 is not comparable to those 128 rows without
accounting for the two records. **`exp_h2_*` also reports `n = 128` but has
`records_available = 128` and no skips — a different cut, which does NOT become
130.** Check `records_available`, not `n`.

### Two things left for someone else

1. **`tests/test_audit_measurement.py::test_the_next_arm_really_does_skip_the_two_records_the_audit_named` is now RED, by construction.** It asserts that the shipped compile call *raises* on those two records — the pre-fix state — and `tests/test_audit_wildcard.py::test_the_two_gate_refused_records_pass_the_gate_after_the_fix` asserts the opposite. Only one can be true, and the fix makes it the latter. The coder may not edit `tests/`; the reviewer rules on which test retires. Nothing else in that module depends on it (its `_FAKE_SKIPS` ids are synthetic).
2. **`compile/tasks/bfcl_python.py` is untouched and unaffected**: `value_regex` dispatches on `type`, so a wildcard property still falls to its own `_scalar_union()` — a 4-way narrowing it declares in a comment. It does not read `WILDCARD_ANYOF_TYPES`, so the JSON and Python grammars now disagree about what `any` means. Pre-existing, harmless today (no Python arm has been run), worth closing when it is.

### Unrelated, found while working: THE RE-RUN QUEUE IS DEAD

PID 3267670 (the 16-arm, ~19.3 h queue logged above) **stopped at 04:19 UTC on
2026-08-12**, ~4 h before this session started and before any edit here. It ran
one smoke arm and then aborted with

```
!! THE TREE MOVED UNDER THE QUEUE (fingerprint 228fc705dc0f9c695cb03111cfe42087 -> 228fc705dc0f9c695cb03111cfe42087).
```

**The two fingerprints in its own message are identical**, so the guard is
comparing something other than what it prints — a false positive that cost the
whole queue. `logs/rerun_2c9f9d9.log` is 1,054 bytes and ends there; the GPU has
been idle at 0 MiB since; `scratchpad/rerun_2c9f9d9_LOCKED.sh` no longer exists,
so the queue cannot be resumed as launched. **Fifteen arms did not run.**

### Review round 1 — one blocking correction to the fix above

**`_is_typeless` over-fired, and its stated justification was false.** It
expanded *every* node carrying none of outlines' nine precedence keywords, on
the written grounds that outlines "either honours or ignores" everything else.
Measured against installed outlines-core 0.2.14, **it ignores nothing — it
raises**:

| typeless sub-schema | raw outlines | `expand=False` | `expand=True`, as first written |
|---|---|---|---|
| `{}` | fall-through | ok | ok |
| `{"pattern": "^a$"}` | **raises** | ValueError | **compiled as any-JSON** |
| `{"minLength": 3}`, `{"format":…}`, `{"required":[…]}`, `{"items":{…}}`, `{"additionalProperties":{…}}` | **raises** | ValueError | **compiled as any-JSON** |

So the expansion converted a loud `ValueError` into a **silent, maximally
permissive grammar with the constraint dropped** — and `check_supported` does
not catch these (`pattern`, `format`, `required`, `items` are not in
`_SILENTLY_DROPPED`; the length/item bounds sit on `tasks.bfcl.ALLOW`). That is
precisely the desensitisation SPEC §4.2's fail-loud pre-pass exists to prevent,
written into the same diff that corrects SPEC for the same mistake. Reach was
**0 of 4,549** — latent, not live — but a wrong keyword list is how the next one
gets through.

**Fixed (reviewer's option (a)):** `_typeless_kind` expands only a node that is
empty **modulo pure annotations** (`description`, `title`, `$comment`,
`examples`, `default`, `deprecated`, `readOnly`, `__wildcard__`) — which is the
real fall-through condition and the semantically correct reading of
`{"description": …}` — and **raises `UnsupportedSchemaError`, naming the
keywords**, when a typeless node carries a real constraint. Re-measured after
the change:

```
{}                                           fall-through   ok             ok
{"description": ...}                         ValueError     ValueError     ok
{"pattern": "^a$"}                           ValueError     ValueError     UnsupportedSchemaError
{"minLength": 3} / {"format"} / {"required"} / {"items"} / {"additionalProperties"}
                                             ValueError     ValueError     UnsupportedSchemaError
```

**Nothing measured moves.** The new raise fires on **0 of 4,549** BFCL-Live
schemas; the gate survey is unchanged (4,538 → **4,549**); and the compiled
regex for both measured schemas is byte-identical to the one the pipeline table
above was produced from (`reverse_input` 15,311 chars `sha df180d694c30`,
`process_data` 15,497 `sha ca76a852950f`), so `|S|`, edges, classes and tree
cannot have moved and the 20-minute compiles were not repeated.

Also in this round:

- `_expand_wildcards` now recurses into draft-04's **list-valued `items`**.
- `synthesize_instance(expand_wildcard=…)` was a no-op unless `from_bfcl=True`,
  while `build_regex` expanded on both paths — so the gate could compare an
  instance against a language it was not drawn from. Wired.
- `docs/RESULTS.md` gains **the third denominator category**, which neither the
  brief nor this log had named: the pre-gate `n=130 / avail=130 / no skips`
  artifacts (`exp_e5_grammar130_*`, `exp_seed{1,2,3}_map`, `exp_abl_*`,
  `exp_oomfix_*`, `exp_f64_*`, `exp_greedy_unconstrained`, `exp_noorder_unc`)
  keep their `n`, but scored both defective records **under the broken
  grammar** — so a post-fix `n=130` is not comparable to a pre-gate `n=130`
  either. 7 artifacts move to 130; the five `exp_h2_*` do not.
- `bfcl_python.value_regex`'s fall-through now cross-references
  `WILDCARD_ANYOF_TYPES` as a second, narrower, **declared** reading of `any`.
- Reviewer's find: `prefixItems: [{}]` was a fifth escape route the tester's
  known-defective list did not name. The fix repairs it too, unchanged.
