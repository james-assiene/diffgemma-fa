# Handoff — state at `bf14e76`, 2026-08-13

Written to make this project resumable after a gap. Nothing here is a plan; it
is a description of where things actually stand, what is known to be broken and
why, and what the cheapest next move is for each open item.

**Read `README.md` first for what the project does.** This file assumes it.

---

## 1. What works, and how strongly

**Constrained JSON decoding works.** This is the headline and it is measured,
not projected:

| | unconstrained | constrained |
|---|---|---|
| schema valid | 0.631 | **1.000** (130/130) |
| constraint satisfaction | 0.000 | **1.000** |
| argument accuracy | 0.639 | 0.663 |
| exact call | 0.500 | 0.531 |

`exp_e5_grammar130_map` at `4f9a6e3`, n = 130 of 130 available, no records
skipped, zero partition failures, zero OOMs, under a coded hard stop that would
have halted the run below 1.000. The unconstrained baseline was re-verified in
the same run and reproduces verbatim at HEAD, so it is current rather than stale.

**All 4,549 BFCL-Live schemas compile** (was 4,538 before `15cdc47`).

**~1,540 tests pass** with one deliberate `xfail(strict)` (§3.2 below). The
verification stack is no longer float64 checking float64: the arbiter is now
tested against exact `fractions.Fraction` enumeration and 60-digit `decimal`
chains.

### What that result does *not* license

- **The accuracy movement is not attributable.** Schema validity 0.992 → 1.000
  is — the failing record is named and its mechanism measured. Argument accuracy
  0.650 → 0.663 is four arguments of 291 across a gap in which four commits
  touched the `j0/map` path. Not a one-variable delta; assign no cause.
- **The constrained block straddles two grammars.** This row is post-`15cdc47`;
  `j0-sample`, `j1-sample`, `j2-sample`, `mask-sample` and `j0-map` are not, and
  they decoded records 117 and 122 under a grammar that rejected the correct
  object. Pairs legitimately against the unconstrained baseline only. The
  per-record `rows` lists in each artifact permit a CPU-only restriction to
  common records for the rest.
- **One split, one seed, 130 of 258 records.** A prefix is not a random sample —
  BFCL ids cluster by schema family.

---

## 2. Not implemented

These are absent from the tree, not broken. Established by search during the
2026-08-08 audit and recorded in `docs/RESULTS.md` under "Coverage and caveats".

| SPEC | what | notes |
|---|---|---|
| **§3.7 `--self-cond=constrained`** | no `logit_shaper` subclass applying a support mask exists anywhere in `diffgemma_fa/` | **The most interesting gap.** SPEC §1913 *recommends* J0 paired with it as a ship option, and it is the only untried lever on the shared path that lets the model re-plan. `docs/RESULTS.md`'s headline names it as the likely missing piece after Mar turned out to be the wrong suspect. |
| §3.3 `--trajectory=R1/R2` | no flag, no code path | every arm is R0 |
| §3.4 `entropy_threshold` | structurally inert | `early_stop_fn` is passed at **zero** production call sites (`eval/run.py`, `eval/run_tasks.py`, `scripts/phase5_calibrate.py`, `scripts/phase4_e2e.py` all omit it); the only installer is `scripts/phase0_smoke.py`. So §3.4's *required* two-threshold sweep has only ever covered `entropy_bound`. |
| §3.8 refusal-branch union | unimplemented | BFCL's 1,124 irrelevance records have no path to a score |
| §7.1 datasets | 5 of 6 never run | xLAM is gated on an `HF_TOKEN`; GSM-Symbolic ≈ 0.5 day; **Spider is structurally infeasible dense** — 19,509 states, 778 GB tree, 389 GB for the `M_i` leaves alone |
| §3.4 bounds | 0.03 and 0.3 | the mar sweep covers 0.01/0.1/0.5 and was extended to 0.003/1.0, but those runs are invalid (§4) |

---

## 3. Not working — known defects, with reasons

### 3.1 The Sudoku `mask` closure — cause unknown, prediction falsified

**Symptom.** `task_sudoku_mask` returns `cs = 1/250`, identical to its published
pre-fix value, *after* the `prefix_suffix` repair (`2c9f9d9`). `solved` did move,
0.012 → 0.024.

**Why this was expected to be fixed, and why that reasoning was wrong.** The
mechanism was measured, not guessed: 16 of 256 positions had empty support
pre-fix, so `where(r > 0, logits, -1e30)` produced an all-sentinel float32 row,
the Gumbel noise was swamped, and `categorical` returned **token 0
deterministically**. The recorded emissions carry exactly that signature
(`'<|channel>124\n4321\n2413\n12422\n\n1'` — first grid row a digit short). All
of that is true. **What was never verified is that it was the only cause, or
that removing it was sufficient.** "This defect explains the symptom" and
"removing it fixes the arm" are different claims.

**Three candidates, none tested:**

1. Another defect in the mask path that the `u` fix does not touch.
2. Support is now correct but the *draw* is still wrong — `sampler.py`'s mask
   branch is deliberately unflagged, so a correct `u` can still produce a bad
   sample and nothing reports it.
3. **`cs` was never the right metric here.** Sudoku's grammar carries SPEC
   §3.6's `<|channel>NAME\n<channel|>` header, which the `mask` variant has no
   reason to emit — so `cs ≈ 0` may be structural rather than a defect, in which
   case the *gate* was wrong and not the fix.

**Cheapest next move:** (3), and it costs no GPU. The emissions are on disk in
`artifacts/task_sudoku_mask.json`; check whether the accepted record differs from
the other 249 by carrying the header. **Do not lower the gate threshold to make
it pass** — that is the move CLAUDE.md's opening anecdotes are about.

### 3.2 `reference.marginals_complement_aware` cancels from 69 nats

**Pinned as `xfail(strict=True)`** in
`tests/test_audit_infer.py::test_reference_complement_aware_marginals_survive_a_10e90_p_span`.

It forms a negated class as the linear difference
`Σ_Neg U[c] − Σ_{Neg, v∈N_c} U[c]`, which cancels catastrophically as the p-row
dynamic range grows. Measured against a `Fraction` oracle: **onset at 69 nats**
(2/254 wrong at 4.19e-06), max err **1.000** at 150 nats, worst relative error
1.32e+10 at 207. Plain `marginals` is exact to ~1e-15 on identical inputs. The
shipped `T = 0.4` admits 150 nats per position, so **the onset is inside
production reach.**

**This is a hole in the arbiter**, so anything the reference certified in that
regime is *uncertified* — not wrong, unverified.

**The shipped JAX path is clean and this was measured:** 420 instances of
`marginals.class_weights` against the same oracle across 0/69/150/207/250 nats,
**0 wrong at every span**, because it forms the class as
`outside + sel @ gathered` — a sum of non-negative terms.

**The repair is small and known.** An earlier note claiming it needs compensated
or log-space summation, or that it changes what SPEC §4.4 Layer 2b prescribes,
is retracted — Layer 2b's kernel already uses the non-subtractive form. The fix
is the move made twice already in this codebase (`prefix_suffix` → reference,
and the MAP floor): **make the arbiter use the form the kernel it certifies
already uses.**

**This is the third instance of one shape.** `class_weights` and
`scatter_edge_mass_to_tokens` were each rebuilt to remove exactly this algebra
(`da1294a`). Worth grepping for subtract-shaped class arithmetic before assuming
there is no fourth.

### 3.3 Countdown's `1 -1=1` degenerate emissions — three dead hypotheses

84 of 250 MAP emissions are degenerate (`1 -0=1`, `1 -1=1`) and **none solve**.
Solve rate rises monotonically with emitted length (1 step: 0.011, 2: 0.100,
3: 0.200).

Explanations proposed and **refuted by measurement**:

1. *Length economics* — MAP taking the shortest admissible string. Refuted:
   `--emission sample` cut degeneracy 33.6% → 11.6% and the solve rate did not
   move (0.036 → 0.032).
2. *The MAP clamp tie-breaking to the lowest token id.* Refuted: across 6 seeds
   with 1,162–1,213 fully-clamped `(class, position)` pairs, the `1e-30` floor
   changed **0/256 tokens and 0.00 nats**.
3. *Measured through defective leaves.* Refuted: `joint_map` calls neither
   `class_weights` nor the scatter nor `prefix_suffix` — verified by grep, three
   times, by different reviewers.

**The mechanism is unknown.** Do not restate any of the three.

### 3.4 Grammar is narrower than RFC 8259

Fully documented with root causes and fixability in **`docs/LIMITATIONS.md`**.
Summary: `\uXXXX` escapes and unsigned exponents (`1e5`) are rejected — both are
one-line deviations in `outlines-core`, both fixable, and both **provably
no-ops on BFCL** (0 real occurrences in 17,046 recorded emissions). Nesting
deeper than 4 is rejected and is a genuine bound, not an oversight.

### 3.5 Smaller open items

- **`prefix_suffix_off_by_one` mutant survives** all tests. `representable` has
  no coverage in a regime where the γ branch is decisive in both directions;
  killing it needs a test asserting `representable` is *True* under moderate γ
  alongside the existing "False at γ≈2000".
- **`up_sweep`'s linear `M_i / max(M_i)` underflows below T ≈ 0.1**, losing live
  edges at *unflagged* positions via `viable == False`. Pre-existing, below every
  asserted temperature, now pinned.
- **`writeOnly` raises while `readOnly` expands** — an asymmetry in
  `_ANNOTATION_KEYWORDS` that errs on the loud side. One word.
- **`artifacts/fa/bfcl/*.npz` is keyed by filename, not `schema_hash`.**
  `expand_wildcard` is in the fingerprint but the fingerprint does not gate that
  path, so a pre-fix automaton would be silently reused. **Clear that directory
  before any BFCL arm runs.** Verified absent as of this writing.

---

## 4. Invalid or unmeasured — do not cite

### Eleven artifacts are archived with no current version

The 16-arm queue archives each tag **before** running it, so its abort at arm 3
left these displaced into `artifacts/stale_pre_1aa8ccf/` with **nothing in
`artifacts/`**:

    eval_bfcl_live_simple_mask_sample   exp_marmap_b0.01
    exp_f64_ws_only_j1                  exp_marmap_b0.1
    exp_h2_grammar_j1                   exp_marmap_b0.5
    exp_h2_marmap                       exp_oomfix_j0
    exp_h2_stock_j1                     task_countdown_mask
                                        task_sudoku_j1_sample

**`scripts/phase5_report.py` globs `artifacts/` and would silently omit those
rows from a regenerated `docs/RESULTS.md` rather than failing.** So:

> **Do not regenerate `docs/RESULTS.md` until they are re-run.** Its committed
> text is the record.

Do **not** restore them to close the gap — they are pre-`2c9f9d9` and
pre-`15cdc47` and were archived for cause.

Separately, `phase5_report.py` reads each artifact's **stored** accuracy fields
and does not apply the 2026-08-08 rescoring, so a regeneration would also mix two
scorers. Rescore from `rows` first.

### Quarantined

`artifacts/unattributable_dirty_tree/` — three artifacts written while an agent
was editing `infer/scans.py` mid-run. Attributable to no commit. Regenerate,
never restore.

### Needing re-runs

- **Every `--emission=sample` arm and every `--confidence=mar` arm**, for the
  `prefix_suffix` fix. The `mar` arms need a *fresh baseline*, not a rescore,
  because the accept mask itself moved.
- **`task_sudoku_mask`** — but not until §3.1 is understood.
- **SPEC §7.3's bucket mix.** The per-bucket timings stand (they are a function
  of `|S|`), but the whitespace fix moved `|S|` from 155 → 59 on a 3-key BFCL
  schema — *below* outlines' stock 71 — so records move **down** a bucket. This
  is probably a **CPU-only** job: buckets 64–512 are already timed, so only the
  mix needs recomputing.
- **Not needed:** every `--emission=map` + `--confidence=mf` arm, exempt on
  traced code paths verified three times, plus a measured null for the MAP floor
  (0/256 tokens across 6 seeds).

---

## 5. How to resume

**Nothing is running.** Tree clean at `bf14e76`, everything pushed, GPU idle.

Approved-and-frozen queue scripts survive in the session scratchpad
(`*_LOCKED.sh`, mode 444) but **live under `/tmp` and will not survive a VM
restart.** Their designs are worth reconstructing from `docs/LOG.md`, which
records what each did and why. The reusable parts:

- a preflight that proves the run's own premise on CPU *and* proves the premise
  check is non-vacuous;
- a fingerprint over `diffgemma_fa/**/*.py` re-checked before every arm, with
  HEAD **recorded not enforced** (a docs commit once killed a 19-hour queue);
- aborts that name *which* condition tripped;
- archive artifact **and** `logs/<tag>.log` before replacing (logs are
  gitignored and `run()` opens with `>`);
- `[skip]` requiring the existing artifact to **parse**;
- per-arm config read back **out of the artifact**, not echoed from the script;
- a falsifiable prediction as a **coded** hard stop, early and cheap.

That last one is the highest-value pattern here: it cost 48 minutes and saved
~17 GPU-hours when a prediction of mine was wrong.

### Suggested order if picking this up cold

1. **§3.1 case (3)** — CPU, free, and it decides whether a whole arm's result is
   a defect or a metric error.
2. **§3.2 repair** — small, known, and it closes the third instance of a defect
   shape that has already appeared twice.
3. **§7.3 bucket mix** — probably CPU-only, and it repairs a table currently
   resting on a superseded grammar.
4. **The sample/mar re-runs** — ~19 GPU-hours, and the only thing standing
   between `docs/RESULTS.md` and being fully current.
5. **§3.7 `--self-cond=constrained`** — the actual research question. Everything
   above is maintenance; this is the untried lever the SPEC recommends.

---

## 6. Process rules that were earned, not assumed

All in `CLAUDE.md`. They exist because each cost real time here.

- **tester → coder → reviewer**, as three separate agents, with the tester
  writing from SPEC before the implementation exists. Six vacuous tests were
  found this way, one with a docstring prescribing the defective formula *as the
  specification*.
- **A reviewer reads every GPU- or artifact-writing command before it runs.** Its
  first application caught a 13.5-hour queue configured against a grammar no
  artifact in this repo had ever used.
- **Never measure a tree an agent is editing.** Cost three quarantined artifacts
  and ~40 minutes of a tester measuring the fix while reading it as the defect.
- **State the whole regime beside every number.** A blast-radius table was wrong
  by four orders of magnitude because two grammars were measured at
  `remaining=0` and a third at `remaining=1000` — which makes `b_L = 1[s live]`
  and *hides the very loss being measured*.
- **Mutate the implementation, not the harness.** A mutation registry that
  perturbs only the test file proves the harness self-consistent and nothing
  about the code.
- **Revert the real prior code, not a paraphrase of it.** Five mutants written
  as `git show HEAD:` splices passed 282 tests that hand-written equivalents
  would have failed.
- **Two runs agreeing is not replication if they share a seed.** Two agents
  independently measured `67/1168` and were both right about the number and
  wrong about it being invariant.
