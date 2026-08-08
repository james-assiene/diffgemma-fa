# Improvement plan — closing the 0.641 → 0.385 argument-accuracy gap

Synthesis of a four-agent investigation (2026-07-29): a manual failure analysis
of all 85 gap items, a project history audit, and three researchers
(grammar/tokenizer lens, sampler/inference lens, adversarial prioritizer). All
counts below were independently re-derived by the prioritizer from the raw
artifacts; discrepancies it found are folded in.

## The diagnosis, in one sentence

**The production grammar assigns probability zero to every single one of the
130 outputs the unconstrained 0.641-accuracy arm actually emitted (0/130
verbatim acceptance)** — so every constrained decode is forced off the model's
plan at multiple canvas positions, and the failure taxonomy is the debris of
that forcing, not a knowledge deficit.

Supporting numbers:

- 58% of the 85-item gap is boundary corruption of values the model knows
  (`600→6000`, `USA→"USA1  "` — the trailing spaces are the model's swallowed
  pretty-print indentation; `\n` is inadmissible, `▁▁` inside a string is).
- 39% is grammar-optional structure collapsing to the shortest member
  (`{"body": {}}`, `[]`, wildcard `any` → literal `1`) — closing early is
  admissible and *free* (the post-stop tail is unscored).
- Genuine wrong-knowledge failures: 4%. Scoring artifacts: 0%.
- Failures are byte-identical across j0/j1/j2 in ~half the cases — a
  grammar×decoder effect, not sampling noise.
- Re-accepting the model's own habits fixes it at the string level:
  whitespace + optional ```` ```json ```` fence → **74/130** verbatim
  acceptance; + case-insensitive enums → **81/130**.
- The paper never loses more than 0.4 points in any of its 40 cells, so a
  25.6-point regression is a **port-configuration effect until proven
  otherwise**. Its two setup differences from us: confidence recomputed from
  the constrained distribution (Mar — unwired here), and prompts that teach
  the output rendering (ours specifies nothing).

## Evidence-quality rulings (prioritizer)

- Only **130 of 258** `live_simple` records were ever evaluated. Every table
  must say so until the back half runs.
- The "entropy_bound ruled out twice" claim is **retracted**: n=12 is 19 keys,
  <10% power for a 10-point effect — and it swept a threshold on the *wrong*
  (unconstrained) signal.
- Native arm denominators differ (unparsed records drop out); use the uniform
  291-key rescore: unconstrained 0.608, j1 0.368.
- Everything is one generation at seed 0; no seed-variance floor exists yet.

## Found and fixed during the investigation

`log_matmul`'s single row/col shift underflowed on real grammars: the unscored
tail pins the shift at 0 while genuine paths sit ~850 nats below; their
contributions fall under float64's subnormal floor and a provably non-empty
language came back Z == 0 (`live_simple_106-63-0`). Fixed with a two-band
shift, commit `fe01dbb`. The constrained n=130 arms were measured with the
buggy kernel and are re-running.

## The experiment plan (merged from all three researchers)

Order: cheapest-per-bit-of-evidence first; each has a decision rule.
GPU cost: each n=130 arm ≈ 80 min, n=30 ≈ 20 min, a --diagnose trace ≈ 2 min.

| # | Experiment | Cost | Decides |
|---|---|---|---|
| E0 | **Re-run constrained arms with the fixed kernel** (in flight) | 4 arms | New honest baseline; does underflow explain part of C/D? |
| E1 | **--diagnose traces on 5 gap records** (`2-2-0`, `3-2-1`, `119-75-0`, `102-61-0`, `120-76-0`): per-step top-5 unconstrained vs renormalized tokens, unconstrained-vs-q entropy at junk positions, stop step | ~10 min | Confirms/kills H1 at token level; tests whether junk enters early (SC dynamic) or only at the final draw; tests the Mar-mismatch signature before paying for E3 |
| E2 | **Prompt contract** — "compact single-line JSON, no code fences, include every listed key"; j1 + unconstrained, n=30 | ~40 min | Zero-code test of the mismatch family; if j1 jumps ≥15 pts the recompile becomes optional |
| E4 | **Grammar bundle** — P1 whitespace `( |\n {0,6})?` + P2 optional fence + P4 case-insensitive enums; n=30 | ~20 min + CPU compile of 130 schemas | The direct fix; measured at string level already (0→81/130) |
| E3 | **Wire Mar** — accept + stopping entropy from `q_i` (`constrained_marginals` is built and tested, never called) | plumbing + 20 min | The paper's larger accuracy lever (+8 pts on Dream); E1 pre-screens it |
| E5 | **Combine winners**, gate at n=30 (≥ +15), confirm at n=130, McNemar-paired | ~100 min | The ship decision |
| E6 | **Controls** — seeds {1,2} n=30; near-greedy T=1e-12 pair | ~80 min | Noise floor; the paper's T=0 column |

## Ranked fix backlog behind the experiments

Grammar (P-series, all string-level verified on the real corpus):
P1 whitespace; P2 fence (required-fence variant if MAP returns); P3
all-required + nullable + null-strip (the model omits keys in 1/126 outputs —
optionality is pure attack surface; GT audit: only 4/333 optional params
demand omission, and nullability saves those); P4 case-insensitive enums
(cannot create a new wrong answer); P5 ThinQ nested-body `anyOf` fix; P6
`minItems: 1` on required arrays; P7 forbid `{}[]` in string interiors (ship
last, over-constraint — 1 known GT loss, declared); P8 wildcard-`any` branch
pruning.

Sampler (ranked, with the J2≡J1≡J0 result honored — variant choice is dead):
S1 constrained self-conditioning (support mask is p-independent → compute once
per block; the only untried lever on the shared path that lets the model
re-plan); S2 tail-weight prior γ (declared prior; exact for the modified
target; complements P1 — γ fixes length economics, P1 fixes closer mass; γ is
two-sided, sweep {0.3, 0.5, 0.7} with the stop-position histogram); S3
automaton-aware stopping + entropy_threshold on the constrained entropy; S4
Mar (corrected expectation: low as standalone accept-rule change, high as
diagnostic + paper fidelity); S5 emission un-tempering (both directions
possible; sharpening protects correct strings in the aligned regime); S6
best-of-K joint draws, length-normalized (decoding heuristic, label it so);
S7 R1/flattening only as the 2×2 cell after S1.

Rejected: digit-count bounds from GT (benchmark leakage, and it wouldn't even
fix C); reusing the old entropy_bound sweep (underpowered + wrong signal —
recalibrate only after regrammaring).

## Standing cautions

- P1/P2 roughly double |S|; ThinQ crosses the 512 bucket → re-check the tree
  overhead crossover and bucket ladder before any §7.3 timing.
- Recalibrate entropy bounds *after* the grammar changes, not before.
- The unconstrained-vs-constrained comparison conflates constraint cost with
  rendering cost — state both numbers, never merge them.

---


> ## ⚠️ RETRACTION (2026-08-08): the Countdown result is a measurement artifact
>
> **The Countdown rows below are withdrawn.** `pipeline.compile_regex` mis-lifts
> an unbounded repeated *group*: `countdown_regex` is
> `step(?:\n step){0,3}`, and the string `3*4=12\n12+5=17` **matches the regex
> while the compiled automaton rejects it**. The grammar admits only
> **single-step** solutions; most Countdown problems need two or three.
>
> So `CS = 1.000, solved = 0.004` did not measure "constrained decoding fails at
> reasoning". It measured a grammar that could not contain the answer — which is
> also why all 250 emissions are single-step.
>
> Verified independently by regex-vs-DFA acceptance on the real compiled
> automaton. Blast radius checked rather than assumed:
>
> | grammar | affected |
> |---|---|
> | Countdown `step(?:\n step){0,3}` | **BROKEN** |
> | JSON arrays `(,item)*` | no — accepted at every length |
> | Sudoku (explicit 4-row concatenation) | no |
>
> **The BFCL and Sudoku conclusions stand.** Countdown must be re-run after the
> lift is fixed; until then it supports no claim in either direction.
>
> Found by the tester→coder→reviewer audit (CLAUDE.md), by a tester that was
> told to check the compiled automaton rather than the regex.

## Cross-task results (2026-08-03): the constraint is not free everywhere

First evaluations outside BFCL, n=250 each. The picture BFCL alone gave was
not general.

> **Superseded 2026-08-08.** This table predates both the Countdown retraction
> above and the scorer fixes of the measurement audit. Every Countdown row is
> withdrawn (the grammar could not contain a multi-step answer), and the
> unconstrained Countdown figure below is wrong for a second, independent
> reason — the scorer charged the last step of every emission with "unparsable
> step" because SPEC §3.5's unscored tail put `<turn|>` inside it; rescored it
> is **0.236**, not 0.048. **`docs/RESULTS.md` is the current table.** Kept here
> only because the argument that follows was built on it.

| task | arm | CS | solved |
|---|---|---|---|
| countdown / unconstrained | | 0.004 | **0.048** |
| countdown / constrained (MAP) | | 1.000 | **0.000** |
| countdown / constrained + 64-token scratchpad | | 1.000 | **0.000** |
| sudoku / unconstrained | | 0.000 | **0.864** |
| sudoku / constrained (MAP) | | 1.000 | **0.000** |
| sudoku / constrained + 64-token scratchpad | | 1.000 | **0.004** |

**The constraint takes an 86%-accurate Sudoku solver to 0%.** On BFCL it cost
nothing; here it costs everything. This is not a measurement artifact: the
Sudoku failures are row and column violations, *not* overwritten givens, so
the grammar is pinning the prefilled cells exactly as designed.

### A hypothesis that was tested and died

First guess: the grammar admits only the grid, so it removes the model's
scratchpad. Wrong on two counts. The unconstrained model emits a correct grid
**directly** (`2143|4321|3214|1432`) with no reasoning at all, so it was never
using a scratchpad; and widening SPEC §3.6's channel header from 8 to 64 free
tokens recovered nothing on either task (sudoku 0.000 -> 0.004, countdown
0.004 -> 0.000). The model spent the space echoing partial grid rows.

A regex-shaped scratchpad was tried first and **OOM-killed the host** — a
near-Σ character class repeated thousands of times, lifted over a
262,144-token vocabulary, which is exactly SPEC §4.2's "regex too large".
The bounded token chain the channel header already uses is the mechanism that
works.

### The surviving explanation

**The constraint is only worth having when the grammar captures the actual
correctness condition.** `sudoku_regex` pins the givens and the digit alphabet
and encodes *nothing* about rows, columns or boxes — so it adds no information
about what makes an answer right, while the MAP emission replaces the model's
own preferred output with the argmax of a product of per-position marginals
over a much larger format-valid set. For BFCL the JSON schema largely *is* the
correctness condition, which is why constraining there is free.

That predicts `--emission sample`, which draws from the constrained posterior
rather than maximising it, should sit closer to the unconstrained rate.
Running.

### What this changes about the project's claim

The honest headline is no longer "the guarantee is free". It is: **the
guarantee is free when the grammar encodes the task's correctness condition,
and can be catastrophic when it only encodes the output's shape.** That is a
more useful finding than the single-dataset version, and it would have been
invisible without a second and third dataset.

