# Phase 5 findings (in progress) — entropy calibration, and J0 vs J1 resolved

Date: **2026-07-28**. SPEC §3.4 calibration plus the diagnosis Phase 4 left open.

---

## 1. Open question 2 is resolved, and the answer is the opposite of the prior

SPEC lists this as open and genuinely uncertain:

> **§3.1 J0 vs J1.** J1 gives a stronger invariant (every canvas ∈ C) but shifts the model's inputs
> off-distribution. **Genuinely open.**

and frames J0 as the safe default — *"J0 — ship this"* — with J1 carrying the risk: *"the model was
trained to denoise uniform random noise, not grammar-valid noise."*

**Measured: J0 is the one that fails, and it fails badly.** Same prompt, same grammar, same
`entropy_bound = 0.1`, per-denoising-step emission traced through `jax.debug.callback`:

| step | J0-map emission | J1-sample emission |
|---|---|---|
| 0 | `{"user_id":7}` | `{"user_id":6}` |
| 10 | `{"user_id":0}` | `{"user_id":5}` |
| 11 | `{"user_id":0}` | `{"user_id":97}` |
| 12 | `{"user_id":0}` | `{"user_id":17890}` |
| 13 | `{"user_id":0}` | `{"user_id":97890,"special":":black"}` |
| 14 | `{"user_id":1}` | **`{"user_id":77890,"special":":black"}`** |
| 21–47 | `{"user_id":1}` (flipping 0/1/6/8) | stable |

Ground truth: `user_id: 7890, special: "black"`.

### Why J0 starves the emission

The tell is steps 21–47 of the J0 column: **mean entropy is `0.0000`** — the model is maximally
confident — and the MAP emission is *still* flipping between `{"user_id":1}`, `{"user_id":0}`,
`{"user_id":6}`. A confident model whose constrained MAP is arbitrary means the confidence is about
something else.

It is. Under J0 the **trajectory** is stock uniform renoising, so the model never sees valid JSON in
its input canvas. It converges confidently on a *natural-language* answer, and its marginals are
peaked on natural-language tokens. The constrained emission is then choosing among tokens that all
carry negligible probability, and the only remaining signal is length — so it takes the shortest
member of the language.

Under J1 the trajectory *is* the constrained draw, so the model sees well-formed JSON in its own
input, its marginals become informative about that JSON, and the emission converges to the answer.

**This is exactly the mechanism SPEC §3.7 describes, from the other direction:**

> The support-projection mask is *sound* but *insufficient*… it does not change the constrained
> posterior at all. **Its entire value is the self-conditioning feedback and the entropy signal.**

J0 has an unconditional guarantee and **zero feedback**. That trade is much worse than SPEC
anticipated: the guarantee is preserved and the *content* collapses.

### What this means for the design

- **J1, or J0 plus constraint feedback, is required — J0 alone is not shippable.** SPEC's "J0 —
  ship this" should be reversed, or J0 must be paired with `--self-cond=constrained` (§3.7's mask in
  `logit_shaper`, **not yet implemented**), which is the other channel by which the constraint could
  reach the model's inputs.
- **This also retires the Phase 4 hypothesis** that MAP's length bias was to blame. The bias is
  real, but it is not the cause: J1 uses the *same* joint machinery over the same grammar and does
  not collapse. What differs is the information in `p_i`.
- The two mechanisms are independently testable and both are cheap: `--variant=j1` (done) and
  `--self-cond=constrained` (outstanding). Phase 5 should ablate them as a 2×2.

---

## 2. SPEC §3.4 entropy-bound calibration

*(Results table filled from `artifacts/phase5_calibrate_j1.json` — see `docs/RESULTS.md`.)*

Sweep as SPEC prescribes: `entropy_bound ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}` (stock default
`0.1`) at a fixed 48-step budget, `--variant=j1 --emission=sample`, `live_simple` slice.

**Coverage caveat, stated up front:** this is **12 records of `live_simple`**, not the 100-example
dev slice SPEC asks for, and one split rather than a dev set spanning tasks. It is enough to see
the shape of the curve and to compare against the stock default; it is **not** enough to select a
production value. `entropy_threshold` (the `EntropyEarlyStop` half of §3.4's 2-D grid) is **not**
swept at all.

### Metric

Accuracy uses **BFCL's own normalisation** — lowercase and strip `",./-_*^` (SPEC §4.8) — rather
than exact string equality. This matters: a predicted `":black"` against ground truth `black` is a
BFCL **match**, and scoring it as a miss understates accuracy. An earlier version of this script
used strict equality; the numbers here supersede it.

Reported alongside, never instead of:

- **CS** (constraint satisfaction), checked by an independent simulator;
- **parsed** — the emission is valid JSON;
- **non-empty** — at least one argument is not `""`/`null`/`[]`/`{}`, which is the quantity Phase 4
  found collapsing.

---

## 3. Known residual: a systematic off-by-one at the value boundary

Both surviving examples show an extra leading character inside the value:

| ground truth | emitted |
|---|---|
| `7890` | `77890` |
| `black` | `:black` |

This is a **token-boundary interaction, not a grammar bug**. Gemma's vocabulary contains multi-
character tokens spanning the separator (`:7`, `":`), so a path that emits `:7` and then `7890`
reads as `":77890"` — and it is a genuinely valid member of the language, so the constrained
sampler is right to allow it. The model's own token-level distribution put mass there.

BFCL's normalisation absorbs the `:black` case (the `:` is stripped) but not `77890`. Worth
reporting as a distinct failure mode from format violation, and a candidate for the grammar to
forbid a leading zero-width duplication if it proves common at scale.
