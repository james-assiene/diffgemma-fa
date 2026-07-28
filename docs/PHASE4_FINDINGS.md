# Phase 4 findings — model integration

Date: **2026-07-28**. Deliverable: `diffgemma_fa/model/`. **670 tests green.**

| module | what it is |
|---|---|
| `model/state.py` | widened `SamplingState` carrying the automaton as **traced** arrays |
| `model/constrained.py` | budget-aware `b_L`, J1's flattened marginals, `joint_draw`, `joint_map`, `advance_states` |
| `model/sampler.py` | `ConstrainedDiffusionSampler` — forked `_sample_step` and denoising loop |
| `scripts/phase4_e2e.py` | the exit criterion: end-to-end on the real checkpoint |

---

## 1. Exit criterion (SPEC §8, Phase 4)

> End-to-end constrained generation on 10 prompts. Diagnostics: stop-token position per block,
> marker position per block, non-accepted count on the final step.

Real 26B-A4B checkpoint, real compiled BFCL-Live grammars, `--variant=j1 --emission=map`:

| | |
|---|---|
| prompts | **10** |
| grammars compiled | **10 / 10** |
| **accepted by an independent simulator** | **10 / 10** |
| prefix viable | 10 / 10 |
| stop-token positions | **7, 4, 13, 13, 5, 4, 5, 4, 5, 5** |
| first generation per new bucket | ~38 s (compile-dominated) |
| repeat on the same bucket | **9.6 s** |

Sample outputs: `{"user_id":0}`, `{"loc":"","type":"black","time":-2}`,
`{"location":":"}` — all schema-valid.

**SPEC §3.5 trap 4's own diagnostic passes decisively.** The trap is that a *scored* post-stop tail
makes a joint decode pay `(255−j)·log p(PAD)` to terminate at position `j`, so "the MAP places the
stop token at position 255 or never… clustering at 255 means this is still open". Measured stop
positions are **4–13**, never near the end. The unscored `ACC --Σ--> ACC` construction works.

**The 9.6 s repeat time is the no-recompile-per-request property showing up.** Different grammars
sharing an `|S|` bucket reuse the compiled program, which is exactly what SPEC §5.3(b) demands and
why the automaton had to be traced rather than stored on `self`.

---

## 2. The fp32 finding, and why it is caused by our own §3.5 fix

**Per-node max-normalization is not sufficient**, and the reason is a genuine interaction between
two parts of the design that each look fine alone.

SPEC §2.4/§2.6 prescribe normalizing every tree node by its max entry. That fixes the *overall*
scale. It does nothing about the **dynamic range inside one matrix**, and at the root that range is
fatal:

- `ACC --Σ--> ACC`, the unscored post-stop tail from §3.5 trap 4, has emission mass **exactly 1.0**
  by construction — so the root's max is **pinned at 1.0** and dividing by it is a no-op;
- a genuine constrained path is a product of per-token probabilities around `4e-6`, so at `L = 64`
  the root entry is `~1e-49`, and the **smallest positive entry measured on a real BFCL grammar was
  1.2e-288**.

float32's smallest subnormal is ~1e-45, so the whole joint underflows to **exactly zero**. The draw
then degenerates *silently*: it still returns 256 tokens of plausible-looking multilingual text and
passes every shape check.

This is SPEC's `Z == 0` cause **(c)**, except that the scaling is *present* and still insufficient.

### 2.1 float64 is necessary but **not sufficient at `L = 256`**

Switching the sampling path to float64 fixed it at `L = 64` (all draws accepted). At the real canvas
length it does **not**: `--emission=sample` over 6 prompts gave

| | |
|---|---|
| accepted | **4 / 6** |
| the 2 failures | no stop token, pure multilingual garbage |

against **10/10** for `--emission=map` on the same grammars. The failures are the same degeneracy —
at `L = 256` even float64's ~1e-308 floor is reached.

**So the sum-product sampling tree needs a log-space formulation** (`logsumexp` combines), not
merely a wider float. That is what §2.7 already does for MAP, and it is why MAP is unaffected:

> Use **log space, not `(max, ×)`**: exact, no scaling discussion, no underflow.

### 2.1a Fixed: `scans.up_sweep_log`, and it keeps the GEMMs

The naive `logsumexp` combine materialises `[S, S, S]` — `1e9` elements at `|S| = 1024`. Shifting
by the **row** max of the left operand and the **column** max of the right leaves an ordinary matmul
of matrices whose entries all lie in `[0, 1]`:

    C[i,j] = ra[i] + cb[j] + log( Σ_k exp(A[i,k] − ra[i]) · exp(B[k,j] − cb[j]) )

so cuBLAS still does the work and the `O(log L)` kernel-count property survives — asserted:
`up_sweep_log` emits **zero `while` loops and exactly `log₂ L` GEMMs**, same as the linear tree.
`jax.random.categorical` takes logits anyway, so the exponentiation that underflowed is never
performed at all.

**Re-measured end to end with the log-space tree, same 10 prompts:**

| `--emission=sample` | before | after |
|---|---|---|
| accepted | 4 / 6 | **10 / 10** |

And the two previously-degenerate `uber.ride` cases now produce
`{"loc":"","type":"comfort","time":22}` and `{"loc":"","type":"plus","time":2}` — whose `type`
values are **exactly BFCL's ground truth** for those two records. So the fix recovered not just
validity but correctness.

**`--emission=map` remains SPEC §3.9's default and is unaffected either way.**

### 2.2 What was done about it

- `model/constrained.require_x64()` refuses to run the sampling path without float64, with the
  measurement in its docstring.
- `infer/tree.sample_tokens` now returns `(tokens, valid)`, where `valid` is False if any position
  had no admissible token — i.e. the boundary draw degenerated. **Surfaced rather than silently
  returned**, because the degenerate output is indistinguishable from real text by inspection.
- Pinned by `test_root_product_underflows_in_fp32_on_a_real_grammar`, which asserts float64 works,
  fp32 gives exactly zero, and the root max really is 1.0.

### 2.3 Consequence for SPEC §5.6

The tree is `(2L−1)·|S|²·**8**` bytes for the sampling path, not `·4`. At the measured ~20 GB of
headroom the `|S|` ceiling falls from **3,128 to 2,211** (and 1,978 → 1,399 at 8 GB). MAP keeps the
4-byte figure. SPEC §2.6 and §5.6 updated.

---

## 3. A second finding: joint MAP prefers the minimal completion

The emissions are schema-valid but semantically empty — `{"location":""}`, `{"repos":""}`,
`{"user_id":0}`.

This is not a bug, it is what joint MAP *means* over a variable-length language. Every additional
token multiplies in a probability `< 1`, so if the grammar admits a short string — and it does,
because `outlines`' JSON string pattern permits `""` — the highest-scoring member of the language is
close to the shortest one. The unscored tail removes the *bias against* stopping (§3.5 trap 4) but
nothing pushes *toward* content.

`--emission=sample` produced visibly richer output where it worked (`{"user_id":77890,"special":
":black"}`), which is consistent with this reading.

**Implications for Phase 5**, worth stating before any accuracy number is quoted:

- A `CS = 100%` claim from these runs would be true and nearly meaningless — SPEC §3.8 already
  demands CS be reported *conditional on the grammar branch being taken*; this adds that
  **content-bearing** output needs checking too, not just membership.
- The grammar should forbid empty strings for required fields where the schema implies content, or
  MAP needs a length prior.
- This sharpens SPEC §7.5's expectation: constrained decoding fixes format, not content — here it
  fixes format *so* aggressively that content collapses.

---

## 4. Fork surface, as built

Phase 0's correction held up exactly. `_sample_step` **is** forked, because:

- `sample_next_canvas` never receives `state`, so `A_k` is unreachable there;
- `max_new_tokens` is not a `SamplingState` field, so `R` is not computable in `_sample_step`
  either — it is now carried explicitly.

`ConstrainedSamplingState` adds `automaton`, `max_new_tokens` and `cache_length`, and
`remaining_budget` bounds `R` by **both** the token budget and cache capacity — SPEC §3.1b's
**fourth** termination path, which Phase 0 added.

The sampler stays `frozen=True, kw_only=True` with only Python ints and strings on it; every
automaton array is traced. Verified: 19 constructor/protocol assertions, plus the accept-mask
recomputation matching the stock rule bit-for-bit.

---

## 5. Not done

- **J0 is not implemented.** Only J1 (trajectory == emission) and the J2 baseline path exist. J0
  needs `_ConstrainedCarry` with a separate `emit_canvas` and a widened `should_stop` fed that
  canvas rather than the trajectory (SPEC §3.1b closure 2). Given §2's finding — that the sampling
  emission is currently broken at `L = 256` — J0-sample would inherit the same problem; J0-map
  would not.
- **The log-space sum-product tree is the identified next step** and is not built.
- **No `EarlyStopFn` widening.** The stock chain is used as-is, so §3.1b's automaton-aware stopping
  conjunct (`A_{k+1} ∩ F ≠ ∅`) is **not** yet enforced.
- **No multi-block run.** All 10 prompts terminated inside one canvas, so the cross-block `A_k`
  threading is exercised by unit tests (`advance_states` vs the simulator) but not end-to-end.
- **`B = 1` only**, and no `--self-cond`, `--trajectory` or `--temp-schedule` variants.
- **The `logit_shaper` hook is untouched**, so `--self-cond=constrained` (SPEC §3.7) does not exist.
- **No accuracy measurement of any kind.** Acceptance by the simulator is not accuracy; that is
  Phase 5.
