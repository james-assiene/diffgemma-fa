# Phase 2 findings — the reference implementation and the exactness harness

Date: **2026-07-27**. Deliverable: `diffgemma_fa/infer/reference.py`.
**542 tests green** (272 of them Phase 2's).

CLAUDE.md calls the exactness harness *"the most valuable artifact in this repo"*, because a
subtly wrong sampler still produces **valid strings** — nothing else in the project would catch it.
So the standard here is exhaustive enumeration in float64, not spot checks.

| file | what it is |
|---|---|
| `diffgemma_fa/infer/reference.py` | plain float64 numpy: `W`, `M_i`, forward–backward, `q_i`, complement-aware `r_i`, chain sampler, reference tree sampler, max-plus MAP, brute-force enumeration |
| `tests/test_exactness.py` | SPEC §6.1's eleven tests — **254 cases, zero skips** |
| `tests/test_numerics.py` | SPEC §6.3 — 18 cases |

---

## 1. Exit criterion (SPEC §8, Phase 2)

> §6.1 fully green, **including tests 4 (NFA tree==chain), 7 (padding), 10 (reverse order) and
> 11 (complement + topk + tie-break)**. The bedrock.

All eleven, all green, on **both DFAs and NFAs**:

| # | test | status |
|---|---|---|
| 1 | `Z` from forward–backward == `Z` from enumeration (rtol 1e-10) | ✅ 36 cases |
| 2 | `q_i` == exact marginal; `Σ_v q_i(v) == 1`; `a_i·b_i` `i`-invariant | ✅ 52 cases |
| 3 | chain sampler χ² against the exact posterior | ✅ 8 cases |
| 4 | **tree == exact posterior on NFAs**, and the `∃` form is measurably wrong | ✅ |
| 5 | constrained MAP == exhaustive argmax, NFAs included | ✅ 48 cases |
| 6 | every sample and MAP output accepted by an independent simulator | ✅ 16 cases |
| 7 | degenerate: `\|C\|=1`, `C=V^L`, `C=∅` raises, dead states, `L=1`, **`L` not a power of two** | ✅ |
| 8 | non-point-mass start vectors | ✅ 8 cases |
| 9 | budget-aware `b_L = 1[d(s) ≤ R]`, incl. `R` too small → raises | ✅ 7 cases |
| 10 | suffix product operand order | ✅ 7 cases |
| 11 | complement-aware `r_i(v)`, topk `K` bound, `segment_min` tie-break | ✅ 44 cases |

**Zero skips.** Test 11 initially skipped 9 cases because random automata rarely produce *mixed*
Pos/Neg polarity at `V = 4`; the generator now searches for a mixed instance rather than skipping,
since mixed polarity is the entire point of that test and a silent skip is exactly the "silently
capped coverage" CLAUDE.md forbids.

Statistical hygiene per SPEC: fixed seeds throughout, **Bonferroni-corrected** χ² thresholds
(`α = 0.001 / 200`), and a documented re-run protocol — bump `SEED_OFFSET` and re-run; a real bug
fails at every offset, a fluke moves.

---

## 2. The one real bug this phase found

### `is_dfa` in the conventional sense is the WRONG gate for eq (8)'s fast path

SPEC §2.6 says to gate the cheap `∃` token draw on "a DFA (one edge per `(s,s')`, disjoint
labels)". **The parenthetical is doing all the work, and I got it wrong first time.**

Consider:

```
0 --{0,1}--> 1
0 --{1,2}--> 1        # parallel, SAME destination, overlaps on token 1
```

Every `(state, token)` has exactly one destination, so this is **deterministic** by every
conventional definition — yet token 1 is carried by two edges between the same state pair, giving
it multiplicity 2. The `∃` form ignores that and draws from the wrong distribution.

`reference.py` now distinguishes them explicitly:

- `is_deterministic` — at most one destination per `(state, token)`. The classic property.
- **`has_unit_multiplicity`** — no `(src, dst, token)` carried by more than one edge. **This is the
  eq (8) gate.**
- `is_dfa` — both.

**Why the compiled artifacts are nonetheless safe.** `compile/automaton.py: _group_edges` collapses
transitions into one edge per `(src, dst)` carrying the *union* of their labels, which makes
`has_unit_multiplicity` true **by construction** — for any automaton, deterministic or not. So the
Phase 1 output can always take the fast path. The distinction bites only on hand-built or ungrouped
automata, which is precisely what a reference implementation deals in.

Pinned by `test_4d_unit_multiplicity_not_determinism_is_the_gate` and
`test_4e_grouping_restores_unit_multiplicity`.

---

## 3. Two test-design corrections worth recording

**Do not assert seed-identity between the tree and the chain, or between the two eq (8) forms.**
SPEC says this for the tree-vs-chain comparison, and the same applies to `∃` vs
multiplicity-weighted: they consume a *different number of RNG variates*, so identical seeds give
different draws even where the distributions coincide exactly. `test_4c` now compares the
**conditional token distributions analytically** given `(s_i, s_{i+1})` — exact, deterministic, and
sharper than any sampling comparison.

**A stress automaton must actually forbid tokens.** The first `ring_automaton` admitted all `V`
tokens from every state, so its `M_i` had unit row sums. Consequences: the unnormalized product
never underflowed (defeating §6.3's whole purpose) and `q` had no exact zeros (leaving §2.4's clamp
rule with nothing to guard). The fixed version leaves 2 of 8 tokens on no edge, which makes every
`M_i` genuinely sub-unit and `q` genuinely zero-valued.

---

## 4. §6.3 numerical stability — measured

| claim | result |
|---|---|
| scaled path finite at `L = 256`, `p ~ softmax(N(0,1)·10)` | ✅ no NaN/Inf, `log Z` finite |
| scaled `log Z` `i`-invariant at `L = 256` | ✅ drift `< 1e-8` across all `i` |
| **unscaled path underflows in fp32 at `L = 256`** | ✅ `a` hits exactly zero well before position 256 |
| scaling is free, not an approximation | ✅ scaled and unscaled `log Z` agree to rtol 1e-12 at short `L` |
| `q` has exact zeros | ✅ |
| naive `-Σ q log q` is `NaN`; the clamp fixes it | ✅ both directions asserted |
| `-inf` logits poison an embedding matmul; `-1e30` does not | ✅ both directions asserted |
| MAP at `L = 256` finite, score == Σ log p of chosen tokens | ✅ rtol 1e-10 |

The underflow test is deliberately written to fail loudly if it ever *stops* failing — that would
mean the scaling argument needs revisiting, not the test.

---

## 5. Not done

- **The dynamax `O(S)` backward-sampling operator is not prototyped.** SPEC §2.6 flags it as worth
  trying, and it is listed under Phase 3's deliverables, not Phase 2's.
- **No JAX.** Everything here is numpy by design; Phase 3 is the JAX port and will be
  differential-tested against this file (SPEC §6.2, up to `|S|=64, |E|=512, L=32`).
- **Enumeration is `V^L`**, so exactness is established at `V=4, L≤8` exactly as SPEC prescribes.
  `L=256` behaviour is covered by §6.3's stability tests and by acceptance checks, not by
  enumeration — that is inherent, not a gap being papered over.
- `marginals_complement_aware` materialises complements densely. Fine at `V=4..32`; the JAX path
  must use the CSR form from `compile/classes.py` instead, and Phase 3 must differential-test the
  two against each other.
