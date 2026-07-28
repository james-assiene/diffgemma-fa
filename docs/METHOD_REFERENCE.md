# METHOD_REFERENCE.md — the method, as an adjudication reference

Authority document for code review of `diffgemma_fa`. Written by the `researcher` agent.
Purpose: let a reviewer decide whether a given line of code is mathematically correct.

Conventions used below:
- **[P]** = stated in the paper (Dang & Ermon, arXiv:2607.07026v1, 8 Jul 2026). Quoted verbatim
  where it matters.
- **[S]** = stated in `SPEC.md` but **not** in the paper (a design decision of this port).
- **[R]** = my own derivation/reasoning, flagged as such.
- **[?]** = I am not certain. Treat as an open question, not as authority.

---

## A. Provenance — what I could and could not access

| Source | Status |
|---|---|
| `https://arxiv.org/abs/2607.07026` (abstract page) | **Accessible.** Title/authors/abstract confirmed: Meihua Dang, Stefano Ermon, "Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata". |
| `https://arxiv.org/pdf/2607.07026` (full PDF) | **Accessible and fully extracted.** The fetch tool could not parse it (returned the compressed byte stream), so I downloaded it and extracted text locally with `pypdf` in a throwaway venv under the scratchpad — **19 pages, 60,142 characters, including Appendix A (forward–backward, edge-space *and* state-space) and Appendix B (the exactness proof)**. Everything marked [P] below is quoted or paraphrased from that extraction. |
| `https://arxiv.org/html/2607.07026v1` (HTML render) | Accessible; used only as a cross-check. Agrees with the PDF extraction. |
| `/home/ubuntu/diffgemma_fa/SPEC.md` | Read §2 in full, plus §3.1, §3.1b, §3.2, §3.3, §3.4, §3.5, §3.6, §3.7, §4.4, §4.5, §4.6, §5.6, §5.7, §6. |

**So: I have the paper. Claims below are attributed individually.** The extraction is text-layer
only — figures are recovered as text fragments and are reliable, but I did not render the math
images, so a *typographic* transcription error in an exotic symbol is conceivable. Every equation
below was cross-checked against SPEC's independent restatement and they agree; where they differ
I say so explicitly.

**Caveat on the extractor:** `pypdf` occasionally drops spaces around inline math (e.g. `zi ∈ Etaking`).
I have corrected obvious de-spacing; I have not silently corrected anything semantic.

### A.1 The two numbering schemes do not match — read this first

**The paper and SPEC.md both number equations (1)…(9), and they mean different things.**
Confusing them is the single easiest way to misread either document. The codebase follows
**SPEC numbering** (its comments say `# SPEC §2.6 eq (8)`).

| Paper eq | Content | SPEC eq | Content |
|---|---|---|---|
| (1) | forward masking process `q_{t\|0}` | (1) | `W[i,e] = Σ_{v∈label(e)} p_i(v)` |
| (2) | denoising step / commit set `U` | (2) | `M_i(s,s')` |
| (3) | **graphical model** `p_M(x_{1:L}) ∝ Σ_z …·1[dst(z_L)∈F]` | (3) | forward recursion `a_i` |
| (4) | **constrained mean-field posterior** (the target) | (4) | backward recursion `b_i` |
| (5) | **tractable product construction** | (5) | `u_i(e)` |
| (6) | midpoint conditional-independence factorization | (6) | `q_i(v)` (constrained marginal) |
| (7) | **`p(z_m\|z_ℓ,z_r) ∝ p(z_m\|z_ℓ)·p(z_r\|z_m)`** | (7) | **same statement** — the one number that coincides |
| (8) | edge-space initial joint draw `(z_1,x_1)` | (8a)/(8b) | state-space token draw |
| (9) | edge-space joint draw `(z_t,x_t)`, `t≥2` | (9) | **max-plus `M̃_i` for MAP — [S], not in the paper** |

SPEC §2.2's citation "`[V, Eq. 3]`" and §2.3's "`[V, Eq. 5]`" refer to the **paper's** (3) and (5).
Those two citations are correct.

---

## B. The graphical model: latents are EDGES, not states — and why

**[P], paper §3.2 and eq (3), verbatim:**

```
p_M(z_1 = e)              := 1[src(e) = s_0]
p_M(z_i = e' | z_{i−1}=e) := 1[dst(e) = src(e')]
p_M(x_i = v | z_i = e)    := 1[v ∈ label(e)]

p_M(x_{1:L}) ∝ Σ_{z_{1:L}} p_M(z_1) ∏_{i=2}^L p_M(z_i|z_{i−1}) ∏_{i=1}^L p_M(x_i|z_i) · 1[dst(z_L) ∈ F]   (3)
```

with `E := {(s,s') ∈ S×S | ∃v. s' ∈ δ(s,v)}`, `src(e)=s`, `dst(e)=s'`, `label(e) := {v | s' ∈ δ(s,v)}`.

**Why edges and not states.** [P] + [R]. The automaton's transition function is `δ: S×V → 2^S`;
the token `v` determines *which* transition is taken. If the latent were the state `s_i`, the
emission `p(x_i | s_i)` would not be well defined — the token also depends on `s_{i+1}`. Making the
latent the **edge** `z_i` makes the emission factor **local and node-local**: `1[v ∈ label(e)]`
depends on `z_i` alone. That locality is what makes the model a textbook HMM, and hence what makes
the mean-field reweighting a *product of a chain model with a fully factorized model*, which is
tractable and stays chain-structured (paper §4.2, citing Vergari et al.).

**[P] paper footnote 1, important:** *"These are unnormalized factors rather than conditional
probabilities; we treat them as such and implicitly normalize globally over length-L sequences."*
The paper repeats the point after eq (5): *"locally normalizing each factor would yield a
distribution that is different from `g`."* This is the source of bug signature **F9**.

### B.1 The state-space reduction is a theorem, not a shortcut

[P] Appendix A.2 gives the state-space form and — crucially — **justifies the reduction**:

> `β_t(s) = β_t(e)` for any edge `e` with `dst(e) = s` (well-defined since edge-space `β_t(e)`
> depends on `e` only through `dst(e)`).

[R] That is the whole content of the reduction: the edge-space backward message
`β_t(e) = Σ_{e'} g(z_{t+1}=e'|z_t=e)(Σ_v g(x_{t+1}=v|z_{t+1}=e'))β_{t+1}(e')` factors through
`dst(e)`, because the transition indicator `1[dst(e)=src(e')]` is the only place `e` appears.
Hence `|S|`-dimensional messages suffice, and cost drops from `O(L|E|²)` to `O(L|S|²)` [P].

**[P] Appendix A.2, verbatim state-space factors:**

```
g(s_0)                       = 1[s_0 = s_0^init]
g(s_t, x_t = v | s_{t−1})    = Σ_{e : src(e)=s_{t−1}, dst(e)=s_t}  g(x_t = v | z_t = e)
M_t(s,s') := Σ_{v∈V} g(s_t = s', x_t = v | s_{t−1} = s)
β_L(s) = 1[s ∈ F],    β_{t−1}(s) = Σ_{s'} M_t(s,s') β_t(s')
P(s_t = s', x_t = v | s_{t−1}) ∝ g(s_t = s', x_t = v | s_{t−1}) · β_t(s')
```

**The `Σ_{e : src=…, dst=…}` in `g(s_t, x_t=v | s_{t−1})` is the paper's own statement of the
edge-multiplicity weighting** that SPEC eq (8a) enforces. It is not a SPEC invention; it is what
"aggregating edges by their endpoints" means. See §C eq (8) and bug signature **F2**.

**Note on `z` in Algorithm 1.** The paper is deliberately loose: Figure 1(b) shows latents
`z_1 … z_L z_{L+1}` (i.e. **L+1** of them for L tokens), Algorithm 1 samples "boundary **states**
`(z_1, z_{L+1})`", and Appendix B says "Step 1 is a categorical sampling problem over the finite
**state space `S`**". [R] So in Algorithm 1 the `z`'s are the **state-space** boundary variables
`s_0 … s_L`, exactly SPEC §2.6's 0-based `s_0 … s_L`. SPEC's rendering is faithful.

---

## C. Every equation, exactly

Below, `p_i(v)` is the mean-field marginal at position `i`, i.e.
`p_i(v) = softmax(shaped_prediction_i)_v` [S] `= p_θ(x_i^0 = v | x^t)` [P].

### The target distribution — paper eq (4)

**[P] verbatim:**

```
x̂^0 ~ p_θ(x^0 | x^t, C) ∝ ( ∏_{i=1}^L p_θ(x_i^0 | x^t) ) · 1[x^0 ∈ C]        (4)
```

This is the **per-denoising-step** target. **[P] explicitly:** the *ideal* object
`p_dLLM(x^0 | x^0 ∈ C)` "is intractable", and the method instead "enforce[s] that samples drawn
from **each diffusion step** must satisfy the constraint". **Do not let a reviewer claim the
end-to-end generative distribution is exact — the paper does not claim it.** See §D.1.

### Paper eq (5) — the product construction

**[P] verbatim:**

```
x̂^0 ~ ( ∏_{i=1}^L p_θ(x_i^0 | x^t) ) · p_M(x_{1:L}^0)  ∝  Equation (4)                (5)

g(z_1 = e)             := p_M(z_1 = e)
g(z_i = e' | z_{i−1}=e):= p_M(z_i = e' | z_{i−1} = e)
g(x_i = v | z_i = e)   := p_M(x_i = v | z_i = e) · p_θ(x_i = v | x^t)
```

**Emissions are reweighted; transitions are untouched.** Reweighting transitions instead is a real
implementation temptation and is wrong (bug signature **F19**).

### SPEC eq (1) — edge emission mass

```
W[i, e] = Σ_{v ∈ label(e)} p_i(v)                                                     (1)
```

*Meaning.* The total mean-field probability mass that edge `e` can emit at position `i`; i.e.
`Σ_v g(x_i=v | z_i=e)`. **Indices:** SPEC §2.4 is **1-based** (`i = 1..L`); SPEC §2.6 and the code
are **0-based** (`i = 0..L−1`). `W` is `[L, E]` or `[E, L]` — check the layout, SPEC §4.4 mandates
`p` stored `[V, L]` column-major and `W` produced as `[E, L]`.

*Plausible-but-wrong.* Computing `max` instead of `sum` (that is eq (9), a different semiring);
or computing it over the *complement* set without adding back `total[i]` (SPEC §4.4 Layer 2:
`W_c^Σ[c,i] = total[i] − Σ_{v ∈ N_c} p_i(v)`). Dropping the `total[i]` term yields a *negative*
weight — usually caught. Using the **max table's** polarity flags to decide (SPEC §4.4: the two
tables have independent Pos/Neg partitions) yields a *plausible positive* wrong weight — not caught
except by SPEC §6.1 test 11.

### SPEC eq (2) — per-position transition matrix

```
M_i(s, s') = Σ_{e : src(e)=s, dst(e)=s'} W[i, e]     ∈ R_{≥0}^{|S|×|S|}                (2)
```

Identical to the paper's `M_t(s,s') := Σ_v g(s_t=s', x_t=v | s_{t−1}=s)` [P, App A.2].

*Meaning.* Sum over **parallel edges** between the same state pair. On a DFA there is at most one
such edge; on an NFA there can be several with overlapping labels. **This sum is what makes the
state-path distribution path-weighted**, and is the reason eq (8a) must be weighted too.

*Plausible-but-wrong.* Scatter with `.at[...].set(...)` instead of `.at[...].add(...)` — silently
drops all but one parallel edge. Shapes are identical, results are a valid distribution over a
*subset* of the paths. Only NFA tests catch it.

### SPEC eq (3)/(4) — forward and backward recursions

```
a_0(s)     = a_start(s)                          # 1[s=s_0] only for a DFA on block 0 — §5.7
a_i(s')    = Σ_s a_{i−1}(s) · M_i(s, s')                                               (3)

b_L(s)     = terminal factor                     # NOT 1[s ∈ F] in this port — §3.1b
b_{i−1}(s) = Σ_{s'} M_i(s, s') · b_i(s')                                               (4)

Z = a_0 M_1 M_2 ⋯ M_L b_L = Σ_{x ∈ C} ∏_i p_i(x_i)
```

**Attribution.** Eq (4) is the paper's `β` recurrence *verbatim*, indices and all [P, App A.2].
Eq (3) — the forward message `a` — **is not displayed in the paper's appendix**, which only defines
`β` and then does *forward sampling*. [R] It is the standard HMM forward message and is *required*
by the paper's own Algorithm 1 line 4 ("Sample boundary states `(z_1, z_{L+1})` from their joint
marginal") and by the constrained marginals of §4.2. SPEC's eq (3) is correct and necessary.

**Indices — the thing to check in code.** `a_{i−1}` is the state **before** position `i`; `b_i` the
state **after** position `i`. There are `L+1` boundary vectors `a_0..a_L` and `b_0..b_L` for `L`
positions. SPEC states this was verified against brute-force enumeration. In 0-based code:
`a[i]` = before position `i`, `b[i+1]` = after position `i`.

**Scaling is mandatory** [S]: `a_i` is a product of `i` sub-unit matrices and underflows fp32 to
exactly zero at `L=256`. Store max-normalized with accumulated log-scales; then
`log Z = log(a_i·b_i) + logscale_a[i] + logscale_b[i]` must be **`i`-invariant** — this is the
single best assertion in the whole stack.

*Plausible-but-wrong.* See **F1** (operand order in the reverse scan) and **F4** (off-by-one).

### SPEC eq (5)/(6) — per-position constrained marginals

```
u_i(e) = a_{i−1}(src e) · b_i(dst e)                                                   (5)
q_i(v) = p_i(v) · ( Σ_{e : v ∈ label(e)} u_i(e) ) / Z                                  (6)
```

**Attribution.** The *quantity* is the paper's: [P] §4.2 "Remasking confidence" — "Under a
constraint `C`, the appropriate score is the constrained marginal `p_θ(x_i^0 = x̂_i | x^t, C)`",
and Table 2's **Mar** column is this signal (Dream xLAM-Python 68.4 → 76.4 over **Mf**). **I did
not find a displayed formula for it in the paper**; eq (5)/(6) is SPEC's rendering of the standard
HMM posterior marginal. [R] It is correct — see the normalization identity below.

**Note `u_i(e)` deliberately does NOT contain `W[i,e]`.** The emission mass enters through the
explicit `p_i(v)` factor in eq (6). [R] Derivation:
`P(x_i=v) = (1/Z) Σ_{e : v∈label(e)} a_{i−1}(src e) · p_i(v) · b_i(dst e)`. Multiplying `u_i` by
`W[i,e]` **double-counts the emission** and is bug signature **F3**.

**The normalization identity — make it a unit test** [S]:
`Σ_v q_i(v) = (1/Z) Σ_e u_i(e) W[i,e] = a_i · b_i / Z = 1`.

**Complement-aware evaluation** [S, SPEC §4.4/§2.4] — under the class representation the scatter
in (6) is not a plain sparse scatter, because negated classes contribute *everywhere*:

```
r_i(v) = Σ_{c ∈ Neg} U_i(c) + Σ_{c ∈ Pos, v ∈ S_c} U_i(c) − Σ_{c ∈ Neg, v ∈ N_c} U_i(c)
q_i(v) = p_i(v) · r_i(v) / Z          where  U_i(c) = Σ_{e ∈ class c} u_i(e)
```

Dropping the first (global) term is bug signature **F17**: it produces a plausible, normalizable,
*wrong* `q`.

**`q_i` has exact zeros.** Never `log q` directly; use `jnp.log(jnp.maximum(q, 1e-30))` so that
`0·(−69) = 0`. For any write-back into a logits slot use a finite sentinel `-1e30`, never `-inf`.

### SPEC eq (7) / paper eq (7) — the midpoint recursion

**[P] verbatim, paper eq (6) then (7):**

```
p(x_{ℓ:r−1} | z_ℓ, z_r) ∝ Σ_{z_m} p(z_m | z_ℓ, z_r) · p(x_{ℓ:m−1} | z_ℓ, z_m) · p(x_{m:r−1} | z_m, z_r),  ℓ<m<r   (6)
p(z_m | z_ℓ, z_r) ∝ p(z_m | z_ℓ) · p(z_r | z_m)                                        (7)
```

SPEC §2.6 states the same with `P_{[ℓ,r)} = M_ℓ M_{ℓ+1} ⋯ M_{r−1}` (the transfer matrix from `s_ℓ`
to `s_r`) so that

```
P(s_m | s_ℓ, s_r) ∝ P_{[ℓ,m)}(s_ℓ, s_m) · P_{[m,r)}(s_m, s_r)                          (7)
```

**Indices.** `m = ⌊(ℓ+r)/2⌋` [P, Alg 1 line 10]. Half-open intervals: `P_{[ℓ,r)}` covers positions
`ℓ … r−1`, i.e. `r − ℓ` positions. SPEC §2.6 is **0-based**; the paper's Algorithm 1 is 1-based over
`1 … L+1`. The dyadic node set required is the **aligned** one (Blelloch/Brent–Kung), not any
prefix set.

**Root draw** [S, SPEC §2.6 / §5.7], the top of the recursion:

```
(s_0, s_L) ~ P ∝ a_start(s_0) · P_{[0,L)}(s_0, s_L) · b_L(s_L)
```

This is a **joint** categorical over the `|S|²` pairs (paper Alg 1 line 4: "Sample boundary states
`(z_1, z_{L+1})` from their **joint marginal**"). Drawing the two independently from their marginals
is bug signature **F8** — and note it still produces valid strings.

**Normalization is per-segment and local to `(z_ℓ, z_r)`** [P, Appendix B "Remark on unnormalized
conditionals"]: *"Different segments use different normalization constants, as they should… the
chain-rule decomposition ensures these segment-local normalizers compose into the correct joint
distribution."* Normalizing the transfer matrices themselves (row-stochastic) instead of normalizing
the `|S|`-vector of scores at draw time is bug signature **F9**.

**Scale factors cancel** [S]: in (7) the scale of `P_{[ℓ,m)}` and of `P_{[m,r)}` are scalars constant
w.r.t. `s_m`, so per-node max-normalization is free for *sampling*. It is **not** free for `Z` —
the log-scale sum must run over all `2L−1` nodes (bug signature **F10**).

**[V-P4, S] Per-node normalization is NOT sufficient in fp32, or even in float64 at `L=256`.**
The sum-product **sampling** tree requires a **log-space** formulation (`logsumexp` combines,
`infer/scans.up_sweep_log`), because the root's dynamic range — an unscored `ACC --Σ--> ACC` tail
pinned at exactly 1.0 alongside genuine paths at `~1e-288` — defeats scalar rescaling. MAP is
unaffected because §2.7 is already in log space. This is bug signature **F12**.

### SPEC eq (8a)/(8b) — the token draw, and why multiplicity is mandatory

```
e_i ~ P(e) ∝ 1[src(e)=s_i, dst(e)=s_{i+1}] · W[i, e]                                   (8a)
x_i ~ P(v) ∝ p_i(v) · 1[v ∈ label(e_i)]                                                (8b)
```

equivalently, in one step:

```
x_i ~ P(v) ∝ p_i(v) · |{e : src(e)=s_i, dst(e)=s_{i+1}, v ∈ label(e)}|
```

**Attribution.** [P] The paper's edge-space eq (8)/(9) are
`P(z_t=e, x_t=v) ∝ g(z_t=e | z_{t−1}) · g(x_t=v | z_t=e) · β_t(e)` — in edge space the latent *is*
the edge, so nothing has to be re-derived. The multiplicity requirement is a consequence of the
**state-space** reduction, and the paper states it there: `g(s_t, x_t=v | s_{t−1}) = Σ_{e: src=s_{t−1},dst=s_t} g(x_t=v|z_t=e)`
[P, App A.2]. That sum **is** the multiplicity count times `p_θ(v)`.

**Why it must be weighted** [R, and [V] in SPEC]. `M_i` (eq 2) sums over parallel edges. So the
state-path distribution you drew `(s_i, s_{i+1})` from is already path-weighted: a token `v` lying
in **two** parallel edges between the same pair contributed **twice** to `M_i(s_i,s_{i+1})`. If the
token draw uses `1[∃e : v ∈ label(e)]` instead of the count, the joint over `(path, string)` is
inconsistent with the marginal you sampled the path from, and you get a **third** distribution —
neither uniform on `C` nor path-weighted. SPEC reports the measured deviation on a 3-state NFA with
parallel overlapping edges at `L=4`: **`∃` form 1.7e-2, weighted form 2.1e-17** against the exact
posterior.

**On a DFA the two coincide** (one edge per `(s,s')`, disjoint labels), so gating the cheap `∃`
path on a verified `is_dfa` flag is legitimate. **Verify the flag is actually verified** — an
`is_dfa` that is set from the compiler's *intent* rather than checked on the arrays is a silent
NFA-correctness hole.

*Plausible-but-wrong forms, all of which emit valid strings:* the `∃` indicator; drawing `x_i` from
`p_i` restricted to `∪_e label(e)` over all matching edges (same as `∃`); drawing the edge uniformly
among matching edges instead of `∝ W[i,e]`; drawing the edge with `W` but then the token uniformly
over `label(e_i)` instead of `∝ p_i(v)`.

### SPEC eq (9) — max-plus MAP

```
M̃_i(s, s') = max_{e : src(e)=s, dst(e)=s'} max_{v ∈ label(e)} log p_i(v)              (9)
```

**Attribution: [S]. This is NOT in the paper.** I searched the full text: the paper mentions
"greedy decoding", reports `T=0` results, and says its approach "can be viewed as a strict
generalization of DINGO… supporting greedy decoding and sampling", where DINGO "leverages a dynamic
programming algorithm to perform MAP inference on deterministic finite automata". **There is no
greedy/MAP/Viterbi algorithm box and no `argmax` in the method section.** SPEC's `[?]` on this is
correct, and SPEC's max-plus construction is a `[D]` design decision of this port.

[R] **It is nonetheless the right object.** The paper's "greedy" is `T=0`, and
`lim_{T→0}` of sampling from `(∏_i softmax(ℓ_i/T)_{x_i})·1[x∈C]` is
`argmax_{x∈C} Σ_i ℓ_{i,x_i}` — exactly max-plus MAP. Replace `(+,×)` with `(max,+)` over
log-probabilities in (3)/(4) and take `argmax` in the tree recursion; `(max,+)` is a semiring, so
the identical tree gives `O(log L)` **exact** MAP.

**MAP is exact on NFAs too.** [R] Proof sketch: for any accepting path `e_{1:L}`, the best string
consistent with it is `x_i = argmax_{v∈label(e_i)} p_i(v)`, which is itself an accepted string; so
`max over paths of ∏_i max_{v∈label(e_i)} p_i(v) = max over accepted strings of ∏_i p_i(x_i)`. Path
multiplicity cannot change a `max`. SPEC states this was verified to error `0.000e+00`. So §2.2's
soft-proxy caveat applies to **sampling only**.

**MAP is exactly temperature-invariant** [S, and [R] agrees]:
`argmax_x ∏_i softmax(ℓ_i/T)_{x_i} = argmax_x Σ_i ℓ_{i,x_i}` for any `T>0`.

**Sentinel:** impossible transitions get a finite `-3e38`, not `-inf`, so fused kernels do not
produce `NaN` from `-inf + -inf`.

**Token recovery needs a `(s,s') → class` table, not just a per-class table** [S]: once the tree
fixes `(s_i, s_{i+1})` you still need which class realized the max on that transition
(`argclass[i, pair]`, then `argmax_token[class, i]`). Deterministic tie-breaking is **lowest token
id, then lowest state id** — see bug signature **F15**.

---

## D. When is the constrained sample EXACT, and when is it an approximation?

This is the section reviewers will misuse. There are **five** distinct layers, and "exact" is true
at some and false at others.

### D.1 Exact with respect to WHAT — the target is per-step, not end-to-end

**[P]** The method samples exactly from eq (4), the **constrained mean-field posterior at a single
denoising step**. The paper is explicit that the ideal object `p_dLLM(x^0 | x^0 ∈ C)` is
**intractable** and that it deliberately substitutes the per-step restriction. **Therefore:**

- ✅ Exact: "the emitted canvas is a draw from `(∏_i p_i)·1[x∈C]`".
- ❌ Not claimed, not true: "the generated text is a draw from the model's generative distribution
  conditioned on `C`".

Appendix B, Proposition B.1 proves exactly the former ("Algorithm 1 returns a sample `x^0`
distributed exactly as `p(x^0|x^t,C) ∝ p(x^0|x^t)·1[x^0 ∈ L(C)]`"), and nothing more.

### D.2 DFA vs NFA — path-weighted vs uniform on `C`

**[P] paper §3.2, verbatim:** *"The value of `p_M(x_{1:L})` is proportional to the number of
accepting paths that generate `x_{1:L}`; if the finite automaton is deterministic, then each
accepted sequence has exactly one accepting path, thus `p_M` becomes uniform over `C`."*
**[P] §4.2:** *"For non-deterministic finite automata (NFAs) … `p_M` now weights each accepted
sequence by the number of accepting paths that generate it; sampling from this weighted
distribution becomes a **soft proxy** for `p_θ(x^0|x^t,C)`."*

| Automaton | Sampled distribution | Exact w.r.t. eq (4)? | Support = `C`? |
|---|---|---|---|
| DFA | `∝ (∏_i p_i(x_i)) · 1[x∈C]` | **Yes** | Yes |
| **Unambiguous** NFA | same | **Yes** [R] — see below | Yes |
| Ambiguous NFA | `∝ (∏_i p_i(x_i)) · #paths(x)` | **No** — soft proxy | **Yes** |
| any, MAP emission | `argmax_{x∈C} Σ log p_i(x_i)` | **Yes**, incl. NFA | Yes |

[R] **Refinement worth having in review:** determinism is *sufficient* but not *necessary*. The
exact condition for `p_M` to be uniform on `C` is **unambiguity** (every accepted string has
exactly one accepting path). A reviewer should not reject a non-deterministic automaton as
"inexact" without checking ambiguity — though in practice checking unambiguity is harder than
determinizing, and SPEC §4.6's eager-determinize-then-fall-back policy is the pragmatic answer.

**Constraint satisfaction is unaffected in every row.** The support of the product is exactly `C`
regardless of ambiguity, because path multiplicity is a strictly positive integer on `C` and zero
off it. This is why the §E guarantee holds even where the distribution is a proxy.

### D.3 Conditions on the *implementation* for the sample to be exact

All of these are necessary; each is a bug signature in §F:

1. **Edge-multiplicity weighting in eq (8a)** — otherwise not even path-weighted (F2).
2. **Joint root draw over `(s_0, s_L)`** — not two marginal draws (F8).
3. **Segment-local normalization at draw time**, not row-normalized transfer matrices (F9).
4. **Aligned dyadic node set** (Blelloch/Brent–Kung), because eq (7) needs `P_{[ℓ,m)}` and
   `P_{[m,r)}` for the *actual* midpoint of the *actual* segment (F11).
5. **Correct operand order in the suffix/backward scan** (F1).
6. **Log-space sum-product at `L=256`**, else the root product underflows and the categorical
   degenerates while still returning tokens (F12).
7. **`a_start` a general vector**, not a hardcoded point mass (F22).

### D.4 The budget-aware terminal factor changes the target — deliberately

[S] With `b_L(s) = 1[d(s) ≤ R]` rather than `1[s ∈ F]`, the per-block target is

```
∝ ( ∏_i p_i(x_i) ) · 1[ x is a viable prefix from A_k that can still reach F within R more tokens ]
```

The sampler is **exact with respect to that** target. It is **not** the same object as eq (4) with
`C = L(M) ∩ V^L`, and it should not be. A reviewer comparing block-local output against a
brute-force enumeration using `1[s∈F]` will see a spurious mismatch.

### D.5 Block-wise decoding: exact per block, and one NFA caveat

[S] Blocks thread the automaton: `A_0 = {s_0}`, and `A_{k+1} = δ*(A_k, truncated canvas_k)`, with
`a_start = 1[s ∈ A_k]`.

[R] **On a DFA this is exact** (`|A_k| = 1`, so the indicator *is* the correct forward message up
to scale, and scale cancels).

[R, flagged] **On an NFA, `a_start = 1[s ∈ A_k]` is an additional approximation.** After
`canvas_k` is committed, the correct forward message into block `k+1` is the *path-weight* vector
`a(s) = #(accepting-prefix paths through the observed tokens ending at s)`, not a 0/1 indicator.
The indicator flattens that to uniform-over-reachable. This does **not** affect support (hence not
the guarantee), only the weighting — and it is arguably *more* faithful to eq (4) than the
path-weighted carry would be, since eq (4) wants uniform-on-`C`. **I am flagging it as a knowing
divergence to be aware of, not as a bug.** SPEC §5.7 prescribes the indicator. **[?]** I did not
find the paper addressing cross-block message carry at all.

### D.6 The edge set is a SET OF PAIRS, not a multigraph — and this is load-bearing

Raised by `review-compile`; the paper settles it. **[P] §3.2, verbatim:**

```
E := {(s, s′) ∈ S × S | ∃v s.t. s′ ∈ δ(s, v)}.
For each edge e = (s, s′), we write src(e) := s, dst(e) := s′ and label(e) := {v | s′ ∈ δ(s, v)}.
```

`E` is a **subset of `S × S`**. An edge *is* a pair. There is exactly **one** edge per `(src,dst)`,
and `label(e)` is the **union** of every token taking `s` to `s'`. Figure 1 confirms it by example:
for `a(b|c)*de*` the edge `e1` is drawn with the single label **"b, c"** — one edge carrying a
two-token label set, not two parallel edges.

**[R] A multigraph reading is inconsistent with the paper's own theorem.** §3.2 states: *"if the
finite automaton is deterministic, then each accepted sequence has exactly one accepting path, thus
`p_M` becomes uniform over `C`."* Suppose edges could be duplicated: split `label = {a,b}` into
parallel edges `{a,b}` and `{b}`. This leaves `δ` untouched, so `|δ(s,v)| = 1` still holds and the
automaton is still deterministic by Definition 3.1 — yet the string `b` would now have two accepting
paths and `p_M` would no longer be uniform on `C`, contradicting the quoted claim. Hence the pair
reading is the only coherent one.

**The one sentence that reads otherwise, quoted so nobody thinks I hid it.** [P] Appendix A.2:
*"The joint transition factor aggregates contributions from all edges that traverse the same state
pair, naturally handling NFAs where multiple edges may connect `s_{t−1}` to `s_t`."* **[R] I rule
this loose prose**, for three reasons: §3.2's definition is formal and explicit; under it the
appendix's `Σ_{e : src=…, dst=…}` degenerates harmlessly to a single term; and the multigraph
reading contradicts §3.2's determinism theorem as shown above. Read "multiple edges" as "multiple
*transitions* `(s,v,s')`" — which is exactly what merging them into a label set handles.

**Consequences.**

1. `compile/automaton.py::_group_edges`, which collapses `(src, label, dst)` triples into one edge
   per `(src,dst)` with the union of labels, is **canonical and correct** — it implements the
   paper's `E` exactly. Not a defect.
2. **Nondeterminism is represented by distinct `(src,dst)` pairs**, not by parallel edges:
   `δ(s,v) = {s1,s2}` means `v ∈ label((s,s1))` *and* `v ∈ label((s,s2))`. Path-weighting on NFAs
   arises entirely from that, and the pair encoding captures it fully.
3. **F2 is vacuous under this encoding** (multiplicity ∈ {0,1}).
4. **An ungrouped multigraph is wrong independently of the token draw.** With parallel edges
   `{a,b}` and `{b,c}`, `M_i(s,s')` counts `p_i(b)` twice, so `p_M` is no longer proportional to the
   number of accepting *state paths* — it is not the paper's distribution at all. Multiplicity
   weighting would make the sampler *self-consistent with a wrong model*. **So the control is
   canonicalization at the boundary, not weighting downstream.**
5. **[?]** SPEC §2.6's measured "1.7e-2 deviation for the `∃` form on a 3-state NFA with parallel
   overlapping edges" is therefore measuring a property of a **non-canonical encoding**. The
   measurement is internally consistent (brute force over multigraph paths), but the encoding it
   uses does not correspond to the paper's `p_M`. SPEC should be annotated accordingly.

---

## E. The guarantee: what is actually guaranteed

**[P] claims:** "guarantees constraint satisfaction **by construction**", "Our method guarantees
100% constraint satisfaction by construction" (Table 1). The paper's mechanism is that every
`x̂^0` drawn at every denoising step lies in `C`, so the final output does too.

**[S] SPEC §3.1b's Proposition (port) — this is the statement to review against:**

> If `emit_canvas` is either the max-plus MAP (§2.7) **or** a joint draw (§2.6) from the
> constrained posterior, with `a_start = 1[s ∈ A_k]` and `b_L = 1[d(s) ≤ R]`, then for every
> block `k`: `δ*(A_k, canvas_k) ≠ ∅` and every state in it can still reach `F` within the
> remaining budget. Hence `canvas_0 ⋯ canvas_k` is a **viable prefix** of `L(M)`. It lies in
> `L(M)` **iff** generation terminates at a block boundary with `A_{k+1} ∩ F ≠ ∅`.

**The argument uses only the SUPPORT of the constrained posterior, not maximality** — which is why
it covers MAP and sampling identically, and why it survives the NFA path-weighting proxy.

### E.1 Per-block versus at-end — the known trap

```
per block k:  δ*(A_k, canvas_k) ≠ ∅   and   ⊆ {s : d(s) ≤ R}      # viable prefix
at end:       simulator.accepts(concat(canvas_0 .. canvas_K))      # this is CS
```

**A non-final block's canvas ends in a live-but-not-accepting state and is NOT accepted on its
own.** Asserting per-canvas acceptance turns `tests/test_guarantee.py` red on every multi-block
generation, and the "fix" is to weaken the test. **If a reviewer proposes weakening
`test_guarantee.py`, that is the bug, not the test.**

### E.2 Membership in `L(M)` is NOT implied by a constrained emission alone

[S] Four failure modes, **all** of which must be closed:

1. **Budget truncation** — block cap reached with `A_k ∩ F = ∅`. Note `Live = {s : d(s) < ∞}` is an
   *unbounded-horizon* predicate and does **not** close this.
2. **Early stopping** — stock early-stop fires on model uncertainty with zero automaton awareness.
3. **Stop token from a non-accepting state** — if the free-text region admits any `end_token`.
4. **[V-P0] Cache exhaustion** — `_sample_loop`'s `cond_fn` includes `~state.cache_info.is_full`;
   the cache filling halts generation mid-grammar and is *not* covered by `max_new_tokens − step`.

### E.3 The three closures of SPEC §3.1b

1. **Budget-aware terminal factor.** `b_L(s) = 1[d(s) ≤ R]` with `d(s) = min tokens from s to F`,
   a BFS on the **reversed** automaton. `d(s) ≤ 0 ⟺ s ∈ F`, so the final block falls out
   automatically — **do not special-case `1[s∈F]` on the last block**.
   - **`d` must be computed AFTER the stop-token augmentation (§3.5 trap 4) and AFTER the
     `FA_grammar | FA_refusal` union (§3.8)**, or it is a different function entirely.
   - **[V-P4] `R = max_new_tokens − state.step − canvas_length`** — `state.step` counts tokens
     committed *before* the block, but `b_L` is evaluated at the state reached *after* the block's
     `L` tokens. The unadjusted `R` is too permissive by exactly `L` and generation **never
     terminates**. Fixed in `model/state.py: terminal_budget`.
   - Also bound by cache. **[corrected, credit `review-model`]** *Both* terms take the `−L`, for
     the same reason: `b_L` is evaluated at the state reached **after** this block's `L` tokens, so
     both budgets must be measured after those `L` tokens are committed. With gemma's
     `used_cache_length = init_cache_length + step`:

     ```
     R = min( max_new_tokens − step − L ,  cache_length − used_cache_length − L )
     ```

     An earlier revision of this document (and SPEC §3.1b's own text, which applies its [V-P4]
     `−L` correction only to the token term) omitted the `−L` on the cache term. That is too
     permissive by `L` on the cache side — the same class of error as §3.5 trap 1.
     **Further [?]:** gemma's `is_full` is `end_index >= total_cache_length − 1`, so usable
     capacity is one *less* than `cache_length`; the fully tight form is
     `cache_length − 1 − used_cache_length − L`. Erring high here does not break the math, it
     re-opens §3.1b failure mode 4.
2. **Automaton-aware stopping.** Conjoin `A_{k+1} ∩ F ≠ ∅` into the block-level done flag.
   **Under J0 this must be fed `emit_canvas`, not the trajectory canvas** — the trajectory is mostly
   uniform random tokens, so `δ*(A_k, trajectory)` is `∅` on essentially every step and a naive
   conjunct is permanently false (the loop then never early-stops). Requires a widened
   `should_stop`; must compose as **AND** with the stock chain.
   - The block-level done flag is **distinct from the carry's monotone `done`**; never flip the
     carry's `done` back to False after a canvas has been truncated at a stop token.
3. **`FREE` must exclude every `end_token` and `PAD`**, not just the channel marker. Handle all of
   `(EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE, *stop_tokens)`.

**Constraining the sampler alone is necessary but not sufficient.** That sentence is the whole
point of §3.1b.

### E.4 The post-stop tail must be UNSCORED

[S, §3.5 trap 4] `ACC --Σ--> ACC`, **not** `ACC --PAD--> ACC`. A joint decode over all 256 positions
that must pay `(255−j)·log p(PAD)` to terminate at position `j` will place the stop token at 255 or
never. Since `_truncate_canvas_at_stop_tokens` overwrites everything after the first stop token
anyway, an accepting tail is semantically free — and it removes the `∏p(PAD)` bias from `Z` and
`q_i` too. **Diagnostic: stop-token position per block clustering at 255 means this is still open.**

---

## F. SUBTLE BUG SIGNATURES

Every one of these produces **valid-looking output** — usually a valid string in `C` — while being
mathematically wrong. Ordered roughly by how easy they are to miss.

**F1 — Reverse-scan operand order.** `lax.associative_scan(fn, x, reverse=True)` yields
`f(f(z,y),x)`, i.e. **reversed operand order**. For non-commutative matmul the suffix pass needs
`lambda a, b: b @ a`. Wrong order gives `M[L-1] @ … @ M[0]` instead of `M[k] @ … @ M[L-1]`.
*Signature:* no shape error; `b` is still a non-negative vector; sampling still yields strings in
`C` (support of `b` is wrong but usually still non-empty). *Detect:* assert
`b_{i−1} == M_i @ b_i` elementwise for all `i`, and assert `log(a_i·b_i) + logscale_a[i] +
logscale_b[i]` is **`i`-invariant**.

**F2 — `∃`-form token draw instead of edge-multiplicity.** eq (8a).
**⚠ SUBSTANTIALLY CORRECTED — see §D.6. Under the paper's edge set this signature is VACUOUS.**
The paper defines `E := {(s,s') ∈ S×S | ∃v. s' ∈ δ(s,v)}` — a set of **state pairs**, one edge per
pair, with `label(e) := {v | s' ∈ δ(s,v)}` the union of all tokens taking `s` to `s'`. So the
multiplicity `|{e : src=s_i, dst=s_{i+1}, v ∈ label(e)}|` is always **0 or 1**, the `∃` form and the
multiplicity form **coincide identically**, and F2 cannot occur. It becomes reachable only if an
implementation feeds the kernels an *ungrouped multigraph* — and such an encoding is already wrong
for a separate and more basic reason (§D.6): it does not represent the paper's `p_M` at all.
*Signature, when a multigraph does reach the kernels:* every emitted string is valid; the sampler is
self-consistent with its (wrong) encoding. *Detect:* the primary control is **not** a distributional
test — it is an assertion of **unit multiplicity at the compile/load boundary**. `compile/automaton.py
::_group_edges` establishes it by construction; assert it again where the arrays are consumed.
*Test-suite trap [measured by `review-tests`]:* duplicating an edge with an **identical** label set
scales `M` uniformly and cancels in normalization, so it produces **zero** deviation — a generator
that builds "parallel edges" that way has no power against F2 at all (the ∃-form substitution
survived a 721-test suite). A real F2 probe needs *overlapping but distinct* labels, e.g. `{a,b}`
and `{b,c}`. Also check `is_dfa` is *verified from the arrays*, not asserted by the compiler.

**F3 — `u_i(e)` multiplied by `W[i,e]` in eq (6).** Double-counts the emission mass. *Signature:*
`q_i` is still non-negative and looks like a distribution *after renormalization*, so if the code
renormalizes `q` the bug is invisible. *Detect:* assert `Σ_v q_i(v) == 1` **without renormalizing**.
Corollary: **any code path that renormalizes `q_i` has destroyed the only cheap test for eq (5)/(6)**.

**F4 — Off-by-one between `a` and `b`.** Using `a_i` instead of `a_{i−1}` in eq (5), or placing
`b_L` at index `L−1`. *Signature:* `Σ_v q_i(v) ≠ 1` — but only if you check; the marginals are
still positive on a valid-looking support. *Detect:* **brute-force comparison at `V=4, L≤8` — and
in this codebase that is the ONLY detector.** The normalization assertion is dead here: both
`infer/reference.py::marginals` and `infer/marginals.py::constrained_marginals` renormalize by the
row total before returning, so `Σ_v q_i(v) == 1.0` holds identically under F3, F4, and even an
injected 1e7 scale error [measured by `review-tests`]. See the note under F3.

**F5 — `b_L = 1[s ∈ F]` instead of `1[d(s) ≤ R]`.** *Signature:* every block is forced to complete
the grammar in exactly `L` tokens; on grammars that fit, everything looks perfect. On grammars that
do not, `Z = 0` (SPEC §6.3 cause (b)) or generation runs to the budget. *Detect:* a cross-block test
with a grammar that provably cannot complete in 256 tokens.

**F6 — `R` off by one canvas.** `max_new_tokens − step` instead of `− step − canvas_length`.
*Signature:* **too permissive** — admits states that cannot actually finish, and generation **never
terminates** because nothing forces completion as the budget runs down. Already caught once in this
repo ("did not terminate in 8 blocks"). *Detect:* multi-block `test_guarantee.py`.

**F7 — `d(s)` computed at the wrong point in the pipeline.** Before the stop-token augmentation, or
before the `FA_grammar | FA_refusal` union. *Signature:* a *different function* — some states get
`d = ∞` that should be finite (spurious `Z = 0` / empty state set) or vice versa (guarantee hole).
*Detect:* assert `d(s) ≤ 0 ⟺ s ∈ F` **on the final, augmented, unioned automaton**, and assert `d`
is recomputed whenever the automaton object changes.

**F8 — Root boundary pair drawn from two marginals instead of jointly.** *Signature:* **the output
is still in `C`** (any `(s_0, s_L)` pair with positive joint mass is reachable — though an
independent draw can even select a pair with **zero** joint mass, in which case the whole tree
degenerates). The distribution is wrong in every case where `a_start` is not a point mass — i.e.
block ≥ 1 and every NFA. *Detect:* SPEC §6.1 test 8 (non-point-mass start vectors); exactness at
block 2.

**F9 — Locally normalizing the factors.** Row-normalizing the transfer matrices to make them
"proper" conditionals, instead of normalizing the `|S|`-vector of unnormalized scores at draw time.
The paper warns about this twice (footnote 1; the sentence after eq (5)). *Signature:* a perfectly
well-formed HMM that samples from **a different distribution** — locally-normalized ≠ globally
normalized. Output always valid. *Detect:* brute-force `Z` comparison; the `i`-invariance assertion.

**F10 — Log-scale sum not taken over all `2L−1` tree nodes.** *Signature:* **samples are correct**
(scales cancel in eq (7) and in the root draw) but `Z` is wrong by a constant factor.
**⚠ CORRECTED — the downstream consequence I originally stated is FALSE for this codebase**
[credit `review-tests`]. A missing node makes `Z` wrong by a factor that is *constant across `v`*,
so `q_i ∝ p_i·r_i/Z` is wrong by a uniform scale — which the renormalization in both `marginals`
implementations divides out **exactly**. `q` comes back bit-identical, so the **Mar** signal, the
entropy rule and the remasking order are all **unaffected**. F10 therefore degrades from "silent
distributional corruption" to "a wrong diagnostic quantity": it corrupts reported `log Z`, the
`i`-invariance assertion, and anything using `Z` as an absolute (e.g. a threshold-based `Z ≈ 0`
test), but not the sample and not the confidence signal. *Detect:* `log Z` against enumeration —
**the only remaining detector**, since `Σ_v q_i(v) == 1` has zero power here (F3/F4 note).

**F11 — Kogge–Stone / Hillis–Steele scan shape instead of Blelloch.** *Signature:* the prefix
products are *correct*, so `a` and `b` are right and everything looks fine — but the retained node
set is not the aligned dyadic set, so if eq (7) reads segment products off those levels it gets
`P_{[ℓ',m')}` for the wrong interval. Also 3.6× the work and 4× the memory, which silently
invalidates the dispatch threshold. *Detect:* assert each retained node equals the hand-computed
`M_ℓ ⋯ M_{r−1}` for its claimed `[ℓ,r)`.

**F12 — Root product underflow in the sum-product sampling tree.** fp32 (and, at `L=256`, even
float64) underflows the root joint to exactly zero. *Signature:* `jax.random.categorical` on an
all-`-inf`/all-zero row still **returns tokens** — degenerate, effectively uniform. Observed
end-to-end as "no stop token, pure multilingual garbage" on 2 of 6 real prompts, while
`--emission=map` was 10/10. *Detect:* `require_x64` guard plus log-space combines
(`infer/scans.up_sweep_log`); pinned by
`tests/test_constrained_draw.py::test_root_product_underflows_in_fp32_on_a_real_grammar`.
This is SPEC §6.3's `Z == 0` cause **(c)** — scaling present but insufficient — and must **not** be
conflated with cause (a) (empty automaton, a real bug) or (b) (no live continuation within budget).

**F13 — `-inf` instead of a finite sentinel.** Masked logits set to `-inf`. *Signature:*
`softmax → @ embed_tokens.weight` produces `-inf`/`NaN` in the **self-conditioning** embedding, and
`-Σ q log q` is `NaN`. The `NaN` may not surface until it corrupts the acceptance mask several
steps later. *Detect:* assert no non-finite values anywhere; use `-1e30` (sum-product) / `-3e38`
(max-plus).

**F14 — `log q` without the clamp.** `q_i` has **exact zeros** by construction. Naive
`-Σ q log q` → `NaN`. *Detect:* `jnp.log(jnp.maximum(q, 1e-30))` at **every** use site (§3.4, §3.7,
§5.2), and a test with a `q` containing exact zeros.

**F15 — Tie-break inverted in the MAP argmax.** The two-pass
`segment_max(where(vals==mx[seg], idx, -1))` workaround resolves ties to the **largest** index; the
convention is **lowest token id, then lowest state id**. *Signature:* MAP output differs from the
reference only on exact ties — rare, deterministic, and it breaks both `test_guarantee` and the
unconstrained-equivalence test in a way that looks like flakiness. *Detect:* use the `segment_min`
variant with an `N` sentinel; test with deliberate ties.

**F16 — Empty-segment sentinel tested as `-1`.** `segment_max`'s identity is `-inf`, **not** `-1`,
and the `segment_min` argmax of an empty segment is `idx.shape[0]`, not `-1`. *Signature:* an
empty segment silently selects token index `-1` → the **last token in the vocabulary**, a valid
token id. *Detect:* test `am == N`, never `am == -1`.

**F17 — Complement-aware `r_i(v)` missing the global Neg term.** Under §4.4's class representation,
negated classes contribute `U_i(c)` at **every** `v`. Dropping
`Σ_{c ∈ Neg} U_i(c)` (or sharing the polarity flag array between the sum and max tables, which have
**independent** Pos/Neg partitions) yields a plausible, positive, wrong `q`. *Signature:* nothing
visible; strings stay valid; only the marginals are wrong. *Detect:* SPEC §6.1 test 11 — force
negated classes at `V=4` with a polarity threshold of `|S_c| > 2`, mixed polarity in one automaton,
empty `N_c` and empty `S_c`; compare against a direct scatter **and** against enumeration.

**F18 — `K` too small in the top-`K` max trick.** `max_{v∈S_c} p_i(v)` for a negated class is "the
first top-`K` entry not in `N_c`", which requires `K ≥ |N_c| + 1`; the bound is **tight** (an
adversarial `p` exists where `K = |N_c|` fails). Measured real grammars have `|N_c|` at 988–1,064,
so `K_max = 256` is wrong; use `K_max ≈ 1,100` set from the compiled grammar's measured
`max_c |N_c|`. *Signature:* silently returns the max over the *wrong* token set — a valid token,
slightly lower probability. *Detect:* build-time assert `K > max_c |N_c|`; test `K = |N_c|` fails
and `K = |N_c|+1` succeeds.

**F19 — Reweighting transitions instead of emissions.** eq (5) reweights the **emission** factor
only. *Signature:* a well-formed chain model whose distribution is not the product of `p_θ` and
`p_M`. Since `M_i` already contains `W`, an implementation that *additionally* scales the transition
factor by anything model-derived is double-counting. *Detect:* brute-force `Z`.

**F20 — Constraint mask installed in `sample_from_predictions` instead of `logit_shaper`.**
*Signature:* the output is still constrained and still valid — but the mask never reaches
self-conditioning (the SC tap is strictly upstream of the sampler) and the entropy acceptance rule
is computed on the **unconstrained** distribution. You lose the paper's **Mar** signal, which is the
larger half of its accuracy gain (Table 2: Dream xLAM-Python Mf 68.4 vs Mar 76.4). Nothing fails.
*Detect:* assert the tensor consumed by `embedder.encode_logits` contains the sentinel.

**F21 — Post-stop tail scored as PAD.** `ACC --PAD--> ACC` rather than `ACC --Σ--> ACC`.
*Signature:* stop token placed at position 255 or never; the model blows through blocks and lands
in the truncation failure mode. Also biases `Z` and `q_i` by `∏ p(PAD)`. *Detect:* log stop-token
position per block; **clustering at 255 is the tell**.

**F22 — `a_start` hardcoded as a point mass `1[s = s_0]`.** *Signature:* correct on block 0 of a
DFA — i.e. every single-block DFA test passes. First fails on **block 2**, or on any NFA, where the
start is a set `A_k`. Either the sample is drawn from the wrong distribution or `Z = 0`. *Detect:*
SPEC §6.1 test 8; a ≥3-canvas cross-block test.

**F23 — Identity padding with `b_L` at the padded length.** For non-power-of-two `L`, pad with
`M_i = I` but place `b_L` at the **true** `L`. Putting it at the padded `L` lets the automaton keep
moving through the padding. *Signature:* accepts strings that should be rejected at the true length.
(`canvas_length = 256` always, so this is latent until someone runs a shorter canvas.) *Detect:*
SPEC §6.1 test 7.

**F24 — Per-request automaton values landing on a static argument.** `self` is a `static_argname`
on both `_sample_loop` and `_sample_step`; anything stored as an attribute of a custom
`sample_from_predictions` / `logit_shaper` is **baked into the trace**. *Signature:* **numerically
perfect output** and a full XLA recompile per schema (~2 h across BFCL). Invisible except by timing.
*Detect:* assert the jit cache size is constant across 50 different automata in the same `|S|`
bucket.

**F25 — Asserting per-canvas acceptance in `test_guarantee.py`.** Not a code bug but a *review*
bug, and the one CLAUDE.md singles out. A non-final block's canvas ends live-but-not-accepting and
is correctly **not** accepted alone. The two assertions (per-block viable prefix; at-end acceptance
of the concatenation) are different statements. **Weakening this test to make something pass is the
failure mode.**

**F26 — Confusing SPEC eq numbers with paper eq numbers.** See §A.1. A comment reading
`# eq (3)` means the forward recursion in SPEC's scheme and the graphical model in the paper's.
Reviewers citing "eq (5)" at each other without saying whose will not converge.

**F27 — A feasibility flag with no detection power.** [measured by `review-infer`] A per-position
"is there an admissible token" check (`all(mult.sum(-1) > 0)`, i.e. eq (8a) multiplicity non-empty)
does **not** detect `Z == 0`. Once the root draw degenerates, the state path is arbitrary, and on a
well-connected automaton an arbitrary path still traverses existing transitions at every step —
measured **valid = True on 200/200 draws from a genuinely empty language, in both float32 and
float64, with 0/200 outputs accepted**. Two distinct mechanisms both produce a confident
`valid = True`: an all-sentinel root (empty language / no length-`L` path) yields an arbitrary but
realizable path; an all-sentinel `b_L` is **shift-invariant** and yields a genuinely *confident*
draw from the budget-unaware posterior. *Detect:* test the root joint **after** both boundary
vectors are applied — `joint = log_a[:,None] + root + log_b[None,:]`, then `joint.max() > threshold`.
That single predicate covers all three of: empty `A_k`, infeasible budget, and no length-`L` path,
i.e. `Z == 0` causes (a) and (b) together. Note three summed `-3e38` sentinels overflow **float32**
to `-inf`.

**F28 — A numerical guard that is inert in the working dtype.** [measured by `review-infer`]
`jnp.maximum(row.sum(), 1e-300)` is a no-op in **float32** because `1e-300` flushes to `0.0` — so
the guard silently does nothing in precisely the dtype that needs it, `q` goes all-`NaN`, and the
`NaN` propagates through `entropy_from_q` into the acceptance mask (§3.4). A float64-correct,
float32-inert clamp is invisible to any test that runs only in x64. *Detect:* derive clamp constants
from `jnp.finfo(dtype).tiny` rather than hardcoded literals, and audit every hardcoded epsilon
against the dtypes it actually runs in (`1e-30` is fine in float32; `1e-300` is not).

---

## G. Things I could not establish (do not treat me as authority here)

1. **[?]** The paper gives **no MAP/greedy algorithm**. SPEC's max-plus eq (9) is this port's own
   design. My argument that it equals the `T→0` limit is [R], and I believe it is sound, but it is
   not the paper's.
2. **[?]** The paper displays **no formula for the constrained marginal `q_i`** — only the statement
   that the constrained marginal is the right remasking confidence. SPEC eq (5)/(6) is the standard
   HMM marginal and is correct [R], but is not a quotation.
3. **[?]** The paper does **not** discuss cross-block automaton message carry (`A_k`), the
   budget-aware terminal factor, cache exhaustion, or the `|S|` compute/memory crossover. All of
   §3.1b, §3.5 and §5.6 are this port's own work.
4. **[?]** I did not verify the paper's Figure 3 / Table 3 numbers beyond the text; the "+4%
   wall-clock, tree variant, 200 BFCL examples (Dream)" figure is quoted from §5.3.
5. **[?]** I read the paper's text layer only. A transcription error in a rarely-used symbol is
   possible, though every equation cross-checks against SPEC's independent restatement.

---

## H. The paper's evaluation setup (added for the eval audit)

Everything here is [P] unless marked. Added in response to `review-eval`'s comparability questions.

### H.1 Metric and n

**[P] §5.1, verbatim:** for xLAM/BFCL the *Metric* is **"Exact match against the ground-truth
function call."** Data is "xlam-function-calling-60k (xLAM) and the **single-turn split** of the
Berkeley Function Calling Leaderboard (BFCL), including both non-live and live settings."

**[?] Tension to be aware of:** Table 9's caption instead says outputs "are still **rejected by the
official scorer** because some argument values differ from the ground truth". So §5.1 says exact
match and the appendix says official scorer. The most likely reading is BFCL's official **AST**
checker (whole call: function name + all args must match), which is close to exact match on the
parsed call. **The paper never states "AST accuracy" or "exec accuracy" in those words.** Do not
report the paper as using BFCL's official AST accuracy without this caveat.

**n, from Table 7's "number of automata N used during evaluation":**

| Split | N |
|---|---|
| BFCL Live (JSON) / Live (Python) | **1,351** |
| BFCL Non-Live (JSON) / (Python) | 1,000 |
| xLAM (JSON) / (Python) | 1,000 |
| Sudoku | 900 |
| Countdown | 993 |
| GSM-Symbolic | 1 automaton (instance count not stated) |
| Spider | 20 automata (one per database) |

**Sub-splits:** the main Table 1 uses Live / Non-Live, **not** `simple`. `live_simple` appears only
in Figure 4 ("BFCL Simple split", the denoising-step sweep) and Table 8 (qualitative examples).
Table 3's runtime numbers use a **200-example subset** of BFCL, Dream, batch size 1, A6000,
mean±std over **3 seeds**.

### H.2 Does the paper claim an accuracy gain? Yes — and it is large

**[P]** Table 1 reports accuracy, not just validity, and the constrained method wins in **39 of 40
cells**. The single exception is **GSM-Symbolic, LLaDA-8B, greedy: base 49.6 → ours 49.2** (−0.4).
Spider gains are near-noise under greedy (Dream 52.7 → 53.0; LLaDA 53.2 → 53.6) but large under
sampling (15.5 → 52.1). Largest gain: **BFCL-Live Python, Dream, sampling: 2.8 → 67.8**.

Headline greedy/sampling for BFCL-Live JSON on Dream: **63.9 → 71.5** and **22.3 → 69.0**.

**There is no task/model/regime in the paper where constrained decoding costs accuracy by more
than 0.4 points.** A large constrained-worse-than-unconstrained regression is *not* an expected
cost of the method and contradicts the paper's reported direction.

### H.3 Constraint satisfaction (CS) — how it is measured is NOT stated

**[P]** Table 1's caption says only: *"CS is the baseline constraint satisfaction rate; our method
guarantees 100% constraint satisfaction by construction."* **There is no methods sentence defining
how CS is computed.** [?]

**[R] What can be inferred from Table 8's four counted violations** (Dream-Instruct, T=0,
`live_simple`), which are the only evidence available:

| Case | Baseline output | Why it violates |
|---|---|---|
| (a) | prose, no tool call at all | "Invalid format: not a JSON tool-call list" |
| (b) | **well-formed JSON**, `"PerPage"` instead of `"perPage"` | wrong argument-name casing → missing required arg |
| (c) | `[todo.complete(...)]` | function name not in the tool list |
| (d) | `Movies_3_FindMovies(ddirected_by=...)` | unexpected parameter name |

So CS is **conformance to the schema FA** — format *and* function name *and* argument names/types —
**not** mere JSON-parseability. Case (b) is valid JSON and still counts as a violation.

[R] This is most consistent with **feeding the whole emitted string to the schema checker/automaton**,
not with extracting a JSON substring: under substring extraction, (a)'s prose would yield "no match"
rather than a violation, and the distinction the paper draws between (a) and (b) would collapse.
**I am inferring this; the paper does not say it.**

**[R] Comparability trap specific to this port — flagging unprompted.** Dream and LLaDA emit the
tool call as the entire response (their prompts say "You SHOULD NOT include any other text in the
response"). DiffusionGemma emits a **channel header** `[100, 45518, 107, 101]` before anything else
(SPEC §3.6). If you measure the unconstrained baseline's CS by feeding the whole emitted string to
`FA_grammar`, you will get **CS ≈ 0** for reasons that have nothing to do with the model's function-
calling ability, and that number is **not comparable** to the paper's 86.7 / 36.0. Whatever
normalization is applied (strip the header, or match against `HEADER · FA_grammar · STOP · Σ*`) must
be applied identically to constrained and unconstrained, and stated in `docs/RESULTS.md`.

### H.4 Prompts

**[P]** Appendix C.2, Table 6 gives one prompt template per task, with in-context examples omitted
for space. The system prompt **contains the function schemas** ("Here is a list of functions in JSON
format that you can invoke: …") and mandates the output format (Python: `[func_name1(params…)]`;
JSON: `[{"name": …, "arguments": {…}}]`).

**[P]** Table 5: "We follow the default decoding configurations released with each model. Generation
length and number of denoising steps are **matched across models within each task** to ensure a fair
comparison."

**[R] No prompt or schema-in-prompt difference between constrained and unconstrained is mentioned
anywhere.** Table 1 presents Base and Ours as the same task, and the schema is in the prompt for
*both* (the constrained run additionally has it in the FA). The natural reading is an identical
prompt. The paper does not state this explicitly, so it is an inference — but there is no evidence
for any difference.

### H.5 Entropy bound / threshold: no sweep, and no analogue in the paper

**[P]** There is **no** hyperparameter sweep or sensitivity analysis for any confidence or entropy
threshold, and **no claim of insensitivity**. Searching the full text for
`threshold|sensitiv|hyperparam|sweep|tuned|tuning` returns exactly one hit: Table 5's "We follow the
default decoding configurations released with each model."

The only sweeps in the paper are **Figure 4** (denoising steps 256→16, BFCL Simple split) and
**Figure 5** (Sudoku prefilled cells 12→4).

**[R] The paper has no analogue of SPEC §3.4's `entropy_bound` because the remasking schemes
differ structurally.** Dream selects the *lowest-entropy positions* and LLaDA the
*highest-probability positions* — a **ranking** that commits a fixed number of positions per step.
DiffusionGemma's `SampleFromPredictions` instead accepts the largest ascending-entropy prefix under
an absolute bound `Σ_{i≤k} H_i − max_{i≤k} H_i ≤ 0.1` nats, so the accept-set *size* is
data-dependent and collapses when `q_i`'s sharper constrained entropies are substituted. **SPEC
§3.4's recalibration requirement is real and is this port's own problem — the paper cannot be cited
for or against it.**

### H.6 Generation configuration (Table 5, verbatim)

All runs use **both** `T = 0` and `T = 1`. "The *Remasking* column reports the algorithm for both
unconstrained and constrained decoding; **for constrained runs, confidence values are recomputed
from the constrained distribution**" — i.e. the headline Table 1 uses **Mar**; Table 2 ablates
Mf vs Mar.

| Task | Model | Seq len | Steps | Block size | Remasking |
|---|---|---|---|---|---|
| BFCL (Python/JSON) | LLaDA-8B-Instruct | 256 | 128 | 32 | low_confidence |
| BFCL (Python/JSON) | Dream-v0-Instruct-7B | 256 | 128 | **—** | entropy |
| xLAM (Python/JSON) | LLaDA / Dream Instruct | 128 | 64 | 32 / — | low_confidence / entropy |
| Sudoku 4×4 | LLaDA / Dream **Base** | 32 | 32 | 32 / — | " |
| Countdown (cd3) | LLaDA / Dream **Base** | 32 | 32 | 32 / — | " |
| GSM-Symbolic | LLaDA / Dream Instruct | 128 | 64 | 32 / — | " |
| Spider | LLaDA / Dream Instruct | 128 | 64 | 32 / — | " |

**Points a reimplementation gets wrong:**

- **Dream has no block size at all** (— in the table): its BFCL runs are a **single 256-token
  block**. Only LLaDA exercises block-wise threading, at **block 32**. So the paper's cross-block
  automaton carry is lightly tested, and none of SPEC §3.5's block traps have a paper analogue.
- **128 denoising steps for 256 tokens** (2 tokens/step). DiffusionGemma runs **48 steps per
  256-token canvas** (~5.3 tokens/step) — a *much* more aggressive parallel schedule. Figure 4 shows
  the unconstrained baseline degrading sharply as steps drop, so the DiffusionGemma baseline sits
  where the paper's baseline is *weakest*, and the constrained method's *relative* gain should be
  larger, not smaller.
- **Greedy is `T = 0`**, i.e. the constrained MAP limit (SPEC §2.7 / `_MIN_TEMP = 1e-12`), not
  temperature-1 sampling with argmax emission.
- **Sudoku and Countdown use the Base variants**, everything else Instruct.
- **Truncation / stop-token handling is never described.** [?] No analogue of
  `_truncate_canvas_at_stop_tokens`, and no discussion of the unscored post-stop tail (SPEC §3.5
  trap 4). This port is on its own there.
- **Automaton sizes (Table 7)** for BFCL-Live JSON: median 385 states, P95 997, max 2,459 — the
  regime where SPEC §0 puts the tree at ~2.9% of a model forward. The "+4% wall-clock" headline is a
  **median-|S|≈385** result and does not transfer to larger automata.
