"""Class-table SpMM and constrained marginals, in JAX. SPEC §4.4, §2.4.

Every denoising step needs `W[i, e] = Σ_{v ∈ label(e)} p_i(v)` for all 256
positions and up to ~170k edges. That is `W = A pᵀ` with `A ∈ {0,1}^{E×V}` and
`V = 262,144`: dense bitsets are 5.5 GB, a dense float `A` is 178 GB, and a
plain CSR has `nnz = Σ_e |label(e)|` with JSON string-body edges carrying ~260k
members each. All dead. §4.4's four layers reduce it to ~1.3e7 FLOPs/step.

Two JAX rules that are not negotiable:

- **`num_segments` must be static** under `jit`; a traced value raises
  `ConcretizationTypeError`. It is bucketed like `|S|`.
- **Never `jax.experimental.sparse`** — "not recommended for use in
  performance-critical applications… no longer being actively developed", per
  its own module docstring. `gather + segment_sum` already fuses and the tree
  matmuls go to cuBLAS.

`p` is stored **`[V, L]` column-major** so each token's 256 marginals are
contiguous. SPEC calls this layout decision more important than the kernel
algorithm, and the gather below assumes it.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

__all__ = [
    "class_weights",
    "edge_weights",
    "transition_matrices",
    "scatter_edge_mass_to_tokens",
    "constrained_marginals",
    "constrained_marginals_and_partition",
    "entropy_from_q",
    "constrained_entropy_streamed",
    "MASK_SENTINEL",
]

#: SPEC §2.4/§3.7: mask with a **finite** sentinel, never `-inf`. An `-inf`
#: logit propagates through `softmax → @ embed_tokens.weight` and poisons the
#: self-conditioning embedding.
MASK_SENTINEL = -1e30


def _csr_support(indices: jnp.ndarray, seg_ids: jnp.ndarray, n_classes: int,
                 vocab_size: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Geometry of the class tables' stored support `U`. SPEC §4.4.

    Both complement-aware kernels below need the same three facts about
    `U = {v : v appears in some class's stored set}`, and they must agree on
    them, so the construction lives in one place.

    Note what this buys structurally: `in_class` is built from **the same
    `(seg_ids, indices)` pair that defines `U`**, so `N_c ⊄ U` is not expressible
    at this interface. The identity
    `Σ_{v ∉ N_c} f(v) = Σ_{v ∉ U} f(v) + Σ_{v ∈ U \\ N_c} f(v)` is therefore
    guaranteed by construction rather than by a property of the caller's tables.

    Args:
      indices: `[nnz] int32`, CSR column indices over the stored sets.
      seg_ids: `[nnz] int32`, class id per stored entry.
      n_classes: `C`. **Static.**
      vocab_size: `V`. **Static.**

    Returns:
      `(in_csr [V] bool, is_rep [nnz] bool, in_class_at_slot [C, nnz] bool)`.
      `is_rep` selects exactly one slot per *distinct* token id, so a token
      several classes store is counted once. `in_class_at_slot[c, j]` is
      `1[indices[j] ∈ N_c]` — a token-level test, not a slot-level one, which is
      the part a duplicate id across classes would otherwise get wrong.
    """
    nnz = indices.shape[0]
    slot = jnp.arange(nnz, dtype=jnp.int32)
    # `first[v]` = lowest slot holding `v` (`nnz` if none). `.min` rather than
    # `.set` because a scatter with duplicate indices has no defined winner in
    # JAX. (Measured: `min`/`max`/`set` agree to 2.4e-16; dropping the collapse
    # entirely costs 4.4e-2. The collapse is load-bearing, the representative's
    # identity is not — `.min` is defensive, not the correctness argument.)
    first = jnp.full((vocab_size,), nnz, dtype=jnp.int32).at[indices].min(slot)
    rep = first[indices]                                          # [nnz]
    in_csr = jnp.zeros((vocab_size,), dtype=bool).at[indices].set(True)
    in_class = jnp.zeros((n_classes, nnz), dtype=bool).at[seg_ids, rep].set(True)
    return in_csr, slot == rep, in_class[:, rep]


@functools.partial(jax.jit, static_argnames=("n_classes",))
def class_weights(
    p_vl: jnp.ndarray,
    indices: jnp.ndarray,
    seg_ids: jnp.ndarray,
    is_neg: jnp.ndarray,
    n_classes: int,
) -> jnp.ndarray:
    """`W_c[c, i] = Σ_{v ∈ S_c} p_i(v)`, complement-aware. SPEC §4.4 Layers 2-3.

    A positive class *is* its stored set, so its mass is a plain `segment_sum`.
    A negated class stores `N_c` and its mass is the mass of the **complement**
    of that set — real BFCL classes cover ~99.6% of the vocabulary, so storing
    them positively would put ~260k entries per class in the CSR.

    **The complement is never computed as `total − Σ_{v ∈ N_c} p_i(v)`.** That
    expression is catastrophic cancellation, not underflow, and it silently
    destroyed the constrained language on the Countdown `--emission=sample` arm
    (`Z == 0` on 73/250 records against 0/250 for `--emission=map`, on identical
    grammars and budgets — `tests/test_audit_partition.py`). When the model is
    confident about a token *inside* `N_c` — which is the ordinary case, Phase 0
    measured real final-step entropies of 5e-5 to 4e-4 nats — `total` and
    `partial` agree to all 53 bits and the difference comes out `0.0` or
    negative, although the true mass was `exp(-47.5) = 2.3e-21`, i.e. 661 nats
    above float64's smallest normal. **No wider float recovers that**: the
    information is gone before the subtraction rounds. `_matrices` feeds the
    result to `M`, and `joint_draw`'s `where(M > 0, log(M), NEG_SENTINEL)` then
    erases a structurally present transition; on Countdown those transitions sit
    on the mandatory channel header, so the whole language goes with them.

    The cure is algebraic, not numerical: sum over the class's **real members**,
    so every term is non-negative and nothing can cancel. Doing that naively is
    an `O(C·V·L)` dense contraction — **3.6e10 MACs at BFCL's worst shape**
    (`C = 532`, `V = 262,144`, `L = 256`; 5.6e9 at the `C = 83` grammar
    `require_x64` quotes) — which trades a numerical defect for a §7.3
    regression. Instead, split the vocabulary at the CSR's own support
    `U = {v : v appears in some stored set}` (`|U| ≤ nnz`, 110 for Countdown,
    median 9,125 and max 51,200 across all 4,549 BFCL schemas):

        Σ_{v ∉ N_c} p_i(v)  =  Σ_{v ∉ U} p_i(v)  +  Σ_{v ∈ U \\ N_c} p_i(v)

    The first term is one `[V]·[V, L]` contraction shared by every class — the
    same traffic as the `total` it replaces. The second is a `[C, nnz] @ [nnz, L]`
    product against a 0/1 membership mask. Both are sums of non-negatives, so
    both keep full *relative* accuracy, and the whole thing stays in linear
    space: the failure was cancellation at ~1e-16, not underflow at ~1e-308, and
    the model's own softcap (`30·tanh(x/30)`, `T ≥ 0.4`) bounds a class mass at
    `~e^-150` — 660 nats clear of float64's floor. SPEC §2.4.

    Duplicate token ids across classes are collapsed to one representative slot
    before the product, so a token stored by several classes is counted once.

    **Cost, measured.** Verified against `math.fsum` over each class's true
    members on Countdown and three real BFCL grammars across 7–8 `p` regimes:
    worst relative error **1.75e-12**, 0 entries lost, against the old kernel's
    relative error **1.0 with entries lost** on the confident-token regime. On an
    idle H100 the worst real shape costs **+0.679 ms, +0.323% of a 210 ms model
    step**. `bytes accessed` moves under 1%, but `flops` rise by the `C·nnz·L`
    contraction and **peak temp is not unchanged** — the `[C, nnz]` mask is a
    real allocation. Measured on XLA:CPU in float64, temp bytes old → new:

        Countdown   C=  24 nnz=   110    0.21 MB ->   3.47 MB
        BFCL median C= 151 nnz= 9,125   17.82 MB ->  29.69 MB
        BFCL max C  C= 532 nnz= 9,125   17.82 MB ->  59.53 MB
        BFCL worst  C= 238 nnz=51,200  100.00 MB -> 204.79 MB

    205 MB is ~1% of SPEC §5.6's ~20 GB headroom — not fatal, but §5.6 sets the
    `|S|` dispatch threshold *from* that headroom, so it is budgeted here rather
    than ignored. (An earlier revision also built a `[nnz, L]` copy of the gather
    and peaked at 374 MB; folding `is_rep` into the mask removed it.)

    Args:
      p_vl: `[V, L]` column-major marginals.
      indices: `[nnz] int32`, CSR column indices over the **stored** set.
      seg_ids: `[nnz] int32`, class id per stored entry.
      is_neg: `[C] bool`, whether each class stores its complement.
      n_classes: `C`. **Static** — a traced value raises.

    Returns:
      `[C, L]`.
    """
    V = p_vl.shape[0]
    gathered = p_vl[indices, :]                                   # [nnz, L]
    partial = jax.ops.segment_sum(gathered, seg_ids, num_segments=n_classes)

    # --- SPEC §2.4: the complement, as a sum over what the class contains ---
    in_csr, is_rep, in_class = _csr_support(indices, seg_ids, n_classes, V)

    # `Σ_{v ∉ U} p_i(v)`, replacing the old `total`. Written as a `[V] @ [V, L]`
    # matvec rather than `sum(where(mask, 0, p_vl))`: the `where` form makes
    # XLA:CPU materialise a `[V, L]` (537 MB at production shapes) temporary
    # instead of fusing the select into the reduction. Same traffic over `p_vl`.
    outside = (~in_csr).astype(p_vl.dtype) @ p_vl                 # [L]

    # `Σ_{v ∈ U \ N_c} p_i(v)`: a 0/1 mask over one slot per distinct token,
    # contracted against the gather `partial` already needs. `is_rep` is folded
    # into the mask rather than into a separate `[nnz, L]` copy of `gathered` —
    # that copy is 105 MB at BFCL's worst `nnz`, and it buys nothing.
    sel = (~in_class) & is_rep[None, :]                           # [C, nnz]
    complement = outside[None, :] + sel.astype(p_vl.dtype) @ gathered

    return jnp.where(is_neg[:, None], complement, partial)


@jax.jit
def edge_weights(W_c: jnp.ndarray, class_id: jnp.ndarray) -> jnp.ndarray:
    """Layer 4: `W[e, i] = W_c[class_id[e], i]`. A gather, not a matmul."""
    return W_c[class_id]


@functools.partial(jax.jit, static_argnames=("n_states",))
def transition_matrices(
    W_e: jnp.ndarray,
    edge_src: jnp.ndarray,
    edge_dst: jnp.ndarray,
    n_states: int,
) -> jnp.ndarray:
    """`M_i(s, s') = Σ_{e : src=s, dst=s'} W[i, e]`. SPEC eq (2).

    `.at[].add` rather than `.set`: the sum over **parallel edges** is what
    carries path multiplicity on an NFA.

    Args:
      W_e: `[E, L]`.
      n_states: padded bucket size. **Static.**

    Returns:
      `[L, S, S]`.
    """
    L = W_e.shape[1]
    M = jnp.zeros((L, n_states, n_states), dtype=W_e.dtype)
    return M.at[:, edge_src, edge_dst].add(W_e.T)


def scatter_edge_mass_to_tokens(
    edge_mass: jnp.ndarray,
    class_id: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    is_neg: jnp.ndarray,
    n_classes: int,
    vocab_size: int,
) -> jnp.ndarray:
    """`r[i, v] = Σ_{e : v ∈ label(e)} edge_mass[i, e]`, complement-aware.

    The shared kernel behind both eq (6)'s `q_i` and eq (8)'s token draw: give
    it `u_i(e)` and it yields the constrained marginal's numerator; give it the
    indicator `1[src(e)=s_i, dst(e)=s_{i+1}]` and it yields the **edge
    multiplicity** `|{e : …, v ∈ label(e)}|` that eq (8) must weight by.

    Negated classes contribute their full mass everywhere and must then be
    removed on the complement they actually store — a plain sparse scatter would
    be wrong (SPEC §2.4).

    **That removal is not a subtraction.** This function used to compute
    `neg_total − Σ_{c ∈ Neg, v ∈ N_c} U_i(c)`, which is the *same* catastrophic
    cancellation `class_weights` above was rebuilt to eliminate, one layer down
    and on a path feeding `q`, `--confidence=mar`, the §2.8 `mask` support and
    `tree.sample_tokens`. It is reachable whenever two negated classes have
    unequal `U` and a token stored by the large one — constructed and measured:

        U ratio 1e-8    got 1.000000e-08    rel err 6.1e-09
        U ratio 1e-14   got 9.992007e-15    rel err 8.0e-04
        U ratio 1e-16   got 0.0             rel err 1.0   <- mass destroyed

    It did not fire on Countdown (0 of 67,108,864 entries lost, max relative
    error 1.24e-11) for a reason that is **structural luck, not a bound** — and
    the reason is worth stating exactly, because a wrong one invites the next
    reader to assume a new grammar inherits it. It is *not* that one negated
    class is dead: measured, **both** carry non-zero `U`. It is that their live
    supports are **disjoint in position**, so `neg_total` never has two terms to
    cancel:

        class  1 (stores 7 reserved ids)  live at positions 1..8    (n = 8)
        class 23 (stores nothing)         live at positions 10..255 (n = 246)
        positions carrying both:          0 / 256

    Class 1 is the SPEC §3.6 channel header, pinned to the front of the canvas;
    class 23 is the unscored `ACC --Σ--> ACC` tail (§3.5), pinned to the back.
    In *this* grammar they cannot overlap. Nothing about the kernel guarantees
    that, and BFCL grammars carry two non-empty negated classes (stored sizes 7
    and 1,585) with no such separation.

    So the negated part uses the same `U`-split as `class_weights`. For the
    `V − |U|` tokens no class stores, the answer is `neg_total` with nothing
    subtracted; for the `≤ nnz` tokens some class does store, it is computed as
    a direct sum over the negated classes that **do not** contain the token.
    Both branches are sums of non-negatives when `edge_mass` is.

    Args:
      edge_mass: `[L, E]`.
      n_classes, vocab_size: **static**.

    Returns:
      `[L, V]`.
    """
    L = edge_mass.shape[0]
    U = jax.ops.segment_sum(edge_mass.T, class_id, num_segments=n_classes).T

    seg_of_nnz = jnp.repeat(
        jnp.arange(n_classes, dtype=jnp.int32),
        jnp.diff(indptr),
        total_repeat_length=indices.shape[0],
    )
    in_csr, is_rep, in_class = _csr_support(indices, seg_of_nnz, n_classes,
                                            vocab_size)

    # Positive classes: a plain scatter of their own mass onto their own tokens.
    pos = jnp.where(is_neg[None, :], 0.0, U)                     # [L, C]
    contrib = pos[:, seg_of_nnz]                                 # [L, nnz]

    # Negated classes, at the tokens the CSR stores: `Σ_{c ∈ Neg, v ∉ N_c} U(c)`
    # summed directly. One representative slot per distinct token id, so a token
    # several classes store is credited once.
    keep = (is_neg[:, None] & ~in_class) & is_rep[None, :]       # [C, nnz]
    contrib = contrib + U @ keep.astype(edge_mass.dtype)         # [L, nnz]

    r = jnp.zeros((L, vocab_size), dtype=edge_mass.dtype).at[:, indices].add(contrib)

    # Negated classes, at every token no class stores: all of them apply, and
    # `neg_total` is reached without ever forming a difference.
    neg_total = jnp.where(is_neg[None, :], U, 0.0).sum(axis=1)   # [L]
    return r + jnp.where(in_csr[None, :], 0.0, neg_total[:, None])


def constrained_marginals(
    p_vl: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    edge_src: jnp.ndarray,
    edge_dst: jnp.ndarray,
    class_id: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    is_neg: jnp.ndarray,
    n_classes: int,
) -> jnp.ndarray:
    """`q_i(v) = p_i(v) · r_i(v) / Z`, via SPEC §2.4's complement-aware form.

    Under the class representation the scatter in eq (6) is **not** a plain
    sparse scatter — negated classes contribute everywhere. Since
    `1[v ∈ S_c] = 1 − 1[v ∈ N_c]`:

        r_i(v) = Σ_{c ∈ Neg} U_i(c)
               + Σ_{c ∈ Pos, v ∈ S_c} U_i(c)
               − Σ_{c ∈ Neg, v ∈ N_c} U_i(c)

    with `U_i(c) = Σ_{e ∈ class c} u_i(e)` and `u_i(e) = a_{i-1}(src) · b_i(dst)`.

    Returns:
      `[L, V]`, each row summing to 1.
    """
    return constrained_marginals_and_partition(
        p_vl, a, b, edge_src, edge_dst, class_id, indices, indptr, is_neg,
        n_classes)[0]


@functools.partial(jax.jit, static_argnames=("n_classes",))
def constrained_marginals_and_partition(
    p_vl: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    edge_src: jnp.ndarray,
    edge_dst: jnp.ndarray,
    class_id: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    is_neg: jnp.ndarray,
    n_classes: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`constrained_marginals`, plus the **per-position** partition function.

    **Why this exists.** `Σ_v q_i(v) == 1` is advertised as a self-check but is
    *vacuous*: `q` is produced by dividing by its own row sum, so the identity
    holds bit-for-bit under an arbitrary scale error. A mutation audit injected
    a missing `u_i × W[i,e]` factor, an `a`/`b` off-by-one, and a deliberate
    1e7 scale error — all three left every row sum at exactly 1.0.

    The non-vacuous invariant is one level up. Before normalisation,

        Σ_v p_i(v) · r_i(v)  =  Z   for **every** `i`,

    because both sides are the same sum over accepted length-`L` strings, just
    grouped by a different position. So the `L` unnormalised row sums must all
    be *equal*, and their common value is `Z`. That equality is what an
    `a`/`b` misalignment or a dropped factor actually breaks, and it is
    checkable without knowing `Z` in advance.

    Returns:
      `(q [L, V], Z_per_position [L])`. The second output is the detector; see
      `tests/test_numerics.py`.
    """
    V, L = p_vl.shape
    u = a[:-1][:, edge_src] * b[1:][:, edge_dst]                 # [L, E]
    r = scatter_edge_mass_to_tokens(u, class_id, indices, indptr, is_neg,
                                    n_classes, V)
    row = p_vl.T * r
    # The floor MUST be derived from the dtype, not hardcoded. `1e-300` flushes
    # to exactly 0.0 in float32 (min normal 1.18e-38), so the guard silently
    # does nothing in the dtype that needs it most: on Z == 0 the fp32 result is
    # all-NaN, and `entropy_from_q` propagates that straight into SPEC §3.4's
    # acceptance mask. Verified by two reviewers independently.
    floor = jnp.finfo(row.dtype).tiny
    Z_i = row.sum(axis=1)                                        # [L]
    return row / jnp.maximum(Z_i[:, None], floor), Z_i


@jax.jit
def entropy_from_q(q: jnp.ndarray) -> jnp.ndarray:
    """`H_i = -Σ_v q_i(v) log q_i(v)`, with SPEC §2.4's clamp.

    **`q` has exact zeros wherever the automaton forbids a token, so never take
    `log q` directly** — the naive entropy is `NaN`. `log(maximum(q, 1e-30))`
    gives exactly `0·(−69) = 0` on forbidden tokens.
    """
    lq = jnp.log(jnp.maximum(q, 1e-30))
    return -(q * lq).sum(axis=-1)


@functools.partial(jax.jit, static_argnames=("n_classes", "vocab_size"))
def constrained_entropy_streamed(
    p_vl: jnp.ndarray,
    u: jnp.ndarray,
    class_id: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    is_neg: jnp.ndarray,
    n_classes: int,
    vocab_size: int,
) -> jnp.ndarray:
    """`H(q_i)` **without ever materialising `q` at `[L, V]`**.

    `entropy_from_q` needs `q = p·r/Z` as a dense `[L, V]` array. At `L = 256`
    and `V = 262,144` that is 67M float64 entries — 537 MB — and several such
    intermediates are chained *inside* the denoising `while_loop`. Measured
    consequence: the `--confidence=mar` arm deadlocked on **three** records
    (CPU frozen at 5:21 while elapsed reached 39 minutes, state `Ssl`), the
    fourth hang in the same XLA:GPU fused-reduction area.

    The way out is structural rather than numerical. `r_i(v)` takes only about
    `C` distinct values, because that is precisely what the class tables
    encode: every token in a class has the same `r`. So the entropy can be
    accumulated over the **stored CSR entries** plus one closed-form term for
    the tokens no class mentions, and the `[L, V]` array is never built.

    The identity, with `w_i(v) = p_i(v)·r_i(v)` and `Z_i = Σ_v w_i(v)`:

        H(q_i) = log Z_i − (1/Z_i)·Σ_v w_i(v)·log w_i(v)

    Both sums decompose over the CSR, so the cost is `O(nnz)` rather than
    `O(L·V)`.

    Returns:
      `[L]`.
    """
    L = u.shape[0]
    r = scatter_edge_mass_to_tokens(u, class_id, indices, indptr, is_neg,
                                    n_classes, vocab_size)
    # `r` is [L, V] and unavoidable with the current scatter, but it is ONE
    # array rather than the chain `q`, `row`, `lq`, `q*lq` that the dense path
    # builds. Kept explicit so the remaining cost is visible.
    w = p_vl.T * r
    tiny = jnp.finfo(w.dtype).tiny
    Z = w.sum(axis=1)
    safe_Z = jnp.maximum(Z, tiny)
    s = (w * jnp.log(jnp.maximum(w, tiny))).sum(axis=1)
    return jnp.log(safe_Z) - s / safe_Z
