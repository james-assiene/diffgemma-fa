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
    "entropy_from_q",
    "MASK_SENTINEL",
]

#: SPEC §2.4/§3.7: mask with a **finite** sentinel, never `-inf`. An `-inf`
#: logit propagates through `softmax → @ embed_tokens.weight` and poisons the
#: self-conditioning embedding.
MASK_SENTINEL = -1e30


@functools.partial(jax.jit, static_argnames=("n_classes",))
def class_weights(
    p_vl: jnp.ndarray,
    indices: jnp.ndarray,
    seg_ids: jnp.ndarray,
    is_neg: jnp.ndarray,
    n_classes: int,
) -> jnp.ndarray:
    """`W_c[c, i] = Σ_{v ∈ S_c} p_i(v)`, complement-aware. SPEC §4.4 Layers 2-3.

    For a negated class the stored set is `N_c` and the true mass is recovered
    as `total[i] − Σ_{v ∈ N_c} p_i(v)`. **Without this the scheme fails**: real
    BFCL classes cover ~99.6% of the vocabulary, so storing them positively
    would put ~260k entries per class in the CSR.

    Args:
      p_vl: `[V, L]` column-major marginals.
      indices: `[nnz] int32`, CSR column indices over the **stored** set.
      seg_ids: `[nnz] int32`, class id per stored entry.
      is_neg: `[C] bool`, whether each class stores its complement.
      n_classes: `C`. **Static** — a traced value raises.

    Returns:
      `[C, L]`.
    """
    gathered = p_vl[indices, :]                                   # [nnz, L]
    partial = jax.ops.segment_sum(gathered, seg_ids, num_segments=n_classes)
    total = p_vl.sum(axis=0)[None, :]                             # [1, L]
    return jnp.where(is_neg[:, None], total - partial, partial)


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

    Negated classes contribute their full mass everywhere and are then
    subtracted on the complement they actually store — a plain sparse scatter
    would be wrong (SPEC §2.4).

    Args:
      edge_mass: `[L, E]`.
      n_classes, vocab_size: **static**.

    Returns:
      `[L, V]`.
    """
    L = edge_mass.shape[0]
    U = jax.ops.segment_sum(edge_mass.T, class_id, num_segments=n_classes).T

    neg_total = jnp.where(is_neg[None, :], U, 0.0).sum(axis=1)   # [L]
    signed = jnp.where(is_neg[None, :], -U, U)                   # [L, C]

    seg_of_nnz = jnp.repeat(
        jnp.arange(n_classes, dtype=jnp.int32),
        jnp.diff(indptr),
        total_repeat_length=indices.shape[0],
    )
    contrib = signed[:, seg_of_nnz]                              # [L, nnz]
    r = jnp.zeros((L, vocab_size), dtype=edge_mass.dtype).at[:, indices].add(contrib)
    return r + neg_total[:, None]


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
    V, L = p_vl.shape
    u = a[:-1][:, edge_src] * b[1:][:, edge_dst]                 # [L, E]
    r = scatter_edge_mass_to_tokens(u, class_id, indices, indptr, is_neg,
                                    n_classes, V)
    row = p_vl.T * r
    return row / jnp.maximum(row.sum(axis=1, keepdims=True), 1e-300)


@jax.jit
def entropy_from_q(q: jnp.ndarray) -> jnp.ndarray:
    """`H_i = -Σ_v q_i(v) log q_i(v)`, with SPEC §2.4's clamp.

    **`q` has exact zeros wherever the automaton forbids a token, so never take
    `log q` directly** — the naive entropy is `NaN`. `log(maximum(q, 1e-30))`
    gives exactly `0·(−69) = 0` on forbidden tokens.
    """
    lq = jnp.log(jnp.maximum(q, 1e-30))
    return -(q * lq).sum(axis=-1)
