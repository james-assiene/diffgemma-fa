"""Tree sampling and constrained MAP over the retained Blelloch levels. SPEC §2.6, §2.7.

Top-down, given `scans.up_sweep`'s retained dyadic products:

    (s_0, s_L) ~ a_start(s_0) · P_{[0,L)}(s_0, s_L) · b_L(s_L)
    P(s_m | s_ℓ, s_r) ∝ P_{[ℓ,m)}(s_ℓ, s_m) · P_{[m,r)}(s_m, s_r)          (7)
    e_i ~ P(e) ∝ 1[src(e)=s_i, dst(e)=s_{i+1}] · W[i, e]                    (8a)
    x_i ~ P(v) ∝ p_i(v) · 1[v ∈ label(e_i)]                                 (8b)

Every level's midpoints are independent given their bracketing states, so each
level is one vmapped categorical — that is the `O(log L)` depth.

**eq (8) must be edge-multiplicity-weighted.** The one-step equivalent is
`x_i ~ p_i(v) · |{e : src=s_i, dst=s_{i+1}, v ∈ label(e)}|`; the `∃`-indicator
form is a real bug on an NFA. The multiplicity is computed by feeding the edge
indicator through the same complement-aware scatter that eq (6) uses.

**PRNG.** `jax.random.split(jax.random.fold_in(key, level), n_nodes)` then vmap
across nodes. `n_nodes` is static because the tree is unrolled, which satisfies
`split`'s static-count requirement for free, and `fold_in` is a pure function of
`(key, index)` so unrolling or reordering cannot change results.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffgemma_fa.infer import marginals as _marginals
from diffgemma_fa.infer.scans import NEG_SENTINEL, TreeLevels

__all__ = ["sample_states", "sample_tokens", "map_states_and_tokens"]


def sample_states(
    tree: TreeLevels,
    a_start: jnp.ndarray,
    b_final: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Draw all `L+1` boundary states. SPEC eq (7).

    Returns:
      `[L+1] int32`.
    """
    L = tree.levels[0].shape[0]
    S = tree.levels[0].shape[1]

    # --- root: (s_0, s_L) jointly. The start is a VECTOR (SPEC §5.7). ------
    joint = a_start[:, None] * tree.root * b_final[None, :]
    flat = jax.random.categorical(
        jax.random.fold_in(key, 0), jnp.log(jnp.maximum(joint.ravel(), 1e-300))
    )
    s0, sL = flat // S, flat % S

    states = jnp.zeros((L + 1,), dtype=jnp.int32)
    states = states.at[0].set(s0.astype(jnp.int32))
    states = states.at[L].set(sL.astype(jnp.int32))

    # --- down-sweep: one vmapped draw per level ---------------------------
    # Level k holds products of width 2^k. To split an interval of width
    # 2^(k+1) we need its two children at level k.
    for k in range(tree.n_levels - 2, -1, -1):
        width = 1 << k
        n_intervals = L // (2 * width)
        if n_intervals == 0:
            continue
        lo = jnp.arange(n_intervals, dtype=jnp.int32) * (2 * width)
        mid = lo + width
        hi = lo + 2 * width

        child = tree.levels[k]                       # [L/2^k, S, S]
        left = child[jnp.arange(n_intervals) * 2]     # P_[lo, mid)
        right = child[jnp.arange(n_intervals) * 2 + 1]  # P_[mid, hi)

        s_lo = states[lo]
        s_hi = states[hi]
        # w[j, m] = P_[lo,mid)(s_lo, m) * P_[mid,hi)(m, s_hi)
        w = (left[jnp.arange(n_intervals), s_lo, :]
             * right[jnp.arange(n_intervals), :, s_hi])

        keys = jax.random.split(jax.random.fold_in(key, k + 1), n_intervals)
        drawn = jax.vmap(
            lambda kk, ww: jax.random.categorical(
                kk, jnp.log(jnp.maximum(ww, 1e-300))
            )
        )(keys, w)
        states = states.at[mid].set(drawn.astype(jnp.int32))

    return states


def sample_tokens(
    p_vl: jnp.ndarray,
    states: jnp.ndarray,
    edge_src: jnp.ndarray,
    edge_dst: jnp.ndarray,
    class_id: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    is_neg: jnp.ndarray,
    n_classes: int,
    key: jax.Array,
) -> jnp.ndarray:
    """eq (8), edge-multiplicity-weighted. All `L` positions in parallel.

    `mult[i, v] = |{e : src=s_i, dst=s_{i+1}, v ∈ label(e)}|` is obtained by
    pushing the edge indicator through the shared complement-aware scatter —
    the same kernel eq (6) uses, which is why the multiplicity comes out right
    on an NFA without a special case.

    Returns:
      `(tokens [L] int32, valid bool)` — see the validity note below.
    """
    V, L = p_vl.shape
    sel = ((edge_src[None, :] == states[:-1, None])
           & (edge_dst[None, :] == states[1:, None])).astype(p_vl.dtype)  # [L, E]
    mult = _marginals.scatter_edge_mass_to_tokens(
        sel, class_id, indices, indptr, is_neg, n_classes, V
    )
    logits = jnp.log(jnp.maximum(p_vl.T * mult, 1e-300))
    keys = jax.random.split(key, L)
    tokens = jax.vmap(jax.random.categorical)(keys, logits).astype(jnp.int32)
    # A position with no admissible token means `sample_states` handed us a
    # state pair with no edge between it — which happens only when the boundary
    # draw degenerated (see `model.constrained.require_x64`). Surfaced rather
    # than silently returning a near-uniform draw over the full 262k vocab,
    # which looks like plausible multilingual text and passes every shape check.
    valid = jnp.all(mult.sum(axis=-1) > 0)
    return tokens, valid


def map_states_and_tokens(
    logp_vl: jnp.ndarray,
    tree_maxplus: TreeLevels,
    edge_src: jnp.ndarray,
    edge_dst: jnp.ndarray,
    class_id: jnp.ndarray,
    class_max: jnp.ndarray,
    class_argmax: jnp.ndarray,
    a_start_log: jnp.ndarray,
    b_final_log: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact constrained MAP over the max-plus tree. SPEC §2.7.

    Token recovery needs a `(s,s') → class` table, not just a class table: once
    the tree fixes `(s_i, s_{i+1})` you still need to know **which** class
    realised the max on that transition. Storing an `O(L|S|²)` backtrace instead
    would be 12.4 GB at `|S| = 2459`.

        argclass[i, pair] = argmax over edges e with (src,dst)=pair of class_max[class_id[e], i]
        x_i               = class_argmax[argclass[i, pair(s_i, s_{i+1})], i]

    Args:
      logp_vl: `[V, L]` log-marginals.
      tree_maxplus: `up_sweep_maxplus` over `M̃`.
      class_max: `[C, L]`, `max_{v ∈ S_c} log p_i(v)`.
      class_argmax: `[C, L] int32`, the token realising it.

    Returns:
      `(tokens [L], states [L+1], score)`.
    """
    L = tree_maxplus.levels[0].shape[0]
    S = tree_maxplus.levels[0].shape[1]

    joint = a_start_log[:, None] + tree_maxplus.root + b_final_log[None, :]
    flat = jnp.argmax(joint.ravel())       # ties -> lowest flat index ->
    s0, sL = flat // S, flat % S           # lowest (s_0, s_L) lexicographically
    score = joint.ravel()[flat]

    states = jnp.zeros((L + 1,), dtype=jnp.int32)
    states = states.at[0].set(s0.astype(jnp.int32))
    states = states.at[L].set(sL.astype(jnp.int32))

    for k in range(tree_maxplus.n_levels - 2, -1, -1):
        width = 1 << k
        n_intervals = L // (2 * width)
        if n_intervals == 0:
            continue
        idx = jnp.arange(n_intervals)
        lo = idx.astype(jnp.int32) * (2 * width)
        mid = lo + width
        hi = lo + 2 * width
        child = tree_maxplus.levels[k]
        left = child[idx * 2]
        right = child[idx * 2 + 1]
        w = (left[idx, states[lo], :] + right[idx, :, states[hi]])
        states = states.at[mid].set(jnp.argmax(w, axis=-1).astype(jnp.int32))

    # Which class realises the max on each realised transition?
    sel = ((edge_src[None, :] == states[:-1, None])
           & (edge_dst[None, :] == states[1:, None]))                 # [L, E]
    per_edge = class_max[class_id, :].T                               # [L, E]
    masked = jnp.where(sel, per_edge, NEG_SENTINEL)
    best_edge = jnp.argmax(masked, axis=1)                            # [L]
    chosen_class = class_id[best_edge]                                # [L]
    tokens = class_argmax[chosen_class, jnp.arange(L)].astype(jnp.int32)
    return tokens, states, score
