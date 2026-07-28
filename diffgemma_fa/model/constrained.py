"""The constrained joint draw, assembled from the Phase 3 kernels.

One function per emission mode, each taking the model's shaped logits plus a
traced `Automaton` and returning a whole `[B, L]` canvas drawn from the
constrained posterior. `sample_from_predictions` receives the entire `[B,L,V]`
logits and returns an entire `[B,L]` canvas, and nothing downstream assumes
per-position independence — **joint sampling is fully permitted** (SPEC §5.1).

SPEC §3.1's emission designs:

| | trajectory (fed back to the model) | emitted | guarantee |
|---|---|---|---|
| **J0-map** | stock: constrained sample at accepted, uniform random elsewhere | constrained MAP | unconditional |
| **J0-sample** | as above | constrained joint draw | unconditional |
| **J1** | single joint draw with **flattened** marginals at non-accepted | same tensor | unconditional, *and every intermediate canvas ∈ C* |
| **J2** | constrained sample at accepted, uniform random elsewhere | same tensor | only at full acceptance |

J1 is built first (SPEC §5.4): it needs no change to the denoising carry, so it
is the shortest path to end-to-end constrained generation and validates the
whole inference stack before the carry is touched.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from diffgemma_fa.infer import marginals as MG
from diffgemma_fa.infer import scans, tree

__all__ = [
    "budget_terminal_factor",
    "flatten_unaccepted",
    "joint_draw",
    "joint_map",
    "advance_states",
    "require_x64",
    "X64Required",
]


class X64Required(RuntimeError):
    """The sum-product sampling path needs float64. See `require_x64`."""


def require_x64() -> None:
    """Assert `jax_enable_x64`, which the **sampling** path genuinely requires.

    **Measured, on a real BFCL grammar at `L = 64`.** SPEC §2.4/§2.6 prescribe
    max-normalizing every tree node, which fixes the *overall* scale — but not
    the **dynamic range inside a single matrix**, and at the root that range is
    what kills fp32:

    - `ACC --Σ--> ACC`, the unscored post-stop tail (§3.5 trap 4), has emission
      mass **exactly 1.0** by construction, so the root's max is pinned at 1.0
      and per-node normalization divides by 1.0 and does nothing;
    - a genuine constrained path is a product of per-token probabilities around
      `4e-6`, so over 64 positions the root entry is `~1e-49` and the smallest
      positive entry measured was **1.2e-288**.

    float32's smallest subnormal is ~1e-45, so the entire joint underflows to
    exactly zero and the root draw degenerates. In float64 the identical code
    samples correctly and the draw is accepted.

    This is SPEC's `Z == 0` cause **(c)** — "fp32 underflow on an unnormalized
    path… means your scaling is missing" — except that the scaling is *present*
    and still insufficient, because the tail edge pins the normalizer.

    Two consequences:

    - **`--emission=map` is unaffected**, because §2.7 puts MAP in log space
      specifically so that "no scaling discussion, no underflow" applies. That
      choice is now empirically vindicated and is an argument for MAP being the
      default emission.
    - **SPEC §5.6's memory table doubles for the sampling path**: the tree is
      `(2L−1)·|S|²·8` bytes, so at the measured ~20 GB of headroom the `|S|`
      ceiling falls from 3,128 to **2,211** (and from 1,978 to 1,399 at 8 GB).

    Raises:
      X64Required: if float64 is disabled, in which case the sampler would
        silently draw from a degenerate distribution rather than fail.
    """
    if jnp.zeros((), dtype=jnp.float64).dtype != jnp.float64:
        raise X64Required(
            "the constrained sampling path needs float64: set "
            "jax.config.update('jax_enable_x64', True) before importing. "
            "In fp32 the root product underflows to exactly zero (measured "
            "min positive entry 1.2e-288 against an fp32 floor of ~1e-45), "
            "because the unscored ACC->ACC tail pins the per-node normalizer "
            "at 1.0. --emission=map is in log space and is unaffected."
        )


def budget_terminal_factor(d: jnp.ndarray, remaining: jnp.ndarray,
                           dtype=jnp.float64) -> jnp.ndarray:
    """`b_L(s) = 1[d(s) ≤ R]`. SPEC §3.1b.

    **Not** `1[s ∈ F]`, and with **no final-block special case**: `d(s) ≤ 0 ⟺
    s ∈ F`, so the last block falls out automatically as `R` runs down. Using
    `1[s ∈ F]` instead would force the grammar to complete in exactly one
    canvas.
    """
    return (d <= remaining).astype(dtype)


def flatten_unaccepted(
    p: jnp.ndarray,
    accepted: jnp.ndarray,
    *,
    temperature: float = 1e6,
) -> jnp.ndarray:
    """J1's `p'`: the model's marginals where accepted, flattened elsewhere.

    Accepted positions keep the model's confident choice; the rest become
    near-uniform **but still grammar-consistent**, because the joint draw that
    follows is constrained. That is what makes every intermediate canvas a
    member of `C` — strictly stronger than the paper, whose intermediate
    canvases contain `[MASK]`s.

    The risk, which is why SPEC wants this ablated against J0: the model was
    trained to denoise **uniform random** noise, not grammar-valid noise.

    Args:
      p: `[L, V]` marginals.
      accepted: `[L] bool`.
      temperature: how flat the non-accepted rows become. Large ⇒ uniform.
    """
    flat = jnp.full_like(p, 1.0 / p.shape[-1])
    blended = jnp.where(accepted[:, None], p, flat)
    if temperature != 1e6:
        blended = jnp.where(accepted[:, None], p, p ** (1.0 / temperature))
        blended = blended / blended.sum(axis=-1, keepdims=True)
    return blended


def _matrices(p_lv, automaton, n_states, n_classes):
    """`W` and `M_i` from the class tables. Invalid (padding) edges are zeroed
    so a bucketed automaton behaves exactly like an unpadded one."""
    p_vl = p_lv.T
    seg = jnp.repeat(
        jnp.arange(n_classes, dtype=jnp.int32),
        jnp.diff(automaton.csr_indptr),
        total_repeat_length=automaton.csr_indices.shape[0],
    )
    W_c = MG.class_weights(p_vl, automaton.csr_indices, seg, automaton.is_neg,
                           n_classes)
    W_e = MG.edge_weights(W_c, automaton.edge_class)
    W_e = jnp.where(automaton.edge_valid[:, None], W_e, 0.0)
    M = MG.transition_matrices(W_e, automaton.edge_src, automaton.edge_dst,
                               n_states)
    return p_vl, W_e, M


@functools.partial(jax.jit, static_argnames=("n_states", "n_classes"))
def joint_draw(
    p_lv: jnp.ndarray,
    automaton,
    remaining: jnp.ndarray,
    key: jax.Array,
    n_states: int,
    n_classes: int,
) -> jnp.ndarray:
    """One draw from the constrained posterior over the whole canvas.

    `a_start = 1[s ∈ A_k]` — a **vector**, not a point mass (SPEC §5.7) — and
    `b_L = 1[d(s) ≤ R]`.

    Returns:
      `[L] int32`.
    """
    p_vl, W_e, M = _matrices(p_lv, automaton, n_states, n_classes)
    a_start = automaton.active.astype(p_lv.dtype)
    b_final = budget_terminal_factor(automaton.d, remaining, dtype=p_lv.dtype)

    tr = scans.up_sweep(M)
    k1, k2 = jax.random.split(key)
    states = tree.sample_states(tr, a_start, b_final, k1)
    return tree.sample_tokens(
        p_vl, states, automaton.edge_src, automaton.edge_dst,
        automaton.edge_class, automaton.csr_indices, automaton.csr_indptr,
        automaton.is_neg, n_classes, k2,
    )


@functools.partial(jax.jit, static_argnames=("n_states", "n_classes"))
def joint_map(
    p_lv: jnp.ndarray,
    automaton,
    remaining: jnp.ndarray,
    n_states: int,
    n_classes: int,
) -> jnp.ndarray:
    """Exact constrained MAP over the max-plus tree. SPEC §2.7, `--emission=map`.

    MAP is **exactly temperature-invariant**, so it is unaffected by the
    0.8 → 0.408 schedule — temperature enters only through sampling and the
    accept rule.

    Returns:
      `[L] int32`.
    """
    L, V = p_lv.shape
    logp = jnp.log(jnp.maximum(p_lv, 1e-30))

    # Per-class max and argmax over the *true* member set, complement-aware.
    # A negated class is evaluated as "the first topk(p, K) entry not in N_c";
    # here the padded CSR makes a direct masked max simpler and exact, which is
    # what the reference does too.
    seg = jnp.repeat(
        jnp.arange(n_classes, dtype=jnp.int32),
        jnp.diff(automaton.csr_indptr),
        total_repeat_length=automaton.csr_indices.shape[0],
    )
    onehot = jnp.zeros((n_classes, V), dtype=bool).at[seg, automaton.csr_indices].set(True)
    member = jnp.where(automaton.is_neg[:, None], ~onehot, onehot)   # [C, V]

    scored = jnp.where(member[:, None, :], logp[None, :, :],
                       jnp.asarray(scans.NEG_SENTINEL, dtype=logp.dtype))
    class_max = jnp.max(scored, axis=-1)                              # [C, L]
    class_arg = jnp.argmax(scored, axis=-1).astype(jnp.int32)         # ties -> lowest id

    per_edge = class_max[automaton.edge_class, :]                     # [E, L]
    per_edge = jnp.where(automaton.edge_valid[:, None], per_edge,
                         scans.NEG_SENTINEL)
    Mt = jnp.full((L, n_states, n_states), scans.NEG_SENTINEL,
                  dtype=logp.dtype)
    Mt = Mt.at[:, automaton.edge_src, automaton.edge_dst].max(per_edge.T)

    neg = jnp.asarray(scans.NEG_SENTINEL, dtype=logp.dtype)
    a_log = jnp.where(automaton.active, jnp.zeros((), logp.dtype), neg)
    b_log = jnp.where(automaton.d <= remaining, jnp.zeros((), logp.dtype), neg)

    tr = scans.up_sweep_maxplus(Mt)
    tokens, _, _ = tree.map_states_and_tokens(
        logp.T, tr, automaton.edge_src, automaton.edge_dst,
        automaton.edge_class, class_max, class_arg, a_log, b_log,
    )
    return tokens


@functools.partial(jax.jit, static_argnames=("n_states", "n_classes", "vocab_size"))
def advance_states(
    automaton,
    tokens: jnp.ndarray,
    n_states: int,
    n_classes: int,
    vocab_size: int,
) -> jnp.ndarray:
    """`A_{k+1} = δ*(A_k, canvas_k)`. SPEC §3.5.

    Computed from the **truncated** canvas, and as traced ops on fixed-shape
    arrays — there is no Python between blocks, so `A_k` is a `[S]` array and
    never a Python set.

    Membership of each token in each class is recovered complement-aware from
    the CSR, so this is correct for negated classes.
    """
    seg = jnp.repeat(
        jnp.arange(n_classes, dtype=jnp.int32),
        jnp.diff(automaton.csr_indptr),
        total_repeat_length=automaton.csr_indices.shape[0],
    )
    onehot = jnp.zeros((n_classes, vocab_size), dtype=bool).at[
        seg, automaton.csr_indices].set(True)
    member = jnp.where(automaton.is_neg[:, None], ~onehot, onehot)  # [C, V]

    def step(active, tok):
        edge_ok = member[automaton.edge_class, tok] & automaton.edge_valid
        live = active[automaton.edge_src] & edge_ok
        nxt = jnp.zeros((n_states,), dtype=bool).at[automaton.edge_dst].max(live)
        # A canvas is PAD-filled after the first stop token; PAD keeps the set
        # unchanged rather than killing it, because the automaton's unscored
        # tail (SPEC §3.5 trap 4) already accepts anything after the stop.
        return jnp.where(nxt.any(), nxt, active), None

    final, _ = jax.lax.scan(step, automaton.active, tokens)
    return final
