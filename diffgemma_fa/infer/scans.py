"""Blelloch up-sweep / down-sweep over transition matrices, in JAX. SPEC §2.6.

The paper's `O(L) → O(log L)` depth reduction is an associative scan over
transition matrices. `jax.lax.associative_scan` computes **exactly the tree we
need and then throws it away** — its `reduced_elems` at recursion depth `k` are
the `P_{[ℓ,r)}` over aligned dyadic blocks of size `2^k`, and there is no API to
retain them. So this hand-rolls the sweep.

Three shape decisions, each of which SPEC says costs a day if taken wrongly:

**Blelloch/Brent–Kung, not Kogge–Stone.** A shallower Hillis–Steele scan does
**3.6× the work** (1,793 combines at `L=256` against 502), does not produce the
aligned dyadic node set eq (7) requires, and materialising its levels costs
`L·log₂L = 2048` nodes against `2L−1 = 511` — a **4×** memory blow-up that
invalidates §5.6's dispatch table. The often-cited shallow-scan win is a
benchmark on 500 *small* matrices; at `S ≥ 512` a single combine is a
multi-GFLOP GEMM and the work-efficient shape strictly wins.

**Python-unrolled at trace time, never `lax.scan` and never `lax.fori_loop`.**
A `lax.scan` compiles to a device `while`, and `WHILE` is **absent** from
XLA:GPU's default `xla_gpu_enable_command_buffer` set — so it escapes CUDA-graph
capture and pays 256 per-kernel launches. That is the mechanism behind the
paper's +114%. `fori_loop` cannot be used either, because the level shapes
differ (`L, L/2, L/4, …`), so it is not a real loop. Unrolling is what buys the
capture.

**Normalize every node.** At `L=256` an unnormalized product underflows fp32 to
exactly zero. The scale factors are scalars that cancel in both the midpoint
conditional and the root draw (SPEC §2.6), so normalisation is *provably free* —
but the log-scale sum must run over **all `2L−1` nodes** to recover `Z`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

__all__ = [
    "TreeLevels",
    "up_sweep",
    "prefix_suffix",
    "NEG_SENTINEL",
    "maxplus_combine",
    "up_sweep_maxplus",
    "log_matmul",
    "up_sweep_log",
]

#: Finite sentinel for "impossible" in the max-plus semiring. **Never `-inf`**:
#: fused kernels produce `NaN` from `-inf + -inf` (SPEC §2.7).
NEG_SENTINEL = -3e38


@dataclasses.dataclass(frozen=True)
class TreeLevels:
    """The retained up-sweep.

    Attributes:
      levels: `levels[k]` has shape `[L / 2^k, S, S]` and holds the aligned
        dyadic products `P_{[j·2^k, (j+1)·2^k)}`. `levels[0]` are the leaves
        `M_i`; the last entry is the single root `P_{[0,L)}`.
      log_scales: `log_scales[k]` is `[L / 2^k]`, the accumulated log of every
        normalisation applied at or below that node. Summing the root's entry
        recovers the true magnitude.
    """

    levels: tuple[jnp.ndarray, ...]
    log_scales: tuple[jnp.ndarray, ...]

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def root(self) -> jnp.ndarray:
        return self.levels[-1][0]

    @property
    def root_log_scale(self) -> jnp.ndarray:
        return self.log_scales[-1][0]

    @property
    def n_nodes(self) -> int:
        """`2L − 1` for a power-of-two `L`. SPEC §5.6's memory table row."""
        return sum(int(x.shape[0]) for x in self.levels)


def _normalize(mats: jnp.ndarray, carried: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Max-normalize each matrix, folding the scale into `carried` (in logs)."""
    m = jnp.max(mats, axis=(-2, -1))
    safe = jnp.where(m > 0, m, 1.0)
    return mats / safe[:, None, None], carried + jnp.log(safe)


def up_sweep(M: jnp.ndarray, *, normalize: bool = True) -> TreeLevels:
    """Bottom-up Blelloch sweep retaining every aligned dyadic product.

    Args:
      M: `[L, S, S]` with `L` a power of two (pad with identities otherwise —
        SPEC §2.6(d)).
      normalize: max-normalize each node, accumulating log-scales. Mandatory in
        fp32 at `L = 256`.

    Returns:
      A `TreeLevels` with `log₂(L) + 1` levels and `2L − 1` nodes.

    The combine is `left @ right`, **left block first**. `reverse=True` on
    `lax.associative_scan` yields `f(f(z,y),x)`, which for non-commutative
    matmul is the wrong order — no shape error, just wrong probabilities
    (SPEC §2.6(c)). That is why the suffix pass below is written out rather than
    delegated.
    """
    L = M.shape[0]
    if L & (L - 1):
        raise ValueError(f"L must be a power of two, got {L}; pad with identities")

    cur = M
    carried = jnp.zeros((L,), dtype=M.dtype)
    if normalize:
        cur, carried = _normalize(cur, carried)

    levels = [cur]
    log_scales = [carried]

    # Python `while`, unrolled at trace time — the level shapes differ, so this
    # cannot be a device loop even in principle.
    while cur.shape[0] > 1:
        left = cur[0::2]
        right = cur[1::2]
        nxt = left @ right                      # left block first
        sc = log_scales[-1][0::2] + log_scales[-1][1::2]
        if normalize:
            nxt, sc = _normalize(nxt, sc)
        levels.append(nxt)
        log_scales.append(sc)
        cur = nxt

    return TreeLevels(levels=tuple(levels), log_scales=tuple(log_scales))


def prefix_suffix(
    tree: TreeLevels,
    a_start: jnp.ndarray,
    b_final: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Down-sweep for the exclusive prefix/suffix vectors `a` and `b`.

    **This is where `a` and `b` come from, and why the down-sweep is needed at
    all** — SPEC is explicit that they must not come from a sequential loop.

    Returns:
      `(a, b, log_scale_a, log_scale_b)` with `a`, `b` shaped `[L+1, S]`:
      `a[i]` is the vector before position `i` (`a[0] == a_start`), and `b[i]`
      likewise with `b[L] == b_final`.
    """
    leaves = tree.levels[0]
    leaf_scales = tree.log_scales[0]
    L, S = leaves.shape[0], leaves.shape[1]

    # Exclusive prefix: a[i+1] = a[i] @ leaves[i]. Unrolled; the tree above is
    # what makes the *products* log-depth, and these vector-matrix products are
    # cheap by comparison, but they are still emitted as straight-line code so
    # nothing becomes a device while-loop.
    a_list = [a_start]
    sa_list = [jnp.zeros((), dtype=leaves.dtype)]
    for i in range(L):
        v = a_list[-1] @ leaves[i]
        s = sa_list[-1] + leaf_scales[i]
        m = jnp.max(v)
        safe = jnp.where(m > 0, m, 1.0)
        a_list.append(v / safe)
        sa_list.append(s + jnp.log(safe))

    b_list = [b_final]
    sb_list = [jnp.zeros((), dtype=leaves.dtype)]
    for i in range(L - 1, -1, -1):
        v = leaves[i] @ b_list[-1]
        s = sb_list[-1] + leaf_scales[i]
        m = jnp.max(v)
        safe = jnp.where(m > 0, m, 1.0)
        b_list.append(v / safe)
        sb_list.append(s + jnp.log(safe))

    a = jnp.stack(a_list)
    log_a = jnp.stack(sa_list)
    b = jnp.stack(b_list[::-1])
    log_b = jnp.stack(sb_list[::-1])
    return a, b, log_a, log_b


# ---------------------------------------------------------------------------
# max-plus semiring, for exact constrained MAP
# ---------------------------------------------------------------------------

def maxplus_combine(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """`(max, +)` matrix product: `out[i,j] = max_k a[i,k] + b[k,j]`.

    `(max, +)` is a semiring, so the identical tree gives exact `O(log L)` MAP.
    Log space, never `(max, ×)`: exact, no scaling discussion, no underflow
    (SPEC §2.7).
    """
    out = jnp.max(a[..., :, :, None] + b[..., None, :, :], axis=-2)
    # Re-clamp to the sentinel. Two sentinels sum to -6e38, which SATURATES TO
    # -inf in float32 at the very first combine — defeating the whole reason
    # this module uses a finite sentinel ("never -inf: fused kernels produce
    # NaN from -inf + -inf"). `log_matmul` already re-clamps; this did not.
    return jnp.maximum(out, jnp.asarray(NEG_SENTINEL, dtype=out.dtype))


def up_sweep_maxplus(M: jnp.ndarray) -> TreeLevels:
    """The same Blelloch shape over the max-plus semiring.

    No normalisation: in log space with `(max, +)` there is nothing to
    underflow, which is the reason SPEC prescribes log space here rather than
    `(max, ×)`.
    """
    L = M.shape[0]
    if L & (L - 1):
        raise ValueError(f"L must be a power of two, got {L}")

    cur = M
    levels = [cur]
    zeros = jnp.zeros((L,), dtype=M.dtype)
    scales = [zeros]
    while cur.shape[0] > 1:
        cur = maxplus_combine(cur[0::2], cur[1::2])
        levels.append(cur)
        scales.append(jnp.zeros((cur.shape[0],), dtype=M.dtype))
    return TreeLevels(levels=tuple(levels), log_scales=tuple(scales))


# ---------------------------------------------------------------------------
# Log-space sum-product — required at the real canvas length
# ---------------------------------------------------------------------------

def log_matmul(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """`C[i,j] = logsumexp_k (A[i,k] + B[k,j])`, **still as a GEMM**.

    Phase 4 measured that linear-space sum-product underflows at `L = 256` even
    in float64: the unscored `ACC --Σ--> ACC` tail has emission mass exactly 1.0
    and so pins every node's max at 1.0, while genuine grammar paths sit near
    1e-49 and below. Per-node normalization cannot fix a dynamic range that
    large *inside* one matrix. Log space can, and it is what SPEC §2.7 already
    chose for MAP — "exact, no scaling discussion, no underflow".

    The naive form materialises `[n, k, m]`, which is `1e9` elements at
    `|S| = 1024`. Shifting by the **row** max of `A` and the **column** max of
    `B` instead leaves an ordinary matmul of matrices whose entries all lie in
    `[0, 1]`, so cuBLAS still does the work and the `O(log L)` kernel-count
    property is preserved:

        C[i,j] = ra[i] + cb[j] + log( Σ_k exp(A[i,k] − ra[i]) · exp(B[k,j] − cb[j]) )

    Impossible entries carry `NEG_SENTINEL`, whose exponential is exactly 0, so
    they propagate correctly without ever producing `NaN` from `-inf + -inf`.
    """
    ra = jnp.max(A, axis=-1, keepdims=True)
    cb = jnp.max(B, axis=-2, keepdims=True)
    ra = jnp.where(ra > NEG_SENTINEL / 2, ra, jnp.zeros_like(ra))
    cb = jnp.where(cb > NEG_SENTINEL / 2, cb, jnp.zeros_like(cb))
    prod = jnp.exp(A - ra) @ jnp.exp(B - cb)
    tiny = jnp.asarray(jnp.finfo(A.dtype).tiny, dtype=A.dtype)
    out = ra + cb + jnp.log(jnp.maximum(prod, tiny))
    return jnp.where(prod > 0, out, jnp.full_like(out, NEG_SENTINEL))


def up_sweep_log(logM: jnp.ndarray) -> TreeLevels:
    """The Blelloch up-sweep over the **log** sum-product semiring.

    Same shape, same `2L−1` nodes, same `log₂ L` combines — only the combine
    changes. No normalisation is needed or meaningful here, which is the point:
    log space removes the scaling question entirely rather than managing it.
    """
    L = logM.shape[0]
    if L & (L - 1):
        raise ValueError(f"L must be a power of two, got {L}")

    cur = logM
    levels = [cur]
    scales = [jnp.zeros((L,), dtype=logM.dtype)]
    while cur.shape[0] > 1:
        cur = log_matmul(cur[0::2], cur[1::2])
        levels.append(cur)
        scales.append(jnp.zeros((cur.shape[0],), dtype=logM.dtype))
    return TreeLevels(levels=tuple(levels), log_scales=tuple(scales))
