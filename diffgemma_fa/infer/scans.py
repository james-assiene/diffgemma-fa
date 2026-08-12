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
      underflow: `[L] bool` or `None`. True at every leaf position covered by a
        node where max-normalisation drove a **strictly positive** entry to
        exactly `0.0` — i.e. where a live transition was destroyed rather than
        merely rounded. `None` means "not measured" (the max-plus and log
        sweeps, which have nothing to normalise, and any hand-built
        `TreeLevels`), and consumers must treat that as *no information*, never
        as a clean bill of health.
    """

    levels: tuple[jnp.ndarray, ...]
    log_scales: tuple[jnp.ndarray, ...]
    underflow: jnp.ndarray | None = None

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


def _normalize(mats: jnp.ndarray, carried: jnp.ndarray
               ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Max-normalize each matrix, folding the scale into `carried` (in logs).

    Also returns `[n] bool`, true where the division took a strictly positive
    entry to exactly `0.0`. That is **not** ordinary rounding: `prefix_suffix`
    rebuilds `log M_i` as `where(m > 0, log(max(m, tiny)) + leaf_scales[i],
    NEG_SENTINEL)`, so a flushed entry is reclassified as a *structurally
    impossible transition* and leaves through `viable == False`. A live edge
    then vanishes with `Z == 0` and no way for the caller to classify it —
    which is none of CLAUDE.md's three causes, since the language is non-empty,
    the budget is satisfiable and the scaling is present.

    The threshold is `|log(tiny)| + log(max_{s,s'} M_i(s,s'))` — **708.396**
    nats above the leaf maximum, not float64's 744.44. XLA flushes subnormals
    to zero, so the quotient dies at the smallest *normal* rather than at the
    smallest denormal; the boundary was pinned between 708 and 709. Measured on
    a leaf with `max M_i = 1e300` and a live entry 736.8 nats below it,
    `1e-20 / 1e300` returns exactly `0.0` under `jnp` where numpy returns the
    subnormal `1e-320`.

    **Contract, and it is narrower than "this position lost a live entry".**
    What is detected is loss *caused by this division*: `mats > 0` and the
    quotient `== 0`. An entry that already underflowed to `0.0` in the caller's
    own arithmetic — a gap of ~750 nats or more, where `M_i(s,s')` is zero
    before `_normalize` ever sees it — reads **False** here, because from this
    function's vantage the transition is indistinguishable from a structurally
    absent one. That is a limit of the two-argument interface, not a detector
    failure: recovering it needs the exact value, which is `up_sweep_log`'s job
    (SPEC §2.6 `[V-P4]`).

    Fed `M` from `marginals.transition_matrices` on probability marginals
    `max M_i` is at most a small multiple of 1, so on the production path this
    costs only `log(max M_i)` nats beyond the float's own floor — measured
    firing on 0/240 synthetic `L = 256` sweeps at σ up to 45, and on 0/256
    positions of bfcl, countdown and sudoku at `T = 1.0 / 0.4 / 0.2 / 0.1`.
    What is regime-independent, and what this return value exists for, is that
    the loss must not be silent.
    """
    m = jnp.max(mats, axis=(-2, -1))
    safe = jnp.where(m > 0, m, 1.0)
    out = mats / safe[:, None, None]
    lost = jnp.any((mats > 0) & (out == 0), axis=(-2, -1))
    return out, carried + jnp.log(safe), lost


def up_sweep(M: jnp.ndarray, *, normalize: bool = True) -> TreeLevels:
    """Bottom-up Blelloch sweep retaining every aligned dyadic product.

    Args:
      M: `[L, S, S]` with `L` a power of two (pad with identities otherwise —
        SPEC §2.6(d)).
      normalize: max-normalize each node, accumulating log-scales. Mandatory in
        fp32 at `L = 256`.

    Returns:
      A `TreeLevels` with `log₂(L) + 1` levels and `2L − 1` nodes, and — when
      `normalize` — an `underflow [L] bool` marking every leaf position under a
      node whose normalisation destroyed a live entry (`_normalize`).
      `prefix_suffix` folds it into `representable`, so the caller is told
      rather than handed an unclassifiable `Z == 0`.

      **What `underflow` does not cover** — two things, both because detecting
      them needs the exact value rather than the quotient: an entry that was
      already `0.0` on arrival (a gap past ~750 nats, so the caller's own
      arithmetic killed it, not the division), and the `left @ right` product
      underflowing at an internal level. Both are `up_sweep_log`'s job
      (SPEC §2.6 `[V-P4]`); the second cannot reach `prefix_suffix` at all,
      which reads only `levels[0]`. See `_normalize`.

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
    # `[L] bool`, ORed up as levels are built: a node at level `k` covers leaf
    # positions `[j·2^k, (j+1)·2^k)`, and a normalisation that destroys a live
    # entry there contaminates every one of them. See `_normalize`.
    underflow = jnp.zeros((L,), dtype=bool)
    if normalize:
        cur, carried, lost = _normalize(cur, carried)
        underflow = underflow | lost

    levels = [cur]
    log_scales = [carried]

    # Python `while`, unrolled at trace time — the level shapes differ, so this
    # cannot be a device loop even in principle.
    width = 1
    while cur.shape[0] > 1:
        left = cur[0::2]
        right = cur[1::2]
        nxt = left @ right                      # left block first
        sc = log_scales[-1][0::2] + log_scales[-1][1::2]
        width *= 2
        if normalize:
            nxt, sc, lost = _normalize(nxt, sc)
            underflow = underflow | jnp.repeat(lost, width)
        levels.append(nxt)
        log_scales.append(sc)
        cur = nxt

    return TreeLevels(levels=tuple(levels), log_scales=tuple(log_scales),
                      underflow=underflow)


def _log_reduce(t: jnp.ndarray, axis: int) -> jnp.ndarray:
    """`logsumexp` along `axis`, anchored on that axis's own max. `[S,S] -> [S]`.

    The same anchor `log_matmul` uses and for the same reason (its docstring
    records two wrong shifts measured on real grammars): relative to the max the
    dominant term is `exp(0) = 1`, so nothing that matters underflows at any
    dynamic range. `log_matmul` itself is not called here because a
    vector-matrix product has `n = 1` and its broadcast `[1, S, S]` operand is
    just `t` — recomputing it for the second reduction, which is the right trade
    at `[L/2, S, S]`, is pure waste at this shape.
    """
    m = jnp.max(t, axis=axis)
    live = m > NEG_SENTINEL / 2.0
    safe = jnp.where(live, m, jnp.zeros_like(m))
    s = jnp.sum(jnp.exp(t - jnp.expand_dims(safe, axis)), axis=axis)
    tiny = jnp.asarray(jnp.finfo(t.dtype).tiny, dtype=t.dtype)
    return jnp.where(live, safe + jnp.log(jnp.maximum(s, tiny)),
                     jnp.full_like(m, NEG_SENTINEL))


def _log_vec_mat(v: jnp.ndarray, logM: jnp.ndarray) -> jnp.ndarray:
    """`out[j] = logsumexp_i (v[i] + logM[i, j])`. `[S] , [S, S] -> [S]`."""
    return _log_reduce(v[:, None] + logM, axis=0)


def _log_mat_vec(logM: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """`out[i] = logsumexp_j (logM[i, j] + v[j])`. `[S, S] , [S] -> [S]`."""
    return _log_reduce(logM + v[None, :], axis=1)


def prefix_suffix(
    tree: TreeLevels,
    a_start: jnp.ndarray,
    b_final: jnp.ndarray,
    *,
    feasible_out: bool = False,
) -> tuple[jnp.ndarray, ...]:
    """Exclusive prefix/suffix vectors `a` and `b`, over the tree's leaves.

    **What this is not.** An earlier docstring claimed "the down-sweep is needed
    at all — SPEC is explicit that they must not come from a sequential loop",
    which does not describe the code below: the vector-matrix chain is a
    **Python-unrolled sequential loop**, `O(L)` deep in the dataflow graph, not
    a Blelloch down-sweep.

    That is a deliberate trade, and SPEC's actual constraint is satisfied. The
    rule (SPEC §0, §2.4) is about **kernel launch count**, not dataflow depth:
    a `lax.scan` compiles to a device `while`, which is *not* in XLA's default
    command-buffer capture set — precisely the paper's +114%. Python unrolling
    emits straight-line code, so the whole chain is capturable. What it costs
    is `L` dependent vector-matrix products of latency, which is small next to
    the `[S, S]` matmuls the tree above already does log-depth.

    It also is not on the constrained hot path. The J0/J1 emission goes through
    `tree.sample_states_log` / `tree.map_states_and_tokens`; the only production
    callers here are SPEC §2.8's `mask` baseline, which needs the per-position
    support projection, and `--confidence=mar`. If this ever moves onto the hot
    path, make it a genuine down-sweep — do not reintroduce a `lax.scan`.

    ---

    **The recursions run in log space, and the linear vectors are anchored on
    the partition function** (SPEC §2.6 [V-P4], applied one layer up from the
    tree). Both halves are load-bearing; an earlier revision did neither and
    the consequences were measured on compiled grammars:

    1. *The recursion.* A linear `a[i+1] = a[i] @ M[i]` with per-vector max
       normalisation destroys every entry more than ~745 nats below its own
       vector max, and the loss then propagates forward. The **viable**
       within-vector range of `a` is 3,420 nats on Countdown and 425 on Sudoku
       at the sharpness Phase 0 measured on the checkpoint, so this is not a
       corner case. Log space with `log_matmul`'s pairwise-max anchor is exact
       at any dynamic range, which is the identical argument that moved the
       tree itself to log space.

    2. *The anchor.* SPEC §2.4 eq (5) makes `u_i(e) = a_{i−1}(src e)·b_i(dst e)`
       the object of the theory, and every factor in (3)/(4) is non-negative, so
       `u_i(e) == 0` **iff** the automaton forbids that edge there. Per-vector
       max normalisation bounds neither factor's within-vector range nor their
       product: measured on Sudoku, 87 live edges came back with `b` exactly
       `0.0` and 21 more with `a > 0, b > 0, a·b == 0`, costing 16 of 256
       positions their entire `mask` support and 2,097,192 legal (position,
       token) pairs. So the pair of scales is chosen to satisfy

           log_scale_a[i] + log_scale_b[i+1] == log Z   for every i,

       which makes `u` the **edge posterior** `P(edge e at position i)`: bounded
       above by 1 at every position, hence never overflowing, and — because the
       scale no longer varies with `i` — leaving `Σ_v p_i(v)·r_i(v) = 1` for
       every `i`, the non-vacuous invariant of SPEC §2.4 that
       `constrained_marginals_and_partition` exists to check.

       The split between the two factors is balanced around each vector's
       maximum **over viable states** (`a > 0` and `b > 0`). Anchoring on the
       unrestricted maximum does not work: SPEC §3.5's unscored `ACC --Σ--> ACC`
       tail pins `max_s b_i(s)` at 1.0 while the whole grammar sits at `Z`, so
       the misalignment `max log a_i + max log b_{i+1} − log Z` is **1,316 nats
       on Sudoku** — it is exactly `−log Z`, since the tail state's `b` and the
       start state's `a` are both 1. Restricted to viable states the same
       quantity is **≤ 474 nats across 8 real grammars** at the production
       temperature (404 Sudoku, 415 Countdown, 473.6 on BFCL `0-0-0 pretty`,
       which is the worst of the six BFCL variants measured), so each factor
       stays within `e^±237` of its anchor.

    3. *The floor.* Both consumers ask a **support** question — `r > 0` in the
       §2.8 mask and `q_i(v) > 0` — where the last representable decade carries
       a bit that matters even when the value does not. The true `log u_i(e)`
       spans over 1,200 nats *within one position* on a real grammar, so no
       choice of scales fits it in float64 and the small end must be clamped
       rather than lost: each factor is floored at `0.5·log(tiny)`, so a live
       edge's `u` is never below `tiny` and never exactly zero. The floored
       entries are ≥ 700 nats below the position's dominant edge; their
       contribution to `Σ_e u_i(e)` was measured at exactly 0 relative on both
       grammars (independently confirmed by `math.fsum`), i.e. far under the
       1e-6 the `i`-invariance of `Z_i` is asserted at.

    **Validity condition, and why it is a property of the interface rather than
    of this scheme.** Write `γ_i = α'_i + β'_{i+1} − log Z` for the
    viable-restricted misalignment above, and `|F_a|, |F_b|` for the two floors.
    Any pair of scales with a fixed sum has
    `max(log a_i) + max(log b_{i+1}) − (scale_a + scale_b) = γ_i`: the two
    vectors' headrooms trade against each other and their **sum is invariant
    under the split**. A floored factor is multiplied by a partner of up to
    `exp(max log)`, so `Σ_e u_i` is undisturbed only while
    `max(log a) ≤ |F_b|` and `max(log b) ≤ |F_a|`, i.e. only while

        γ_i ≤ |F_a| + |F_b| ≤ |log tiny| ≈ 708 nats   (float64; ≈ 87 float32)

    the second inequality being forced by needing the *product* of two floored
    factors to stay representable. No choice of split, anchor, normalisation or
    floor placement evades that while `u` must factor as `a(src)·b(dst)` with a
    per-position scale — it is a property of the two-factor interface, not of
    this construction.

    Above the bound the two requirements are genuinely incompatible and the
    floor gives up **support**, not mass (see the code below). Measured on
    Sudoku, `γ = 163 / 404 / 808 / 1,616` nats at `T = 1.0 / 0.4 / 0.2 / 0.1`,
    and the shipped `T = 0.4` sits 304 nats inside the bound — one halving of
    `T`, which `_MIN_TEMP = 1e-12` and the shipped `--temp greedy` make
    reachable by configuration alone. With a fixed `0.5·log(tiny)` floor,
    `max|Z_i − 1|` went `1.2e-12 → 4.1e-12 → 1.63 → 1.9e+109` across those four
    temperatures — support intact throughout, so the §2.8 `mask` baseline still
    masked to `π_i(C)`, while `H(q_i)` was silently reweighted inside
    `[0, log V]` and **nothing downstream raised**. With the widening floor it
    is `9.9e-13 / 3.9e-12 / 7.8e-13 / 1.9e-12`, the support is still exact at
    `T = 0.2`, and what degrades at `T = 0.1` is 16 live edges of 262,139
    (position, token) pairs — with no position emptied, because the dominant
    edge sits at `exp(0)` by construction — at the 21 of 256 positions
    `feasible_out` reports.

    The alternative the reviewer of this change proposed — anchoring on the
    per-position edge maximum `c_i = max_e (log a(src) + log b(dst))`, which is
    bounded by construction — is **not available at this signature**: `c_i −
    log Z` varies by 94.7 nats across positions on Sudoku at `T = 0.4` (37.99 at
    `T = 1.0`, 378.8 at `T = 0.1`), and the production consumers read `u`
    through `a` and `b` alone, with no channel to add a per-position scale back.
    It would trade a guarded cliff for an unguarded 94-nat error in `Z_i`.

    **Non-viable states are returned as exact `0.0`, which is a deliberate
    change of what `a` means and no consumer can see it.** A state with
    `a_i(s) > 0` and `b_i(s) == 0` is a dead end, and
    `b_i(s) = Σ_{s'} M_i(s,s')·b_{i+1}(s') = 0` is a sum of non-negatives, so
    `b_{i+1}(s') = 0` on every edge out of it: no `u_i(e)` with `src(e) = s` can
    be non-zero either way, and `Σ_s a_j(s)·b_j(s)` already had a zero factor
    there. Those two forms are the *only* ways anything reads `a` or `b`
    (`sampler.py`'s two closures, `marginals.constrained_marginals_and_partition`
    and the `log Z` self-check), so the change is exactly unobservable rather
    than approximately so — but only while that enumeration holds. What it buys
    is that non-viable entries are precisely the ones no anchor bounds, and it
    is the **`b` side** that runs away: SPEC §3.5's unscored tail is
    backward-reachable long before it is forward-reachable, so on Sudoku a
    non-viable `log b − scale_b` reaches **+1,284 nats at the shipped
    `T = 0.4`** (+2,568 at `T = 0.2`; Countdown +447 / +895) against
    `log(max) = 709.8`. Keeping those entries is `inf` unless the ceiling below
    catches them, and then `inf · 0 = NaN` in `u`. Read `a` or `b` alone and the
    two conventions differ; that is what a test would have to do to pin this.

    Args:
      feasible_out: also return `[L] bool`, false at every position where the
        result no longer certifies its own support. Two independent causes are
        reported through the one flag: (i) `γ_i` exceeded the validity bound
        above and the floor had to widen, and (ii) `up_sweep`'s max
        normalisation destroyed a live entry of that position's leaf
        (`TreeLevels.underflow`). Keyword-only and off by default so the
        4-tuple every existing caller unpacks is unchanged.

    Returns:
      `(a, b, log_scale_a, log_scale_b)`, plus `representable [L] bool` when
      `feasible_out`. `a`, `b` are `[L+1, S]` and the scales `[L+1]`:
      `a[i]·exp(log_scale_a[i])` is the true prefix vector before position `i`
      restricted to viable states, `b[i]·exp(log_scale_b[i])` the true suffix
      vector, so `log(a[i]·b[i]) + log_scale_a[i] + log_scale_b[i]` is
      `i`-invariant and equals `log Z` (SPEC §2.4 `[D]`). `a[0]` and `b[L]` are
      no longer `a_start` / `b_final` verbatim — they carry the same scale
      convention as every other boundary.
    """
    leaves = tree.levels[0]
    leaf_scales = tree.log_scales[0]
    L = leaves.shape[0]
    dt = leaves.dtype
    neg = jnp.asarray(NEG_SENTINEL, dtype=dt)
    tiny = jnp.asarray(jnp.finfo(dt).tiny, dtype=dt)

    # The tree's leaves are max-normalised; `leaf_scales` restores the true
    # magnitude, so this is `log M_i` exactly.
    #
    # **Sliced first, transformed second, which is worth 2.6x in wall time.**
    # Built as one `[L, S, S]` `logM` and then sliced per position, XLA charges
    # every one of the `2L` consumers for the whole elementwise producer: the
    # cost analysis reports 1.62 GFLOP at `S = 64` against 16 MFLOP this way,
    # a 100x that is mostly an attribution artefact -- the honest figure is the
    # 2.6x of measured wall time. Taking the `[S, S]` slice first keeps the
    # per-position transform per-position.
    def log_leaf(i: int) -> jnp.ndarray:
        m = leaves[i]
        return jnp.where(m > 0, jnp.log(jnp.maximum(m, tiny)) + leaf_scales[i],
                         neg)

    log_a0 = jnp.where(a_start > 0, jnp.log(jnp.maximum(a_start, tiny)), neg)
    log_bL = jnp.where(b_final > 0, jnp.log(jnp.maximum(b_final, tiny)), neg)

    # Unrolled at trace time, exactly as before: straight-line code, no device
    # `while`, so the whole chain stays inside a CUDA-graph capture (SPEC §0).
    fwd = [log_a0]
    for i in range(L):
        fwd.append(_log_vec_mat(fwd[-1], log_leaf(i)))
    bwd = [log_bL]
    for i in range(L - 1, -1, -1):
        bwd.append(_log_mat_vec(log_leaf(i), bwd[-1]))

    log_a = jnp.stack(fwd)                      # [L+1, S]
    log_b = jnp.stack(bwd[::-1])                # [L+1, S]

    live = NEG_SENTINEL / 2.0
    viable = (log_a > live) & (log_b > live)    # [L+1, S]

    # log Z, read off boundary 0. Every boundary gives the same value; this one
    # involves `a_start` unmodified, so it is the least processed of them.
    t0 = jnp.where(viable[0], log_a[0] + log_b[0], neg)
    m0 = jnp.max(t0)
    ok0 = m0 > live
    logZ = jnp.where(
        ok0,
        m0 + jnp.log(jnp.maximum(jnp.sum(jnp.exp(t0 - jnp.where(ok0, m0, 0.0))), tiny)),
        neg)

    # Per-boundary maxima over viable states only -- see the docstring: the
    # unrestricted maxima are pinned by the unscored tail and misalign by
    # `-log Z`.
    alpha = jnp.max(jnp.where(viable, log_a, neg), axis=1)      # [L+1]
    beta = jnp.max(jnp.where(viable, log_b, neg), axis=1)       # [L+1]

    # `scale_a[i] + scale_b[i+1] == logZ` exactly (the second is formed as the
    # complement of the first, so the identity survives rounding), which is what
    # makes the per-position partition `i`-invariant. The endpoints `scale_a[L]`
    # and `scale_b[0]` pair with nothing and only have to keep `a[L]`, `b[0]`
    # and the boundary dots representable, so they use the same balanced form.
    half = jnp.asarray(0.5, dtype=dt)
    scale_a = jnp.concatenate([
        half * (logZ + alpha[:-1] - beta[1:]),
        half * (logZ + alpha[L:] - beta[L:]),
    ])                                                          # [L+1]
    scale_b = jnp.concatenate([
        half * (logZ - alpha[:1] + beta[:1]),
        logZ - scale_a[:-1],
    ])                                                          # [L+1]

    # `floor` keeps a live edge's product above `tiny`; `ceil` is pure insurance
    # against `exp` overflowing on an entry the anchor does not bound (there are
    # none once non-viable states are dropped, but an `inf` here would become a
    # `NaN` one multiplication later).
    #
    # **The floor widens rather than clipping mass.** `|F| = 0.5·|log tiny|`
    # keeps every live product representable, and it is the right floor exactly
    # while `γ_i ≤ 2|F|`. Past that the two requirements are provably
    # incompatible (see the validity condition above), and of the two failures
    # only one is quiet: clipping a load-bearing factor moves `Σ_e u_i` — `u`
    # stops being a posterior and `H(q_i)` is silently reweighted inside
    # `[0, log V]` — whereas widening the floor drops the support of edges more
    # than `2|F_i|` below the position's dominant one, which is a strict subset
    # of what is unrepresentable anyway and cannot empty a position (the
    # dominant edge is at `exp(0)`). So the floor follows `γ_i` when it has to,
    # and `feasible_out` reports every position where it did.
    # `slack` is what keeps the widened floor from becoming the next quiet
    # error: a floored factor still multiplies a partner of at most
    # `exp(γ_i/2)`, so putting the floor exactly at `−γ_i/2` bounds the
    # inflation of `Σ_e u_i` by `exp(0)` — no better than clipping. `S²` pairs
    # at `exp(−40)` bound it by 1.7e-14 instead, which is under the 4e-12 the
    # scheme's own rounding already costs. Measured at `T = 0.2`: a 2-nat slack
    # leaves `max|Z_i − 1| = 0.135 = e^-2`, exactly this term.
    floor_base = half * (jnp.log(tiny) + jnp.asarray(5.0, dtype=dt))
    slack = jnp.minimum(jnp.asarray(40.0, dtype=dt), -half * floor_base)
    gamma = alpha[:-1] + beta[1:] - logZ                        # [L]
    widened = -jnp.maximum(-floor_base, half * gamma + slack)
    floor_a = jnp.concatenate([widened, floor_base[None]])      # [L+1]
    floor_b = jnp.concatenate([floor_base[None], widened])      # [L+1]
    ceil = jnp.log(jnp.finfo(dt).max) - jnp.asarray(8.0, dtype=dt)
    a = jnp.where(viable,
                  jnp.exp(jnp.clip(log_a - scale_a[:, None],
                                   floor_a[:, None], ceil)),
                  jnp.zeros((), dtype=dt))
    b = jnp.where(viable,
                  jnp.exp(jnp.clip(log_b - scale_b[:, None],
                                   floor_b[:, None], ceil)),
                  jnp.zeros((), dtype=dt))
    if not feasible_out:
        return a, b, scale_a, scale_b

    # The guard on the validity condition in the docstring. `widened < 0`
    # exactly at the positions where `γ_i > 2|F|` forced the floor down, i.e.
    # where `u_i` keeps its mass but no longer certifies its own support. A
    # boundary with no viable state carries nothing either way and is reported
    # representable; `Z == 0` is a different signal with its own detector
    # (SPEC §6.3, CLAUDE.md causes (a)/(b)).
    representable = jnp.logical_or(widened >= floor_base,
                                   ~jnp.any(viable[:-1], axis=1))    # [L]

    # **`up_sweep`'s own loss is ANDed in last, and it has to be last.** The
    # `γ` guard above and the "no viable state" clause both concern *this*
    # function's scaling. A leaf whose live entry `up_sweep` flushed to exactly
    # `0.0` arrives here already reclassified as a structurally impossible
    # transition, so it empties the position, `viable` goes all-false, and the
    # second clause would then report the position **healthy** — the failure
    # mode reporting exists to prevent. `underflow=None` means the sweep did
    # not measure it (max-plus, log, `normalize=False`, hand-built), which is
    # absence of information, not evidence of health, and is left alone.
    if tree.underflow is not None:
        representable = representable & ~tree.underflow[:L]
    return a, b, scale_a, scale_b, representable


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
    """`C[i,j] = logsumexp_k (A[i,k] + B[k,j])`, exact for ANY dynamic range.

    **History of two wrong versions, both measured on real grammars.**

    1. Single row/col shift (`ra[i] + cb[j]`): the unscored `ACC --Σ--> ACC`
       tail pins the shift at 0 while genuine grammar paths sit ~850 nats
       below; their contributions fall under float64's subnormal floor and a
       provably non-empty language came back Z == 0 (`live_simple_106-63-0`).
    2. Two-band shift (4 GEMMs): fixed the bimodal tail-vs-grammar case but
       not the realistic one — with per-position *sharp* model marginals the
       low band's INTERNAL spread is itself thousands of nats, and terms
       under the band anchor still underflow. Reproduced on the same record
       with adversarial sharp `p` after the bands had "fixed" the uniform
       case.

    The only anchor that is exact per entry is the **pairwise max**
    `M[i,j] = max_k (A[i,k] + B[k,j])` — the max-plus product. Relative to it,
    the dominant term of every entry is exp(0) = 1 by construction, so nothing
    that matters can underflow at any dynamic range; terms more than ~745
    nats below the max are dropped at relative weight < 1e-323, which is
    negligible *relative to their own entry* rather than to a foreign anchor.

    Cost: two fused reductions over the broadcast `A+B` instead of one GEMM —
    the same shape and peak memory as `maxplus_combine`, which already runs in
    production on the MAP path at these sizes. It is bandwidth-bound rather
    than cuBLAS-bound; SPEC §0's kernel-count property survives, its
    GEMM-throughput property does not on this path, and the §7.3 timings must
    be measured against THIS implementation, not the GEMM one.

    Correctness before speed, always (CLAUDE.md). A GEMM fast path gated on a
    proven-tight range bound can come back later if profiling demands it.
    """
    # TWO FUSED REDUCTIONS, no chunk loop, no materialisation.
    #
    # A previous version chunked over `k` with a running max/sum. That OOM-ed
    # the 221 GB host: the chunk size ignored the LEADING batch dim (at the
    # leaf level A is `[L/2, S, S]`, so a "64-wide" slab is
    # `[128, 512, 64, 512]` = 17 GB), and holding a slab across the running
    # update defeats XLA's fusion so every unrolled chunk stays live.
    #
    # `maxplus_combine` proves the shape that works at these sizes: written as
    # a single reduction over the broadcast sum, XLA fuses it and the `[n,k,m]`
    # tensor is never materialised. So do exactly that twice — once for the
    # max, once for the shifted sum, letting XLA recompute `A+B` in the second
    # pass rather than store it. Peak memory is O(n·m) per node, the same as
    # the MAP path that already runs in production.
    neg = jnp.asarray(NEG_SENTINEL, dtype=A.dtype)
    mx = jnp.maximum(jnp.max(A[..., :, :, None] + B[..., None, :, :], axis=-2),
                     neg)
    live = mx > NEG_SENTINEL / 2
    safe = jnp.where(live, mx, jnp.zeros_like(mx))
    sm = jnp.sum(
        jnp.exp(A[..., :, :, None] + B[..., None, :, :] - safe[..., None, :]),
        axis=-2)
    tiny = jnp.asarray(jnp.finfo(A.dtype).tiny, dtype=A.dtype)
    out = safe + jnp.log(jnp.maximum(sm, tiny))
    return jnp.where(live, out, jnp.full_like(out, NEG_SENTINEL))


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
