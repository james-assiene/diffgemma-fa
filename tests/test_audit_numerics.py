"""Audit of the numerical core, at production shapes. SPEC §2.4, §2.6, §2.7.

Written by the TESTER agent of the tester → coder → reviewer loop (CLAUDE.md).
It exists because this layer shipped **three** successive wrong `log_matmul`
kernels and one wrong dtype claim, and every one of them was caught by the
runtime `Z == 0` detector or by a live eval arm rather than by a test:

1. a single row/col max shift — underflowed when the unscored `ACC --Σ--> ACC`
   tail pins the shift at 0 while genuine grammar paths sit ~850 nats below;
2. a two-band shift — fixed that case but not per-position sharp `p`, where the
   low band's *internal* spread is itself thousands of nats;
3. a chunked streaming version — correct, but the chunk size ignored the
   leading batch dim and OOM-killed the host;

plus float32, declared safe on a toy check (`L = 4`, `|S| ≤ 8`) and then found
to drive spurious `Z == 0` on more than half of a 130-record arm at `L = 256`.

**The one property that would have caught all of them.** The tree's result must
equal a *straightforward sequential* float64 fold — matrices combined one at a
time, each entry anchored on its own running max — at the shapes production
actually runs. That reference is written here, in numpy, and shares no code
with the implementation; that is the entire point. Every kernel above is exact
on toy shapes and on a matrix with a benign dynamic range, so the shapes and the
regimes are load-bearing, not decoration. Kernels #1 and #2 are replayed below
(#2 verbatim, recovered from the 2026-07-29 working copy in the session
scratchpad) and the property is asserted to reject them, so this file cannot
quietly become a suite that passes on everything.

**Shapes.** `L = 256` (the real `canvas_length`, verified [V] in SPEC §1.2) and
`|S| ∈ {64, 128, 256}` — the first three rungs of SPEC §5.5's power-of-two
bucket ladder. Measured cost of the sequential reference on this CPU:
`|S| = 64` → 0.5 s, `128` → 4 s, `256` → 54 s. The `|S| = 256` rung therefore
runs in exactly one test rather than across the regime sweep; nothing is
silently shrunk, and any regime marked below as running at a smaller `|S|` says
why.

**What is deliberately NOT at production `V`.** `V = 262,144`, `L = 256` makes
one `[L, V]` float64 array 537 MB, and the marginal/scatter path chains several
of them; `joint_map` additionally materialises `[C, L, V]`. Those tests run at
`V ∈ {1024, 4096}` with mixed-polarity classes, which exercises every branch of
the complement-aware scatter — the `V`-dependence of the *arithmetic* is nil,
only the memory grows. Said out loud rather than left implicit (CLAUDE.md:
"never silently cap coverage").
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.infer import marginals as MG  # noqa: E402
from diffgemma_fa.infer import scans  # noqa: E402
from diffgemma_fa.model import constrained as C  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402

NEG = scans.NEG_SENTINEL
DEAD = NEG / 2.0          # "below this is the sentinel, not a real value"
L_PROD = 256              # the real canvas_length


# ===========================================================================
# The independent reference: a plain sequential float64 fold
# ===========================================================================

def sequential_log_fold(mats: np.ndarray) -> np.ndarray:
    """`logsumexp`-product of `mats[0] ⋯ mats[L-1]`, folded one at a time.

    Deliberately the dumbest correct thing: float64, `O(L)` sequential, each
    output entry anchored on its **own** running max over the contraction axis.
    No tree, no banding, no shared anchor, no JAX. Correctness before speed
    (CLAUDE.md) — this is the "reference" side of the differential test, and it
    must stay this boring.

    Args:
      mats: `[L, S, S]` log-space matrices, `NEG_SENTINEL` for impossible.

    Returns:
      `[S, S]`, `NEG_SENTINEL` where the product has no live path.
    """
    acc = np.asarray(mats[0], dtype=np.float64).copy()
    for i in range(1, mats.shape[0]):
        B = np.asarray(mats[i], dtype=np.float64)
        terms = acc[:, :, None] + B[None, :, :]           # [S, k, S]
        m = terms.max(axis=1)                             # per-entry anchor
        live = m > DEAD
        anchor = np.where(live, m, 0.0)
        s = np.exp(terms - anchor[:, None, :]).sum(axis=1)
        acc = np.where(live, anchor + np.log(np.maximum(s, 1e-320)), NEG)
    return acc


def sequential_maxplus_fold(mats: np.ndarray) -> np.ndarray:
    """`(max, +)` product of `mats[0] ⋯ mats[L-1]`, folded one at a time."""
    acc = np.asarray(mats[0], dtype=np.float64).copy()
    for i in range(1, mats.shape[0]):
        B = np.asarray(mats[i], dtype=np.float64)
        acc = np.maximum(np.max(acc[:, :, None] + B[None, :, :], axis=1), NEG)
    return acc


def assert_matches_reference(got: np.ndarray, ref: np.ndarray, *, what: str,
                             atol: float = 1e-6) -> None:
    """The audit property, stated once.

    Three separate failures, reported separately because they mean different
    things:

    * **lost** — the reference has a live entry the kernel returned as
      sentinel. This is the `Z == 0` bug: a provably non-empty language
      reported empty. All three historical kernels failed here.
    * **spurious** — the kernel invented a live entry. That is worse than
      losing one: it makes the feasibility detector report success on an empty
      language.
    * **value** — both live, numbers disagree. The two-band kernel failed here
      by ~885 nats while losing nothing, i.e. it would have drawn from a
      visibly wrong distribution with the detector silent.
    """
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    live_ref, live_got = ref > DEAD, got > DEAD
    lost = int((live_ref & ~live_got).sum())
    spurious = int((~live_ref & live_got).sum())
    assert lost == 0, (
        f"{what}: {lost}/{int(live_ref.sum())} entries the sequential float64 "
        f"reference keeps came back at the sentinel. That is a spurious "
        f"`Z == 0`: a provably non-empty language reported empty. Lowest live "
        f"reference entry was {ref[live_ref].min():.1f} nats."
    )
    assert spurious == 0, (
        f"{what}: {spurious} entries are live in the kernel but dead in the "
        f"reference — the feasibility detector would pass on an empty language."
    )
    both = live_ref & live_got
    if both.any():
        err = float(np.abs(got[both] - ref[both]).max())
        assert err <= atol, (
            f"{what}: max |kernel − reference| = {err:.4g} nats over "
            f"{int(both.sum())} live entries (tol {atol:g}). The values are "
            f"log-probabilities, so this is a wrong distribution, not noise."
        )


# ===========================================================================
# Adversarial regimes — the structures that actually broke things
# ===========================================================================

def regime_matrices(name: str, S: int, L: int = L_PROD, seed: int = 3) -> np.ndarray:
    """`[L, S, S]` log-space transition matrices for one adversarial regime.

    Every regime except `dense_random` has the shape a compiled grammar
    actually has (SPEC §3.5 trap 4):

      * states `0 .. S-3` are the grammar,
      * state `S-2` is "grammar complete", the only source of the stop token,
      * state `S-1` is `ACC` with the **unscored** `ACC --Σ--> ACC` self-loop
        at log-weight exactly `0.0`.

    That self-loop is what makes this hard: it pins every row/column/global max
    at 0 while the grammar's own paths sit hundreds to thousands of nats below,
    so any anchor that is not per-entry underflows the grammar away.
    """
    rng = np.random.default_rng(seed)
    ACC, DONE = S - 1, S - 2
    M = np.full((L, S, S), NEG, dtype=np.float64)

    if name == "dense_random":
        # No sentinels, no tail: a plain accuracy check over a wide range, so
        # a failure elsewhere cannot be blamed on the sentinel bookkeeping.
        return rng.normal(size=(L, S, S)) * 50.0

    deg = 3
    succ = {s: list(rng.choice(S - 2, size=min(deg, S - 2), replace=False))
            for s in range(S - 2)}
    for s in range(S - 2):
        succ[s].append((s + 1) % (S - 2))          # keep it strongly connected

    for i in range(L):
        for s in range(S - 2):
            ts = succ[s]
            if name == "tail_vs_grammar":
                # 9-16 nats per grammar step, i.e. per-token probabilities
                # around `4e-6` — SPEC §2.6's own measured figure for "a
                # genuine constrained path" on a real BFCL grammar. Over 256
                # positions that puts the grammar ~2,800 nats below the free
                # tail, and the half-products ~1,400 below, which is what
                # takes a foreign (row/column) anchor past float64's ~745-nat
                # exp floor. A milder cost — 4-9 nats, net ~5 after the
                # branching sum — lands at ~620 nats and a single-shift kernel
                # SURVIVES it: the regime, not just the shape, is load-bearing.
                # `test_the_regimes_actually_span_the_dynamic_range_they_claim`
                # asserts the gap directly so this cannot silently regress.
                for t in ts:
                    M[i, s, t] = -rng.uniform(9.0, 16.0)
            elif name == "sharp_per_position":
                # Near-one-hot `p`: one cheap continuation, the rest hundreds
                # of nats down. `_MIN_TEMP = 1e-12` (CLAUDE.md) makes this
                # reachable by configuration alone, and it is the regime that
                # kills a band anchor — the low band's internal spread is
                # itself thousands of nats.
                k = int(rng.integers(len(ts)))
                for j, t in enumerate(ts):
                    M[i, s, t] = (-rng.uniform(0.0, 2.0) if j == k
                                  else -rng.uniform(400.0, 1200.0))
            else:
                raise ValueError(name)
            if s == 0:
                M[i, s, DONE] = -rng.uniform(3.0, 6.0)
        M[i, DONE, ACC] = -rng.uniform(1.0, 3.0)
        M[i, ACC, ACC] = 0.0                        # unscored post-stop tail
    return M


def modular_counting_matrices(n_cycle: int, S: int, L: int = L_PROD) -> np.ndarray:
    """A cyclic counter: state `s → (s+1) mod n_cycle`, everything else dead.

    Used for the empty-language case. After `L` steps the only reachable state
    is `L mod n_cycle`, so restricting the terminal factor to state 0 makes the
    language **genuinely empty** whenever `L % n_cycle != 0` — cause (a)/(b) of
    CLAUDE.md's `Z == 0` taxonomy, which must be reported, not swallowed.
    """
    M = np.full((L, S, S), NEG, dtype=np.float64)
    for i in range(L):
        for s in range(n_cycle):
            M[i, s, (s + 1) % n_cycle] = -1.0
    return M


# ===========================================================================
# 1. The killer property: tree == sequential float64 reference
# ===========================================================================

@pytest.mark.parametrize("regime", ["tail_vs_grammar", "sharp_per_position",
                                    "dense_random"])
@pytest.mark.parametrize("S", [64, 128])
def test_log_tree_root_matches_sequential_float64_reference(regime, S):
    """SPEC §2.6. `up_sweep_log`'s root **is** `M_0 ⋯ M_255` in log space.

    Nothing here knows how the tree is built. Any anchor that is not per-entry
    — one shift, two bands, a chunk-local running max — shows up as lost
    entries or as nats of error, at `L = 256`, in the regimes that occur.
    """
    M = regime_matrices(regime, S)
    ref = sequential_log_fold(M)
    root = np.asarray(scans.up_sweep_log(jnp.asarray(M)).root)
    assert_matches_reference(root, ref, what=f"root[{regime}, S={S}, L=256]")


def test_log_tree_root_matches_sequential_reference_at_S256():
    """The third bucket rung. Split out because the numpy reference costs ~54 s
    at `|S| = 256`; running it across the whole regime sweep would put minutes
    into the suite for no extra coverage of the anchor question.
    """
    M = regime_matrices("tail_vs_grammar", 256)
    ref = sequential_log_fold(M)
    root = np.asarray(scans.up_sweep_log(jnp.asarray(M)).root)
    assert_matches_reference(root, ref, what="root[tail_vs_grammar, S=256]")


def test_the_regimes_actually_span_the_dynamic_range_they_claim():
    """Guard on the guard, and the sharpest statement in this file.

    A regime whose entries all sit within ~700 nats of the tail cannot
    discriminate any of the historical kernels, and the suite above would then
    be vacuous — passing on everything, which is the tester's own failure mode
    that CLAUDE.md calls out.

    Two things are asserted, and the second is the one that matters:

    1. the unscored `ACC --Σ--> ACC` tail pins the maximum at exactly `0.0`;
    2. **the foreign-anchor gap exceeds float64's ~745-nat `exp` floor.** For
       the final combine `root = A @ B`, a row/column-anchored kernel
       exponentiates against `max_k A[i,k] + max_k B[k,j]`. Where that anchor
       sits more than 745 nats above the entry's true value, every term
       underflows to zero and the entry is lost. Measuring the gap directly
       certifies that *any* non-per-entry anchor fails here, without needing a
       copy of the kernel — so it stays a valid guard against the fourth wrong
       `log_matmul`, whatever shape it takes.
    """
    for regime in ("tail_vs_grammar", "sharp_per_position"):
        M = regime_matrices(regime, 64)
        ref = sequential_log_fold(M)
        live = ref > DEAD
        assert float(ref[live].max()) == pytest.approx(0.0, abs=1e-9), (
            f"{regime}: the unscored ACC tail should pin the max at exactly 0"
        )
        halves = scans.up_sweep_log(jnp.asarray(M)).levels[7]
        A, B = np.asarray(halves[0]), np.asarray(halves[1])
        anchor = A.max(axis=1)[:, None] + B.max(axis=0)[None, :]
        gap = float(np.max(np.where(live, anchor - ref, -np.inf)))
        assert gap > 745.0, (
            f"{regime}: the largest foreign-anchor overshoot is only "
            f"{gap:.0f} nats. Below float64's ~745-nat exp floor a row/column "
            f"anchor still produces the right answer, so this regime cannot "
            f"discriminate the kernel that shipped."
        )


def test_every_retained_tree_level_matches_its_reference_block_product():
    """SPEC §2.6(b)/eq (7): the retained nodes are the **aligned dyadic**
    products `P_{[j·2^k, (j+1)·2^k)}`, and eq (7)'s midpoint conditional reads
    them directly. A root that is right while an interior node is wrong gives
    a correct `Z` and a wrong *sample* — which no `Z == 0` detector can see.
    """
    S = 64
    M = regime_matrices("tail_vs_grammar", S)
    tr = scans.up_sweep_log(jnp.asarray(M))
    assert tr.n_nodes == 2 * L_PROD - 1, (
        f"{tr.n_nodes} nodes, expected 2L-1 = {2 * L_PROD - 1}: the shape is "
        f"not Blelloch (Kogge-Stone would materialise L·log2 L = 2048)"
    )
    rng = np.random.default_rng(0)
    for k in (0, 3, 5, 7, 8):
        width = 1 << k
        n_nodes = L_PROD // width
        for j in sorted(rng.choice(n_nodes, size=min(3, n_nodes), replace=False)):
            ref = sequential_log_fold(M[j * width:(j + 1) * width])
            got = np.asarray(tr.levels[k][j])
            assert_matches_reference(
                got, ref, what=f"level {k} node {j} = P_[{j*width},{(j+1)*width})")


def test_the_tree_keeps_the_left_to_right_matrix_order():
    """SPEC §2.6(c). `lax.associative_scan(..., reverse=True)` yields
    `f(f(z,y),x)`, i.e. the operands swapped — for non-commutative matmul that
    is silently wrong probabilities and **no shape error**.

    Detected by making the reversed product a genuinely different matrix and
    asserting the tree is far from it, as well as close to the forward one.
    """
    S = 64
    M = regime_matrices("tail_vs_grammar", S)
    fwd = sequential_log_fold(M)
    rev = sequential_log_fold(M[::-1])
    root = np.asarray(scans.up_sweep_log(jnp.asarray(M)).root)
    assert_matches_reference(root, fwd, what="root vs forward order")
    live = (rev > DEAD) & (fwd > DEAD)
    assert live.any(), "degenerate fixture: reversed product has no live entry"
    gap = float(np.abs(rev[live] - fwd[live]).max())
    assert gap > 1.0, (
        f"fixture is order-blind (max forward/reverse gap {gap:.3g} nats), so "
        f"this test could not detect an operand swap"
    )


def test_log_matmul_single_combine_is_exact_over_a_5000_nat_span():
    """One combine, checked entry by entry against a per-entry `logsumexp`.

    Includes fully-dead rows and columns so the sentinel bookkeeping is
    exercised: SPEC §2.7 mandates a **finite** sentinel precisely so fused
    kernels cannot make `NaN` out of `-inf + -inf`, and a kernel that returns
    `-inf` or `NaN` for a dead entry poisons everything downstream.
    """
    rng = np.random.default_rng(11)
    n, k, m = 48, 48, 48
    A = rng.uniform(-5000.0, 0.0, size=(n, k))
    B = rng.uniform(-5000.0, 0.0, size=(k, m))
    A[3, :] = NEG
    B[:, 5] = NEG
    A[:, 7] = NEG
    got = np.asarray(scans.log_matmul(jnp.asarray(A), jnp.asarray(B)))
    assert np.isfinite(got).all(), "log_matmul produced NaN/inf, not a sentinel"

    ref = np.full((n, m), NEG)
    for i in range(n):
        for j in range(m):
            t = np.array([A[i, kk] + B[kk, j] for kk in range(k)
                          if A[i, kk] > DEAD and B[kk, j] > DEAD])
            if t.size:
                mx = t.max()
                ref[i, j] = mx + np.log(np.exp(t - mx).sum())
    assert_matches_reference(got, ref, what="single log_matmul combine",
                             atol=1e-9)


def test_linear_prefix_suffix_recovers_the_log_space_partition_at_L256():
    """SPEC §2.4. `prefix_suffix` is the one production path still in **linear**
    space (it feeds §2.8's `mask` baseline), with per-vector max normalisation
    and accumulated log-scales.

    Two things at once, both at `L = 256` with the unscored tail present:
    `log(a_i · b_i) + logscale_a[i] + logscale_b[i]` must be `i`-invariant
    (SPEC marks this `[D]` — "assert it across all `i`"), and its common value
    must equal the log-space tree's `log Z`. Self-consistency alone would be
    satisfied by a uniformly scaled-wrong `a`/`b`; anchoring to the tree is
    what makes it a differential test.

    It also pins the suffix pass's operand order: `b_{i-1} = M_i b_i`, which is
    the direction `lax.associative_scan(..., reverse=True)` gets backwards
    (SPEC §2.6(c)).
    """
    S = 64
    logM = regime_matrices("tail_vs_grammar", S)
    M = np.where(logM > DEAD, np.exp(np.minimum(logM, 0.0)), 0.0)
    a_start = np.zeros(S)
    a_start[0] = 1.0
    b_final = np.zeros(S)
    b_final[S - 1] = 1.0

    tree_lin = scans.up_sweep(jnp.asarray(M))
    a, b, la, lb = scans.prefix_suffix(tree_lin, jnp.asarray(a_start),
                                       jnp.asarray(b_final))
    a, b = np.asarray(a), np.asarray(b)
    dot = (a * b).sum(axis=1)
    assert np.all(dot > 0), (
        "a_i · b_i underflowed to zero at some position: the per-vector max "
        "normalisation is pinned by the unscored tail while the grammar decays"
    )
    logZ_i = np.log(dot) + np.asarray(la) + np.asarray(lb)
    spread = float(logZ_i.max() - logZ_i.min())
    assert spread < 1e-8, (
        f"log Z varies by {spread:.3g} nats across the 257 boundaries"
    )

    ref = sequential_log_fold(logM)
    truth = float(ref[0, S - 1])
    assert float(logZ_i[0]) == pytest.approx(truth, abs=1e-8), (
        f"prefix/suffix log Z = {logZ_i[0]:.6f} but the sequential float64 "
        f"reference says {truth:.6f}"
    )


# ===========================================================================
# 2. Does the property have teeth? Replay the historical kernels.
# ===========================================================================
#
# A test that passes on the broken implementation is worse than no test — it is
# the specific failure CLAUDE.md warns the tester about ("`Σ_v q_i(v) == 1` was
# asserted for weeks and was mathematically incapable of failing"). So the two
# historical kernels are replayed here and the audit property is asserted to
# REJECT them. Kernel #3 (the chunked streaming form) is not replayed: its
# defect was peak host memory, not arithmetic, and no assertion about values
# can see it — see the report's "not covered" list.

def _historical_single_shift(A, B):
    """Kernel #1, exactly as documented in `scans.log_matmul`'s docstring:
    exponentiate against `ra[i] + cb[j]`, the row max of `A` plus the column
    max of `B`."""
    ra = jnp.max(A, axis=-1)
    cb = jnp.max(B, axis=-2)
    sh = ra[..., :, None] + cb[..., None, :]
    live = sh > DEAD
    safe = jnp.where(live, sh, 0.0)
    s = jnp.exp(A[..., :, :, None] + B[..., None, :, :]
                - safe[..., :, None, :]).sum(axis=-2)
    out = safe + jnp.log(jnp.maximum(s, jnp.finfo(A.dtype).tiny))
    return jnp.where(live & (s > 0), out, NEG)


#: `_BAND` as it stood in kernel #2.
_HISTORICAL_BAND = 350.0


def _historical_band_shift(A, B):
    """Kernel #2 — **verbatim**, not a paraphrase.

    Recovered from `scans.py` as it stood on 2026-07-29 (the working copy kept
    in the session scratchpad; the repository has a single commit, so `git log`
    cannot supply it). Each operand is split into a high band, within
    `_BAND = 350` nats of its row/column max, and a low band shifted by its own
    max; the four shifted GEMMs are recombined with an exact logsumexp.

    Its stated justification — "each pairwise band sum spans ≤ 2·_BAND < 745
    nats, so no contribution underflows" — is true of the *bands* and false of
    the entries: the low band's own internal spread is unbounded, and with
    per-position sharp `p` it is itself thousands of nats.
    """
    ra = jnp.max(A, axis=-1, keepdims=True)
    cb = jnp.max(B, axis=-2, keepdims=True)
    ra = jnp.where(ra > DEAD, ra, jnp.zeros_like(ra))
    cb = jnp.where(cb > DEAD, cb, jnp.zeros_like(cb))

    def _bands(X, m, axis):
        hi = jnp.where(X > m - _HISTORICAL_BAND, X, NEG)
        lo = jnp.where(X <= m - _HISTORICAL_BAND, X, NEG)
        ml = jnp.max(lo, axis=axis, keepdims=True)
        ml = jnp.where(ml > DEAD, ml, jnp.zeros_like(ml))
        return (jnp.exp(hi - m), m), (jnp.exp(lo - ml), ml)

    (Ah, sa_h), (Al, sa_l) = _bands(A, ra, axis=-1)
    (Bh, sb_h), (Bl, sb_l) = _bands(B, cb, axis=-2)

    tiny = jnp.asarray(jnp.finfo(A.dtype).tiny, dtype=A.dtype)
    parts = []
    for X, sx in ((Ah, sa_h), (Al, sa_l)):
        for Y, sy in ((Bh, sb_h), (Bl, sb_l)):
            p = X @ Y
            v = sx + sy + jnp.log(jnp.maximum(p, tiny))
            parts.append(jnp.where(p > 0, v, jnp.full_like(v, NEG)))
    stacked = jnp.stack(parts)
    m = jnp.max(stacked, axis=0)
    safe_m = jnp.where(m > DEAD, m, jnp.zeros_like(m))
    out = safe_m + jnp.log(jnp.sum(jnp.exp(stacked - safe_m[None]), axis=0))
    return jnp.where(m > DEAD, out, jnp.full_like(out, NEG))


def _up_sweep_with(combine, logM):
    cur = jnp.asarray(logM)
    while cur.shape[0] > 1:
        cur = combine(cur[0::2], cur[1::2])
    return np.asarray(cur[0])


@pytest.mark.parametrize("kernel,regime", [
    (_historical_single_shift, "tail_vs_grammar"),
    (_historical_single_shift, "sharp_per_position"),
    (_historical_band_shift, "sharp_per_position"),
])
def test_the_audit_property_rejects_the_historical_kernels(kernel, regime):
    """Each of these shipped and was caught in production, not by a test.

    If this test starts passing trivially — i.e. a historical kernel is no
    longer rejected — the regime has drifted and the tests above have stopped
    proving anything.
    """
    M = regime_matrices(regime, 64)
    ref = sequential_log_fold(M)
    got = _up_sweep_with(kernel, M)
    with pytest.raises(AssertionError):
        assert_matches_reference(got, ref, what="historical kernel")


def test_the_cheap_2x2_witness_separates_the_kernels():
    """The `L = 256` sweep above is the real guard, but a 2×2 witness that
    actually discriminates is worth keeping — it runs in microseconds and names
    the mechanism.

    Replaces a test that asserted the *opposite*: that
    `tests/test_numerics.py`'s old `[[-423, -423], [NEG, 0]]` repro could not
    tell the kernels apart. It could not, and that test pinned the hole open
    rather than closing it. `test_numerics.py` now carries this witness as its
    own assertion, so what is checked here is only that the two kernels really
    do diverge on it — i.e. that the cheap guard has teeth.

    `ra[0] = 0` and `cb[0] = 0` put the foreign anchor at 0, while the only
    live path is `A[0,1] + B[1,0] = -900`; `exp(-900)` is zero in float64.
    """
    A = jnp.asarray([[0.0, -900.0], [NEG, NEG]], dtype=jnp.float64)
    B = jnp.asarray([[NEG, NEG], [0.0, NEG]], dtype=jnp.float64)
    current = float(scans.log_matmul(A, B)[0, 0])
    broken = float(_historical_single_shift(A, B)[0, 0])
    assert current == pytest.approx(-900.0, abs=1e-9), (
        f"the current kernel got {current}, expected -900"
    )
    assert broken <= DEAD, (
        f"the single-shift kernel got {broken} rather than the sentinel — this "
        f"witness has stopped discriminating and test_numerics.py's cheap "
        f"guard is vacuous again"
    )


# ===========================================================================
# 3. max-plus / MAP tree. SPEC §2.7.
# ===========================================================================

@pytest.mark.parametrize("regime", ["tail_vs_grammar", "sharp_per_position"])
@pytest.mark.parametrize("S", [64, 128])
def test_maxplus_tree_root_matches_sequential_float64_reference(regime, S):
    """`(max, +)` is a semiring, so the identical tree must give the identical
    answer as an `O(L)` fold. MAP is the default emission, so this path carries
    production traffic."""
    M = regime_matrices(regime, S)
    ref = sequential_maxplus_fold(M)
    root = np.asarray(scans.up_sweep_maxplus(jnp.asarray(M)).root)
    assert_matches_reference(root, ref, what=f"maxplus root[{regime}, S={S}]")


def test_maxplus_in_float32_does_not_saturate_the_sentinel_to_minus_inf():
    """SPEC §2.7's finite sentinel exists so `NEG + NEG` cannot become `NaN`.

    But `-3e38 + -3e38 = -6e38` **overflows float32 to `-inf`** at the very
    first combine, so `maxplus_combine`'s re-clamp is load-bearing rather than
    cosmetic — without it the sentinel discipline is gone by level 1 and a
    later `-inf + inf` is a `NaN`. Checked at `L = 256`, over 8 combine levels.
    """
    M = regime_matrices("tail_vs_grammar", 64)
    tr = scans.up_sweep_maxplus(jnp.asarray(M, dtype=jnp.float32))
    for k, lvl in enumerate(tr.levels):
        arr = np.asarray(lvl, dtype=np.float64)
        assert np.isfinite(arr).all(), (
            f"level {k} contains NaN/inf: the finite sentinel saturated "
            f"(NEG + NEG = -6e38 overflows float32 to -inf)"
        )
        assert arr.min() > 1.5 * NEG, (
            f"level {k} reached {arr.min():.3g}, below the sentinel — the "
            f"re-clamp in maxplus_combine is missing"
        )
    ref = sequential_maxplus_fold(M)
    assert_matches_reference(np.asarray(tr.root, dtype=np.float64), ref,
                             what="maxplus float32 root", atol=1e-2)


# ===========================================================================
# 4. Genuinely empty languages must be reported, never sampled from
# ===========================================================================

def test_empty_language_root_is_all_sentinel_at_L256():
    """`256 % 63 != 0`, so no length-256 path returns the counter to state 0
    and the (0, 0) entry has no support. `jax.random.categorical` and `argmax`
    are shift-invariant, so an all-sentinel row is indistinguishable from a
    uniform one — this must be visible in the numbers, not inferred later.
    """
    S = 64
    M = modular_counting_matrices(63, S)
    ref = sequential_log_fold(M)
    root = np.asarray(scans.up_sweep_log(jnp.asarray(M)).root)
    assert_matches_reference(root, ref, what="empty-language root")
    assert float(root[0, 0]) <= DEAD, (
        "a length-256 path from state 0 back to state 0 in a 63-cycle does "
        "not exist, but the kernel reports one"
    )
    assert float(ref[0, 4]) > DEAD, "fixture broken: 256 mod 63 == 4 is live"


# ---------------------------------------------------------------------------
# The same question one level up, through `model/constrained.py`.
# ---------------------------------------------------------------------------

def _cycle_automaton(n_cycle: int, s_bucket: int, vocab: int) -> Automaton:
    """A traced `Automaton`: an `n_cycle` counter with **mixed class polarity**.

    Even states emit even tokens (a *positive* class storing the evens); odd
    states emit odd tokens (a *negated* class storing the same evens, i.e.
    meaning their complement). Both polarities therefore ride the same CSR
    payload, which is the configuration SPEC §2.4 says only the exactness suite
    catches: a plain sparse scatter gets the negated class silently wrong.

    Accepting state is 0, so `d(s) = (n_cycle − s) mod n_cycle` and with
    `remaining = 0` the terminal factor `1[d(s) ≤ R]` admits state 0 alone.
    """
    evens = np.arange(0, vocab, 2, dtype=np.int32)
    src = np.arange(n_cycle, dtype=np.int32)
    dst = ((src + 1) % n_cycle).astype(np.int32)
    cls = (src % 2).astype(np.int32)                    # 0 = positive, 1 = negated
    d = np.full(s_bucket, 10 ** 6, dtype=np.int32)
    d[:n_cycle] = [(n_cycle - s) % n_cycle for s in range(n_cycle)]
    is_final = np.zeros(s_bucket, dtype=bool)
    is_final[0] = True
    active = np.zeros(s_bucket, dtype=bool)
    active[0] = True
    return Automaton(
        edge_src=jnp.asarray(src), edge_dst=jnp.asarray(dst),
        edge_class=jnp.asarray(cls),
        edge_valid=jnp.ones(n_cycle, dtype=bool),
        csr_indices=jnp.asarray(np.concatenate([evens, evens])),
        csr_indptr=jnp.asarray(np.array([0, evens.size, 2 * evens.size],
                                        dtype=np.int32)),
        is_neg=jnp.asarray(np.array([False, True])),
        d=jnp.asarray(d), is_final=jnp.asarray(is_final),
        active=jnp.asarray(active),
    )


def _cycle_accepts(tokens: np.ndarray, n_cycle: int) -> bool:
    """Independent simulator for `_cycle_automaton`, written from the spec of
    the fixture rather than from any code under test."""
    s = 0
    for tok in np.asarray(tokens):
        want_even = (s % 2) == 0
        if bool(int(tok) % 2 == 0) != want_even:
            return False
        s = (s + 1) % n_cycle
    return s == 0


def _sharp_marginals(L: int, V: int, seed: int = 0, scale: float = 8.0,
                     dtype=jnp.float64) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((L, V)) * scale
    return jax.nn.softmax(jnp.asarray(logits, dtype=dtype), axis=-1)


def test_joint_draw_reports_infeasible_on_a_genuinely_empty_language():
    """SPEC §6.3 cause (a)/(b). Measured historically: `valid == True` on
    200/200 draws from a provably empty language, because `valid` only asks
    whether the drawn state path traverses existing edges. The returned flag
    must be the root-mass predicate, not that."""
    S, V, n = 64, 1024, 63                      # 256 % 63 != 0 -> empty
    aut = _cycle_automaton(n, S, V)
    p = _sharp_marginals(L_PROD, V)
    _tokens, ok = C.joint_draw(p, aut, jnp.asarray(0), jax.random.key(0),
                               n_states=S, n_classes=2)
    assert not bool(ok), (
        "joint_draw reported a feasible draw from a language with no "
        "length-256 string: the Z == 0 detector is not firing"
    )


def test_joint_map_reports_infeasible_on_a_genuinely_empty_language():
    """The same, through the max-plus path — whose `score` used to be
    discarded by the caller and is a perfect detector."""
    S, V, n = 64, 1024, 63
    aut = _cycle_automaton(n, S, V)
    p = _sharp_marginals(L_PROD, V)
    _tokens, ok = C.joint_map(p, aut, jnp.asarray(0), n_states=S, n_classes=2)
    assert not bool(ok), "joint_map reported MAP support on an empty language"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_joint_draw_is_feasible_and_accepted_on_a_non_empty_language(seed):
    """The converse, without which the two tests above are satisfied by a
    detector that always says False.

    `256 % 64 == 0`, so the language is non-empty; the draw must be reported
    feasible **and** the emitted tokens must satisfy an independent simulator.
    Mixed class polarity throughout, at `L = 256`.
    """
    S, V, n = 64, 1024, 64
    aut = _cycle_automaton(n, S, V)
    p = _sharp_marginals(L_PROD, V, seed=seed)
    tokens, ok = C.joint_draw(p, aut, jnp.asarray(0), jax.random.key(seed),
                              n_states=S, n_classes=2)
    assert bool(ok), "spurious Z == 0 on a language with 2^256 members"
    assert _cycle_accepts(np.asarray(tokens), n), (
        "the emitted canvas is not in the language"
    )


def test_joint_map_is_feasible_and_accepted_on_a_non_empty_language():
    S, V, n = 64, 1024, 64
    aut = _cycle_automaton(n, S, V)
    p = _sharp_marginals(L_PROD, V, seed=7)
    tokens, ok = C.joint_map(p, aut, jnp.asarray(0), n_states=S, n_classes=2)
    assert bool(ok), "spurious Z == 0 on the MAP path"
    assert _cycle_accepts(np.asarray(tokens), n)


def test_joint_map_beats_every_perturbation_of_itself():
    """MAP must be the **argmax**, not merely a member of the language — a
    feasible-but-suboptimal decode passes every acceptance test in the suite.

    Checked by single-position perturbation: swapping any one token for the
    best alternative that keeps the string in the language must not raise the
    score. That is a necessary condition for a global argmax and is computable
    without enumerating `V^L`.
    """
    S, V, n = 64, 1024, 64
    aut = _cycle_automaton(n, S, V)
    p = _sharp_marginals(L_PROD, V, seed=5)
    tokens, ok = C.joint_map(p, aut, jnp.asarray(0), n_states=S, n_classes=2)
    assert bool(ok)
    logp = np.log(np.asarray(p))
    toks = np.asarray(tokens)
    # Position `i` sits at cycle state `i % n`; the admissible tokens there are
    # the evens (even state) or the odds (odd state), independent of the rest.
    for i in range(L_PROD):
        admissible = np.arange(0 if (i % n) % 2 == 0 else 1, V, 2)
        best = admissible[np.argmax(logp[i, admissible])]
        assert logp[i, toks[i]] >= logp[i, best] - 1e-9, (
            f"position {i}: MAP chose log p = {logp[i, toks[i]]:.4f} when "
            f"{logp[i, best]:.4f} was admissible — this is not the argmax"
        )


# ===========================================================================
# 5. The dtype claim. `require_x64` is a no-op; that is a numerical claim.
# ===========================================================================

def test_float32_at_production_scale_is_either_shown_safe_or_refused():
    """`require_x64` was made a no-op on a toy check and the claim did not hold.

    The claim it encoded was "float32 is safe on the sampling path". This test
    measures that claim where it has to hold — a `[256, V]` sharp `p` run
    through the real class-table → `M` → `up_sweep_log` chain — and asserts
    **both** halves unconditionally:

      1. float32 *is* lossy here, so the fixture cannot silently stop
         exercising the hazard;
      2. `require_x64()` refuses when x64 is off, rather than letting the
         sampler draw from a degenerate distribution.

    **[Corrected 2026-08-08.]** This used to return early when (1) came back
    clean — "the no-op is justified" — which meant a drifting fixture would
    make the test pass while asserting nothing *and* silently drop its guard on
    (2). An escape hatch that disables the assertion it guards is not a
    two-outcome contract, it is a hole.
    """
    S, V, n = 64, 4096, 64
    aut = _cycle_automaton(n, S, V)

    def log_root(dtype):
        p = _sharp_marginals(L_PROD, V, seed=1, scale=12.0, dtype=dtype)
        _p_vl, _W, M = C._matrices(p, aut, S, 2)
        tiny = jnp.finfo(dtype).tiny
        logM = jnp.where(M > 0, jnp.log(jnp.maximum(M, tiny)),
                         jnp.asarray(NEG, dtype))
        return np.asarray(scans.up_sweep_log(logM).root, dtype=np.float64)

    r64 = log_root(jnp.float64)
    r32 = log_root(jnp.float32)
    live64, live32 = r64 > DEAD, r32 > DEAD
    lost = int((live64 & ~live32).sum())
    both = live64 & live32
    err = float(np.abs(r64[both] - r32[both]).max()) if both.any() else np.inf

    # UNCONDITIONAL. A previous version returned early when the fixture came
    # back clean, which made the test pass while asserting nothing AND stop
    # protecting the guard — the failure mode this file exists to prevent.
    # Both halves are now assertions: the fixture must be lossy, and the guard
    # must refuse.
    assert lost > 0 or err > 1e-2, (
        f"the fixture is no longer lossy in float32 (lost {lost}, err "
        f"{err:.3g} nats), so the second half of this test would be vacuous. "
        f"Either the leaf construction was fixed to build `M` in float64 — in "
        f"which case retarget this at whatever still runs in float32 — or the "
        f"sharpness of `p` has drifted below the regime real grammars occupy."
    )

    jax.config.update("jax_enable_x64", False)
    raised = False
    try:
        C.require_x64()
    except C.X64Required:
        raised = True
    finally:
        jax.config.update("jax_enable_x64", True)

    assert raised, (
        f"float32 is measurably lossy at L = 256: {lost}/{int(live64.sum())} "
        f"live root entries vanish and the survivors are off by {err:.3g} nats "
        f"— a spurious `Z == 0` on every one of those state pairs, which is "
        f"the 70/130 failure `require_x64`'s own docstring records. But "
        f"`require_x64()` returns silently with x64 disabled, so nothing stops "
        f"the sampler running there. The loss is in the LEAF construction "
        f"(`_matrices` -> `transition_matrices`), upstream of the tree: `p` "
        f"entries below float32's 1.18e-38 flush to zero and take their edges "
        f"with them, which the pairwise-max anchor never addressed."
    )


# ===========================================================================
# 6. Constrained marginals and the class tables. SPEC §2.4, §4.4.
# ===========================================================================

def _mixed_polarity_tables(V: int, n_classes: int, seed: int = 0):
    """CSR class tables with **independent** polarity per class, half of them
    negated. SPEC §4.4 warns against sharing the polarity flag array between
    the sum and max tables; here the point is simply that both branches of the
    complement-aware scatter carry real mass."""
    rng = np.random.default_rng(seed)
    stored = [np.sort(rng.choice(V, size=rng.integers(3, 12), replace=False))
              for _ in range(n_classes)]
    is_neg = np.array([c % 2 == 1 for c in range(n_classes)])
    indptr = np.zeros(n_classes + 1, np.int32)
    for c, st in enumerate(stored):
        indptr[c + 1] = indptr[c] + st.size
    indices = np.concatenate(stored).astype(np.int32)
    return indices, indptr, is_neg, stored


def test_scatter_is_complement_aware_against_a_dense_float64_reference():
    """`r_i(v) = Σ_{e : v ∈ label(e)} u_i(e)` at `L = 256` with mixed polarity.

    The reference is a dense membership matrix built directly from the class
    definitions — no CSR, no complement algebra — which is exactly the thing
    the optimised form replaces. SPEC §2.4: getting this wrong "silently
    produces a *plausible* wrong distribution".
    """
    L, V, Cn, E = L_PROD, 2048, 12, 40
    rng = np.random.default_rng(4)
    indices, indptr, is_neg, stored = _mixed_polarity_tables(V, Cn, seed=4)
    class_id = rng.integers(0, Cn, E).astype(np.int32)
    u = rng.random((L, E))

    got = np.asarray(MG.scatter_edge_mass_to_tokens(
        jnp.asarray(u), jnp.asarray(class_id), jnp.asarray(indices),
        jnp.asarray(indptr), jnp.asarray(is_neg), Cn, V))

    member = np.zeros((Cn, V), dtype=bool)
    for c, st in enumerate(stored):
        member[c, st] = True
        if is_neg[c]:
            member[c] = ~member[c]
    ref = u @ member[class_id]                       # [L, V], dense and dumb

    err = float(np.abs(got - ref).max())
    assert err < 1e-9, (
        f"complement-aware scatter differs from the dense reference by {err:.3g}"
    )
    assert is_neg.any() and (~is_neg).any(), "fixture lost its mixed polarity"


def test_partition_is_position_invariant_at_L256_with_mixed_polarity():
    """SPEC §2.4's real invariant: `Σ_v p_i(v)·r_i(v) = Z` for **every** `i`.

    `Σ_v q_i(v) == 1` cannot fail — `q` is divided by its own row sum — so the
    unnormalised row sums are the only checkable statement, and an `a`/`b`
    off-by-one or a dropped `W` factor is exactly what breaks them. Run at the
    production `L`, because a misalignment of one position is proportionally
    invisible at `L = 8`.
    """
    L, V, Cn, S = L_PROD, 512, 8, 16
    rng = np.random.default_rng(6)
    indices, indptr, is_neg, _stored = _mixed_polarity_tables(V, Cn, seed=6)
    # A spanning cycle guarantees strong connectivity — without it a random
    # edge set can leave `a_i` with no live state and the invariant would be
    # measuring an empty language (Z == 0 cause (a)) instead of an alignment.
    # The extra random edges give parallel (src, dst) pairs, i.e. an NFA.
    edge_src = np.concatenate([np.arange(S), rng.integers(0, S, 24)]).astype(np.int32)
    edge_dst = np.concatenate([(np.arange(S) + 1) % S,
                               rng.integers(0, S, 24)]).astype(np.int32)
    E = edge_src.size
    class_id = rng.integers(0, Cn, E).astype(np.int32)

    p = np.asarray(_sharp_marginals(L, V, seed=6, scale=2.0)).T   # [V, L]
    W_c = np.asarray(MG.class_weights(
        jnp.asarray(p), jnp.asarray(indices),
        jnp.asarray(np.repeat(np.arange(Cn), np.diff(indptr))),
        jnp.asarray(is_neg), Cn))
    W_e = W_c[class_id]                                            # [E, L]
    M = np.zeros((L, S, S))
    np.add.at(M, (slice(None), edge_src, edge_dst), W_e.T)

    # Scaled forward/backward, longhand — the only sequential loop in this file
    # and deliberately so: it is the reference, not the shipped path.
    a = np.zeros((L + 1, S))
    b = np.zeros((L + 1, S))
    la = np.zeros(L + 1)
    lb = np.zeros(L + 1)
    a[0] = 1.0 / S
    b[L] = 1.0
    for i in range(L):
        v = a[i] @ M[i]
        sc = max(v.max(), 1e-300)
        a[i + 1], la[i + 1] = v / sc, la[i] + np.log(sc)
    for i in range(L - 1, -1, -1):
        v = M[i] @ b[i + 1]
        sc = max(v.max(), 1e-300)
        b[i], lb[i] = v / sc, lb[i + 1] + np.log(sc)

    _q, Z_i = MG.constrained_marginals_and_partition(
        jnp.asarray(p), jnp.asarray(a), jnp.asarray(b),
        jnp.asarray(edge_src), jnp.asarray(edge_dst), jnp.asarray(class_id),
        jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(is_neg), Cn)
    Z_i = np.asarray(Z_i)
    assert np.all(Z_i > 0), "empty fixture; Z == 0 cause (a)"
    logZ = np.log(Z_i) + la[:L] + lb[1:L + 1]
    spread = float(logZ.max() - logZ.min())
    assert spread < 1e-8, (
        f"log Z varies by {spread:.3g} nats across the 256 positions — an "
        f"`a`/`b` misalignment, a dropped edge-weight factor, or a log-scale "
        f"sum that does not run over every node"
    )


def test_eq8_token_draw_is_edge_multiplicity_weighted_not_an_existence_flag():
    """SPEC §2.6 eq (8). On an NFA with parallel, label-overlapping edges the
    `∃`-indicator form deviates from the exact posterior by 1.7e-2 while the
    multiplicity form is exact to 2e-17 — and on a DFA the two coincide, so
    this is invisible unless NFAs are tested (CLAUDE.md).

    Asserted on the shared scatter kernel: fed the edge indicator, it must
    return the **count** of admitting edges, so a token on two parallel edges
    comes back as 2 and not as 1.
    """
    V, Cn = 512, 2
    # class 0: tokens 0..15 ; class 1: tokens 8..23  -> 8..15 overlap
    stored = [np.arange(0, 16, dtype=np.int32), np.arange(8, 24, dtype=np.int32)]
    indices = np.concatenate(stored)
    indptr = np.array([0, 16, 32], np.int32)
    is_neg = np.array([False, False])
    class_id = np.array([0, 1], np.int32)            # two parallel edges, 0 -> 1
    sel = np.ones((4, 2))                            # both edges admitted

    mult = np.asarray(MG.scatter_edge_mass_to_tokens(
        jnp.asarray(sel), jnp.asarray(class_id), jnp.asarray(indices),
        jnp.asarray(indptr), jnp.asarray(is_neg), Cn, V))
    assert np.allclose(mult[:, 8:16], 2.0), (
        "tokens carried by BOTH parallel edges came back with multiplicity "
        f"{mult[0, 8]} — that is the `∃` form, which draws from the wrong "
        "posterior on every NFA"
    )
    assert np.allclose(mult[:, 0:8], 1.0)
    assert np.allclose(mult[:, 16:24], 1.0)
    assert np.allclose(mult[:, 24:], 0.0)


def test_entropy_clamp_survives_exact_zeros_at_production_width():
    """SPEC §2.4: `q` has exact zeros wherever the automaton forbids a token,
    so `log q` is `-inf` and the naive entropy is `NaN`, which then rides into
    §3.4's acceptance mask. Checked at `L = 256` on a `q` that is mostly zero,
    which is what a real grammar produces.

    **[Strengthened 2026-08-08 after a mutation review.]** This previously
    asserted only `isfinite(H)` and `0 ≤ H ≤ log 6`. A mutant returning
    **constant zero** satisfied both and the test passed — the exact
    `Σ_v q_i(v) == 1` failure mode this file's own docstring names, committed
    by the file that names it. The fix is to assert the *value* against a
    float64 reference computed from the known 6-token support, and to require
    the 256 entropies to actually differ, which no constant can do.
    """
    L, V, K = L_PROD, 4096, 6
    rng = np.random.default_rng(8)
    q = np.zeros((L, V))
    support = np.zeros((L, K))
    for i in range(L):
        keep = rng.choice(V, size=K, replace=False)
        w = rng.random(K)
        w = w / w.sum()
        q[i, keep] = w
        support[i] = w

    # Reference: -Σ w log w over the K non-zero weights only. The zeros never
    # enter, so this needs no clamp and is independent of the kernel's.
    ref = -(support * np.log(support)).sum(axis=1)

    H = np.asarray(MG.entropy_from_q(jnp.asarray(q)))
    assert np.isfinite(H).all(), "entropy is NaN/inf on a q with exact zeros"
    err = float(np.abs(H - ref).max())
    assert err < 1e-12, (
        f"entropy differs from the float64 reference by {err:.3g} nats. Either "
        f"the clamp is contributing mass on the {V - K} forbidden tokens "
        f"(`0 · log(1e-30)` must be exactly 0) or the sum is wrong."
    )
    # A constant — the mutation this test used to pass under — cannot do this.
    assert float(H.max() - H.min()) > 0.1, (
        "all 256 entropies are (near) identical; the fixture has degenerated "
        "and a constant-valued kernel would pass"
    )
    assert 0.0 < float(ref.min()) and float(ref.max()) < np.log(K), (
        "fixture check: the reference must be strictly inside (0, log K), so "
        "neither bound can be hit by accident"
    )
