"""Numerical stability. SPEC §6.3, §2.4.

Two things to establish, and the second is the unusual one:

1. The **scaled** path is clean at `L = 256` with sharp marginals — no NaN, no
   Inf, `Z > 0`.
2. The **unscaled** path underflows there. SPEC asks for that to be tested
   deliberately, because it is the documentation of why the scaling exists.
   `Z == 0` from underflow is cause (c) in SPEC's taxonomy — *expected*, and
   distinct from the two causes that are bugs.

Plus §2.4's clamp rule: `q_i` has exact zeros wherever the automaton forbids a
token, so `log q` must never be taken directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.infer import marginals  # noqa: E402
from diffgemma_fa.infer import scans  # noqa: E402
from diffgemma_fa.infer import reference as R  # noqa: E402

V = 8


def sharp_probs(rng: np.random.Generator, L: int, scale: float = 10.0) -> np.ndarray:
    """`p ~ softmax(N(0,1) · scale)` — SPEC §6.3's stress distribution."""
    logits = rng.standard_normal((L, V)) * scale
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def ring_automaton(n_states: int = 6) -> R.Automaton:
    """A strongly connected automaton that **forbids part of the vocabulary**.

    Tokens 6 and 7 appear on no edge. That matters for two reasons and both are
    the point of this file:

      - it makes every `M_i` genuinely **sub-unit** (row mass `p(0..5) < 1`), so
        the unnormalized product actually underflows at `L = 256` — an
        automaton admitting all `V` tokens has unit row sums and would never
        demonstrate the failure the scaling exists to prevent;
      - it gives `q_i` **exact zeros**, without which §2.4's `log q` clamp rule
        has nothing to guard against.
    """
    edges = []
    for s in range(n_states):
        edges.append((s, (s + 1) % n_states, frozenset({0, 1, 2})))
        edges.append((s, s, frozenset({3, 4, 5})))
    start = np.zeros(n_states)
    start[0] = 1.0
    return R.Automaton(n_states=n_states, vocab_size=V, edges=tuple(edges),
                       start=start, finals=frozenset({0, n_states - 1}))



def class_tables(A: R.Automaton, threshold: int | None = None):
    """Intern labels and build the CSR the JAX path consumes."""
    threshold = V // 2 if threshold is None else threshold
    lookup, members, class_of = {}, [], []
    for _, _, lab in A.edges:
        if lab not in lookup:
            lookup[lab] = len(members)
            members.append(lab)
        class_of.append(lookup[lab])
    is_neg = [len(m) > threshold for m in members]
    stored = [sorted(set(range(V)) - set(m)) if is_neg[c] else sorted(m)
              for c, m in enumerate(members)]
    indptr = np.zeros(len(members) + 1, np.int32)
    for c, st in enumerate(stored):
        indptr[c + 1] = indptr[c] + len(st)
    indices = np.array([x for st in stored for x in st], np.int32)
    seg = np.array([c for c, st in enumerate(stored) for _ in st], np.int32)
    return (np.array(class_of, np.int32), members, np.array(is_neg),
            indices, indptr, seg)


def _viable_instance(rng, L, nfa):
    """A ring automaton (DFA) or a ring with an extra overlapping parallel edge
    (NFA), plus sharp marginals, guaranteed to have `Z > 0` at this `L`."""
    A = ring_automaton()
    if nfa:
        # Overlapping but DISTINCT label on an existing state pair, so the
        # multiplicity is genuinely 2 on the shared token.
        edges = list(A.edges) + [(0, 1, frozenset({2, 3}))]
        A = R.Automaton(n_states=A.n_states, vocab_size=V, edges=tuple(edges),
                        start=A.start, finals=A.finals)
    for _ in range(40):
        p = sharp_probs(rng, L, scale=2.0)
        W = R.edge_weights(p, A)
        M = R.transition_matrices(W, A)
        fb = R.forward_backward(M, A.start, np.ones(A.n_states))
        if np.all(fb.a[1:].sum(axis=1) > 0) and np.all(fb.b[:-1].sum(axis=1) > 0):
            return A, p
    raise AssertionError("no viable instance; fix the generator")


# ---------------------------------------------------------------------------
# The scaled path is clean at L = 256
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_scaled_path_is_finite_at_L256(seed):
    rng = np.random.default_rng(60000 + seed)
    L = 256
    A = ring_automaton()
    p = sharp_probs(rng, L)
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    fb = R.forward_backward(M, A.start, A.final_vector(), scaled=True)

    assert np.isfinite(fb.a).all(), "scaled forward vectors must stay finite"
    assert np.isfinite(fb.b).all(), "scaled backward vectors must stay finite"
    assert np.isfinite(fb.log_scale_a).all()
    assert np.isfinite(fb.log_scale_b).all()
    assert np.isfinite(fb.log_Z), "log Z must be finite"
    assert fb.log_Z < 0.0, "Z is a probability mass over a huge space"
    assert not np.isnan(fb.a).any() and not np.isnan(fb.b).any()


@pytest.mark.parametrize("seed", range(5))
def test_scaled_log_Z_is_i_invariant_at_L256(seed):
    """The `i`-invariance assertion is the one that actually proves the scaling
    is bookkept correctly — magnitudes alone would not."""
    rng = np.random.default_rng(61000 + seed)
    L = 256
    A = ring_automaton()
    p = sharp_probs(rng, L)
    M = R.transition_matrices(R.edge_weights(p, A), A)
    fb = R.forward_backward(M, A.start, A.final_vector(), scaled=True)

    values = np.array([fb.log_Z_at(i) for i in range(0, L + 1, 16)])
    assert np.isfinite(values).all()
    spread = values.max() - values.min()
    assert spread < 1e-8, f"log Z drifts across i by {spread:.3g}"


# ---------------------------------------------------------------------------
# The unscaled path underflows — SPEC §6.3 asks for this explicitly
# ---------------------------------------------------------------------------

def test_unscaled_path_underflows_at_L256_in_float32():
    """SPEC §2.4: "`a_i` is a product of `i` sub-unit matrices; at `L = 256` it
    underflows fp32 to exactly zero."

    This is `Z == 0` cause **(c)** — expected, not a bug — and the reason the
    scaling exists. Demonstrated in float32, which is the dtype the JAX path
    will actually use.
    """
    rng = np.random.default_rng(62000)
    L = 256
    A = ring_automaton()
    p = sharp_probs(rng, L)
    M = R.transition_matrices(R.edge_weights(p, A), A).astype(np.float32)

    a = A.start.astype(np.float32)
    underflow_at = None
    for i in range(L):
        a = a @ M[i]
        if not a.any() and underflow_at is None:
            underflow_at = i
            break
    assert underflow_at is not None, (
        "expected fp32 underflow before L=256; if this stops failing the "
        "scaling argument needs revisiting, not the test"
    )
    assert underflow_at < L


def test_unscaled_float64_still_reaches_the_same_log_Z_at_short_L():
    """At a short `L` the unscaled path is fine, and must agree with the scaled
    one exactly — so the scaling is provably free, not an approximation."""
    rng = np.random.default_rng(62500)
    L = 16
    A = ring_automaton()
    p = sharp_probs(rng, L, scale=3.0)
    M = R.transition_matrices(R.edge_weights(p, A), A)
    scaled = R.forward_backward(M, A.start, A.final_vector(), scaled=True)
    plain = R.forward_backward(M, A.start, A.final_vector(), scaled=False)
    assert scaled.log_Z == pytest.approx(plain.log_Z, rel=1e-12)


# ---------------------------------------------------------------------------
# §2.4's clamp rule: q has exact zeros, so never take log q directly
# ---------------------------------------------------------------------------

def test_q_has_exact_zeros():
    """The premise of the clamp rule — if `q` had no exact zeros there would be
    nothing to guard against."""
    rng = np.random.default_rng(63000)
    L = 8
    A = ring_automaton()
    p = sharp_probs(rng, L, scale=2.0)
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    fb = R.forward_backward(M, A.start, A.final_vector())
    q = R.marginals(p, A, fb, W)
    assert (q == 0.0).any(), "expected forbidden tokens to have exactly zero mass"


def test_naive_entropy_of_q_is_nan_and_the_clamp_fixes_it():
    """SPEC §2.4: naive `-Σ q log q` gives `nan`; the fix is
    `lq = log(maximum(q, 1e-30))`, which yields exactly `0·(−69) = 0` on
    forbidden tokens."""
    rng = np.random.default_rng(63500)
    L = 8
    A = ring_automaton()
    p = sharp_probs(rng, L, scale=2.0)
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    fb = R.forward_backward(M, A.start, A.final_vector())
    q = R.marginals(p, A, fb, W)

    with np.errstate(divide="ignore", invalid="ignore"):
        naive = -(q * np.log(q)).sum(axis=-1)
    assert np.isnan(naive).any(), "the hazard this rule guards against is real"

    lq = np.log(np.maximum(q, 1e-30))
    clamped = -(q * lq).sum(axis=-1)
    assert np.isfinite(clamped).all()
    assert (clamped >= -1e-12).all(), "entropy must be non-negative"


def test_finite_sentinel_not_neg_inf_for_masked_logits():
    """SPEC §2.4/§3.7: use `-1e30`, never `-inf`. An `-inf` logit propagates
    through `softmax → @ embedding` and poisons the self-conditioning matmul;
    a finite sentinel does not."""
    # The constants are READ FROM THE SOURCE, not retyped. A mutation audit
    # found this test hardcoding `-1e30` in its own body, so flipping
    # `marginals.MASK_SENTINEL` to `-inf` left it green: it demonstrated the
    # hazard without checking the code.
    assert np.isfinite(marginals.MASK_SENTINEL), (
        "MASK_SENTINEL must be finite (SPEC §2.4/§3.7); an -inf logit poisons "
        "the self-conditioning matmul"
    )
    assert np.isfinite(scans.NEG_SENTINEL), (
        "NEG_SENTINEL must be finite (SPEC §2.7); fused max-plus kernels turn "
        "-inf + -inf into NaN"
    )
    # ...and it must have room to be halved without saturating, since
    # `maxplus_combine` adds two of them and re-clamps.
    assert scans.NEG_SENTINEL / 2.0 > np.finfo(np.float32).min, (
        "NEG_SENTINEL is too close to the fp32 floor for the sum of two to be "
        "distinguishable from -inf"
    )

    logits = np.array([[1.0, 2.0, 3.0, 4.0]])
    support = np.array([[True, False, True, False]])

    inf_masked = np.where(support, logits, -np.inf)
    finite_masked = np.where(support, logits, marginals.MASK_SENTINEL)

    embed = np.ones((4, 3))

    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    # Both give the same probabilities...
    np.testing.assert_allclose(softmax(inf_masked), softmax(finite_masked),
                               atol=1e-12)
    # ...but only the finite one survives being used as raw logits.
    with np.errstate(invalid="ignore"):
        poisoned = inf_masked @ embed
    clean = finite_masked @ embed
    assert not np.isfinite(poisoned).all(), "-inf must be shown to be harmful"
    assert np.isfinite(clean).all(), "the finite sentinel must stay finite"


def test_map_uses_a_finite_sentinel():
    """SPEC §2.7: represent impossible transitions with a finite sentinel, not
    `-inf`, so fused kernels cannot produce `NaN` from `-inf + -inf`."""
    rng = np.random.default_rng(64000)
    L = 6
    A = ring_automaton()
    p = sharp_probs(rng, L, scale=4.0)
    tokens, states, score = R.map_decode(p, A, A.start, A.final_vector())
    assert np.isfinite(score)
    assert R.accepts(A, tokens)


def test_map_matches_scored_path_at_L256():
    """A long MAP decode must stay finite and its reported score must equal the
    sum of the chosen tokens' log-probabilities."""
    rng = np.random.default_rng(64500)
    L = 256
    A = ring_automaton()
    p = sharp_probs(rng, L)
    tokens, states, score = R.map_decode(p, A, A.start, A.final_vector())
    assert np.isfinite(score)
    direct = float(sum(np.log(p[i, t]) for i, t in enumerate(tokens)))
    assert score == pytest.approx(direct, rel=1e-10)
    assert R.accepts(A, tokens)


def test_sampling_at_L256_stays_in_the_language():
    """End-to-end: the scaled path must sample cleanly at the real canvas
    length, not just at the enumerable sizes the exactness suite uses."""
    rng = np.random.default_rng(65000)
    L = 256
    A = ring_automaton()
    p = sharp_probs(rng, L)
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    fb = R.forward_backward(M, A.start, A.final_vector())
    for _ in range(5):
        tokens, _ = R.sample_chain(p, A, fb, W, rng)
        assert len(tokens) == L
        assert R.accepts(A, tokens)


# ==========================================================================
# The NON-vacuous replacement for `Σ_v q_i(v) == 1`
# ==========================================================================
#
# A mutation audit confirmed `Σ_v q_i(v) == 1` is vacuous in both code paths:
# `q` is produced by dividing by its own row sum, so the identity survives a
# dropped `u_i × W[i,e]` factor, an `a`/`b` off-by-one, and a deliberate 1e7
# scale error — all three leave every row sum at exactly 1.0, and an error in
# the log-scale accumulation leaves `q` bit-identical.
#
# The live invariant is one level up: before normalisation,
#
#     Σ_v p_i(v) · r_i(v) = Z   for EVERY i
#
# because both sides sum over the same set of accepted length-`L` strings,
# merely grouped by a different position. The `L` unnormalised row sums must
# therefore all be equal — checkable without knowing `Z`, and precisely what a
# misalignment breaks.

def _log_partition_by_position(A, p):
    """`[L]` of `log Z` recovered independently at each position.

    `forward_backward` is **scaled** (mandatory in fp32 at `L = 256`), so the
    raw row sum at position `i` is `Z` divided by the scales that `a_i` and
    `b_{i+1}` carry. Adding the log-scales back is what makes the `L` values
    comparable — and it puts SPEC §2.4's log-scale accumulation *inside* the
    invariant, so an error there is caught too. That is the mutation the
    advertised `Σ_v q_i == 1` check leaves `q` bit-identical under.
    """
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    fb = R.forward_backward(M, A.start, A.final_vector())
    class_of, members, is_neg, indices, indptr, _ = class_tables(A)
    _q, Z_i = marginals.constrained_marginals_and_partition(
        jnp.asarray(p.T), jnp.asarray(fb.a), jnp.asarray(fb.b),
        jnp.asarray([e[0] for e in A.edges], jnp.int32),
        jnp.asarray([e[1] for e in A.edges], jnp.int32),
        jnp.asarray(class_of, jnp.int32), jnp.asarray(indices),
        jnp.asarray(indptr), jnp.asarray(is_neg), len(members))
    Z_i = np.asarray(Z_i)
    assert np.all(Z_i > 0), "empty language slipped past the instance filter"
    # `u_i(e) = a_i(src) * b_{i+1}(dst)` -- see `constrained_marginals`.
    L = p.shape[0]
    return np.log(Z_i) + fb.log_scale_a[:L] + fb.log_scale_b[1:L + 1]


@pytest.mark.parametrize("nfa", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_the_partition_function_is_the_same_at_every_position(nfa, seed):
    """`Σ_v p_i(v) r_i(v)` must not depend on `i`. This is the assertion that
    `Σ_v q_i(v) == 1` was pretending to be."""
    rng = np.random.default_rng(91000 + seed * 13 + int(nfa))
    A, p = _viable_instance(rng, 8, nfa)
    logZ_i = _log_partition_by_position(A, p)
    spread = float(logZ_i.max() - logZ_i.min())
    assert spread < 1e-9, (
        f"log Z varies by {spread:.3g} nats across positions: {logZ_i} — an "
        f"`a`/`b` misalignment, a dropped edge-weight factor, or a log-scale "
        f"sum that does not run over every node"
    )


@pytest.mark.parametrize("nfa", [False, True])
def test_that_partition_equals_the_brute_force_Z(nfa):
    """And its common value is the enumerated `Z`, so the invariant is anchored
    to ground truth rather than merely self-consistent."""
    rng = np.random.default_rng(92000 + int(nfa))
    A, p = _viable_instance(rng, 4, nfa)
    _post, Z = R.enumerate_posterior(p, A)
    logZ_i = _log_partition_by_position(A, p)
    assert float(np.exp(logZ_i[0])) == pytest.approx(Z, rel=1e-9)


# ==========================================================================
# log_matmul's two-band shift (found live: live_simple_106-63-0)
# ==========================================================================

def test_log_matmul_survives_the_tail_vs_grammar_dynamic_range():
    """A single row/col max shift underflows on the REAL structure.

    The unscored `ACC --Σ--> ACC` tail pins the row/col maxes at 0.0 while a
    genuine grammar path across L = 256 sits at log Z ≈ -846: each of its
    contributions is exp(-423)·exp(-423) ≈ 1e-368, below float64's smallest
    subnormal, so the whole (start, ACC) entry underflowed to sentinel.
    Measured live on `live_simple_106-63-0` (403 states): `joint_draw`
    reported a provably non-empty language as Z == 0, and before the
    feasibility detector existed this emitted silent garbage. The sequential
    reference gets -846 easily — it folds into a running max and never
    multiplies two tiny halves together.

    Minimal reproduction: two states, a self-loop at weight 1 (log 0) and a
    start→ACC chain at log -423 per half.
    """
    from diffgemma_fa.infer import scans
    import jax.numpy as jnp

    neg = scans.NEG_SENTINEL
    # A = B = one 128-length half-product: [[ -423 (start→start), -423 (start→acc)],
    #                                       [ sentinel,            0 (acc→acc)  ]]
    half = jnp.asarray([[-423.0, -423.0], [neg, 0.0]], dtype=jnp.float64)
    root = scans.log_matmul(half, half)
    got = float(root[0, 1])
    # exact: logsumexp(-423 + -423, -423 + 0) = -423 + log1p(exp(-423)) ≈ -423
    assert got == pytest.approx(-423.0, abs=1e-6), (
        f"(start, acc) came back {got}; the single-shift form underflowed "
        "this to sentinel"
    )


def test_log_matmul_two_band_is_still_exact_in_the_easy_regime():
    """The 4-GEMM path must agree with brute-force logsumexp on ordinary
    matrices, including mixed sentinel patterns."""
    from diffgemma_fa.infer import scans
    import jax.numpy as jnp

    rng = np.random.default_rng(5)
    A = jnp.asarray(rng.normal(size=(5, 7)) * 10, dtype=jnp.float64)
    B = jnp.asarray(rng.normal(size=(7, 4)) * 10, dtype=jnp.float64)
    A = A.at[2, :].set(scans.NEG_SENTINEL)          # dead row
    B = B.at[:, 1].set(scans.NEG_SENTINEL)          # dead column
    got = np.asarray(scans.log_matmul(A, B))
    # brute force
    ref = np.full((5, 4), scans.NEG_SENTINEL)
    An, Bn = np.asarray(A), np.asarray(B)
    for i in range(5):
        for j in range(4):
            terms = [An[i, k] + Bn[k, j] for k in range(7)
                     if An[i, k] > scans.NEG_SENTINEL / 2
                     and Bn[k, j] > scans.NEG_SENTINEL / 2]
            if terms:
                m = max(terms)
                ref[i, j] = m + np.log(sum(np.exp(t - m) for t in terms))
    live = ref > scans.NEG_SENTINEL / 2
    assert np.allclose(got[live], ref[live], atol=1e-9)
    assert (got[~live] <= scans.NEG_SENTINEL / 2).all()


def test_log_matmul_survives_per_position_sharp_marginals():
    """The regime that killed the two-band form: per-position sharp `p` gives
    a LOW band whose internal spread is itself thousands of nats, so terms
    still underflowed against the band anchor. The streaming pairwise-max
    anchor cannot underflow the dominant term of any entry by construction.

    Chain automaton in log space with per-step costs drawn to span ~4000 nats
    total — the shape real annealed marginals produce on `live_simple_106-63-0`.
    """
    from diffgemma_fa.infer import scans
    import jax.numpy as jnp

    rng = np.random.default_rng(9)
    neg = scans.NEG_SENTINEL
    S, L = 6, 64
    # chain 0->1->...->5 with huge per-step costs, plus a free self-loop at 5
    mats = np.full((L, S, S), neg)
    for i in range(L):
        for s in range(S - 1):
            mats[i, s, s + 1] = -rng.uniform(20, 80)     # model hates these
        mats[i, S - 1, S - 1] = 0.0                       # unscored tail
    tr = scans.up_sweep_log(jnp.asarray(mats))
    root = np.asarray(tr.root)
    # reference: sequential logsumexp product
    ref = mats[0]
    for i in range(1, L):
        Aexp = ref
        out = np.full((S, S), neg)
        for a_ in range(S):
            for b_ in range(S):
                terms = [Aexp[a_, k] + mats[i][k, b_] for k in range(S)
                         if Aexp[a_, k] > neg / 2 and mats[i][k, b_] > neg / 2]
                if terms:
                    m = max(terms)
                    out[a_, b_] = m + np.log(sum(np.exp(t - m) for t in terms))
        ref = out
    live = ref > neg / 2
    assert (root[live] > neg / 2).all(), (
        "tree lost entries the sequential product keeps — the anchor "
        "underflow is back"
    )
    assert np.allclose(root[live], ref[live], rtol=0, atol=1e-8)


def test_streamed_constrained_entropy_equals_the_dense_form():
    """`--confidence=mar` needs `H(q_i)`, and the dense route builds `q`,
    `row`, `lq` and `q*lq` at `[L, V] = [256, 262144]` — 537 MB each — inside
    the denoising `while_loop`. That deadlocked on three records (CPU frozen at
    5:21 while elapsed reached 39 minutes).

    The streamed form uses the identity
    `H = log Z - (1/Z)·Σ_v w log w` with `w = p·r`, accumulated over the CSR,
    so the cost is `O(nnz)` rather than `O(L·V)`. It must be *exactly* the same
    number, including on negated classes — where `r` is recovered as a
    complement and a naive sparse sum would be wrong.
    """
    from diffgemma_fa.infer import marginals as M
    import jax.numpy as jnp

    rng = np.random.default_rng(3)
    L, V, C, E = 6, 32, 4, 10
    p = jnp.asarray(rng.random((V, L)))
    p = p / p.sum(0, keepdims=True)
    u = jnp.asarray(rng.random((L, E)))
    cid = jnp.asarray(rng.integers(0, C, E), jnp.int32)
    idx = jnp.asarray(np.concatenate(
        [rng.choice(V, 5, replace=False) for _ in range(C)]), jnp.int32)
    iptr = jnp.asarray(np.arange(C + 1) * 5, jnp.int32)
    neg = jnp.asarray([False, True, False, True])   # mixed polarity on purpose

    r = M.scatter_edge_mass_to_tokens(u, cid, idx, iptr, neg, C, V)
    q = p.T * r
    q = q / jnp.maximum(q.sum(1, keepdims=True), jnp.finfo(q.dtype).tiny)
    dense = np.asarray(M.entropy_from_q(q))
    streamed = np.asarray(
        M.constrained_entropy_streamed(p, u, cid, idx, iptr, neg, C, V))
    assert np.abs(dense - streamed).max() < 1e-12, (
        f"streamed entropy differs from dense by "
        f"{np.abs(dense - streamed).max():.3g}"
    )
