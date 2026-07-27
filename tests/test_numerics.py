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

from diffgemma_fa.infer import reference as R

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
    logits = np.array([[1.0, 2.0, 3.0, 4.0]])
    support = np.array([[True, False, True, False]])

    inf_masked = np.where(support, logits, -np.inf)
    finite_masked = np.where(support, logits, -1e30)

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
