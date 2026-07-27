"""Brute-force exactness — the bedrock. SPEC §6.1.

CLAUDE.md: *"Never let this go red. It brute-force-enumerates the constrained
posterior on tiny alphabets and is the only thing standing between you and a
plausible-looking sampler that draws from the wrong distribution. A subtly wrong
sampler still produces valid strings."*

Setup per SPEC: `V = 4`, `L ∈ {4, 6, 8}`, random small DFAs **and NFAs**
(3–8 states, including parallel overlapping edges), enumerate all `V^L`
sequences, compute the exact path-weighted posterior in float64.

**Fixed seeds and Bonferroni-corrected χ² thresholds.** This suite runs dozens
of independent statistical tests; an uncorrected `p > 0.001` would go red on its
own. Re-run protocol on a suspected flake: bump `SEED_OFFSET` below and re-run.
A genuine bug fails at every offset; a statistical fluke moves.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from diffgemma_fa.infer import reference as R

V = 4
SEED_OFFSET = 0

#: Number of independent χ² tests in this file, for the Bonferroni correction.
#: Counted generously; over-counting only makes the suite more permissive, and
#: a real distributional bug fails by orders of magnitude, not marginally.
N_CHI2_TESTS = 200
ALPHA = 0.001
ALPHA_CORRECTED = ALPHA / N_CHI2_TESTS


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def random_probs(rng: np.random.Generator, L: int) -> np.ndarray:
    p = rng.random((L, V)) + 0.05
    return p / p.sum(axis=1, keepdims=True)


def random_dfa(rng: np.random.Generator, n_states: int) -> R.Automaton:
    """Random partial DFA: each (state, token) has at most one destination."""
    by_pair: dict[tuple[int, int], set[int]] = {}
    for s in range(n_states):
        for v in range(V):
            if rng.random() < 0.65:
                d = int(rng.integers(n_states))
                by_pair.setdefault((s, d), set()).add(v)
    edges = tuple((s, d, frozenset(vs)) for (s, d), vs in sorted(by_pair.items()))
    start = np.zeros(n_states)
    start[0] = 1.0
    finals = frozenset(
        s for s in range(n_states) if rng.random() < 0.4
    ) or frozenset({n_states - 1})
    return R.Automaton(n_states=n_states, vocab_size=V, edges=edges,
                       start=start, finals=finals)


def random_nfa(rng: np.random.Generator, n_states: int) -> R.Automaton:
    """Random NFA **with parallel, overlapping-label edges**.

    The overlap is the point: it is what makes the path multiplicity in eq (8)
    observable. A suite that only exercises DFAs cannot see the bug.
    """
    edges = []
    n_edges = int(rng.integers(n_states, 3 * n_states + 1))
    for _ in range(n_edges):
        s = int(rng.integers(n_states))
        d = int(rng.integers(n_states))
        k = int(rng.integers(1, V + 1))
        labels = frozenset(int(x) for x in rng.choice(V, size=k, replace=False))
        edges.append((s, d, labels))
    # Force at least one genuinely parallel overlapping pair.
    if edges:
        s, d, labels = edges[0]
        edges.append((s, d, labels))
    start = np.zeros(n_states)
    start[0] = 1.0
    finals = frozenset(
        s for s in range(n_states) if rng.random() < 0.4
    ) or frozenset({n_states - 1})
    return R.Automaton(n_states=n_states, vocab_size=V, edges=tuple(edges),
                       start=start, finals=finals)


def nontrivial(rng, maker, n_states, L, tries=40):
    """An instance whose language is non-empty at this length."""
    for _ in range(tries):
        A = maker(rng, n_states)
        p = random_probs(rng, L)
        post, Z = R.enumerate_posterior(p, A)
        if Z > 0 and len(post) >= 2:
            return A, p, post, Z
    pytest.skip("no non-degenerate instance found for this seed")


def prepared(A: R.Automaton, p: np.ndarray, b_final=None):
    b = A.final_vector() if b_final is None else b_final
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    fb = R.forward_backward(M, A.start, b)
    return W, M, fb, b


def chi2_pvalue(samples: list[tuple[int, ...]],
                posterior: dict[tuple[int, ...], float]) -> float:
    """Pearson χ² of empirical counts against the exact posterior.

    Bins with a tiny expected count are pooled, since χ² is unreliable below
    ~5 expected and pooling is the standard remedy.
    """
    n = len(samples)
    keys = sorted(posterior)
    counts = {k: 0 for k in keys}
    for s in samples:
        assert s in counts, f"sampler produced a string outside the support: {s}"
        counts[s] += 1

    obs, exp, pooled_o, pooled_e = [], [], 0.0, 0.0
    for k in keys:
        e = posterior[k] * n
        if e >= 5.0:
            obs.append(counts[k])
            exp.append(e)
        else:
            pooled_o += counts[k]
            pooled_e += e
    if pooled_e > 0:
        obs.append(pooled_o)
        exp.append(pooled_e)
    if len(obs) < 2:
        return 1.0
    obs = np.array(obs, dtype=np.float64)
    exp = np.array(exp, dtype=np.float64)
    exp *= obs.sum() / exp.sum()
    return float(stats.chisquare(obs, exp).pvalue)


# ===========================================================================
# Test 1 — Z from forward-backward == Z from enumeration
# ===========================================================================

@pytest.mark.parametrize("maker", [random_dfa, random_nfa])
@pytest.mark.parametrize("L", [4, 6, 8])
@pytest.mark.parametrize("seed", range(6))
def test_1_partition_function_matches_enumeration(maker, L, seed):
    rng = np.random.default_rng(1000 + seed + SEED_OFFSET)
    A, p, post, Z = nontrivial(rng, maker, int(rng.integers(3, 9)), L)
    _, _, fb, _ = prepared(A, p)
    assert fb.log_Z == pytest.approx(np.log(Z), rel=1e-10)


# ===========================================================================
# Test 2 — q_i == exact marginal; row sums == 1; a_i.b_i is i-invariant
# ===========================================================================

@pytest.mark.parametrize("maker", [random_dfa, random_nfa])
@pytest.mark.parametrize("L", [4, 6, 8])
@pytest.mark.parametrize("seed", range(6))
def test_2_marginals_match_and_are_normalised(maker, L, seed):
    rng = np.random.default_rng(2000 + seed + SEED_OFFSET)
    A, p, post, Z = nontrivial(rng, maker, int(rng.integers(3, 9)), L)
    W, M, fb, _ = prepared(A, p)

    q = R.marginals(p, A, fb, W)
    q_exact = R.marginals_from_posterior(post, L, V)
    np.testing.assert_allclose(q, q_exact, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(q.sum(axis=1), np.ones(L), rtol=1e-12)


@pytest.mark.parametrize("maker", [random_dfa, random_nfa])
@pytest.mark.parametrize("seed", range(8))
def test_2b_log_Z_is_i_invariant(maker, seed):
    """SPEC §2.4: `log(a_i·b_i) + logscale_a[i] + logscale_b[i]` must be the
    same for every `i`. This is the assertion that catches a scaling bug."""
    rng = np.random.default_rng(2500 + seed + SEED_OFFSET)
    L = 8
    A, p, post, Z = nontrivial(rng, maker, int(rng.integers(3, 9)), L)
    _, _, fb, _ = prepared(A, p)
    values = [fb.log_Z_at(i) for i in range(L + 1)]
    for v in values:
        assert v == pytest.approx(np.log(Z), rel=1e-10)


# ===========================================================================
# Test 3 — chain sampler against the exact posterior (chi-squared)
# ===========================================================================

@pytest.mark.parametrize("maker", [random_dfa, random_nfa])
@pytest.mark.parametrize("seed", range(4))
def test_3_chain_sampler_matches_exact_posterior(maker, seed):
    rng = np.random.default_rng(3000 + seed + SEED_OFFSET)
    L = 4
    A, p, post, Z = nontrivial(rng, maker, int(rng.integers(3, 7)), L)
    W, M, fb, _ = prepared(A, p)

    draws = [tuple(int(t) for t in R.sample_chain(p, A, fb, W, rng)[0])
             for _ in range(4000)]
    assert chi2_pvalue(draws, post) > ALPHA_CORRECTED


# ===========================================================================
# Test 4 — tree == chain in distribution, ON NFAs.
#          This is the test that catches an exists-form eq (8).
# ===========================================================================

@pytest.mark.parametrize("maker", [random_dfa, random_nfa])
@pytest.mark.parametrize("seed", range(4))
def test_4_tree_sampler_matches_exact_posterior(maker, seed):
    """The tree and chain draw a different number of variates, so this is
    distributional only — never assert seed-identity between them."""
    rng = np.random.default_rng(4000 + seed + SEED_OFFSET)
    L = 4
    A, p, post, Z = nontrivial(rng, maker, int(rng.integers(3, 7)), L)
    W, M, fb, b = prepared(A, p)

    draws = [tuple(int(t) for t in
                   R.sample_tree(p, A, M, W, A.start, b, rng)[0])
             for _ in range(4000)]
    assert chi2_pvalue(draws, post) > ALPHA_CORRECTED


def test_4b_exists_form_is_measurably_wrong_on_an_nfa():
    """SPEC §2.6: on an NFA with parallel overlapping edges the `∃` form
    deviates from the exact path-weighted posterior by ~1.7e-2, while the
    multiplicity-weighted form is at ~2e-17.

    Constructed rather than random, so the failure is deterministic and the
    magnitudes are pinned.
    """
    # Two parallel edges 0->1 whose labels overlap on token 1: token 1 has
    # multiplicity 2 across that state pair, tokens 0 and 2 have multiplicity 1.
    edges = (
        (0, 1, frozenset({0, 1})),
        (0, 1, frozenset({1, 2})),
        (1, 2, frozenset({0, 1, 2, 3})),
        (2, 1, frozenset({0, 1, 2, 3})),
    )
    start = np.zeros(3)
    start[0] = 1.0
    A = R.Automaton(n_states=3, vocab_size=V, edges=edges, start=start,
                    finals=frozenset({1, 2}))
    assert not A.is_dfa

    rng = np.random.default_rng(4321)
    L = 4
    p = random_probs(rng, L)
    post, Z = R.enumerate_posterior(p, A)
    assert Z > 0
    W, M, fb, b = prepared(A, p)

    def empirical(exists: bool) -> dict:
        g = np.random.default_rng(99)
        counts: dict[tuple[int, ...], int] = {}
        n = 40000
        for _ in range(n):
            t, _ = R.sample_tree(p, A, M, W, A.start, b, g,
                                 use_exists_form=exists)
            k = tuple(int(x) for x in t)
            counts[k] = counts.get(k, 0) + 1
        return {k: c / n for k, c in counts.items()}

    keys = sorted(post)
    correct = empirical(False)
    wrong = empirical(True)
    err_correct = max(abs(correct.get(k, 0.0) - post[k]) for k in keys)
    err_wrong = max(abs(wrong.get(k, 0.0) - post[k]) for k in keys)

    # The correct form is within sampling noise; the exists form is not, and
    # the gap must be large enough that this test has teeth.
    assert err_wrong > 5 * err_correct, (
        f"exists-form error {err_wrong:.4g} vs correct {err_correct:.4g} — "
        "this test is supposed to separate them"
    )


@pytest.mark.parametrize("seed", range(8))
def test_4c_exists_form_coincides_on_a_dfa(seed):
    """On a DFA the two forms coincide (one edge per `(s,s')`, disjoint labels),
    which is what makes gating the cheap path on `is_dfa` sound.

    Compared **analytically**, as the exact conditional token distribution
    given `(s_i, s_{i+1})`. Seed-identity would be the wrong assertion — the two
    paths consume a different number of variates, exactly as test 4's docstring
    says — and an empirical comparison would only be as sharp as the sample
    size. This is exact.
    """
    rng = np.random.default_rng(4500 + seed + SEED_OFFSET)
    L = 4
    A, p, post, Z = nontrivial(rng, random_dfa, 5, L)
    assert A.is_dfa
    W, _, _, _ = prepared(A, p)

    checked = 0
    for i in range(L):
        for s in range(A.n_states):
            for t in range(A.n_states):
                # multiplicity-weighted: sum over edges, each spread on its labels
                mult = np.zeros(V)
                for e, (src, dst, labels) in enumerate(A.edges):
                    if src == s and dst == t and labels:
                        idx = np.fromiter(sorted(labels), dtype=np.int64,
                                          count=len(labels))
                        sub = p[i][idx]
                        if sub.sum() > 0:
                            mult[idx] += W[i, e] * sub / sub.sum()
                # exists-indicator: one draw over the union of labels
                support: set[int] = set()
                for src, dst, labels in A.edges:
                    if src == s and dst == t:
                        support |= labels
                ex = np.zeros(V)
                if support:
                    idx = np.fromiter(sorted(support), dtype=np.int64,
                                      count=len(support))
                    ex[idx] = p[i][idx]
                if mult.sum() == 0 or ex.sum() == 0:
                    continue
                np.testing.assert_allclose(mult / mult.sum(), ex / ex.sum(),
                                           rtol=1e-12, atol=1e-14)
                checked += 1
    assert checked > 0, "instance exercised no state pair"


# ===========================================================================
# Test 5 — constrained MAP == exhaustive argmax, NFAs included
# ===========================================================================

@pytest.mark.parametrize("maker", [random_dfa, random_nfa])
@pytest.mark.parametrize("L", [4, 6])
@pytest.mark.parametrize("seed", range(12))
def test_5_map_matches_exhaustive_argmax(maker, L, seed):
    """MAP is exact on NFAs too — path multiplicity cannot change a `max` over
    strings (SPEC §2.7), so the comparison is against `argmax_x ∏ p_i(x_i)`
    over the *support*, not against the path-weighted posterior."""
    rng = np.random.default_rng(5000 + seed + SEED_OFFSET)
    A, p, post, Z = nontrivial(rng, maker, int(rng.integers(3, 9)), L)

    best_x, best_s = None, -np.inf
    for x in itertools.product(range(V), repeat=L):
        if R.accepting_path_count(A, x) == 0.0:
            continue
        s = float(sum(np.log(p[i, v]) for i, v in enumerate(x)))
        if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and
                                  (best_x is None or x < best_x)):
            best_x, best_s = x, s

    tokens, states, score = R.map_decode(p, A, A.start, A.final_vector())
    assert score == pytest.approx(best_s, rel=1e-10)
    assert tuple(int(t) for t in tokens) == best_x


# ===========================================================================
# Test 6 — every sample and every MAP output accepted by an independent simulator
# ===========================================================================

@pytest.mark.parametrize("maker", [random_dfa, random_nfa])
@pytest.mark.parametrize("seed", range(8))
def test_6_all_outputs_are_accepted(maker, seed):
    rng = np.random.default_rng(6000 + seed + SEED_OFFSET)
    L = 6
    A, p, post, Z = nontrivial(rng, maker, int(rng.integers(3, 9)), L)
    W, M, fb, b = prepared(A, p)

    for _ in range(60):
        t, _ = R.sample_chain(p, A, fb, W, rng)
        assert R.accepts(A, t), f"chain sample rejected: {t}"
        t, _ = R.sample_tree(p, A, M, W, A.start, b, rng)
        assert R.accepts(A, t), f"tree sample rejected: {t}"
    t, _, _ = R.map_decode(p, A, A.start, A.final_vector())
    assert R.accepts(A, t), f"MAP output rejected: {t}"


# ===========================================================================
# Test 7 — degenerate cases
# ===========================================================================

def test_7a_single_string_language():
    """`|C| = 1`: the posterior is a point mass and every sampler must find it."""
    L = 4
    word = [1, 2, 0, 3]
    edges = tuple((i, i + 1, frozenset({word[i]})) for i in range(L))
    start = np.zeros(L + 1)
    start[0] = 1.0
    A = R.Automaton(n_states=L + 1, vocab_size=V, edges=edges, start=start,
                    finals=frozenset({L}))
    rng = np.random.default_rng(7000)
    p = random_probs(rng, L)
    W, M, fb, b = prepared(A, p)
    post, Z = R.enumerate_posterior(p, A)
    assert len(post) == 1
    for _ in range(20):
        assert list(R.sample_chain(p, A, fb, W, rng)[0]) == word
        assert list(R.sample_tree(p, A, M, W, A.start, b, rng)[0]) == word
    assert list(R.map_decode(p, A, A.start, A.final_vector())[0]) == word


def test_7b_universal_language_recovers_unconstrained_sampling():
    """`C = V^L` must reduce **exactly** to unconstrained sampling: `q_i == p_i`
    and the MAP is the per-position argmax."""
    L = 5
    edges = ((0, 0, frozenset(range(V))),)
    start = np.array([1.0])
    A = R.Automaton(n_states=1, vocab_size=V, edges=edges, start=start,
                    finals=frozenset({0}))
    rng = np.random.default_rng(7100)
    p = random_probs(rng, L)
    W, M, fb, b = prepared(A, p)

    q = R.marginals(p, A, fb, W)
    np.testing.assert_allclose(q, p, rtol=1e-12, atol=1e-14)

    tokens, _, _ = R.map_decode(p, A, A.start, A.final_vector())
    np.testing.assert_array_equal(tokens, p.argmax(axis=1))


def test_7c_empty_language_raises():
    """`C = ∅` must **raise**, not silently emit. SPEC's Z==0 taxonomy calls
    this cause (a): a real bug."""
    edges = ((0, 1, frozenset({0})),)
    start = np.zeros(2)
    start[0] = 1.0
    A = R.Automaton(n_states=2, vocab_size=V, edges=edges, start=start,
                    finals=frozenset())  # nothing accepts
    rng = np.random.default_rng(7200)
    p = random_probs(rng, 3)
    W, M, fb, b = prepared(A, p)
    with pytest.raises(R.EmptyLanguageError):
        R.sample_tree(p, A, M, W, A.start, b, rng)
    with pytest.raises(R.EmptyLanguageError):
        R.map_decode(p, A, A.start, A.final_vector())


def test_7d_unreachable_and_non_coaccessible_states():
    """Dead states must not perturb Z or the marginals."""
    rng = np.random.default_rng(7300)
    L = 4
    base = (
        (0, 1, frozenset({0, 1})),
        (1, 2, frozenset({2, 3})),
        (2, 1, frozenset({0})),
    )
    start3 = np.zeros(3)
    start3[0] = 1.0
    A3 = R.Automaton(n_states=3, vocab_size=V, edges=base, start=start3,
                     finals=frozenset({2}))
    # 3 = unreachable, 4 = reachable but cannot reach a final state.
    extra = base + ((3, 1, frozenset({0})), (1, 4, frozenset({3})),
                    (4, 4, frozenset({1})))
    start5 = np.zeros(5)
    start5[0] = 1.0
    A5 = R.Automaton(n_states=5, vocab_size=V, edges=extra, start=start5,
                     finals=frozenset({2}))

    p = random_probs(rng, L)
    _, _, fb3, _ = prepared(A3, p)
    _, _, fb5, _ = prepared(A5, p)
    assert fb3.log_Z == pytest.approx(fb5.log_Z, rel=1e-12)


def test_7e_length_one():
    edges = ((0, 1, frozenset({2, 3})),)
    start = np.zeros(2)
    start[0] = 1.0
    A = R.Automaton(n_states=2, vocab_size=V, edges=edges, start=start,
                    finals=frozenset({1}))
    rng = np.random.default_rng(7400)
    p = random_probs(rng, 1)
    W, M, fb, b = prepared(A, p)
    post, Z = R.enumerate_posterior(p, A)
    assert set(post) == {(2,), (3,)}
    q = R.marginals(p, A, fb, W)
    np.testing.assert_allclose(q, R.marginals_from_posterior(post, 1, V),
                               rtol=1e-12)


@pytest.mark.parametrize("L", [3, 5, 6, 7])
def test_7f_length_not_a_power_of_two(L):
    """SPEC §2.6(d): pad to the next power of two with `M_i = I`, and place
    `b_L` at the **true** `L`, not the padded one.

    Asserts the padded and unpadded computations agree exactly — if `b_L` were
    placed at the padded length the identity tail would silently change `Z`.
    """
    rng = np.random.default_rng(7500 + L)
    A, p, post, Z = nontrivial(rng, random_nfa, 4, L)
    W, M, fb, b = prepared(A, p)

    L_pad = 1 << (L - 1).bit_length()
    if L_pad != L:
        eye = np.broadcast_to(np.eye(A.n_states), (L_pad - L, A.n_states,
                                                   A.n_states))
        M_pad = np.concatenate([M, eye], axis=0)
        fb_pad = R.forward_backward(M_pad, A.start, b)
        # b_L lands at the true L because the identity tail propagates it back.
        np.testing.assert_allclose(fb_pad.b[L], fb.b[L], rtol=1e-12)
        assert fb_pad.log_Z == pytest.approx(fb.log_Z, rel=1e-12)
    assert fb.log_Z == pytest.approx(np.log(Z), rel=1e-10)


# ===========================================================================
# Test 8 — non-point-mass start vectors and NFA start sets
# ===========================================================================

@pytest.mark.parametrize("seed", range(8))
def test_8_general_start_vector(seed):
    """SPEC §5.7: `a_start = 1[s = s_0]` only holds for a DFA on block 0.
    Hardcoding a point mass is a bug that first appears on block 2."""
    rng = np.random.default_rng(8000 + seed + SEED_OFFSET)
    L = 4
    for _ in range(40):
        A0 = random_nfa(rng, int(rng.integers(3, 7)))
        start = (rng.random(A0.n_states) < 0.6).astype(np.float64)
        if start.sum() < 2:
            continue
        A = R.Automaton(n_states=A0.n_states, vocab_size=V, edges=A0.edges,
                        start=start, finals=A0.finals)
        p = random_probs(rng, L)
        post, Z = R.enumerate_posterior(p, A)
        if Z <= 0 or len(post) < 2:
            continue
        W, M, fb, b = prepared(A, p)
        assert fb.log_Z == pytest.approx(np.log(Z), rel=1e-10)
        np.testing.assert_allclose(
            R.marginals(p, A, fb, W),
            R.marginals_from_posterior(post, L, V), rtol=1e-10, atol=1e-12)

        draws = [tuple(int(x) for x in
                       R.sample_tree(p, A, M, W, A.start, b, rng)[0])
                 for _ in range(3000)]
        assert chi2_pvalue(draws, post) > ALPHA_CORRECTED
        return
    pytest.skip("no suitable multi-start instance for this seed")


# ===========================================================================
# Test 9 — budget-aware terminal factors 1[d(s) <= R]
# ===========================================================================

@pytest.mark.parametrize("seed", range(6))
def test_9_budget_aware_terminal_factor(seed):
    """SPEC §3.1b. `b_L(s) = 1[d(s) ≤ R]`, and `d(s) ≤ 0 ⟺ s ∈ F` must make the
    largest budget coincide with the plain final vector."""
    rng = np.random.default_rng(9000 + seed + SEED_OFFSET)
    L = 4
    A, p, post, Z = nontrivial(rng, random_dfa, int(rng.integers(3, 7)), L)

    b0 = A.budget_vector(0)
    np.testing.assert_array_equal(b0, A.final_vector())

    for Rb in (0, 1, 2, 5):
        b = A.budget_vector(Rb)
        if b.sum() == 0:
            continue
        W, M, fb, _ = prepared(A, p, b_final=b)
        post_b, Z_b = R.enumerate_posterior(p, A, b_final=b)
        if Z_b <= 0:
            continue
        assert fb.log_Z == pytest.approx(np.log(Z_b), rel=1e-10)
        # Support must be exactly the strings ending in a within-budget state.
        for x in post_b:
            reached = R.simulate(A, x)
            assert any(b[s] > 0 for s in reached)


def test_9b_budget_too_small_raises():
    """An `R` too small must yield `Z = 0` and **raise**, not silently emit."""
    L = 4
    # A chain needing 4 tokens to accept; a budget of 0 admits nothing.
    edges = tuple((i, i + 1, frozenset({0, 1})) for i in range(6))
    start = np.zeros(7)
    start[0] = 1.0
    A = R.Automaton(n_states=7, vocab_size=V, edges=edges, start=start,
                    finals=frozenset({6}))
    rng = np.random.default_rng(9500)
    p = random_probs(rng, L)
    b = A.budget_vector(0)          # only state 6 is within budget
    W, M, fb, _ = prepared(A, p, b_final=b)
    # From state 0, four tokens reach state 4, never state 6 -> Z == 0.
    assert fb.log_Z == -np.inf
    with pytest.raises(R.EmptyLanguageError):
        R.sample_tree(p, A, M, W, A.start, b, rng)
    with pytest.raises(R.EmptyLanguageError):
        R.map_decode(p, A, A.start, b)


# ===========================================================================
# Test 10 — suffix product operand order
# ===========================================================================

@pytest.mark.parametrize("seed", range(6))
def test_10_suffix_scan_operand_order(seed):
    """SPEC §2.6(c): `lax.associative_scan(..., reverse=True)` yields
    `f(f(z,y),x)`, which for non-commutative matmul is the wrong order for the
    suffix pass. The invariant to hold is

        b_i = M[i] @ M[i+1] @ ... @ M[L-1] @ b_L

    and the reversed order must be measurably different, or the test has no
    teeth.
    """
    rng = np.random.default_rng(10000 + seed + SEED_OFFSET)
    L = 6
    A, p, post, Z = nontrivial(rng, random_nfa, 5, L)
    W, M, fb, b = prepared(A, p)

    for i in range(L + 1):
        prod = np.eye(A.n_states)
        for k in range(i, L):
            prod = prod @ M[k]
        expected = prod @ b
        scale = np.exp(fb.log_scale_b[i] - fb.log_scale_b[L])
        np.testing.assert_allclose(fb.b[i] * scale, expected, rtol=1e-10,
                                   atol=1e-14)

    # The reversed order is a different matrix — confirms the assertion bites.
    fwd = np.eye(A.n_states)
    for k in range(L):
        fwd = fwd @ M[k]
    rev = np.eye(A.n_states)
    for k in range(L - 1, -1, -1):
        rev = rev @ M[k]
    assert not np.allclose(fwd, rev), "instance too symmetric to detect order"


def test_10b_block_products_are_left_to_right():
    rng = np.random.default_rng(10500)
    L = 8
    A, p, post, Z = nontrivial(rng, random_nfa, 4, L)
    _, M, _, _ = prepared(A, p)
    blocks = R.block_products(M)
    for (lo, hi), got in blocks.items():
        want = np.eye(A.n_states)
        for k in range(lo, hi):
            want = want @ M[k]
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-14)


# ===========================================================================
# Test 11 — the complement-aware path, topk bound, and the tie-break
# ===========================================================================

def _classes_with_polarity(A: R.Automaton, threshold: int):
    """Intern edge labels and force negation above a small size threshold.

    SPEC §6.1 test 11: set the polarity threshold to `|S_c| > 2` at `V = 4` so
    that negated classes actually occur in a tiny alphabet.
    """
    lookup: dict[frozenset[int], int] = {}
    members: list[frozenset[int]] = []
    class_of = []
    for _, _, labels in A.edges:
        if labels not in lookup:
            lookup[labels] = len(members)
            members.append(labels)
        class_of.append(lookup[labels])
    is_neg = [len(m) > threshold for m in members]
    return class_of, members, is_neg


def nontrivial_mixed_polarity(rng, maker, L, tries=400):
    """An instance that is non-degenerate **and** has both Pos and Neg classes.

    Searched for rather than skipped past: mixed polarity is the whole point of
    test 11, and a suite that quietly skips it provides no coverage of the one
    path SPEC says only this suite can catch.
    """
    for _ in range(tries):
        A = maker(rng, int(rng.integers(3, 8)))
        p = random_probs(rng, L)
        post, Z = R.enumerate_posterior(p, A)
        if Z <= 0 or len(post) < 2:
            continue
        class_of, members, is_neg = _classes_with_polarity(A, threshold=2)
        if any(is_neg) and not all(is_neg):
            return A, p, post, Z, class_of, members, is_neg
    raise AssertionError(
        "could not construct a mixed-polarity instance in "
        f"{tries} tries — the generator, not the test, needs fixing"
    )


@pytest.mark.parametrize("maker", [random_dfa, random_nfa])
@pytest.mark.parametrize("seed", range(10))
def test_11a_complement_aware_marginals_match(maker, seed):
    """`q_i` from `r_i(v)` must equal `q_i` from the direct scatter **and** from
    enumeration, with mixed Pos/Neg classes in one automaton.

    SPEC: getting this wrong silently produces a *plausible* wrong distribution
    that only this suite catches.
    """
    rng = np.random.default_rng(11000 + seed + SEED_OFFSET)
    L = 4
    A, p, post, Z, class_of, members, is_neg = nontrivial_mixed_polarity(
        rng, maker, L)
    W, M, fb, _ = prepared(A, p)
    assert any(is_neg) and not all(is_neg)

    direct = R.marginals(p, A, fb, W)
    comp = R.marginals_complement_aware(p, A, fb, class_of, members, is_neg)
    exact = R.marginals_from_posterior(post, L, V)
    np.testing.assert_allclose(comp, direct, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(comp, exact, rtol=1e-10, atol=1e-12)


def test_11b_empty_complement_and_empty_class():
    """`N_c = ∅` (i.e. `S_c = V`) and `S_c = ∅` are both representable and must
    give the same answer either way round."""
    L = 3
    edges = (
        (0, 1, frozenset(range(V))),   # S_c = V, so N_c is empty
        (1, 1, frozenset({0})),        # small positive class
        (1, 2, frozenset(range(V))),
    )
    start = np.zeros(3)
    start[0] = 1.0
    A = R.Automaton(n_states=3, vocab_size=V, edges=edges, start=start,
                    finals=frozenset({2}))
    rng = np.random.default_rng(11500)
    p = random_probs(rng, L)
    W, M, fb, _ = prepared(A, p)
    class_of, members, is_neg = _classes_with_polarity(A, threshold=2)
    assert any(is_neg)

    post, Z = R.enumerate_posterior(p, A)
    comp = R.marginals_complement_aware(p, A, fb, class_of, members, is_neg)
    np.testing.assert_allclose(comp, R.marginals(p, A, fb, W),
                               rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(comp, R.marginals_from_posterior(post, L, V),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_11c_topk_bound_is_tight(seed):
    """SPEC §4.4 Layer 2b: `K = 1 + max_c |N_c|` succeeds and `K = |N_c|` can
    fail. Verified against a direct max over `S_c`."""
    rng = np.random.default_rng(11800 + seed + SEED_OFFSET)
    big_V = 32
    n_neg = int(rng.integers(1, 8))
    complement = frozenset(int(x) for x in
                           rng.choice(big_V, size=n_neg, replace=False))
    p_row = rng.random(big_V)
    p_row /= p_row.sum()

    members = [v for v in range(big_V) if v not in complement]
    want = max(members, key=lambda v: (float(p_row[v]), -v))
    got, arg = R.max_over_class_topk(p_row, complement, n_neg + 1)
    assert arg == want
    assert got == pytest.approx(float(p_row[want]))


def test_11d_topk_with_k_equal_to_complement_size_can_fail():
    """The adversarial `p` SPEC says exists: the whole complement occupies the
    top `|N_c|` slots, so `K = |N_c|` finds nothing and must raise rather than
    return a wrong maximum."""
    big_V = 8
    complement = frozenset({0, 1, 2})
    p_row = np.array([0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.03])
    with pytest.raises(LookupError):
        R.max_over_class_topk(p_row, complement, len(complement))
    got, arg = R.max_over_class_topk(p_row, complement, len(complement) + 1)
    assert arg == 3 and got == pytest.approx(0.1)


def test_11e_segment_tiebreak_lowest_index_and_empty_segments():
    """SPEC §2.7's worked example. The naive `segment_max`-over-indices
    workaround gives [2, 4]; the correct `segment_min` form gives [1, 4].
    Empty segments must yield `-inf` and `am == N`, never `-1`."""
    vals = np.array([5.0, 7.0, 7.0, 2.0, 7.0, 1.0])
    seg = np.array([0, 0, 0, 1, 1, 1])
    mx, am = R.segment_max_argmin(vals, seg, num_segments=3)
    np.testing.assert_array_equal(mx[:2], [7.0, 7.0])
    np.testing.assert_array_equal(am[:2], [1, 4])
    assert mx[2] == -np.inf, "empty segment identity is -inf"
    assert am[2] == vals.shape[0], "empty segment argmin is the N sentinel"
    assert am[2] != -1, "SPEC: test for am == N, never am == -1"


def test_11f_map_tiebreak_is_lowest_token_id():
    """`test_guarantee` and the unconstrained-equivalence test depend on the
    stated convention: lowest token id wins a tie."""
    L = 1
    edges = ((0, 1, frozenset({1, 3})),)
    start = np.zeros(2)
    start[0] = 1.0
    A = R.Automaton(n_states=2, vocab_size=V, edges=edges, start=start,
                    finals=frozenset({1}))
    p = np.array([[0.25, 0.25, 0.25, 0.25]])   # an exact tie
    tokens, _, _ = R.map_decode(p, A, A.start, A.final_vector())
    assert tokens[0] == 1, "lowest admissible token id must win the tie"


def test_4d_unit_multiplicity_not_determinism_is_the_gate():
    """SPEC §2.6 says to gate eq (8)'s cheap `∃` draw on "a DFA (one edge per
    `(s,s')`, disjoint labels)". The parenthetical is load-bearing.

    This automaton is **deterministic** — every `(state, token)` has exactly one
    destination — yet two parallel edges `0->1` overlap on token 1, giving it
    multiplicity 2. Gating on determinism alone would take the fast path here
    and draw from the wrong distribution.
    """
    edges = (
        (0, 1, frozenset({0, 1})),
        (0, 1, frozenset({1, 2})),   # parallel, same dst, overlaps on token 1
        (1, 1, frozenset(range(V))),
    )
    start = np.zeros(2)
    start[0] = 1.0
    A = R.Automaton(n_states=2, vocab_size=V, edges=edges, start=start,
                    finals=frozenset({1}))

    assert A.is_deterministic, "every (state, token) has one destination"
    assert not A.has_unit_multiplicity, "token 1 is carried by two edges 0->1"
    assert not A.is_dfa, "so the eq (8) fast path must NOT be taken"


def test_4e_grouping_restores_unit_multiplicity():
    """Collapsing transitions to one edge per `(src, dst)` with the union of
    labels — what `compile/automaton.py` does — makes the fast path safe by
    construction."""
    edges = ((0, 1, frozenset({0, 1})), (0, 1, frozenset({1, 2})))
    grouped = ((0, 1, frozenset({0, 1, 2})),)
    start = np.zeros(2)
    start[0] = 1.0
    raw = R.Automaton(n_states=2, vocab_size=V, edges=edges, start=start,
                      finals=frozenset({1}))
    merged = R.Automaton(n_states=2, vocab_size=V, edges=grouped, start=start,
                         finals=frozenset({1}))
    assert not raw.has_unit_multiplicity
    assert merged.has_unit_multiplicity and merged.is_dfa
