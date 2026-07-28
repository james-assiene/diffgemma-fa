"""JAX paths differential-tested against the float64 reference. SPEC §6.2.

CLAUDE.md: *"If the fast path and the reference disagree, the fast path is
wrong."* So nothing here re-derives an expected value — every assertion compares
against `infer/reference.py`, which the Phase 2 exactness suite pins to
brute-force enumeration.

x64 is enabled so the comparison measures the *algorithm*, not float32 noise.
Production runs fp32; `test_numerics.py` and the fp32 case below cover that.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.infer import marginals as MG  # noqa: E402
from diffgemma_fa.infer import reference as R  # noqa: E402
from diffgemma_fa.infer import scans, tree  # noqa: E402

V = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def random_automaton(rng, n_states, nfa: bool):
    edges = []
    if nfa:
        for _ in range(int(rng.integers(n_states, 3 * n_states))):
            s, d = int(rng.integers(n_states)), int(rng.integers(n_states))
            k = int(rng.integers(1, V))
            edges.append((s, d, frozenset(int(x) for x in
                                          rng.choice(V, size=k, replace=False))))
        if edges:
            # Force a parallel pair that **overlaps but is not identical**.
            #
            # This used to be `edges.append(edges[0])` — a duplicate of the
            # same `(src, dst, label)`. That has ZERO power against eq (8)'s
            # multiplicity: duplicating a label scales `M` uniformly, and the
            # scale cancels in normalisation. A mutation audit measured the
            # consequence directly: the `∃`-indicator form of eq (8) survived
            # all 721 tests, with deviation ≤ 9.4e-4 (five of six instances at
            # ~1e-18) against a 2e-2 threshold, because λ was exactly 0 on the
            # generated instances — not because the χ² threshold was loose.
            #
            # Distinct-but-overlapping labels on the *same* `(src, dst)` are
            # what make the `∃` form and the edge-multiplicity-weighted form
            # actually disagree: a token in the intersection contributes twice
            # under the weighted form and once under `∃`.
            s0, d0, lab0 = edges[0]
            lab0 = set(lab0)
            other = set(range(V)) - lab0
            if lab0 and other:
                overlap = {sorted(lab0)[0]}
                fresh = {sorted(other)[0]}
                edges.append((s0, d0, frozenset(overlap | fresh)))
            else:
                edges.append(edges[0])
    else:
        by_pair = {}
        for s in range(n_states):
            # Destinations are drawn from a small per-state fan-out rather than
            # uniformly over all states. Uniform fan-out scatters the V tokens
            # so thinly that a state pair almost never accumulates > V/2 labels,
            # and then no class is ever stored negatively — leaving the
            # complement-aware path untested on DFAs.
            fan = rng.choice(n_states, size=min(n_states, 2), replace=False)
            for v in range(V):
                if rng.random() < 0.75:
                    d = int(fan[int(rng.integers(len(fan)))])
                    by_pair.setdefault((s, d), set()).add(v)
        edges = [(s, d, frozenset(vs)) for (s, d), vs in sorted(by_pair.items())]
    start = np.zeros(n_states)
    start[0] = 1.0
    finals = frozenset(s for s in range(n_states) if rng.random() < 0.4) \
        or frozenset({n_states - 1})
    return R.Automaton(n_states=n_states, vocab_size=V, edges=tuple(edges),
                       start=start, finals=finals)


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
    for c, s in enumerate(stored):
        indptr[c + 1] = indptr[c] + len(s)
    indices = np.array([x for s in stored for x in s], np.int32)
    seg = np.array([c for c, s in enumerate(stored) for _ in s], np.int32)
    return (np.array(class_of, np.int32), members, np.array(is_neg),
            indices, indptr, seg)


def viable(rng, n_states, L, nfa, tries=60, require_negated=False):
    """A usable instance.

    `require_negated` searches for one with at least one **negated** class
    rather than skipping when the random draw happens not to produce one —
    the complement-aware path is the whole point of those tests, and a silent
    skip is exactly the "silently capped coverage" CLAUDE.md forbids.
    """
    for _ in range(tries):
        A = random_automaton(rng, n_states, nfa)
        p = rng.random((L, V)) + 0.05
        p /= p.sum(axis=1, keepdims=True)
        W = R.edge_weights(p, A)
        M = R.transition_matrices(W, A)
        fb = R.forward_backward(M, A.start, A.final_vector())
        if not (np.isfinite(fb.log_Z) and A.n_edges > 0):
            continue
        if require_negated and not class_tables(A)[2].any():
            continue
        return A, p, W, M, fb
    raise AssertionError(
        f"no viable instance in {tries} tries (require_negated={require_negated}); "
        "the generator, not the test, needs fixing"
    )


# ---------------------------------------------------------------------------
# Class SpMM (SPEC §4.4 layers 1-4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nfa", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_class_spmm_matches_reference(nfa, seed):
    rng = np.random.default_rng(70000 + seed)
    L = 8
    A, p, W, M, fb = viable(rng, int(rng.integers(3, 9)), L, nfa)
    class_of, members, is_neg, indices, indptr, seg = class_tables(A)

    W_c = MG.class_weights(jnp.asarray(p.T), jnp.asarray(indices),
                           jnp.asarray(seg), jnp.asarray(is_neg), len(members))
    W_e = MG.edge_weights(W_c, jnp.asarray(class_of))
    np.testing.assert_allclose(np.asarray(W_e).T, W, rtol=1e-12, atol=1e-14)

    M_j = MG.transition_matrices(
        W_e,
        jnp.asarray(np.array([e[0] for e in A.edges], np.int32)),
        jnp.asarray(np.array([e[1] for e in A.edges], np.int32)),
        A.n_states,
    )
    np.testing.assert_allclose(np.asarray(M_j), M, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("nfa", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_constrained_marginals_match_reference(nfa, seed):
    """The complement-aware path, against the reference's direct scatter."""
    rng = np.random.default_rng(71000 + seed)
    L = 8
    A, p, W, M, fb = viable(rng, int(rng.integers(3, 9)), L, nfa,
                            require_negated=True)
    class_of, members, is_neg, indices, indptr, seg = class_tables(A)
    assert is_neg.any(), "the complement-aware path must actually be exercised"

    tr = scans.up_sweep(jnp.asarray(M))
    a, b, _, _ = scans.prefix_suffix(tr, jnp.asarray(A.start),
                                     jnp.asarray(A.final_vector()))
    q = MG.constrained_marginals(
        jnp.asarray(p.T), a, b,
        jnp.asarray(np.array([e[0] for e in A.edges], np.int32)),
        jnp.asarray(np.array([e[1] for e in A.edges], np.int32)),
        jnp.asarray(class_of), jnp.asarray(indices), jnp.asarray(indptr),
        jnp.asarray(is_neg), len(members))
    q_ref = R.marginals(p, A, fb, W)
    np.testing.assert_allclose(np.asarray(q), q_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(q).sum(axis=1), np.ones(L), rtol=1e-12)


# ---------------------------------------------------------------------------
# Blelloch tree (SPEC §2.6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [4, 8, 16, 32])
@pytest.mark.parametrize("seed", range(4))
def test_up_sweep_retains_every_dyadic_product(L, seed):
    """The retained nodes must be exactly the aligned dyadic block products —
    the ones eq (7) needs and `associative_scan` discards."""
    rng = np.random.default_rng(72000 + seed + L)
    S = int(rng.integers(3, 9))
    M = np.abs(rng.random((L, S, S))) * 0.5
    tr = scans.up_sweep(jnp.asarray(M))
    blocks = R.block_products(M)

    assert tr.n_nodes == 2 * L - 1, "Blelloch has 2L-1 nodes; Kogge-Stone has L log L"
    for k, level in enumerate(tr.levels):
        w = 1 << k
        assert level.shape[0] == L // w
        for j in range(level.shape[0]):
            got = np.asarray(level[j]) * np.exp(float(tr.log_scales[k][j]))
            np.testing.assert_allclose(got, blocks[(j * w, (j + 1) * w)],
                                       rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("seed", range(6))
def test_prefix_suffix_matches_reference_forward_backward(seed):
    rng = np.random.default_rng(73000 + seed)
    L = 16
    A, p, W, M, fb = viable(rng, int(rng.integers(3, 9)), L, nfa=True)
    tr = scans.up_sweep(jnp.asarray(M))
    a, b, la, lb = scans.prefix_suffix(tr, jnp.asarray(A.start),
                                       jnp.asarray(A.final_vector()))
    for i in range(L + 1):
        dot = float(np.asarray(a[i]) @ np.asarray(b[i]))
        if dot <= 0:
            continue
        log_Z = np.log(dot) + float(la[i]) + float(lb[i])
        assert log_Z == pytest.approx(fb.log_Z, rel=1e-9), f"at i={i}"


def test_operand_order_is_left_block_first():
    """SPEC §2.6(c): `reverse=True` reverses the operand order and for
    non-commutative matmul that silently yields wrong probabilities."""
    rng = np.random.default_rng(73500)
    L, S = 8, 4
    M = np.abs(rng.random((L, S, S))) * 0.5
    tr = scans.up_sweep(jnp.asarray(M), normalize=False)
    fwd = np.eye(S)
    for k in range(L):
        fwd = fwd @ M[k]
    rev = np.eye(S)
    for k in range(L - 1, -1, -1):
        rev = rev @ M[k]
    root = np.asarray(tr.root)
    np.testing.assert_allclose(root, fwd, rtol=1e-9, atol=1e-13)
    assert not np.allclose(root, rev), "test would not detect a reversed order"


def test_non_power_of_two_length_is_rejected_loudly():
    """SPEC §2.6(d): pad with identities; silently accepting would give a wrong
    dyadic decomposition."""
    with pytest.raises(ValueError, match="power of two"):
        scans.up_sweep(jnp.zeros((6, 3, 3)))


# ---------------------------------------------------------------------------
# Tree sampling (SPEC eq (7)/(8))
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nfa", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_tree_samples_are_accepted(nfa, seed):
    rng = np.random.default_rng(74000 + seed)
    L = 8
    A, p, W, M, fb = viable(rng, int(rng.integers(3, 8)), L, nfa)
    class_of, members, is_neg, indices, indptr, seg = class_tables(A)

    tr = scans.up_sweep(jnp.asarray(M))
    key = jax.random.PRNGKey(seed)
    for t in range(25):
        k1, k2 = jax.random.split(jax.random.fold_in(key, t))
        states = tree.sample_states(tr, jnp.asarray(A.start),
                                    jnp.asarray(A.final_vector()), k1)
        toks, _valid = tree.sample_tokens(
            jnp.asarray(p.T), states,
            jnp.asarray(np.array([e[0] for e in A.edges], np.int32)),
            jnp.asarray(np.array([e[1] for e in A.edges], np.int32)),
            jnp.asarray(class_of), jnp.asarray(indices), jnp.asarray(indptr),
            jnp.asarray(is_neg), len(members), k2)
        toks = [int(x) for x in toks]
        assert R.accepts(A, toks), f"tree sample rejected: {toks}"


def draw_batch(A, p, M, class_of, members, is_neg, indices, indptr, n, seed=0):
    """`n` tree samples in **one** vmapped dispatch.

    Drawing them one at a time costs `2n` separate JAX dispatches, which made
    this test take 19 minutes on its own — untenable for a suite CLAUDE.md says
    to run constantly. `vmap` over the key batch makes it seconds and exercises
    the same code path.
    """
    tr = scans.up_sweep(jnp.asarray(M))
    src = jnp.asarray(np.array([e[0] for e in A.edges], np.int32))
    dst = jnp.asarray(np.array([e[1] for e in A.edges], np.int32))
    p_vl = jnp.asarray(p.T)
    a0 = jnp.asarray(A.start)
    bf = jnp.asarray(A.final_vector())
    cid, idx, iptr, neg = (jnp.asarray(class_of), jnp.asarray(indices),
                           jnp.asarray(indptr), jnp.asarray(is_neg))

    def one(k):
        k1, k2 = jax.random.split(k)
        states = tree.sample_states(tr, a0, bf, k1)
        return tree.sample_tokens(p_vl, states, src, dst, cid, idx, iptr, neg,
                                  len(members), k2)[0]

    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return np.asarray(jax.jit(jax.vmap(one))(keys))


@pytest.mark.parametrize("nfa", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_tree_sampler_matches_exact_posterior(nfa, seed):
    """Distributional agreement with brute-force enumeration, on an NFA too.

    This is the JAX-side counterpart of exactness test 4 — the one that catches
    an `∃`-form eq (8).
    """
    rng = np.random.default_rng(75000 + seed * 17 + int(nfa))
    L = 4
    for _ in range(60):
        A, p, W, M, fb = viable(rng, 4, L, nfa)
        post, Z = R.enumerate_posterior(p, A)
        if Z > 0 and len(post) >= 2:
            break
    else:
        raise AssertionError("no non-degenerate instance; fix the generator")

    class_of, members, is_neg, indices, indptr, seg = class_tables(A)
    n = 20000
    draws = draw_batch(A, p, M, class_of, members, is_neg, indices, indptr,
                       n, seed=seed)

    counts: dict[tuple, int] = {}
    for row in draws:
        k = tuple(int(x) for x in row)
        counts[k] = counts.get(k, 0) + 1
    for k in counts:
        assert k in post, f"sampled a string outside the support: {k}"
    err = max(abs(counts.get(k, 0) / n - post[k]) for k in post)
    assert err < 0.02, f"max deviation from the exact posterior: {err:.4g}"


# ---------------------------------------------------------------------------
# max-plus MAP (SPEC §2.7)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nfa", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_maxplus_map_matches_reference(nfa, seed):
    """MAP is exact on NFAs too — multiplicity cannot change a max over
    strings — so this must agree with the reference on both."""
    rng = np.random.default_rng(76000 + seed)
    L = 8
    A, p, W, M, fb = viable(rng, int(rng.integers(3, 8)), L, nfa)
    class_of, members, is_neg, indices, indptr, seg = class_tables(A)

    with np.errstate(divide="ignore"):
        logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), scans.NEG_SENTINEL)

    # class_max / class_argmax, complement-aware via the reference helper.
    C = len(members)
    class_max = np.full((C, L), scans.NEG_SENTINEL)
    class_arg = np.zeros((C, L), np.int32)
    for c, mem in enumerate(members):
        if not mem:
            continue
        idx = np.array(sorted(mem))
        for i in range(L):
            j = int(np.argmax(logp[i][idx]))
            class_max[c, i] = logp[i][idx[j]]
            class_arg[c, i] = idx[j]

    # M-tilde from the class table.
    S = A.n_states
    Mt = np.full((L, S, S), scans.NEG_SENTINEL)
    for e, (s, d, _) in enumerate(A.edges):
        c = class_of[e]
        for i in range(L):
            Mt[i, s, d] = max(Mt[i, s, d], class_max[c, i])

    tr = scans.up_sweep_maxplus(jnp.asarray(Mt))
    a_log = np.where(A.start > 0, 0.0, scans.NEG_SENTINEL)
    b_log = np.where(A.final_vector() > 0, 0.0, scans.NEG_SENTINEL)

    toks, states, score, feasible = tree.map_states_and_tokens(
        jnp.asarray(logp.T), tr,
        jnp.asarray(np.array([e[0] for e in A.edges], np.int32)),
        jnp.asarray(np.array([e[1] for e in A.edges], np.int32)),
        jnp.asarray(class_of), jnp.asarray(class_max), jnp.asarray(class_arg),
        jnp.asarray(a_log), jnp.asarray(b_log))

    ref_toks, ref_states, ref_score = R.map_decode(p, A, A.start,
                                                   A.final_vector())
    assert float(score) == pytest.approx(ref_score, rel=1e-9)
    toks = [int(x) for x in toks]
    assert R.accepts(A, toks), f"MAP output rejected: {toks}"
    got = float(sum(logp[i, t] for i, t in enumerate(toks)))
    assert got == pytest.approx(ref_score, rel=1e-9), (
        "the recovered tokens must realise the reported score"
    )


def test_maxplus_combine_matches_reference_semiring():
    rng = np.random.default_rng(77000)
    S = 6
    a = rng.standard_normal((S, S))
    b = rng.standard_normal((S, S))
    got = np.asarray(scans.maxplus_combine(jnp.asarray(a), jnp.asarray(b)))
    want = np.max(a[:, :, None] + b[None, :, :], axis=-2)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Entropy clamp (SPEC §2.4)
# ---------------------------------------------------------------------------

def test_entropy_from_q_is_nan_free_with_exact_zeros():
    q = jnp.asarray(np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    h = MG.entropy_from_q(q)
    assert np.isfinite(np.asarray(h)).all()
    assert float(h[0]) == pytest.approx(np.log(2.0), rel=1e-12)


@pytest.mark.parametrize("L", [4, 8, 16, 32])
@pytest.mark.parametrize("seed", range(3))
def test_log_space_tree_matches_the_linear_tree(L, seed):
    """`up_sweep_log` must agree with `up_sweep` wherever the linear form is
    still numerically valid — it is the same semiring, computed differently."""
    rng = np.random.default_rng(78000 + seed + L)
    S = int(rng.integers(3, 8))
    M = np.abs(rng.random((L, S, S))) * 0.5 + 1e-3
    lin = scans.up_sweep(jnp.asarray(M))
    log = scans.up_sweep_log(jnp.asarray(np.log(M)))

    blocks = R.block_products(M)
    for k in range(log.n_levels):
        w = 1 << k
        for j in range(log.levels[k].shape[0]):
            got = np.exp(np.asarray(log.levels[k][j]))
            np.testing.assert_allclose(got, blocks[(j * w, (j + 1) * w)],
                                       rtol=1e-9, atol=1e-13)


def test_log_space_survives_a_range_that_kills_the_linear_form():
    """The Phase 4 failure, reduced to a unit test.

    A matrix with one unit entry (standing in for the unscored `ACC --Σ--> ACC`
    tail, whose emission mass is exactly 1.0) and the rest at 1e-30: the linear
    product underflows to zero, the log-space product does not.
    """
    L, S = 16, 4
    M = np.full((L, S, S), 1e-30)
    M[:, 0, 0] = 1.0

    lin = np.asarray(scans.up_sweep(jnp.asarray(M, dtype=jnp.float32)).root)
    log = np.asarray(scans.up_sweep_log(jnp.asarray(np.log(M), dtype=jnp.float32)).root)

    # The dominant 1->1 path is 1e-30 * 1.0^14 * 1e-30 = 1e-60.
    assert log[1, 1] == pytest.approx(np.log(1e-60), rel=1e-4)
    assert lin[1, 1] == 0.0, "the linear form is expected to underflow here"
    assert np.isfinite(log).all()


# ---------------------------------------------------------------------------
# eq (8)'s edge multiplicity — a DETERMINISTIC probe (SPEC §2.6)
# ---------------------------------------------------------------------------
#
# A mutation audit found the `∃`-indicator form of eq (8) surviving all 721
# tests. Sharpening the random NFA generator to emit overlapping-but-distinct
# parallel labels was necessary but **not sufficient**: the randomly chosen
# instance still has to put a multiplicity-2 pair on a path the sampler
# actually walks, and measured, it usually does not.
#
# So this probe is constructed rather than sampled. It is the minimal instance
# on which the two forms disagree, and the disagreement is 0.167 in total
# variation — nowhere near a threshold question.

def _multiplicity_probe():
    """`0 --{a,b}--> 1` and `0 --{b,c}--> 1`, one token, uniform `p`.

    Token `b` lies on **two** distinct edges between the same state pair, so it
    carries twice the path mass:

        weighted:  p * mult = [1, 2, 1, 0] / 4  ->  [.25, .50, .25, 0]
        ∃-form:    p * 1    = [1, 1, 1, 0] / 3  ->  [.33, .33, .33, 0]

    `reference.enumerate_posterior` independently confirms the first — it
    enumerates strings and sums over latent edge paths, so the multiplicity
    falls out of the model definition rather than being asserted here.
    """
    A = R.Automaton(
        n_states=2, vocab_size=V,
        edges=((0, 1, frozenset({0, 1})), (0, 1, frozenset({1, 2}))),
        start=np.array([1.0, 0.0]), finals=frozenset({1}))
    p = np.full((1, V), 1.0 / V)
    return A, p


def test_the_reference_posterior_is_edge_multiplicity_weighted():
    """Pins the arbiter itself, so the JAX assertion below cannot be 'fixed' by
    quietly changing what the reference means."""
    A, p = _multiplicity_probe()
    post, Z = R.enumerate_posterior(p, A)
    assert Z > 0
    assert post[(1,)] == pytest.approx(0.5), "token on two edges must get 2x"
    assert post[(0,)] == pytest.approx(0.25)
    assert post[(2,)] == pytest.approx(0.25)
    assert (3,) not in post


def test_jax_sample_tokens_weights_by_edge_multiplicity():
    """The JAX-side counterpart. `∃` here is off by 8.3e-2 per token against
    a 1.5e-2 tolerance; on a DFA the two forms coincide exactly, which is why
    CLAUDE.md says to gate the fast path on `is_dfa` rather than assume it."""
    A, p = _multiplicity_probe()
    post, _ = R.enumerate_posterior(p, A)
    class_of, members, is_neg, indices, indptr, seg = class_tables(A)

    src = jnp.asarray([e[0] for e in A.edges], jnp.int32)
    dst = jnp.asarray([e[1] for e in A.edges], jnp.int32)
    cid = jnp.asarray(class_of, jnp.int32)   # already per-edge
    p_vl = jnp.asarray(p.T)                                    # [V, L=1]
    states = jnp.asarray([0, 1], jnp.int32)                    # the only path

    n = 40_000
    keys = jax.random.split(jax.random.PRNGKey(11), n)
    draw = jax.jit(jax.vmap(lambda k: tree.sample_tokens(
        p_vl, states, src, dst, cid, jnp.asarray(indices),
        jnp.asarray(indptr), jnp.asarray(is_neg), len(members), k)[0]))
    got = np.asarray(draw(keys)).reshape(-1)

    freq = np.bincount(got, minlength=V) / n
    assert freq[3] == 0.0, "token on no edge was sampled"
    for v in range(3):
        assert freq[v] == pytest.approx(post[(v,)], abs=1.5e-2), (
            f"token {v}: got {freq[v]:.4f}, exact {post[(v,)]:.4f} — the "
            f"∃-indicator form predicts {1/3:.4f} for every one of the three"
        )
