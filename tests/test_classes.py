"""Tests for the token-set representation. SPEC §4.4.

The properties that matter here are the ones that fail *silently*: a shared
polarity array, or a `K` too small for the topk recovery of
`max_{v in S_c} p_i(v)`, both produce a plausible wrong distribution rather
than an error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import jax

# `class_weights` is exercised below at class masses down to 1e-40; without x64
# every `jnp.float64` here is silently a float32 and the sweep would fail for
# the dtype rather than for the kernel. Set at import, as the other audit
# modules do.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.compile.classes import (  # noqa: E402
    DEFAULT_K_MAX,
    build_tables,
    intern_labels,
)
from diffgemma_fa.infer import marginals as MG  # noqa: E402

V = 64  # small vocab so complements are enumerable


def members(tables, c: int, which: str) -> set[int]:
    """Reconstruct class `c`'s true member set from a CSR table."""
    if which == "sum":
        indptr, indices, is_neg = tables.sum_indptr, tables.sum_indices, tables.sum_is_neg
    else:
        indptr, indices, is_neg = tables.max_indptr, tables.max_indices, tables.max_is_neg
    stored = set(int(x) for x in indices[indptr[c]:indptr[c + 1]])
    return set(range(tables.vocab_size)) - stored if is_neg[c] else stored


def test_interning_deduplicates():
    labels = [frozenset({1, 2}), frozenset({3}), frozenset({2, 1}), frozenset({3})]
    class_id, classes = intern_labels(labels)
    assert len(classes) == 2
    assert class_id[0] == class_id[2]
    assert class_id[1] == class_id[3]


def test_dedup_ratio_is_edges_per_class():
    labels = [frozenset({1})] * 10 + [frozenset({2})] * 5
    t = build_tables(labels, vocab_size=V)
    assert t.n_edges == 15
    assert t.n_classes == 2
    assert t.dedup_ratio == pytest.approx(7.5)


@pytest.mark.parametrize("seed", range(30))
def test_both_tables_roundtrip_to_the_true_member_set(seed: int):
    """Whatever the polarity, decoding a table must give back S_c exactly."""
    rng = np.random.default_rng(seed)
    labels = []
    for _ in range(rng.integers(1, 8)):
        k = int(rng.integers(0, V + 1))
        labels.append(frozenset(int(x) for x in rng.choice(V, size=k, replace=False)))
    t = build_tables(labels, vocab_size=V)
    _, classes = intern_labels(labels)
    for c, true_set in enumerate(classes):
        assert members(t, c, "sum") == set(true_set)
        assert members(t, c, "max") == set(true_set)


def test_sum_polarity_uses_the_size_rule():
    big = frozenset(range(V - 3))       # |S_c| > V/2 -> store complement
    small = frozenset({0, 1})           # -> store positively
    t = build_tables([big, small], vocab_size=V)
    _, classes = intern_labels([big, small])
    i_big = classes.index(big)
    i_small = classes.index(small)
    assert t.sum_is_neg[i_big]
    assert not t.sum_is_neg[i_small]
    # The complement really is the smaller thing to store.
    assert t.sum_indptr[i_big + 1] - t.sum_indptr[i_big] == 3


def test_max_and_sum_polarity_are_independent_arrays():
    """SPEC §4.4: 'do not share the polarity flag array'. They must be distinct
    objects AND must be allowed to disagree."""
    big = frozenset(range(V - 20))  # |S_c| > V/2, but |N_c| = 20
    t = build_tables([big], vocab_size=V, k_max=5, auto_k_max=False)
    assert t.sum_is_neg is not t.max_is_neg
    assert t.sum_is_neg[0], "sum table should negate a >V/2 class"
    assert not t.max_is_neg[0], "max table must NOT negate when |N_c| > k_max"


def test_k_max_bound_is_enforced():
    """K = k_max + 1 must exceed max_c |N_c| among negated classes."""
    cls = frozenset(range(V - 10))  # |N_c| = 10
    t = build_tables([cls], vocab_size=V, k_max=10, auto_k_max=False)
    assert t.max_is_neg[0]
    assert t.max_neg_size == 10
    assert t.k_max + 1 > t.max_neg_size


def test_auto_k_max_raises_the_threshold_rather_than_demoting():
    """The failure this guards: with a fixed small k_max, a 260k-member class
    falls back to positive storage and the CSR budget blows up by orders of
    magnitude. Phase 1 measured real |N_c| in the 988-1064 range against SPEC's
    original k_max default of 256."""
    cls = frozenset(range(V - 20))  # |N_c| = 20, above k_max=5
    auto = build_tables([cls], vocab_size=V, k_max=5, auto_k_max=True)
    fixed = build_tables([cls], vocab_size=V, k_max=5, auto_k_max=False)
    assert auto.max_is_neg[0] and auto.k_max >= 20
    assert not fixed.max_is_neg[0]
    assert auto.nnz_max < fixed.nnz_max


def test_default_k_max_is_not_256():
    """SPEC originally defaulted to 256; measured |N_c| on real BFCL is
    988-1064, so 256 would demote every real class."""
    assert DEFAULT_K_MAX >= 1064


def test_full_vocab_class_is_free():
    """The unscored tail (SPEC §3.5 trap 4) labels an edge with the whole
    vocabulary. Its complement is empty, so it must cost nothing to store and
    its emission mass is 1."""
    t = build_tables([frozenset(range(V))], vocab_size=V)
    assert t.max_is_neg[0] and t.sum_is_neg[0]
    assert t.nnz_sum == 0 and t.nnz_max == 0
    assert members(t, 0, "sum") == set(range(V))


def test_empty_class_is_representable():
    t = build_tables([frozenset()], vocab_size=V)
    assert members(t, 0, "sum") == set()


@pytest.mark.parametrize("mass", [1.0, 1e-8, 1e-20, 1e-40])
def test_emission_mass_matches_direct_sum(mass):
    """`W_c` from `marginals.class_weights` must equal the direct member sum.

    **Two things were wrong with the previous version of this test and both
    were invisible while it was green** (see `tests/test_audit_partition.py`):

    1. *It called nothing.* It re-derived `total - p[stored].sum()` in the test
       body and compared that to a direct sum — an identity of arithmetic, true
       of any table, and it would have stayed green through any change to the
       kernel it was named after. It now calls `class_weights`.
    2. *Its docstring stated the defective formula as the requirement.*
       `W_c = total - Σ_{v ∈ N_c} p` is catastrophic cancellation, not a
       specification; it destroyed the constrained language on 73/250 Countdown
       records. The requirement is `W_c = Σ_{v ∈ S_c} p_i(v)` — how it is
       reached is the kernel's business, and this test must not prescribe an
       implementation that a fix has to violate.

    The tolerance is **relative** and the class mass is swept down to 1e-40. An
    absolute `1e-12` on an `O(1)` fixture — what this asserted before — cannot
    distinguish a correct small answer from a destroyed one.
    """
    rng = np.random.default_rng(7)
    labels = [frozenset(range(V - 4)), frozenset({0, 5, 9})]
    t = build_tables(labels, vocab_size=V)
    _, classes = intern_labels(labels)

    # Concentrate `1 - mass` on a token in NEITHER class, so both classes' true
    # mass is O(mass) and the small-answer regime is actually reached.
    heavy = V - 1
    assert not any(heavy in s for s in classes), (
        "fixture broken: the heavy token must lie outside every class, or the "
        "masses stay O(1) and the sweep tests nothing"
    )
    p = rng.random(V) * mass / V
    p[heavy] = 1.0 - mass
    p = p / math.fsum(p)

    seg = np.repeat(np.arange(len(classes), dtype=np.int32),
                    np.diff(t.sum_indptr))
    W = np.asarray(MG.class_weights(
        jnp.asarray(p[:, None], dtype=jnp.float64),
        jnp.asarray(t.sum_indices), jnp.asarray(seg),
        jnp.asarray(t.sum_is_neg), len(classes)))[:, 0]

    for c, true_set in enumerate(classes):
        # `math.fsum` over the members: exactly rounded, so the reference has no
        # error budget of its own at any magnitude.
        want = math.fsum(p[v] for v in sorted(true_set))
        assert W[c] > 0.0, (
            f"class {c} (mass concentration {mass:g}) has true mass "
            f"{want:.6e} and came back {W[c]:.6e}"
        )
        assert W[c] == pytest.approx(want, rel=1e-12), (
            f"class {c}: {W[c]:.17g} against {want:.17g}"
        )
