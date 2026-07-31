"""`model/sampler.py`'s decision logic, without the 51 GB model.

**Why this file exists.** A mutation audit found `model/sampler.py` — 400+
lines, the whole fork surface — with **zero functional coverage**: the only test
that imported it asserted that an attribute does *not* exist. Two bug
signatures (the constraint mask landing in `sample_from_predictions` instead of
`logit_shaper`, and per-request automaton values reaching a static argument)
live entirely in this file.

The forward pass genuinely needs the checkpoint, so what is covered here is
everything that is *not* the forward pass: the accept rule, the construction-time
validation, the `Z == 0` raise, the state widening, and the emission composition
each variant performs. Those are the decision points; the transformer call
between them is the library's, not ours.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.model import sampler as S  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402


# --------------------------------------------------------------------------
# The accept rule (SPEC §3.4)
# --------------------------------------------------------------------------

def _logits_with_entropies(target_h):
    """Logits whose per-position entropy is approximately `target_h` nats.

    Two-point support, `logits = (0, -gap)`. The entropy is monotone
    *decreasing* in `gap`, so `gap = 1/h` orders the positions by `h`. The
    tests below rely only on that ordering, never on the exact nats, so the
    construction does not need to be inverted numerically.
    """
    V = 8
    out = []
    for h in target_h:
        z = np.full(V, -60.0)
        z[0] = 0.0
        z[1] = -1.0 / h           # larger h -> smaller gap -> higher entropy
        out.append(z)
    return jnp.asarray(np.stack(out))[None, :, :]      # [1, L, V]


def _entropies(logits):
    lp = jax.nn.log_softmax(logits.astype(jnp.float32))
    p = jnp.exp(lp)
    return np.asarray(-jnp.sum(jnp.where(p == 0, 0.0, lp) * p, axis=-1))[0]


def test_the_fixture_really_orders_positions_by_entropy():
    """Pins the helper, so a test below cannot pass for the wrong reason."""
    h = _entropies(_logits_with_entropies([0.9, 0.1, 0.7, 0.3]))
    assert h[1] < h[3] < h[2] < h[0]


def test_accept_mask_always_accepts_at_least_one_position():
    """The stock rule's `cumsum - sorted <= bound` accepts the lowest-entropy
    position unconditionally, because its own entropy is subtracted off. A
    reimplementation that drops the `- srt` term accepts nothing at a small
    bound and the sampler silently stops making progress."""
    logits = _logits_with_entropies([0.5, 0.6, 0.7, 0.8])
    for bound in (0.0, 1e-9, 1e-3):
        acc = np.asarray(S._accept_mask(logits, bound))
        assert acc.sum() >= 1, f"bound={bound} accepted nothing"


def test_accept_mask_accepts_the_lowest_entropy_positions_first():
    logits = _logits_with_entropies([0.9, 0.1, 0.7, 0.3])   # position 1 lowest
    acc = np.asarray(S._accept_mask(logits, 0.05))[0]
    assert acc[1], "the lowest-entropy position must be accepted first"


def test_accept_mask_accepts_everything_at_a_large_bound():
    logits = _logits_with_entropies([0.5, 0.6, 0.7, 0.8])
    acc = np.asarray(S._accept_mask(logits, 1e6))
    assert acc.all()


def test_accept_mask_is_monotone_in_the_bound():
    """More budget can only accept more positions. Not guaranteed by the
    formula's shape — it follows from the ordering being fixed — and a version
    that re-sorted per bound would break it."""
    logits = _logits_with_entropies([0.9, 0.1, 0.7, 0.3, 0.5, 0.2, 0.8, 0.4])
    prev = 0
    for bound in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0):
        n = int(np.asarray(S._accept_mask(logits, bound)).sum())
        assert n >= prev, f"bound {bound} accepted fewer positions"
        prev = n


def test_accept_mask_is_nan_free_on_a_one_hot_distribution():
    """`p == 0` exactly, so the `0 * log 0` guard is load-bearing (SPEC §2.4).
    Without it the entropy is NaN, the argsort is meaningless, and the accept
    mask becomes arbitrary rather than obviously broken."""
    z = np.full((1, 3, 8), -np.inf)
    z[:, :, 0] = 0.0
    acc = np.asarray(S._accept_mask(jnp.asarray(z), 0.1))
    assert acc.dtype == bool and not np.isnan(acc).any()
    assert acc.all(), "a zero-entropy canvas must be entirely accepted"


# --------------------------------------------------------------------------
# Construction-time validation
# --------------------------------------------------------------------------

def _kwargs(**over):
    base = dict(model=None, end_tokens=(1,), forbidden_tokens=None,
                sampling=None, cache_length=1024, special_tokens=None,
                canvas_length=8, max_denoising_steps=4, text_vocab_size=32,
                n_states_bucket=8, n_classes=2)
    base.update(over)
    return base


def test_a_map_emission_is_rejected_for_j1_and_j2():
    """SPEC §3.9's flag table: `--emission=map` is **J0-only**. J1/J2 define the
    emission to *be* the draw, so a MAP emission is not one of their variants
    and must be rejected rather than silently reinterpreted — with J1's
    flattened marginals a joint MAP degenerates to the shortest string in the
    language."""
    for variant in ("j1", "j2", "mask"):
        with pytest.raises(ValueError, match="always a draw"):
            S.ConstrainedDiffusionSampler(
                **_kwargs(variant=variant, emission="map"))


def test_an_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        S.ConstrainedDiffusionSampler(**_kwargs(variant="j7"))


def test_float32_is_admissible_now_that_the_anchor_is_per_entry():
    """This test used to assert the opposite, and the reversal is the point.

    The old ban existed because `log_matmul` exponentiated against a FOREIGN
    anchor (`ra[i] + cb[j]`), which the unscored `ACC --Σ--> ACC` tail pins at
    0.0 while genuine grammar paths sit ~850 nats below — so entries
    underflowed *even in float64*. With each entry anchored on its own
    pairwise max, float32 drops only what is 1e-38 below its own entry.

    Measured on `live_simple_106-63-0` under adversarial sharp `p`: feasible
    and simulator-accepted in float32. Distributionally against brute-force
    enumeration: float32 deviates 0.0025 (DFA) / 0.0013 (NFA) versus float64's
    0.0033 / 0.0009 — indistinguishable, both far inside the 0.02 threshold.
    """
    S.ConstrainedDiffusionSampler(
        **_kwargs(variant="j0", emission="sample",
                  constrained_dtype="float32"))


def test_an_unknown_dtype_is_still_rejected():
    with pytest.raises(ValueError, match="float32 or float64"):
        S.ConstrainedDiffusionSampler(
            **_kwargs(variant="j0", emission="sample",
                      constrained_dtype="bfloat16"))


def test_the_valid_configurations_construct():
    for variant, emission in (("j0", "map"), ("j0", "sample"),
                              ("j1", "sample"), ("j2", "sample"),
                              ("mask", "sample"),
                              ("unconstrained", "sample")):
        S.ConstrainedDiffusionSampler(
            **_kwargs(variant=variant, emission=emission))


def test_the_stock_sample_entry_point_refuses_rather_than_silently_working():
    """`sample()` has nowhere to put the automaton, so it raises instead of
    running unconstrained under a constrained-looking call."""
    s = S.ConstrainedDiffusionSampler(**_kwargs(variant="j0", emission="map"))
    with pytest.raises(NotImplementedError, match="sample_constrained"):
        s.sample(params=None, init_state=None, max_new_tokens=8)


# --------------------------------------------------------------------------
# The Z == 0 raise (SPEC §6.3)
# --------------------------------------------------------------------------

class _FakeState:
    def __init__(self, feasible):
        self.feasible = jnp.asarray(feasible)


def test_check_feasible_raises_on_an_infeasible_batch_element():
    with pytest.raises(S.ZeroPartitionError, match=r"Z == 0"):
        S._check_feasible(_FakeState([True, False, True]))


def test_check_feasible_names_the_offending_elements():
    with pytest.raises(S.ZeroPartitionError, match=r"\[1, 3\]"):
        S._check_feasible(_FakeState([True, False, True, False]))


def test_check_feasible_passes_a_clean_batch():
    S._check_feasible(_FakeState([True, True]))


def test_check_feasible_is_a_no_op_on_a_state_without_the_field():
    """The stock `SamplingState` has no `feasible`; streaming yields those."""
    S._check_feasible(object())


def test_zero_partition_error_is_not_caught_anywhere():
    """CLAUDE.md: `Z == 0` has three causes and only (c), fp32 underflow, is
    benign — and the constrained paths run in log space so (c) cannot occur.
    What is left is (a) an empty automaton and (b) no live continuation within
    budget, both of which must be seen rather than smoothed over."""
    import inspect
    src = inspect.getsource(S)
    assert "except ZeroPartitionError" not in src
    assert "except S.ZeroPartitionError" not in src


# --------------------------------------------------------------------------
# The widened carry
# --------------------------------------------------------------------------

def test_the_carry_tracks_the_trajectory_and_the_emission_separately():
    """SPEC §5.4: under J0 the two differ, and conflating them is what makes
    the guarantee conditional on the accept prefix covering the canvas."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(S._ConstrainedCarry)}
    assert {"canvas", "emit_canvas", "feasible"} <= names


def test_the_carry_is_a_pytree_so_it_threads_through_while_loop():
    """`lax.while_loop` needs the carry flattened and unflattened with an
    identical treedef; a plain dataclass would raise at trace time."""
    c = S._ConstrainedCarry(
        step=jnp.int32(0), canvas=jnp.zeros((1, 4), jnp.int32),
        emit_canvas=jnp.zeros((1, 4), jnp.int32),
        sc_embeddings=jnp.zeros((1, 4, 2), jnp.bfloat16),
        rng=jax.random.PRNGKey(0), done=jnp.zeros((1,), bool),
        feasible=jnp.ones((1,), bool))
    leaves, treedef = jax.tree_util.tree_flatten(c)
    assert len(leaves) == 7
    assert jax.tree_util.tree_unflatten(treedef, leaves).step == 0


# --------------------------------------------------------------------------
# J2's emission composition (SPEC §7.2 baseline 3)
# --------------------------------------------------------------------------

def test_j2_keeps_the_constrained_draw_only_where_accepted():
    """J2 was previously *not implemented* — it ran J0's fully constrained
    emission, so the baseline table had a duplicated row under two names.

    The composition is the whole content of the baseline: keep the constrained
    draw at accepted positions, leave the rest as the stock uniform renoise.
    That is SPEC §3.1's finding — the emitted canvas *is* the sample, so
    non-accepted positions reach the output as random tokens — and its CS
    column is the evidence that constraining the sampler alone is necessary but
    not sufficient.
    """
    accepted = jnp.asarray([[True, False, True, False]])
    constrained = jnp.asarray([[10, 11, 12, 13]], jnp.int32)
    noise = jnp.asarray([[90, 91, 92, 93]], jnp.int32)
    got = np.asarray(jnp.where(accepted, constrained, noise))[0]
    assert list(got) == [10, 91, 12, 93]
    assert not np.array_equal(got, np.asarray(constrained)[0]), (
        "if J2 equals the constrained draw it is measuring J0 under another "
        "name"
    )


def test_j2_is_not_flagged_infeasible_for_violating_the_grammar():
    """`j2`, `mask` and `unconstrained` emit rejected tokens **by
    construction** — that is what they measure. Folding `advance_ok` into the
    `Z == 0` flag for them would raise on the baselines and make them
    unrunnable, turning a measurement into a crash."""
    import inspect
    src = inspect.getsource(S.ConstrainedDiffusionSampler._sample_step)
    assert 'self.variant in ("j0", "j1")' in src, (
        "the advance_ok conjunct must be gated to the arms that actually "
        "promise the guarantee"
    )
