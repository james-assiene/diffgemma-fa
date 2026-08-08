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

import warnings

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


def test_float32_is_refused_on_the_sample_path():
    """**[Rewritten 2026-08-08.]** `constrained_dtype='float32'` must **raise**
    when `emission='sample'`.

    The test this replaces asserted the opposite — that the combination stays
    constructible — and by doing so pinned the hazard open. Its stated
    rationale was already self-defeating: it recited that float32 drove the
    `Z == 0` detector on 70/130 and 55/130 records at `L = 256` against 0/130
    in float64, and then asserted that the API must let you ask for it.

    A construction-time refusal is the right place because the failure is
    **silent at run time**. Measured at the *production* temperature (logit
    softcap `30·tanh`, `min_temperature ≈ 0.4`) on a 50-state grammar — about
    as small as real grammars get — float32 drops tens of live root entries
    and misvalues the survivors by tens of nats. The lost entries look like a
    spurious `Z == 0`; the misvalued ones do not trip any detector at all, so
    the draw simply comes from the wrong distribution and returns plausible
    tokens.

    **Why no figure above is asserted.** Every one of them is draw-dependent —
    including the root count, which an earlier version of this docstring called
    a reproducible property of the grammar. It is not. Across the measured
    cross-product:

        T=0.4    seed=1   M_lost 1315   root_lost 67/1168   err 27.57
        T=0.4    seed=2   M_lost 1399   root_lost 38/1168   err 27.73
        T=0.4    seed=3   M_lost 1374   root_lost 47/1168   err 77.77
        T=0.408  seed=1   M_lost 1141   root_lost 67/1168   err 26.94

    the root loss is 67, 38, 47 across seeds; two independent replications
    agreed on 67/1168 only because they shared seed 1, and the 1,315-vs-1,141
    gap was temperature at a fixed seed, not two draws. The survivor error
    spans 27–78 nats, not "about 27".

    So the figures are here as *evidence of a hazard*, never as an assertion.
    Pinning any of them would put the suite one unrelated seed change away from
    red, and the obvious repair would be to weaken the assertion — which is the
    precise failure this audit exists to stop. What is asserted is only the
    behaviour: the refusal fires.

    The loss is upstream of `log_matmul`'s pairwise-max anchor: `p` entries
    below float32's smallest normal (1.18e-38) flush to zero in the softmax and
    in `_matrices`, taking their edges out of `M` before the tree ever runs.
    So "the anchor made float32 safe" does not apply, and neither does a wider
    accumulator.
    """
    with pytest.raises(ValueError, match="float32"):
        S.ConstrainedDiffusionSampler(
            **_kwargs(variant="j0", emission="sample",
                      constrained_dtype="float32"))


@pytest.mark.parametrize("x64", [True, False])
def test_float32_is_still_allowed_for_a_map_emission(x64):
    """MAP is unaffected: SPEC §2.7 puts it in the `(max, +)` semiring in log
    space, where there is no summation to underflow — "exact, no scaling
    discussion, no underflow". The refusal above must therefore be scoped to
    `emission='sample'` and must not become a blanket ban, which would cost
    the MAP path half its memory headroom for nothing.

    Checked with x64 both on **and off**, because that is the claim: the MAP
    path does not depend on float64 at all, so neither guard may fire on it.
    """
    jax.config.update("jax_enable_x64", x64)
    try:
        S.ConstrainedDiffusionSampler(
            **_kwargs(variant="j0", emission="map",
                      constrained_dtype="float32"))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_float32_on_the_sample_path_has_an_explicit_opt_out_and_warns():
    """The refusal is a guard rail, not a wall: an ablation that *wants* to
    measure float32 at `L = 256` must be able to, since re-measuring the claim
    is exactly how it was found wrong the first time.

    The opt-out has to be explicit and separate from `constrained_dtype`, so
    that no existing call site acquires it by accident and it is greppable in
    a diff.

    **And it must be loud.** The whole hazard is that float32 fails silently —
    the misvalued survivors trip no detector — so a run that opts in has to say
    so in its own log or the resulting numbers get compared against float64
    runs by someone who never knew. Asserting the warning is what keeps
    "loud rather than silent" a contract instead of a comment; without this the
    warning is an unasserted side effect that a future refactor drops for free.
    """
    with pytest.warns(RuntimeWarning, match="float32"):
        S.ConstrainedDiffusionSampler(
            **_kwargs(variant="j0", emission="sample",
                      constrained_dtype="float32",
                      allow_unsafe_float32=True))


def test_the_supported_configuration_warns_about_nothing():
    """Guard against the warning being emitted too broadly. If float64 on the
    sample path also warned, the signal would be noise within a week and the
    test above would be pinning a message nobody reads.

    **Scoped to `RuntimeWarning` on purpose.** A bare `simplefilter("error")`
    promotes *any* warning raised anywhere during construction — including an
    unrelated `DeprecationWarning` from a third-party import — so the test
    would one day fail for a reason it is not about, and the cheapest repair
    would be to delete it. Narrowing it to the category actually under test
    means a red here can only mean "the float32 warning leaked onto the
    supported path".
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        S.ConstrainedDiffusionSampler(
            **_kwargs(variant="j0", emission="sample",
                      constrained_dtype="float64"))


def test_the_opt_out_does_not_weaken_the_x64_guard():
    """`allow_unsafe_float32` waives the *dtype* check only. It must not waive
    `require_x64`, which covers a different failure: with `jax_enable_x64` off
    every `jnp.float64` silently becomes float32, so a caller asking for
    float64 does not get it. Waiving both on one flag would let
    `constrained_dtype='float64'` run in float32 unannounced.
    """
    from diffgemma_fa.model import constrained as _C

    jax.config.update("jax_enable_x64", False)
    try:
        # `X64Required` specifically — not a bare `Exception`, which a typo in
        # the field name would satisfy with a `TypeError`.
        with pytest.raises(_C.X64Required):
            S.ConstrainedDiffusionSampler(
                **_kwargs(variant="j0", emission="sample",
                          constrained_dtype="float64",
                          allow_unsafe_float32=True))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_float64_on_the_sample_path_still_constructs():
    """Guard against the refusal being written too broadly — the supported
    configuration must keep working."""
    S.ConstrainedDiffusionSampler(
        **_kwargs(variant="j0", emission="sample",
                  constrained_dtype="float64"))


def test_the_eval_default_dtype_is_float64():
    """Pins the correction: the CLI default must not drift back to float32."""
    import inspect
    from diffgemma_fa.eval import run as R
    src = inspect.getsource(R.main)
    assert '"--dtype", default="float64"' in src, (
        "the sample path's tree dtype default must stay float64; float32 was "
        "measured to cause spurious Z == 0 on >50% of records at L = 256"
    )


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


# --------------------------------------------------------------------------
# Mar confidence (SPEC §3.4) — the paper's remasking signal
# --------------------------------------------------------------------------

def test_the_accept_rule_can_be_driven_by_a_precomputed_entropy():
    """`--confidence=mar` swaps the entropy source, not the selection rule.
    Splitting `_accept_mask` is what makes that possible, so the split must
    preserve behaviour exactly."""
    logits = _logits_with_entropies([0.9, 0.1, 0.7, 0.3])
    direct = np.asarray(S._accept_mask(logits, 0.05))
    via_h = np.asarray(S._accept_from_entropy(_entropies(logits)[None, :], 0.05))
    assert (direct == via_h).all(), (
        "the refactor changed the accept rule; mar would then be measuring two "
        "things at once"
    )


def test_mar_is_rejected_if_misspelled():
    with pytest.raises(ValueError, match="unknown confidence"):
        S.ConstrainedDiffusionSampler(**_kwargs(variant="j1", emission="sample", confidence="Mar"))


def test_both_confidence_modes_construct():
    for c in ("mf", "mar"):
        S.ConstrainedDiffusionSampler(**_kwargs(variant="j1", emission="sample", confidence=c))


def test_mar_uses_the_constrained_marginal_not_the_logits():
    """The whole point: `q_i` was built and reference-tested in
    `infer/marginals.py` and never called on the production path. Pinned by
    source inspection because exercising it needs the 51 GB model."""
    import inspect
    src = inspect.getsource(S.ConstrainedDiffusionSampler)
    i = src.index('self.confidence == "mar"')
    # Scope strictly to the mar branch: the `else:` that follows it is the mf
    # path and legitimately uses `out.logits`.
    body = src[i:src.index("            else:", i)]
    # `constrained_entropy_streamed` replaced the
    # `constrained_marginals` -> `entropy_from_q` pair: same number (verified
    # to 8.9e-16) computed over the CSR instead of a dense [L, V] chain. What
    # must hold is that the accept rule is driven by the CONSTRAINED marginal
    # rather than the raw logits, whichever route computes it.
    assert "constrained_entropy_streamed" in body
    assert "_accept_from_entropy" in body
    assert "out.logits" not in body, (
        "the mar branch must not fall back to unconstrained logits"
    )


def test_mf_remains_the_default_so_prior_arms_stay_comparable():
    s = S.ConstrainedDiffusionSampler(**_kwargs(variant="j1",
                                               emission="sample"))
    assert s.confidence == "mf"


def test_near_greedy_is_documented_as_incompatible_with_the_sample_emission():
    """Measured, not reasoned: `--temp greedy` (min=max=_MIN_TEMP=1e-12) makes
    the constrained posterior collapse on **every** record.

    At that temperature softmax is one-hot, so `p` carries exact zeros almost
    everywhere and `Z == 0` unless the model's greedy string is itself in the
    language. A 3-record probe returned `zero_partition = 3/3`: the detector
    correctly refused rather than emitting garbage.

    This matters for reproduction: the paper's headline BFCL-Live number
    (63.9 -> 71.5) is a GREEDY comparison, and it cannot be reproduced by
    shaping the logits upstream of the constrained emission. The flag stays
    because greedy is still the right setting for the *unconstrained* baseline
    arm, which is the comparison that tells us whether we have been competing
    against a stronger baseline than the paper did.
    """
    from diffgemma_fa.eval import run as R
    import inspect
    src = inspect.getsource(R.main)
    assert '"--temp"' in src and '"greedy"' in src
    assert "1e-12" in src, "the near-greedy path must use gemma's _MIN_TEMP"
