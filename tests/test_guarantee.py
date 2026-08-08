"""The guarantee. CLAUDE.md: *"The one thing that must not break."*

It asserts **two different things**, and CLAUDE.md flags conflating them as a
known trap:

    per block k:  δ*(A_k, canvas_k) ≠ ∅  and  ⊆ {s : d(s) ≤ R}     # viable prefix
    at end:       simulator.accepts(concat(canvas_0 .. canvas_K))   # this is CS

> A non-final block's canvas ends in a live-but-not-accepting state and is
> **not** accepted on its own. Asserting per-canvas acceptance makes this red on
> every multi-block generation, and the "fix" is to weaken it — which must not
> happen.

The guarantee comes from making the **emitted** canvas a constrained object
(SPEC §3.1) **plus** the three closures of §3.1b:

  1. a budget-aware terminal factor `b_L = 1[d(s) ≤ R]`;
  2. an automaton-aware stopping conjunct fed `emit_canvas`, not the trajectory;
  3. a `FREE` region excluding every `end_token` and `PAD`.

**Constraining the sampler alone is necessary but not sufficient.** Closure 2 is
tested here as a *property of the state set*; it is **not yet enforced in the
sampler** (see `docs/PHASE4_FINDINGS.md` §5), and
`test_closure_2_is_not_yet_enforced` records that honestly rather than letting
the file imply coverage it does not have.

If you find yourself weakening anything in this file to make something pass:
stop and write up why instead.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.compile import pipeline  # noqa: E402
from diffgemma_fa.compile.automaton import INF_DISTANCE  # noqa: E402
from diffgemma_fa.compile.validate import Simulator  # noqa: E402
from diffgemma_fa.compile.vocab import END_TOKENS, PAD_TOKEN  # noqa: E402
from gemma.diffusion import _sampler as _diffusion_sampler  # noqa: E402
from diffgemma_fa.model import constrained as C  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402


def traced(a) -> Automaton:
    return Automaton(
        edge_src=jnp.asarray(a.edge_src), edge_dst=jnp.asarray(a.edge_dst),
        edge_class=jnp.asarray(a.edge_class),
        edge_valid=jnp.ones(a.n_edges, bool),
        csr_indices=jnp.asarray(a.tables.sum_indices),
        csr_indptr=jnp.asarray(a.tables.sum_indptr),
        is_neg=jnp.asarray(a.tables.sum_is_neg),
        d=jnp.asarray(a.d), is_final=jnp.asarray(a.is_final),
        active=jnp.asarray(a.start_vector),
    )


# -- shims for the pair-returning APIs -------------------------------------
# `joint_map`/`joint_draw`/`advance_states` each return a feasibility flag
# beside their value (SPEC §6.3's Z == 0 detector). These wrappers keep the
# existing assertions readable AND assert the flag, so a silent Z == 0 fails
# the test rather than sliding past it.
#
# A `_map_tokens` wrapper used to sit here and was called by nothing: every
# multi-block assertion in this file went through `joint_draw`, so
# `--emission=map` — the DEFAULT of SPEC §3.9's flag table — was never run
# against the proposition it relies on, while a dead MAP helper implied it was.
# The emission is a parameter of the two multi-block tests now.


def _adv(fn):
    def go(*a, **kw):
        active, ok = fn(*a, **kw)
        assert bool(ok), "state set emptied while advancing across the canvas"
        return active
    return go


@pytest.fixture(scope="module")
def long_grammar():
    """A grammar that **cannot** complete inside one short canvas.

    Many required string fields, so the shortest member of the language is far
    longer than the block size used below. That is what forces the multi-block
    path — the one where a non-final canvas is live-but-not-accepting, and where
    a naive per-canvas acceptance assertion would go red.
    """
    schema = {
        "type": "object",
        "properties": {f"field_{i}": {"type": "string"} for i in range(8)},
        "required": [f"field_{i}" for i in range(8)],
    }
    a = pipeline.compile_json_schema(schema, name="long").automaton
    return a, traced(a)


def marginals(L, V, seed):
    rng = np.random.default_rng(seed)
    return jax.nn.softmax(
        jnp.asarray(rng.standard_normal((L, V)) * 2.0, dtype=jnp.float64), axis=-1)


def emit(emission, p, aut, terminal, key, a):
    """One canvas from whichever emission mode is under test (SPEC §3.9).

    `map` is the **default**; `sample` is `--emission=sample`. Both must satisfy
    the proposition of §3.1b — the argument uses only the *support* of the
    constrained posterior, not maximality, which is exactly why it covers both.
    """
    if emission == "map":
        toks, ok = C.joint_map(p, aut, terminal, a.n_states_bucket,
                               a.tables.n_classes)
    else:
        toks, ok = C.joint_draw(p, aut, terminal, key, a.n_states_bucket,
                                a.tables.n_classes)
    assert bool(ok), f"Z == 0: the {emission} emission has no support"
    return [int(x) for x in toks]


def truncate_at_stop(tokens: list[int], *, canvas_length: int | None = None,
                     done: bool = False) -> tuple[list[int], bool]:
    """`_truncate_canvas_at_stop_tokens`, in Python — **production semantics**.

    This used to truncate the Python *list*, which is not what production does
    and meant the PAD tail was never fed to `advance_states` from this file.
    gemma keeps the first stop token, rewrites everything after it to
    `PAD_TOKEN`, and **keeps the canvas at full length**; that PAD-padded canvas
    is what enters the KV cache, what `predicted_tokens` records, and what `δ*`
    must be recomputed from (SPEC §3.5 trap 2). It therefore only works because
    the unscored `ACC --Σ--> ACC` tail of trap 4 absorbs PAD.

    `done` reproduces gemma's `keep_mask &= ~done`: an already-finished element
    emits an **all-PAD** canvas. `test_the_python_truncation_matches_gemmas`
    pins this against the real function rather than trusting the reimplementation.
    """
    L = len(tokens) if canvas_length is None else canvas_length
    if done:
        return [PAD_TOKEN] * L, False
    for i, t in enumerate(tokens):
        if t in END_TOKENS:
            return tokens[: i + 1] + [PAD_TOKEN] * (L - i - 1), True
    return list(tokens), False


def test_the_python_truncation_matches_gemmas():
    """The helper above is a reimplementation, so it is differential-tested
    against `gemma.diffusion._sampler._truncate_canvas_at_stop_tokens` itself —
    including the `done` row, which emits an all-PAD canvas."""
    L = 8
    canvas = jnp.asarray([
        [9, 9, END_TOKENS[0], 7, 7, 7, 7, 7],     # stops at index 2
        [9, 9, 9, 9, 9, 9, 9, 9],                 # never stops
        [9, END_TOKENS[1], 5, 5, 5, 5, 5, 5],     # stops, but already done
    ], jnp.int32)
    done = jnp.asarray([False, False, True])
    got, has_stop = _diffusion_sampler._truncate_canvas_at_stop_tokens(  # noqa: SLF001
        canvas, end_tokens=tuple(END_TOKENS), canvas_length=L, done=done)
    for row in range(3):
        mine, stopped = truncate_at_stop(
            [int(x) for x in canvas[row]], canvas_length=L, done=bool(done[row]))
        assert mine == [int(x) for x in got[row]], f"row {row}"
        if not bool(done[row]):
            assert stopped == bool(has_stop[row]), f"row {row}"
    assert list(np.asarray(got[2])) == [PAD_TOKEN] * L, (
        "gemma's `keep_mask &= ~done` makes a finished element emit an all-PAD "
        "canvas; the guarantee must survive a block that is entirely PAD"
    )


# ===========================================================================
# Assertion 1 — the per-block viable-prefix property
# ===========================================================================

@pytest.mark.parametrize("emission", ["map", "sample"])
@pytest.mark.parametrize("seed", range(6))
def test_per_block_state_set_is_non_empty_and_within_budget(long_grammar, seed,
                                                            emission):
    """`δ*(A_k, canvas_k) ≠ ∅` **and** `⊆ {s : d(s) ≤ R}`.

    Both halves matter. Non-emptiness alone is the *unbounded-horizon* predicate
    `Live = {s : d(s) < ∞}`, which SPEC §3.1b says explicitly does **not** close
    the budget-truncation failure.

    Run for **both** emissions. `map` is the default of SPEC §3.9's flag table
    and used to be absent from this file entirely; §3.1b's proposition is stated
    over the *support* of the constrained posterior, so it makes exactly the
    same claim for MAP as for the joint draw and must be tested that way.

    **If this ever goes red on an NFA grammar, correct it in this direction and
    no other.** The subset half is asserted over the *whole* reached set, which
    is §3.1b's proposition verbatim and is correct on the compiled minimized
    DFAs this fixture uses. But `b_L` constrains the **drawn path**, not every
    state in `δ*` — so a genuine NFA with two live branches, one of which
    exceeds the budget, can fail this assertion while the guarantee holds
    perfectly. In that case the *assertion* is over-strong and the fix is to
    restrict it to the states reachable along the drawn path. It is **never**
    to relax `d(s) ≤ R`, drop the subset half, or assert non-emptiness alone —
    that is `Live = {d < ∞}`, the unbounded-horizon predicate §3.1b exists to
    reject, and the `b_l_unbounded` mutant in `tests/test_audit_sampler.py`
    exists to catch. CLAUDE.md: if you find yourself weakening this test to make
    something pass, stop and write up why instead.
    """
    a, aut = long_grammar
    L, blocks = 64, 4
    sim = Simulator(a)
    max_new = L * blocks

    active = aut.active
    for k in range(blocks):
        # b_L is evaluated at the state reached AFTER this canvas's L tokens,
        # so the terminal budget is the remainder once they are spent.
        remaining = max_new - k * L
        terminal = remaining - L
        p = marginals(L, a.vocab_size, seed * 100 + k)
        toks = emit(emission, p, aut.with_active(active), jnp.int64(terminal),
                    jax.random.PRNGKey(seed * 100 + k), a)
        canvas, stopped = truncate_at_stop(toks, canvas_length=L)

        reached = sim.run(canvas, states={int(i) for i in np.nonzero(np.asarray(active))[0]})
        assert reached, f"block {k}: delta*(A_k, canvas_k) is EMPTY"

        d = np.asarray(a.d)
        # The whole canvas is committed, PAD tail included: `_sample_step`
        # advances `step += canvas_length` regardless of where the stop landed.
        rem_after = remaining - L
        assert all(d[s] <= max(rem_after, 0) for s in reached), (
            f"block {k}: a reached state cannot finish within the remaining "
            f"budget {rem_after} (d = {[int(d[s]) for s in reached]})"
        )

        active = _adv(C.advance_states)(aut.with_active(active), jnp.asarray(canvas),
                                  a.n_states_bucket, a.tables.n_classes,
                                  a.vocab_size)
        if stopped:
            break


def test_a_non_final_canvas_is_NOT_accepted_on_its_own(long_grammar):
    """**The trap, asserted from the other side.**

    CLAUDE.md: "A non-final block's canvas ends in a live-but-not-accepting
    state and is **not** accepted on its own. Asserting per-canvas acceptance
    makes this red on every multi-block generation, and the 'fix' is to weaken
    it — which must not happen."

    This test exists so that anyone who *does* add a per-canvas acceptance
    assertion finds a test telling them why it is wrong.
    """
    a, aut = long_grammar
    L = 32                       # deliberately far too short to finish
    sim = Simulator(a)
    p = marginals(L, a.vocab_size, 7)
    toks, valid = C.joint_draw(p, aut, jnp.int64(L * 8),
                               jax.random.PRNGKey(7),
                               a.n_states_bucket, a.tables.n_classes)
    assert bool(valid)
    canvas, stopped = truncate_at_stop([int(x) for x in toks], canvas_length=L)
    assert not stopped, "this grammar should not be finishable in 32 tokens"

    reached = sim.run(canvas)
    assert reached, "must still be a VIABLE PREFIX"
    assert not sim.accepts(canvas), (
        "a non-final canvas must NOT be accepted on its own — if this starts "
        "passing, the grammar got shorter, not the guarantee stronger"
    )


# ===========================================================================
# Assertion 2 — constraint satisfaction, on the CONCATENATION
# ===========================================================================

@pytest.mark.parametrize("emission", ["map", "sample"])
@pytest.mark.parametrize("seed", range(4))
def test_concatenation_of_all_blocks_is_accepted(long_grammar, seed, emission):
    """`simulator.accepts(concat(canvas_0 .. canvas_K))` — **this is CS.**

    Membership in `L(M)` is *not* implied by a constrained emission alone
    (SPEC §3.1b); it holds **iff** generation terminates at a block boundary
    with `A_{k+1} ∩ F ≠ ∅`, which is what the loop below checks before
    asserting.

    Run for both emissions, and on the **PAD-padded** canvases production
    actually commits — so the concatenation contains the PAD tails too, and
    acceptance of it is a statement about the unscored `ACC --Σ--> ACC` tail as
    much as about the emission.
    """
    a, aut = long_grammar
    L, blocks = 64, 8
    sim = Simulator(a)
    max_new = L * blocks

    active = aut.active
    whole: list[int] = []
    finished = False
    for k in range(blocks):
        remaining = max_new - k * L
        terminal = remaining - L
        p = marginals(L, a.vocab_size, seed * 50 + k)
        toks = emit(emission, p, aut.with_active(active), jnp.int64(terminal),
                    jax.random.PRNGKey(seed * 50 + k), a)
        canvas, stopped = truncate_at_stop(toks, canvas_length=L)
        whole.extend(canvas)
        active = _adv(C.advance_states)(aut.with_active(active), jnp.asarray(canvas),
                                  a.n_states_bucket, a.tables.n_classes,
                                  a.vocab_size)
        if stopped:
            finished = True
            break

    assert finished, (
        f"generation did not terminate in {blocks} blocks; CS is only claimable "
        "when it ends at a block boundary in an accepting state (SPEC §3.1b)"
    )
    assert sim.accepts(whole), (
        f"the CONCATENATION is not in L(M): {len(whole)} tokens, "
        f"first 16 = {whole[:16]}"
    )


# ===========================================================================
# The three closures of SPEC §3.1b
# ===========================================================================

def test_closure_1_budget_aware_terminal_factor(long_grammar):
    """`b_L = 1[d(s) ≤ R]`, **not** `1[s ∈ F]` and **not** `Live = {d < ∞}`.

    `Live` is unbounded-horizon and does not close budget truncation; the whole
    point of `d` is that it does.
    """
    a, aut = long_grammar
    d = np.asarray(a.d)
    live = (d < INF_DISTANCE).sum()
    within_small = np.asarray(C.budget_terminal_factor(aut.d, jnp.int64(4))).sum()
    assert within_small < live, (
        "a small budget must admit strictly fewer states than `Live` — "
        "otherwise b_L is the unbounded-horizon predicate SPEC rejects"
    )
    at_zero = np.asarray(C.budget_terminal_factor(aut.d, jnp.int64(0))).astype(bool)
    np.testing.assert_array_equal(at_zero, np.asarray(a.is_final),
                                  err_msg="d(s) <= 0 must be exactly s in F")


def test_closure_3_free_region_excludes_end_tokens_and_pad(long_grammar):
    """`FREE` must exclude **every** `end_token` and `PAD`, not just the marker.

    Enforced at the alphabet level by `vocab.RESERVED_TOKENS`: if a grammar
    could emit a stop token internally, the canvas would be truncated
    mid-grammar and `A_{k+1}` would go empty (§3.1b failure mode 3).
    """
    a, _ = long_grammar
    sim = Simulator(a)
    # Reconstruct which tokens each class admits and check no non-ACC edge
    # carries a reserved token.
    acc_states = {int(i) for i in np.nonzero(np.asarray(a.is_final))[0]}
    for e in range(a.n_edges):
        src = int(a.edge_src[e])
        dst = int(a.edge_dst[e])
        if src in acc_states:
            continue                      # the unscored tail may carry anything
        members = sim._members[int(a.edge_class[e])]  # noqa: SLF001
        # An edge INTO an accepting state is the stop-token edge and must carry
        # only end tokens; every other edge must carry none.
        if dst in acc_states:
            assert members <= set(END_TOKENS) | {0}, (
                f"edge {src}->{dst} into ACC carries non-stop tokens"
            )
        else:
            assert not (members & set(END_TOKENS)), (
                f"edge {src}->{dst} inside the grammar can emit a stop token — "
                "the canvas would be truncated mid-grammar (§3.1b mode 3)"
            )
            assert 0 not in members, f"edge {src}->{dst} can emit PAD"


def test_closure_2_is_not_yet_enforced_in_the_sampler(long_grammar):
    """**Honest gap marker.** SPEC §3.1b closure 2 requires conjoining
    `A_{k+1} ∩ F ≠ ∅` into the block-level done flag, fed `emit_canvas` rather
    than the trajectory canvas.

    `ConstrainedDiffusionSampler` currently uses the stock `early_stop_fn`
    unchanged, so that conjunct is **not** enforced end to end. The property it
    would guarantee is checkable here, and
    `test_concatenation_of_all_blocks_is_accepted` asserts the *consequence*
    (termination in an accepting state) explicitly rather than assuming it.

    This test documents the gap so it is not mistaken for coverage. It should be
    replaced by a real end-to-end assertion once the widened `EarlyStopFn`
    exists — not deleted.
    """
    from diffgemma_fa.model.sampler import ConstrainedDiffusionSampler

    assert not hasattr(ConstrainedDiffusionSampler, "should_stop_constrained"), (
        "if a widened EarlyStopFn now exists, replace this marker with a real "
        "end-to-end assertion of A_{k+1} ∩ F != {}"
    )
    a, aut = long_grammar
    # The property itself is well-defined and checkable: after a canvas that
    # does not finish, the state set must be non-empty but not yet accepting.
    L = 32
    p = marginals(L, a.vocab_size, 3)
    toks, valid = C.joint_draw(p, aut, jnp.int64(L * 8), jax.random.PRNGKey(3),
                               a.n_states_bucket, a.tables.n_classes)
    assert bool(valid)
    canvas, stopped = truncate_at_stop([int(x) for x in toks], canvas_length=L)
    nxt = np.asarray(_adv(C.advance_states)(aut, jnp.asarray(canvas),
                                      a.n_states_bucket, a.tables.n_classes,
                                      a.vocab_size))
    assert nxt.any(), "A_{k+1} must be non-empty"
    if not stopped:
        assert not (nxt & np.asarray(a.is_final)).any(), (
            "A_{k+1} ∩ F must still be empty mid-grammar — that is exactly the "
            "condition closure 2 conjoins into the done flag"
        )


# ==========================================================================
# The Z == 0 detector actually fires (SPEC §6.3)
# ==========================================================================
#
# These are the tests whose absence let a real defect survive all 721 others.
# Two reviewers measured it independently: drawing from a **provably empty**
# language returned `valid == True` on 200/200 draws and a canvas that looked
# like a confident sample. The cause is that `jax.random.categorical` and
# `argmax` are **shift-invariant**, so an all-sentinel logit vector is
# indistinguishable from a uniform one — the failure is silent by construction
# and only a root-mass predicate can see it.
#
# CLAUDE.md's taxonomy: (a) empty automaton and (b) no live continuation within
# budget must RAISE; only (c) fp32 underflow is benign, and the constrained
# paths run in log space so (c) cannot arise here.


def _empty_language_automaton(*, n_bucket=8, vocab=8):
    """An automaton with **no accepting state reachable at all**: cause (a).

    Two states, one edge, `is_final` nowhere. `d(s) = INF_DISTANCE` for every
    state, so `b_L` is all-false and the root mass is exactly the sentinel.
    """
    n_edges = 1
    return Automaton(
        edge_src=jnp.zeros(n_edges, jnp.int32),
        edge_dst=jnp.ones(n_edges, jnp.int32),
        edge_class=jnp.zeros(n_edges, jnp.int32),
        edge_valid=jnp.ones(n_edges, bool),
        csr_indices=jnp.arange(vocab, dtype=jnp.int32),
        csr_indptr=jnp.asarray([0, vocab], jnp.int32),
        is_neg=jnp.zeros(1, bool),
        d=jnp.full(n_bucket, INF_DISTANCE, jnp.int32),
        is_final=jnp.zeros(n_bucket, bool),
        active=jnp.zeros(n_bucket, bool).at[0].set(True),
    )


def _uniform(L: int, V: int) -> jnp.ndarray:
    return jnp.full((L, V), 1.0 / V, dtype=jnp.float64)


def test_joint_draw_reports_infeasible_on_an_empty_language():
    """Cause (a). `valid` alone is **not** a detector: it only asks whether the
    drawn state path traverses existing edges, which an all-sentinel draw
    satisfies by accident. `feasible` is the root-mass predicate."""
    V, L = 8, 8
    aut = _empty_language_automaton(vocab=V)
    n_infeasible = 0
    for seed in range(25):
        _toks, ok = C.joint_draw(_uniform(L, V), aut, jnp.int64(64),
                                 jax.random.PRNGKey(seed), 8, 1)
        n_infeasible += int(not bool(ok))
    assert n_infeasible == 25, (
        f"Z == 0 went undetected on {25 - n_infeasible}/25 draws from a "
        "provably empty language"
    )


def test_joint_map_reports_infeasible_on_an_empty_language():
    V, L = 8, 8
    aut = _empty_language_automaton(vocab=V)
    _toks, ok = C.joint_map(_uniform(L, V), aut, jnp.int64(64), 8, 1)
    assert not bool(ok), "constrained MAP claimed support for an empty language"


def test_the_budget_makes_a_nonempty_language_infeasible():
    """Cause (b), which is the one that actually happens in production: the
    automaton is fine, but `b_L = 1[d(s) ≤ R]` admits nothing because no live
    continuation can finish inside `R`. Distinguishing (b) from (a) matters —
    (b) is a grammar/budget condition, not a compiler bug."""
    V, L = 8, 4
    n = 8
    # A chain 0 -> 1 -> ... needing `n - 1` more tokens after the canvas.
    aut = Automaton(
        edge_src=jnp.arange(n - 1, dtype=jnp.int32),
        edge_dst=jnp.arange(1, n, dtype=jnp.int32),
        edge_class=jnp.zeros(n - 1, jnp.int32),
        edge_valid=jnp.ones(n - 1, bool),
        csr_indices=jnp.arange(V, dtype=jnp.int32),
        csr_indptr=jnp.asarray([0, V], jnp.int32),
        is_neg=jnp.zeros(1, bool),
        # After L tokens we are at state L, needing `n - 1 - L` more.
        d=jnp.asarray([n - 1 - i for i in range(n)], jnp.int32),
        is_final=jnp.zeros(n, bool).at[n - 1].set(True),
        active=jnp.zeros(n, bool).at[0].set(True),
    )
    need = n - 1 - L                       # tokens still needed after the canvas
    _t, ok_ample = C.joint_draw(_uniform(L, V), aut, jnp.int64(need),
                                jax.random.PRNGKey(0), n, 1)
    assert bool(ok_ample), "an exactly-sufficient budget was called infeasible"
    _t, ok_tight = C.joint_draw(_uniform(L, V), aut, jnp.int64(need - 1),
                                jax.random.PRNGKey(0), n, 1)
    assert not bool(ok_tight), (
        "R one token short of the shortest completion must be Z == 0 (cause b)"
    )


def test_advance_states_reports_the_state_set_emptying():
    """The stale-carry fallback this replaces made SPEC §3.1b closure 2
    *unsound*, not merely unenforced: substituting the previous state set for an
    empty one lets `A_{k+1} ∩ F ≠ ∅` be TRUE for a string the automaton
    rejects, i.e. the system affirmatively reports acceptance of a rejected
    string. The justification given for the fallback — PAD after a stop token —
    is false for compiled automata, whose unscored `ACC --Σ--> ACC` tail already
    absorbs PAD and every end token."""
    V = 8
    aut = _empty_language_automaton(vocab=V)      # one edge: 0 -> 1 only
    # Two tokens: the first is fine, the second has no edge out of state 1.
    _active, ok = C.advance_states(aut, jnp.asarray([0, 0], jnp.int32), 8, 1, V)
    assert not bool(ok), "advance_states hid an empty state set"
