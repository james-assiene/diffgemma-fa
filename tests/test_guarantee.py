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
from diffgemma_fa.compile.vocab import END_TOKENS  # noqa: E402
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


def truncate_at_stop(tokens: list[int]) -> tuple[list[int], bool]:
    """`_truncate_canvas_at_stop_tokens`, in Python.

    Keeps the first stop token and PADs after it — and the PAD-truncated canvas
    is what enters the KV cache and what `δ*` must be recomputed from
    (SPEC §3.5 trap 2).
    """
    for i, t in enumerate(tokens):
        if t in END_TOKENS:
            return tokens[: i + 1], True
    return tokens, False


# ===========================================================================
# Assertion 1 — the per-block viable-prefix property
# ===========================================================================

@pytest.mark.parametrize("seed", range(6))
def test_per_block_state_set_is_non_empty_and_within_budget(long_grammar, seed):
    """`δ*(A_k, canvas_k) ≠ ∅` **and** `⊆ {s : d(s) ≤ R}`.

    Both halves matter. Non-emptiness alone is the *unbounded-horizon* predicate
    `Live = {s : d(s) < ∞}`, which SPEC §3.1b says explicitly does **not** close
    the budget-truncation failure.
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
        toks, valid = C.joint_draw(p, aut.with_active(active), jnp.int64(terminal),
                                   jax.random.PRNGKey(seed * 100 + k),
                                   a.n_states_bucket, a.tables.n_classes)
        assert bool(valid), f"block {k}: the boundary draw degenerated"
        canvas, stopped = truncate_at_stop([int(x) for x in toks])

        reached = sim.run(canvas, states={int(i) for i in np.nonzero(np.asarray(active))[0]})
        assert reached, f"block {k}: delta*(A_k, canvas_k) is EMPTY"

        d = np.asarray(a.d)
        rem_after = remaining - len(canvas)
        assert all(d[s] <= max(rem_after, 0) for s in reached), (
            f"block {k}: a reached state cannot finish within the remaining "
            f"budget {rem_after} (d = {[int(d[s]) for s in reached]})"
        )

        active = C.advance_states(aut.with_active(active), jnp.asarray(canvas),
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
    canvas, stopped = truncate_at_stop([int(x) for x in toks])
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

@pytest.mark.parametrize("seed", range(4))
def test_concatenation_of_all_blocks_is_accepted(long_grammar, seed):
    """`simulator.accepts(concat(canvas_0 .. canvas_K))` — **this is CS.**

    Membership in `L(M)` is *not* implied by a constrained emission alone
    (SPEC §3.1b); it holds **iff** generation terminates at a block boundary
    with `A_{k+1} ∩ F ≠ ∅`, which is what the loop below checks before
    asserting.
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
        toks, valid = C.joint_draw(p, aut.with_active(active), jnp.int64(terminal),
                                   jax.random.PRNGKey(seed * 50 + k),
                                   a.n_states_bucket, a.tables.n_classes)
        assert bool(valid)
        canvas, stopped = truncate_at_stop([int(x) for x in toks])
        whole.extend(canvas)
        active = C.advance_states(aut.with_active(active), jnp.asarray(canvas),
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
    canvas, stopped = truncate_at_stop([int(x) for x in toks])
    nxt = np.asarray(C.advance_states(aut, jnp.asarray(canvas),
                                      a.n_states_bucket, a.tables.n_classes,
                                      a.vocab_size))
    assert nxt.any(), "A_{k+1} must be non-empty"
    if not stopped:
        assert not (nxt & np.asarray(a.is_final)).any(), (
            "A_{k+1} ∩ F must still be empty mid-grammar — that is exactly the "
            "condition closure 2 conjoins into the done flag"
        )
