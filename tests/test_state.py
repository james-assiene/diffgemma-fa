"""The widened `SamplingState`'s budget arithmetic. SPEC §3.1b, §3.5, §5.3.

**Why this file exists.** A mutation audit found that `terminal_budget` was
called by *no test in the suite*: `test_guarantee.py` recomputes
`remaining - canvas_length` inline, so it validates its own arithmetic and not
the source's. Deleting `- canvas_length` from `model/state.py` — the exact
off-by-one-canvas bug the docstring there says "`tests/test_guarantee.py`
catches" — was invisible to all 721 tests.

`R` is the budget fed to the terminal factor `b_L(s) = 1[d(s) ≤ R]`. Too
permissive and the grammar is never forced to close, so generation runs to the
token cap and the canvas is truncated mid-object. Too restrictive and a
completion that genuinely exists is reported as `Z == 0`. Both directions are
tested here.
"""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.model.state import ConstrainedSamplingState  # noqa: E402


def _state(*, step, max_new_tokens, cache_length, used, batch=1):
    """A `ConstrainedSamplingState` with only the budget fields populated.

    Everything the budget arithmetic does not read is a placeholder; the point
    is to call the **source's** `remaining_budget` / `terminal_budget`, which is
    exactly what the suite was not doing.
    """
    return ConstrainedSamplingState(
        step=jnp.int32(step),
        done=jnp.zeros((batch,), bool),
        last_token=jnp.zeros((batch,), jnp.int32),
        last_token_pos=jnp.zeros((batch,), jnp.int32),
        predicted_tokens=jnp.zeros((batch, 8), jnp.int32),
        cache={},
        rng=jax.random.PRNGKey(0),
        init_cache_length=jnp.int32(used),
        full_attention_mask=jnp.zeros((batch, 8), bool),
        automaton=None,
        max_new_tokens=jnp.int32(max_new_tokens),
        cache_length=jnp.int32(cache_length),
        feasible=jnp.ones((batch,), bool),
    )


# --------------------------------------------------------------------------
# The token budget
# --------------------------------------------------------------------------

def test_remaining_budget_counts_the_block_about_to_be_emitted():
    s = _state(step=0, max_new_tokens=256, cache_length=100_000, used=0)
    assert int(s.remaining_budget) == 256


def test_remaining_budget_shrinks_by_a_canvas_per_block():
    s = _state(step=256, max_new_tokens=1024, cache_length=100_000, used=0)
    assert int(s.remaining_budget) == 768


# --------------------------------------------------------------------------
# `R` for `b_L` — the off-by-one-canvas bug
# --------------------------------------------------------------------------

def test_terminal_budget_excludes_the_current_canvas():
    """SPEC §3.5 writes `R <- max_new_tokens - state.step`, but `state.step`
    counts tokens committed *before* the block while `b_L` is evaluated at the
    state reached *after* its `L` tokens. Unadjusted, `R` is too permissive by
    exactly `canvas_length`: it admits states that cannot actually finish, and
    generation then never terminates because nothing forces completion."""
    s = _state(step=0, max_new_tokens=256, cache_length=100_000, used=0)
    assert int(s.terminal_budget(256)) == 0, (
        "with a one-canvas budget, nothing is left after this canvas: b_L must "
        "collapse to 1[d(s) <= 0], i.e. 1[s in F]"
    )
    assert int(s.terminal_budget(64)) == 192


def test_a_state_needing_one_more_token_than_R_is_excluded():
    """The property `b_L` exists for, stated directly on `d`."""
    s = _state(step=192, max_new_tokens=256, cache_length=100_000, used=0)
    R = int(s.terminal_budget(64))
    assert R == 0
    assert not (1 <= R), "a state one token short of finishing must fail b_L"
    assert 0 <= R, "an accepting state (d == 0) must still pass b_L"


def test_the_final_block_needs_no_special_case():
    """CLAUDE.md: `b_L` is `1[d(s) <= R]` with *no* final-block special case,
    because `d(s) <= 0 <=> s in F` handles it. That identity is what makes the
    single formula sufficient, and it is only true because `R` reaches exactly
    0 — never -1, never 1 — on the last block."""
    for L in (32, 64, 128, 256):
        s = _state(step=1024 - L, max_new_tokens=1024, cache_length=100_000,
                   used=0)
        assert int(s.terminal_budget(L)) == 0, L


# --------------------------------------------------------------------------
# SPEC §3.1b's fourth termination path: cache exhaustion
# --------------------------------------------------------------------------

def test_the_cache_bounds_the_budget_when_it_is_tighter():
    s = _state(step=0, max_new_tokens=100_000, cache_length=1024, used=512)
    # gemma's `is_full` is `end_index >= total - 1`, so the usable capacity is
    # `total - 1 - used`, not `total - used`.
    assert int(s.remaining_budget) == 1024 - 1 - 512


def test_the_off_by_one_in_is_full_is_accounted_for():
    """gemma defines `is_full` as `end_index >= total_cache_length - 1` — one
    short of the literal capacity, with its own comment "maybe will lose the
    last token". A bound of `cache_length - used` claims a token the loop will
    never emit, and `b_L` then admits a state that cannot finish."""
    # Positioned so that this canvas exactly fills the cache to `total - 1`.
    L = 64
    s = _state(step=0, max_new_tokens=100_000, cache_length=1024,
               used=1024 - 1 - L)
    assert int(s.terminal_budget(L)) == 0, (
        "after this canvas `end_index == total - 1`, `is_full` is true and the "
        "block loop exits: zero tokens remain, so b_L must force completion now"
    )


def test_the_token_budget_wins_when_it_is_tighter():
    s = _state(step=0, max_new_tokens=64, cache_length=100_000, used=0)
    assert int(s.remaining_budget) == 64


@pytest.mark.parametrize("used", [0, 1, 100, 4095])
def test_remaining_budget_is_never_larger_than_either_bound(used):
    s = _state(step=32, max_new_tokens=256, cache_length=4096, used=used)
    r = int(s.remaining_budget)
    assert r <= 256 - 32
    assert r <= 4096 - 1 - (used + 32)
