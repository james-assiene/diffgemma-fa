"""The constrained emission path, on real compiled BFCL grammars.

This is where the Phase 1 compiler and the Phase 3 kernels meet: a real
automaton, real class tables, and an emission that an independent simulator has
to accept.

Pins the fp32-underflow finding, which is the one thing in Phase 4 that changes
a documented design assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.compile import bfcl_data, pipeline  # noqa: E402
from diffgemma_fa.compile.validate import Simulator  # noqa: E402
from diffgemma_fa.compile.vocab import END_TOKENS  # noqa: E402
from diffgemma_fa.infer import scans  # noqa: E402
from diffgemma_fa.model import constrained as C  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402


# -- shims for the pair-returning APIs -------------------------------------
# `joint_map`/`joint_draw`/`advance_states` each return a feasibility flag
# beside their value (SPEC §6.3's Z == 0 detector). These wrappers keep the
# existing assertions readable AND assert the flag, so a silent Z == 0 fails
# the test rather than sliding past it.

def _map_tokens(fn):
    def go(*a, **kw):
        toks, feasible = fn(*a, **kw)
        assert bool(feasible), "Z == 0: constrained MAP has no support"
        return toks
    return go


def _adv(fn):
    def go(*a, **kw):
        active, ok = fn(*a, **kw)
        assert bool(ok), "state set emptied while advancing across the canvas"
        return active
    return go


@pytest.fixture(scope="module")
def grammar():
    """A real BFCL-Live grammar, compiled end to end."""
    rec = next(iter(bfcl_data.iter_split("BFCL_v4_live_simple.json")))
    fn = rec.functions[0]
    a = pipeline.compile_json_schema(fn["parameters"], name=fn["name"],
                                     from_bfcl=True).automaton
    aut = Automaton(
        edge_src=jnp.asarray(a.edge_src), edge_dst=jnp.asarray(a.edge_dst),
        edge_class=jnp.asarray(a.edge_class),
        edge_valid=jnp.ones(a.n_edges, bool),
        csr_indices=jnp.asarray(a.tables.sum_indices),
        csr_indptr=jnp.asarray(a.tables.sum_indptr),
        is_neg=jnp.asarray(a.tables.sum_is_neg),
        d=jnp.asarray(a.d), is_final=jnp.asarray(a.is_final),
        active=jnp.asarray(a.start_vector),
    )
    return a, aut


def marginals(L, V, seed=0, scale=2.0):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((L, V)) * scale
    return jax.nn.softmax(jnp.asarray(logits, dtype=jnp.float64), axis=-1)


# ---------------------------------------------------------------------------
# The emission must be in the language
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_joint_draw_is_accepted(grammar, seed):
    a, aut = grammar
    L = 64
    p = marginals(L, a.vocab_size, seed)
    toks, valid = C.joint_draw(p, aut, jnp.int64(L), jax.random.PRNGKey(seed),
                               a.n_states_bucket, a.tables.n_classes)
    assert bool(valid), "the boundary draw degenerated (see require_x64)"
    toks = [int(x) for x in toks]
    assert len(toks) == L
    assert Simulator(a).accepts(toks), f"constrained draw rejected: {toks[:12]}"


@pytest.mark.parametrize("seed", range(4))
def test_joint_map_is_accepted(grammar, seed):
    a, aut = grammar
    L = 64
    p = marginals(L, a.vocab_size, seed)
    toks = [int(x) for x in _map_tokens(C.joint_map)(p, aut, jnp.int64(L),
                                        a.n_states_bucket, a.tables.n_classes)]
    assert Simulator(a).accepts(toks)


def test_emission_contains_a_stop_token(grammar):
    """The grammar can only reach an accepting state through one of
    `end_tokens` (SPEC §3.5), so a valid emission must contain one."""
    a, aut = grammar
    L = 64
    p = marginals(L, a.vocab_size, 11)
    toks = [int(x) for x in _map_tokens(C.joint_map)(p, aut, jnp.int64(L),
                                        a.n_states_bucket, a.tables.n_classes)]
    assert any(t in END_TOKENS for t in toks), "no stop token in the emission"


def test_stop_token_is_not_pinned_to_the_last_position(grammar):
    """SPEC §3.5 trap 4's diagnostic: with a **scored** post-stop tail a joint
    decode pays `(L-1-j)·log p(PAD)` to terminate at `j`, so the MAP puts the
    stop token at the last position or never. The unscored `ACC --Σ--> ACC`
    tail is what prevents that — clustering at the end means it is still open.
    """
    a, aut = grammar
    L = 64
    positions = []
    for seed in range(8):
        p = marginals(L, a.vocab_size, seed)
        toks = [int(x) for x in _map_tokens(C.joint_map)(p, aut, jnp.int64(L),
                                            a.n_states_bucket,
                                            a.tables.n_classes)]
        first = next((i for i, t in enumerate(toks) if t in END_TOKENS), None)
        assert first is not None
        positions.append(first)
    assert max(positions) < L - 1, (
        f"stop token clustering at the end: {positions} — the post-stop tail "
        "is being scored (SPEC §3.5 trap 4)"
    )


# ---------------------------------------------------------------------------
# The budget-aware terminal factor
# ---------------------------------------------------------------------------

def test_budget_factor_is_d_le_R_not_membership_of_F(grammar):
    a, aut = grammar
    b0 = C.budget_terminal_factor(aut.d, jnp.int64(0))
    np.testing.assert_array_equal(np.asarray(b0).astype(bool),
                                  np.asarray(a.is_final))
    b_big = C.budget_terminal_factor(aut.d, jnp.int64(10_000))
    assert np.asarray(b_big).sum() > np.asarray(b0).sum(), (
        "a larger budget must admit strictly more states"
    )


def test_budget_shrinks_the_admissible_set_monotonically(grammar):
    a, aut = grammar
    sizes = [float(np.asarray(C.budget_terminal_factor(aut.d, jnp.int64(r))).sum())
             for r in (0, 1, 2, 4, 8, 16)]
    assert sizes == sorted(sizes), f"not monotone in R: {sizes}"


# ---------------------------------------------------------------------------
# Block threading
# ---------------------------------------------------------------------------

def test_advance_states_matches_the_reference_simulator(grammar):
    """`A_{k+1} = δ*(A_k, canvas_k)`, computed as traced ops on fixed-shape
    arrays, must agree with the plain-Python simulator."""
    a, aut = grammar
    L = 64
    p = marginals(L, a.vocab_size, 3)
    toks, _ = C.joint_draw(p, aut, jnp.int64(L), jax.random.PRNGKey(3),
                           a.n_states_bucket, a.tables.n_classes)
    got = np.asarray(_adv(C.advance_states)(aut, toks, a.n_states_bucket,
                                      a.tables.n_classes, a.vocab_size))
    want = Simulator(a).run([int(x) for x in toks])
    assert set(np.nonzero(got)[0].tolist()) == want


def test_advance_states_never_empties_on_a_valid_canvas(grammar):
    """An empty state set at a block boundary is the failure SPEC §3.5 trap 3
    calls the most likely bug in the whole port."""
    a, aut = grammar
    L = 64
    for seed in range(6):
        p = marginals(L, a.vocab_size, seed)
        toks, _ = C.joint_draw(p, aut, jnp.int64(L), jax.random.PRNGKey(seed),
                               a.n_states_bucket, a.tables.n_classes)
        nxt = np.asarray(_adv(C.advance_states)(aut, toks, a.n_states_bucket,
                                          a.tables.n_classes, a.vocab_size))
        assert nxt.any(), f"A_k+1 went empty on seed {seed}"


# ---------------------------------------------------------------------------
# The fp32 underflow finding
# ---------------------------------------------------------------------------

def test_root_product_underflows_in_fp32_on_a_real_grammar(grammar):
    """The measurement behind `require_x64`.

    Per-node max-normalization fixes the overall scale but **not** the dynamic
    range inside one matrix. `ACC --Σ--> ACC` has emission mass exactly 1.0, so
    the root's max is pinned at 1.0 while genuine grammar paths sit around
    1e-49 — and the smallest positive entry measured here is ~1e-288, against
    an fp32 floor of ~1e-45.
    """
    a, aut = grammar
    L = 64
    p = marginals(L, a.vocab_size, 0)
    _, _, M = C._matrices(p, aut, a.n_states_bucket, a.tables.n_classes)

    tr64 = scans.up_sweep(jnp.asarray(M, dtype=jnp.float64))
    root64 = np.asarray(tr64.root)
    start = np.asarray(aut.active).astype(np.float64)
    b = np.asarray(C.budget_terminal_factor(aut.d, jnp.int64(L)))
    joint64 = start[:, None] * root64 * b[None, :]

    assert joint64.sum() > 0, "float64 must produce a usable joint"
    assert joint64[joint64 > 0].min() < 1e-45, (
        "this grammar is supposed to exercise the underflow regime"
    )

    tr32 = scans.up_sweep(jnp.asarray(M, dtype=jnp.float32))
    joint32 = (start.astype(np.float32)[:, None]
               * np.asarray(tr32.root) * b.astype(np.float32)[None, :])
    assert joint32.sum() == 0.0, (
        "fp32 is expected to underflow to exactly zero here; if it stops "
        "doing so, require_x64's justification needs revisiting"
    )
    # And the max really is pinned at 1.0 by the unscored tail.
    assert np.asarray(tr64.root).max() == pytest.approx(1.0)


def test_require_x64_passes_when_x64_is_on():
    C.require_x64()


def test_map_path_survives_fp32(grammar):
    """MAP is in log space (SPEC §2.7), so it is immune to the above — which is
    an argument for `--emission=map` as the default."""
    a, aut = grammar
    L = 64
    p32 = jnp.asarray(marginals(L, a.vocab_size, 5), dtype=jnp.float32)
    toks = [int(x) for x in _map_tokens(C.joint_map)(p32, aut, jnp.int32(L),
                                        a.n_states_bucket, a.tables.n_classes)]
    assert Simulator(a).accepts(toks), "MAP must work in fp32"
