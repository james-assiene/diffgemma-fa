"""Audit of the sampler / guarantee slice. Tests written from SPEC, not from code.

Scope: `model/sampler.py`, `model/state.py`, `model/constrained.py`,
`infer/marginals.py`, and the coverage of `tests/test_guarantee.py`.

Every assertion below is derived from a numbered claim in SPEC.md or CLAUDE.md
and is annotated with it. Where a property could only be checked by reading the
implementation back to itself, the test was not written — the finding is in the
audit report instead.

Three structural choices worth stating, because they are what make these tests
able to fail:

1. **The denoising loop is exercised for real**, with only the transformer
   stubbed out (`_StubbedSampler`). `sample_next_canvas_constrained` — the
   emission composition, the accept rule, the trajectory/emission decoupling,
   the feasibility threading — runs unmodified. SPEC §3.1's residual hole (the
   emitted canvas *is* the sample, and non-accepted positions are uniform random
   tokens over the full vocabulary) is a property of that loop and cannot be
   checked at kernel level.
2. **Oracles are independent.** `_PySim` and `_distance_to_final` here are
   re-derivations from the definitions in SPEC §3.5 / §3.1b, not calls into
   `compile/`. A test that used the compiler's own `d` to check the sampler's
   use of `d` would be circular.
3. **The default emission is `map`** (SPEC §3.9's flag table). `test_guarantee.py`
   used to exercise the multi-block guarantee only through `joint_draw`, i.e.
   only through `--emission=sample`; it is parametrized over both now, and the
   MAP path is additionally tested here end to end.

Every mutant used to validate this file is registered in `_MUTANTS` below and
re-runnable with `DGFA_MUT=<name>`.

**Counting note:** one test here — `test_report_string_body_saturation_in_
existing_artifacts` — asserts nothing by design and only prints. Quote this
file's result as "N-1 assertions plus a diagnostic", never as N tests passing.
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402
from gemma.diffusion import _sampler as DS  # noqa: E402

from diffgemma_fa.compile import pipeline  # noqa: E402
from diffgemma_fa.compile.automaton import INF_DISTANCE  # noqa: E402
from diffgemma_fa.compile.validate import Simulator  # noqa: E402
from diffgemma_fa.compile import vocab as VOCAB  # noqa: E402
from diffgemma_fa.infer import marginals as MG  # noqa: E402
from diffgemma_fa.infer import scans  # noqa: E402
from diffgemma_fa.infer import tree as TREE  # noqa: E402
from diffgemma_fa.model import constrained as C  # noqa: E402
from diffgemma_fa.model import sampler as S  # noqa: E402
from diffgemma_fa.model import state as STATE  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402


# ===========================================================================
# Reproducible mutation registry.  `DGFA_MUT=<name> pytest tests/…`
# ===========================================================================
"""Every mutant used to validate this file, in a form anyone can re-run.

An unreproducible verification is worth much less than a reproducible one:
"these tests catch that bug" is a claim about the tests, and a claim nobody else
can check is not evidence. Each entry below breaks the implementation in one
specific way **in process** — no repository file is edited, so this is safe to
run in a tree several agents are working in.

    DGFA_MUT=b_l_slack JAX_PLATFORMS=cpu pytest tests/test_audit_sampler.py -q

Each mutant names the test it was written to kill. Collateral kills are
expected and fine; a mutant with **no** kill, or one whose named victim survives,
is a hole in this file.
"""

_MUTANTS: dict[str, str] = {
    # name                     -> the test it must kill
    "terminal_budget_no_canvas": "test_terminal_budget_is_what_the_sampler_actually_passes",
    "b_l_is_final":              "test_b_L_is_not_membership_in_F",
    "b_l_unbounded":             "test_map_emission_concatenation_is_accepted",
    "b_l_slack":                 "test_a_high_mass_self_loop_does_not_capture_the_decode",
    "a_start_point_mass":        "test_map_follows_the_active_set_not_a_point_mass",
    "emission_unconstrained":    "test_the_emitted_canvas_carries_no_unconstrained_random_token",
    "advance_stale_fallback":    "test_advance_states_matches_an_independent_simulator_on_an_nfa",
    "scatter_complement_blind":  "test_the_constrained_marginal_support_is_exactly_the_reachable_tokens",
    "mar_unconstrained":         "test_mar_confidence_changes_which_positions_are_accepted",
    "feasible_always_true":      "test_a_block_that_cannot_finish_is_reported_infeasible",
}


def _rewrap(orig, fn):
    """Keep `clear_cache`/`_cache_size` alive so the tracing tests still run."""
    for attr in ("clear_cache", "_cache_size"):
        if hasattr(orig, attr):
            setattr(fn, attr, getattr(orig, attr))
    return fn


def _apply_mutation(name: str) -> None:
    if name not in _MUTANTS:
        raise SystemExit(f"unknown DGFA_MUT={name!r}; known: {sorted(_MUTANTS)}")
    omap, odraw = C.joint_map, C.joint_draw

    def wrap_budget(f):
        """Rewrite the `remaining` argument the emission kernels receive."""
        def jmap(p_lv, aut, remaining, n_states, n_classes):
            return omap(p_lv, aut, f(remaining, p_lv.shape[0]), n_states,
                        n_classes)

        def jdraw(p_lv, aut, remaining, key, n_states, n_classes):
            return odraw(p_lv, aut, f(remaining, p_lv.shape[0]), key, n_states,
                         n_classes)
        C.joint_map = _rewrap(omap, jmap)
        C.joint_draw = _rewrap(odraw, jdraw)

    if name == "terminal_budget_no_canvas":
        # SPEC §3.5 trap 1 [V-P4]: drop the `- canvas_length` correction.
        STATE.ConstrainedSamplingState.terminal_budget = (
            lambda self, canvas_length: self.remaining_budget)

    elif name == "b_l_is_final":
        # b_L := 1[s in F].  `d >= 0` everywhere, so `d <= 0` is exactly `s in F`.
        wrap_budget(lambda r, L: jnp.int64(0))

    elif name == "b_l_unbounded":
        # b_L := Live = {s : d(s) < INF}, the unbounded-horizon predicate.
        wrap_budget(lambda r, L: jnp.int64(1 << 23))

    elif name == "b_l_slack":
        # R = max_new_tokens - step, i.e. permissive by exactly one canvas.
        wrap_budget(lambda r, L: r + L)

    elif name == "a_start_point_mass":
        # SPEC §5.7: hardcode `a_start = 1[s = s0]` instead of using A_k.
        def point(aut):
            return aut.with_active(
                jnp.zeros_like(aut.active).at[..., 0].set(True))

        def jmap(p_lv, aut, remaining, n_states, n_classes):
            return omap(p_lv, point(aut), remaining, n_states, n_classes)

        def jdraw(p_lv, aut, remaining, key, n_states, n_classes):
            return odraw(p_lv, point(aut), remaining, key, n_states, n_classes)
        C.joint_map = _rewrap(omap, jmap)
        C.joint_draw = _rewrap(odraw, jdraw)

    elif name == "emission_unconstrained":
        # SPEC §3.1's residual hole, reopened: the emission stops being a
        # constrained object and becomes a stock per-position draw.
        def jmap(p_lv, aut, remaining, n_states, n_classes):
            return jnp.argmax(p_lv, axis=-1).astype(jnp.int32), jnp.bool_(True)

        def jdraw(p_lv, aut, remaining, key, n_states, n_classes):
            return (jax.random.categorical(key, jnp.log(p_lv)).astype(jnp.int32),
                    jnp.bool_(True))
        C.joint_map = _rewrap(omap, jmap)
        C.joint_draw = _rewrap(odraw, jdraw)

    elif name == "advance_stale_fallback":
        # The `where(nxt.any(), nxt, active)` fallback whose removal SPEC §3.1b
        # records: it makes closure 2 unsound rather than merely unenforced.
        oadv = C.advance_states

        def adv(automaton, tokens, n_states, n_classes, vocab_size):
            active, _ok = oadv(automaton, tokens, n_states, n_classes,
                               vocab_size)
            stale = automaton.active
            return (jnp.where(active.any(), active, stale), jnp.bool_(True))
        C.advance_states = _rewrap(oadv, adv)

    elif name == "scatter_complement_blind":
        # A plain sparse scatter: drop the negated classes' `neg_total` term.
        orig = MG.scatter_edge_mass_to_tokens

        def broken(edge_mass, class_id, indices, indptr, is_neg, n_classes,
                   vocab_size):
            return orig(edge_mass, class_id, indices, indptr,
                        jnp.zeros_like(is_neg), n_classes, vocab_size)
        MG.scatter_edge_mass_to_tokens = broken

    elif name == "mar_unconstrained":
        # --confidence=mar silently falls back to the unconstrained entropy.
        def flat(p_vl, u, class_id, indices, indptr, is_neg, n_classes,
                 vocab_size):
            return jnp.full((u.shape[0],), jnp.log(jnp.float64(vocab_size)))
        MG.constrained_entropy_streamed = flat

    elif name == "feasible_always_true":
        # SPEC §6.3's Z == 0 detector discarded at the source.
        ostates, omapst = TREE.sample_states_log, TREE.map_states_and_tokens

        def st(*a, **kw):
            s, _f = ostates(*a, **kw)
            return s, jnp.bool_(True)

        def mp(*a, **kw):
            out = omapst(*a, **kw)
            return out[:-1] + (jnp.bool_(True),)
        # `constrained.py` looks these up on the module at call time, and the
        # jit cache is empty at import, so patching here does reach the traced
        # body.
        TREE.sample_states_log, TREE.map_states_and_tokens = st, mp


if os.environ.get("DGFA_MUT"):
    _apply_mutation(os.environ["DGFA_MUT"])
    print(f"\n*** DGFA_MUT={os.environ['DGFA_MUT']} ACTIVE — the implementation "
          f"is deliberately broken; expected victim: "
          f"{_MUTANTS[os.environ['DGFA_MUT']]} ***")


# ===========================================================================
# Independent oracles — deliberately NOT the compiler's
# ===========================================================================

def _distance_to_final(n_states, edges, finals):
    """`d(s)` = min tokens from `s` to an accepting state (SPEC §3.1b).

    Reverse BFS, every edge one token. Re-derived here so a test of the
    sampler's *use* of `d` never validates the compiler's `d` against itself.
    """
    rev = [[] for _ in range(n_states)]
    for src, dst in edges:
        rev[dst].append(src)
    d = [INF_DISTANCE] * n_states
    frontier = list(finals)
    for f in frontier:
        d[f] = 0
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for q in frontier:
            for p in rev[q]:
                if d[p] == INF_DISTANCE:
                    d[p] = depth
                    nxt.append(p)
        frontier = nxt
    return np.asarray(d, dtype=np.int32)


class _PySim:
    """Plain-Python set-tracking simulator for a hand-built automaton.

    Tracks a **set** of states, so it is correct for NFAs: CLAUDE.md says
    several bugs (eq (8)'s edge multiplicity, start vectors) are invisible on
    DFAs.
    """

    def __init__(self, spec):
        self.spec = spec

    def step(self, states, tok):
        out = set()
        for s in states:
            for (src, dst, cls) in self.spec.edges:
                if src == s and tok in self.spec.members[cls]:
                    out.add(dst)
        return out

    def run(self, tokens, states=None):
        cur = set(self.spec.start if states is None else states)
        for t in tokens:
            cur = self.step(cur, t)
            if not cur:
                return cur
        return cur

    def accepts(self, tokens):
        return bool(self.run(tokens) & set(self.spec.finals))


@dataclasses.dataclass(frozen=True)
class _Spec:
    n_states: int
    bucket: int
    vocab: int
    #: `(src, dst, class_id)`
    edges: tuple
    #: class id -> the **true** member set
    members: tuple
    #: class id -> whether the CSR stores the complement
    is_neg: tuple
    finals: frozenset
    start: frozenset

    def traced(self, active=None, batched=False) -> Automaton:
        indices, indptr = [], [0]
        for c, neg in zip(self.members, self.is_neg):
            stored = sorted(set(range(self.vocab)) - set(c)) if neg else sorted(c)
            indices.extend(stored)
            indptr.append(len(indices))
        d_real = _distance_to_final(self.n_states,
                                    [(s, t) for (s, t, _) in self.edges],
                                    self.finals)
        d = np.full(self.bucket, INF_DISTANCE, np.int32)
        d[: self.n_states] = d_real
        is_final = np.zeros(self.bucket, bool)
        is_final[sorted(self.finals)] = True
        act = np.zeros(self.bucket, bool)
        for s in (self.start if active is None else active):
            act[s] = True
        return Automaton(
            edge_src=jnp.asarray([e[0] for e in self.edges], jnp.int32),
            edge_dst=jnp.asarray([e[1] for e in self.edges], jnp.int32),
            edge_class=jnp.asarray([e[2] for e in self.edges], jnp.int32),
            edge_valid=jnp.ones(len(self.edges), bool),
            csr_indices=jnp.asarray(indices, jnp.int32),
            csr_indptr=jnp.asarray(indptr, jnp.int32),
            is_neg=jnp.asarray(self.is_neg, bool),
            d=jnp.asarray(d), is_final=jnp.asarray(is_final),
            active=jnp.asarray(act[None, :] if batched else act),
        )

    @property
    def n_classes(self):
        return len(self.members)


def _enumerate_language(spec, L):
    """Every length-`L` string the automaton accepts. Brute force, `V**L`."""
    sim = _PySim(spec)
    out = []
    def rec(prefix, states):
        if not states:
            return
        if len(prefix) == L:
            if states & set(spec.finals):
                out.append(tuple(prefix))
            return
        for v in range(spec.vocab):
            rec(prefix + [v], sim.step(states, v))
    rec([], set(spec.start))
    return out


# ===========================================================================
# Fixtures — small hand-built automata
# ===========================================================================
# Token convention throughout: 0 = PAD, 1 = EOS, >= 2 = grammar body. That is
# `compile/vocab.py`'s convention and it keeps the stop-token traps honest.


@pytest.fixture(scope="module")
def chain():
    """`body body EOS`, then the unscored `ACC --Σ--> ACC` tail (§3.5 trap 4)."""
    V = 8
    members = ({2, 3}, {2, 3}, {1}, set(range(V)))
    edges = ((0, 1, 0), (1, 2, 1), (2, 3, 2), (3, 3, 3))
    return _Spec(n_states=4, bucket=4, vocab=V, edges=edges, members=members,
                 is_neg=(False, False, False, False),
                 finals=frozenset({3}), start=frozenset({0}))


@pytest.fixture(scope="module")
def non_product():
    """`{(2,3), (3,2)}` then EOS — the classic non-product language.

    SPEC §2.8's whole argument: the per-position projections are `{2,3}` at both
    positions, so a factorized sampler admits `(2,2)` and `(3,3)`, which are not
    in the language. This is the fixture that makes the `mask` baseline mean
    something.
    """
    V = 6
    members = ({2}, {3}, {1}, set(range(V)))
    edges = ((0, 1, 0), (0, 2, 1),      # 2 -> A, 3 -> B
             (1, 3, 1), (2, 3, 0),      # A-3-> C, B-2-> C
             (3, 4, 2),                 # C-EOS-> ACC
             (4, 4, 3))                 # ACC -Σ-> ACC
    return _Spec(n_states=5, bucket=8, vocab=V, edges=edges, members=members,
                 is_neg=(False, False, False, False),
                 finals=frozenset({4}), start=frozenset({0}))


@pytest.fixture(scope="module")
def two_branches():
    """Two disjoint branches out of *different* start states.

    Used for SPEC §5.7: `a_start = 1[s = s₀]` holds only for a DFA on block 0.
    From block 1 on, and always for an NFA, the start is a **set**, so a
    hardcoded point mass is a bug that first shows up on block 2.
    """
    V = 8
    members = ({2}, {3}, {1}, set(range(V)))
    edges = ((0, 2, 0),   # branch L: state 0 -2-> 2
             (1, 2, 1),   # branch R: state 1 -3-> 2
             (2, 3, 2),   # -EOS-> ACC
             (3, 3, 3))
    return _Spec(n_states=4, bucket=4, vocab=V, edges=edges, members=members,
                 is_neg=(False, False, False, False),
                 finals=frozenset({3}), start=frozenset({0, 1}))


@pytest.fixture(scope="module")
def nfa():
    """A genuine NFA: parallel, overlapping edges out of one state.

    CLAUDE.md: "Test NFAs, not just DFAs." Class 0 and class 1 overlap on token
    3, so token 3 from state 0 reaches **both** 1 and 2 — and there are two
    parallel edges carrying it, which is exactly the multiplicity eq (8) has to
    weight by. One branch also uses a **negated** class, so the complement-aware
    path is exercised too.
    """
    V = 6
    members = ({2, 3}, {3, 4}, set(range(V)) - {0, 1}, {1}, set(range(V)))
    edges = ((0, 1, 0), (0, 2, 1),
             (1, 3, 2), (2, 3, 2),
             (3, 4, 3), (4, 4, 4))
    return _Spec(n_states=5, bucket=8, vocab=V, edges=edges, members=members,
                 is_neg=(False, False, True, False, False),
                 finals=frozenset({4}), start=frozenset({0}))


# -- a real compiled grammar, for the multi-block guarantee -----------------

def _traced_compiled(a) -> Automaton:
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
    """A JSON grammar whose shortest member is far longer than the block size.

    Same shape as `test_guarantee.py`'s fixture, deliberately: the point of the
    tests below is to run the **default** emission (`map`) down the path that
    file only runs with `sample`.
    """
    schema = {
        "type": "object",
        "properties": {f"field_{i}": {"type": "string"} for i in range(4)},
        "required": [f"field_{i}" for i in range(4)],
    }
    a = pipeline.compile_json_schema(schema, name="audit_long").automaton
    return a, _traced_compiled(a)


def _marginals(L, V, seed, scale=2.0):
    rng = np.random.default_rng(seed)
    return jax.nn.softmax(
        jnp.asarray(rng.standard_normal((L, V)) * scale, dtype=jnp.float64),
        axis=-1)


# ===========================================================================
# 1. The two-part guarantee under the DEFAULT emission (`--emission=map`)
#    SPEC §3.1b, §3.5, §3.9's flag table.
# ===========================================================================

def _run_blocks(a, aut, *, L, blocks, seed, emission):
    """One multi-block generation, following SPEC §3.5's protocol verbatim.

    Returns `(records, whole_tokens, finished)` where each record is
    `(A_k, canvas_k, remaining_after)`.
    """
    sim = Simulator(a)
    max_new = L * blocks
    active = aut.active
    records, whole, finished = [], [], False
    for k in range(blocks):
        remaining = max_new - k * L                 # R, including this canvas
        terminal = remaining - L                    # `state.terminal_budget`
        p = _marginals(L, a.vocab_size, seed * 97 + k)
        if emission == "map":
            toks, ok = C.joint_map(p, aut.with_active(active),
                                   jnp.int64(terminal), a.n_states_bucket,
                                   a.tables.n_classes)
        else:
            toks, ok = C.joint_draw(p, aut.with_active(active),
                                    jnp.int64(terminal),
                                    jax.random.PRNGKey(seed * 97 + k),
                                    a.n_states_bucket, a.tables.n_classes)
        assert bool(ok), f"block {k}: Z == 0 on the {emission} emission"
        toks = [int(x) for x in toks]

        # Production truncation semantics: keep the first stop token, PAD the
        # rest, keep the full canvas length (SPEC §3.5 trap 2).
        stopped = any(t in VOCAB.END_TOKENS for t in toks)
        if stopped:
            j = min(i for i, t in enumerate(toks) if t in VOCAB.END_TOKENS)
            canvas = toks[: j + 1] + [VOCAB.PAD_TOKEN] * (L - j - 1)
        else:
            canvas = toks

        records.append((active, canvas, remaining - L))
        whole.extend(canvas)
        active, adv_ok = C.advance_states(
            aut.with_active(active), jnp.asarray(canvas, jnp.int32),
            a.n_states_bucket, a.tables.n_classes, a.vocab_size)
        assert bool(adv_ok), f"block {k}: the state set emptied"
        if stopped:
            finished = True
            break
    return sim, records, whole, finished


@pytest.mark.parametrize("seed", range(3))
def test_map_emission_per_block_is_a_viable_prefix_within_budget(long_grammar, seed):
    """Part (a) of the guarantee, for `--emission=map`. SPEC §3.1b.

    `δ*(A_k, canvas_k) ≠ ∅` **and** `⊆ {s : d(s) ≤ R}`. `test_guarantee.py`
    asserts this only for `joint_draw`; `map` is the default emission, so the
    default configuration was untested against the proposition it relies on.

    Both halves matter: non-emptiness alone is the unbounded-horizon predicate
    `Live = {s : d(s) < ∞}`, which SPEC §3.1b says explicitly does **not** close
    budget truncation.
    """
    a, aut = long_grammar
    sim, records, _whole, _fin = _run_blocks(a, aut, L=32, blocks=8, seed=seed,
                                             emission="map")
    d = np.asarray(a.d)
    for k, (active, canvas, rem_after) in enumerate(records):
        starts = {int(i) for i in np.nonzero(np.asarray(active))[0]}
        reached = sim.run(canvas, states=starts)
        assert reached, f"block {k}: delta*(A_k, canvas_k) is EMPTY"
        assert all(d[s] <= max(rem_after, 0) for s in reached), (
            f"block {k}: a reached state cannot finish within the remaining "
            f"budget {rem_after} (d = {sorted(int(d[s]) for s in reached)})"
        )


@pytest.mark.parametrize("seed", range(3))
def test_map_emission_concatenation_is_accepted(long_grammar, seed):
    """Part (b) — constraint satisfaction — for `--emission=map`. SPEC §3.1b.

    Membership in `L(M)` is not implied by a constrained emission alone; it
    holds **iff** generation terminates at a block boundary with
    `A_{k+1} ∩ F ≠ ∅`. The budget-aware `b_L` is what forces that to happen
    before the budget runs out, so "did not terminate" is itself a failure.
    """
    a, aut = long_grammar
    sim, _rec, whole, finished = _run_blocks(a, aut, L=32, blocks=8, seed=seed,
                                             emission="map")
    assert finished, (
        "generation did not terminate inside the budget; with b_L = 1[d(s) <= R] "
        "and R running down to 0, completion is forced (SPEC §3.5 trap 1)"
    )
    assert sim.accepts(whole), (
        f"the CONCATENATION is not in L(M): {len(whole)} tokens, "
        f"first 16 = {whole[:16]}"
    )


def test_the_multi_block_guarantee_is_not_vacuous(long_grammar):
    """**Vacuity check on the fixture itself.**

    Part (a) of the guarantee is only stressed if generation actually crosses a
    block boundary — if everything fits in one canvas, `A_k` is never anything
    but `A_0` and `test_guarantee.py`'s loop degenerates to a single-block test
    that a per-canvas acceptance assertion would also pass.
    """
    a, aut = long_grammar
    _sim, records, _whole, finished = _run_blocks(a, aut, L=32, blocks=8, seed=1,
                                                  emission="map")
    assert len(records) >= 2, (
        f"the generation used {len(records)} block(s); the multi-block path is "
        "not being exercised at all"
    )
    assert finished
    first, second = records[0][0], records[1][0]
    assert not np.array_equal(np.asarray(first), np.asarray(second)), (
        "A_1 == A_0: the automaton is not being advanced across the boundary, "
        "so `a_start` is effectively still a point mass (SPEC §5.7)"
    )


# ===========================================================================
# 2. `b_L = 1[d(s) <= R]` is budget-aware.  SPEC §3.1b, §3.5 trap 1.
# ===========================================================================

def test_b_L_is_not_membership_in_F(chain):
    """`b_L` must **not** be `1[s ∈ F]`, and there is no final-block special case.

    A canvas shorter than the shortest member of the language must still be
    emittable as a viable prefix whenever the remaining budget can absorb the
    rest. `1[s ∈ F]` would force the grammar to complete inside one canvas and
    report `Z == 0` here.
    """
    spec = chain
    aut = spec.traced()
    L = 2                                   # the language needs 3 tokens
    p = _marginals(L, spec.vocab, 11)
    toks, ok = C.joint_map(p, aut, jnp.int64(8), spec.bucket, spec.n_classes)
    assert bool(ok), (
        "a two-token viable prefix with 8 tokens of budget left was reported "
        "infeasible: b_L is behaving like 1[s in F] (SPEC §3.5 trap 1)"
    )
    reached = _PySim(spec).run([int(x) for x in toks])
    assert reached and not (reached & set(spec.finals)), (
        "the emission should be live-but-not-accepting here"
    )


def test_map_is_infeasible_one_token_short_of_the_shortest_completion(chain):
    """The other direction: `R` one token short of `d(s)` must be `Z == 0`.

    CLAUDE.md's cause (b) — a grammar/budget condition, not a compiler bug — and
    it must raise rather than silently emit. Only `joint_draw` was covered.
    """
    spec = chain
    aut = spec.traced()
    L = 2
    p = _marginals(L, spec.vocab, 12)
    # After 2 tokens the only live state is 2, with d(2) = 1.
    _t, ok_exact = C.joint_map(p, aut, jnp.int64(1), spec.bucket, spec.n_classes)
    assert bool(ok_exact), "an exactly-sufficient budget was called infeasible"
    _t, ok_short = C.joint_map(p, aut, jnp.int64(0), spec.bucket, spec.n_classes)
    assert not bool(ok_short), (
        "R one token short of the shortest completion must be Z == 0; MAP's own "
        "score is the detector"
    )


def test_a_tight_budget_forces_the_shorter_of_two_completions():
    """The property `b_L` exists for, on a grammar with a real choice.

    Two branches: a short one the model dislikes and a long one it likes. With
    a generous `R` the MAP takes the likely long branch; with `R` too small for
    it, the MAP must switch to the short branch rather than emit something it
    cannot finish. A `b_L` that ignored `R` would take the long branch in both
    cases.
    """
    V = 8
    # 0 -2-> 1 -1(EOS)-> ACC          (short: needs 1 more token after 2)
    # 0 -3-> 4 -3-> 5 -3-> 6 -1-> ACC (long : needs 3 more)
    members = ({2}, {3}, {1}, set(range(V)))
    edges = ((0, 1, 0), (1, 3, 2),
             (0, 4, 1), (4, 5, 1), (5, 6, 1), (6, 3, 2),
             (3, 3, 3))
    spec = _Spec(n_states=7, bucket=8, vocab=V, edges=edges, members=members,
                 is_neg=(False, False, False, False),
                 finals=frozenset({3}), start=frozenset({0}))
    aut = spec.traced()
    L = 1
    p = np.full((L, V), 1e-6)
    p[:, 3] = 0.9                     # the model strongly prefers the long branch
    p[:, 2] = 0.05
    p = jnp.asarray(p / p.sum(axis=-1, keepdims=True))

    generous, _ok = C.joint_map(p, aut, jnp.int64(4), spec.bucket, spec.n_classes)
    assert int(generous[0]) == 3, "with budget to spare, MAP should follow `p`"

    tight, ok = C.joint_map(p, aut, jnp.int64(1), spec.bucket, spec.n_classes)
    assert bool(ok)
    assert int(tight[0]) == 2, (
        "with only 1 token left after the canvas, the long branch (3 more) "
        "cannot finish; b_L = 1[d(s) <= R] must exclude it"
    )


def test_a_high_mass_self_loop_does_not_capture_the_decode():
    """**The attractor `b_L` actually exists to escape.**

    Measured on the real compiled BFCL grammars: the JSON string-body state
    carries a self-loop over **99.4% of Σ**. Nothing in the grammar prevents
    that and nothing should — an unbounded JSON string genuinely has that shape.
    Its emission mass is therefore ~1.0, so on a joint decode the per-token
    factor for *staying in the string* dominates the factor for closing the
    quote at essentially every position. The only thing that forces the decoder
    out is `b_L = 1[d(s) ≤ R]` running down.

    Both arms below run the **same** automaton and the **same** `p`; only `R`
    differs, and the wrong `R` is the specific off-by-one-canvas of SPEC §3.5
    trap 1 [V-P4] — `max_new_tokens − step` instead of
    `max_new_tokens − step − canvas_length`. Slack by exactly one canvas turns a
    grammar that closes into one that emits `L` tokens of string interior and
    never closes the quote. It is not a crash; it is a silently saturated
    canvas, which is why it needs a test rather than a `Z == 0`.
    """
    V, L = 256, 8
    QUOTE, body = 2, set(range(3, V))          # |body| / |Σ| = 99.2%
    members = (body, {QUOTE}, {1}, set(range(V)))
    edges = ((0, 0, 0),        # string body self-loop, mass ~1.0
             (0, 1, 1),        # close the quote
             (1, 2, 2),        # EOS -> ACC
             (2, 2, 3))        # unscored tail
    spec = _Spec(n_states=3, bucket=4, vocab=V, edges=edges, members=members,
                 is_neg=(False, False, False, False), finals=frozenset({2}),
                 start=frozenset({0}))
    aut = spec.traced()
    p = jnp.asarray(np.full((L, V), 1.0 / V))   # flat: the model has no opinion

    correct, ok = C.joint_map(p, aut, jnp.int64(0), spec.bucket, spec.n_classes)
    assert bool(ok)
    correct = [int(x) for x in correct]
    assert QUOTE in correct, (
        f"the decode never closed the string: {correct}. With R = 0 the "
        "terminal factor must admit only ACC, which forces the quote and the "
        "stop token inside the canvas."
    )
    assert _PySim(spec).accepts(correct)

    # The off-by-one-canvas arm: R slack by exactly one canvas length.
    slack, ok_s = C.joint_map(p, aut, jnp.int64(L), spec.bucket, spec.n_classes)
    slack = [int(x) for x in slack]
    assert bool(ok_s)
    assert QUOTE not in slack, (
        "control failed: even with a slack budget the decode left the string "
        "self-loop, so the test above is not demonstrating that b_L is what "
        "forced it out"
    )


def test_d_is_the_reverse_bfs_of_the_augmented_automaton(long_grammar):
    """`d(s)` must be computed **after** stop-token augmentation and after the
    refusal-branch union, "or it is a completely different function"
    (SPEC §3.1b).

    Recomputed here from the compiled edge list and `is_final` with an
    independent reverse BFS. A `d` measured on the un-augmented machine is
    uniformly *larger* (the `ACC` sink and its stop edges are missing), so `b_L`
    excludes states that can in fact finish and the decode is reported
    infeasible; a `d` measured before the refusal union is *smaller* on the
    branch states and `b_L` then admits states that cannot finish — which is the
    silent saturation the string-body self-loop turns into 256 tokens of string
    interior.
    """
    a, _aut = long_grammar
    edges = list(zip((int(x) for x in a.edge_src), (int(x) for x in a.edge_dst)))
    finals = [int(i) for i in np.nonzero(np.asarray(a.is_final))[0]]
    mine = _distance_to_final(a.n_states_bucket, edges, finals)
    theirs = np.asarray(a.d)
    # Padding states beyond `n_states` are unreachable in both.
    np.testing.assert_array_equal(
        mine[: a.n_states], theirs[: a.n_states],
        err_msg="the compiled `d` is not the reverse BFS of the automaton the "
                "sampler is handed; b_L is then computing a different predicate")


def test_terminal_budget_is_what_the_sampler_actually_passes():
    """`_sample_step` must feed `b_L` the budget left **after** the canvas.

    SPEC §3.5 trap 1 as corrected in [V-P4]: `state.step` counts tokens
    committed *before* the block, `b_L` is evaluated at the state reached
    *after* it. Off by exactly `canvas_length` in the permissive direction, and
    generation then never terminates.

    Asserted on the value, by constructing the state the sampler would have.
    """
    from diffgemma_fa.model.state import ConstrainedSamplingState
    L = 64
    st = ConstrainedSamplingState(
        step=jnp.int32(128), done=jnp.zeros((1,), bool),
        last_token=jnp.zeros((1,), jnp.int32),
        last_token_pos=jnp.zeros((1,), jnp.int32),
        predicted_tokens=jnp.zeros((1, 8), jnp.int32), cache={},
        rng=jax.random.PRNGKey(0), init_cache_length=jnp.int32(0),
        full_attention_mask=jnp.zeros((1, 8), bool), automaton=None,
        max_new_tokens=jnp.int32(256), cache_length=jnp.int32(1 << 20),
        feasible=jnp.ones((1,), bool))
    assert int(st.terminal_budget(L)) == 256 - 128 - L


# ===========================================================================
# 3. The post-stop tail must be UNSCORED.  SPEC §3.5 trap 4.
# ===========================================================================

def _stop_position(tokens):
    idx = [i for i, t in enumerate(tokens) if t == 1]
    return idx[0] if idx else None


@pytest.mark.parametrize("tail,expected", [("sigma", 2), ("pad", 7)])
def test_the_unscored_tail_is_what_lets_the_stop_token_land_early(tail, expected):
    """SPEC §3.5 trap 4, with its own diagnostic turned into an assertion.

    > If the tail is constrained to PAD, terminating at position `j` costs
    > `log p(EOS) + (255−j)·log p(PAD)` … **The MAP places the stop token at
    > position 255 or never.**
    > Diagnostic: log the stop-token position per block. Clustering at 255 means
    > this is still open.

    Two automata identical but for the tail edge. `p` is built so that EOS is
    relatively most attractive at position 2. With `ACC --Σ--> ACC` the tail is
    free and the MAP stops there; with `ACC --PAD--> ACC` the `log p(PAD)`
    penalty per remaining position dominates and pushes the stop to the last
    position, which is the failure this trap describes.
    """
    V, L = 8, 8
    tail_members = set(range(V)) if tail == "sigma" else {0}
    members = ({2, 3, 4, 5, 6, 7}, {1}, tail_members)
    edges = ((0, 0, 0), (0, 1, 1), (1, 1, 2))
    spec = _Spec(n_states=2, bucket=2, vocab=V, edges=edges, members=members,
                 is_neg=(False, False, False), finals=frozenset({1}),
                 start=frozenset({0}))
    aut = spec.traced()

    p = np.full((L, V), 1e-8)
    p[:, 2:] = 0.2                       # body tokens are the likely ones
    p[:, 1] = 1e-4                       # EOS is unlikely almost everywhere ...
    p[2, 1] = 0.15                       # ... but much less so at position 2
    p = jnp.asarray(p / p.sum(axis=-1, keepdims=True))

    toks, ok = C.joint_map(p, aut, jnp.int64(0), spec.bucket, spec.n_classes)
    assert bool(ok)
    got = _stop_position([int(x) for x in toks])
    assert got == expected, (
        f"tail={tail}: stop token at {got}, expected {expected}. With the "
        f"unscored Sigma tail the stop must land where `p` wants it; with a "
        f"PAD-only tail the joint decode pays (L-1-j)*log p(PAD) and is driven "
        f"to the last position — that is the bias SPEC §3.5 trap 4 removes."
    )


@pytest.mark.parametrize("tail", ["sigma", "pad"])
def test_the_unscored_tail_removes_the_pad_bias_from_the_posterior(tail):
    """§3.5 trap 4's second claim, stated on the posterior rather than on a
    class weight.

    > Since `_truncate_canvas_at_stop_tokens` overwrites everything after the
    > first stop token anyway, letting the tail accept any token is
    > *semantically free*, and **it removes the corresponding `∏ p(PAD)` bias
    > from `Z` and `q_i` too**.

    "Semantically free" has an exact meaning: at a post-stop position the
    constrained marginal must be **`q_i = p_i`**, i.e. the tail reweights
    nothing. A PAD-only tail instead collapses `q_i` onto a point mass at PAD,
    which is precisely the bias.

    The grammar here is one token long (`EOS -> ACC`), so positions 1..L-1 are
    unambiguously in the tail and the two cases are exact rather than
    approximate. An earlier version of this test asserted
    `W_c[Σ-class] == 1.0`, which is a property of `class_weights` and would hold
    for a tail that was never reached at all.
    """
    V, L = 8, 4
    tail_members = set(range(V)) if tail == "sigma" else {0}
    members = ({1}, tail_members)
    edges = ((0, 1, 0), (1, 1, 1))
    spec = _Spec(n_states=2, bucket=2, vocab=V, edges=edges, members=members,
                 is_neg=(False, False), finals=frozenset({1}),
                 start=frozenset({0}))
    aut = spec.traced()
    p = _marginals(L, V, 13)
    p_vl, a_v, b_v = _forward_backward(spec, aut, p, jnp.int64(0))
    q = np.asarray(MG.constrained_marginals(
        p_vl, a_v, b_v, aut.edge_src, aut.edge_dst, aut.edge_class,
        aut.csr_indices, aut.csr_indptr, aut.is_neg, spec.n_classes))

    # Position 0 is the grammar: EOS with certainty, in both variants.
    assert np.isclose(q[0, 1], 1.0), "the one grammar token must be forced"

    for i in range(1, L):
        if tail == "sigma":
            np.testing.assert_allclose(
                q[i], np.asarray(p[i]), rtol=1e-9, atol=1e-12,
                err_msg=f"position {i}: an unscored tail must leave q_i == p_i; "
                        "any deviation is the tail reweighting the posterior")
        else:
            assert np.isclose(q[i, 0], 1.0), (
                f"position {i}: a PAD-only tail must pin q_i on PAD — this is "
                "the `prod p(PAD)` bias trap 4 exists to remove"
            )


# ===========================================================================
# 4. Start vectors are VECTORS; NFAs.  SPEC §5.7, CLAUDE.md.
# ===========================================================================

def test_map_follows_the_active_set_not_a_point_mass(two_branches):
    """SPEC §5.7: `a_start = 1[s = s₀]` holds only for a DFA on block 0.

    Selecting a single non-zero start state must change the emission. A sampler
    that hardcoded `s₀ = 0`, or that read the compiled `start_vector` instead of
    the carried `A_k`, passes every block-0 DFA test and fails from block 1 on.
    """
    spec = two_branches
    L = 1
    p = jnp.asarray(np.full((L, spec.vocab), 1.0 / spec.vocab))
    left, ok_l = C.joint_map(p, spec.traced(active={0}), jnp.int64(2),
                             spec.bucket, spec.n_classes)
    right, ok_r = C.joint_map(p, spec.traced(active={1}), jnp.int64(2),
                              spec.bucket, spec.n_classes)
    assert bool(ok_l) and bool(ok_r)
    assert int(left[0]) == 2, "from state 0 the only legal token is 2"
    assert int(right[0]) == 3, (
        "from state 1 the only legal token is 3 — the emission ignored A_k"
    )


def test_a_two_state_start_reaches_both_branches(two_branches):
    """`a_start` is a general **vector**, so both branches must have support.

    With equal marginals the two branches are equiprobable, so the split is a
    fair coin. Asserted with a Bonferroni-corrected two-sided normal test
    (2 statistical tests in this file, alpha 0.01 -> 0.005 each) rather than a
    bare "both appeared", which would also pass on a 199/1 split.
    """
    spec = two_branches
    L, n = 1, 400
    p = jnp.asarray(np.full((L, spec.vocab), 1.0 / spec.vocab))
    aut = spec.traced(active={0, 1})
    firsts = []
    for seed in range(n):
        toks, ok = C.joint_draw(p, aut, jnp.int64(2), jax.random.PRNGKey(seed),
                                spec.bucket, spec.n_classes)
        assert bool(ok)
        firsts.append(int(toks[0]))
    assert set(firsts) <= {2, 3}, f"illegal first tokens: {set(firsts) - {2, 3}}"
    k = firsts.count(2)
    z = abs(k - n / 2) / math.sqrt(n * 0.25)
    assert z < 2.807, (                       # two-sided, alpha = 0.005
        f"branch split {k}/{n} is not consistent with the uniform posterior "
        f"over a two-state start vector (z = {z:.2f})"
    )


def test_advance_states_matches_an_independent_simulator_on_an_nfa(nfa):
    """`A_{k+1} = δ*(A_k, canvas_k)` on an NFA with parallel, overlapping edges
    and a **negated** class. SPEC §3.5, CLAUDE.md "Test NFAs, not just DFAs".
    """
    spec = nfa
    sim = _PySim(spec)
    aut = spec.traced()
    rng = np.random.default_rng(0)
    for trial in range(30):
        toks = [int(t) for t in rng.integers(0, spec.vocab, size=3)]
        expected = sim.run(toks)
        got, ok = C.advance_states(aut, jnp.asarray(toks, jnp.int32),
                                   spec.bucket, spec.n_classes, spec.vocab)
        got_set = {int(i) for i in np.nonzero(np.asarray(got))[0]}
        assert bool(ok) == bool(expected), (
            f"{toks}: ok={bool(ok)} but the simulator reached {expected}"
        )
        if expected:
            assert got_set == expected, f"{toks}: {got_set} != {expected}"


def test_the_draw_on_an_nfa_only_emits_accepted_strings(nfa):
    """Support correctness on an NFA, checked against brute-force enumeration.

    `L = 4` is exactly the shortest accepted length here, so with `R = 0` every
    draw must be a member of the language.
    """
    spec = nfa
    L = 4
    language = set(_enumerate_language(spec, L))
    assert language, "fixture broken: no accepted string of this length"
    aut = spec.traced()
    p = _marginals(L, spec.vocab, 21)
    seen = set()
    for seed in range(60):
        toks, ok = C.joint_draw(p, aut, jnp.int64(0), jax.random.PRNGKey(seed),
                                spec.bucket, spec.n_classes)
        assert bool(ok)
        w = tuple(int(x) for x in toks)
        assert w in language, f"emitted {w}, which the automaton rejects"
        seen.add(w)
    assert len(seen) > 1, "the draw is degenerate — it never leaves one string"


# ===========================================================================
# 5. Block threading.  SPEC §3.2, §3.5 trap 2.
# ===========================================================================

def test_advance_states_composes_across_a_block_boundary(nfa):
    """`δ*(δ*(A, c₀), c₁) == δ*(A, c₀·c₁)`.

    SPEC §3.2: there is no monotone commit set, "which is also why automaton
    state can be recomputed statelessly each step". That claim is only sound if
    advancing block by block equals advancing over the concatenation — the
    property the per-block protocol of §3.5 rests on.
    """
    spec = nfa
    aut = spec.traced()
    rng = np.random.default_rng(5)
    for _ in range(20):
        c0 = [int(t) for t in rng.integers(0, spec.vocab, size=2)]
        c1 = [int(t) for t in rng.integers(0, spec.vocab, size=2)]
        mid, ok0 = C.advance_states(aut, jnp.asarray(c0, jnp.int32),
                                    spec.bucket, spec.n_classes, spec.vocab)
        if not bool(ok0):
            continue
        end, _ = C.advance_states(aut.with_active(mid),
                                  jnp.asarray(c1, jnp.int32),
                                  spec.bucket, spec.n_classes, spec.vocab)
        whole, _ = C.advance_states(aut, jnp.asarray(c0 + c1, jnp.int32),
                                    spec.bucket, spec.n_classes, spec.vocab)
        np.testing.assert_array_equal(np.asarray(end), np.asarray(whole))


def test_the_pad_tail_of_a_truncated_canvas_is_absorbed(long_grammar):
    """SPEC §3.5 trap 2, with gemma's **real** truncation.

    `_truncate_canvas_at_stop_tokens` keeps the first stop token and rewrites
    everything after it to `PAD_TOKEN`, keeping the canvas at full length — and
    that PAD-padded canvas is what enters the KV cache and what `δ*` is
    recomputed from. `advance_states` therefore consumes the PAD tail as
    ordinary tokens.

    `test_guarantee.py` truncates the Python **list** instead, so the PAD tail
    is never fed to `advance_states` there. If the unscored `ACC --Σ--> ACC`
    tail did not absorb PAD, production would empty the state set at every
    terminating block while that test stayed green.
    """
    a, aut = long_grammar
    L = 32
    active = aut.active
    # Walk forward until a block emits a stop token.
    for k in range(8):
        p = _marginals(L, a.vocab_size, 300 + k)
        toks, ok = C.joint_map(p, aut.with_active(active),
                               jnp.int64(L * 8 - (k + 1) * L),
                               a.n_states_bucket, a.tables.n_classes)
        assert bool(ok)
        canvas = jnp.asarray([int(x) for x in toks], jnp.int32)[None, :]
        truncated, has_stop = DS._truncate_canvas_at_stop_tokens(  # noqa: SLF001
            canvas, end_tokens=VOCAB.END_TOKENS, canvas_length=L,
            done=jnp.zeros((1,), bool))
        row = [int(x) for x in truncated[0]]
        active, adv_ok = C.advance_states(
            aut.with_active(active), jnp.asarray(row, jnp.int32),
            a.n_states_bucket, a.tables.n_classes, a.vocab_size)
        assert bool(adv_ok), (
            f"block {k}: the state set emptied on the PAD-padded canvas; the "
            "unscored tail is not absorbing PAD (SPEC §3.5 traps 2/4)"
        )
        if bool(has_stop[0]):
            assert (np.asarray(active) & np.asarray(a.is_final)).any(), (
                "after a stop token the state set must contain an accepting "
                "state (SPEC §3.1b closure 2)"
            )
            return
    pytest.fail("no block emitted a stop token inside the budget")


# ===========================================================================
# 6. End tokens.  SPEC §3.5 trap 3.
# ===========================================================================

def test_the_samplers_end_tokens_and_the_compilers_agree():
    """"Handle **all** of `end_tokens`… the single most likely source of an
    'empty state set' bug." (SPEC §3.5 trap 3.)

    The sampler truncates on `(EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE)` read
    off the tokenizer; the compiler builds stop edges from
    `vocab.END_TOKENS`. If the two sets ever diverge, a canvas gets truncated on
    a token for which the automaton has no stop edge, `δ*` walks off the machine
    and `A_{k+1} = ∅`.
    """
    tok = VOCAB.gemma_tokenizer()
    st = tok.special_tokens
    sampler_side = {int(st.EOS), int(st.END_OF_TURN),
                    int(st.BEGIN_OF_TOOL_RESPONSE)}
    assert set(VOCAB.END_TOKENS) == sampler_side, (
        f"compiler stop edges {sorted(VOCAB.END_TOKENS)} != sampler truncation "
        f"set {sorted(sampler_side)}"
    )


def test_every_end_token_completes_the_grammar(long_grammar):
    """Each of the three end tokens must lead from a grammar-final state into
    `F`, not just EOS. Missing one is invisible until the model happens to emit
    it. SPEC §3.5 trap 3.
    """
    a, aut = long_grammar
    sim = Simulator(a)
    L = 32
    active = aut.active
    prefix: list[int] = []
    for k in range(8):
        p = _marginals(L, a.vocab_size, 400 + k)
        toks, ok = C.joint_map(p, aut.with_active(active),
                               jnp.int64(L * 8 - (k + 1) * L),
                               a.n_states_bucket, a.tables.n_classes)
        assert bool(ok)
        row = [int(x) for x in toks]
        j = next((i for i, t in enumerate(row) if t in VOCAB.END_TOKENS), None)
        if j is not None:
            prefix.extend(row[:j])           # everything before the stop token
            break
        prefix.extend(row)
        active, _ = C.advance_states(aut.with_active(active),
                                     jnp.asarray(row, jnp.int32),
                                     a.n_states_bucket, a.tables.n_classes,
                                     a.vocab_size)
    else:
        pytest.fail("never reached a grammar-final state")

    for t in VOCAB.END_TOKENS:
        assert sim.accepts(prefix + [t]), (
            f"end token {t} does not complete the grammar from a grammar-final "
            "state; a canvas truncated on it empties the state set"
        )


# ===========================================================================
# 7. The `mask` baseline must really be unsound.  SPEC §2.8, §7.2.
# ===========================================================================

def _support_projection(spec, aut, p, remaining):
    """`r_i(v)` exactly as `sampler.py`'s `mask` branch computes it."""
    p_vl, _W_e, M = C._matrices(p, aut, spec.bucket, spec.n_classes)  # noqa: SLF001
    tr = scans.up_sweep(M)
    a_v, b_v, _, _ = scans.prefix_suffix(
        tr, aut.active.astype(p.dtype),
        C.budget_terminal_factor(aut.d, remaining, dtype=p.dtype))
    u = a_v[:-1][:, aut.edge_src] * b_v[1:][:, aut.edge_dst]
    return MG.scatter_edge_mass_to_tokens(u, aut.edge_class, aut.csr_indices,
                                          aut.csr_indptr, aut.is_neg,
                                          spec.n_classes, spec.vocab)


def test_per_position_masking_leaves_the_language(non_product):
    """SPEC §2.8 / §7.2 baseline 2: the `mask` arm's CS column must be able to
    drop below 1.

    The support projection `π_i(C)` is `{2,3}` at both grammar positions, so an
    independent per-position sampler admits `(2,2)` and `(3,3)`, neither of
    which is in `C`. If this test ever passed trivially — i.e. if the masked
    sampler only ever produced members — the `mask` baseline would be measuring
    nothing, and the headline "joint decoding is necessary" claim would have no
    evidence behind it.
    """
    spec = non_product
    L = 4
    aut = spec.traced()
    language = set(_enumerate_language(spec, L))
    assert len(language) == 2 * spec.vocab, "fixture: expected exactly |Σ| * 2"
    p = jnp.asarray(np.full((L, spec.vocab), 1.0 / spec.vocab))
    r = _support_projection(spec, aut, p, jnp.int64(0))
    assert np.asarray(r[0] > 0).sum() == 2 and np.asarray(r[1] > 0).sum() == 2

    logits = jnp.where(r > 0, jnp.zeros_like(r), MG.MASK_SENTINEL)
    violations = 0
    for seed in range(60):
        toks = jax.random.categorical(jax.random.PRNGKey(seed), logits)
        w = tuple(int(x) for x in toks)
        assert all(v in {2, 3} for v in w[:2]), "the mask itself is wrong"
        if w not in language:
            violations += 1
    assert violations > 0, (
        "per-position masking never left the language on a NON-PRODUCT grammar; "
        "the `mask` baseline is not measuring SPEC §2.8's failure mode"
    )


def test_the_joint_draw_never_leaves_the_same_language(non_product):
    """The complement of the test above, on the identical automaton. Without
    this pair the `mask` result could be blamed on the fixture rather than on
    factorization."""
    spec = non_product
    L = 4
    aut = spec.traced()
    language = set(_enumerate_language(spec, L))
    p = jnp.asarray(np.full((L, spec.vocab), 1.0 / spec.vocab))
    for seed in range(60):
        toks, ok = C.joint_draw(p, aut, jnp.int64(0), jax.random.PRNGKey(seed),
                                spec.bucket, spec.n_classes)
        assert bool(ok)
        assert tuple(int(x) for x in toks) in language


# ===========================================================================
# 8. MAP is temperature-invariant.  SPEC §2.7, §3.9.
# ===========================================================================

@pytest.mark.parametrize("T", [0.5, 1.0, 2.0, 4.0])
def test_map_is_temperature_invariant(nfa, T):
    """`argmax_w Σ_i log softmax(z_i/T)[w_i]` is independent of `T`.

    The per-position log-normalizers differ with `T` but every candidate string
    pays all `L` of them, so they cancel. This is what lets `--emission=map` be
    compared across `--temp-schedule={default,near-greedy}` without the
    temperature confounding the result (SPEC §3.9). If it failed, the MAP arm
    would silently be measuring the annealing schedule.
    """
    spec = nfa
    L = 4
    rng = np.random.default_rng(31)
    z = jnp.asarray(rng.standard_normal((L, spec.vocab)) * 3.0)
    aut = spec.traced()
    base, _ = C.joint_map(jax.nn.softmax(z, axis=-1), aut, jnp.int64(0),
                          spec.bucket, spec.n_classes)
    got, ok = C.joint_map(jax.nn.softmax(z / T, axis=-1), aut, jnp.int64(0),
                          spec.bucket, spec.n_classes)
    assert bool(ok)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(base))


# ===========================================================================
# 9. Constrained marginals and `--confidence=mar`.  SPEC §2.4, §3.4.
# ===========================================================================

def _forward_backward(spec, aut, p, remaining):
    p_vl, _W_e, M = C._matrices(p, aut, spec.bucket, spec.n_classes)  # noqa: SLF001
    tr = scans.up_sweep(M)
    a_v, b_v, _, _ = scans.prefix_suffix(
        tr, aut.active.astype(p.dtype),
        C.budget_terminal_factor(aut.d, remaining, dtype=p.dtype))
    return p_vl, a_v, b_v


def test_the_streamed_entropy_equals_the_dense_constrained_marginal(nfa):
    """`constrained_entropy_streamed` must equal `entropy_from_q(q)`.

    `infer/marginals.py` holds two routes to the same number: the dense one
    (`constrained_marginals` -> `entropy_from_q`), which is the **definition**
    in SPEC §2.4, and the streamed CSR one, which is what production actually
    calls under `--confidence=mar`. The dense route is reachable from no
    production path, so nothing but a differential test keeps the fast one
    honest — and if they diverge, the accept rule is driven by a quantity that
    is not `H(q_i)`.
    """
    spec = nfa
    L = 4
    aut = spec.traced()
    p = _marginals(L, spec.vocab, 41)
    p_vl, a_v, b_v = _forward_backward(spec, aut, p, jnp.int64(0))
    q = MG.constrained_marginals(p_vl, a_v, b_v, aut.edge_src, aut.edge_dst,
                                 aut.edge_class, aut.csr_indices,
                                 aut.csr_indptr, aut.is_neg, spec.n_classes)
    dense = np.asarray(MG.entropy_from_q(q))
    u = a_v[:-1][:, aut.edge_src] * b_v[1:][:, aut.edge_dst]
    streamed = np.asarray(MG.constrained_entropy_streamed(
        p_vl, u, aut.edge_class, aut.csr_indices, aut.csr_indptr, aut.is_neg,
        spec.n_classes, spec.vocab))
    np.testing.assert_allclose(streamed, dense, rtol=1e-10, atol=1e-12)


def test_the_constrained_marginal_support_is_exactly_the_reachable_tokens(nfa):
    """`q_i(v) > 0` **iff** some accepted length-`L` string has `v` at `i`.

    Checked against brute-force enumeration, not against another kernel. This
    is what makes `q` usable as a support mask (SPEC §3.7) and as the `mask`
    baseline's projection: an over-broad support silently readmits forbidden
    tokens, an under-broad one deletes valid strings from the posterior.
    """
    spec = nfa
    L = 4
    aut = spec.traced()
    language = _enumerate_language(spec, L)
    assert language
    expected = [set(w[i] for w in language) for i in range(L)]

    p = _marginals(L, spec.vocab, 42)
    p_vl, a_v, b_v = _forward_backward(spec, aut, p, jnp.int64(0))
    q = np.asarray(MG.constrained_marginals(
        p_vl, a_v, b_v, aut.edge_src, aut.edge_dst, aut.edge_class,
        aut.csr_indices, aut.csr_indptr, aut.is_neg, spec.n_classes))
    for i in range(L):
        got = {int(v) for v in np.nonzero(q[i] > 1e-12)[0]}
        assert got == expected[i], f"position {i}: {got} != {expected[i]}"


def test_the_constrained_entropy_is_bounded_by_its_support(nfa):
    """SPEC §3.4: "Constrained marginals `q_i` have restricted support and are
    far sharper… The defaults are calibrated for unconstrained entropies and
    will silently produce garbage."

    The sharpening is not a hope, it is `H(q_i) ≤ log |supp(q_i)|`. Asserting
    the bound (rather than `H(q) < H(p)`, which is not a theorem) pins the
    magnitude the recalibration of §3.4 has to cover.
    """
    spec = nfa
    L = 4
    aut = spec.traced()
    language = _enumerate_language(spec, L)
    p = _marginals(L, spec.vocab, 43)
    p_vl, a_v, b_v = _forward_backward(spec, aut, p, jnp.int64(0))
    u = a_v[:-1][:, aut.edge_src] * b_v[1:][:, aut.edge_dst]
    h = np.asarray(MG.constrained_entropy_streamed(
        p_vl, u, aut.edge_class, aut.csr_indices, aut.csr_indptr, aut.is_neg,
        spec.n_classes, spec.vocab))
    assert np.isfinite(h).all(), "H(q) must never be NaN — `q` has exact zeros"
    for i in range(L):
        k = len({w[i] for w in language})
        assert h[i] <= math.log(k) + 1e-9, (
            f"position {i}: H(q) = {h[i]:.4f} exceeds log|supp| = "
            f"{math.log(k):.4f}"
        )


# ===========================================================================
# 10. The denoising loop itself, with the transformer stubbed out.
#     SPEC §3.1 (the emission IS the sample), §5.4 (the widened carry).
# ===========================================================================

#: Logits the stubbed `sample_step` returns. Module state, not an attribute:
#: `self` is a `static_argname`, so anything mutable on the sampler would be
#: baked into the trace (SPEC §5.3b) — the same reason `DIAGNOSTICS` lives at
#: module level.
_STUB = {}


class _StubbedSampler(S.ConstrainedDiffusionSampler):
    """`ConstrainedDiffusionSampler` with the transformer replaced.

    **Only `sample_step` is overridden** — the forward pass, which genuinely
    needs the 51 GB checkpoint. `sample_next_canvas_constrained` (the accept
    rule, the emission composition, the trajectory/emission split, the
    feasibility threading) is the code under test and runs untouched.
    """

    def sample_step(self, *, canvas, sc_embeddings, cache, positions,
                    attention_mask, sliding_attention_mask,
                    current_noise_proportion, target_noise_proportion,
                    params, rng):
        return DS.SampleStepOutput(
            sc_embeddings=jnp.zeros_like(sc_embeddings),
            logits=jnp.broadcast_to(_STUB["logits"],
                                    (canvas.shape[0],) + _STUB["logits"].shape[-2:]),
            sampled_tokens=canvas,
            modified_tokens_mask=jnp.zeros_like(canvas, bool),
        )


class _FakeConfig:
    embed_dim = 4


class _FakeModel:
    config = _FakeConfig()


def _run_denoise(spec, *, variant, emission, logits, L, steps=3, seed=0,
                 remaining=0, entropy_bound=0.1, early_stop_fn=None,
                 confidence="mf", diagnose=False):
    """Drive the real denoising loop for one block."""
    _STUB["logits"] = jnp.asarray(logits)
    kw = dict(
        model=_FakeModel(), end_tokens=(1,), forbidden_tokens=None,
        sampling=None, cache_length=64, special_tokens=None, canvas_length=L,
        max_denoising_steps=steps, text_vocab_size=spec.vocab,
        n_states_bucket=spec.bucket, n_classes=spec.n_classes,
        variant=variant, emission=emission, confidence=confidence,
        diagnose=diagnose,
        sample_from_predictions=DS.SampleFromPredictions(
            entropy_bound=entropy_bound, text_vocab_size=spec.vocab),
    )
    if early_stop_fn is not None:
        kw["early_stop_fn"] = early_stop_fn
    sampler = _StubbedSampler(**kw)
    cache = {"l0": {"k": jnp.zeros((1, 64, 1, 1)),
                    "end_index": jnp.zeros((1,), jnp.int32)}}
    emit, feasible = sampler.sample_next_canvas_constrained(
        canvas_length=L, max_denoising_steps=steps, batch_size=1, cache=cache,
        params=None, rng=jax.random.PRNGKey(seed),
        full_attention_mask=jnp.ones((1, 64), bool),
        automaton=spec.traced(batched=True), remaining=jnp.int64(remaining))
    return [int(x) for x in emit[0]], bool(feasible[0])


@pytest.mark.parametrize("variant,emission",
                         [("j0", "map"), ("j0", "sample"), ("j1", "sample")])
def test_the_emitted_canvas_carries_no_unconstrained_random_token(chain, variant,
                                                                  emission):
    """**SPEC §3.1's residual hole, closed.**

    > The emitted canvas **is** the sample… Positions not accepted on the final
    > executed step are emitted as uniform random tokens over the whole 262k
    > vocab, with no cleanup pass. Measured [V-P0]: 31/31 blocks early-stopped,
    > 0/31 hit the budget, **and one still emitted a random token**.

    The entropy bound is set to 0 so the accept rule keeps exactly one position
    — the worst case for the stock design, where every other position would be
    uniform noise. Under J0/J1 the emission is a constrained object regardless
    of the accept prefix, so the canvas must still be in the language.

    This is the property the whole J0 design exists for, and it was previously
    checked only at kernel level, one canvas at a time, outside the loop that
    composes the emission.
    """
    spec = chain
    L = 4
    logits = jnp.zeros((1, L, spec.vocab))            # maximal entropy
    toks, ok = _run_denoise(spec, variant=variant, emission=emission,
                            logits=logits, L=L, entropy_bound=0.0)
    assert ok, "the loop reported Z == 0 on a satisfiable block"
    sim = _PySim(spec)
    assert sim.accepts(toks), (
        f"{variant}/{emission} emitted {toks}, which the automaton rejects — a "
        "non-accepted position reached the emission as an unconstrained token"
    )


def test_the_j2_baseline_does_leak_random_tokens(chain):
    """The other half of the same measurement, and the reason J0 exists.

    J2 keeps the constrained draw only at accepted positions and leaves the rest
    as the stock uniform renoise (SPEC §3.1's table, §7.2 baseline 3). With the
    accept prefix pinned to one position its emission must be able to leave the
    language — otherwise J2 is J0 under another name and its CS column is not
    evidence of anything.
    """
    spec = chain
    L = 4
    logits = jnp.zeros((1, L, spec.vocab))
    sim = _PySim(spec)
    rejected = 0
    for seed in range(20):
        toks, _ok = _run_denoise(spec, variant="j2", emission="sample",
                                 logits=logits, L=L, seed=seed,
                                 entropy_bound=0.0)
        if not sim.accepts(toks):
            rejected += 1
    assert rejected > 0, (
        "J2 never emitted a rejected string; it is running J0's fully "
        "constrained emission and the baseline measures nothing"
    )


def test_j0_keeps_the_trajectory_off_the_emission(chain):
    """SPEC §5.4: the widened carry exists precisely so the two differ.

    Under J0 the trajectory keeps stock uniform renoising — that is what keeps
    the model's *inputs* on its training distribution — while the emission is
    constrained. If a refactor ever collapsed the two fields, J0 would silently
    become J1 (grammar-valid noise fed back to a model never trained on it) and
    nothing else in the suite would notice.
    """
    spec = chain
    L = 4
    logits = jnp.zeros((1, L, spec.vocab))
    sim = _PySim(spec)

    S.DIAGNOSTICS.reset()
    toks, ok = _run_denoise(spec, variant="j0", emission="sample", logits=logits,
                            L=L, entropy_bound=0.0, steps=3, diagnose=True)
    rows = list(S.DIAGNOSTICS.rows)
    S.DIAGNOSTICS.reset()

    assert ok and sim.accepts(toks), "the J0 emission must be in the language"
    # `rows[k]["trajectory"]` is the canvas the model was fed at step `k`, i.e.
    # the previous step's trajectory. With the accept prefix pinned to one
    # position, J0's rule (`where(accepted, denoiser, uniform noise)`) leaves
    # three positions as uniform draws over the whole vocabulary.
    fed_back = [r["trajectory"] for r in rows[1:]]
    assert fed_back, "diagnostics recorded no steps"
    assert any(not sim.accepts(t) for t in fed_back), (
        "every canvas fed back to the model was already grammar-valid — the "
        "trajectory has collapsed onto the emission and J0 has silently become "
        "J1, feeding the model noise it was never trained to denoise"
    )
    assert any(t != rows[-1]["emitted"] for t in fed_back), (
        "trajectory == emission: the widened carry of SPEC §5.4 is not "
        "separating them"
    )


def test_a_block_that_cannot_finish_is_reported_infeasible(chain):
    """The `Z == 0` flag must survive the loop, not just the kernel.

    CLAUDE.md's causes (a) and (b) both have to raise, and the raise happens
    outside `jit` off the carried flag. A loop that dropped the flag would turn
    a provably impossible block into a confident-looking canvas —
    `jax.random.categorical` and `argmax` are shift-invariant, so an
    all-sentinel root is indistinguishable from a uniform one.
    """
    spec = chain
    L = 2
    logits = jnp.zeros((1, L, spec.vocab))
    # `chain` needs 3 tokens; with R = 0 after a 2-token canvas nothing finishes.
    _toks, ok = _run_denoise(spec, variant="j0", emission="map", logits=logits,
                             L=L, remaining=0)
    assert not ok, (
        "a block with no accepted continuation inside the budget was reported "
        "feasible; the Z == 0 signal is not reaching the carry"
    )


@pytest.fixture(scope="module")
def rigid():
    """An automaton admitting exactly ONE string of length 4: `(2, 3, 4, EOS)`.

    So `H(q_i) = 0` at every position while the stub logits stay uniform at
    `H(p) = log V`. That is the largest possible gap between the two entropy
    sources of SPEC §3.4, which is what makes the `mar` tests below decisive
    rather than suggestive.
    """
    V = 8
    members = ({2}, {3}, {4}, {1}, set(range(V)))
    edges = ((0, 1, 0), (1, 2, 1), (2, 3, 2), (3, 4, 3), (4, 4, 4))
    return _Spec(n_states=5, bucket=8, vocab=V, edges=edges, members=members,
                 is_neg=(False,) * 5, finals=frozenset({4}),
                 start=frozenset({0}))


def test_mar_confidence_changes_which_positions_are_accepted(rigid):
    """SPEC §3.4's substantive half: `--confidence=mar` must drive the accept
    rule from `H(q_i)`, not `H(p_i)`.

    > Constrained marginals `q_i` have restricted support and are far sharper.
    > Substituting `q_i` (the **Mar** variant) collapses entropies, accepts far
    > more positions per step…

    Measured through the per-step accept count, not by reading the source. With
    `entropy_bound = 0` and uniform logits the stock rule can only accept the
    single lowest-entropy position (its own entropy is subtracted off); under
    `mar` every `H(q_i)` is 0, so the whole canvas must accept on step one.

    This is the arm that carries the paper's larger accuracy gain (68.4 -> 76.4
    in its own ablation), so a `mar` flag that silently did nothing would be a
    reproduction failure, not a cosmetic one.
    """
    spec = rigid
    L = 4
    assert _enumerate_language(spec, L) == [(2, 3, 4, 1)]
    logits = jnp.zeros((1, L, spec.vocab))

    accepted = {}
    for conf in ("mf", "mar"):
        S.DIAGNOSTICS.reset()
        _run_denoise(spec, variant="j0", emission="map", logits=logits, L=L,
                     steps=1, confidence=conf, diagnose=True,
                     entropy_bound=0.0)
        rows = list(S.DIAGNOSTICS.rows)
        assert rows, f"{conf}: diagnostics recorded no step"
        accepted[conf] = rows[0]["n_accepted"]
    S.DIAGNOSTICS.reset()

    assert accepted["mf"] == 1, (
        f"sanity: with uniform logits (~{math.log(spec.vocab):.2f} nats at "
        f"every position) the stock rule accepts exactly one, got "
        f"{accepted['mf']}"
    )
    assert accepted["mar"] == L, (
        f"`--confidence=mar` accepted {accepted['mar']} of {L} positions on a "
        "canvas where the automaton leaves no choice at all. H(q_i) is "
        "identically 0 here, so the accept rule is still being driven by the "
        "unconstrained logits and the mar arm is measuring mf under another "
        "name."
    )


def test_the_early_stop_default_is_no_early_stop():
    """SPEC §3.1 [V-P0]: "a directly-constructed `DiffusionSampler` defaults to
    `NoEarlyStop` and so *always* runs the full 48 steps".

    This is load-bearing for two separate claims and is asserted here because
    both are silently wrong if it ever changes:

    1. **It is what makes the `mf`/`mar` split safe.** SPEC §3.4 warns that
       swapping the entropy source "changes stopping as well as acceptance
       (the same tensor feeds both)" — and offers "or split the flag" as the
       alternative. The implementation takes the split: `mar` re-drives the
       accept rule but `should_stop` still receives `out.logits`. That is
       sound *only* while no entropy-driven stopping rule is installed, which
       is the case for every eval arm, none of which passes `early_stop_fn`.
       Install one and the `mar` arm silently accepts on constrained entropies
       while stopping on unconstrained ones.
    2. **Every arm therefore pays the same step count**, so the §7.3 overhead
       comparison is not confounded by differential early stopping.
    """
    from gemma.diffusion import _early_stopping

    field = {f.name: f for f in dataclasses.fields(
        S.ConstrainedDiffusionSampler)}["early_stop_fn"]
    default = (field.default_factory() if field.default_factory is not
               dataclasses.MISSING else field.default)
    assert isinstance(default, _early_stopping.NoEarlyStop), (
        f"early_stop_fn now defaults to {default!r}. SPEC §3.4's "
        "acceptance/stopping confound is no longer inert: `--confidence=mar` "
        "re-drives acceptance from H(q) but leaves `should_stop` reading the "
        "unconstrained logits, and the two arms no longer run the same number "
        "of denoising steps."
    )


# ===========================================================================
# 11. Traced values, static shapes.  SPEC §5.3(b).
# ===========================================================================
# `tests/test_constrained_draw.py` already pins this for `joint_draw` and it is
# a good test. It covers **only** `joint_draw`, so the default emission (`map`)
# and the per-block `advance_states` — both of which take the automaton as an
# argument and both of which run once per block — were unpinned.


def _bucket_automaton(i, *, n, V=8, n_classes=2, rng=None):
    """Structurally different automata sharing one `|S|` bucket."""
    rng = rng or np.random.default_rng(i)
    return Automaton(
        edge_src=jnp.asarray(rng.integers(0, n, size=6), jnp.int32),
        edge_dst=jnp.asarray(rng.integers(0, n, size=6), jnp.int32),
        edge_class=jnp.asarray(rng.integers(0, n_classes, size=6), jnp.int32),
        edge_valid=jnp.ones(6, bool),
        csr_indices=jnp.arange(V * n_classes, dtype=jnp.int32) % V,
        csr_indptr=jnp.asarray([0, V, 2 * V], jnp.int32),
        is_neg=jnp.zeros(n_classes, bool),
        d=jnp.asarray(rng.integers(0, 3, size=n), jnp.int32),
        is_final=jnp.zeros(n, bool).at[i % n].set(True),
        active=jnp.zeros(n, bool).at[i % n].set(True),
    )


def test_the_default_map_emission_traces_once_per_bucket():
    """SPEC §5.3(b), for `--emission=map`.

    "Automaton values must be traced; only shapes may be static." At 4,549
    BFCL-Live schemas a per-grammar retrace is hours of compilation and it
    defeats the compilation cache. Measured with XLA's own compile counter, not
    by reading the code.
    """
    n, V, L, nC = 8, 8, 4, 2
    C.joint_map.clear_cache()
    p = jnp.full((L, V), 1.0 / V)
    for i in range(10):
        C.joint_map(p, _bucket_automaton(i, n=n, V=V, n_classes=nC),
                    jnp.int64(16), n, nC)
    got = C.joint_map._cache_size()
    assert got == 1, (
        f"{got} XLA traces for 10 automata of identical shape — something "
        "per-request landed on a static argument (SPEC §5.3)"
    )
    C.joint_map.clear_cache()
    for m in (8, 16):
        C.joint_map(p, _bucket_automaton(0, n=m, V=V, n_classes=nC),
                    jnp.int64(16), m, nC)
    assert C.joint_map._cache_size() == 2, (
        "a different |S| bucket must retrace; without this the check above "
        "would also pass on a function that never specialises"
    )


def test_advance_states_traces_once_per_bucket():
    """The same, for the per-block boundary op. It runs once per block inside
    the outer `lax.while_loop`, so a retrace here is a retrace per request."""
    n, V, nC = 8, 8, 2
    C.advance_states.clear_cache()
    toks = jnp.zeros((4,), jnp.int32)
    for i in range(10):
        C.advance_states(_bucket_automaton(i, n=n, V=V, n_classes=nC), toks,
                         n, nC, V)
    got = C.advance_states._cache_size()
    assert got == 1, f"{got} traces for 10 same-shape automata (SPEC §5.3)"

    # The control, matching the `joint_map` sibling: bucketing is what bounds
    # the compile count, so a genuinely different `|S|` MUST retrace. Without
    # this, a function that never specialises on `|S|` at all would also satisfy
    # the assertion above.
    C.advance_states.clear_cache()
    for m in (8, 16):
        C.advance_states(_bucket_automaton(0, n=m, V=V, n_classes=nC), toks,
                         m, nC, V)
    assert C.advance_states._cache_size() == 2, (
        "a different |S| bucket did not retrace, so the check above is not "
        "evidence that the automaton is traced rather than static"
    )



# ===========================================================================
# 12. Diagnostic (report-only) — string-body saturation in shipped runs.
# ===========================================================================

def test_report_string_body_saturation_in_existing_artifacts(capsys):
    """**A diagnostic, not an assertion — deliberately.**

    SPEC §3.5 trap 4 prescribes the sibling of this one ("log the stop-token
    position per block; clustering at 255 means this is still open"). The
    string-body analogue is: did any decode spend a whole canvas inside the
    99.4%-of-Σ self-loop? There is **no threshold that is right in general** — a
    long string value can be the correct answer — so this reports and asserts
    nothing about the distribution.

    Two limits, stated because a diagnostic that overstates itself is worse
    than none:

    - the shipped eval artifacts store **decoded text**, not token ids and not
      block boundaries, so "tokens of string interior per block" is not
      recoverable; the character length of the longest string literal is the
      available proxy;
    - the unbalanced-quote count is computed on the raw emission including the
      channel header and any markdown fence, and the `--think` and free-text
      arms are unconstrained there, so a nonzero count is not by itself a
      defect.

    What it is good for: a large gap between the median and the maximum string
    length, or a rising unbalanced-quote count in the `j0`/`j1` arms, is the
    signature of `b_L` only just managing to force closure at the budget edge.
    """
    import glob
    import json
    import os
    import statistics

    paths = sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts", "*.json")))
    if not paths:
        pytest.skip("no artifacts/ to scan")

    def longest_literal(t):
        best = cur = 0
        inside = False
        i = 0
        while i < len(t):
            c = t[i]
            if inside and c == "\\":
                i += 2
                cur += 2
                continue
            if c == '"':
                if inside:
                    best = max(best, cur)
                inside = not inside
                cur = 0
            elif inside:
                cur += 1
            i += 1
        return best, inside

    per_variant: dict[str, list] = {}
    unbalanced: dict[str, list] = {}
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:                     # noqa: BLE001 - artifact scan only
            continue
        rows, variant = d.get("rows"), d.get("variant")
        if not isinstance(rows, list) or variant is None:
            continue
        for r in rows:
            t = r.get("text")
            if not isinstance(t, str):
                continue
            n, open_at_end = longest_literal(t)
            per_variant.setdefault(variant, []).append(n)
            if open_at_end:
                unbalanced.setdefault(variant, []).append(
                    (os.path.basename(p), r.get("id")))

    if not per_variant:
        pytest.skip("no artifact carries per-record emissions")

    lines = ["", "string-body saturation diagnostic (report only)",
             f"{'variant':16s} {'n':>6s} {'median':>7s} {'p90':>6s} {'max':>6s} "
             f"{'open-quote':>11s}"]
    for v in sorted(per_variant):
        xs = sorted(per_variant[v])
        p90 = xs[min(len(xs) - 1, int(len(xs) * 0.9))]
        lines.append(f"{v:16s} {len(xs):6d} {statistics.median(xs):7.0f} "
                     f"{p90:6d} {xs[-1]:6d} {len(unbalanced.get(v, [])):11d}")
    for v in ("j0", "j1"):
        for rec in unbalanced.get(v, [])[:5]:
            lines.append(f"  open quote in a guarantee arm: {v} {rec}")
    with capsys.disabled():
        print("\n".join(lines))


def test_the_denoising_carry_has_no_monotone_commit_set():
    """SPEC §3.2: "`selection_mask` is rebuilt from `jnp.zeros_like` every call;
    nothing mask-shaped is in the carry… **This is also why automaton state can
    be recomputed statelessly each step.** … Do not add a monotone commit set to
    make it look like LLaDA."

    The statelessness of the per-step automaton computation is a *consequence*
    of there being no accumulated accept set. A field carrying one would make
    every recomputation in `sample_next_canvas_constrained` unsound without
    changing any test that only looks at outputs.
    """
    names = {f.name for f in dataclasses.fields(S._ConstrainedCarry)}
    assert names == {"step", "canvas", "emit_canvas", "sc_embeddings", "rng",
                     "done", "feasible"}, (
        f"the denoising carry grew a field: {names}. If it is mask-shaped, the "
        "accept set has become monotone and SPEC §3.2's stateless recomputation "
        "argument no longer holds."
    )
