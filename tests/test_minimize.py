"""Tests for the Valmari port. SPEC §4.5.

Minimization sits under everything else in Phase 1, and a subtly wrong
minimizer produces an automaton that still *looks* fine — it accepts a
plausible language, just not the right one. So the property test here is
language equivalence by exhaustive enumeration, not state counts.
"""

from __future__ import annotations

import itertools
import random

import pytest

from diffgemma_fa.compile.minimize import Dfa, minimize


def simulate(dfa: Dfa, word) -> bool:
    """Reference simulator. Partial: a missing transition rejects."""
    if dfa.n_states == 0:
        return False
    table = {(s, l): d for s, l, d in dfa.transitions}
    q = dfa.start
    for sym in word:
        if (q, sym) not in table:
            return False
        q = table[(q, sym)]
    return q in dfa.finals


def language(dfa: Dfa, alphabet, max_len: int) -> set[tuple]:
    out = set()
    for n in range(max_len + 1):
        for w in itertools.product(alphabet, repeat=n):
            if simulate(dfa, w):
                out.add(w)
    return out


def random_dfa(rng: random.Random, n_states: int, alphabet, density: float) -> Dfa:
    """A random *partial*, deterministic automaton."""
    trans = []
    for s in range(n_states):
        for a in alphabet:
            if rng.random() < density:
                trans.append((s, a, rng.randrange(n_states)))
    finals = frozenset(s for s in range(n_states) if rng.random() < 0.35)
    return Dfa(n_states=n_states, transitions=tuple(trans), start=0, finals=finals)


# --------------------------------------------------------------------------
# Hand-built cases
# --------------------------------------------------------------------------

def test_already_minimal_is_unchanged():
    # Accepts exactly the words over {0,1} ending in 1.
    dfa = Dfa(
        n_states=2,
        transitions=((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)),
        start=0,
        finals=frozenset({1}),
    )
    r = minimize(dfa)
    assert r.states_after == 2
    assert language(r.dfa, [0, 1], 6) == language(dfa, [0, 1], 6)


def test_merges_equivalent_states():
    # Three states that all accept the same language; must collapse.
    dfa = Dfa(
        n_states=4,
        transitions=((0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 3),
                     (2, 0, 3), (2, 1, 3)),
        start=0,
        finals=frozenset({3}),
    )
    r = minimize(dfa)
    assert r.states_after == 3, "states 1 and 2 are equivalent and must merge"
    assert language(r.dfa, [0, 1], 6) == language(dfa, [0, 1], 6)


def test_trims_unreachable_states():
    dfa = Dfa(
        n_states=4,
        transitions=((0, 0, 1), (2, 0, 3)),  # 2 and 3 unreachable from 0
        start=0,
        finals=frozenset({1, 3}),
    )
    r = minimize(dfa)
    assert r.states_after == 2
    assert r.state_map[2] is None and r.state_map[3] is None


def test_trims_non_coaccessible_states():
    # State 2 is reachable but no final state is reachable from it.
    dfa = Dfa(
        n_states=3,
        transitions=((0, 0, 1), (0, 1, 2), (2, 0, 2)),
        start=0,
        finals=frozenset({1}),
    )
    r = minimize(dfa)
    assert r.state_map[2] is None
    assert language(r.dfa, [0, 1], 5) == language(dfa, [0, 1], 5)


def test_empty_language_reports_zero_states():
    """SPEC's Z==0 taxonomy: a genuinely empty automaton is a real bug, so it
    must be visible, not papered over with a sink state."""
    dfa = Dfa(n_states=2, transitions=((0, 0, 1),), start=0, finals=frozenset())
    r = minimize(dfa)
    assert r.states_after == 0
    assert all(x is None for x in r.state_map)


def test_start_state_is_zero():
    dfa = Dfa(
        n_states=3,
        transitions=((0, 0, 1), (1, 0, 2), (2, 0, 2)),
        start=0,
        finals=frozenset({2}),
    )
    r = minimize(dfa)
    assert r.dfa.start == 0


def test_never_completes_with_a_sink():
    """THE COMPLETION TRAP (SPEC §4.5). The output must stay partial: total
    transitions must not blow up to n * |alphabet|."""
    alphabet = list(range(50))
    dfa = Dfa(
        n_states=3,
        transitions=((0, 7, 1), (1, 9, 2)),
        start=0,
        finals=frozenset({2}),
    )
    r = minimize(dfa)
    assert r.transitions_after <= 2
    assert r.dfa.n_states * len(alphabet) > r.transitions_after * 10


def test_large_alphabet_is_not_enumerated():
    """|Sigma| must never appear in the cost. Labels are 262k-scale ids here;
    if the implementation iterated the alphabet this would not finish."""
    trans = tuple((i, 200_000 + i, i + 1) for i in range(200))
    dfa = Dfa(n_states=201, transitions=trans, start=0, finals=frozenset({200}))
    r = minimize(dfa)
    assert r.states_after == 201
    assert simulate(r.dfa, [200_000 + i for i in range(200)])


# --------------------------------------------------------------------------
# Property test: language preservation on random partial DFAs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(40))
def test_language_is_preserved(seed: int):
    rng = random.Random(seed)
    alphabet = [0, 1, 2]
    dfa = random_dfa(rng, rng.randint(2, 8), alphabet, density=0.6)
    r = minimize(dfa)
    assert language(r.dfa, alphabet, 5) == language(dfa, alphabet, 5)


@pytest.mark.parametrize("seed", range(40))
def test_result_is_minimal_and_deterministic(seed: int):
    """Minimizing twice must be a no-op, and the output must stay a DFA."""
    rng = random.Random(seed + 500)
    alphabet = [0, 1]
    dfa = random_dfa(rng, rng.randint(3, 10), alphabet, density=0.7)
    once = minimize(dfa)
    twice = minimize(once.dfa)
    assert twice.states_after == once.states_after

    seen = set()
    for s, l, _ in once.dfa.transitions:
        assert (s, l) not in seen, "output is nondeterministic"
        seen.add((s, l))


@pytest.mark.parametrize("seed", range(20))
def test_state_map_agrees_with_simulation(seed: int):
    """Every surviving old state must map to a new state accepting the same
    residual language."""
    rng = random.Random(seed + 900)
    alphabet = [0, 1]
    dfa = random_dfa(rng, rng.randint(3, 7), alphabet, density=0.8)
    r = minimize(dfa)
    if r.dfa.n_states == 0:
        return
    old_tbl = {(s, l): d for s, l, d in dfa.transitions}
    new_tbl = {(s, l): d for s, l, d in r.dfa.transitions}

    def residual(table, q, finals, max_len):
        out = set()
        for n in range(max_len + 1):
            for w in itertools.product(alphabet, repeat=n):
                cur = q
                ok = True
                for sym in w:
                    if (cur, sym) not in table:
                        ok = False
                        break
                    cur = table[(cur, sym)]
                if ok and cur in finals:
                    out.add(w)
        return out

    for old in range(dfa.n_states):
        new = r.state_map[old]
        if new is None:
            continue
        assert residual(old_tbl, old, dfa.finals, 4) == \
               residual(new_tbl, new, r.dfa.finals, 4)


# ==========================================================================
# Valmari's precondition, checked rather than assumed
# ==========================================================================

def test_a_duplicate_transition_raises_instead_of_changing_the_language():
    """Valmari 2012's `mark()` has no re-mark guard — the port is faithful, and
    the *precondition* is what was unchecked. Marking the same element twice
    pushes `marked[s]` past the set size and corrupts the partition.

    This exact instance was measured: it accepts `{(1,), (0,1)}` and used to
    minimize to an infinite language, with no error. Silent language change is
    strictly worse than a raise.
    """
    with pytest.raises(ValueError, match="duplicate transition"):
        minimize(Dfa(3, ((0, 0, 1), (0, 0, 1), (1, 1, 2), (0, 1, 2)), 0,
                     frozenset({2})))


def test_a_nondeterministic_pair_raises_with_a_usable_message():
    """Previously an `IndexError` from deep inside the refine loop, where the
    cause is unrecoverable from the traceback."""
    with pytest.raises(ValueError, match="nondeterministic"):
        minimize(Dfa(3, ((0, 0, 1), (0, 0, 2)), 0, frozenset({1, 2})))


def test_the_realistic_accident_is_a_union_of_two_transition_tuples():
    """SPEC §3.8's `FA_grammar | FA_refusal` would naturally be built by
    concatenating two transition tuples over a shared state space. That is what
    makes this a realistic accident rather than a theoretical one."""
    grammar = ((0, 0, 1), (1, 1, 2))
    refusal = ((0, 0, 1), (1, 2, 2))        # shares (0, 0, 1)
    with pytest.raises(ValueError):
        minimize(Dfa(3, grammar + refusal, 0, frozenset({2})))
