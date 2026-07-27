"""Simulation and round-trip validation. SPEC §8 (Phase 1 exit), §6.4.

Two directions, and both are needed:

- **FA -> validator.** Random-walk the automaton, decode the token sequence, and
  hand it to an *independent* validator (`json.loads` + a real JSON Schema
  checker, or `ast.parse` for the Python format). Catches a grammar that emits
  garbage.
- **Ground truth -> FA.** Run the benchmark's own reference answers through the
  automaton. Catches a grammar that is *over*-constrained — which is the far
  more dangerous direction, because the model's output will still look
  well-formed while scoring zero.

The simulator here is deliberately plain Python over the compiled artifact, so
it is independent of the JAX inference path it will later be used to check.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Iterable, Sequence

import numpy as np

from diffgemma_fa.compile.automaton import INF_DISTANCE, CompiledAutomaton

__all__ = ["Simulator", "SampleResult", "sample_strings"]


class Simulator:
    """Reference simulator over a `CompiledAutomaton`.

    Tracks a *set* of states, so it is correct for NFAs as well as DFAs — SPEC
    warns repeatedly that DFA-only testing hides real bugs (eq (8)'s edge
    multiplicity, start vectors).
    """

    def __init__(self, automaton: CompiledAutomaton) -> None:
        self.a = automaton
        self._members: list[set[int]] = self._materialize_classes()
        # (src, dst) -> class id, and src -> [(dst, class)]
        self._out: dict[int, list[tuple[int, int]]] = {}
        for e in range(automaton.n_edges):
            self._out.setdefault(int(automaton.edge_src[e]), []).append(
                (int(automaton.edge_dst[e]), int(automaton.edge_class[e]))
            )

    def _materialize_classes(self) -> list[set[int]]:
        """Decode every class back to its true member set."""
        t = self.a.tables
        out = []
        for c in range(t.n_classes):
            stored = set(int(x) for x in t.sum_indices[t.sum_indptr[c]:t.sum_indptr[c + 1]])
            if t.sum_is_neg[c]:
                out.append(set(range(t.vocab_size)) - stored)
            else:
                out.append(stored)
        return out

    def start_states(self) -> set[int]:
        return {int(i) for i in np.nonzero(self.a.start_vector)[0]}

    def step(self, states: Iterable[int], token: int) -> set[int]:
        """`delta*(states, token)`."""
        nxt: set[int] = set()
        for s in states:
            for dst, cls in self._out.get(s, ()):
                if token in self._members[cls]:
                    nxt.add(dst)
        return nxt

    def run(self, tokens: Sequence[int], *, states: Iterable[int] | None = None) -> set[int]:
        cur = set(states) if states is not None else self.start_states()
        for tok in tokens:
            cur = self.step(cur, tok)
            if not cur:
                return cur
        return cur

    def accepts(self, tokens: Sequence[int]) -> bool:
        return any(self.a.is_final[s] for s in self.run(tokens))

    def is_viable_prefix(self, tokens: Sequence[int], *, budget: int | None = None) -> bool:
        """`delta*(A, w) != {}` and, with a budget, every state can still finish.

        This is exactly `test_guarantee.py`'s per-block assertion (SPEC §6.4),
        which is a *different* claim from acceptance: a non-final block's canvas
        ends live-but-not-accepting and is not accepted on its own.
        """
        reached = self.run(tokens)
        if not reached:
            return False
        if budget is None:
            return True
        return all(self.a.d[s] <= budget for s in reached)


@dataclasses.dataclass(frozen=True)
class SampleResult:
    tokens: tuple[int, ...]
    states: tuple[int, ...]
    accepted: bool
    hit_length_cap: bool


def sample_strings(
    automaton: CompiledAutomaton,
    *,
    n: int,
    max_len: int = 256,
    seed: int = 0,
    stop_at_final: bool = True,
) -> list[SampleResult]:
    """Random-walk the automaton to produce accepted token sequences.

    Walks only along edges that keep `d(s) < INF`, and once `max_len` is close
    prefers edges that reduce `d`, so the walk terminates inside the budget
    rather than wandering the unscored tail forever.
    """
    sim = Simulator(automaton)
    rng = random.Random(seed)
    a = automaton
    out: list[SampleResult] = []

    for i in range(n):
        state = next(iter(sim.start_states()))
        toks: list[int] = []
        states = [state]
        capped = False
        while True:
            if stop_at_final and a.is_final[state] and toks:
                break
            if len(toks) >= max_len:
                capped = True
                break
            edges = sim._out.get(state, [])  # noqa: SLF001
            viable = [(d, c) for d, c in edges
                      if a.d[d] < INF_DISTANCE and sim._members[c]]
            if not viable:
                break
            remaining = max_len - len(toks)
            # Prefer progress toward acceptance when the budget is tight.
            closing = [(d, c) for d, c in viable if a.d[d] < a.d[state]]
            pool = closing if (closing and remaining <= a.d[state] + 2) else viable
            dst, cls = rng.choice(pool)
            members = sim._members[cls]
            # Sampling from a 260k-member set: take a bounded random probe
            # rather than materialising a sorted list every step.
            tok = rng.choice(sorted(members)) if len(members) <= 512 else \
                rng.choice([t for t in rng.sample(range(a.vocab_size), 64)
                            if t in members] or sorted(members)[:1])
            toks.append(int(tok))
            state = int(dst)
            states.append(state)

        out.append(SampleResult(
            tokens=tuple(toks),
            states=tuple(states),
            accepted=bool(a.is_final[state]),
            hit_length_cap=capped,
        ))
    return out
