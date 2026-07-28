"""Valmari 2012 DFA minimization. SPEC §4.5.

Antti Valmari, *Fast brief practical DFA minimization*, Information Processing
Letters 112(6):213-217, 2012. https://www.cs.cmu.edu/~cdm/resources/Valmari2012.pdf

**Why not Hopcroft.** Hopcroft's bound is `O(|Σ|·n·log n)` because its splitter
worklist holds `(symbol, block)` pairs. At `|Σ| = 262,144` and `n = 20,000` that
is 7.5e10 operations and a 21 GB dense table. Valmari's bound is
`n + m·log₂ m` where `m` is the number of *transitions actually present* —
**`|Σ|` never appears** — giving 3.0e6 here, ~25,000x less. Pure Python is
therefore fine; this runs in seconds.

**THE COMPLETION TRAP (SPEC §4.5).** Never complete the partial DFA with a sink
state to make it total. That converts `m = 170k` real transitions into
`m = n·|Σ| = 5.24e9`. This implementation handles partial DFAs natively — a
missing `(state, label)` pair is simply a transition that is not in the list.

The algorithm also *trims*: states unreachable from the start, and states from
which no final state is reachable, are removed before refinement.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Sequence

__all__ = ["Dfa", "minimize", "MinimizationResult"]


@dataclasses.dataclass(frozen=True)
class Dfa:
    """A partial DFA over integer labels.

    Attributes:
      n_states: states are `0 .. n_states-1`.
      transitions: `(src, label, dst)` triples. Partial — absent pairs are
        simply missing, never routed to a sink.
      start: the initial state.
      finals: accepting states.
    """

    n_states: int
    transitions: tuple[tuple[int, int, int], ...]
    start: int
    finals: frozenset[int]

    def __post_init__(self) -> None:
        if self.n_states == 0:
            # The canonical empty-language automaton. Deliberately
            # representable: SPEC's Z==0 taxonomy wants "genuinely empty" to be
            # a visible, checkable state rather than something smoothed over.
            if self.transitions or self.finals:
                raise ValueError("n_states == 0 but transitions/finals present")
            return
        for s, _, d in self.transitions:
            if not (0 <= s < self.n_states and 0 <= d < self.n_states):
                raise ValueError(f"transition out of range: ({s}, _, {d})")
        if not (0 <= self.start < self.n_states):
            raise ValueError(f"start state {self.start} out of range")
        for f in self.finals:
            if not (0 <= f < self.n_states):
                raise ValueError(f"final state {f} out of range")

    @property
    def is_empty(self) -> bool:
        """True if this accepts nothing at all."""
        return self.n_states == 0

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    @property
    def alphabet(self) -> frozenset[int]:
        return frozenset(lbl for _, lbl, _ in self.transitions)


@dataclasses.dataclass(frozen=True)
class MinimizationResult:
    dfa: Dfa
    #: old state id -> new state id, or None if the state was trimmed away.
    state_map: tuple[int | None, ...]
    states_before: int
    states_after: int
    transitions_before: int
    transitions_after: int

    @property
    def state_ratio(self) -> float:
        return self.states_before / max(1, self.states_after)

    @property
    def transition_ratio(self) -> float:
        return self.transitions_before / max(1, self.transitions_after)


class _RefinablePartition:
    """Valmari's refinable partition, ported verbatim in structure.

    Elements are `0 .. n-1`. `mark(e)` moves `e` to the front of its set;
    `split()` then cuts every touched set into its marked and unmarked halves,
    always making the *smaller* half the new set — which is what buys the
    `log m` factor.
    """

    __slots__ = ("n_sets", "elems", "loc", "sidx", "first", "past", "marked",
                 "touched", "n_touched")

    def __init__(self, n: int) -> None:
        self.n_sets = 1 if n > 0 else 0
        self.elems = list(range(n))
        self.loc = list(range(n))
        self.sidx = [0] * n
        self.first = [0] * (n + 1)
        self.past = [0] * (n + 1)
        self.marked = [0] * (n + 1)
        self.touched = [0] * (n + 1)
        self.n_touched = 0
        if self.n_sets:
            self.first[0] = 0
            self.past[0] = n

    def mark(self, e: int) -> None:
        s = self.sidx[e]
        i = self.loc[e]
        j = self.first[s] + self.marked[s]
        # swap e into the marked prefix of its set
        self.elems[i] = self.elems[j]
        self.loc[self.elems[i]] = i
        self.elems[j] = e
        self.loc[e] = j
        if self.marked[s] == 0:
            self.touched[self.n_touched] = s
            self.n_touched += 1
        self.marked[s] += 1

    def split(self) -> None:
        while self.n_touched:
            self.n_touched -= 1
            s = self.touched[self.n_touched]
            j = self.first[s] + self.marked[s]
            if j == self.past[s]:
                # everything in s was marked: no split
                self.marked[s] = 0
                continue
            # Make the smaller half the new set — the source of the log factor.
            if self.marked[s] <= self.past[s] - j:
                self.first[self.n_sets] = self.first[s]
                self.past[self.n_sets] = j
                self.first[s] = j
            else:
                self.past[self.n_sets] = self.past[s]
                self.first[self.n_sets] = j
                self.past[s] = j
            for i in range(self.first[self.n_sets], self.past[self.n_sets]):
                self.sidx[self.elems[i]] = self.n_sets
            self.marked[s] = 0
            self.marked[self.n_sets] = 0
            self.n_sets += 1


def _reachable(n: int, adj_start: Sequence[int], adj: Sequence[int],
               ends: Sequence[int], roots: Iterable[int]) -> bytearray:
    """Depth-first reachability over an adjacency built by `_make_adjacent`."""
    seen = bytearray(n)
    stack = []
    for r in roots:
        if not seen[r]:
            seen[r] = 1
            stack.append(r)
    while stack:
        q = stack.pop()
        for j in range(adj_start[q], adj_start[q + 1]):
            t = adj[j]
            nxt = ends[t]
            if not seen[nxt]:
                seen[nxt] = 1
                stack.append(nxt)
    return seen


def _make_adjacent(n: int, m: int, key: Sequence[int]) -> tuple[list[int], list[int]]:
    """Bucket transition indices by `key[t]` (a state), CSR-style.

    Returns `(start, adj)` with the transitions of state `q` at
    `adj[start[q] : start[q+1]]`.
    """
    start = [0] * (n + 1)
    for t in range(m):
        start[key[t]] += 1
    for q in range(n):
        start[q + 1] += start[q]
    adj = [0] * m
    for t in range(m - 1, -1, -1):
        start[key[t]] -= 1
        adj[start[key[t]]] = t
    return start, adj


def minimize(dfa: Dfa) -> MinimizationResult:
    """Minimize a partial DFA. Trims first, then refines.

    Returns the minimized DFA plus the old->new state map.

    Determinism is **checked here**, not assumed. Valmari 2012 requires at most
    one transition per `(state, label)`: its `mark()` has no re-mark guard, so
    marking the same element twice pushes `marked[s]` past the set size and
    corrupts the partition. The port is faithful; the *precondition* was what
    went unchecked.

    Both failure modes are worse than a raise. An exact duplicate `(s, a, d)`
    silently CHANGES THE LANGUAGE -- verified:
    `Dfa(3, ((0,0,1),(0,0,1),(1,1,2),(0,1,2)), 0, {2})` accepts `{(1,), (0,1)}`
    and minimizes to an infinite one. A genuine nondeterministic pair raises
    `IndexError` from deep inside the refine loop, where the cause is
    unrecoverable from the traceback.

    Concatenating two transition tuples is exactly how one would build SPEC
    §3.8's `FA_grammar | FA_refusal` union, which makes this a realistic
    accident rather than a theoretical one. For a genuine NFA (Spider, SPEC
    §4.6) this is not the right algorithm and `minimize` must not be called --
    the check is what makes that a loud failure instead of a silent one.

    The validation lives here rather than on `Dfa.__post_init__` because `Dfa`
    doubles as the plain transition container `compile_automaton` accepts, and
    that path legitimately carries NFAs (it computes `is_dfa` from them).

    Raises:
      ValueError: on a duplicate or nondeterministic transition.
    """
    seen: dict[tuple[int, int], int] = {}
    for s_, a_, d_ in dfa.transitions:
        prev = seen.get((s_, a_))
        if prev is None:
            seen[(s_, a_)] = d_
        elif prev == d_:
            raise ValueError(
                f"duplicate transition ({s_}, {a_}, {d_}); Valmari requires at "
                "most one transition per (state, label) and silently corrupts "
                "the partition otherwise"
            )
        else:
            raise ValueError(
                f"nondeterministic: ({s_}, {a_}) goes to both {prev} and {d_}. "
                "This is a DFA minimizer; determinize first (SPEC §4.5)"
            )

    n0, m0 = dfa.n_states, dfa.n_transitions
    if n0 == 0:
        return MinimizationResult(dfa, (), 0, 0, 0, 0)

    tails = [t[0] for t in dfa.transitions]
    labels = [t[1] for t in dfa.transitions]
    heads = [t[2] for t in dfa.transitions]

    # --- trim: forward-reachable from start, backward-reachable from finals --
    f_start, f_adj = _make_adjacent(n0, m0, tails)
    fwd = _reachable(n0, f_start, f_adj, heads, [dfa.start])

    b_start, b_adj = _make_adjacent(n0, m0, heads)
    bwd = _reachable(n0, b_start, b_adj, tails, [f for f in dfa.finals if fwd[f]])

    live = [q for q in range(n0) if fwd[q] and bwd[q]]
    if not live or not fwd[dfa.start] or not bwd[dfa.start]:
        # The language is empty. Report it rather than inventing a sink state:
        # SPEC's Z==0 taxonomy calls a genuinely empty automaton a real bug.
        empty = Dfa(n_states=0, transitions=(), start=0, finals=frozenset())
        return MinimizationResult(empty, tuple([None] * n0), n0, 0, m0, 0)

    old_to_live: dict[int, int] = {q: i for i, q in enumerate(live)}
    n = len(live)
    kept = [t for t in range(m0)
            if (fwd[tails[t]] and bwd[tails[t]] and fwd[heads[t]] and bwd[heads[t]])]
    m = len(kept)
    T = [old_to_live[tails[t]] for t in kept]
    L = [labels[t] for t in kept]
    H = [old_to_live[heads[t]] for t in kept]
    start = old_to_live[dfa.start]
    finals = {old_to_live[f] for f in dfa.finals if f in old_to_live}

    # --- initial block partition: finals vs non-finals ----------------------
    blocks = _RefinablePartition(n)
    for q in finals:
        blocks.mark(q)
    blocks.split()

    # --- initial cord partition: group transitions by label -----------------
    # Sorting by label first is what lets the bound be O(n + m log n); SPEC §4.5
    # notes this. |Σ| is never enumerated — only labels that actually occur.
    cords = _RefinablePartition(m)
    if m:
        order = sorted(range(m), key=lambda t: L[t])
        cords.n_sets = 0
        cords.marked[0] = 0
        a = L[order[0]]
        for i, t in enumerate(order):
            if L[t] != a:
                a = L[t]
                cords.past[cords.n_sets] = i
                cords.n_sets += 1
                cords.first[cords.n_sets] = i
                cords.marked[cords.n_sets] = 0
            cords.elems[i] = t
            cords.sidx[t] = cords.n_sets
            cords.loc[t] = i
        cords.past[cords.n_sets] = m
        cords.n_sets += 1

    # --- refine -------------------------------------------------------------
    # Incoming transitions per state, for propagating block splits to cords.
    in_start, in_adj = _make_adjacent(n, m, H)

    b, c = 1, 0
    while c < cords.n_sets:
        for i in range(cords.first[c], cords.past[c]):
            blocks.mark(T[cords.elems[i]])
        blocks.split()
        c += 1
        while b < blocks.n_sets:
            for i in range(blocks.first[b], blocks.past[b]):
                q = blocks.elems[i]
                for j in range(in_start[q], in_start[q + 1]):
                    cords.mark(in_adj[j])
            cords.split()
            b += 1

    # --- rebuild ------------------------------------------------------------
    # Number blocks so the start block is 0, purely for readability.
    block_of = blocks.sidx
    remap: dict[int, int] = {block_of[start]: 0}
    for q in range(n):
        remap.setdefault(block_of[q], len(remap))

    new_trans = sorted({
        (remap[block_of[T[t]]], L[t], remap[block_of[H[t]]]) for t in range(m)
    })
    new_finals = frozenset(remap[block_of[q]] for q in finals)

    state_map: list[int | None] = [None] * n0
    for old, i in old_to_live.items():
        state_map[old] = remap[block_of[i]]

    result = Dfa(
        n_states=len(remap),
        transitions=tuple(new_trans),
        start=0,
        finals=new_finals,
    )
    return MinimizationResult(
        dfa=result,
        state_map=tuple(state_map),
        states_before=n0,
        states_after=result.n_states,
        transitions_before=m0,
        transitions_after=result.n_transitions,
    )
