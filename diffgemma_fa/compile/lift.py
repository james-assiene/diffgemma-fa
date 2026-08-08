"""Byte regex -> token-level DFA. SPEC §4.3.

`outlines_core` is the only shipped library that materialises an explicit
`state -> {token_id -> state}` table (`Index(regex, vocabulary).get_transitions()`).
XGrammar and llguidance are per-step bitmask engines with no automaton export.

Traps handled here, all verified in Phase 0:

- **State ids are raw `regex-automata` dense ids** — byte offsets, observed with
  a uniform stride of **64** (not 8, as SPEC originally said), running 256…2,816
  for a small schema. Not dense. They are renumbered immediately.
- `get_transitions()` deep-clones the whole nested map into Python dicts on
  every call, so it is called exactly once.
- Matching is **anchored**, and a token is allowed only if the DFA consumes
  **all** of its bytes.
- `guide.advance(eos)` raises even though `T[final][eos]` exists, so stop tokens
  are handled out of band (SPEC §3.5) and never enter this table.
- **`outlines_core` silently drops every token transition that leaves an
  accepting state for a non-accepting one.** See `_EOI_ANCHOR` below; this is
  the bug that made every repeated *group* lift as its minimum count.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from diffgemma_fa.compile.minimize import Dfa, minimize

__all__ = ["LiftResult", "lift_regex", "anchor_at_eoi"]

#: The end-of-haystack anchor appended to every regex before it reaches
#: `outlines_core.Index`. **Not cosmetic — it is a correctness fix.**
#:
#: `outlines_core` 0.2.14 `src/index.rs`, in the token loop:
#:
#: ```rust
#: let is_intermediate_state = !dfa.is_match_state(next_state);
#: let is_full_match_state = dfa.is_match_state(dfa.next_eoi_state(next_state));
#: if is_intermediate_state || is_full_match_state { ...record the edge... }
#: ```
#:
#: `regex-automata` reports matches one byte late: `is_match_state(s)` is true
#: of a state entered by reading a byte *after* a match ended. So a token that
#: leaves an accepting state and lands somewhere that is not itself accepting
#: satisfies neither disjunct and **the edge is thrown away** — while the
#: destination is still pushed onto the BFS frontier, so it appears in the table
#: as a source with no incoming edge and nothing looks obviously broken.
#:
#: Consequences measured here: `1(?:x2)*` rejected `1x2x2`; `a(?:ba)*` rejected
#: `ababa`; `countdown_regex(max_steps=4)` admitted exactly one step, so every
#: Countdown arm was scored against a grammar that could not contain a
#: multi-step answer (`cs_rate=1.000, solve_rate=0.000`). Character repetition
#: (`x*`, `[0-9]{0,3}`, `[^"]*`) is unaffected, because the state after the
#: repeated character is accepting too — which is why this survived for weeks.
#: It is also the cause of the lost closing fence recorded in
#: `tests/test_pipeline_flags.py`.
#:
#: With `\z` the pattern can only match at end-of-haystack, so no state reached
#: by reading a byte is ever `is_match_state` and `is_intermediate_state` is
#: universally true: every edge is recorded. Acceptance is unchanged, because
#: the finality `outlines_core` reports is already `is_match_state(next_eoi_state
#: (s))` — "would match if the input ended here" — which is exactly what `\z`
#: asks.
#:
#: **Cost: `|S|` generally GROWS, and that is the fix working.** The dropped
#: edges were the only thing making the mid-repetition states unreachable, so
#: the minimizer was collapsing a language the grammar was supposed to have.
#: Restoring them restores the states that distinguish it. Measured, post-
#: `compile_automaton`: `1(?:x2)*` 3 -> 4, `(?:ab|a)(?:b|ba)*` 3 -> 4,
#: `[1-4]{2}(?:\n[1-4]{2}){1,2}` 7 -> 10, `countdown_regex(max_steps=4,
#: max_value=999)` **40 -> 124** (bucket 128). Grammars with no repeated group
#: are untouched: `ab*`, `a(?:bc)*d`, `(?:ab)*c` are unchanged, and 15/15 BFCL
#: `live_simple` schemas compile to an identical `|S|`, so no BFCL result moves.
#:
#: (An earlier revision of this comment claimed `|S|` was "equal or smaller,
#: since the flagged and unflagged copies of a state merge (`ab*`: 3 -> 2)".
#: That was a raw `outlines_core.Index` state count for one regex with no
#: repeated group, generalised without checking; it is false of the compiled
#: automaton and false in direction.)
_EOI_ANCHOR = r"\z"


def anchor_at_eoi(regex: str) -> str:
    """Wrap `regex` so it can only match at end-of-haystack. See `_EOI_ANCHOR`.

    Non-capturing so that top-level alternation binds correctly: `a|b` must
    become `(?:a|b)\\z`, never `a|b\\z`.
    """
    return f"(?:{regex}){_EOI_ANCHOR}"


@dataclasses.dataclass(frozen=True)
class LiftResult:
    """A lifted (and optionally minimized) token DFA, with the measurements
    Phase 1 is required to report."""

    dfa: Dfa
    #: raw regex-automata id -> dense id, before minimization.
    raw_state_ids: tuple[int, ...]
    seconds_index: float
    seconds_transitions: float
    seconds_minimize: float
    states_raw: int
    states_minimized: int
    transitions_raw: int
    transitions_minimized: int

    @property
    def minimization_state_ratio(self) -> float:
        """SPEC §4.5: the post-lift reduction ratio. Nobody in the literature
        does this second pass, so this number is unpublished."""
        return self.states_raw / max(1, self.states_minimized)

    @property
    def minimization_transition_ratio(self) -> float:
        return self.transitions_raw / max(1, self.transitions_minimized)


def lift_regex(
    regex: str,
    vocabulary: Any,
    *,
    do_minimize: bool = True,
    drop_labels: frozenset[int] | None = None,
) -> LiftResult:
    """Compile `regex` against `vocabulary` and return a dense token DFA.

    Args:
      regex: an anchored byte-level regex (from `schema.build_regex`).
      vocabulary: an `outlines_core.Vocabulary` (from `vocab.build_vocabulary`).
      do_minimize: run the Valmari pass. SPEC §4.5 recommends minimizing twice
        — once on the byte DFA (inside outlines) and once here, on the token
        DFA. The lift is a non-injective relabeling with path compression: the
        byte DFA needs a state per byte inside every literal, the token DFA only
        at token boundaries, and `outlines_core` prunes dead states but never
        merges equivalent ones.
      drop_labels: token ids to strip from the lifted table. **Defaults to
        `vocab.RESERVED_TOKENS`, and this matters.** `outlines_core` injects
        `T[final][eos] = final` on its own — SPEC §4.3 records that
        `guide.advance(eos)` raises "even though `T[final][eos]` exists". Since
        stop tokens are handled out of band by
        `automaton.augment_with_stop_tokens`, leaving that edge in place gives
        the grammar-final state two destinations on the EOS label, which makes
        the augmented automaton **spuriously nondeterministic** and silently
        drops eq (8) onto its slow multiplicity-weighted path (SPEC §2.6).

    Returns:
      A `LiftResult`.
    """
    import outlines_core as oc

    if drop_labels is None:
        from diffgemma_fa.compile.vocab import RESERVED_TOKENS

        drop_labels = RESERVED_TOKENS

    t0 = time.perf_counter()
    index = oc.Index(anchor_at_eoi(regex), vocabulary)
    seconds_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    transitions = index.get_transitions()  # deep-clones; call exactly once
    seconds_transitions = time.perf_counter() - t0

    if drop_labels:
        transitions = {
            src: {tok: dst for tok, dst in row.items() if int(tok) not in drop_labels}
            for src, row in transitions.items()
        }

    initial = index.get_initial_state()
    finals = set(index.get_final_states())

    # --- renumber ---------------------------------------------------------
    # Include every id that appears as a source OR a destination OR is the
    # initial/final marker: a final state may have no outgoing transitions and
    # so never appear as a key.
    raw_ids = set(transitions) | {initial} | finals
    for row in transitions.values():
        raw_ids.update(row.values())
    ordered = sorted(raw_ids)
    dense = {raw: i for i, raw in enumerate(ordered)}

    triples = tuple(
        (dense[src], int(tok), dense[dst])
        for src, row in transitions.items()
        for tok, dst in row.items()
    )

    raw_dfa = Dfa(
        n_states=len(ordered),
        transitions=triples,
        start=dense[initial],
        finals=frozenset(dense[f] for f in finals),
    )

    states_raw = raw_dfa.n_states
    transitions_raw = raw_dfa.n_transitions

    if not do_minimize:
        return LiftResult(
            dfa=raw_dfa, raw_state_ids=tuple(ordered),
            seconds_index=seconds_index, seconds_transitions=seconds_transitions,
            seconds_minimize=0.0,
            states_raw=states_raw, states_minimized=states_raw,
            transitions_raw=transitions_raw, transitions_minimized=transitions_raw,
        )

    t0 = time.perf_counter()
    result = minimize(raw_dfa)
    seconds_minimize = time.perf_counter() - t0

    return LiftResult(
        dfa=result.dfa,
        raw_state_ids=tuple(ordered),
        seconds_index=seconds_index,
        seconds_transitions=seconds_transitions,
        seconds_minimize=seconds_minimize,
        states_raw=states_raw,
        states_minimized=result.states_after,
        transitions_raw=transitions_raw,
        transitions_minimized=result.transitions_after,
    )
