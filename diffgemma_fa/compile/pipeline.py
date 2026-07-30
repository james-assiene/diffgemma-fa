"""End-to-end compilation. SPEC §4.1.

```
grammar spec (JSON Schema | regex | task DSL)
      |  §4.2      ->  byte-level regex
      |  regex-automata dense::DFA (anchored, via outlines_core)
   byte DFA        ->  minimize (inside outlines, over ByteClasses)
      |  §4.3         token lift
   token DFA/NFA   ->  minimize (Valmari, §4.5)
      |  §4.4         label interning -> classes -> CSR x2 (sum and max)
   Automaton artifact (padded to a power-of-two |S| bucket)
```
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Sequence

from diffgemma_fa.compile import schema as schema_mod
from diffgemma_fa.compile import vocab as vocab_mod
from diffgemma_fa.compile.automaton import CompiledAutomaton, compile_automaton
from diffgemma_fa.compile.lift import lift_regex

__all__ = ["CompileReport", "compile_regex", "compile_json_schema"]


@dataclasses.dataclass(frozen=True)
class CompileReport:
    """The artifact plus the numbers Phase 1 must report."""

    automaton: CompiledAutomaton
    regex_len: int
    seconds_regex: float
    seconds_index: float
    seconds_transitions: float
    seconds_minimize: float
    seconds_classes: float
    states_raw: int
    states_minimized: int
    transitions_raw: int
    transitions_minimized: int

    @property
    def seconds_total(self) -> float:
        return (self.seconds_regex + self.seconds_index + self.seconds_transitions
                + self.seconds_minimize + self.seconds_classes)

    @property
    def minimization_state_ratio(self) -> float:
        """SPEC §4.5: unpublished — nobody does the post-lift pass."""
        return self.states_raw / max(1, self.states_minimized)

    def summary(self) -> dict:
        d = self.automaton.summary()
        d.update({
            "regex_len": self.regex_len,
            "seconds_total": round(self.seconds_total, 3),
            "seconds_regex": round(self.seconds_regex, 4),
            "seconds_index": round(self.seconds_index, 3),
            "seconds_transitions": round(self.seconds_transitions, 3),
            "seconds_minimize": round(self.seconds_minimize, 3),
            "seconds_classes": round(self.seconds_classes, 3),
            "states_raw": self.states_raw,
            "states_minimized": self.states_minimized,
            "minimization_state_ratio": round(self.minimization_state_ratio, 3),
            "transitions_raw": self.transitions_raw,
            "transitions_minimized": self.transitions_minimized,
        })
        return d


def compile_regex(
    regex: str,
    *,
    name: str,
    vocabulary: Any = None,
    end_tokens: Sequence[int] = vocab_mod.END_TOKENS,
    vocab_size: int | None = None,
    do_minimize: bool = True,
    ladder: Sequence[int] | None = None,
    k_max: int | None = None,
    schema_hash: str = "",
    channel_header: bool = True,
) -> CompileReport:
    """Compile an anchored byte-level regex into a `CompiledAutomaton`.

    `channel_header` prefixes SPEC §3.6's `<|channel>NAME\n<channel|>` header.
    **On by default**: Phase 0 measured that every generation from the released
    model opens with it, and without it the grammar forbids the model's first
    token outright.
    """
    vocabulary = vocabulary if vocabulary is not None else vocab_mod.build_vocabulary()
    if vocab_size is None:
        vocab_size = int(vocab_mod.gemma_tokenizer().vocab_size)

    lifted = lift_regex(regex, vocabulary, do_minimize=do_minimize)

    grammar = lifted.dfa
    if channel_header:
        from diffgemma_fa.compile.automaton import prepend_channel_header

        grammar = prepend_channel_header(
            grammar, vocab_size=vocab_size,
            reserved=tuple(end_tokens) + (vocab_mod.PAD_TOKEN,))

    t0 = time.perf_counter()
    automaton = compile_automaton(
        grammar,
        name=name,
        end_tokens=end_tokens,
        vocab_size=vocab_size,
        ladder=ladder,
        k_max=k_max,
        schema_hash=schema_hash,
    )
    seconds_classes = time.perf_counter() - t0

    return CompileReport(
        automaton=automaton,
        regex_len=len(regex),
        seconds_regex=0.0,
        seconds_index=lifted.seconds_index,
        seconds_transitions=lifted.seconds_transitions,
        seconds_minimize=lifted.seconds_minimize,
        seconds_classes=seconds_classes,
        states_raw=lifted.states_raw,
        states_minimized=lifted.states_minimized,
        transitions_raw=lifted.transitions_raw,
        transitions_minimized=lifted.transitions_minimized,
    )


def compile_json_schema(
    json_schema: dict,
    *,
    name: str,
    from_bfcl: bool = False,
    allow: Sequence[str] = (),
    allow_wildcard: bool = False,
    whitespace_pattern: str | None = None,
    nonempty_required_strings: bool = False,
    channel_header: bool = True,
    fence: bool = False,
    **kwargs: Any,
) -> CompileReport:
    """Compile a JSON Schema (or BFCL parameter block) end to end.

    `whitespace_pattern` defaults to outlines' own (optional whitespace).

    **[V-P5] It used to default to `""` — forbidding whitespace — on the
    reasoning that this "shrinks the automaton and costs nothing the benchmark
    scores". That reasoning was wrong.** Gemma's natural tokenisation of JSON
    puts the space *inside* the separator token (`": "`), so forbidding
    whitespace makes the model's own rendering
    `{"user_id": 7890, "special": "black"}` **unacceptable to the grammar**, and
    it is forced onto an off-distribution path. The measured symptom was a
    leading `:` on essentially every string value (`":Divinópolis, MG"` where
    the ground truth is `Divinópolis, MG`) — a one-character defect that BFCL's
    normalisation does not strip and that therefore failed every such argument.

    The cost of allowing whitespace is 8 states on a representative BFCL schema
    (43 -> 51), against ~20 GB of measured tree headroom. Not a trade worth
    making.

    `fence` (SPEC §3.6 extension, experiment E4/P2) wraps the object in an
    **optional** ```` ```json ```` … ```` ``` ```` markdown fence. 126 of 130
    unconstrained outputs are fenced, and with no slot for it the model writes
    the fence into the channel-header name instead (measured: 82/130 junk
    headers, all variants of ` ```json\n{ `) and its whole canvas plan is
    shifted. Measured cost: +8 states. Whitespace and fence together take
    verbatim acceptance of the unconstrained outputs from **0/130 to 74/130**;
    neither does anything alone.
    """
    t0 = time.perf_counter()
    prepared = json_schema
    if nonempty_required_strings:
        prepared = schema_mod.normalize_bfcl_schema(prepared) if from_bfcl else prepared
        prepared = schema_mod.require_nonempty_strings(prepared)
        from_bfcl = False
    regex = schema_mod.build_regex(
        prepared, from_bfcl=from_bfcl, allow=allow,
        allow_wildcard=allow_wildcard, whitespace_pattern=whitespace_pattern,
    )
    if fence:
        # Wrapped at the REGEX level, i.e. before `lift_regex` and therefore
        # before `augment_with_stop_tokens` and `distance_to_final` -- `d(s)`
        # must see the fence or it is a different function (SPEC §3.5).
        regex = r"(```json\n)?" + regex + r"(\n```)?"
    seconds_regex = time.perf_counter() - t0

    from diffgemma_fa.compile.automaton import schema_fingerprint

    report = compile_regex(
        regex, name=name, channel_header=channel_header,
        # Every option that can change the compiled language goes into the key.
        # Hashing the schema alone made two materially different grammars share
        # a cache entry -- see `schema_fingerprint`.
        schema_hash=schema_fingerprint(
            json_schema, whitespace_pattern=whitespace_pattern,
            nonempty_required_strings=nonempty_required_strings,
            channel_header=channel_header, allow=allow,
            allow_wildcard=allow_wildcard, from_bfcl=from_bfcl, fence=fence),
        **kwargs)
    return dataclasses.replace(report, seconds_regex=seconds_regex)
