"""The compiled automaton artifact. SPEC §3.1b, §3.5, §3.6, §4.4, §5.5.

Turns a minimized token DFA into the arrays the JAX side consumes, applying the
three constructions the guarantee depends on:

1. **Stop-token augmentation with an UNSCORED tail** (§3.5 trap 4). From every
   grammar-final state, each of `end_tokens` leads to a single accepting state
   `ACC`, and `ACC --Σ--> ACC` over the *whole vocabulary*.

   The tail must be Σ, not PAD. A joint decode scores all 256 positions, so if
   the tail were pinned to PAD, terminating at position `j` would cost
   `log p(EOS) + (255−j)·log p(PAD)`. With `p(PAD) ≈ 1e-3` against a grammar
   continuation at `p ≈ 0.3`, stopping at `j=100` scores ≈ −1080 versus ≈ −188
   for carrying on: the MAP would place the stop token at position 255 or never.
   Labelling the tail Σ makes its emission mass `Σ_v p_i(v) = 1` — literally
   unscored — and it is semantically free because
   `_truncate_canvas_at_stop_tokens` overwrites everything after the first stop
   token anyway.

2. **Budget-aware `d(s)`** (§3.1b). `d(s) = ` min tokens from `s` to an
   accepting state, by BFS on the **reversed** automaton, computed **after**
   stop augmentation and **after** any refusal-branch union — otherwise it is a
   different function. `b_L(s) = 1[d(s) ≤ R]`, and `d(s) ≤ 0 ⟺ s ∈ F` makes the
   final block fall out automatically, so there is no last-block special case.

3. **Power-of-two `|S|` bucketing** (§5.5) with an absorbing dead state at
   `d = ∞`, so XLA compiles once per bucket rather than once per grammar.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Iterable, Sequence

import numpy as np

from diffgemma_fa.compile.classes import ClassTables, build_tables
from diffgemma_fa.compile.minimize import Dfa

__all__ = [
    "CompiledAutomaton",
    "INF_DISTANCE",
    "augment_with_stop_tokens",
    "distance_to_final",
    "compile_automaton",
    "bucket_size",
]

#: Sentinel for "cannot reach an accepting state at all". Deliberately a large
#: finite int32, never `inf` — it is compared against a traced budget on device
#: and must survive integer arithmetic.
INF_DISTANCE = np.int32(1 << 24)


def bucket_size(n: int, *, ladder: Sequence[int] | None = None) -> int:
    """Round `|S|` up to the next power-of-two bucket. SPEC §5.5.

    The ladder must be **derived from the memory budget**, not hardcoded: an
    unrolled tree over `[256, S, S]` costs `(2L−1)·S²·4 B`, so warming a bucket
    you can never dispatch to would try to compile and allocate a program that
    does not fit. Phase 0 measured ~20 GB free on this H100, which admits 2048
    (8.59 GB) but not 4096 (34.3 GB).
    """
    if ladder is None:
        ladder = (16, 32, 64, 128, 256, 512, 1024, 2048)
    for b in ladder:
        if n <= b:
            return b
    raise ValueError(
        f"|S| = {n} exceeds the largest usable bucket {ladder[-1]}; this grammar "
        "must go to the chain sampler (SPEC §5.6)"
    )


def augment_with_stop_tokens(
    dfa: Dfa,
    *,
    end_tokens: Iterable[int],
    vocab_size: int,
) -> tuple[Dfa, int, frozenset[int]]:
    """Add the `ACC` sink and the unscored Σ self-loop.

    Returns:
      `(augmented_dfa, acc_state, full_alphabet_label)`. `ACC` is the **only**
      accepting state afterwards: a string is in the language exactly when it
      has emitted a stop token from a grammar-final state.
    """
    end_tokens = tuple(sorted(set(int(t) for t in end_tokens)))
    if not end_tokens:
        raise ValueError(
            "end_tokens is empty; SPEC §3.5 trap 3 — handle all of "
            "(EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE, *stop_tokens) or the "
            "state set goes empty at a block boundary"
        )
    if dfa.is_empty:
        raise ValueError("cannot augment an empty automaton (SPEC: a real bug)")

    acc = dfa.n_states
    trans = list(dfa.transitions)

    # grammar-final --end_token--> ACC
    for f in sorted(dfa.finals):
        for t in end_tokens:
            trans.append((f, t, acc))

    # ACC --Σ--> ACC, over the FULL vocabulary. This is the unscored tail.
    for v in range(vocab_size):
        trans.append((acc, v, acc))

    aug = Dfa(
        n_states=acc + 1,
        transitions=tuple(trans),
        start=dfa.start,
        finals=frozenset({acc}),
    )
    return aug, acc, frozenset(range(vocab_size))


def distance_to_final(
    n_states: int,
    edges: Sequence[tuple[int, int]],
    finals: Iterable[int],
) -> np.ndarray:
    """`d(s)` = min number of tokens from `s` to an accepting state.

    A reverse BFS: every edge costs exactly one token regardless of how many
    labels it carries, so this is unweighted.

    Args:
      n_states: state count.
      edges: `(src, dst)` pairs (labels are irrelevant to the distance).
      finals: accepting states.

    Returns:
      `[n_states] int32`, with `INF_DISTANCE` where `F` is unreachable.
    """
    rev: list[list[int]] = [[] for _ in range(n_states)]
    for src, dst in edges:
        rev[dst].append(src)

    d = np.full(n_states, INF_DISTANCE, dtype=np.int32)
    frontier = [f for f in finals]
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
    return d


@dataclasses.dataclass(frozen=True)
class CompiledAutomaton:
    """Everything the inference side needs, as fixed-shape arrays.

    Shapes/dtypes:
      edge_src, edge_dst   [E]        int32
      edge_class           [E]        int32   index into `tables`
      d                    [S_bucket] int32   min tokens to F; INF_DISTANCE if dead
      is_final             [S_bucket] bool
      start_vector         [S_bucket] bool    `A_0`; a VECTOR, not a point mass (§5.7)
    """

    name: str
    n_states: int          # real states, before bucketing
    n_states_bucket: int   # padded power of two
    n_edges: int
    vocab_size: int
    is_dfa: bool

    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_class: np.ndarray

    d: np.ndarray
    is_final: np.ndarray
    start_vector: np.ndarray

    tables: ClassTables
    schema_hash: str
    tokenizer_hash: str
    compiler_version: str

    @property
    def dead_state(self) -> int:
        """The absorbing pad state. Every bucketed index >= n_states is dead."""
        return self.n_states

    @property
    def tree_bytes(self) -> int:
        """`(2L−1)·S²·4 B` at L=256 — SPEC §5.6's dispatch quantity.

        Do not drop the factor 2: it flips the decision.
        """
        return (2 * 256 - 1) * self.n_states_bucket ** 2 * 4

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_states": self.n_states,
            "n_states_bucket": self.n_states_bucket,
            "n_edges": self.n_edges,
            "n_classes": self.tables.n_classes,
            "dedup_ratio": round(self.tables.dedup_ratio, 2),
            "nnz_sum": self.tables.nnz_sum,
            "nnz_max": self.tables.nnz_max,
            "k_max": self.tables.k_max,
            "max_neg_size": self.tables.max_neg_size,
            "tables_mb": round(self.tables.nbytes / 1e6, 3),
            "tree_gb": round(self.tree_bytes / 1e9, 3),
            "is_dfa": self.is_dfa,
            "max_finite_d": int(self.d[self.d < INF_DISTANCE].max())
            if (self.d < INF_DISTANCE).any() else None,
            "n_dead_states": int((self.d >= INF_DISTANCE).sum()),
        }


def _group_edges(
    dfa: Dfa,
) -> tuple[list[tuple[int, int]], list[frozenset[int]], bool]:
    """Collapse transitions into `(src, dst)` edges carrying label *sets*.

    `M_i(s, s') = Σ_{e : src(e)=s, dst(e)=s'} W[i, e]`, so for a DFA — where the
    labels out of a state are disjoint — one edge per `(src, dst)` pair with the
    union of its labels is exact. Returns `is_dfa` too, because eq (8)'s cheap
    `∃` token draw is only valid when it holds (SPEC §2.6).
    """
    grouped: dict[tuple[int, int], set[int]] = {}
    seen: dict[tuple[int, int], int] = {}
    is_dfa = True
    for src, lbl, dst in dfa.transitions:
        key = (src, lbl)
        if key in seen and seen[key] != dst:
            is_dfa = False
        seen[key] = dst
        grouped.setdefault((src, dst), set()).add(lbl)

    pairs = sorted(grouped)
    labels = [frozenset(grouped[p]) for p in pairs]
    return pairs, labels, is_dfa


def compile_automaton(
    dfa: Dfa,
    *,
    name: str,
    end_tokens: Iterable[int],
    vocab_size: int,
    schema_hash: str = "",
    tokenizer_hash: str = "",
    compiler_version: str = "phase1",
    ladder: Sequence[int] | None = None,
    k_max: int | None = None,
) -> CompiledAutomaton:
    """Augment, measure `d`, intern labels, bucket. The whole back half of §4.

    Order matters: `d` is computed **after** stop augmentation, exactly as
    §3.1b requires. Any refusal-branch union (§3.8) must therefore be folded
    into `dfa` *before* this call.
    """
    aug, acc, _ = augment_with_stop_tokens(
        dfa, end_tokens=end_tokens, vocab_size=vocab_size
    )
    pairs, labels, is_dfa = _group_edges(aug)

    kwargs = {} if k_max is None else {"k_max": k_max}
    tables = build_tables(labels, vocab_size=vocab_size, **kwargs)

    d_real = distance_to_final(aug.n_states, pairs, aug.finals)

    n = aug.n_states
    bucket = bucket_size(n + 1, ladder=ladder)  # +1 for the dead state

    d = np.full(bucket, INF_DISTANCE, dtype=np.int32)
    d[:n] = d_real
    is_final = np.zeros(bucket, dtype=bool)
    is_final[sorted(aug.finals)] = True
    start_vector = np.zeros(bucket, dtype=bool)
    start_vector[aug.start] = True

    return CompiledAutomaton(
        name=name,
        n_states=n,
        n_states_bucket=bucket,
        n_edges=len(pairs),
        vocab_size=vocab_size,
        is_dfa=is_dfa,
        edge_src=np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs)),
        edge_dst=np.fromiter((p[1] for p in pairs), dtype=np.int32, count=len(pairs)),
        edge_class=tables.class_id,
        d=d,
        is_final=is_final,
        start_vector=start_vector,
        tables=tables,
        schema_hash=schema_hash,
        tokenizer_hash=tokenizer_hash,
        compiler_version=compiler_version,
    )


def schema_fingerprint(schema: dict) -> str:
    """Stable hash for the compilation cache key. SPEC §4.7(3).

    Cache on `(schema_hash, tokenizer_hash, compiler_version)`. Do **not** reuse
    `~/.cache/outlines`: its key ignores the tokenizer.
    """
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
