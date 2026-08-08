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

   **That "literally unscored" argument is sum-product-specific.** Under
   max-plus (`--emission=map`, SPEC §2.7) the tail contributes
   `log max_v p_i(v)` per position, not 0. The construction is still correct in
   both semirings — the tail is the cheapest available continuation either way,
   which is the property that matters — but only the sum-product path gets it
   for free. Worth remembering before reusing the identity elsewhere: it is
   also why `require_x64` exists, since a node max pinned at exactly 1.0 makes
   per-node normalization a no-op in the sampling path.

2. **Budget-aware `d(s)`** (§3.1b). `d(s) = ` min tokens from `s` to an
   accepting state, by BFS on the **reversed** automaton, computed **after**
   stop augmentation and **after** any refusal-branch union — otherwise it is a
   different function. `b_L(s) = 1[d(s) ≤ R]`, and `d(s) ≤ 0 ⟺ s ∈ F` makes the
   final block fall out automatically, so there is no last-block special case.

3. **Power-of-two `|S|` bucketing** (§5.5) with an absorbing dead state at
   `d = ∞`, so XLA compiles once per bucket rather than once per grammar.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import itertools
import json
from typing import Iterable, Sequence

import numpy as np

from diffgemma_fa.compile.classes import ClassTables, build_tables
from diffgemma_fa.compile.minimize import Dfa

__all__ = [
    "CompiledAutomaton",
    "INF_DISTANCE",
    "prepend_channel_header",
    "augment_with_stop_tokens",
    "distance_to_final",
    "compile_automaton",
    "bucket_size",
]

#: Sentinel for "cannot reach an accepting state at all". Deliberately a large
#: finite int32, never `inf` — it is compared against a traced budget on device
#: and must survive integer arithmetic.
INF_DISTANCE = np.int32(1 << 24)


#: Powers of two up to the largest the measured HBM headroom admits.
#: Phase 0 measured ~20 GB free on this H100: 2048 needs 8.59 GB, 4096 needs
#: 34.3 GB. Derive this from the budget, never hardcode past it — AOT-warming a
#: bucket you can never dispatch to tries to compile and allocate that program.
DEFAULT_LADDER: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048)


def bucket_size(
    n: int,
    *,
    ladder: Sequence[int] | None = None,
    allow_oversize: bool = False,
) -> int:
    """Round `|S|` up to the next power-of-two bucket. SPEC §5.5.

    Args:
      n: state count including the dead padding state.
      ladder: usable buckets; defaults to `DEFAULT_LADDER`.
      allow_oversize: return the next power of two beyond the ladder instead of
        raising. The caller must then route the grammar to the chain sampler
        (SPEC §5.6) — measured on BFCL-Live, the largest raw lifted automaton
        has 3,573 states, above this ladder and above the 2,459 the paper
        quotes, so this case is real and must not simply crash the run.

    Raises:
      ValueError: when `n` exceeds the ladder and `allow_oversize` is False.
    """
    ladder = DEFAULT_LADDER if ladder is None else ladder
    for b in ladder:
        if n <= b:
            return b
    if allow_oversize:
        b = ladder[-1]
        while b < n:
            b *= 2
        return b
    raise ValueError(
        f"|S| = {n} exceeds the largest usable bucket {ladder[-1]}; this grammar "
        "must go to the chain sampler (SPEC §5.6)"
    )


def prepend_channel_header(
    dfa: Dfa,
    *,
    vocab_size: int,
    open_token: int = 100,
    close_token: int = 101,
    newline_token: int = 107,
    reserved: Iterable[int] = (),
    max_name_tokens: int = 8,
) -> Dfa:
    """Prefix the grammar with SPEC §3.6's channel header.

        FA_total = HEADER · FA_grammar · STOP · Σ*
        HEADER   = 100 · Σ_name+ · 107 · 101      (`<|channel>NAME\n<channel|>`)

    **Why this is load-bearing rather than cosmetic.** Phase 0 measured that
    *every* generation from the released model opens with exactly
    `[100, 45518, 107, 101]`. Without the header the grammar admits only `{` at
    canvas position 0 — measured: 3 tokens, all variants of `{`, with token 100
    forbidden — so the very first thing the model wants to emit is impossible,
    and the whole canvas is decoded from an off-distribution prefix.

    The channel *name* is left open (`Σ_name{1,max_name_tokens}`): only
    `thought` was observed in Phase 0, but `final`/`answer` tokenise fine and
    hardcoding one would be fragile. The bound is **not** optional — see the
    comment on the name states below.

    Args:
      dfa: the lifted grammar, whose start becomes the header's continuation.
      reserved: tokens that may not appear inside the name — the end tokens and
        PAD, so the header cannot itself terminate the canvas.

    Returns:
      A new `Dfa` with `max_name_tokens + 2` states prepended; existing state
      ids shift by that amount.
    """
    if max_name_tokens < 1:
        raise ValueError("max_name_tokens must be >= 1")

    name_alphabet = sorted(
        frozenset(range(vocab_size))
        - {open_token, close_token, newline_token, *reserved}
    )

    # Layout: h_open, then one state per accepted name token, then h_close.
    #
    # **The name repetition must be BOUNDED, never a Σ* self-loop.** A self-loop
    # over the whole vocabulary has emission mass ~1.0 — exactly like the
    # unscored `ACC --Σ--> ACC` tail — so staying in it is free, and a joint MAP
    # will sit there for the entire canvas rather than pay a specific token's
    # probability to leave. Measured with a self-loop: the MAP consumed all 64
    # positions inside the name loop and never closed the header, leaving a
    # viable-but-not-accepting prefix and a rejected emission.
    # `consumed[i]` means "i name tokens have been read". The newline may close
    # the name from `consumed[1..n]` but **not** from `consumed[0]`, which is
    # what makes the repetition `Σ_name+` rather than `Σ_name*`.
    n_name = max_name_tokens
    h_open = 0
    consumed = list(range(1, 2 + n_name))          # consumed[0] .. consumed[n]
    h_close = consumed[-1] + 1
    shift = h_close + 1

    trans: list[tuple[int, int, int]] = [(h_open, open_token, consumed[0])]
    for i in range(n_name):
        trans.extend((consumed[i], v, consumed[i + 1]) for v in name_alphabet)
    for i in range(1, n_name + 1):
        trans.append((consumed[i], newline_token, h_close))
    trans.append((h_close, close_token, dfa.start + shift))

    trans.extend((src + shift, lbl, dst + shift) for src, lbl, dst in dfa.transitions)

    return Dfa(
        n_states=dfa.n_states + shift,
        transitions=tuple(trans),
        start=h_open,
        finals=frozenset(f + shift for f in dfa.finals),
    )


def wrap_with_fence(
    dfa: Dfa,
    *,
    open_tokens: Sequence[int] = (2717, 3723, 107),   # "```", "json", "\n"
    close_tokens: Sequence[int] = (107, 2717),        # "\n", "```"
) -> Dfa:
    """Wrap the grammar in an **optional** markdown fence, as DFA surgery.

        FA = (```json\n)? · FA_grammar · (\n```)?

    **Why surgery and not a regex wrap.** Prepending
    `r"(```json\n)?" + regex + r"(\n```)?"` looks equivalent and is correct
    under `re.fullmatch`, but the closing branch is **lost in the lift**:
    `lift_regex` walks the opening fence fine and then dies on the `\n` after
    the object (measured on a 2-key schema: 37 lifted states, two finals, and
    no outgoing newline edge from the grammar-final state). Whatever the cause
    inside `outlines_core`'s index construction, the resulting automaton
    accepts only the compact form — so the flag would have looked like it
    worked while measuring nothing. `prepend_channel_header` already
    establishes that literal token prefixes belong here rather than in the
    regex; this is the same argument for a suffix.

    **Why it is worth having.** 126 of 130 unconstrained emissions are fenced.
    With no slot for the fence the model writes it into the channel-header
    name instead (82/130 junk names, all variants of `` ```json\n{ ``) and its
    whole canvas plan is shifted. Whitespace tolerance **alone** was measured
    to accept 0/130 of the model's own outputs; whitespace plus fence accepts
    74/130. Neither does anything without the other.

    Optional in both directions: the fence is a habit, not a guarantee, so a
    model that skips it must still be able to emit. Both the fenced and bare
    renderings therefore end accepting.

    Args:
      dfa: the lifted grammar. Its finals gain a path through the closing
        fence; its start gains an optional prefix.

    Returns:
      A new `Dfa`. Existing state ids shift by `len(open_tokens)`.
    """
    shift = len(open_tokens)
    trans = [(src + shift, lbl, dst + shift) for src, lbl, dst in dfa.transitions]

    # Opening fence: a chain 0 -> 1 -> ... -> len(open)-1 -> grammar start.
    for i, t in enumerate(open_tokens):
        nxt = (i + 1) if i + 1 < shift else dfa.start + shift
        trans.append((i, t, nxt))

    # Closing fence: a fresh chain hanging off EVERY grammar-final state.
    base = dfa.n_states + shift
    close_states = list(range(base, base + len(close_tokens)))
    for f in dfa.finals:
        trans.append((f + shift, close_tokens[0], close_states[0]))
    for i, t in enumerate(close_tokens[1:], start=1):
        trans.append((close_states[i - 1], t, close_states[i]))

    return Dfa(
        n_states=base + len(close_tokens),
        # The bare start must remain reachable, so the grammar start is the
        # automaton start and the fence chain is entered from it -- no: an
        # optional PREFIX needs the fence chain to be the start, with the
        # grammar reachable from state 0 directly. Both are wired below.
        transitions=tuple(trans + [
            (0, lbl, dst + shift)
            for src, lbl, dst in dfa.transitions if src == dfa.start
        ]),
        start=0,
        finals=frozenset([f + shift for f in dfa.finals] + [close_states[-1]]),
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

    #: True when `|S|` exceeded the usable bucket ladder. Such a grammar cannot
    #: use the tree sampler and must be routed to the chain path (SPEC §5.6).
    #: Log which path each request took and report the split.
    needs_chain_path: bool = False

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
            "needs_chain_path": self.needs_chain_path,
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
    allow_oversize: bool = True,
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
    bucket = bucket_size(n + 1, ladder=ladder, allow_oversize=allow_oversize)
    usable_top = (DEFAULT_LADDER if ladder is None else ladder)[-1]
    needs_chain_path = bucket > usable_top

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
        needs_chain_path=needs_chain_path,
    )


def save(automaton: CompiledAutomaton, path: str) -> None:
    """Serialize to a single `.npz`.

    The cache key is `(schema_hash, tokenizer_hash, compiler_version)` and it is
    stored alongside the arrays. **Do not reuse `~/.cache/outlines`** — its key
    ignores the tokenizer, so a vocabulary change silently returns a stale
    automaton (SPEC §4.7(3)).
    """
    t = automaton.tables
    np.savez_compressed(
        path,
        meta=np.frombuffer(
            json.dumps({
                "name": automaton.name,
                "n_states": automaton.n_states,
                "n_states_bucket": automaton.n_states_bucket,
                "n_edges": automaton.n_edges,
                "vocab_size": automaton.vocab_size,
                "is_dfa": automaton.is_dfa,
                "n_classes": t.n_classes,
                "k_max": t.k_max,
                "max_neg_size": t.max_neg_size,
                "schema_hash": automaton.schema_hash,
                "tokenizer_hash": automaton.tokenizer_hash,
                "compiler_version": automaton.compiler_version,
                "needs_chain_path": automaton.needs_chain_path,
            }).encode(),
            dtype=np.uint8,
        ),
        edge_src=automaton.edge_src, edge_dst=automaton.edge_dst,
        edge_class=automaton.edge_class, d=automaton.d,
        is_final=automaton.is_final, start_vector=automaton.start_vector,
        class_size=t.class_size,
        sum_is_neg=t.sum_is_neg, sum_indptr=t.sum_indptr, sum_indices=t.sum_indices,
        max_is_neg=t.max_is_neg, max_indptr=t.max_indptr, max_indices=t.max_indices,
        class_id=t.class_id,
    )


def load(path: str) -> CompiledAutomaton:
    """Inverse of `save`, with the structural invariants re-checked.

    `_group_edges` guarantees **one edge per `(src, dst)`** carrying the union
    of its labels, and eq (8)'s multiplicity is 0/1 as a consequence. That
    guarantee holds at build time and was then never re-established: a hand-made
    or stale `.npz` with a duplicated pair loaded silently, and the damage would
    surface as a wrong *distribution* — valid strings drawn with the wrong
    probabilities — which no acceptance check can see.

    So the invariant is proved on the way in, before anything consumes it.
    """
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta"]).decode())
    tables = ClassTables(
        n_classes=meta["n_classes"], n_edges=meta["n_edges"],
        vocab_size=meta["vocab_size"], k_max=meta["k_max"],
        class_id=z["class_id"], class_size=z["class_size"],
        sum_is_neg=z["sum_is_neg"], sum_indptr=z["sum_indptr"],
        sum_indices=z["sum_indices"],
        max_is_neg=z["max_is_neg"], max_indptr=z["max_indptr"],
        max_indices=z["max_indices"],
        max_neg_size=meta["max_neg_size"],
    )
    a = CompiledAutomaton(
        name=meta["name"], n_states=meta["n_states"],
        n_states_bucket=meta["n_states_bucket"], n_edges=meta["n_edges"],
        vocab_size=meta["vocab_size"], is_dfa=meta["is_dfa"],
        edge_src=z["edge_src"], edge_dst=z["edge_dst"], edge_class=z["edge_class"],
        d=z["d"], is_final=z["is_final"], start_vector=z["start_vector"],
        tables=tables, schema_hash=meta["schema_hash"],
        tokenizer_hash=meta["tokenizer_hash"],
        compiler_version=meta["compiler_version"],
        needs_chain_path=meta.get("needs_chain_path", False),
    )
    _assert_structural_invariants(a, path)
    return a


def _assert_structural_invariants(a: CompiledAutomaton, path: str = "") -> None:
    """Unit edge multiplicity per `(src, dst)`, and `is_dfa` as stored.

    Raises:
      ValueError: on a duplicated state pair, or on an `is_dfa` flag that
        disagrees with the transition data.
    """
    where = f" in {path}" if path else ""
    pairs = np.stack([np.asarray(a.edge_src), np.asarray(a.edge_dst)], axis=1)
    uniq = np.unique(pairs, axis=0)
    if uniq.shape[0] != pairs.shape[0]:
        raise ValueError(
            f"{pairs.shape[0] - uniq.shape[0]} duplicated (src, dst) pair(s)"
            f"{where}. `_group_edges` emits one edge per pair carrying the "
            "union of its labels, so eq (8)'s multiplicity is 0/1 by "
            "construction; a duplicate silently changes the sampling "
            "DISTRIBUTION, which no acceptance check can detect."
        )
    # `is_dfa` gates eq (8)'s cheap `∃` token draw (SPEC §2.6), so a flag that
    # over-claims is a correctness bug, not a performance one.
    if a.is_dfa:
        by_src_class: dict[tuple[int, int], int] = {}
        for e in range(len(a.edge_src)):
            key = (int(a.edge_src[e]), int(a.edge_class[e]))
            prev = by_src_class.get(key)
            if prev is not None and prev != int(a.edge_dst[e]):
                raise ValueError(
                    f"is_dfa=True{where} but state {key[0]} has two "
                    f"destinations for class {key[1]}"
                )
            by_src_class[key] = int(a.edge_dst[e])
        _assert_outgoing_labels_are_disjoint(a, where)


def _class_tokens(a: CompiledAutomaton, c: int) -> tuple[np.ndarray, bool]:
    """The **stored** member ids of class `c` and whether they are its
    complement. `[nnz_c] int64`, sorted, plus the polarity flag."""
    t = a.tables
    lo, hi = int(t.sum_indptr[c]), int(t.sum_indptr[c + 1])
    return np.sort(np.asarray(t.sum_indices[lo:hi], dtype=np.int64)), \
        bool(t.sum_is_neg[c])


def _assert_outgoing_labels_are_disjoint(
    a: CompiledAutomaton, where: str = ""
) -> None:
    """Determinism, checked on **tokens** rather than on interned class ids.

    **[AUDIT-D4]** The `(src, edge_class)` check above can only see a conflict
    between two edges that interned to the *same* class. Two edges out of one
    state whose label sets **overlap but are not identical** get different class
    ids, so a stale or hand-made artifact claiming `is_dfa=True` used to load
    clean — silently licensing eq (8)'s `∃` fast path on an NFA, where it is off
    by ~1.7e-2 against the exact posterior (SPEC §2.6). `_group_edges` derives
    `is_dfa` from the raw token-level transitions, so a freshly compiled
    automaton always passes; this closes the gap on everything that arrives from
    disk.

    The stored side of every class is small — that is the whole point of the
    polarity rule in `classes.py` — so each comparison is over `nnz`, never over
    `V = 262,144`.
    """
    V = int(a.vocab_size)
    by_src: dict[int, list[int]] = collections.defaultdict(list)
    for e in range(len(a.edge_src)):
        by_src[int(a.edge_src[e])].append(e)

    def fail(e1: int, e2: int, n: int) -> None:
        raise ValueError(
            f"is_dfa=True{where} but state {int(a.edge_src[e1])} sends {n} "
            f"token(s) to both {int(a.edge_dst[e1])} (class "
            f"{int(a.edge_class[e1])}) and {int(a.edge_dst[e2])} (class "
            f"{int(a.edge_class[e2])}). Their label sets OVERLAP without being "
            "identical, so the `(src, class)` check cannot see it; eq (8)'s "
            "`∃` token draw would be applied to an NFA."
        )

    for src, edges in by_src.items():
        if len(edges) < 2:
            continue
        pos: list[tuple[int, np.ndarray]] = []
        neg: list[tuple[int, np.ndarray]] = []
        for e in edges:
            idx, is_neg = _class_tokens(a, int(a.edge_class[e]))
            (neg if is_neg else pos).append((e, idx))

        # positive x positive: a token listed twice across the stored sets.
        if len(pos) > 1:
            stacked = np.concatenate([i for _, i in pos])
            if np.unique(stacked).size != stacked.size:
                for (e1, i1), (e2, i2) in itertools.combinations(pos, 2):
                    n = int(np.intersect1d(i1, i2, assume_unique=True).size)
                    if n:
                        fail(e1, e2, n)
        # positive x negative: `P ∩ ~N = P \ N`.
        for e1, p in pos:
            for e2, n_idx in neg:
                extra = int(p.size - np.isin(p, n_idx).sum())
                if extra:
                    fail(e1, e2, extra)
        # negative x negative: `~N1 ∩ ~N2 = ~(N1 ∪ N2)`.
        for (e1, n1), (e2, n2) in itertools.combinations(neg, 2):
            n = V - int(np.union1d(n1, n2).size)
            if n:
                fail(e1, e2, n)


def schema_fingerprint(schema: dict, **options) -> str:
    """Stable hash for the compilation cache key. SPEC §4.7(3).

    Cache on `(schema_hash, tokenizer_hash, compiler_version)`. Do **not** reuse
    `~/.cache/outlines`: its key ignores the tokenizer.

    **`options` is not optional in practice.** This used to hash the schema
    alone, which is the mistake SPEC §4.7(3) warns about one level down: the
    *same* schema compiles to materially different grammars under
    `whitespace_pattern`, `nonempty_required_strings`, `channel_header`,
    `allow` and `from_bfcl`. Two of those change the accepted language
    outright. The key collided for them, so a cache hit would silently return
    the wrong automaton — and since every returned automaton is internally
    consistent, nothing downstream could tell.

    Harmless while `tasks/bfcl.py` keys artifacts by filename, dangerous the
    moment the key is actually used, which is why it is fixed before that
    happens rather than after.
    """
    payload = {
        "schema": schema,
        # `sorted` on the items, not on a set: `allow` is an iterable of
        # keywords whose order must not affect the key, but whose *content*
        # must.
        "options": {k: (sorted(v) if isinstance(v, (set, frozenset, tuple, list))
                        else v)
                    for k, v in sorted(options.items())},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode()
    ).hexdigest()[:16]
