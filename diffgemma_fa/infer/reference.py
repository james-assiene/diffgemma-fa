"""Reference implementation. Plain float64 numpy, no cleverness. SPEC §2, §6.2.

**Every optimized path is differential-tested against this file.** If the fast
path and the reference disagree, the fast path is wrong. So this module
deliberately favours transparency over speed: explicit loops, dense matrices,
no fused kernels, no scaling tricks except where SPEC requires them and then
written out longhand.

Notation follows SPEC §2. `L` positions, `V` vocabulary, `S` states, `E` edges.
Latents are **edges, not states** (SPEC §2.2, the paper's Eq. 3) — that is what
makes the emission factor local, and it is why an NFA's parallel edges carry
multiplicity through the whole derivation.

Indexing: §2.4 is 1-indexed and §2.6 is 0-indexed. **This module is 0-indexed
throughout** — positions `0..L-1`, boundary states `s_0..s_L` where `s_i` is the
state *before* position `i`.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Automaton",
    "EmptyLanguageError",
    "edge_weights",
    "transition_matrices",
    "label_count_matrix",
    "forward_backward",
    "ForwardBackward",
    "marginals",
    "marginals_complement_aware",
    "sample_chain",
    "sample_tree",
    "map_decode",
    "enumerate_posterior",
    "accepting_path_count",
    "block_products",
]


class EmptyLanguageError(ValueError):
    """`Z == 0`.

    SPEC is explicit that this has **three distinct causes** and that two of
    them are bugs:

      (a) the automaton is genuinely empty — a real bug;
      (b) no live continuation of `L` tokens from `A_k` within budget — a
          grammar/budget condition (§3.1b/§3.5);
      (c) fp32 underflow on an unnormalized path — expected, and §6.3
          deliberately tests for it.

    Raise on (a) and (b). (c) means the scaling is missing, so it must never
    reach here from the scaled path.
    """


@dataclasses.dataclass(frozen=True)
class Automaton:
    """A small automaton, in the edge-centric form the method needs.

    Attributes:
      n_states: `S`.
      vocab_size: `V`.
      edges: `(src, dst, labels)` triples. **Parallel edges between the same
        state pair are permitted and meaningful** — on an NFA they carry the
        path multiplicity that eq (8) must respect.
      start: `[S] float64`. A **vector**, not a point mass: `1[s = s_0]` only
        holds for a DFA on block 0 (SPEC §5.7).
      finals: accepting states, used to build a default `b_L`.
    """

    n_states: int
    vocab_size: int
    edges: tuple[tuple[int, int, frozenset[int]], ...]
    start: np.ndarray
    finals: frozenset[int]

    def __post_init__(self) -> None:
        if self.start.shape != (self.n_states,):
            raise ValueError(f"start must be [{self.n_states}], got {self.start.shape}")
        for src, dst, labels in self.edges:
            if not (0 <= src < self.n_states and 0 <= dst < self.n_states):
                raise ValueError(f"edge out of range: ({src}, {dst})")
            for v in labels:
                if not 0 <= v < self.vocab_size:
                    raise ValueError(f"label {v} outside vocab")

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def is_deterministic(self) -> bool:
        """Classic DFA property: at most one destination per `(state, token)`."""
        seen: dict[tuple[int, int], int] = {}
        for src, dst, labels in self.edges:
            for v in labels:
                if seen.get((src, v), dst) != dst:
                    return False
                seen[(src, v)] = dst
        return True

    @property
    def has_unit_multiplicity(self) -> bool:
        """No `(src, dst, token)` is carried by more than one edge.

        **This, not `is_deterministic`, is the correct gate for eq (8)'s cheap
        `∃` token draw** — and the distinction is a real trap. SPEC §2.6 says
        "on a DFA the two coincide (one edge per `(s,s')`, disjoint labels), so
        gate the fast path on an `is_dfa` flag", and the parenthetical is doing
        all the work: two *parallel* edges to the **same** destination with
        overlapping labels are perfectly deterministic, yet give that token
        multiplicity 2. The `∃` form would then draw from the wrong
        distribution on an automaton every conventional check calls a DFA.

        Note that grouping transitions into one edge per `(src, dst)` carrying
        the union of their labels — which `compile/automaton.py` does — makes
        this hold **by construction**, so compiled artifacts are always safe
        for the fast path. The predicate matters for hand-built or ungrouped
        automata.
        """
        seen: set[tuple[int, int, int]] = set()
        for src, dst, labels in self.edges:
            for v in labels:
                key = (src, dst, v)
                if key in seen:
                    return False
                seen.add(key)
        return True

    @property
    def is_dfa(self) -> bool:
        """Deterministic **and** unit-multiplicity — safe for eq (8)'s `∃` form."""
        return self.is_deterministic and self.has_unit_multiplicity

    def final_vector(self) -> np.ndarray:
        b = np.zeros(self.n_states, dtype=np.float64)
        b[sorted(self.finals)] = 1.0
        return b

    def budget_vector(self, remaining: int) -> np.ndarray:
        """`b_L(s) = 1[d(s) ≤ R]`. SPEC §3.1b.

        Not `1[s ∈ F]`, and with **no final-block special case** —
        `d(s) ≤ 0 ⟺ s ∈ F` makes that fall out as `R` runs down.
        """
        d = self.distance_to_final()
        return (d <= remaining).astype(np.float64)

    def distance_to_final(self) -> np.ndarray:
        """Min tokens from each state to an accepting state, by reverse BFS."""
        inf = np.iinfo(np.int32).max
        d = np.full(self.n_states, inf, dtype=np.int64)
        rev: list[list[int]] = [[] for _ in range(self.n_states)]
        for src, dst, _ in self.edges:
            rev[dst].append(src)
        frontier = sorted(self.finals)
        for f in frontier:
            d[f] = 0
        depth = 0
        while frontier:
            depth += 1
            nxt = []
            for q in frontier:
                for p in rev[q]:
                    if d[p] == inf:
                        d[p] = depth
                        nxt.append(p)
            frontier = nxt
        return d


# ---------------------------------------------------------------------------
# §2.3 — the tractable product
# ---------------------------------------------------------------------------

def edge_weights(p: np.ndarray, automaton: Automaton) -> np.ndarray:
    """`W[i, e] = Σ_{v ∈ label(e)} p_i(v)`. SPEC eq (1).

    Args:
      p: `[L, V] float64` mean-field marginals.
      automaton: the automaton.

    Returns:
      `[L, E] float64`.
    """
    L = p.shape[0]
    W = np.zeros((L, automaton.n_edges), dtype=np.float64)
    for e, (_, _, labels) in enumerate(automaton.edges):
        if labels:
            idx = np.fromiter(sorted(labels), dtype=np.int64, count=len(labels))
            W[:, e] = p[:, idx].sum(axis=1)
    return W


def transition_matrices(W: np.ndarray, automaton: Automaton) -> np.ndarray:
    """`M_i(s, s') = Σ_{e : src(e)=s, dst(e)=s'} W[i, e]`. SPEC eq (2).

    The sum over **parallel edges** is what makes the state-path distribution
    path-weighted on an NFA.

    Returns:
      `[L, S, S] float64`.
    """
    L = W.shape[0]
    S = automaton.n_states
    M = np.zeros((L, S, S), dtype=np.float64)
    for e, (src, dst, _) in enumerate(automaton.edges):
        M[:, src, dst] += W[:, e]
    return M


def label_count_matrix(automaton: Automaton, token: int) -> np.ndarray:
    """`c(s, s') = #{e : src=s, dst=s', token ∈ label(e)}`.

    The multiplicity that distinguishes the path-weighted posterior from the
    `∃`-indicator one. `M_i = Σ_v p_i(v) · c(·, ·, v)`, which is the identity
    tying `enumerate_posterior` to `forward_backward`.
    """
    S = automaton.n_states
    C = np.zeros((S, S), dtype=np.float64)
    for src, dst, labels in automaton.edges:
        if token in labels:
            C[src, dst] += 1.0
    return C


# ---------------------------------------------------------------------------
# §2.4 — forward–backward
# ---------------------------------------------------------------------------

def _log(x: np.ndarray) -> np.ndarray:
    """Elementwise `log` with exact zeros mapped to `-inf`, no warning.

    `-inf` and not a finite sentinel: this module is plain numpy, there is no
    fused kernel to produce `NaN` from `-inf + -inf`, and every reduction below
    anchors on its own max so `-inf` never participates in an arithmetic that
    could produce `NaN`. (SPEC §2.7's finite-sentinel rule is about the JAX
    kernels; `scans.NEG_SENTINEL` exists for that reason and this does not.)
    """
    x = np.asarray(x, dtype=np.float64)
    pos = x > 0.0
    out = np.full(x.shape, -np.inf, dtype=np.float64)
    np.log(np.where(pos, x, 1.0), out=out, where=pos)
    return out


def _lse(t: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """`logsumexp` anchored on the reduced axis's own max.

    Relative to the max the dominant term is `exp(0) = 1`, so nothing that
    matters underflows at any dynamic range — the same argument that moved
    `scans.log_matmul` to a pairwise-max anchor after two wrong shifts were
    measured on real grammars. An all-`-inf` slice reduces to `-inf` rather
    than `NaN`, which is what `m - m = -inf - -inf` would give.
    """
    t = np.asarray(t, dtype=np.float64)
    m = np.max(t, axis=axis)
    finite = np.isfinite(m)
    safe = np.where(finite, m, 0.0)
    shifted = t - (safe if axis is None else np.expand_dims(safe, axis))
    s = np.sum(np.exp(shifted), axis=axis)
    with np.errstate(divide="ignore"):
        out = safe + np.log(s)
    out = np.where(finite, out, -np.inf)
    return float(out) if axis is None else out


@dataclasses.dataclass(frozen=True)
class ForwardBackward:
    """`a`/`b` in **both** representations: log space, and scaled linear.

    `a[i]` is `a_i` in SPEC's notation: the vector **before** position `i`, so
    `a[0] == start` and `a[L]` is the vector after the last position.
    Likewise `b[i]` is the vector before position `i`, with `b[L] == b_L`.

    **`log_a` / `log_b` are the primary representation and everything in this
    module reads them.** `a`, `b`, `log_scale_a`, `log_scale_b` are the same
    vectors max-normalised — `a[i] · exp(log_scale_a[i])` is the true prefix
    vector — kept because the JAX kernels take linear factors and the
    differential tests feed them straight across. They are lossy by
    construction: a vector whose internal dynamic range exceeds ~745 nats
    cannot be held in a float64 linear vector at any single scale, and that
    loss is exactly the defect the log fields exist to remove (see
    `forward_backward`).
    """

    a: np.ndarray        # [L+1, S]
    b: np.ndarray        # [L+1, S]
    log_scale_a: np.ndarray  # [L+1]
    log_scale_b: np.ndarray  # [L+1]
    log_Z: float
    scaled: bool
    log_a: np.ndarray | None = None   # [L+1, S], `-inf` where exactly zero
    log_b: np.ndarray | None = None   # [L+1, S]

    @property
    def la(self) -> np.ndarray:
        """`log a_i`, unscaled. Reconstructed for hand-built instances."""
        if self.log_a is not None:
            return self.log_a
        return _log(self.a) + np.asarray(self.log_scale_a)[:, None]

    @property
    def lb(self) -> np.ndarray:
        """`log b_i`, unscaled."""
        if self.log_b is not None:
            return self.log_b
        return _log(self.b) + np.asarray(self.log_scale_b)[:, None]

    def log_Z_at(self, i: int) -> float:
        """`logsumexp_s (log a_i(s) + log b_i(s))`.

        Identical in exact arithmetic to `log(a_i · b_i) + logscale_a[i] +
        logscale_b[i]`, and that is still what it computes for an unscaled
        `ForwardBackward` — but evaluated on the log vectors it survives a
        within-vector range the linear dot cannot hold.

        SPEC §2.4 requires this be **`i`-invariant**; §6.1 test 2 asserts it
        across all `i`.
        """
        return float(_lse(self.la[i] + self.lb[i]))


def forward_backward(
    M: np.ndarray,
    a_start: np.ndarray,
    b_final: np.ndarray,
    *,
    scaled: bool = True,
) -> ForwardBackward:
    """SPEC eq (3)/(4), longhand.

    **`scaled=True` runs the recursions in log space.** Max-normalising each
    vector and accumulating log-scales — what this used to do — bounds a
    vector's *maximum*, not its internal dynamic range, and that is not the
    same thing. It is the identical construction `scans.prefix_suffix` was
    rewritten to eliminate in commit `2c9f9d9`, and it fails here for the same
    reason: with SPEC §3.5's mandatory unscored `ACC --Σ--> ACC` tail beside a
    live chain, the tail's emission mass is exactly 1.0, so it pins `max_s
    b_i(s) = 1` while the chain sits at `e^{-30(L-i)}`. At `L = 32` the start
    state's `b_0` divides straight to exactly `0.0`, `log Z` comes back `-inf`,
    and `marginals` / `sample_chain` / `sample_tree` then raise
    `EmptyLanguageError` on a language whose true `log Z` is `-956.566` —
    CLAUDE.md's cause **(c)** delivered to the caller as **(a)/(b)**, the
    misclassification the taxonomy exists to prevent. Measured rate on random
    `V = 4` instances: 0/33 at `L=32, σ=45`; 1/35 at `L=64, σ=20`; 1/32 at
    `L=256, σ=10`; 2/31 at `L=256, σ=45`.

    Log space is the **simplest** correct fix rather than a port of the fast
    path's machinery, and this file is required to prefer that (CLAUDE.md:
    "plain float64 numpy with no cleverness"). `scans.prefix_suffix` cannot do
    this: its callers are JAX kernels that read a *pair of linear factors*, so
    it is forced into a balanced pair of scales, a widening floor and a
    `feasible_out` flag — a whole apparatus whose only purpose is to survive the
    two-factor linear interface. Here the consumers are three functions in this
    module, so the log vectors can simply be handed to them and every one of
    those trade-offs disappears. There is no accuracy question left to guard: an
    `L`-fold product of logs is exact to a rounding, at any dynamic range.

    `a`, `b`, `log_scale_a` and `log_scale_b` are still returned, still
    max-normalised, and still satisfy `a[i]·exp(log_scale_a[i]) == a_i`,
    because the JAX differential tests feed them straight into the kernels —
    `tests/test_numerics.py` (:102, :121–127, :356, :365) and
    `tests/test_exactness.py` (:516, :632–633) read them, and the convention is
    preserved: measured difference against the pre-log-space code is 3.3e-12.
    They remain lossy — that is a property of a linear vector at one scale, not
    of how it was computed.

    **No consumer in this module reads them on any path this module
    constructs.** The one thing that can is `ForwardBackward.la`/`.lb`, which
    fall back to reconstructing the log vectors from `a` and `log_scale_a` when
    `log_a is None` — i.e. only for a hand-built `ForwardBackward`, of which
    there are none in `diffgemma_fa/`, `tests/` or `scripts/`. Dead in
    practice, not dead by construction, and the difference is worth stating
    rather than rounding off.

    Args:
      M: `[L, S, S]`.
      a_start: `[S]`, the start **vector**.
      b_final: `[S]`, the terminal factor — `1[d(s) ≤ R]` in the port, **not**
        `1[s ∈ F]` (SPEC §3.1b).
      scaled: run in log space and return max-normalised linear vectors with
        their log-scales. `scaled=False` keeps the plain unnormalised linear
        recursion, so §6.3 can still demonstrate the underflow that motivates
        all of this.

    Returns:
      A `ForwardBackward`.
    """
    L, S, _ = M.shape
    a = np.zeros((L + 1, S), dtype=np.float64)
    b = np.zeros((L + 1, S), dtype=np.float64)
    ls_a = np.zeros(L + 1, dtype=np.float64)
    ls_b = np.zeros(L + 1, dtype=np.float64)

    if not scaled:
        # SPEC §6.3's demonstration path: no scaling of any kind, so the
        # underflow it exists to exhibit still happens.
        a[0] = a_start
        for i in range(L):
            a[i + 1] = a[i] @ M[i]
        b[L] = b_final
        for i in range(L - 1, -1, -1):
            b[i] = M[i] @ b[i + 1]
        dot = float(a[0] @ b[0])
        log_Z = -np.inf if dot <= 0.0 else float(np.log(dot))
        return ForwardBackward(a=a, b=b, log_scale_a=ls_a, log_scale_b=ls_b,
                               log_Z=log_Z, scaled=False,
                               log_a=_log(a), log_b=_log(b))

    logM = _log(M)                                        # [L, S, S]
    log_a = np.full((L + 1, S), -np.inf, dtype=np.float64)
    log_b = np.full((L + 1, S), -np.inf, dtype=np.float64)

    log_a[0] = _log(a_start)
    for i in range(L):
        # log a_{i+1}(t) = logsumexp_s (log a_i(s) + log M_i(s, t))
        log_a[i + 1] = _lse(log_a[i][:, None] + logM[i], axis=0)

    log_b[L] = _log(b_final)
    for i in range(L - 1, -1, -1):
        # log b_i(s) = logsumexp_t (log M_i(s, t) + log b_{i+1}(t))
        log_b[i] = _lse(logM[i] + log_b[i + 1][None, :], axis=1)

    # Max-normalised linear view. An all-`-inf` vector keeps the previous
    # scale — exactly what the old carried `ls_a[i+1] = ls_a[i]` did — so a
    # dead boundary does not silently reset the bookkeeping.
    for i in range(L + 1):
        m = float(np.max(log_a[i]))
        if np.isfinite(m):
            ls_a[i] = m
        elif i > 0:
            ls_a[i] = ls_a[i - 1]
        a[i] = np.exp(log_a[i] - ls_a[i])
    for i in range(L, -1, -1):
        m = float(np.max(log_b[i]))
        if np.isfinite(m):
            ls_b[i] = m
        elif i < L:
            ls_b[i] = ls_b[i + 1]
        b[i] = np.exp(log_b[i] - ls_b[i])

    log_Z = float(_lse(log_a[0] + log_b[0]))
    return ForwardBackward(a=a, b=b, log_scale_a=ls_a, log_scale_b=ls_b,
                           log_Z=log_Z, scaled=True,
                           log_a=log_a, log_b=log_b)


# ---------------------------------------------------------------------------
# §2.4 — per-position constrained marginals
# ---------------------------------------------------------------------------

def marginals(
    p: np.ndarray,
    automaton: Automaton,
    fb: ForwardBackward,
    W: np.ndarray,
) -> np.ndarray:
    """`q_i(v) = p_i(v) · (Σ_{e : v ∈ label(e)} u_i(e)) / Z`. SPEC eq (5)/(6).

    `u_i(e) = a_{i-1}(src e) · b_i(dst e)` — in 0-indexed terms, `a[i]` is the
    vector before position `i` and `b[i+1]` the one after it. Verified against
    brute-force enumeration; there is no off-by-one.

    **`u` is formed in log space and anchored on the position's own dominant
    edge.** `q_i` is normalised by its row total, so a per-position constant
    cancels exactly — which makes `max_e log u_i(e)` the anchor that loses
    least, and it is *available here* only because the row is closed over `i`.
    (`scans.prefix_suffix`' docstring rejects the same anchor for the JAX path,
    correctly: its consumers read `u` through `a` and `b` alone, with no
    channel to add a per-position scale back.) With this anchor the dominant
    edge sits at `exp(0)` at every position, so a live position can never come
    back empty at any dynamic range.

    Returns:
      `[L, V] float64` with `Σ_v q_i(v) == 1` for every `i`.
    """
    L, V = p.shape
    la, lb = fb.la, fb.lb
    q = np.zeros((L, V), dtype=np.float64)
    for i in range(L):
        # u_i over edges, then scatter each edge's mass onto its label set.
        log_u = np.array(
            [la[i][src] + lb[i + 1][dst] for src, dst, _ in automaton.edges],
            dtype=np.float64) if automaton.n_edges else np.zeros(0)
        anchor = np.max(log_u) if log_u.size else -np.inf
        acc = np.zeros(V, dtype=np.float64)
        if np.isfinite(anchor):
            u_all = np.exp(log_u - anchor)
            for e, (_, _, labels) in enumerate(automaton.edges):
                if u_all[e] == 0.0 or not labels:
                    continue
                idx = np.fromiter(sorted(labels), dtype=np.int64,
                                  count=len(labels))
                acc[idx] += u_all[e]
        row = p[i] * acc
        total = row.sum()
        if total <= 0.0:
            raise EmptyLanguageError(
                f"Z == 0 at position {i}: no live continuation. Classify before "
                "working around it (SPEC: (a) empty automaton, (b) budget, "
                "(c) underflow)."
            )
        q[i] = row / total
    return q


def marginals_complement_aware(
    p: np.ndarray,
    automaton: Automaton,
    fb: ForwardBackward,
    class_of: Sequence[int],
    class_members: Sequence[frozenset[int]],
    is_neg: Sequence[bool],
) -> np.ndarray:
    """`q_i` via SPEC §2.4's complement-aware inner sum.

    Under §4.4's class representation the scatter in eq (6) is **not** a plain
    sparse scatter: a negated class contributes everywhere. Since
    `1[v ∈ S_c] = 1 − 1[v ∈ N_c]`,

        r_i(v) = Σ_{c ∈ Neg} U_i(c)
               + Σ_{c ∈ Pos, v ∈ S_c} U_i(c)
               − Σ_{c ∈ Neg, v ∈ N_c} U_i(c)
        q_i(v) = p_i(v) · r_i(v) / Z          where  U_i(c) = Σ_{e ∈ class c} u_i(e)

    Getting this wrong silently produces a *plausible* wrong distribution that
    only the exactness suite catches.

    Args:
      class_of: `[E]`, edge -> class id.
      class_members: per class, the **true** member set `S_c`.
      is_neg: per class, whether it is stored as its complement.
    """
    L, V = p.shape
    n_classes = len(class_members)
    la, lb = fb.la, fb.lb
    q = np.zeros((L, V), dtype=np.float64)

    for i in range(L):
        # Same log-space, position-anchored `u` as `marginals` — see there.
        log_u = np.array(
            [la[i][src] + lb[i + 1][dst] for src, dst, _ in automaton.edges],
            dtype=np.float64) if automaton.n_edges else np.zeros(0)
        anchor = np.max(log_u) if log_u.size else -np.inf
        U = np.zeros(n_classes, dtype=np.float64)
        if np.isfinite(anchor):
            u_all = np.exp(log_u - anchor)
            for e in range(automaton.n_edges):
                U[class_of[e]] += u_all[e]

        r = np.zeros(V, dtype=np.float64)
        # Every negated class contributes its full mass everywhere...
        for c in range(n_classes):
            if is_neg[c]:
                r += U[c]
        # ...then positives add on their members, and negatives subtract on
        # the complement they actually store.
        for c in range(n_classes):
            if U[c] == 0.0:
                continue
            if is_neg[c]:
                comp = set(range(V)) - set(class_members[c])
                if comp:
                    idx = np.fromiter(sorted(comp), dtype=np.int64, count=len(comp))
                    r[idx] -= U[c]
            else:
                mem = class_members[c]
                if mem:
                    idx = np.fromiter(sorted(mem), dtype=np.int64, count=len(mem))
                    r[idx] += U[c]

        row = p[i] * r
        total = row.sum()
        if total <= 0.0:
            raise EmptyLanguageError(f"Z == 0 at position {i}")
        q[i] = row / total
    return q


# ---------------------------------------------------------------------------
# §2.5 — ancestral (chain) sampling, O(L) depth
# ---------------------------------------------------------------------------

def _choice(rng: np.random.Generator, weights: np.ndarray) -> int:
    total = weights.sum()
    if total <= 0.0:
        raise EmptyLanguageError("no admissible choice; Z == 0 along this path")
    return int(rng.choice(len(weights), p=weights / total))


def _exp_anchored(log_w: np.ndarray) -> np.ndarray:
    """`exp(log_w − max log_w)`, i.e. the same weights up to the one constant
    a subsequent normalisation removes. All-`-inf` in, all-zero out — so
    `_choice` still raises `EmptyLanguageError` on a genuinely dead draw.
    """
    log_w = np.asarray(log_w, dtype=np.float64)
    if log_w.size == 0:
        return np.zeros(0, dtype=np.float64)
    m = np.max(log_w)
    if not np.isfinite(m):
        return np.zeros_like(log_w)
    return np.exp(log_w - m)


def sample_chain(
    p: np.ndarray,
    automaton: Automaton,
    fb: ForwardBackward,
    W: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact, **path-weighted** ancestral sampling. SPEC §2.5.

    It draws the *edge* first, which is what makes it path-weighted and hence
    the reference the tree sampler must match in distribution.

    Every draw here is a categorical over one position's candidates, and
    `_choice` normalises by their sum — so each weight vector is formed in log
    space and anchored on its own maximum. That is exact at any dynamic range
    and it is what stops SPEC §3.5's unscored tail from normalising the live
    chain's `b` to zero (see `forward_backward`).

    Returns:
      `(tokens [L], states [L+1])`.
    """
    L = p.shape[0]
    la, lb = fb.la, fb.lb
    # The start is a vector, so draw s_0 ~ a_start(s) * b_0(s) (SPEC §5.7).
    s = _choice(rng, _exp_anchored(la[0] + lb[0]))

    tokens = np.zeros(L, dtype=np.int64)
    states = np.zeros(L + 1, dtype=np.int64)
    states[0] = s

    for i in range(L):
        # e_i ~ P(e) proportional to 1[src(e)=s] * W[i,e] * b_i(dst e)
        log_w = np.full(automaton.n_edges, -np.inf, dtype=np.float64)
        logW_i = _log(W[i])
        for e, (src, dst, _) in enumerate(automaton.edges):
            if src == s:
                log_w[e] = logW_i[e] + lb[i + 1][dst]
        e = _choice(rng, _exp_anchored(log_w))
        _, dst, labels = automaton.edges[e]

        # x_i ~ P(v) proportional to p_i(v) * 1[v in label(e_i)]
        idx = np.fromiter(sorted(labels), dtype=np.int64, count=len(labels))
        tokens[i] = idx[_choice(rng, p[i][idx])]
        s = dst
        states[i + 1] = s

    return tokens, states


# ---------------------------------------------------------------------------
# §2.6 — tree (log-depth) sampling, reference non-parallel form
# ---------------------------------------------------------------------------

def block_products(M: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """All aligned dyadic block products `P_{[ℓ,r)} = M_ℓ ⋯ M_{r−1}`.

    These are exactly the `reduced_elems` `lax.associative_scan` computes and
    throws away (SPEC §2.6(a)) — the JAX path hand-rolls a Blelloch up-sweep to
    retain them. Here they are built directly, which is the point of a
    reference.

    Note the operand order: `P_{[ℓ,m)} @ P_{[m,r)}`, left block first. Getting
    it backwards is silent — no shape error, just wrong probabilities
    (SPEC §2.6(c)).
    """
    L = M.shape[0]
    out: dict[tuple[int, int], np.ndarray] = {}
    for i in range(L):
        out[(i, i + 1)] = M[i]
    width = 2
    while width <= L:
        for lo in range(0, L - width + 1, width):
            mid = lo + width // 2
            hi = lo + width
            if (lo, mid) in out and (mid, hi) in out:
                out[(lo, hi)] = out[(lo, mid)] @ out[(mid, hi)]
        width *= 2
    return out


def _range_product(M: np.ndarray, lo: int, hi: int,
                   cache: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """`P_{[lo,hi)}`, memoized. Falls back to a direct product off the dyadic
    grid, which is what makes this handle `L` that is not a power of two."""
    if (lo, hi) in cache:
        return cache[(lo, hi)]
    if hi - lo == 1:
        cache[(lo, hi)] = M[lo]
        return M[lo]
    mid = (lo + hi) // 2
    prod = _range_product(M, lo, mid, cache) @ _range_product(M, mid, hi, cache)
    cache[(lo, hi)] = prod
    return prod


def _log_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """`C[i,j] = logsumexp_k (A[i,k] + B[k,j])`, pairwise-max anchored.

    The numpy twin of `scans.log_matmul`, and for the reason recorded there at
    length: only the pairwise max makes every *entry's* dominant term `exp(0)`,
    so nothing that matters underflows at any dynamic range. A single row/col
    shift returns `Z == 0` on a provably non-empty language as soon as SPEC
    §3.5's unscored tail sits beside a real grammar path.
    """
    t = A[:, :, None] + B[None, :, :]          # [S, S, S]
    return np.asarray(_lse(t, axis=1))


def _log_range_product(logM: np.ndarray, lo: int, hi: int,
                       cache: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """`log P_{[lo,hi)}`, memoized, left block first (SPEC §2.6(c))."""
    if (lo, hi) in cache:
        return cache[(lo, hi)]
    if hi - lo == 1:
        cache[(lo, hi)] = logM[lo]
        return logM[lo]
    mid = (lo + hi) // 2
    prod = _log_matmul(_log_range_product(logM, lo, mid, cache),
                       _log_range_product(logM, mid, hi, cache))
    cache[(lo, hi)] = prod
    return prod


def sample_tree(
    p: np.ndarray,
    automaton: Automaton,
    M: np.ndarray,
    W: np.ndarray,
    a_start: np.ndarray,
    b_final: np.ndarray,
    rng: np.random.Generator,
    *,
    use_exists_form: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1, written as a plain recursion. SPEC §2.6.

    Draws the boundary pair jointly, recurses on midpoints, then draws every
    token independently once all boundary states are fixed:

        (s_0, s_L) ~ a_start(s_0) · P_{[0,L)}(s_0, s_L) · b_L(s_L)
        P(s_m | s_ℓ, s_r) ∝ P_{[ℓ,m)}(s_ℓ, s_m) · P_{[m,r)}(s_m, s_r)     (7)
        e_i ~ 1[src=s_i, dst=s_{i+1}] · W[i, e]                            (8a)
        x_i ~ p_i(v) · 1[v ∈ label(e_i)]                                   (8b)

    **The dyadic products are formed in log space** (`_log_matmul`), for the
    same reason `forward_backward` is and `tree.sample_states_log` is on the
    JAX side: an `L`-fold linear product of matrices whose entries span SPEC
    §3.5's unscored tail (mass 1.0) beside a real grammar path (`e^{-30}` a
    token) underflows to exactly zero, and the root draw then raises
    `EmptyLanguageError` — cause (c) reported as (a)/(b). Each draw is a
    categorical, so the log weights are anchored on their own maximum and the
    one constant that leaves cancels in the normalisation.

    Args:
      use_exists_form: draw `x_i ∝ p_i(v) · 1[∃e : …, v ∈ label(e)]` instead of
        the edge-multiplicity-weighted form. **This is the bug**, exposed
        deliberately so the exactness suite can prove it wrong: on an NFA with
        parallel overlapping edges it deviates from the exact posterior by
        ~1.7e-2 where the correct form is at 2e-17. On a DFA the two coincide.

    Returns:
      `(tokens [L], states [L+1])`.
    """
    L = p.shape[0]
    S = automaton.n_states
    logM = _log(M)
    cache: dict[tuple[int, int], np.ndarray] = {}

    # --- root: draw (s_0, s_L) jointly -----------------------------------
    log_P0L = _log_range_product(logM, 0, L, cache)
    log_joint = (_log(a_start)[:, None] + log_P0L + _log(b_final)[None, :])
    joint = _exp_anchored(log_joint.ravel())
    total = joint.sum()
    if total <= 0.0:
        raise EmptyLanguageError(
            "Z == 0: no accepted string of this length within budget"
        )
    flat = int(rng.choice(S * S, p=joint / total))
    s0, sL = divmod(flat, S)

    states = np.full(L + 1, -1, dtype=np.int64)
    states[0] = s0
    states[L] = sL

    # --- recurse on midpoints, eq (7) ------------------------------------
    def recurse(lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        left = _log_range_product(logM, lo, mid, cache)
        right = _log_range_product(logM, mid, hi, cache)
        log_w = left[states[lo], :] + right[:, states[hi]]
        states[mid] = _choice(rng, _exp_anchored(log_w))
        recurse(lo, mid)
        recurse(mid, hi)

    recurse(0, L)

    # --- tokens, independently, given the fixed boundary states ----------
    tokens = np.zeros(L, dtype=np.int64)
    for i in range(L):
        s, t = states[i], states[i + 1]
        if use_exists_form:
            support: set[int] = set()
            for e, (src, dst, labels) in enumerate(automaton.edges):
                if src == s and dst == t:
                    support |= labels
            if not support:
                raise EmptyLanguageError(f"no edge {s}->{t} at position {i}")
            idx = np.fromiter(sorted(support), dtype=np.int64, count=len(support))
            tokens[i] = idx[_choice(rng, p[i][idx])]
        else:
            w = np.zeros(automaton.n_edges, dtype=np.float64)
            for e, (src, dst, _) in enumerate(automaton.edges):
                if src == s and dst == t:
                    w[e] = W[i, e]
            e = _choice(rng, w)
            labels = automaton.edges[e][2]
            idx = np.fromiter(sorted(labels), dtype=np.int64, count=len(labels))
            tokens[i] = idx[_choice(rng, p[i][idx])]

    return tokens, states


# ---------------------------------------------------------------------------
# §2.7 — constrained MAP via the max-plus semiring, in log space
# ---------------------------------------------------------------------------

_NEG = -3e38  # finite sentinel, never -inf: -inf + -inf gives NaN in fused kernels


def map_decode(
    p: np.ndarray,
    automaton: Automaton,
    a_start: np.ndarray,
    b_final: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact constrained MAP. SPEC §2.7.

        M̃_i(s, s') = max_{e : src=s, dst=s'} max_{v ∈ label(e)} log p_i(v)   (9)

    `(max, +)` is a semiring, so the identical recursion gives exact MAP.
    Log space throughout: exact, no scaling discussion, no underflow.

    **MAP is exact on NFAs too** — path multiplicity cannot change a `max` over
    strings — so §2.2's soft-proxy caveat applies to *sampling only*.

    **Tie-break: lowest token id, then lowest state id.** `test_guarantee` and
    the unconstrained-equivalence test depend on it.

    **The floor under `p` is `finfo(dtype).tiny`, derived from the dtype, and
    it must stay that way.** It was `1e-300`, which is eight orders of
    magnitude above the JAX path's — `model/constrained.map_log_floor` was
    moved from `1e-30` to `finfo(dtype).tiny` and its docstring records that
    finding at length. A floor is not a clamp on a value that is about to be
    multiplied by zero (SPEC §2.4's `1e-30` rule about `q`, which has exact
    zeros); `p` is a softmax with no exact zeros, so every token underneath the
    floor scores the same `log(floor)` and the §2.7 tie-break then emits the
    *smallest admissible token id* rather than the most probable one, with
    `feasible` still True. Measured on the old floor: MAP emitted token 0 at
    `p = 1.000e-305` over token 1 at `p = 1.000e-301`. Reachable — softcapped
    logits with a 56-nat gap at `T = 0.08`, and `_MIN_TEMP = 1e-12` makes any
    `T` reachable by configuration alone (SPEC §3.9). While the two floors
    differed, **the arbiter was less exact than the path it certifies** and any
    differential test landing in the window would have blamed the fast path.

    Returns:
      `(tokens [L], states [L+1], log_score)`.
    """
    L, _ = p.shape
    S = automaton.n_states
    # Smallest normal of the input dtype, exactly `constrained.map_log_floor`.
    floor = float(np.finfo(np.asarray(p).dtype).tiny)
    with np.errstate(divide="ignore"):
        logp = np.where(p > 0.0, np.log(np.maximum(p, floor)), _NEG)

    # M_tilde[i, s, s'] and the token realising it.
    Mt = np.full((L, S, S), _NEG, dtype=np.float64)
    arg = np.full((L, S, S), -1, dtype=np.int64)
    for src, dst, labels in automaton.edges:
        if not labels:
            continue
        idx = np.fromiter(sorted(labels), dtype=np.int64, count=len(labels))
        for i in range(L):
            sub = logp[i][idx]
            k = int(np.argmax(sub))          # argmax ties -> lowest index ->
            best, tok = float(sub[k]), int(idx[k])   # lowest token id, as required
            if best > Mt[i, src, dst] or (
                best == Mt[i, src, dst] and 0 <= tok < arg[i, src, dst]
            ):
                Mt[i, src, dst] = best
                arg[i, src, dst] = tok

    with np.errstate(divide="ignore"):
        log_a_start = np.where(a_start > 0.0,
                               np.log(np.maximum(a_start, floor)), _NEG)
        log_b_final = np.where(b_final > 0.0,
                               np.log(np.maximum(b_final, floor)), _NEG)

    # Forward max-plus, retaining backpointers.
    alpha = np.full((L + 1, S), _NEG, dtype=np.float64)
    back = np.full((L + 1, S), -1, dtype=np.int64)
    alpha[0] = log_a_start
    for i in range(L):
        cand = alpha[i][:, None] + Mt[i]          # [S, S]
        best = cand.max(axis=0)
        # Lowest state id wins ties: argmax returns the first maximum.
        src = cand.argmax(axis=0)
        alpha[i + 1] = best
        back[i + 1] = np.where(best > _NEG / 2, src, -1)

    final = alpha[L] + log_b_final
    if not np.any(final > _NEG / 2):
        raise EmptyLanguageError(
            "no accepted string of this length within budget (MAP)"
        )
    sL = int(np.argmax(final))
    score = float(final[sL])

    states = np.zeros(L + 1, dtype=np.int64)
    states[L] = sL
    for i in range(L, 0, -1):
        states[i - 1] = back[i][states[i]]
    tokens = np.array(
        [arg[i, states[i], states[i + 1]] for i in range(L)], dtype=np.int64
    )
    return tokens, states, score


# ---------------------------------------------------------------------------
# Brute force, for the exactness harness
# ---------------------------------------------------------------------------

def accepting_path_count(
    automaton: Automaton,
    tokens: Sequence[int],
    b_final: np.ndarray | None = None,
) -> float:
    """`a_start^T C(x_0) ⋯ C(x_{L−1}) b_L`, the weighted accepting-path count.

    On a DFA this is 0 or 1. On an NFA it is the multiplicity that makes the
    constrained posterior path-weighted rather than uniform on `C`
    (SPEC §2.2).
    """
    b = automaton.final_vector() if b_final is None else b_final
    v = automaton.start.astype(np.float64)
    for t in tokens:
        v = v @ label_count_matrix(automaton, int(t))
        if not v.any():
            return 0.0
    return float(v @ b)


def enumerate_posterior(
    p: np.ndarray,
    automaton: Automaton,
    b_final: np.ndarray | None = None,
) -> tuple[dict[tuple[int, ...], float], float]:
    """Exhaustive `V^L` enumeration of the exact path-weighted posterior.

    The ground truth the whole suite is checked against. Only tractable at the
    `V = 4, L ≤ 8` scale SPEC §6.1 prescribes.

    Returns:
      `(posterior, Z)` with `posterior` normalised over its support.
    """
    L, V = p.shape
    b = automaton.final_vector() if b_final is None else b_final
    weights: dict[tuple[int, ...], float] = {}
    Z = 0.0
    for x in itertools.product(range(V), repeat=L):
        n_paths = accepting_path_count(automaton, x, b)
        if n_paths == 0.0:
            continue
        w = n_paths
        for i, v in enumerate(x):
            w *= p[i, v]
        if w > 0.0:
            weights[x] = w
            Z += w
    if Z > 0.0:
        weights = {k: v / Z for k, v in weights.items()}
    return weights, Z


def marginals_from_posterior(
    posterior: dict[tuple[int, ...], float], L: int, V: int
) -> np.ndarray:
    """Per-position marginals of an enumerated posterior, for checking eq (6)."""
    q = np.zeros((L, V), dtype=np.float64)
    for x, w in posterior.items():
        for i, v in enumerate(x):
            q[i, v] += w
    return q


def simulate(automaton: Automaton, tokens: Iterable[int]) -> set[int]:
    """`δ*(start, tokens)` as a **set**, so it is correct for NFAs too."""
    cur = {s for s in range(automaton.n_states) if automaton.start[s] > 0}
    for t in tokens:
        nxt: set[int] = set()
        for src, dst, labels in automaton.edges:
            if src in cur and t in labels:
                nxt.add(dst)
        cur = nxt
        if not cur:
            return cur
    return cur


def accepts(automaton: Automaton, tokens: Iterable[int]) -> bool:
    """Independent acceptance check, used to validate every sample and MAP."""
    return bool(simulate(automaton, tokens) & automaton.finals)


# ---------------------------------------------------------------------------
# §4.4 Layer 2b / §2.7 — the max table's topk recovery and the tie-break rule
# ---------------------------------------------------------------------------

def max_over_class_topk(
    p_row: np.ndarray,
    complement: frozenset[int],
    k: int,
) -> tuple[float, int]:
    """`max_{v ∈ S_c} p_i(v)` and its argmax, for a class stored as `N_c`.

    `max` has no complement trick (SPEC §4.4 Layer 2b), so a negated class is
    evaluated as **the first `topk(p, K)` entry not in `N_c`**. The bound
    `K > |N_c|` is tight: with `K = |N_c|` an adversarial `p` puts the whole
    complement in the top `K` and the answer is missed.

    Ties resolve to the **lowest token id**, matching §2.7's stated convention.

    Raises:
      LookupError: if no top-`k` entry lies outside `N_c` — i.e. `k` was too
        small. Raising rather than returning a wrong maximum is the point.
    """
    V = p_row.shape[0]
    k = min(k, V)
    # Sort by (-p, token) so ties go to the lowest token id.
    order = sorted(range(V), key=lambda v: (-float(p_row[v]), v))[:k]
    for v in order:
        if v not in complement:
            return float(p_row[v]), int(v)
    raise LookupError(
        f"no top-{k} entry outside N_c (|N_c| = {len(complement)}); "
        "K must exceed max_c |N_c| (SPEC §4.4 Layer 2b)"
    )


def segment_max_argmin(
    values: np.ndarray,
    segments: np.ndarray,
    num_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Segmented max with **lowest-index** tie-breaking. SPEC §2.7.

    JAX exports no `segment_argmax`, and the obvious two-pass workaround gets
    the tie-break backwards — `segment_max` over indices resolves ties to the
    *largest* index. The correct form uses `segment_min` with an `N` sentinel:

        mx = segment_max(vals, seg)
        am = segment_min(where(vals == mx[seg], idx, N), seg)

    Empty segments give `-inf` (segment_max's identity, **not** `-1`) and
    `am == N`. **Test for `am == N`, never `am == -1`.**

    Returns:
      `(max_values [num_segments], argmin_indices [num_segments])`.
    """
    n = values.shape[0]
    mx = np.full(num_segments, -np.inf, dtype=np.float64)
    for i in range(n):
        s = int(segments[i])
        if values[i] > mx[s]:
            mx[s] = values[i]
    am = np.full(num_segments, n, dtype=np.int64)  # N sentinel, not -1
    for i in range(n):
        s = int(segments[i])
        if values[i] == mx[s] and i < am[s]:
            am[s] = i
    return mx, am
