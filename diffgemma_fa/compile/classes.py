"""Token-set representation: label interning and the two CSR tables. SPEC §4.4.

Every denoising step needs `W[i, e] = Σ_{v ∈ label(e)} p_i(v)` for all 256
positions and every edge. That is `W = A pᵀ` with `A ∈ {0,1}^{E×V}` and
`V = 262,144`. Naively: dense bitsets 5.5 GB, dense float `A` 178 GB, plain CSR
with `nnz = Σ_e |label(e)|` where a JSON string-body edge carries ~260k members.
All dead.

Four layers:

1. **Intern** labels into equivalence classes — the same `STRING_INNER` set
   recurs at every string position.
2. **Polarity for the SUM table** — store the complement `N_c` when
   `|S_c| > V/2`. Without this the scheme fails: measured on real BFCL, the
   largest class covers **~99.6% of the vocab**.

   The requirement this table has to satisfy is `W_c[c, i] = Σ_{v ∈ S_c} p_i(v)`
   — the class's mass, *whatever* polarity it is stored in. **How** the consumer
   recovers that from `N_c` is `infer/marginals.py::class_weights`' business and
   is deliberately not specified here.

   This paragraph used to read "use `W_c = total[i] − Σ_{v ∈ N_c} p_i(v)`", and
   that sentence is why the defect of 2026-08-10 survived review as
   *documentation*: a subtraction of two near-equal doubles was written down as
   the specification, so an implementation matching it looked correct by
   inspection, and the identity — true in exact arithmetic, catastrophically
   cancelling in float64 whenever the model is confident about a token inside
   `N_c` — was never the thing under test. It destroyed the constrained language
   on 73 of 250 Countdown records. Prescribing an implementation in a spec
   removes the gap an implementation can be found wrong in; state the quantity,
   not the expression.
3. **2b. Polarity for the MAX table, computed independently.** `max` has no
   complement trick, so a negated class is evaluated as "the first `topk(p, K)`
   entry not in `N_c`", which forces `K > max_c |N_c|`. Storing the complement
   whenever `|S_c| > V/2` would make `K` as large as 131,073. So the max table
   uses its own rule — complement only when `|N_c| ≤ K_max` — and therefore its
   own Pos/Neg partition. **The two flag arrays must never be shared.**
4. **CSR over classes**, int32 indices, no values array (all ones).

Measured in Phase 0/1: real BFCL classes have `|N_c|` in the **988–1,064**
range, so SPEC's original `K_max = 256` default would have pushed every one of
them to positive storage and blown the CSR budget by ~4 orders of magnitude.
The default here is **1,100**, and `K_max` is derived from the grammar when
possible.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["ClassTables", "DEFAULT_K_MAX", "intern_labels", "build_tables"]

#: See module docstring. NOT 256.
DEFAULT_K_MAX = 1100


@dataclasses.dataclass(frozen=True)
class ClassTables:
    """Interned label classes plus the two independently-polarised CSR tables.

    Shapes and dtypes:
      class_id            [E]        int32   edge -> class
      sum_is_neg          [C]        bool    SUM table polarity
      sum_indptr          [C+1]      int32   CSR over the *stored* member set
      sum_indices         [nnz_sum]  int32   token ids
      max_is_neg          [C]        bool    MAX table polarity (INDEPENDENT)
      max_indptr          [C+1]      int32
      max_indices         [nnz_max]  int32
      class_size          [C]        int32   true |S_c|, before complementing
    """

    n_classes: int
    n_edges: int
    vocab_size: int
    k_max: int

    class_id: np.ndarray
    class_size: np.ndarray

    sum_is_neg: np.ndarray
    sum_indptr: np.ndarray
    sum_indices: np.ndarray

    max_is_neg: np.ndarray
    max_indptr: np.ndarray
    max_indices: np.ndarray

    #: `max_c |N_c|` over classes stored negatively in the MAX table. The topk
    #: buffer must satisfy `K > this`; SPEC §4.4's verified bound is tight at
    #: `K = |N_c|`.
    max_neg_size: int

    def __post_init__(self) -> None:
        if self.max_neg_size >= self.k_max + 1:
            raise ValueError(
                f"K = k_max + 1 = {self.k_max + 1} must exceed max_c |N_c| = "
                f"{self.max_neg_size}; the topk recovery of "
                "max_{v in S_c} p_i(v) is provably wrong otherwise (SPEC §4.4)"
            )
        if self.sum_is_neg is self.max_is_neg:
            raise ValueError("SUM and MAX polarity arrays must not be shared")

    @property
    def dedup_ratio(self) -> float:
        """SPEC §4.4 Layer 1 / open question 9 — unpublished. Edges per class."""
        return self.n_edges / max(1, self.n_classes)

    @property
    def nnz_sum(self) -> int:
        return int(self.sum_indices.size)

    @property
    def nnz_max(self) -> int:
        return int(self.max_indices.size)

    @property
    def nbytes(self) -> int:
        return int(sum(
            a.nbytes for a in (
                self.class_id, self.class_size,
                self.sum_is_neg, self.sum_indptr, self.sum_indices,
                self.max_is_neg, self.max_indptr, self.max_indices,
            )
        ))


def intern_labels(
    labels: list[frozenset[int]],
) -> tuple[np.ndarray, list[frozenset[int]]]:
    """Layer 1. Group identical label sets.

    Args:
      labels: one canonical label set per edge.

    Returns:
      `(class_id[E] int32, classes)` where `classes[c]` is the member set.
    """
    lookup: dict[frozenset[int], int] = {}
    classes: list[frozenset[int]] = []
    class_id = np.empty(len(labels), dtype=np.int32)
    for e, lab in enumerate(labels):
        c = lookup.get(lab)
        if c is None:
            c = len(classes)
            lookup[lab] = c
            classes.append(lab)
        class_id[e] = c
    return class_id, classes


def _build_csr(
    classes: list[frozenset[int]],
    is_neg: np.ndarray,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """CSR over whichever member set each class stores (S_c or N_c)."""
    n = len(classes)
    indptr = np.zeros(n + 1, dtype=np.int32)
    chunks: list[np.ndarray] = []
    for c, members in enumerate(classes):
        if is_neg[c]:
            stored = np.fromiter(
                (v for v in range(vocab_size) if v not in members),
                dtype=np.int32,
                count=vocab_size - len(members),
            )
        else:
            stored = np.fromiter(sorted(members), dtype=np.int32, count=len(members))
        chunks.append(stored)
        indptr[c + 1] = indptr[c] + stored.size
    indices = (np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32))
    return indptr, indices.astype(np.int32, copy=False)


def build_tables(
    edge_labels: list[frozenset[int]],
    *,
    vocab_size: int,
    k_max: int = DEFAULT_K_MAX,
    auto_k_max: bool = True,
) -> ClassTables:
    """Build the interned classes and both CSR tables.

    Args:
      edge_labels: canonical label set per edge.
      vocab_size: `V`, 262,144 for Gemma.
      k_max: MAX-table polarity threshold — complement only when
        `|N_c| ≤ k_max`. Ignored when `auto_k_max` finds a larger requirement.
      auto_k_max: raise `k_max` to cover the grammar's actual complements. This
        is nearly always what you want: the alternative is silently demoting a
        260k-member class to positive storage.

    Returns:
      A `ClassTables`.
    """
    class_id, classes = intern_labels(edge_labels)
    n_classes = len(classes)
    class_size = np.fromiter((len(c) for c in classes), dtype=np.int32,
                             count=n_classes)

    # Layer 2 — SUM polarity: pure size rule, the complement is exact.
    sum_is_neg = class_size > (vocab_size // 2)

    # Layer 2b — MAX polarity: INDEPENDENT rule keyed on the complement size,
    # because K (the topk width) is bounded by max_c |N_c|, not by |S_c|.
    complement_size = vocab_size - class_size
    if auto_k_max and n_classes:
        # Consider only classes we would actually want to negate (those whose
        # complement is smaller than the class itself); take the largest.
        worth_negating = complement_size < class_size
        if worth_negating.any():
            needed = int(complement_size[worth_negating].max())
            k_max = max(k_max, needed)
    max_is_neg = complement_size <= k_max

    sum_indptr, sum_indices = _build_csr(classes, sum_is_neg, vocab_size)
    max_indptr, max_indices = _build_csr(classes, max_is_neg, vocab_size)

    max_neg_size = int(complement_size[max_is_neg].max()) if max_is_neg.any() else 0

    return ClassTables(
        n_classes=n_classes,
        n_edges=len(edge_labels),
        vocab_size=vocab_size,
        k_max=k_max,
        class_id=class_id,
        class_size=class_size,
        sum_is_neg=sum_is_neg,
        sum_indptr=sum_indptr,
        sum_indices=sum_indices,
        max_is_neg=max_is_neg,
        max_indptr=max_indptr,
        max_indices=max_indices,
        max_neg_size=max_neg_size,
    )
