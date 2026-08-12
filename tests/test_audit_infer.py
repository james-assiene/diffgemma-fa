"""Audit of `diffgemma_fa/infer/` — the math core. SPEC §2.4, §2.6, §2.7, §6.1.

**Why this file exists.** The four-slice audit covered `compile/`, `model/`,
`eval/` and the sampler. It had no slice for `infer/`, and three defects have
since been found there — `class_weights`' subtract-shaped complement,
`scatter_edge_mass_to_tokens`' identical algebra, and `prefix_suffix`'
per-vector normalisation — none of them by the audit. All three shared one
shape: *a scaling assumption that holds at toy scale and fails at production
sharpness, with no detector.*

**The highest-stakes thing here is `infer/reference.py` itself.** CLAUDE.md:
*"Every optimized path is differential-tested against it. If the fast path and
the reference disagree, the fast path is wrong."* Nothing had ever tested the
reference. `tests/test_exactness.py` checks it against
`reference.enumerate_posterior` — which lives **in the same file**, so the
bedrock suite is a self-consistency argument, not an external one.

So Part 1 below checks `reference.py` against an oracle that shares no code with
it and no arithmetic with it:

  * `_ExactOracle` — the constrained posterior of SPEC §2.2 evaluated in
    **exact rational arithmetic** (`fractions.Fraction`, unbounded precision, no
    rounding anywhere). Path multiplicity comes from a memoized depth-first
    search over edges; there is no matrix, no forward-backward, no `Z`, and no
    line of `reference.py` involved. Two independent things are being checked at
    once: the values, and the fact that `enumerate_posterior` — the oracle the
    *existing* suite trusts — is itself right.
  * `_decimal_log_partition` — a stdlib `decimal` forward pass at 60 significant
    digits with unbounded exponent and **no scaling of any kind**, for `L = 32`
    where enumeration is out of reach. This is the reviewer-endorsed reference
    for anything in SPEC §2.4.

Parts 2 and 3 cover `infer/tree.py` (only ever exercised incidentally, via the
sampler slice) and the corners of `infer/scans.py` that the three post-hoc files
were not written to reach.

**Regime, stated beside every number** — because "toy" means the wrong regime,
not merely small. Every generator here is swept over: DFA *and* NFA with
parallel overlapping edges; flat `p`, `softmax(N(0,1)·σ)` at σ up to 45, and
`wide_span_p` at a **stated** 207-nat (`10^90`) span; `b_L = 1[s ∈ F]` *and* the
budget-aware `1[d(s) ≤ R]` at several `R`; point-mass, set-valued and weighted
start vectors; positive, narrow-negated and mixed-polarity class tables.

**Statistics.** Fixed seeds; every χ² threshold Bonferroni-corrected by
`N_CHI2_TESTS`. Re-run protocol on a suspected flake: bump `SEED_OFFSET`. A real
distributional bug fails at every offset; a fluke moves.

**Mutation.** Every mutant used to validate this file is registered in
`_MUTANTS` below and reproducible with `DGFA_MUT=<name>`; a claim nobody can
re-run is not evidence (the house rule, from `test_audit_prefix_suffix.py`).
Unusually for this repo, these mutants are **the exact pre-slice functions**,
spliced back in from `git show HEAD:` and bound onto the live module — not
paraphrases — so a kill is evidence about the implementation, not about the
harness.

**Known red at the time of writing**, both pre-existing and neither introduced
by the `infer/` repair slice:

  * `test_reference_complement_aware_marginals_survive_a_10e90_p_span` — the
    subtract-shaped complement algebra in `marginals_complement_aware`, the
    third instance of the defect that `class_weights` (`da1294a`) and
    `scatter_edge_mass_to_tokens` were each rebuilt to remove. 6/82
    mixed-polarity instances lose mass outright; plain `marginals` is exact on
    the same inputs.
Closed since the first draft of this file, and listed so the record is not
re-litigated: `forward_backward` and every consumer of it (the unscored-tail
set), `map_decode`'s `1e-300` floor, `tree.map_states_and_tokens`' tie-break,
and `up_sweep`'s silent leaf flush — the last via the `TreeLevels.underflow`
flag, whose narrow contract is stated at
`test_the_underflow_flag_reports_the_division_not_the_position` and whose
threshold is `XLA_FLUSH_NATS` (708.4 nats under XLA, **not** float64's 744.4 —
they differ because XLA:CPU flushes subnormals, and the distinction is what
places every fixture in this file).
"""

from __future__ import annotations

import decimal
import itertools
import math
import os
import subprocess
import sys
import types
from decimal import Decimal
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np
import pytest
from scipy import stats

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.infer import marginals as MG  # noqa: E402
from diffgemma_fa.infer import reference as R  # noqa: E402
from diffgemma_fa.infer import scans, tree  # noqa: E402

V = 4                    # SPEC §6.1's alphabet: exhaustive enumeration is the point
SEED_OFFSET = 0

#: Bonferroni denominator. Counted generously — over-counting only makes the
#: suite more permissive, and a real distributional bug fails by orders of
#: magnitude rather than marginally.
N_CHI2_TESTS = 120
ALPHA = 0.001
ALPHA_CORRECTED = ALPHA / N_CHI2_TESTS

decimal.getcontext().prec = 60
decimal.getcontext().Emin = -999999999
decimal.getcontext().Emax = 999999999


# ===========================================================================
# Reproducible mutation registry.  `DGFA_MUT=<name> pytest tests/test_audit_infer.py`
# ===========================================================================
#
# **These mutants are the exact pre-slice functions, not paraphrases.** Each one
# is read out of `git show HEAD:diffgemma_fa/infer/<module>.py`, exec'd into a
# fresh namespace, and bound onto the *live* module — so what runs is the code
# that shipped before this slice, in place, against the shipped callers. A
# paraphrase would only prove the harness self-consistent; CLAUDE.md's rule is
# to mutate the implementation, and a registry nobody can re-run is not evidence.
#
# `fb_linear` needs one adapter and it is worth stating precisely: HEAD's
# `forward_backward` returns HEAD's `ForwardBackward`, which has no `log_a` /
# `log_b`. It is re-wrapped into the *current* dataclass with both set to
# `None`, so `la` / `lb` fall back to `_log(a) + log_scale_a[:, None]` — the
# lossy linear representation, reconstructed. That is a faithful revert of the
# information content, not a weakening: the pre-slice module had nothing else.

_MUTANTS: dict[str, str] = {
    # --- reverts of `infer/reference.py` --------------------------------
    "fb_linear":
        "forward_backward — max-normalised linear recursion. Kills the whole "
        "unscored-tail set: log_Z, log_Z_at, marginals, both samplers.",
    "logZ_at_linear":
        "ForwardBackward.log_Z_at — `log(a_i·b_i) + ls_a + ls_b`. Kills "
        "test_reference_forward_backward_survives_an_unscored_tail_... at "
        "assert 2, and the grid test at every cell. Not equivalent: handed the "
        "FIXED forward_backward it still returns -inf at interior boundaries.",
    "marginals_old":
        "marginals — linear `u = a[i][src]·b[i+1][dst]`. Kills the tail set at "
        "assert 3 (EmptyLanguageError on a non-empty language).",
    "chain_old":
        "sample_chain — linear edge weights. Kills the tail set at assert 4.",
    "tree_old":
        "sample_tree — linear dyadic products. Kills the tail set at assert 5. "
        "This is the fix nobody briefed and it was wholly unpinned.",
    "ca_old":
        "marginals_complement_aware — linear `U`. Kills the tail-consumer set "
        "and the complement-aware regime tests.",
    "map_floor_1e300":
        "map_decode — the `1e-300` clamp. Kills "
        "test_reference_map_is_exact_on_marginals_the_model_can_produce and "
        "test_reference_map_floor_sits_under_what_the_model_can_represent.",
    # --- reverts of `infer/tree.py` -------------------------------------
    "tiebreak_edge_index":
        "map_states_and_tokens — ties by lowest EDGE index. Kills "
        "test_tree_map_tie_break_is_lowest_token_id.",
    # --- reverts of `infer/scans.py` ------------------------------------
    "underflow_always_false":
        "TreeLevels.underflow forced to all-False. Kills "
        "test_up_sweep_normalisation_does_not_delete_a_live_transition, whose "
        "assertion is the disjunction 'the value survives OR the caller is "
        "told' — with the flag muted, neither holds.",
}

#: **Measured 2026-08-12**, CPU, float64, `JAX_PLATFORMS=cpu`, against the
#: shipped implementation. `killed` is *net of the baseline*, which stands at 1
#: — `test_reference_complement_aware_marginals_survive_a_10e90_p_span`, a
#: pre-existing defect this slice does not touch and every row below inherits.
#:
#:   DGFA_MUT                  killed   what dies
#:   <none>                         0   290 collected, 289 pass, 1 pre-existing red
#:   fb_linear                      7   assert 1 + all 5 grid cells + the σ=45 arm
#:   logZ_at_linear                 6   assert 2 + all 5 grid cells
#:   marginals_old                  6   assert 3 + all 5 grid cells
#:   chain_old                      6   assert 4 + all 5 grid cells
#:   tree_old                       1   assert 5 (the tree runs only at L = 32)
#:   ca_old                         1   assert 6, the mixed-polarity tail consumer
#:   map_floor_1e300                2   both MAP-floor tests
#:   tiebreak_edge_index            1   the tie-break test
#:   underflow_always_false         1   the up_sweep flush test
#:
#: **The five entries a review battery found at zero kills** — `logZ_at_linear`,
#: `marginals_old`, `chain_old`, `tree_old`, `ca_old` — are all closed, and all
#: five by asserts 2-6 on the single `unscored_tail_fixture`. None of them is an
#: equivalent mutant: handed the *fixed* `forward_backward`, every one still
#: raises `EmptyLanguageError` on a provably non-empty language, because the
#: linear `a`/`b` remain lossy by construction. An earlier version of this file
#: asserted only `math.isfinite(fb.log_Z)` and would have signed off a repair
#: that fixed the partition function and left every consumer reading `fb.a`.


def _head_module(relpath: str):
    """`git show HEAD:<relpath>` exec'd into a fresh module namespace.

    Read-only — no `checkout`, `restore`, `stash` or `reset` is involved, and
    the working tree is untouched.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = subprocess.run(["git", "show", f"HEAD:{relpath}"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    name = "_head_" + relpath.replace("/", "_").removesuffix(".py")
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(root, relpath)
    # `@dataclasses.dataclass` resolves `cls.__module__` through `sys.modules`,
    # so the fresh module has to be registered before the source is exec'd.
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _apply_mutation(spec: str) -> None:
    """`DGFA_MUT=a` or `DGFA_MUT=a,b` — applied left to right."""
    for name in spec.split(","):
        if name not in _MUTANTS:
            raise SystemExit(
                f"unknown DGFA_MUT={name!r}; known: {sorted(_MUTANTS)}")

        if name in ("fb_linear", "logZ_at_linear", "marginals_old", "chain_old",
                    "tree_old", "ca_old", "map_floor_1e300"):
            head = _head_module("diffgemma_fa/infer/reference.py")
            if name == "fb_linear":
                old_fb = head.forward_backward

                def _fb(M, a_start, b_final, *, scaled=True, _o=old_fb):
                    r = _o(M, a_start, b_final, scaled=scaled)
                    # Re-wrap into the CURRENT dataclass with no log fields, so
                    # `la`/`lb` reconstruct from the lossy linear vectors.
                    return R.ForwardBackward(
                        a=r.a, b=r.b, log_scale_a=r.log_scale_a,
                        log_scale_b=r.log_scale_b, log_Z=r.log_Z,
                        scaled=r.scaled, log_a=None, log_b=None)
                R.forward_backward = _fb
            elif name == "logZ_at_linear":
                def _log_Z_at(self, i, _R=R):
                    dot = float(self.a[i] @ self.b[i])
                    if dot <= 0.0:
                        return -np.inf
                    return float(np.log(dot) + self.log_scale_a[i]
                                 + self.log_scale_b[i])
                R.ForwardBackward.log_Z_at = _log_Z_at
            else:
                attr = {"marginals_old": "marginals",
                        "chain_old": "sample_chain",
                        "tree_old": "sample_tree",
                        "ca_old": "marginals_complement_aware",
                        "map_floor_1e300": "map_decode"}[name]
                setattr(R, attr, getattr(head, attr))

        elif name == "tiebreak_edge_index":
            head = _head_module("diffgemma_fa/infer/tree.py")
            tree.map_states_and_tokens = head.map_states_and_tokens

        elif name == "underflow_always_false":
            real = scans.up_sweep

            def _up(M, *, normalize=True, _r=real):
                t = _r(M, normalize=normalize)
                return scans.TreeLevels(
                    levels=t.levels, log_scales=t.log_scales,
                    underflow=jnp.zeros((M.shape[0],), dtype=bool))
            scans.up_sweep = _up


if os.environ.get("DGFA_MUT"):
    _apply_mutation(os.environ["DGFA_MUT"])
    print(f"\n*** DGFA_MUT={os.environ['DGFA_MUT']} ACTIVE — "
          f"{_MUTANTS[os.environ['DGFA_MUT'].split(',')[0]]} ***")


# ===========================================================================
# The independent oracle. Shares no code and no arithmetic with reference.py.
# ===========================================================================

def _multiplicity_by_end_state(
    edges: Sequence[tuple[int, int, frozenset[int]]],
    n_states: int,
    x: Sequence[int],
) -> dict[int, dict[int, int]]:
    """`{s_0: {s_L: #paths}}` for the string `x`, as exact integers.

    A memoized DFS over edges — SPEC §2.2's graphical model read literally.
    Deliberately *not* a matrix power: `reference.accepting_path_count` is a
    product of `label_count_matrix`es, and an oracle that reused that would be
    checking the implementation against itself.
    """
    L = len(x)
    memo: dict[tuple[int, int], dict[int, int]] = {}

    def go(s: int, i: int) -> dict[int, int]:
        if i == L:
            return {s: 1}
        key = (s, i)
        if key in memo:
            return memo[key]
        acc: dict[int, int] = {}
        for src, dst, labels in edges:
            if src == s and x[i] in labels:
                for end, c in go(dst, i + 1).items():
                    acc[end] = acc.get(end, 0) + c
        memo[key] = acc
        return acc

    return {s0: go(s0, 0) for s0 in range(n_states)}


class _ExactOracle:
    """The exact constrained posterior of SPEC §2.2/§2.3, in `Fraction`.

        w(x) = [Σ_{s_0,s_L} a_start(s_0)·#paths(s_0 →x→ s_L)·b_L(s_L)] · Π_i p_i(x_i)
        P(x) = w(x) / Z ,   Z = Σ_x w(x)

    Exact: `Fraction(float)` is lossless, and nothing here rounds. `Z`, the
    per-position marginals and the MAP argmax are therefore ground truth in the
    strict sense, not a better-conditioned approximation.
    """

    def __init__(self, p: np.ndarray, automaton: R.Automaton,
                 b_final: np.ndarray | None = None):
        self.L, self.V = p.shape
        self.p = [[Fraction(float(v)) for v in row] for row in p]
        self.aut = automaton
        b = automaton.final_vector() if b_final is None else b_final
        self.b = [Fraction(float(v)) for v in b]
        self.a0 = [Fraction(float(v)) for v in automaton.start]
        self.weights: dict[tuple[int, ...], Fraction] = {}
        self.Z = Fraction(0)
        for x in itertools.product(range(self.V), repeat=self.L):
            mult = self.path_weight(x)
            if mult == 0:
                continue
            w = mult
            for i, v in enumerate(x):
                w *= self.p[i][v]
                if w == 0:
                    break
            if w != 0:
                self.weights[x] = w
                self.Z += w

    def path_weight(self, x: Sequence[int]) -> Fraction:
        """`a_start^T · (path multiplicity) · b_L`, exactly."""
        table = _multiplicity_by_end_state(self.aut.edges, self.aut.n_states, x)
        tot = Fraction(0)
        for s0 in range(self.aut.n_states):
            if self.a0[s0] == 0:
                continue
            for sL, c in table[s0].items():
                if self.b[sL] != 0:
                    tot += self.a0[s0] * Fraction(c) * self.b[sL]
        return tot

    @property
    def nonempty(self) -> bool:
        return self.Z != 0

    def log_Z(self) -> float:
        assert self.Z != 0
        return float(Decimal(self.Z.numerator).ln() - Decimal(self.Z.denominator).ln())

    def posterior(self) -> dict[tuple[int, ...], float]:
        return {k: float(v / self.Z) for k, v in self.weights.items()}

    def marginals(self) -> np.ndarray:
        q = [[Fraction(0)] * self.V for _ in range(self.L)]
        for x, w in self.weights.items():
            for i, v in enumerate(x):
                q[i][v] += w
        return np.array([[float(c / self.Z) for c in row] for row in q])

    def map_score(self, x: Sequence[int]) -> Fraction:
        """`max over accepting paths of a_start(s_0)·b_L(s_L) · Π_i p_i(x_i)`.

        This is exactly what SPEC §2.7's max-plus recursion optimises: the start
        and terminal factors are *inside* the max, so a weighted `a_start` or a
        budget-aware `b_L` changes the answer. Path **multiplicity** is not
        inside it — a max over strings cannot see it, which is §2.7's "MAP is
        exact on NFAs too".
        """
        table = _multiplicity_by_end_state(self.aut.edges, self.aut.n_states, x)
        best_end = Fraction(0)
        for s0 in range(self.aut.n_states):
            if self.a0[s0] == 0:
                continue
            for sL, c in table[s0].items():
                if c and self.b[sL] != 0:
                    best_end = max(best_end, self.a0[s0] * self.b[sL])
        if best_end == 0:
            return Fraction(0)
        sc = best_end
        for i, v in enumerate(x):
            sc *= self.p[i][v]
        return sc

    def map_string(self) -> tuple[tuple[int, ...], Fraction]:
        best_score, best_x = Fraction(0), None
        for x in self.weights:
            sc = self.map_score(x)
            if sc > best_score:
                best_score, best_x = sc, x
        return best_x, best_score


def _decimal_log_partition(M: np.ndarray, a_start: np.ndarray,
                           b_final: np.ndarray) -> float:
    """`log Z` from a 60-digit `decimal` chain with **no scaling at all**.

    Unbounded exponent, so the underflow that forces SPEC §2.4's scaling on
    float64 simply cannot happen here. This is the reviewer-endorsed oracle for
    §2.4 at lengths where enumeration is out of reach.
    """
    L, S, _ = M.shape
    a = [Decimal(float(x)) for x in a_start]
    for i in range(L):
        col = [Decimal(0)] * S
        for s in range(S):
            if a[s] == 0:
                continue
            for t in range(S):
                m = M[i, s, t]
                if m != 0.0:
                    col[t] += a[s] * Decimal(float(m))
        a = col
    Z = sum((a[s] * Decimal(float(b_final[s])) for s in range(S)), Decimal(0))
    if Z == 0:
        return float("-inf")
    return float(Z.ln())


def _lse(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x > -np.inf]
    if not xs:
        return -np.inf
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _float64_log_chain(M: np.ndarray, a_start: np.ndarray,
                       b_final: np.ndarray) -> float:
    """`log Z` by a plain logsumexp chain. Independent of `scans.py`."""
    L, S, _ = M.shape
    logM = np.where(M > 0, np.log(np.maximum(M, 1e-320)), -np.inf)
    la = np.where(a_start > 0, np.log(np.maximum(a_start, 1e-320)), -np.inf)
    for i in range(L):
        la = np.array([_lse([la[s] + logM[i, s, t] for s in range(S)])
                       for t in range(S)])
    return _lse([la[s] + (0.0 if b_final[s] > 0 else -np.inf) for s in range(S)])


# ===========================================================================
# Generators. Written here rather than imported from another test file, so a
# generator bug cannot be inherited silently.
# ===========================================================================

def flat_p(rng: np.random.Generator, L: int, vocab: int = V) -> np.ndarray:
    p = rng.random((L, vocab)) + 0.05
    return p / p.sum(axis=1, keepdims=True)


def sharp_p(rng: np.random.Generator, L: int, sigma: float,
            vocab: int = V) -> np.ndarray:
    """`softmax(N(0,1)·σ)`. σ = 25 puts the smallest entry near 1e-30 — the
    order the released checkpoint actually produces (SPEC §4.4, Phase 0)."""
    z = rng.standard_normal((L, vocab)) * sigma
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def random_dfa(rng: np.random.Generator, n_states: int,
               vocab: int = V) -> R.Automaton:
    by_pair: dict[tuple[int, int], set[int]] = {}
    for s in range(n_states):
        for v in range(vocab):
            if rng.random() < 0.65:
                by_pair.setdefault((s, int(rng.integers(n_states))), set()).add(v)
    edges = tuple((s, d, frozenset(vs)) for (s, d), vs in sorted(by_pair.items()))
    start = np.zeros(n_states)
    start[0] = 1.0
    finals = frozenset(s for s in range(n_states) if rng.random() < 0.4) \
        or frozenset({n_states - 1})
    return R.Automaton(n_states=n_states, vocab_size=vocab, edges=edges,
                       start=start, finals=finals)


def random_nfa(rng: np.random.Generator, n_states: int,
               vocab: int = V) -> R.Automaton:
    """An NFA with **parallel, distinct-but-overlapping** labels on one pair.

    An exact duplicate edge would not do: duplicating a label set scales `M`
    uniformly and the scale cancels in normalisation, so eq (8)'s multiplicity
    becomes unobservable. Distinct label sets sharing a token are what make the
    weighted and `∃` forms disagree.
    """
    edges: list[tuple[int, int, frozenset[int]]] = []
    for _ in range(int(rng.integers(n_states, 3 * n_states + 1))):
        s, d = int(rng.integers(n_states)), int(rng.integers(n_states))
        k = int(rng.integers(1, vocab + 1))
        edges.append((s, d, frozenset(int(x) for x in
                                      rng.choice(vocab, size=k, replace=False))))
    s0, d0, lab0 = edges[0]
    rest = set(range(vocab)) - set(lab0)
    if lab0 and rest:
        edges.append((s0, d0, frozenset({sorted(lab0)[0], sorted(rest)[0]})))
    start = np.zeros(n_states)
    start[0] = 1.0
    finals = frozenset(s for s in range(n_states) if rng.random() < 0.4) \
        or frozenset({n_states - 1})
    return R.Automaton(n_states=n_states, vocab_size=vocab,
                       edges=tuple(edges), start=start, finals=finals)


MAKERS = {"dfa": random_dfa, "nfa": random_nfa}


def instance(seed: int, kind: str, L: int, *, sigma: float | None = None,
             start_kind: str = "point", budget: int | None = None):
    """One fully-specified instance. **Every regime axis is a named argument**,
    so the regime is readable beside the assertion rather than buried."""
    rng = np.random.default_rng(seed + SEED_OFFSET)
    A = MAKERS[kind](rng, int(rng.integers(3, 6)))
    if start_kind != "point":
        st = (rng.random(A.n_states) < 0.5).astype(np.float64)
        if st.sum() == 0:
            st[0] = 1.0
        if start_kind == "weighted":
            st = st * (rng.random(A.n_states) + 0.1)
        A = R.Automaton(n_states=A.n_states, vocab_size=A.vocab_size,
                        edges=A.edges, start=st, finals=A.finals)
    p = flat_p(rng, L) if sigma is None else sharp_p(rng, L, sigma)
    b = A.final_vector() if budget is None else A.budget_vector(budget)
    return A, p, b


def nondegenerate(seed: int, kind: str, L: int, *, tries: int = 60,
                  require_negated: bool = False, **kw):
    """An instance whose exact language is non-empty and has ≥ 2 strings.

    `require_negated` **searches** for an automaton with at least one negated
    class rather than skipping when the draw happens not to produce one. A
    `pytest.skip` there would silently cap the coverage of the complement-aware
    path, which is the one SPEC §2.4 says only this kind of suite can check.
    """
    for k in range(tries):
        A, p, b = instance(seed * 977 + k, kind, L, **kw)
        if require_negated and not class_tables(A, threshold=2)[2].any():
            continue
        oracle = _ExactOracle(p, A, b)
        if oracle.nonempty and len(oracle.weights) >= 2:
            return A, p, b, oracle
    raise AssertionError(
        f"no non-degenerate instance in {tries} tries at seed={seed} kind={kind} "
        f"L={L} require_negated={require_negated} {kw}; the generator, not the "
        f"assertion, needs fixing")


def prepared(A: R.Automaton, p: np.ndarray, b: np.ndarray):
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    fb = R.forward_backward(M, A.start, b)
    return W, M, fb


def chi2_p(samples: list[tuple[int, ...]],
           posterior: dict[tuple[int, ...], float]) -> tuple[float, int]:
    """Pearson χ² p-value, plus the count of draws **outside the support**.

    Support violations are returned rather than pooled: a sampler that emits an
    inadmissible string is not a distributional near-miss, and χ² would dilute
    it.
    """
    n = len(samples)
    counts: dict[tuple[int, ...], int] = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    outside = sum(c for k, c in counts.items() if k not in posterior)
    obs, exp, pooled_o, pooled_e = [], [], 0.0, 0.0
    for k in sorted(posterior):
        e = posterior[k] * n
        if e >= 5.0:
            obs.append(counts.get(k, 0))
            exp.append(e)
        else:
            pooled_o += counts.get(k, 0)
            pooled_e += e
    if pooled_e > 0:
        obs.append(pooled_o)
        exp.append(pooled_e)
    if len(obs) < 2:
        return 1.0, outside
    o = np.asarray(obs, dtype=np.float64)
    e = np.asarray(exp, dtype=np.float64)
    e *= o.sum() / e.sum()
    return float(stats.chisquare(o, e).pvalue), outside


def class_tables(A: R.Automaton, threshold: int | None = None):
    """Intern edge labels into SPEC §4.4's class CSR.

    `threshold` is the polarity rule: a class with `|S_c| > threshold` is stored
    as its complement `N_c`. SPEC §6.1 test 11 prescribes `|S_c| > 2` at `V = 4`
    precisely so negated classes are *forced* rather than left to chance.
    """
    vocab = A.vocab_size
    threshold = (vocab // 2) if threshold is None else threshold
    lookup: dict[frozenset[int], int] = {}
    members: list[frozenset[int]] = []
    class_of: list[int] = []
    for _, _, lab in A.edges:
        if lab not in lookup:
            lookup[lab] = len(members)
            members.append(lab)
        class_of.append(lookup[lab])
    is_neg = [len(m) > threshold for m in members]
    stored = [sorted(set(range(vocab)) - set(m)) if is_neg[c] else sorted(m)
              for c, m in enumerate(members)]
    indptr = np.zeros(len(members) + 1, np.int32)
    for c, s in enumerate(stored):
        indptr[c + 1] = indptr[c] + len(s)
    indices = np.array([x for s in stored for x in s], np.int32)
    return (np.asarray(class_of, np.int32), members, np.asarray(is_neg),
            indices, indptr)


# ===========================================================================
# Part 0 — the oracle is not the implementation
# ===========================================================================

def test_the_oracle_is_not_the_implementation():
    """Guard against the failure mode that produced five vacuous tests here.

    A test that re-implements the thing under test proves nothing. So: assert
    the oracle **disagrees** with a deliberately wrong posterior — one that
    drops eq (8)'s edge multiplicity and treats every accepted string as equally
    weighted by its `p`-product alone. On an NFA with parallel overlapping edges
    SPEC §2.2 says these are genuinely different distributions; if the oracle
    could not tell them apart it would be measuring nothing.
    """
    A = R.Automaton(
        n_states=3, vocab_size=V,
        edges=((0, 1, frozenset({0, 1})), (0, 1, frozenset({1, 2})),
               (1, 2, frozenset({0, 1, 2, 3}))),
        start=np.array([1.0, 0.0, 0.0]), finals=frozenset({2}))
    p = np.array([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    oracle = _ExactOracle(p, A)
    assert oracle.nonempty

    # Token 1 lies on BOTH parallel edges, so its multiplicity is 2.
    assert oracle.path_weight((1, 0)) == 2, "the oracle lost the parallel edge"
    assert oracle.path_weight((0, 0)) == 1
    assert oracle.path_weight((2, 0)) == 1

    unweighted = {x: float(np.prod([p[i, v] for i, v in enumerate(x)]))
                  for x in oracle.weights}
    tot = sum(unweighted.values())
    unweighted = {k: v / tot for k, v in unweighted.items()}
    true = oracle.posterior()
    tv = 0.5 * sum(abs(true[k] - unweighted[k]) for k in true)
    assert tv > 1e-2, (
        f"the oracle cannot distinguish the path-weighted posterior from the "
        f"unweighted one (TV = {tv:.2e}); it is not measuring SPEC §2.2")


def test_the_decimal_oracle_disagrees_with_an_unscaled_float64_chain():
    """The `decimal` oracle must be able to see what float64 cannot.

    Otherwise it is a slower copy of the thing it certifies. At `L = 32` with
    `σ = 25` marginals the unscaled float64 product underflows to exactly zero
    (SPEC §6.3) while the 60-digit unbounded-exponent chain returns a finite
    `log Z`.
    """
    L = 32
    for attempt in range(40):
        rng = np.random.default_rng(4242 + attempt + SEED_OFFSET)
        A = random_dfa(rng, 4)
        p = sharp_p(rng, L, 45.0)
        W = R.edge_weights(p, A)
        M = R.transition_matrices(W, A)
        b = A.final_vector()
        exact = _decimal_log_partition(M, A.start, b)
        if exact == float("-inf"):
            continue
        naive = A.start.astype(np.float64).copy()
        for i in range(L):
            naive = naive @ M[i]
        if float(naive @ b) == 0.0:
            assert math.isfinite(exact), "decimal chain lost a non-empty language"
            assert exact < math.log(np.finfo(np.float64).tiny), (
                f"log Z = {exact} is representable, so no separation was shown")
            return
    pytest.fail(
        "no instance in 40 tries where the unscaled float64 chain underflows "
        "while the language is non-empty — the decimal oracle has not been "
        "shown to see anything float64 cannot")


# ===========================================================================
# Part 1 — infer/reference.py against the exact rational oracle
# ===========================================================================

REGIMES = [
    # (kind,  L, sigma, start_kind, budget)   -- the whole regime, stated.
    ("dfa", 4, None, "point", None),
    ("nfa", 4, None, "point", None),
    ("dfa", 6, None, "point", None),
    ("nfa", 6, None, "point", None),
    ("nfa", 5, 12.0, "point", None),
    ("dfa", 5, 25.0, "point", None),
    ("nfa", 4, 40.0, "point", None),
    ("nfa", 5, None, "set", None),
    ("dfa", 5, None, "weighted", None),
    ("nfa", 4, 12.0, "weighted", None),
    ("dfa", 4, None, "point", 0),
    ("nfa", 4, None, "point", 1),
    ("nfa", 5, 12.0, "set", 2),
    ("dfa", 6, 25.0, "point", 3),
]
REGIME_IDS = [f"{k}-L{L}-sig{s}-{st}-R{b}" for k, L, s, st, b in REGIMES]


@pytest.mark.parametrize("kind,L,sigma,start_kind,budget", REGIMES, ids=REGIME_IDS)
@pytest.mark.parametrize("seed", range(3))
def test_reference_partition_matches_the_exact_rational_Z(
        kind, L, sigma, start_kind, budget, seed):
    """SPEC §2.4: `Z = a_0 M_1 ⋯ M_L b_L = Σ_{x ∈ C} Π_i p_i(x_i)`.

    Checked against exact rational enumeration, not against
    `reference.enumerate_posterior`.
    """
    A, p, b, oracle = nondegenerate(seed, kind, L, sigma=sigma,
                                    start_kind=start_kind, budget=budget)
    _, _, fb = prepared(A, p, b)
    exact = oracle.log_Z()
    assert fb.log_Z == pytest.approx(exact, rel=1e-12, abs=1e-12), (
        f"log Z: got {fb.log_Z!r}, exact {exact!r} "
        f"(regime: {kind} L={L} sigma={sigma} start={start_kind} R={budget})")


@pytest.mark.parametrize("kind,L,sigma,start_kind,budget", REGIMES, ids=REGIME_IDS)
@pytest.mark.parametrize("seed", range(2))
def test_reference_marginals_match_the_exact_rational_marginals(
        kind, L, sigma, start_kind, budget, seed):
    """SPEC eq (5)/(6): `q_i(v) = p_i(v)·(Σ_{e : v ∈ label(e)} u_i(e))/Z`.

    An `a`/`b` off-by-one is invisible in `Σ_v q_i(v) == 1` (the row is divided
    by its own sum) and invisible in `Z`; it shows up here and only here.
    """
    A, p, b, oracle = nondegenerate(seed, kind, L, sigma=sigma,
                                    start_kind=start_kind, budget=budget)
    W, _, fb = prepared(A, p, b)
    q = R.marginals(p, A, fb, W)
    exact = oracle.marginals()
    assert np.allclose(q, exact, rtol=1e-11, atol=1e-13), (
        f"max |q - exact| = {np.max(np.abs(q - exact)):.3e} "
        f"(regime: {kind} L={L} sigma={sigma} start={start_kind} R={budget})")
    # And the support is exactly the automaton's, with no smearing.
    assert np.array_equal(q > 0, exact > 0)


@pytest.mark.parametrize("kind,L,sigma,start_kind,budget", REGIMES, ids=REGIME_IDS)
@pytest.mark.parametrize("seed", range(2))
def test_reference_enumerate_posterior_is_itself_correct(
        kind, L, sigma, start_kind, budget, seed):
    """**The load-bearing test in this file.**

    `tests/test_exactness.py` — CLAUDE.md's "bedrock", the thing standing
    between the project and a sampler drawing from the wrong distribution —
    checks every other function in `reference.py` against
    `reference.enumerate_posterior`. That oracle has never been checked against
    anything. If it is wrong, the bedrock is wrong and so is every "verified"
    result that rests on it.
    """
    A, p, b, oracle = nondegenerate(seed, kind, L, sigma=sigma,
                                    start_kind=start_kind, budget=budget)
    post, Z = R.enumerate_posterior(p, A, b)
    assert set(post) == set(oracle.weights), (
        f"support differs: reference has {len(post)}, exact has "
        f"{len(oracle.weights)}; symmetric difference "
        f"{sorted(set(post) ^ set(oracle.weights))[:5]}")
    assert math.log(Z) == pytest.approx(oracle.log_Z(), rel=1e-12, abs=1e-12)
    truth = oracle.posterior()
    worst = max(abs(post[k] - truth[k]) for k in truth)
    assert worst < 1e-12, f"worst |P - P_exact| = {worst:.3e}"


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(4))
def test_reference_accepting_path_count_matches_a_dfs_path_count(kind, seed):
    """SPEC §2.2: on an NFA the weight of a string is its **number of accepting
    paths**. `accepting_path_count` gets it from a product of
    `label_count_matrix`es; the oracle gets it from a DFS over edges."""
    A, p, b, _ = nondegenerate(seed, kind, 4)
    oracle = _ExactOracle(p, A, b)
    for x in itertools.product(range(V), repeat=4):
        got = R.accepting_path_count(A, x, b)
        exact = oracle.path_weight(x)
        assert got == pytest.approx(float(exact), rel=1e-12, abs=1e-12), \
            f"path count for {x}: got {got}, exact {float(exact)}"


@pytest.mark.parametrize("sigma", [1.0, 8.0, 20.0, 45.0])
@pytest.mark.parametrize("seed", range(4))
def test_reference_forward_backward_matches_a_60_digit_decimal_chain(sigma, seed):
    """SPEC §2.4 beyond the enumerable regime: `L = 32`, up to 8 states.

    The `decimal` chain has unbounded exponent and does **no scaling**, so it
    isolates exactly the thing `scaled=True` exists to protect: whether the
    max-normalisation plus accumulated log-scales reproduces the true `log Z`.

    `σ = 45` is the arm that makes the scaling load-bearing rather than
    decorative: at that sharpness the unscaled float64 product underflows to
    exactly zero at `L = 32` (asserted in
    `test_the_decimal_oracle_disagrees_with_an_unscaled_float64_chain`), so a
    reference that quietly stopped normalising would return `-inf` here. The
    milder σ arms cannot see that and are kept only as controls.

    **Cross-reference:** the `σ = 45` arm also surfaces, on some seeds, the
    per-vector-normalisation defect that
    `test_reference_forward_backward_survives_an_unscored_tail_beside_a_live_chain`
    pins deterministically. If both are red, they are one defect, not two: fix
    the deterministic one first.
    """
    L = 32
    # The σ = 45 arm additionally requires an instance on which the *unscaled*
    # float64 product underflows to exactly zero — otherwise the scaling is not
    # load-bearing on that instance and the arm degenerates into another control.
    require_underflow = sigma >= 40.0
    for attempt in range(200):
        rng = np.random.default_rng(31000 + seed * 7 + int(sigma) * 101
                                    + attempt * 7919 + SEED_OFFSET)
        A = (random_dfa if seed % 2 else random_nfa)(rng, int(rng.integers(3, 9)))
        p = sharp_p(rng, L, sigma)
        W = R.edge_weights(p, A)
        M = R.transition_matrices(W, A)
        b = A.final_vector()
        exact = _decimal_log_partition(M, A.start, b)
        if exact == float("-inf"):
            continue
        if require_underflow:
            naive = A.start.astype(np.float64).copy()
            for i in range(L):
                naive = naive @ M[i]
            if float(naive @ b) != 0.0:
                continue
        break
    else:
        raise AssertionError(
            f"no length-32 instance in 200 tries with a non-empty language"
            f"{' that underflows unscaled float64' if require_underflow else ''}; "
            f"the generator, not the assertion, needs fixing")
    fb = R.forward_backward(M, A.start, b)
    assert fb.log_Z == pytest.approx(exact, rel=1e-12), (
        f"log Z at L=32, sigma={sigma}, S={A.n_states}: got {fb.log_Z!r}, "
        f"60-digit decimal chain {exact!r}"
        f"{'; the unscaled product underflows on this instance, so the '
           'accumulated log-scales are the only thing that can recover it'
           if require_underflow else ''}")


def unscored_tail_fixture(L: int = 32, nats: float = 30.0):
    """SPEC §3.5's shape as a **real `Automaton` plus `p`**, not a raw `M`.

    One unscored `ACC --Σ--> ACC` tail — the edge carries the *whole* alphabet,
    so `W[i, e] = Σ_v p_i(v) = 1.0` exactly, which is what §2.6 `[V-P4]` means
    by "the root's max is pinned at 1.0 by construction" — beside a live chain
    whose every edge carries the single token `1` at `p_i(1) = e^{-nats}`.

    Building it this way rather than as a hand-written `M` is what lets the
    whole consumer chain be exercised on one fixture: `marginals`,
    `sample_chain` and `sample_tree` all need the automaton and `p`, not just
    the transition matrices.

    The tail state 3 is **unreachable from the start**, so:
      * the only accepted strings are `1^L`, and every accepted path leaves
        state 0 exactly once;
      * `u_i(e) = 0` on the tail edge at every `i`, so `q_i` must be the point
        mass on token 1 — an exact, checkable claim rather than a row-sum
        identity that holds by construction.

    Returns `(automaton, p, b_final)`.
    """
    step = math.exp(-nats)
    A = R.Automaton(
        n_states=4, vocab_size=V,
        edges=(
            (3, 3, frozenset(range(V))),      # ACC --Σ--> ACC, mass exactly 1.0
            (0, 0, frozenset({1})),           # the live chain, e^-nats a token
            (0, 1, frozenset({1})),
            (1, 2, frozenset({1})),
            (2, 2, frozenset({1})),
        ),
        start=np.array([1.0, 0.0, 0.0, 0.0]),
        finals=frozenset({2, 3}))
    p = np.tile(np.array([1.0 - 3.0 * step, step, step, step]), (L, 1))
    return A, p, A.final_vector()


def test_reference_forward_backward_survives_an_unscored_tail_beside_a_live_chain():
    """`forward_backward` and **every consumer that reads it** must survive
    SPEC §3.5's unscored tail beside a live chain.

    `forward_backward`'s docstring used to say `scaled=True` was *"Mandatory in
    fp32 at `L = 256`, where the unnormalized product underflows to exactly
    zero"* — i.e. the scaling made `log Z` recoverable. It did not, for the
    reason `scans.prefix_suffix`' docstring already recorded against the
    identical construction (commit `2c9f9d9`): a **per-vector max** bounds a
    vector's maximum, not its internal dynamic range. The tail's emission mass
    is exactly 1.0, so it pins `max_s b_i(s) = 1` while the chain sits at
    `e^{-30(L-i)}`; at `L = 32` the start state's `b_0` divided straight to
    `0.0`, `log Z` came back `-inf`, and `marginals` / `sample_chain` /
    `sample_tree` all raised `EmptyLanguageError` on a language whose true
    `log Z` is `-956.566` — CLAUDE.md's cause **(c)** delivered to the caller as
    **(a)/(b)**, the misclassification the taxonomy exists to prevent.

    **This test asserts the whole consumer chain, not just `log_Z`.** A review
    battery that spliced the exact pre-slice functions back in
    (`git show HEAD:`) found that an earlier version of this test, which
    asserted only `math.isfinite(fb.log_Z)`, killed `fb_linear` but left
    `logZ_at_linear`, `marginals_old`, `chain_old`, `tree_old` and `ca_old`
    **all at zero kills** — and none of them is an equivalent mutant: handed the
    *fixed* `forward_backward`, every one still raises `EmptyLanguageError` here,
    because the linear `a`/`b` remain lossy by construction. Fixing the
    partition function and leaving the consumers reading `fb.a`/`fb.b` would
    have passed. Hence `_MUTANTS` below, and hence the six asserts.

    **Rate on random instances**, so the blast radius is stated rather than
    implied: with `V = 4`, `S ∈ [3,8]` automata and `p = softmax(N(0,1)·σ)`, the
    pre-slice `forward_backward` reported a non-empty language empty in 0/33
    draws at `L = 32, σ = 45`, 1/35 at `L = 64, σ = 20`, 1/32 at `L = 256,
    σ = 10` and 2/31 at `L = 256, σ = 45`. A structural corner, not a common
    one — but the corner is SPEC §3.5's *mandatory* unscored tail.
    """
    L = 32
    A, p, b = unscored_tail_fixture(L)
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)

    # --- fixture checks: the shape is the one the docstring claims ----------
    assert W[0, 0] == pytest.approx(1.0, abs=0.0), (
        "the tail edge must carry emission mass exactly 1.0 — that is what "
        "pins the per-vector maximum and makes the normalisation lossy")
    assert M[0].max() == 1.0
    assert np.all(np.isfinite(M))
    assert R.simulate(A, [1] * L) & A.finals, "fixture check: 1^L is accepted"
    assert 3 not in R.simulate(A, [1] * L), (
        "fixture check: the tail must be unreachable from the start, or `q` is "
        "not the point mass this test asserts")

    exact = _decimal_log_partition(M, A.start, b)
    assert math.isfinite(exact) and exact < -900, (
        f"fixture check: the language must be non-empty and deep; log Z = {exact}")

    # --- 1. the partition function -----------------------------------------
    fb = R.forward_backward(M, A.start, b, scaled=True)
    assert math.isfinite(fb.log_Z), (
        f"the language is non-empty (60-digit decimal chain: log Z = "
        f"{exact:.4f}) but the scaled reference forward-backward returned "
        f"log_Z = {fb.log_Z!r}.")
    assert fb.log_Z == pytest.approx(exact, rel=1e-9)

    # --- 2. log_Z_at(i), which reads the vectors rather than the scalar -----
    # SPEC §2.4 [D] requires i-invariance; the point here is that the *linear*
    # dot `a_i · b_i` cannot hold this instance at any single pair of scales, so
    # an accessor built on it returns -inf at interior boundaries while `log_Z`
    # itself is fine. Every boundary is checked, not just one.
    at = [fb.log_Z_at(i) for i in range(L + 1)]
    bad = [i for i, v in enumerate(at) if not math.isfinite(v)]
    assert not bad, (
        f"log_Z_at returned non-finite at boundaries {bad[:6]} while log_Z = "
        f"{fb.log_Z:.4f}; the accessor is reading a representation that cannot "
        f"hold this instance")
    assert max(at) - min(at) < 1e-6, f"log Z varies with i: {at}"
    for i, v in enumerate(at):
        assert v == pytest.approx(exact, rel=1e-9), \
            f"log_Z_at({i}) = {v!r}, 60-digit decimal chain {exact!r}"

    # --- 3. marginals: finite, normalised, and the exact point mass ---------
    q = R.marginals(p, A, fb, W)
    assert np.all(np.isfinite(q)), "q contains NaN or inf"
    assert np.allclose(q.sum(axis=1), 1.0, rtol=0, atol=1e-12), \
        f"rows do not sum to 1: {q.sum(axis=1)}"
    # Non-vacuous: the row sum is 1 by construction (q is divided by its own
    # total), so the *support* is what carries the information. Only token 1 is
    # on any edge reachable from the start, so q must be its point mass.
    expect = np.zeros((L, V))
    expect[:, 1] = 1.0
    assert np.allclose(q, expect, rtol=0, atol=1e-12), (
        f"q is not the point mass on token 1; worst deviation "
        f"{np.max(np.abs(q - expect)):.3e} at position "
        f"{int(np.argmax(np.abs(q - expect)) // V)}")

    # --- 4. the chain sampler ----------------------------------------------
    rng = np.random.default_rng(20260812 + SEED_OFFSET)
    for _ in range(8):
        toks, states = R.sample_chain(p, A, fb, W, rng)
        got = tuple(int(t) for t in toks)
        assert R.accepts(A, got), f"sample_chain emitted an unaccepted string {got}"
        assert got == (1,) * L, f"the only accepted string is 1^{L}, got {got}"
        assert b[states[-1]] > 0

    # --- 5. the tree sampler -----------------------------------------------
    for _ in range(8):
        toks, states = R.sample_tree(p, A, M, W, A.start, b, rng)
        got = tuple(int(t) for t in toks)
        assert R.accepts(A, got), f"sample_tree emitted an unaccepted string {got}"
        assert got == (1,) * L, f"the only accepted string is 1^{L}, got {got}"
        assert b[states[-1]] > 0

    # --- 6. the complement-aware marginals ---------------------------------
    # Same claim as assert 3, through SPEC §2.4's class representation. It is a
    # separate function reading the same `fb`, so fixing `marginals` alone
    # leaves it broken — and with the polarity threshold at `|S_c| > 2` the
    # tail's `Σ` class (|S_c| = 4) is stored **negated** while the chain's `{1}`
    # is positive, which is the mixed-polarity case §2.4 says only this kind of
    # suite catches.
    cid, members, is_neg, _, _ = class_tables(A, threshold=2)
    assert is_neg[cid[0]] and not is_neg[cid[1]], (
        "fixture check: the tail class must be stored negated and the chain "
        "class positive, or this assert does not exercise mixed polarity")
    q_ca = R.marginals_complement_aware(p, A, fb, list(cid), members,
                                        list(is_neg))
    assert np.all(np.isfinite(q_ca)), "complement-aware q contains NaN or inf"
    assert np.allclose(q_ca, expect, rtol=0, atol=1e-12), (
        f"complement-aware q is not the point mass on token 1; worst deviation "
        f"{np.max(np.abs(q_ca - expect)):.3e}")

    # --- and MAP, which SPEC §2.7 already put in log space ------------------
    map_toks, _, score = R.map_decode(p, A, A.start, b)
    assert tuple(int(t) for t in map_toks) == (1,) * L
    assert score == pytest.approx(-30.0 * L, rel=1e-12)


@pytest.mark.parametrize("L,nats", [(32, 30.0), (64, 20.0), (128, 10.0),
                                    (256, 10.0), (256, 30.0)])
def test_reference_consumers_survive_the_tail_across_the_measured_grid(L, nats):
    """The same fixture over the `(L, nats)` grid on which the pre-slice
    functions were measured to fail, so the fix is pinned where it was broken
    rather than only at one point.

    Kept separate from the test above — that one is the detailed, deterministic
    account at a single point; this one is the coverage. Only the cheap
    consumers run here (`log_Z`, `log_Z_at`, `marginals`, `sample_chain`),
    because `sample_tree` builds `O(L)` dyadic `[S,S]` log-products and the
    point of the grid is breadth, not depth.
    """
    A, p, b = unscored_tail_fixture(L, nats)
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    exact = _decimal_log_partition(M, A.start, b)
    assert math.isfinite(exact), "fixture check: the language is non-empty"

    fb = R.forward_backward(M, A.start, b)
    assert fb.log_Z == pytest.approx(exact, rel=1e-9), (
        f"log Z at L={L}, {nats} nats/token: got {fb.log_Z!r}, exact {exact!r}")
    for i in (0, L // 2, L):
        assert fb.log_Z_at(i) == pytest.approx(exact, rel=1e-9), \
            f"log_Z_at({i}) = {fb.log_Z_at(i)!r} at L={L}, {nats} nats"

    q = R.marginals(p, A, fb, W)
    assert np.all(np.isfinite(q))
    expect = np.zeros((L, V))
    expect[:, 1] = 1.0
    assert np.allclose(q, expect, rtol=0, atol=1e-12)

    toks, _ = R.sample_chain(p, A, fb, W,
                             np.random.default_rng(7 + L + SEED_OFFSET))
    assert R.accepts(A, [int(t) for t in toks])


@pytest.mark.parametrize("kind,L,sigma,start_kind,budget", REGIMES, ids=REGIME_IDS)
def test_reference_log_Z_is_i_invariant_and_equals_the_exact_value(
        kind, L, sigma, start_kind, budget):
    """SPEC §2.4 `[D]`: `log(a_i·b_i) + ls_a[i] + ls_b[i]` is **`i`-invariant**.

    `i`-invariance alone is *not* enough — a uniform scale error passes it. So
    the invariant value is also pinned to the exact `log Z`.
    """
    A, p, b, oracle = nondegenerate(0, kind, L, sigma=sigma,
                                    start_kind=start_kind, budget=budget)
    _, _, fb = prepared(A, p, b)
    exact = oracle.log_Z()
    vals = [fb.log_Z_at(i) for i in range(L + 1)]
    assert max(vals) - min(vals) < 1e-9, f"log Z varies with i: {vals}"
    for i, v in enumerate(vals):
        assert v == pytest.approx(exact, rel=1e-12, abs=1e-12), \
            f"log_Z_at({i}) = {v}, exact {exact}"


@pytest.mark.parametrize("seed", range(6))
def test_reference_scaled_and_unscaled_agree_where_unscaled_survives(seed):
    """SPEC §2.4: scaling is a numerical device, not a change of semantics.

    At `L ≤ 6` with flat `p` the unscaled product does not underflow, so both
    paths must return the same `log Z` — and both must equal the exact one.
    """
    A, p, b, oracle = nondegenerate(seed, "nfa", 6)
    _, M, _ = prepared(A, p, b)
    f1 = R.forward_backward(M, A.start, b, scaled=True)
    f0 = R.forward_backward(M, A.start, b, scaled=False)
    exact = oracle.log_Z()
    assert f0.log_Z == pytest.approx(exact, rel=1e-12, abs=1e-12)
    assert f1.log_Z == pytest.approx(exact, rel=1e-12, abs=1e-12)
    # The scaled path must actually be scaling: if every log-scale is zero the
    # test above proves nothing about the mechanism.
    assert np.any(f1.log_scale_a != 0.0) or np.any(f1.log_scale_b != 0.0), (
        "no normalisation happened, so this instance cannot separate the "
        "scaled and unscaled paths")


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("sigma", [None, 12.0, 30.0])
@pytest.mark.parametrize("seed", range(3))
def test_reference_complement_aware_marginals_match_the_exact_marginals(
        kind, sigma, seed):
    """SPEC §2.4 / §6.1 test 11 — the one path §2.4 says only this suite catches.

    Polarity threshold `|S_c| > 2` at `V = 4`, so negated classes are forced,
    with mixed Pos/Neg in one automaton and `N_c` as narrow as a single token.
    `1[v ∈ S_c] = 1 − 1[v ∈ N_c]` makes a negated class contribute *everywhere*;
    a plain sparse scatter silently produces a plausible wrong distribution.
    """
    A, p, b, oracle = nondegenerate(seed, kind, 5, sigma=sigma,
                                    require_negated=True)
    _, _, fb = prepared(A, p, b)
    class_of = list(range(A.n_edges))
    members = [lab for _, _, lab in A.edges]
    is_neg = [len(m) > 2 for m in members]
    assert any(is_neg), "regime check: the search must have found a negated class"
    narrow = [len(set(range(V)) - set(m)) for c, m in enumerate(members) if is_neg[c]]
    q = R.marginals_complement_aware(p, A, fb, class_of, members, is_neg)
    exact = oracle.marginals()
    assert np.allclose(q, exact, rtol=1e-11, atol=1e-13), (
        f"max |q - exact| = {np.max(np.abs(q - exact)):.3e}; "
        f"{sum(is_neg)}/{len(is_neg)} classes negated, narrowest |N_c| = "
        f"{min(narrow)}, sigma={sigma}")


def wide_span_p(rng: np.random.Generator, L: int, nats: float,
                vocab: int = V) -> np.ndarray:
    """`p` rows deliberately spanning `nats` between their largest and smallest
    entry — `nats = 207` is a span of `10^90`.

    Distinct from `sharp_p`: that draws logits from `N(0,1)·σ`, so the realised
    span is random and the tail of the distribution decides whether a regime is
    exercised at all. Here the span is the parameter, which is what makes "the
    whole regime, stated" possible for the arm below.
    """
    z = rng.random((L, vocab)) * nats
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PRE-EXISTING defect in the ARBITER, not a regression from this slice "
        "(`ca_old` gives identical answers). `reference.marginals_complement_"
        "aware` forms a negated class as the linear difference "
        "`sum_Neg U[c] - sum_{Neg, v in N_c} U[c]`, which cancels "
        "catastrophically as the p-row dynamic range grows.\n\n"
        "ONSET IS 69 NATS, and that is inside production reach: the shipped "
        "T = 0.4 admits 150 nats per position "
        "(`test_audit_prefix_suffix.MAX_NATS_PER_POSITION = 2*SOFTCAP/T`). "
        "Measured against the Fraction oracle -- 69 nats: 2/254 wrong at "
        "4.19e-06; 150 nats: 6/252 wrong at max err 1.000; 207 nats: 6/82 "
        "wrong, worst rel err 1.32e+10; 300 nats: 1/62 at 1.000. Plain "
        "`marginals` is exact to ~1e-15 on the same inputs, so the failure is "
        "attributable to the complement algebra and not to `forward_backward`. "
        "Quoting only the 207-nat figure would read as an unreachable corner; "
        "it is not.\n\n"
        "THE SHIPPED JAX PATH IS CLEAN AND THIS WAS MEASURED, NOT ASSUMED. "
        "`marginals.class_weights` vs the same oracle, 420 instances, mixed "
        "polarity, spans 0/69/150/207/250 nats: 0 wrong at every span, worst "
        "rel err 1.96e-16. It forms the negated class as "
        "`outside + sel @ gathered` -- a sum of NON-NEGATIVE terms, which is "
        "precisely the `da1294a` rebuild. So the blast radius is confined to "
        "the reference.\n\n"
        "REPAIR IS NOT A DESIGN CHANGE, and an earlier note in docs/LOG.md "
        "saying it needs compensated or log-space summation of the Pos/Neg "
        "algebra -- or that it changes what SPEC 4.4 Layer 2b prescribes -- is "
        "retracted. Layer 2b's kernel already uses the non-subtractive form "
        "and is exact at 250 nats. The fix is the same move this slice made "
        "twice already (prefix_suffix -> reference, and the MAP floor): make "
        "the arbiter use the form the kernel it certifies already uses.\n\n"
        "xfail(strict) rather than a standing red, per the reviewer: a "
        "permanent red in the arbiter's own suite is what trains people to "
        "stop reading red, and strict=True turns an accidental fix into a "
        "failure rather than letting it pass unnoticed."
    ),
)
def test_reference_complement_aware_marginals_survive_a_10e90_p_span():
    """**RED — `marginals_complement_aware` destroys mass by cancellation.**

    SPEC §2.4's complement-aware inner sum is evaluated as

        r_i(v) = Σ_{c ∈ Neg} U_i(c) + Σ_{c ∈ Pos, v ∈ S_c} U_i(c)
                                    − Σ_{c ∈ Neg, v ∈ N_c} U_i(c)

    and the reference implements that literally: `r += U[c]` for every negated
    class, then `r[idx] -= U[c]` on the complement each one stores. **That is
    the subtract-shaped algebra twice removed from production already.**
    `marginals.class_weights` was rebuilt to eliminate it (commit `da1294a`,
    which measured relative error 1.0 with entries lost), and
    `marginals.scatter_edge_mass_to_tokens` was rebuilt to eliminate the
    identical expression one layer down (5,013 entries destroyed on
    `github_star`, 4,033 on `uber.ride`). This is the third instance, and it is
    in the arbiter.

    **Measured, with the whole regime stated.** `V = 4`, `L = 4`, `S ∈ [3,6)`,
    DFA and NFA, polarity threshold `|S_c| > 2` so both polarities are present
    in every instance, `p` rows spanning `207` nats (`10^90`) — inside the
    checkpoint's own reachable sharpness once `_MIN_TEMP = 1e-12` is available
    by configuration (SPEC §3.9). Over 82 non-degenerate instances: **6 come
    back with relative error ≥ 1.0** against the exact rational posterior — mass
    destroyed outright, and the *support* differs by 1–2 (position, token) pairs
    — while worst-case error over the other 76 is 5.7e-14.

    **Plain `marginals` is exact on the identical inputs** (worst 5.7e-14 over
    all 82). That contrast is the point and it is asserted below: the two
    functions read the same `fb`, the same `u`, the same anchor. The difference
    is entirely the complement algebra, so this is **not** reachable by any
    further work on the forward-backward — it is downstream of the anchor, in
    linear space, and no log-space fix upstream touches it.

    **Pre-existing and unchanged by this slice**: the pre-slice
    `marginals_complement_aware` gives the identical answers here (`ca_old` in
    `_MUTANTS` does not change this test's verdict).
    """
    span = 207.0
    failures = []
    plain_worst = 0.0
    n = 0
    for seed in range(200):
        rng = np.random.default_rng(90000 + seed + SEED_OFFSET)
        A = (random_dfa if seed % 2 else random_nfa)(rng, int(rng.integers(3, 6)))
        L = 4
        p = wide_span_p(rng, L, span)
        b = A.final_vector()
        members = [lab for _, _, lab in A.edges]
        is_neg = [len(m) > 2 for m in members]
        if not any(is_neg) or all(is_neg):
            continue                      # this arm is about MIXED polarity
        oracle = _ExactOracle(p, A, b)
        if not oracle.nonempty or len(oracle.weights) < 2:
            continue
        n += 1
        W = R.edge_weights(p, A)
        M = R.transition_matrices(W, A)
        fb = R.forward_backward(M, A.start, b)
        exact = oracle.marginals()
        sup = exact > 0

        q_plain = R.marginals(p, A, fb, W)
        plain_worst = max(plain_worst,
                          float(np.max(np.abs(q_plain[sup] - exact[sup])
                                       / exact[sup])))
        q_ca = R.marginals_complement_aware(p, A, fb, list(range(A.n_edges)),
                                            members, is_neg)
        err = float(np.max(np.abs(q_ca[sup] - exact[sup]) / exact[sup]))
        if err > 1e-6:
            failures.append((seed, err, int(np.sum((q_ca > 0) != sup)),
                             int(np.sum(is_neg)), len(is_neg)))

    assert n >= 40, f"only {n} mixed-polarity instances found; regime too narrow"
    assert plain_worst < 1e-9, (
        f"plain `marginals` is itself inexact here (worst relative error "
        f"{plain_worst:.3e}), so this arm cannot attribute the failure to the "
        f"complement algebra")
    assert not failures, (
        f"{len(failures)}/{n} mixed-polarity instances at a {span:.0f}-nat "
        f"({10 ** (span / math.log(10)):.0e}) p-span come back wrong from "
        f"`marginals_complement_aware`, while plain `marginals` is exact to "
        f"{plain_worst:.1e} on the same inputs. Worst offenders "
        f"(seed, rel_err, support_diff, n_neg/n_classes): "
        f"{sorted(failures, key=lambda r: -r[1])[:4]}")


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(3))
def test_reference_chain_sampler_matches_the_exact_rational_posterior(kind, seed):
    """SPEC §2.5: ancestral sampling is exact and **path-weighted** — it draws
    the edge first. χ² against the exact rational posterior."""
    A, p, b, oracle = nondegenerate(seed, kind, 5)
    truth = oracle.posterior()
    if len(truth) < 3:
        pytest.skip("support too small for a meaningful χ²")
    W, _, fb = prepared(A, p, b)
    rng = np.random.default_rng(777 + seed + SEED_OFFSET)
    draws = [tuple(int(t) for t in R.sample_chain(p, A, fb, W, rng)[0])
             for _ in range(4000)]
    pv, outside = chi2_p(draws, truth)
    assert outside == 0, f"{outside}/4000 draws lie outside the language"
    assert pv > ALPHA_CORRECTED, f"chain χ² p = {pv:.3e} (regime: {kind}, L=5)"


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(3))
def test_reference_tree_sampler_matches_the_exact_rational_posterior(kind, seed):
    """SPEC §2.6 Algorithm 1. Same target as the chain sampler, different
    algorithm — and on an NFA this is what an `∃`-form eq (8) fails."""
    A, p, b, oracle = nondegenerate(seed, kind, 4)
    truth = oracle.posterior()
    if len(truth) < 3:
        pytest.skip("support too small for a meaningful χ²")
    W, M, _ = prepared(A, p, b)
    rng = np.random.default_rng(888 + seed + SEED_OFFSET)
    draws = [tuple(int(t) for t in R.sample_tree(p, A, M, W, A.start, b, rng)[0])
             for _ in range(4000)]
    pv, outside = chi2_p(draws, truth)
    assert outside == 0, f"{outside}/4000 draws lie outside the language"
    assert pv > ALPHA_CORRECTED, f"tree χ² p = {pv:.3e} (regime: {kind}, L=4)"


def test_reference_exists_form_is_measurably_wrong_against_the_exact_posterior():
    """The adversarial case: `use_exists_form=True` is SPEC §2.6's documented
    bug, and the exact rational posterior must be able to *see* it.

    If this test cannot separate the two forms, neither can any differential
    test built on the same fixtures — which is precisely how the `∃`-form
    mutant survived 721 tests once already.
    """
    A = R.Automaton(
        n_states=3, vocab_size=V,
        edges=((0, 1, frozenset({0, 1})), (0, 1, frozenset({1, 2})),
               (1, 2, frozenset({0, 1, 2, 3}))),
        start=np.array([1.0, 0.0, 0.0]), finals=frozenset({2}))
    p = np.array([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    oracle = _ExactOracle(p, A)
    truth = oracle.posterior()
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    b = A.final_vector()

    rng = np.random.default_rng(99 + SEED_OFFSET)
    N = 40000
    good = [tuple(int(t) for t in R.sample_tree(p, A, M, W, A.start, b, rng)[0])
            for _ in range(N)]
    bad = [tuple(int(t) for t in
                 R.sample_tree(p, A, M, W, A.start, b, rng,
                               use_exists_form=True)[0])
           for _ in range(N)]

    def tv(draws):
        emp: dict[tuple[int, ...], float] = {}
        for d in draws:
            emp[d] = emp.get(d, 0.0) + 1.0 / len(draws)
        return 0.5 * sum(abs(emp.get(k, 0.0) - truth.get(k, 0.0))
                         for k in set(emp) | set(truth))

    tv_good, tv_bad = tv(good), tv(bad)
    assert tv_good < 0.02, f"the correct form is off by TV = {tv_good:.3e}"
    assert tv_bad > 0.05, (
        f"the ∃ form is only TV = {tv_bad:.3e} from the truth, so this fixture "
        "cannot see eq (8)'s multiplicity — the fixture is the bug")


@pytest.mark.parametrize("kind,L,sigma,start_kind,budget", REGIMES, ids=REGIME_IDS)
@pytest.mark.parametrize("seed", range(2))
def test_reference_map_is_the_exact_rational_argmax(
        kind, L, sigma, start_kind, budget, seed):
    """SPEC §2.7: `(max, +)` is a semiring, so the recursion is **exact MAP** —
    on NFAs too, since path multiplicity cannot change a max over strings.

    Both halves are asserted: the emitted string attains the exact maximum
    `p`-product, and the returned `score` is its log.
    """
    A, p, b, oracle = nondegenerate(seed, kind, L, sigma=sigma,
                                    start_kind=start_kind, budget=budget)
    best_x, best_score = oracle.map_string()
    toks, states, score = R.map_decode(p, A, A.start, b)
    got = tuple(int(t) for t in toks)
    got_score = oracle.map_score(got)
    assert got in oracle.weights, f"MAP emitted {got}, which is not in the language"
    assert got_score == best_score, (
        f"MAP is suboptimal: emitted {got} at {float(got_score):.6e}, exact "
        f"argmax {best_x} at {float(best_score):.6e} "
        f"(regime: {kind} L={L} sigma={sigma} start={start_kind} R={budget})")
    assert score == pytest.approx(
        float(Decimal(best_score.numerator).ln()
              - Decimal(best_score.denominator).ln()), rel=1e-12, abs=1e-12)
    # The reported state path must be a real path for that string, and it must
    # land where the terminal factor allows. Under a budget `b_L = 1[d(s) ≤ R]`
    # the last state need NOT be accepting — SPEC §3.1b, and asserting
    # acceptance here is exactly the conflation CLAUDE.md calls a known trap.
    assert R.simulate(A, got), f"MAP's string {got} dies in the automaton"
    assert states[0] < A.n_states and A.start[states[0]] > 0
    assert b[states[-1]] > 0
    if budget is None:
        assert R.accepts(A, got)


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(4))
def test_reference_distance_to_final_matches_an_exhaustive_search(kind, seed):
    """SPEC §3.1b: `d(s)` is the min number of tokens from `s` to an accepting
    state, by BFS on the **reversed** automaton. Checked forwards instead: the
    shortest `ℓ` for which `δ*(s, ·)` can reach `F` in `ℓ` steps."""
    rng = np.random.default_rng(6100 + seed + SEED_OFFSET)
    A = MAKERS[kind](rng, int(rng.integers(3, 8)))
    d = A.distance_to_final()
    inf = np.iinfo(np.int32).max
    for s in range(A.n_states):
        frontier, found = {s}, None
        for ell in range(A.n_states + 2):
            if frontier & A.finals:
                found = ell
                break
            nxt = {dst for src, dst, lab in A.edges if src in frontier and lab}
            if not nxt:
                break
            frontier = nxt
        got = None if d[s] == inf else int(d[s])
        assert got == found, f"d({s}) = {got}, exhaustive search says {found}"


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(3))
def test_reference_budget_vector_is_the_indicator_of_d_le_R(kind, seed):
    """SPEC §3.1b: `b_L(s) = 1[d(s) ≤ R]`, **not** `1[s ∈ F]`, and with no
    final-block special case — `d(s) ≤ 0 ⟺ s ∈ F` makes it fall out."""
    rng = np.random.default_rng(6200 + seed + SEED_OFFSET)
    A = MAKERS[kind](rng, 5)
    d = A.distance_to_final()
    assert np.array_equal(A.budget_vector(0), A.final_vector()), (
        "R = 0 must reproduce 1[s ∈ F] exactly; it does not, so the "
        "no-special-case claim in SPEC §3.1b is false")
    for Rem in (0, 1, 2, 5, 1000):
        got = A.budget_vector(Rem)
        assert np.array_equal(got, (d <= Rem).astype(np.float64))
    # Monotone in R, and never empty once R exceeds the largest finite d.
    prev = A.budget_vector(0)
    for Rem in range(1, 8):
        cur = A.budget_vector(Rem)
        assert np.all(cur >= prev), "the admissible set must grow with R"
        prev = cur


@pytest.mark.parametrize("seed", range(6))
def test_reference_simulate_and_accepts_match_a_subset_construction(seed):
    """`simulate`/`accepts` are the *independent simulator* every guarantee
    assertion in this project appeals to (SPEC §6.1 test 6, §6.4). They have
    never been checked against anything either."""
    A = random_nfa(np.random.default_rng(6300 + seed + SEED_OFFSET), 4)
    rng = np.random.default_rng(11 + seed)
    for _ in range(80):
        x = [int(t) for t in rng.integers(0, V, size=int(rng.integers(0, 7)))]
        cur = {s for s in range(A.n_states) if A.start[s] > 0}
        for t in x:
            cur = {dst for src, dst, lab in A.edges if src in cur and t in lab}
        assert R.simulate(A, x) == cur, f"simulate({x}) = {R.simulate(A, x)}, want {cur}"
        assert R.accepts(A, x) is bool(cur & A.finals)


@pytest.mark.parametrize("L", [1, 2, 4, 8, 16])
def test_reference_block_products_are_the_aligned_dyadic_left_to_right_products(L):
    """SPEC §2.6(a)/(c): `P_{[ℓ,r)} = M_ℓ ⋯ M_{r−1}`, **left block first**, over
    the aligned dyadic grid. Getting the operand order backwards is silent."""
    rng = np.random.default_rng(6400 + L + SEED_OFFSET)
    S = 3
    M = rng.random((L, S, S))
    bp = R.block_products(M)
    want = set()
    w = 1
    while w <= L:
        for lo in range(0, L - w + 1, w):
            want.add((lo, lo + w))
        w *= 2
    assert want <= set(bp), f"missing dyadic blocks: {sorted(want - set(bp))}"
    for (lo, hi), P in bp.items():
        ex = np.eye(S)
        for i in range(lo, hi):
            ex = ex @ M[i]
        assert np.allclose(P, ex, rtol=1e-12, atol=1e-13), \
            f"P_[{lo},{hi}) is not the left-to-right product"
    if L >= 2:
        # An order-sensitive witness: the reversed product must differ, else
        # the assertion above cannot detect a swap.
        rev = M[1] @ M[0]
        assert not np.allclose(bp[(0, 2)], rev, atol=1e-9), \
            "M[0]@M[1] == M[1]@M[0] at this seed; the fixture cannot see a swap"


def test_reference_max_over_class_topk_matches_the_definition():
    """SPEC §4.4 Layer 2b: `max` has **no complement trick**. A negated class is
    the first `topk(p, K)` entry outside `N_c`; ties go to the lowest token id;
    and `K = |N_c|` must be able to fail while `K = |N_c| + 1` succeeds."""
    rng = np.random.default_rng(6500 + SEED_OFFSET)
    Vv = 8
    for _ in range(600):
        p_row = rng.random(Vv)
        if rng.random() < 0.3:                       # force exact ties
            p_row = np.round(p_row * 3) / 3
        n_comp = int(rng.integers(0, Vv))
        comp = frozenset(int(x) for x in
                         rng.choice(Vv, size=n_comp, replace=False))
        k = int(rng.integers(1, Vv + 1))
        order = sorted(range(Vv), key=lambda v: (-float(p_row[v]), v))[:k]
        reachable = [v for v in order if v not in comp]
        if not reachable:
            with pytest.raises(LookupError):
                R.max_over_class_topk(p_row, comp, k)
            continue
        val, arg = R.max_over_class_topk(p_row, comp, k)
        assert arg == reachable[0], f"tie-break: got {arg}, want {reachable[0]}"
        assert val == float(p_row[reachable[0]])

    # The bound is tight, in both directions, on an adversarial p.
    p_row = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.02, 0.01, 0.005])
    comp = frozenset({0, 1, 2})
    with pytest.raises(LookupError):
        R.max_over_class_topk(p_row, comp, len(comp))
    val, arg = R.max_over_class_topk(p_row, comp, len(comp) + 1)
    assert (val, arg) == (0.1, 3)


def test_reference_segment_max_argmin_matches_the_definition():
    """SPEC §2.7: segmented max with **lowest-index** tie-breaking, an `N`
    sentinel for empty segments, and `-inf` — never `-1` — for their max."""
    rng = np.random.default_rng(6600 + SEED_OFFSET)
    for _ in range(400):
        n = int(rng.integers(1, 14))
        ns = int(rng.integers(1, 6))
        vals = rng.integers(0, 3, size=n).astype(np.float64)   # ties guaranteed
        seg = rng.integers(0, ns, size=n)
        mx, am = R.segment_max_argmin(vals, seg, ns)
        for s in range(ns):
            idxs = [i for i in range(n) if seg[i] == s]
            if not idxs:
                assert mx[s] == -np.inf, f"empty segment max = {mx[s]}, want -inf"
                assert am[s] == n, f"empty segment arg = {am[s]}, want N = {n}"
                assert am[s] != -1
            else:
                m = max(vals[i] for i in idxs)
                assert mx[s] == m
                assert am[s] == min(i for i in idxs if vals[i] == m)


def test_reference_map_tie_break_is_lowest_token_then_lowest_state():
    """SPEC §2.7: *"Deterministic tie-breaking — lowest token id, then lowest
    state id. `test_guarantee` and the unconstrained-equivalence test depend on
    it."*

    Both halves are exercised separately, because a single fixture that ties on
    tokens *and* states cannot tell which rule produced the answer:

      * token tie — two parallel edges on the same `(0,1)` carrying tokens 2 and
        1 at identical `p`; the answer must be 1;
      * state tie — two distinct accepting destinations reachable at identical
        score; the answer must be the lower state id.
    """
    token_tie = R.Automaton(
        n_states=2, vocab_size=V,
        edges=((0, 1, frozenset({2})), (0, 1, frozenset({1}))),
        start=np.array([1.0, 0.0]), finals=frozenset({1}))
    p = np.array([[0.1, 0.45, 0.45, 0.0]])
    assert p[0, 1] == p[0, 2], "fixture check: the token tie must be exact"
    toks, _, _ = R.map_decode(p, token_tie, token_tie.start,
                              token_tie.final_vector())
    assert int(toks[0]) == 1, f"token tie went to {int(toks[0])}, want 1"

    # A single edge whose label set ties internally: same rule, different code
    # path (the `argmax(sub)` inside the M̃ build rather than the `arg` update).
    one_edge = R.Automaton(
        n_states=2, vocab_size=V, edges=((0, 1, frozenset({1, 2})),),
        start=np.array([1.0, 0.0]), finals=frozenset({1}))
    toks2, _, _ = R.map_decode(p, one_edge, one_edge.start,
                               one_edge.final_vector())
    assert int(toks2[0]) == 1, f"intra-class tie went to {int(toks2[0])}, want 1"

    state_tie = R.Automaton(
        n_states=3, vocab_size=V,
        edges=((0, 1, frozenset({0})), (0, 2, frozenset({0}))),
        start=np.array([1.0, 0.0, 0.0]), finals=frozenset({1, 2}))
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    _, states, _ = R.map_decode(q, state_tie, state_tie.start,
                                state_tie.final_vector())
    assert int(states[-1]) == 1, (
        f"state tie went to {int(states[-1])}, want the lower id 1")


def test_reference_map_is_exact_on_marginals_the_model_can_produce():
    """**RED — a real defect in `reference.py`. SPEC §2.7.**

    §2.7 mandates log space precisely so MAP is *exact*: "exact, no scaling
    discussion, no underflow". But `map_decode` computes

        logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), _NEG)

    so every token below `1e-300` scores the same `log(1e-300)` and `argmax`
    resolves the tie by lowest id — MAP emits the *smallest admissible token id*
    rather than the most probable one.

    This is the **same defect, in the same place**, that was already found and
    fixed on the JAX path: `model/constrained.map_log_floor` used to be `1e-30`
    and its docstring records the finding at length. The fix moved that floor to
    `finfo(dtype).tiny` (2.2e-308 in float64, `log` ≈ −708) "so it sits under
    everything the model can represent". `reference.py` was never given the same
    fix, and it is 8 orders of magnitude above `finfo.tiny` — which means the
    **reference is now less exact than the path it certifies**, and a future
    differential test that lands in the window will blame the fast path.

    Reachable: with the released model's softcap (`30·tanh(x/30)`) a logit gap
    of 56 nats at `T = 0.08` — `_MIN_TEMP = 1e-12` makes any `T` reachable by
    configuration alone (SPEC §3.9) — gives `p = 9.86e-305`, inside the window.
    """
    A = R.Automaton(
        n_states=2, vocab_size=V, edges=((0, 1, frozenset({0, 1})),),
        start=np.array([1.0, 0.0]), finals=frozenset({1}))
    p = np.zeros((1, V))
    p[0, 0] = 1e-305
    p[0, 1] = 1e-301           # 1e4 times more probable than token 0
    p[0, 3] = 1.0 - p[0, 0] - p[0, 1]

    oracle = _ExactOracle(p, A)
    best_x, best_score = oracle.map_string()
    assert best_x == (1,), "fixture check: the exact argmax is token 1"

    toks, _, score = R.map_decode(p, A, A.start, A.final_vector())
    assert int(toks[0]) == 1, (
        f"MAP emitted token {int(toks[0])} (p = {p[0, int(toks[0])]:.3e}) instead "
        f"of token 1 (p = {p[0, 1]:.3e}); the 1e-300 clamp flattened them")
    assert score == pytest.approx(math.log(1e-301), rel=1e-12)


def test_reference_map_floor_sits_under_what_the_model_can_represent():
    """**RED — the same defect, stated as the invariant rather than a witness.**

    `model/constrained.map_log_floor`: *"The floor has to sit under everything
    the model can represent, not under an unrelated round number, so it is
    derived from the dtype."* The reference's floor must satisfy the same rule,
    or the two paths rank tokens differently in the gap between them.

    The gap is measured here rather than asserted abstractly: `map_decode` is
    order-preserving on every pair of distinct float64 normals iff its floor is
    at or below `finfo(float64).tiny`.
    """
    fast_floor = float(np.finfo(np.float64).tiny)          # 2.225e-308
    A = R.Automaton(
        n_states=2, vocab_size=V, edges=((0, 1, frozenset({0, 1})),),
        start=np.array([1.0, 0.0]), finals=frozenset({1}))
    # Two probabilities that are distinct float64 normals above the fast path's
    # floor. MAP must prefer the larger one.
    lo, hi = fast_floor * 4.0, fast_floor * 4.0e3
    p = np.zeros((1, V))
    p[0, 0], p[0, 1] = lo, hi
    p[0, 3] = 1.0 - lo - hi
    assert lo > fast_floor and hi > lo, "fixture check"
    toks, _, _ = R.map_decode(p, A, A.start, A.final_vector())
    assert int(toks[0]) == 1, (
        f"the reference's MAP floor is above {hi:.3e}, while the JAX path's is "
        f"{fast_floor:.3e}; between the two floors the reference and the path "
        f"it certifies disagree by construction")


# ===========================================================================
# Part 2 — infer/tree.py
# ===========================================================================

def _exact_state_path_distribution(M: np.ndarray, a_start: np.ndarray,
                                   b_final: np.ndarray):
    """`P(s_0..s_L) ∝ a(s_0)·Π_i M_i(s_i,s_{i+1})·b(s_L)`, exact rationals.

    This is what SPEC §2.6's top-down recursion — the root joint plus eq (7) at
    every midpoint — must reproduce. Enumerated directly over `S^{L+1}` paths,
    with no tree, no scan and no down-sweep anywhere in sight.
    """
    L, S, _ = M.shape
    Mf = [[[Fraction(float(M[i, s, t])) for t in range(S)] for s in range(S)]
          for i in range(L)]
    af = [Fraction(float(x)) for x in a_start]
    bf = [Fraction(float(x)) for x in b_final]
    out: dict[tuple[int, ...], Fraction] = {}
    Z = Fraction(0)
    for path in itertools.product(range(S), repeat=L + 1):
        w = af[path[0]] * bf[path[-1]]
        if w == 0:
            continue
        for i in range(L):
            w *= Mf[i][path[i]][path[i + 1]]
            if w == 0:
                break
        if w != 0:
            out[path] = w
            Z += w
    if Z == 0:
        return {}, Z
    return {k: float(v / Z) for k, v in out.items()}, Z


def _log_tree(M: np.ndarray):
    logM = jnp.where(jnp.asarray(M) > 0,
                     jnp.log(jnp.maximum(jnp.asarray(M), 1e-320)),
                     scans.NEG_SENTINEL)
    return scans.up_sweep_log(logM)


def _log_vec(x: np.ndarray):
    return jnp.where(jnp.asarray(x) > 0,
                     jnp.log(jnp.maximum(jnp.asarray(x), 1e-320)),
                     scans.NEG_SENTINEL)


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("L", [4, 8])
@pytest.mark.parametrize("seed", range(2))
def test_tree_boundary_states_are_drawn_from_the_exact_path_distribution(
        kind, L, seed):
    """SPEC §2.6: the root joint `a(s_0)·P_{[0,L)}(s_0,s_L)·b(s_L)` and eq (7)
    at every midpoint must jointly reproduce the exact state-path distribution.

    χ² over the **whole `L+1`-tuple**, not per-position marginals: a swapped
    left/right child or a reversed operand order can leave every marginal right
    while the joint is wrong. Both the linear and the log tree are checked —
    production runs the log one (`model/constrained.py`), and it has never been
    compared to anything but the linear one.
    """
    rng = np.random.default_rng(7100 + seed * 13 + L + SEED_OFFSET)
    A = MAKERS[kind](rng, int(rng.integers(3, 5)))
    p = flat_p(rng, L)
    b = A.final_vector()
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)
    exact, Z = _exact_state_path_distribution(M, A.start, b)
    if Z == 0 or len(exact) < 3:
        pytest.skip("degenerate: fewer than 3 live state paths at this seed")

    N = 4000
    keys = jax.random.split(jax.random.PRNGKey(4000 + seed), N)

    linear = scans.up_sweep(jnp.asarray(M))
    drawn = np.asarray(jax.jit(jax.vmap(lambda k: tree.sample_states(
        linear, jnp.asarray(A.start), jnp.asarray(b), k)))(keys))
    counts: dict[tuple[int, ...], int] = {}
    for row in drawn:
        key = tuple(int(x) for x in row)
        counts[key] = counts.get(key, 0) + 1
    outside = sum(c for k, c in counts.items() if k not in exact)
    assert outside == 0, (
        f"sample_states drew {outside}/{N} state paths with zero weight "
        f"(regime: {kind} L={L} S={A.n_states})")
    pv, _ = chi2_p([tuple(int(x) for x in r) for r in drawn], exact)
    assert pv > ALPHA_CORRECTED, f"linear tree χ² p = {pv:.3e}"

    lt = _log_tree(M)
    drawn2 = np.asarray(jax.jit(jax.vmap(lambda k: tree.sample_states_log(
        lt, _log_vec(A.start), _log_vec(b), k)[0]))(keys))
    counts2: dict[tuple[int, ...], int] = {}
    for row in drawn2:
        key = tuple(int(x) for x in row)
        counts2[key] = counts2.get(key, 0) + 1
    outside2 = sum(c for k, c in counts2.items() if k not in exact)
    assert outside2 == 0, (
        f"sample_states_log drew {outside2}/{N} state paths with zero weight")
    pv2, _ = chi2_p([tuple(int(x) for x in r) for r in drawn2], exact)
    assert pv2 > ALPHA_CORRECTED, f"log tree χ² p = {pv2:.3e}"


def test_tree_down_sweep_pairs_each_interval_with_its_own_two_children():
    """SPEC eq (7): `P(s_m | s_ℓ, s_r) ∝ P_{[ℓ,m)}(s_ℓ,s_m)·P_{[m,r)}(s_m,s_r)`.

    Adversarial fixture for the index arithmetic in the down-sweep
    (`child[2j]` / `child[2j+1]`). Every position admits exactly one token and
    the automaton is a chain, so the **entire** boundary sequence is forced:
    `s_i = i`. Any mis-pairing of an interval with a child — a swap, an off-by-
    one, or a level shift — puts a zero-weight state on the path, which is a
    hard assertion rather than a distributional one.

    Position-dependent weights are used so a swap cannot cancel.
    """
    L, S = 8, 9
    M = np.zeros((L, S, S))
    for i in range(L):
        M[i, i, i + 1] = 0.1 * (i + 1)        # distinct per position
    a0 = np.zeros(S); a0[0] = 1.0
    bL = np.zeros(S); bL[L] = 1.0
    want = tuple(range(L + 1))

    linear = scans.up_sweep(jnp.asarray(M))
    for k in range(6):
        st = tree.sample_states(linear, jnp.asarray(a0), jnp.asarray(bL),
                                jax.random.PRNGKey(k))
        assert tuple(int(x) for x in st) == want, (
            f"linear down-sweep produced {tuple(int(x) for x in st)}, and the "
            f"only path of weight > 0 is {want}")

    lt = _log_tree(M)
    for k in range(6):
        st, feasible = tree.sample_states_log(lt, _log_vec(a0), _log_vec(bL),
                                              jax.random.PRNGKey(k))
        assert bool(feasible)
        assert tuple(int(x) for x in st) == want, (
            f"log down-sweep produced {tuple(int(x) for x in st)}, want {want}")


def test_tree_token_draw_is_edge_multiplicity_weighted_not_an_existence_flag():
    """SPEC eq (8): `x_i ~ p_i(v)·|{e : src=s_i, dst=s_{i+1}, v ∈ label(e)}|`.

    Two parallel edges `0→1` with **overlapping but distinct** labels, so token
    1 has multiplicity 2 and tokens 0 and 2 multiplicity 1. The `∃` form and the
    weighted form differ by TV = 0.164 here; on a DFA they coincide, which is
    why the fixture is an NFA.

    The separation between the two candidate answers is asserted first: without
    it a green χ² would prove only that the fixture is blind.
    """
    A = R.Automaton(
        n_states=2, vocab_size=V,
        edges=((0, 1, frozenset({0, 1})), (0, 1, frozenset({1, 2}))),
        start=np.array([1.0, 0.0]), finals=frozenset({1}))
    p = np.array([[0.4, 0.3, 0.2, 0.1]])
    cid, members, is_neg, indices, indptr = class_tables(A, threshold=2)

    mult = np.array([1.0, 2.0, 1.0, 0.0])
    weighted = p[0] * mult
    weighted /= weighted.sum()
    exists = p[0] * (mult > 0)
    exists /= exists.sum()
    assert 0.5 * np.abs(weighted - exists).sum() > 0.1, (
        "fixture check: the two candidate answers must differ, or a χ² against "
        "either proves nothing")

    N = 30000
    src = jnp.asarray([e[0] for e in A.edges])
    dst = jnp.asarray([e[1] for e in A.edges])
    states = jnp.asarray([0, 1], dtype=jnp.int32)
    draw = jax.jit(jax.vmap(lambda k: tree.sample_tokens(
        jnp.asarray(p.T), states, src, dst, jnp.asarray(cid),
        jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(is_neg),
        len(members), k)[0]))
    toks = np.asarray(draw(jax.random.split(jax.random.PRNGKey(5), N)))[:, 0]
    counts = np.bincount(toks, minlength=V)
    assert counts[3] == 0, "token 3 is on no edge and must never be drawn"

    keep = weighted > 0
    pv = float(stats.chisquare(counts[keep],
                               weighted[keep] * counts.sum()).pvalue)
    assert pv > ALPHA_CORRECTED, (
        f"χ² against the multiplicity-weighted law p = {pv:.3e}; empirical "
        f"{counts / N}, weighted {weighted}, ∃-form {exists}")
    pv_bad = float(stats.chisquare(counts[keep],
                                   exists[keep] * counts.sum()).pvalue)
    assert pv_bad < ALPHA_CORRECTED, (
        f"the sample is ALSO consistent with the ∃ form (p = {pv_bad:.3e}); "
        "raise N or sharpen the fixture")


def test_tree_sample_tokens_reports_invalid_on_a_state_pair_with_no_edge():
    """`sample_tokens` must surface an inadmissible boundary pair rather than
    return a near-uniform draw over the vocabulary.

    SPEC §3.1 / §2.6 `[V-P5]`: `categorical` is shift-invariant, so a degenerate
    weight vector still yields a confident-looking token that passes every shape
    and validity check downstream. `valid` is the only thing between that and a
    canvas of plausible multilingual garbage.
    """
    A = R.Automaton(
        n_states=2, vocab_size=V,
        edges=((0, 1, frozenset({0, 1})),),
        start=np.array([1.0, 0.0]), finals=frozenset({1}))
    cid, members, is_neg, indices, indptr = class_tables(A, threshold=2)
    src = jnp.asarray([e[0] for e in A.edges])
    dst = jnp.asarray([e[1] for e in A.edges])
    args = (jnp.asarray(np.array([[0.4], [0.3], [0.2], [0.1]])),)

    _, ok = tree.sample_tokens(*args, jnp.asarray([0, 1], jnp.int32), src, dst,
                               jnp.asarray(cid), jnp.asarray(indices),
                               jnp.asarray(indptr), jnp.asarray(is_neg),
                               len(members), jax.random.PRNGKey(0))
    assert bool(ok), "a real transition must be reported valid"

    _, bad = tree.sample_tokens(*args, jnp.asarray([1, 0], jnp.int32), src, dst,
                                jnp.asarray(cid), jnp.asarray(indices),
                                jnp.asarray(indptr), jnp.asarray(is_neg),
                                len(members), jax.random.PRNGKey(0))
    assert not bool(bad), (
        "1 → 0 carries no edge, yet sample_tokens reported the draw valid")


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(3))
def test_tree_map_is_the_exact_rational_argmax(kind, seed):
    """SPEC §2.7 over the max-plus tree, including token recovery through
    `argclass` — checked against the exact rational argmax, not against
    `reference.map_decode` (which Part 1 shows has its own floor defect)."""
    A, p, b, oracle = nondegenerate(seed, kind, 4)
    best_x, best_score = oracle.map_string()
    L = 4
    cid, members, is_neg, indices, indptr = class_tables(A, threshold=2)
    C = len(members)
    logp = np.where(p > 0, np.log(p), scans.NEG_SENTINEL)

    class_max = np.full((C, L), scans.NEG_SENTINEL)
    class_arg = np.zeros((C, L), np.int32)
    for c, m in enumerate(members):
        for i in range(L):
            best, tok = scans.NEG_SENTINEL, 0
            for v in sorted(m):
                if logp[i, v] > best:
                    best, tok = logp[i, v], v
            class_max[c, i], class_arg[c, i] = best, tok
    Mt = np.full((L, A.n_states, A.n_states), scans.NEG_SENTINEL)
    for e, (s, d, _) in enumerate(A.edges):
        for i in range(L):
            Mt[i, s, d] = max(Mt[i, s, d], class_max[cid[e], i])

    trm = scans.up_sweep_maxplus(jnp.asarray(Mt))
    a_log = np.where(A.start > 0, np.log(np.maximum(A.start, 1e-320)),
                     scans.NEG_SENTINEL)
    b_log = np.where(b > 0, 0.0, scans.NEG_SENTINEL)
    toks, states, score, feasible = tree.map_states_and_tokens(
        jnp.asarray(logp.T), trm, jnp.asarray([e[0] for e in A.edges]),
        jnp.asarray([e[1] for e in A.edges]), jnp.asarray(cid),
        jnp.asarray(class_max), jnp.asarray(class_arg),
        jnp.asarray(a_log), jnp.asarray(b_log))
    assert bool(feasible)
    got = tuple(int(t) for t in toks)
    got_score = Fraction(1)
    for i, v in enumerate(got):
        got_score *= oracle.p[i][v]
    assert got in oracle.weights, f"tree MAP emitted {got}, outside the language"
    assert got_score == best_score, (
        f"tree MAP is suboptimal: {got} at {float(got_score):.6e} vs exact "
        f"{best_x} at {float(best_score):.6e} (regime: {kind}, L=4)")


def test_tree_map_reports_infeasible_on_an_empty_language():
    """SPEC §2.6 `[V-P5]`: `argmax` is shift-invariant, so an all-sentinel root
    produces a confident-looking draw. `score`/`feasible` is the only detector,
    and it was discarded once already."""
    A = R.Automaton(
        n_states=2, vocab_size=V, edges=((0, 1, frozenset({0})),),
        start=np.array([1.0, 0.0]), finals=frozenset({0}))   # 0 is not reachable back
    L = 4
    p = np.full((L, V), 0.25)
    cid, members, is_neg, indices, indptr = class_tables(A, threshold=2)
    C = len(members)
    logp = np.log(p)
    class_max = np.array([[max(logp[i, v] for v in sorted(m)) for i in range(L)]
                          for m in members])
    class_arg = np.zeros((C, L), np.int32)
    Mt = np.full((L, 2, 2), scans.NEG_SENTINEL)
    for e, (s, d, _) in enumerate(A.edges):
        for i in range(L):
            Mt[i, s, d] = max(Mt[i, s, d], class_max[cid[e], i])
    trm = scans.up_sweep_maxplus(jnp.asarray(Mt))
    a_log = np.array([0.0, scans.NEG_SENTINEL])
    b_log = np.array([0.0, scans.NEG_SENTINEL])
    _, _, score, feasible = tree.map_states_and_tokens(
        jnp.asarray(logp.T), trm, jnp.asarray([e[0] for e in A.edges]),
        jnp.asarray([e[1] for e in A.edges]), jnp.asarray(cid),
        jnp.asarray(class_max), jnp.asarray(class_arg),
        jnp.asarray(a_log), jnp.asarray(b_log))
    assert not bool(feasible), (
        f"no string of length {L} is accepted, yet MAP reported feasible "
        f"(score = {float(score)})")


def test_tree_map_tie_break_is_lowest_token_id():
    """**RED — `tree.map_states_and_tokens` breaks ties by lowest EDGE index.**

    SPEC §2.7: *"Deterministic tie-breaking — lowest token id, then lowest state
    id. `test_guarantee` and the unconstrained-equivalence test depend on it."*
    `reference.map_decode` implements exactly that. `tree.py` instead does

        best_edge = jnp.argmax(masked, axis=1)      # ties -> lowest EDGE index
        chosen_class = class_id[best_edge]

    so when two edges on the same `(s, s')` belong to different classes whose
    maxima tie, the winner is decided by edge ordering, not by token id.
    Measured here: the reference emits token 1, the tree emits token 2, on
    `p[1] == p[2]` exactly.

    **Reachability, stated rather than assumed.** `compile/automaton._group_edges`
    emits **one edge per `(src, dst)`** carrying the union of its labels, so a
    compiled artifact never presents two edges for the tree to choose between,
    and the token then comes from `class_argmax`, whose `jnp.argmax` does resolve
    to the lowest id. This is therefore **not** a live production defect on the
    compiled path. What it is: (i) a divergence from `reference.py` on an input
    the function's own signature accepts, and `reference.py` is what every
    differential test in this repo treats as the arbiter; (ii) an undocumented
    precondition — the docstring derives `argclass` over "edges `e` with
    `(src,dst) = pair`", plural, and says nothing about grouping; (iii) live for
    the SPEC §4.6 NFA fallback and for any hand-built automaton. The fix is
    either to honour SPEC's rule or to state the precondition and assert it.
    """
    A = R.Automaton(
        n_states=2, vocab_size=V,
        edges=((0, 1, frozenset({2})), (0, 1, frozenset({1}))),
        start=np.array([1.0, 0.0]), finals=frozenset({1}))
    L = 1
    p = np.array([[0.1, 0.45, 0.45, 0.0]])      # tokens 1 and 2 tie exactly
    assert p[0, 1] == p[0, 2], "fixture check: the tie must be exact"

    ref_toks, _, _ = R.map_decode(p, A, A.start, A.final_vector())
    assert int(ref_toks[0]) == 1, "reference check: SPEC's rule is lowest token id"

    cid, members, is_neg, indices, indptr = class_tables(A, threshold=2)
    C = len(members)
    logp = np.where(p > 0, np.log(np.maximum(p, 1e-320)), scans.NEG_SENTINEL)
    class_max = np.array([[max(logp[i, v] for v in sorted(m)) for i in range(L)]
                          for m in members])
    class_arg = np.array([[min(v for v in sorted(m)
                               if logp[i, v] == class_max[c, i])
                           for i in range(L)] for c, m in enumerate(members)],
                         dtype=np.int32)
    Mt = np.full((L, 2, 2), scans.NEG_SENTINEL)
    for e, (s, d, _) in enumerate(A.edges):
        for i in range(L):
            Mt[i, s, d] = max(Mt[i, s, d], class_max[cid[e], i])
    trm = scans.up_sweep_maxplus(jnp.asarray(Mt))
    toks, _, _, feasible = tree.map_states_and_tokens(
        jnp.asarray(logp.T), trm, jnp.asarray([e[0] for e in A.edges]),
        jnp.asarray([e[1] for e in A.edges]), jnp.asarray(cid),
        jnp.asarray(class_max), jnp.asarray(class_arg),
        jnp.asarray(np.array([0.0, scans.NEG_SENTINEL])),
        jnp.asarray(np.array([scans.NEG_SENTINEL, 0.0])))
    assert bool(feasible)
    assert int(toks[0]) == int(ref_toks[0]) == 1, (
        f"tie-break: reference emits {int(ref_toks[0])}, tree emits "
        f"{int(toks[0])}; SPEC §2.7 says lowest token id")


# ===========================================================================
# Part 3 — infer/scans.py: the corners the three post-hoc files do not reach
# ===========================================================================

def test_prefix_suffix_partition_equals_an_external_log_chain_with_non_unit_leaves():
    """SPEC §2.4 `[D]`: `log(a_i·b_i) + scale_a[i] + scale_b[i] == log Z`.

    **Pins the `forget_leaf_scales` mutant.** `prefix_suffix` rebuilds `log M_i`
    from the tree's *max-normalised* leaves plus `tree.log_scales[0]`; drop the
    `+ leaf_scales[i]` and `log Z` shifts by `−Σ_i log max M_i` — a constant, so
    `i`-invariance survives untouched and so does every existing assertion. The
    audit fixture that names this case pins `max M_i = 1.0`, which makes
    `leaf_scales == 0` and the mutant unobservable **by construction**.

    So the leaf maxima here are deliberately **not** 1: they span 1e-3 to 7,
    giving `Σ log max M_i = -14.5` nats of separation, and the target comes from
    a plain float64 logsumexp chain that shares no code with `scans.py`.
    """
    rng = np.random.default_rng(8100 + SEED_OFFSET)
    L, S = 8, 4
    M = rng.random((L, S, S)) * rng.choice([1e-3, 1.0, 7.0], size=(L, 1, 1))
    M = M * (rng.random((L, S, S)) > 0.3)
    a0 = np.zeros(S); a0[0] = 1.0
    bL = np.ones(S)
    exact = _float64_log_chain(M, a0, bL)
    assert math.isfinite(exact)

    trm = scans.up_sweep(jnp.asarray(M))
    leaf_scale_sum = float(jnp.sum(trm.log_scales[0]))
    assert abs(leaf_scale_sum) > 1.0, (
        f"Σ log max M_i = {leaf_scale_sum:.3e}; with leaf maxima at 1 the "
        "leaf-scale term is unobservable and this fixture proves nothing")

    a, b, sa, sb = scans.prefix_suffix(trm, jnp.asarray(a0), jnp.asarray(bL))
    got = [float(jnp.log(a[i] @ b[i]) + sa[i] + sb[i]) for i in range(L + 1)]
    assert max(got) - min(got) < 1e-9, f"log Z varies with i: {got}"
    for i, g in enumerate(got):
        assert g == pytest.approx(exact, rel=1e-11, abs=1e-11), (
            f"log Z at boundary {i}: got {g!r}, external chain {exact!r} "
            f"(Σ log max M_i = {leaf_scale_sum:.4f})")


def test_prefix_suffix_returns_representable_factors_at_a_gamma_that_overflows():
    """**Pins the `no_ceil` mutant.** SPEC §2.4 / `prefix_suffix`'s own bound.

    The docstring derives `max(log a_i) − scale_a[i] ≤ γ_i/2`, so once
    `γ_i > 2·log(float64 max) ≈ 1418` nats the exponential overflows and `a`
    contains `inf`. One multiplication later `inf · 0` is `NaN` and it is loose
    in `u` — the ceiling is what stops that, and it is described in the code as
    "pure insurance… there are none once non-viable states are dropped".

    A large `γ` on **viable** states is the regime that claim misses. Two
    parallel accepting chains, one front-heavy and one back-heavy, put the
    forward maximum on chain A and the backward maximum on chain B at the same
    boundary, with no path between them: `γ ≈ 2·c·(L/2)`. Sudoku measures
    `γ = 1,616` nats at `T = 0.1` (`prefix_suffix`'s own docstring), so this is
    the shipped grammar's regime one halving of temperature away, not a
    hypothetical.

    What is asserted is representability only — `a` and `b` finite, `u = a·b`
    free of `NaN`. Accuracy past the bound is a *separate*, already-guarded
    question, and `feasible_out` is checked to be reporting the position rather
    than staying silent.
    """
    L, S, c = 8, 4, 250.0                   # γ ≈ 2·250·4 = 2000 nats
    M = np.zeros((L, S, S))
    for i in range(L):
        w1 = math.exp(c if i < L // 2 else -c)
        w2 = math.exp(-c if i < L // 2 else c)
        if i == 0:
            M[i, 0, 1], M[i, 0, 2] = w1, w2
        elif i == L - 1:
            M[i, 1, 3], M[i, 2, 3] = w1, w2
        else:
            M[i, 1, 1], M[i, 2, 2] = w1, w2
    a0 = np.zeros(S); a0[0] = 1.0
    bL = np.zeros(S); bL[3] = 1.0
    assert np.all(np.isfinite(M)), "fixture check: the leaves themselves are fine"

    trm = scans.up_sweep(jnp.asarray(M))
    a, b, sa, sb, representable = scans.prefix_suffix(
        trm, jnp.asarray(a0), jnp.asarray(bL), feasible_out=True)

    assert bool(jnp.all(jnp.isfinite(a))), (
        f"`a` overflowed: max = {float(jnp.max(a)):.3e}; the ceiling is not "
        "insurance, it is load-bearing at this γ")
    assert bool(jnp.all(jnp.isfinite(b))), \
        f"`b` overflowed: max = {float(jnp.max(b)):.3e}"
    u = a[:-1] * b[1:]
    assert not bool(jnp.any(jnp.isnan(u))), "inf · 0 produced NaN in u = a·b"
    assert not bool(jnp.all(representable)), (
        "γ is ~2000 nats, far past the docstring's own 708-nat validity bound, "
        "yet feasible_out reported every position representable")


#: **The threshold is XLA's, and it is not float64's.** `up_sweep` divides on
#: the JAX backend, and XLA:CPU flushes subnormals to zero — so a quotient is
#: destroyed as soon as it falls below the smallest **normal**
#: `finfo(float64).tiny`, at `-log(tiny) = 708.396` nats, and *not* at the
#: smallest **subnormal**'s `744.44` nats that plain numpy reaches. Measured
#: here: `1e-20 / 1e300` is exactly `0.0` under `jnp` and `1e-320` under numpy,
#: and a sweep puts the boundary in `(708, 708.4]` — 708 nats below the max
#: still returns `3.3e-308`, 708.4 returns `0.0`.
#:
#: This is why the 736.8-nat fixture below is the one that bites: it sits
#: **between the two thresholds**, so the entry is a perfectly good numpy
#: float64 that `jnp` cannot keep. A fixture past 744.44 would prove nothing —
#: numpy would have destroyed it before `_normalize` ever saw it (see the
#: `underflow`-contract note in the test).
XLA_FLUSH_NATS = -math.log(float(np.finfo(np.float64).tiny))     # 708.396
NUMPY_SUBNORMAL_NATS = -math.log(5e-324)                          # 744.44


@pytest.mark.parametrize("small,live_nats", [(1e-5, 702.3), (1e-20, 736.8)])
def test_up_sweep_normalisation_does_not_delete_a_live_transition(small, live_nats):
    """`up_sweep`'s linear `M_i / max(M_i)` must not silently delete a live edge.

    `up_sweep` divides every leaf by its own maximum. `prefix_suffix` then
    rebuilds `log M_i` as `where(m > 0, log(max(m, tiny)) + leaf_scales[i], neg)`
    — so any entry that the division flushes to exactly `0.0` is reclassified as
    a **structurally impossible transition**, not as an underflow, and leaves
    via `viable == False`. Before this slice `feasible_out` could not see it:
    that flag guarded `γ_i`, an entirely different quantity.

    **Where the threshold actually is.** `745` is the wrong number and it was in
    an earlier version of this docstring. The division happens under `jnp`, and
    XLA:CPU flushes subnormals, so the boundary is `-log(finfo.tiny)` =
    **708.396 nats**, not numpy's `-log(5e-324)` = 744.44. See
    `XLA_FLUSH_NATS` above for the measurement. The `1e-20` arm's span of 736.8
    nats sits between the two, which is exactly why it is red under `jnp` and
    would be green under numpy — that gap *is* the fixture's value, not an
    arbitrary constant.

    **Regime, stated rather than assumed — the magnitude is not the mechanism.**
    The loss needs a leaf whose live entries lie more than
    `708.4 + log(max_{s,s'} M_i(s,s'))` nats below its maximum. Fed `M` from
    `marginals.transition_matrices` on probability marginals, `max M_i` is at
    most a small multiple of 1, so the normalisation costs only `log(max M_i)`
    nats beyond the float's own floor — on the production path a few nats, not
    a cliff, and the cliff is `up_sweep_log`'s job (SPEC §2.6 `[V-P4]`). The
    fixture uses `max M_i = 1e300` to exhibit the mechanism in isolation; that
    magnitude is **not** production-reachable through `transition_matrices`,
    and this test should not be read as one.

    What *is* regime-independent is that the loss must not be **unreported**: a
    caller otherwise gets `Z == 0` with no way to classify it, and CLAUDE.md
    requires (a) and (b) to raise while (c) means the scaling is missing. This
    is none of the three — the language is non-empty, the budget is satisfiable,
    the scaling is present. So the assertion is the disjunction: either the
    value survives, or the caller is told it did not.

    **What `TreeLevels.underflow` promises, and what it does not.** It reports
    that *the division* destroyed a live entry — `(mats > 0) & (out == 0)`. It
    is therefore silent when the entry was already `0.0` on the way in, which
    is what happens past `NUMPY_SUBNORMAL_NATS`: measured at a 750-nat gap the
    raw `M` entry is `0.0` in numpy before `_normalize` sees it, so
    `underflow` is False and `representable` is all-True while `log Z` is the
    sentinel. That is **not** a detector failure — nothing was destroyed by the
    division, and float64 could not have held the entry either way — but the
    flag's contract is narrower than "this position lost a live edge", and a
    caller must not read it as the latter. The `1e-5` control (702.3 nats, below
    both thresholds) and the `1e-20` case (736.8, between them) are the two arms
    that bracket the mechanism; the >744.44 regime is out of this flag's scope
    by construction.
    """
    L, S, big = 8, 3, 1e300
    M = np.zeros((L, S, S))
    for i in range(L):
        M[i, 0, 0] = big          # the unscored tail: pins the leaf maximum
        M[i, 0, 1] = small        # the only entry into the accepting chain
        M[i, 1, 1] = small
    a0 = np.zeros(S); a0[0] = 1.0
    bL = np.zeros(S); bL[1] = 1.0
    assert np.all(np.isfinite(M)) and M[0, 0, 1] > 0.0, \
        "fixture check: every leaf entry is a representable float64"
    span = math.log(big) - math.log(small)
    assert abs(span - live_nats) < 1.0, f"fixture check: span = {span:.1f} nats"

    exact = _float64_log_chain(M, a0, bL)
    assert math.isfinite(exact), "fixture check: the language is non-empty"

    trm = scans.up_sweep(jnp.asarray(M))
    a, b, sa, sb, representable = scans.prefix_suffix(
        trm, jnp.asarray(a0), jnp.asarray(bL), feasible_out=True)
    got = float(jnp.log(jnp.maximum(a[0] @ b[0], jnp.finfo(jnp.float64).tiny))
                + sa[0] + sb[0])

    lost = (not math.isfinite(got)) or got < scans.NEG_SENTINEL / 2.0
    reported = not bool(jnp.all(representable))
    assert (not lost) or reported, (
        f"the language is non-empty (external log Z = {exact:.2f}) but "
        f"prefix_suffix returned log Z = {got!r} after up_sweep normalised a "
        f"leaf whose live entry sat {span:.1f} nats below its maximum, and "
        f"feasible_out reported all_representable=True. SPEC §6.3 requires "
        f"`Z == 0` to be classified before it is acted on; this is none of "
        f"causes (a), (b) or (c) — the language is non-empty, the budget is "
        f"satisfiable, and the scaling is present.")
    if not lost:
        assert got == pytest.approx(exact, rel=1e-9, abs=1e-6), (
            f"log Z = {got!r}, external chain {exact!r} (span {span:.1f} nats)")


def test_the_flush_threshold_is_xlas_not_numpys():
    """Pins `XLA_FLUSH_NATS`, so the constant above is a measurement rather
    than folklore — and so the 736.8-nat fixture's placement is checkable.

    XLA:CPU flushes subnormals to zero; numpy does not. The consequence is that
    `up_sweep`'s division destroys a live entry `708.4` nats below the leaf max,
    where the same arithmetic in numpy survives to `744.44`. An earlier version
    of this file cited `745` for both, which would have made the fixture look
    arbitrary and would have mis-sized every regime statement built on it.
    """
    assert XLA_FLUSH_NATS == pytest.approx(708.396, abs=0.01)
    assert NUMPY_SUBNORMAL_NATS == pytest.approx(744.44, abs=0.01)

    big = 1e300
    # numpy keeps it as a subnormal; jnp does not.
    small = 1e-20
    assert small / big > 0.0, "fixture check: numpy holds the quotient"
    assert float(jnp.asarray(small, jnp.float64)
                 / jnp.asarray(big, jnp.float64)) == 0.0, (
        "XLA did NOT flush this quotient, so the 736.8-nat fixture no longer "
        "sits between the two thresholds and its regime statement is stale")

    # The boundary itself: just inside survives, just outside does not.
    inside = big * math.exp(-(XLA_FLUSH_NATS - 0.5))
    outside = big * math.exp(-(XLA_FLUSH_NATS + 0.5))
    assert float(jnp.asarray(inside, jnp.float64)
                 / jnp.asarray(big, jnp.float64)) > 0.0
    assert float(jnp.asarray(outside, jnp.float64)
                 / jnp.asarray(big, jnp.float64)) == 0.0


def test_the_underflow_flag_reports_the_division_not_the_position():
    """`TreeLevels.underflow`'s contract is narrower than "this position lost a
    live edge", and the difference must be stated where it is relied on.

    Past `NUMPY_SUBNORMAL_NATS` the entry is already `0.0` in the `M` handed to
    `up_sweep`, so the division destroyed nothing and the flag correctly reads
    False — while `log Z` is still the sentinel, because there was never a
    representable value to keep. A caller that reads `underflow == False` as
    "this position is healthy" is wrong for that regime, and this test exists so
    nobody has to rediscover it from a green suite.
    """
    L, S, big = 8, 3, 1e300
    small = big * math.exp(-750.0)
    assert small == 0.0, (
        "fixture check: at 750 nats below 1e300 the entry must already be 0.0 "
        "in numpy, which is the whole point")
    M = np.zeros((L, S, S))
    for i in range(L):
        M[i, 0, 0] = big
        M[i, 0, 1] = small
        M[i, 1, 1] = small
    a0 = np.zeros(S); a0[0] = 1.0
    bL = np.zeros(S); bL[1] = 1.0

    trm = scans.up_sweep(jnp.asarray(M))
    assert trm.underflow is not None
    assert not bool(jnp.any(trm.underflow)), (
        "the division destroyed nothing here — the entry arrived as 0.0 — so "
        "`underflow` must not claim otherwise")
    _, _, _, _, representable = scans.prefix_suffix(
        trm, jnp.asarray(a0), jnp.asarray(bL), feasible_out=True)
    assert bool(jnp.all(representable)), (
        "`representable` is about this scheme's own scaling; an input that was "
        "already dead is not its business")
    # And the language really is unrepresentable, not merely unreported: the
    # external chain agrees there is nothing to keep.
    assert _float64_log_chain(M, a0, bL) == -np.inf


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(3))
def test_jax_transition_matrices_carry_parallel_edge_multiplicity(kind, seed):
    """SPEC eq (2): `M_i(s,s') = Σ_{e : src=s, dst=s'} W[i,e]` — a **sum** over
    parallel edges, which is what makes the state-path distribution
    path-weighted on an NFA. `.at[].add`, never `.set`.

    Checked against `reference.transition_matrices` *and* against a direct
    Fraction sum, so a shared bug in the two dense paths cannot hide.
    """
    A, p, b, _ = nondegenerate(seed, kind, 4)
    W = R.edge_weights(p, A)
    M_ref = R.transition_matrices(W, A)
    W_e = jnp.asarray(W.T)
    M_jax = np.asarray(MG.transition_matrices(
        W_e, jnp.asarray([e[0] for e in A.edges]),
        jnp.asarray([e[1] for e in A.edges]), A.n_states))
    assert np.allclose(M_jax, M_ref, rtol=1e-12, atol=1e-14)

    L = p.shape[0]
    for i in range(L):
        exact = [[Fraction(0)] * A.n_states for _ in range(A.n_states)]
        for src, dst, labels in A.edges:
            exact[src][dst] += sum((Fraction(float(p[i, v])) for v in sorted(labels)),
                                   Fraction(0))
        for s in range(A.n_states):
            for t in range(A.n_states):
                assert M_jax[i, s, t] == pytest.approx(float(exact[s][t]),
                                                       rel=1e-12, abs=1e-15)
    # And the fixture must actually contain a repeated (src, dst) pair, or the
    # `add`-vs-`set` distinction is untested.
    pairs = [(s, d) for s, d, _ in A.edges]
    if kind == "nfa":
        assert len(pairs) != len(set(pairs)), (
            "no parallel edges at this seed: the fixture cannot separate "
            "`.at[].add` from `.at[].set`")


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(3))
def test_jax_scatter_edge_mass_matches_an_exact_rational_scatter(kind, seed):
    """SPEC §2.4's complement-aware inner sum, against exact rationals.

    `test_audit_partition.py` checks this kernel against a float64 dense
    reference and against `math.fsum`; neither is exact, and both were written
    to catch the specific subtraction defect. Here `r_i(v) = Σ_{e : v ∈ label(e)}
    edge_mass[i,e]` is evaluated as a rational sum straight from the definition,
    with the polarity threshold at `|S_c| > 2` so negated classes are forced.
    """
    A, p, b, _ = nondegenerate(seed, kind, 4, require_negated=True)
    L = p.shape[0]
    cid, members, is_neg, indices, indptr = class_tables(A, threshold=2)
    assert is_neg.any(), "regime check: at least one class must be stored negated"
    rng = np.random.default_rng(8300 + seed + SEED_OFFSET)
    edge_mass = rng.random((L, A.n_edges))

    got = np.asarray(MG.scatter_edge_mass_to_tokens(
        jnp.asarray(edge_mass), jnp.asarray(cid), jnp.asarray(indices),
        jnp.asarray(indptr), jnp.asarray(is_neg), len(members), V))

    for i in range(L):
        for v in range(V):
            exact = sum((Fraction(float(edge_mass[i, e]))
                         for e, (_, _, lab) in enumerate(A.edges) if v in lab),
                        Fraction(0))
            assert got[i, v] == pytest.approx(float(exact), rel=1e-12, abs=1e-14), (
                f"r[{i},{v}] = {got[i, v]!r}, exact {float(exact)!r}; "
                f"{int(is_neg.sum())}/{len(members)} classes negated")


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(3))
def test_jax_class_weights_match_an_exact_rational_class_mass(kind, seed):
    """SPEC §4.4 Layers 2–3: `W_c[c, i] = Σ_{v ∈ S_c} p_i(v)`, complement-aware.

    `test_audit_partition.py` checks this against `math.fsum` and a log-space
    reference — both float, and both written after the fact to catch the
    subtract-shaped complement. Here the target is an exact rational sum over
    the class's true members, with the polarity threshold at `|S_c| > 2` so
    negated classes are forced and their `N_c` is as narrow as one token.
    """
    A, p, b, _ = nondegenerate(seed, kind, 4, sigma=25.0, require_negated=True)
    L = p.shape[0]
    cid, members, is_neg, indices, indptr = class_tables(A, threshold=2)
    seg = np.concatenate([np.full(indptr[c + 1] - indptr[c], c, np.int32)
                          for c in range(len(members))]) if indices.size \
        else np.zeros(0, np.int32)
    got = np.asarray(MG.class_weights(
        jnp.asarray(p.T), jnp.asarray(indices), jnp.asarray(seg),
        jnp.asarray(is_neg), len(members)))
    for c, m in enumerate(members):
        for i in range(L):
            exact = sum((Fraction(float(p[i, v])) for v in sorted(m)), Fraction(0))
            assert got[c, i] == pytest.approx(float(exact), rel=1e-12, abs=1e-300), (
                f"W_c[{c},{i}] = {got[c, i]!r}, exact {float(exact)!r} "
                f"(class stored {'negated' if is_neg[c] else 'positive'}, "
                f"|S_c| = {len(m)}, |N_c| = {V - len(m)}, sigma=25)")
            assert got[c, i] > 0.0 or len(m) == 0, (
                f"a non-empty class came back with zero mass — the failure mode "
                f"commit da1294a fixed")


@pytest.mark.parametrize("kind", ["dfa", "nfa"])
@pytest.mark.parametrize("seed", range(3))
def test_the_jax_constrained_marginals_match_the_exact_rational_posterior(kind, seed):
    """SPEC eq (5)/(6) end to end **in JAX**, against exact rationals.

    This is the one composition nothing else checks against ground truth:
    `up_sweep → prefix_suffix → constrained_marginals_and_partition`. Every
    existing check of it is differential — against `reference.py`, or against a
    float64 dense re-derivation — so a defect shared by the reference and the
    kernel is invisible to all of them. `Σ_v q_i(v) == 1` cannot help: the row
    is divided by its own sum.

    The per-position partition `Z_i` is asserted `i`-invariant **and** equal to
    the exact `Z` up to the scale convention, which is the non-vacuous form.
    """
    A, p, b, oracle = nondegenerate(seed, kind, 4, require_negated=True)
    L = p.shape[0]
    cid, members, is_neg, indices, indptr = class_tables(A, threshold=2)
    W = R.edge_weights(p, A)
    M = R.transition_matrices(W, A)

    trm = scans.up_sweep(jnp.asarray(M))
    a, bb, sa, sb = scans.prefix_suffix(trm, jnp.asarray(A.start), jnp.asarray(b))
    q, Z_i = MG.constrained_marginals_and_partition(
        jnp.asarray(p.T), a, bb, jnp.asarray([e[0] for e in A.edges]),
        jnp.asarray([e[1] for e in A.edges]), jnp.asarray(cid),
        jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(is_neg),
        len(members))
    q = np.asarray(q)
    Z_i = np.asarray(Z_i)

    exact = oracle.marginals()
    assert np.allclose(q, exact, rtol=1e-10, atol=1e-13), (
        f"max |q - exact| = {np.max(np.abs(q - exact)):.3e} "
        f"(regime: {kind} L={L} S={A.n_states}, "
        f"{int(is_neg.sum())}/{len(members)} classes negated)")
    assert np.array_equal(q > 0, exact > 0), (
        "the JAX support differs from the exact one: "
        f"{int(np.sum((q > 0) != (exact > 0)))} (position, token) pairs")
    # The non-vacuous invariant (SPEC §2.4): the L unnormalised row sums are the
    # same quantity grouped by a different position, so they must be equal.
    assert np.max(np.abs(Z_i - Z_i[0])) <= 1e-9 * max(abs(Z_i[0]), 1.0), \
        f"Z_i varies with position: {Z_i}"


def test_entropy_from_q_matches_an_exact_rational_entropy_with_forbidden_tokens():
    """SPEC §2.4: `q` has exact zeros, so `log q` is never taken directly, and
    `log(max(q, 1e-30))` must give exactly `0·(−69) = 0` there.

    The existing test asserts only that the result is NaN-free. That passes for
    an entropy that is merely *finite* and wrong. Here the value is pinned to a
    high-precision entropy of the same distribution.
    """
    q = np.zeros((3, 8))
    q[0, [0, 3]] = [0.25, 0.75]
    q[1, :] = 1.0 / 8.0
    q[2, 2] = 1.0
    got = np.asarray(MG.entropy_from_q(jnp.asarray(q)))
    assert np.all(np.isfinite(got))
    for i in range(3):
        exact = -sum(float(x) * math.log(float(x)) for x in q[i] if x > 0)
        assert got[i] == pytest.approx(exact, rel=1e-12, abs=1e-14), \
            f"H(q_{i}) = {got[i]!r}, exact {exact!r}"
    assert got[2] == pytest.approx(0.0, abs=1e-15), \
        "a point mass must have exactly zero entropy"
