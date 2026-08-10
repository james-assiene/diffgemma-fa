"""Audit of the `Z == 0` asymmetry between `--emission=map` and `--emission=sample`.

Written by the TESTER agent of CLAUDE.md's tester → coder → reviewer loop, from
SPEC §2.4, §2.6, §2.7, §3.1b and §6.3 — **not** from the implementation. It
exists because one Countdown arm, run twice on the identical grammar, records
and budget, produced:

    --emission map      Z == 0 on   0/250   CS 1.000
    --emission sample   Z == 0 on  73/250   CS 0.708

(`artifacts/task_countdown_j0{map,sample}_refixed.json`; the two arms cover the
identical 250 record ids, and all 73 of the sample arm's `Z == 0` records come
back from the MAP arm non-empty and `accepted: True`.)

Causes (a) "the automaton is genuinely empty" and (b) "no live continuation of
`L` tokens from `A_k` within budget" are properties of `(automaton, A_k, R)`.
Both emissions are handed the *same* three, and `joint_map`'s feasibility flag
is a function of exactly those and nothing else — see
`test_map_feasibility_does_not_depend_on_p` for why, and note that it is the
1e-30 clamp on `log p` that makes it so. So the MAP arm's 0/250 settles (a) and
(b) for every record individually, whatever each record's `R` happened to be.
What is left is cause **(c)**, and this file is about where the mass goes.

**The oracle this file is built on, and why it needs no ground truth.**
SPEC §3.1b's proposition covers both emission modes because "the argument uses
only the *support* of the constrained posterior, not maximality". The support
is the same set for both. Hence, for any automaton, any `p` and any `R`:

    joint_map reports feasible   ⟹   joint_draw must not report Z == 0

That is a theorem, not a heuristic, and it is checkable without knowing the
right answer. It is only worth anything alongside its converse — a detector
wired to `True` satisfies it vacuously — so the empty-language direction is
asserted here too, and both are mutation-tested (see `_MUTANTS`).

**Shapes.** The real compiled Countdown grammar at the real `L = 256`,
`V = 262,144`, float64, and a minimal fixture at `L = 8`, `|S| = 4`, `V = 64`.
CLAUDE.md warns that "a numerical claim validated only on toy shapes is not
validated"; the finding here runs the *other* way, and that is itself a result:
the defect reproduces identically at both scales, so it was never a large-`L`
phenomenon and no amount of `L = 256` coverage was ever going to be the thing
that caught it.

**Cost, and a host-safety warning.** The Countdown automaton takes ~190 s to
compile (token lift over a 262k vocabulary). It is compiled once and cached
under `artifacts/fa/countdown/` (gitignored); delete that file to force a
rebuild. Run this module on its own — **never** `pytest tests/`, which grabs the
whole GPU — and **always with `-x`**:

    JAX_PLATFORMS=cpu pytest tests/test_audit_partition.py -x -q

> **`-x` is not a convenience here, it is a memory bound.** Several tests hold
> `[L, V]` and `[C, V]` float64 arrays (537 MB and up at `L = 256`,
> `V = 262,144`). pytest keeps every failing test's **traceback frames**, and
> therefore its locals, alive until the session ends. Measured while this file
> was red: a `-x`-less run retaining 22 failures reached **136 GB RSS** and
> OOM-killed an unrelated process on this box. CLAUDE.md already records a
> `--think` regex that OOM-killed this host; this is the same failure with a
> different trigger. Measured green with `-x`: **7.6 GB peak RSS, 67 s**
> (`/usr/bin/time -v`), which is the budget to hold this file to.

**Counting note.** Two tests here are **characterisations, not regressions**:
`test_a_wide_negated_class_cannot_exercise_the_hazard` and
`test_the_driver_is_confidence_in_the_complement_not_L[1e-12-*]` pass both
before and after the fix, by design — the first measures why the existing
fixtures cannot reach the hazard, the second establishes that `L` is not the
variable. Quote this file as "regression tests plus 3 documentary
measurements", never as a flat test count.
"""

from __future__ import annotations

import os
import pathlib
import pickle

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402

from diffgemma_fa.infer import marginals as MG  # noqa: E402
from diffgemma_fa.infer import scans  # noqa: E402
from diffgemma_fa.model import constrained as C  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402

NEG = scans.NEG_SENTINEL
DEAD = NEG / 2.0            # below this is the sentinel, not a real value
L_PROD = 256                # the real canvas_length [V, SPEC §1.2]
V_PROD = 262_144            # the real vocabulary

#: `run_tasks.py` passes `state.terminal_budget(canvas_length)`, which at
#: `max_new_tokens = 256`, `step = 0`, `canvas_length = 256` is **0** — so
#: `b_L(s) = 1[d(s) <= 0] = 1[s in F]` on the first (and for Countdown only)
#: block. Using anything else here would test a budget production never sees.
R_BLOCK0 = 0


# ===========================================================================
# Reproducible mutation registry.  `DGFA_MUT=<name> pytest tests/…`
# ===========================================================================
"""Every mutant used to validate this file, re-runnable by anyone.

Mutations are applied **in process** — no repository file is edited, so this is
safe in a tree several agents are working in (CLAUDE.md forbids
`git checkout`/`stash`/`restore` here).

    DGFA_MUT=feasible_always_true JAX_PLATFORMS=cpu \
        pytest tests/test_audit_partition.py -q

Each mutant names the test it was written to kill. A mutant with **no** kill, or
one whose named victim survives, is a hole in this file.

Measured 2026-08-10 against the fixed implementation (baseline: 43 passed):

    class_weight_subtractive         18 killed
    scatter_subtractive               5 killed   (exactly the 5 ratios)
    map_floor_1e30                    2 killed   (exactly the 2 floor tests)
    feasible_always_true              1 killed
    leaf_no_dead_sentinel             1 killed
    class_weight_complement_blind    16 killed
    map_feasible_reads_p             13 killed
    log_matmul_row_col_anchor         1 killed
    up_sweep_drop_a_level_scale       1 killed
    no_dup_collapse                   5 killed

Two of those rows are load-bearing beyond "it went red". `no_dup_collapse`
leaves `test_class_mass_matches_a_log_space_reference` **alive** at every `eps`
— that fixture is one class holding one token, so it has no duplicate to
collapse and cannot see the failure at all; only the shared-token fixture and
the real grammar catch it. And `class_weight_subtractive` kills the shared-token
test only at `1e-20`/`1e-40` while `no_dup_collapse` kills it at all four
magnitudes, so the two failure modes are distinguishable from the pattern alone
rather than only from the messages.
"""

_MUTANTS: dict[str, str] = {
    # name                        -> the test it must kill  [measured 2026-08-10]
    # -- reverts: the three defects this file was written to find. Each restores
    #    the shipped code as it stood on 2026-08-09, verbatim.
    "class_weight_subtractive":    "test_class_mass_matches_a_log_space_reference",
    "scatter_subtractive":         "test_scatter_keeps_the_mass_of_a_small_negated_class",
    "map_floor_1e30":              "test_map_is_the_argmax_even_when_the_whole_class_sits_below_the_floor",
    # -- and mutants aimed at the tests themselves
    "feasible_always_true":        "test_an_empty_language_is_infeasible_on_both_paths",
    "leaf_no_dead_sentinel":       "test_the_log_tree_root_matches_a_sequential_fold_on_the_real_leaves",
    "class_weight_complement_blind": "test_class_mass_matches_a_log_space_reference",
    "map_feasible_reads_p":        "test_map_feasibility_does_not_depend_on_p",
    "log_matmul_row_col_anchor":   "test_the_log_tree_root_matches_a_sequential_fold_on_the_real_leaves",
    "up_sweep_drop_a_level_scale": "test_the_log_scale_is_accumulated_over_all_2L_minus_1_nodes",
    "no_dup_collapse":             "test_class_mass_with_a_token_shared_by_two_classes",
}

# `leaf_no_dead_sentinel` was WRITTEN to kill the empty-language test and does
# **not**: measured, that test still passes under it. Recorded rather than
# quietly re-aimed, because the reason is a fact about the implementation worth
# knowing — `joint_draw` returns `feasible & valid`, and `sample_tokens`'
# `valid` (every drawn `(s_i, s_{i+1})` must carry an edge admitting some token)
# independently rejects the draw once the sentinel stops marking absent
# transitions. So the spurious-live failure mode has two independent detectors,
# and only `feasible_always_true` disables both. The mutant is kept because it
# does kill the tree-vs-reference test, which is a real kill.

# **On the two `_REPAIRS` this file used to carry.** While the defects were
# live, the registry also held two *repairs* — a naive `O(C·V·L)` dense
# complement for `class_weights`, and a rescaling of `joint_map`'s input to lift
# it clear of the floor. They were diagnostic instruments, not proposals: a red
# test says *something* is wrong, and a repair that turns a specific set green
# says *what*. The dense complement took the file from 18 failures to 2, which
# is what attributed 16 of them to one expression; the rescale flipped exactly
# one test, which is what isolated the floor as a second, independent defect.
# Both are now redundant — the shipped code contains real fixes and the reverts
# above are the durable form of the same evidence — so they are removed rather
# than left as dead configuration.
#
# One of them also caught a bad test, and that is the transferable lesson:
# `test_the_map_floor_is_below_the_marginals...` stayed **red under the repair
# that turned all 32 other tests green**, because it compared against a literal
# `1e-30` and called no implementation function at all. **When a repair fails to
# move a test, that is a fact about the test, not about the repair.**


def _apply_mutation(name: str) -> None:
    if name not in _MUTANTS:
        raise SystemExit(f"unknown DGFA_MUT={name!r}; known: {sorted(_MUTANTS)}")

    if name == "class_weight_subtractive":
        # REVERT to `W_c = total − Σ_{v ∈ N_c} p_i(v)` — `marginals.py` as
        # shipped before 2026-08-10, verbatim. This is the defect that emptied
        # the constrained language on 73/250 Countdown records.
        def cw(p_vl, indices, seg_ids, is_neg, n_classes):
            gathered = p_vl[indices, :]
            partial = jax.ops.segment_sum(gathered, seg_ids,
                                          num_segments=n_classes)
            total = p_vl.sum(axis=0)[None, :]
            return jnp.where(is_neg[:, None], total - partial, partial)

        MG.class_weights = cw
        return

    if name == "map_floor_1e30":
        # REVERT `map_log_floor` to the constant it was. Every `p` below 1e-30
        # then scores the same and `argmax` ties to the lowest token id.
        C.map_log_floor = lambda dtype: 1e-30
        return

    if name == "feasible_always_true":
        # The detector wired shut. Every "must be infeasible" assertion here
        # exists to stop the theorem test being satisfiable this way.
        odraw, omap = C.joint_draw, C.joint_map

        def jdraw(*a, **k):
            t, _ok = odraw(*a, **k)
            return t, jnp.bool_(True)

        def jmap(*a, **k):
            t, _ok = omap(*a, **k)
            return t, jnp.bool_(True)

        C.joint_draw, C.joint_map = jdraw, jmap

    elif name == "leaf_no_dead_sentinel":
        # Drop the `M > 0` guard, so a structurally absent transition becomes
        # `log(tiny) = -708` rather than the sentinel and every state pair looks
        # reachable. This is the "spurious live entry" failure, which is worse
        # than losing one.
        oup = scans.up_sweep_log

        def up(logM):
            return oup(jnp.maximum(logM, -708.0))

        scans.up_sweep_log = up

    elif name == "class_weight_complement_blind":
        # A plain sparse scatter: negated classes return the mass of the set
        # they *store* rather than of its complement. SPEC §2.4 says only the
        # exactness suite catches this; it must be caught here too, because the
        # class mass is the quantity this file blames.
        ocw = MG.class_weights

        def cw(p_vl, indices, seg_ids, is_neg, n_classes):
            return ocw(p_vl, indices, seg_ids,
                       jnp.zeros_like(is_neg), n_classes)

        MG.class_weights = cw

    elif name == "scatter_subtractive":
        # `r = neg_total + Σ signed`, the pre-fix shape: every negated class's
        # mass is added and the ones storing the token are subtracted back off.
        # Verbatim from `marginals.py` before 2026-08-10.
        def sc(edge_mass, class_id, indices, indptr, is_neg, n_classes,
               vocab_size):
            L = edge_mass.shape[0]
            U = jax.ops.segment_sum(edge_mass.T, class_id,
                                    num_segments=n_classes).T
            neg_total = jnp.where(is_neg[None, :], U, 0.0).sum(axis=1)
            signed = jnp.where(is_neg[None, :], -U, U)
            seg = jnp.repeat(jnp.arange(n_classes, dtype=jnp.int32),
                             jnp.diff(indptr),
                             total_repeat_length=indices.shape[0])
            r = jnp.zeros((L, vocab_size), dtype=edge_mass.dtype).at[
                :, indices].add(signed[:, seg])
            return r + neg_total[:, None]

        MG.scatter_edge_mass_to_tokens = sc

    elif name == "no_dup_collapse":
        # Drop the representative-slot collapse: a token stored by two classes
        # is then counted once per CSR slot in the complement sum. Everything
        # else is the shipped algebra, so a survivor here means no fixture in
        # the file has a cross-class duplicate.
        def cw(p_vl, indices, seg_ids, is_neg, n_classes):
            V = p_vl.shape[0]
            nnz = indices.shape[0]
            gathered = p_vl[indices, :]
            partial = jax.ops.segment_sum(gathered, seg_ids,
                                          num_segments=n_classes)
            in_csr = jnp.zeros((V,), dtype=bool).at[indices].set(True)
            outside = (~in_csr).astype(p_vl.dtype) @ p_vl
            slot = jnp.arange(nnz, dtype=jnp.int32)          # <- rep := slot
            in_class = jnp.zeros((n_classes, nnz), dtype=bool).at[
                seg_ids, slot].set(True)
            outside_class = (~in_class[:, slot]).astype(p_vl.dtype)
            complement = outside[None, :] + outside_class @ gathered
            return jnp.where(is_neg[:, None], complement, partial)

        MG.class_weights = cw

    elif name == "map_feasible_reads_p":
        # Make MAP's feasibility a function of `p`. If this survives, the
        # witness argument ("MAP found a decode, so the language is non-empty")
        # is not actually pinned by anything in this file.
        omap = C.joint_map

        def jmap(p_lv, aut, remaining, n_states, n_classes):
            t, ok = omap(p_lv, aut, remaining, n_states, n_classes)
            return t, ok & (p_lv.max() < 0.9)

        C.joint_map = jmap

    elif name == "log_matmul_row_col_anchor":
        # Historical kernel #1 (SPEC §2.6, `log_matmul`'s own docstring): a
        # single row/column max shift instead of the pairwise max.
        def lm(A, B):
            ra = jnp.max(A, axis=-1)
            cb = jnp.max(B, axis=-2)
            sh = ra[..., :, None] + cb[..., None, :]
            s = jnp.sum(jnp.exp(A[..., :, :, None] + B[..., None, :, :]
                                - sh[..., :, None, :]), axis=-2)
            tiny = jnp.asarray(jnp.finfo(A.dtype).tiny, dtype=A.dtype)
            out = sh + jnp.log(jnp.maximum(s, tiny))
            return jnp.maximum(out, jnp.asarray(NEG, dtype=A.dtype))

        scans.log_matmul = lm

        def up(logM):
            cur = logM
            levels, sc = [cur], [jnp.zeros((logM.shape[0],), logM.dtype)]
            while cur.shape[0] > 1:
                cur = lm(cur[0::2], cur[1::2])
                levels.append(cur)
                sc.append(jnp.zeros((cur.shape[0],), logM.dtype))
            return scans.TreeLevels(levels=tuple(levels), log_scales=tuple(sc))

        scans.up_sweep_log = up

    elif name == "up_sweep_drop_a_level_scale":
        # Accumulate the log-scale over every node **except** the `L` leaves.
        # SPEC §2.6: "Accumulate `Σ log scale` over **all** `2L−1` nodes."
        oup = scans.up_sweep

        def up(M, *, normalize=True):
            tr = oup(M, normalize=normalize)
            leaf = tr.log_scales[0]
            out = [jnp.zeros_like(leaf)]
            acc = leaf
            for k in range(1, len(tr.log_scales)):
                acc = acc[0::2] + acc[1::2]
                out.append(tr.log_scales[k] - acc)
            return scans.TreeLevels(levels=tr.levels, log_scales=tuple(out))

        scans.up_sweep = up


if os.environ.get("DGFA_MUT"):
    _apply_mutation(os.environ["DGFA_MUT"])


# ===========================================================================
# Independent references. numpy float64, log space, no code shared with the
# implementation.
# ===========================================================================

def dense_membership(indices, indptr, is_neg, n_classes, vocab) -> np.ndarray:
    """`[C, V] bool` membership, straight from SPEC §4.4's class definition.

    A positive class *is* its stored set; a negated class is the complement of
    its stored set. That is the definition, so writing it out is not a
    re-implementation of the complement algebra — it is the specification the
    complement algebra is supposed to compute.
    """
    indices = np.asarray(indices)
    indptr = np.asarray(indptr)
    is_neg = np.asarray(is_neg)
    member = np.zeros((n_classes, vocab), dtype=bool)
    for c in range(n_classes):
        member[c, indices[indptr[c]:indptr[c + 1]]] = True
    member[is_neg[:n_classes]] = ~member[is_neg[:n_classes]]
    return member


def reference_log_class_mass(logp_lv: np.ndarray, member: np.ndarray) -> np.ndarray:
    """`log W_c[c, i] = log Σ_{v ∈ S_c} p_i(v)`, evaluated in **log space**.

    SPEC §4.4 defines `W_c` as a sum of probabilities; this computes exactly
    that quantity via `logsumexp`, which is the same real number and cannot
    cancel. `-inf` iff the class is empty or every member has `p = 0`, i.e.
    iff the quantity really is zero.

    Args:
      logp_lv: `[L, V]` natural logs of the marginals.
      member: `[C, V]` bool.

    Returns:
      `[C, L]` float64.
    """
    out = np.full((member.shape[0], logp_lv.shape[0]), -np.inf)
    for c in range(member.shape[0]):
        sel = logp_lv[:, member[c]]                      # [L, |S_c|]
        if sel.shape[1] == 0:
            continue
        mx = sel.max(axis=1)
        finite = mx > -np.inf
        acc = np.full(logp_lv.shape[0], -np.inf)
        acc[finite] = mx[finite] + np.log(
            np.exp(sel[finite] - mx[finite][:, None]).sum(axis=1))
        out[c] = acc
    return out


def reference_log_leaves(log_W_c: np.ndarray, edge_src, edge_dst, edge_class,
                         n_states: int) -> np.ndarray:
    """`[L, S, S]` log-space `M_i(s,s') = log Σ_{e: src,dst} W[i, e]`. SPEC eq (2).

    `.at[].add` on the linear side is a sum over **parallel** edges (path
    multiplicity on an NFA); here that is a `logaddexp` accumulation, which is
    the same number computed where it cannot cancel or underflow.
    """
    L = log_W_c.shape[1]
    M = np.full((L, n_states, n_states), -np.inf)
    for e, (s, d, c) in enumerate(zip(np.asarray(edge_src), np.asarray(edge_dst),
                                      np.asarray(edge_class))):
        M[:, s, d] = np.logaddexp(M[:, s, d], log_W_c[c])
    return np.where(np.isfinite(M), M, NEG)


def sequential_log_fold(mats: np.ndarray) -> np.ndarray:
    """`log(M_0 M_1 ⋯ M_{L-1})` folded one matrix at a time, float64.

    Deliberately the dumbest correct thing: no tree, no banding, each output
    entry anchored on its own running max over the contracted axis. This is the
    reference side of the differential test and it must stay this boring
    (CLAUDE.md: correctness before speed).
    """
    acc = np.asarray(mats[0], dtype=np.float64).copy()
    for i in range(1, mats.shape[0]):
        B = np.asarray(mats[i], dtype=np.float64)
        terms = acc[:, :, None] + B[None, :, :]
        m = terms.max(axis=1)
        live = m > DEAD
        anchor = np.where(live, m, 0.0)
        s = np.exp(terms - anchor[:, None, :]).sum(axis=1)
        acc = np.where(live, anchor + np.log(np.maximum(s, 1e-320)), NEG)
    return acc


def reference_log_partition(log_leaves: np.ndarray, active: np.ndarray,
                            d: np.ndarray, remaining: int) -> float:
    """`log Z` for SPEC §3.1b's `a_start = 1[s ∈ A_k]`, `b_L = 1[d(s) ≤ R]`.

    Returns `-inf` iff the constrained language is empty at this canvas length
    and budget — i.e. iff CLAUDE.md's cause (a) or (b) holds. Nothing else in
    this function can produce `-inf`: it is a `logsumexp` throughout.
    """
    root = sequential_log_fold(log_leaves)
    a = np.where(np.asarray(active), 0.0, -np.inf)
    b = np.where(np.asarray(d) <= remaining, 0.0, -np.inf)
    joint = a[:, None] + np.where(root > DEAD, root, -np.inf) + b[None, :]
    if not np.isfinite(joint).any():
        return -np.inf
    mx = joint[np.isfinite(joint)].max()
    return float(mx + np.log(np.exp(joint[np.isfinite(joint)] - mx).sum()))


# ===========================================================================
# Fixtures
# ===========================================================================

_CACHE = pathlib.Path("/home/ubuntu/diffgemma_fa/artifacts/fa/countdown")
_CACHE_FILE = _CACHE / "countdown_maxsteps4_maxvalue999.pkl"


def _compile_countdown():
    """The **production** Countdown grammar, exactly as `run_tasks.load_task`
    builds it: `countdown_regex(max_steps=4, max_value=999, step_separator=r"\\n")`
    through `pipeline.compile_regex` with the default 8-token channel header."""
    from diffgemma_fa.compile import pipeline
    from diffgemma_fa.compile.tasks.grammars import countdown_regex

    return pipeline.compile_regex(
        countdown_regex(max_steps=4, max_value=999, step_separator=r"\n"),
        name="countdown", header_tokens=8).automaton


@pytest.fixture(scope="session")
def countdown():
    """`CompiledAutomaton` for Countdown, cached on disk (~190 s to build)."""
    if _CACHE_FILE.exists():
        with _CACHE_FILE.open("rb") as f:
            return pickle.load(f)
    a = _compile_countdown()
    _CACHE.mkdir(parents=True, exist_ok=True)
    with _CACHE_FILE.open("wb") as f:
        pickle.dump(a, f)
    return a


def traced(a, *, active=None) -> Automaton:
    """`compile.CompiledAutomaton -> model.state.Automaton`, as `run_tasks.py`
    does it (batch dim dropped; the kernels are `vmap`-ed over it upstream)."""
    return Automaton(
        edge_src=jnp.asarray(a.edge_src), edge_dst=jnp.asarray(a.edge_dst),
        edge_class=jnp.asarray(a.edge_class),
        edge_valid=jnp.ones(a.n_edges, dtype=bool),
        csr_indices=jnp.asarray(a.tables.sum_indices),
        csr_indptr=jnp.asarray(a.tables.sum_indptr),
        is_neg=jnp.asarray(a.tables.sum_is_neg),
        d=jnp.asarray(a.d), is_final=jnp.asarray(a.is_final),
        active=jnp.asarray(a.start_vector if active is None else active))


def gemma_shaped_logits(L: int, V: int, *, seed: int = 0, spread: float = 8.0,
                        softcap: float = 30.0, temperature: float = 0.408
                        ) -> np.ndarray:
    """Logits in the shape the real sampler produces.

    `gemma/diffusion/_transformer.py:182` caps the final logits at
    `30·tanh(x/30)`, and `gemma/diffusion/_sampler.py:236`'s schedule bottoms
    out at `min_temperature = 0.4` (0.408 at the last step). The span is
    therefore bounded at `2·softcap/T ≈ 147` nats, which matters: a plain
    Gaussian fixture at scale 40 spans 458 nats and would make `p` underflow for
    reasons production cannot reach.
    """
    rng = np.random.default_rng(seed)
    return softcap * np.tanh(rng.standard_normal((L, V)) * spread / softcap) \
        / temperature


def softmax64(logits: np.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(jnp.asarray(logits, dtype=jnp.float64), axis=-1)


# --- the minimal fixture ---------------------------------------------------

MIN_V, MIN_S, MIN_L = 64, 4, 8


def minimal_automaton(n_cycle: int = 2) -> Automaton:
    """An `n_cycle`-cycle over a single **negated** class `N = {0}`.

    The class means "any token except 0", which is the shape SPEC §4.4 says the
    representation exists for: real grammars store the complement because their
    classes cover ~99.6% of the vocabulary, and the Countdown grammar's
    channel-header class stores just 7 ids.

    State 0 is the only accepting state and `d(s) = (n − s) mod n`, so with
    `R = 0` the terminal factor admits state 0 alone and the language at canvas
    length `L` is non-empty **iff** `L mod n_cycle == 0`. At `n_cycle = 2`,
    `MIN_L = 8` gives `63^8 ≈ 2.5e14` members; at `n_cycle = 3` the same `L = 8`
    gives none, which is the empty-language control. (`L` must stay a power of
    two — SPEC §2.6(d) — so the emptiness is arranged through the cycle length,
    not through an odd `L`.)
    """
    src = np.arange(n_cycle, dtype=np.int32)
    dst = ((src + 1) % n_cycle).astype(np.int32)
    d = np.full(MIN_S, 10 ** 6, dtype=np.int32)
    d[:n_cycle] = [(n_cycle - s) % n_cycle for s in range(n_cycle)]
    is_final = np.zeros(MIN_S, dtype=bool)
    is_final[0] = True
    active = np.zeros(MIN_S, dtype=bool)
    active[0] = True
    return Automaton(
        edge_src=jnp.asarray(src), edge_dst=jnp.asarray(dst),
        edge_class=jnp.zeros(n_cycle, jnp.int32),
        edge_valid=jnp.ones(n_cycle, dtype=bool),
        csr_indices=jnp.asarray([0], jnp.int32),
        csr_indptr=jnp.asarray([0, 1], jnp.int32),
        is_neg=jnp.asarray([True]),
        d=jnp.asarray(d), is_final=jnp.asarray(is_final),
        active=jnp.asarray(active))


def minimal_p(eps: float, L: int = MIN_L, V: int = MIN_V) -> jnp.ndarray:
    """`p_i(0) = 1 − eps`, the remaining `eps` spread over tokens `1..V−1`.

    So the class `N = {0}`'s complement carries **exactly `eps`** of the mass at
    every position, and the constrained language's true `log Z` is
    `L·log(eps/(V−1)) + log((V−1)^L)` — finite for every `eps > 0`.
    """
    p = np.full((L, V), eps / (V - 1), dtype=np.float64)
    p[:, 0] = 1.0 - eps
    return jnp.asarray(p, dtype=jnp.float64)


# ===========================================================================
# 1. Fixture integrity — the premises this file's conclusions rest on
# ===========================================================================

def test_the_countdown_fixture_is_the_grammar_the_arm_actually_ran(countdown):
    """If this drifts, every number below is about a different automaton."""
    assert countdown.n_states == 124, countdown.n_states
    assert countdown.n_states_bucket == 128
    assert countdown.tables.n_classes == 24
    assert countdown.n_edges == 268


def test_the_countdown_grammar_has_a_small_negated_class_on_a_mandatory_path(countdown):
    """The structural precondition for the mechanism this file measures.

    SPEC §4.4 stores a class by its complement when the class is wide; §3.6's
    channel header is `100 · Σ_name{1,8} · 107 · 101` with `Σ_name` the
    complement of the reserved tokens, so `|N_c|` is a handful. Every accepted
    string must traverse those edges — the header is prepended unconditionally
    (`pipeline.compile_regex(..., channel_header=True)`), so if that class's
    weight is lost the whole language is lost.
    """
    is_neg = np.asarray(countdown.tables.sum_is_neg)
    indptr = np.asarray(countdown.tables.sum_indptr)
    sizes = np.diff(indptr)
    neg = np.flatnonzero(is_neg[:countdown.tables.n_classes])
    assert neg.size >= 1, "no negated class: the complement path is untested"
    small = [c for c in neg if 0 < sizes[c] <= 16]
    assert small, (
        f"expected a negated class with a small stored set; sizes {sizes[neg]}"
    )
    cls = np.asarray(countdown.edge_class)
    used = [c for c in small if (cls == c).any()]
    assert used, "the small negated class is on no edge"


# ===========================================================================
# 2. The theorem: MAP feasible ⟹ the draw must not report Z == 0
# SPEC §3.1b — "the argument uses only the *support* of the constrained
# posterior, not maximality, which is why it covers both emission modes".
# ===========================================================================

def test_map_feasibility_does_not_depend_on_p():
    """Why a MAP decode is a *witness* that the language is non-empty.

    SPEC §2.7 builds `M̃_i(s,s') = max_e max_{v ∈ label(e)} log p_i(v)`. A max
    over a non-empty class of a finite `log p` is finite, so no `p` can remove a
    structurally present edge from the max-plus tree. Therefore `joint_map`'s
    feasibility flag is a statement about the **automaton and the budget alone**
    — which is exactly what makes "MAP produced a decode" a proof that
    CLAUDE.md's causes (a) and (b) do not hold.

    Adversarial case: a `p` that is a point mass on a token **no class
    contains**. If MAP's feasibility leaked any dependence on `p`, that is where
    it would show.
    """
    aut = minimal_automaton()
    flags = []
    for p in [minimal_p(0.5),
              minimal_p(1e-40),
              jnp.asarray(np.full((MIN_L, MIN_V), 1.0 / MIN_V)),
              jnp.asarray(np.eye(MIN_V, dtype=np.float64)[
                  np.zeros(MIN_L, dtype=int)])]:      # all mass on token 0 ∉ S_c
        flags.append(bool(C.joint_map(p, aut, jnp.asarray(0),
                                      n_states=MIN_S, n_classes=1)[1]))
    assert flags == [True] * 4, (
        f"joint_map's feasibility changed with `p` ({flags}). The witness "
        f"argument used to rule out causes (a)/(b) — 'MAP emitted an accepted "
        f"decode on all 73 affected records, so the language is non-empty' — "
        f"is only valid if this flag is a property of the automaton."
    )


@pytest.mark.parametrize("eps", [1e-1, 1e-6, 1e-14, 1e-16, 1e-20, 1e-40])
def test_map_feasible_implies_draw_feasible__minimal(eps):
    """SPEC §3.1b, at `L = 8`, `|S| = 4`, `V = 64`.

    The language `{x : x_i ≠ 0}` is non-empty for **every** `eps > 0` and has
    the same `2.5e14` members at every `eps`; only the probability of each
    member changes. `log Z` is `≈ 8·log(eps/63) + 8·log 63`, i.e. `8·log eps` —
    finite and comfortably inside float64 for every value above (at `eps = 1e-40`
    that is `-737` nats, and the log-space tree SPEC §2.6 mandates represents it
    exactly).

    So a `Z == 0` here is not "the mass is too small to represent". It is mass
    that was destroyed before it reached the representation that was chosen to
    hold it.
    """
    aut = minimal_automaton()
    p = minimal_p(eps)
    _t, ok_map = C.joint_map(p, aut, jnp.asarray(0), n_states=MIN_S, n_classes=1)
    _t, ok_draw = C.joint_draw(p, aut, jnp.asarray(0), jax.random.key(0),
                               n_states=MIN_S, n_classes=1)
    assert bool(ok_map), "fixture broken: MAP must find a decode here"
    assert bool(ok_draw), (
        f"eps = {eps:g}: joint_map found a decode, so the constrained language "
        f"is non-empty within budget — but joint_draw reports Z == 0. "
        f"log Z is about {MIN_L * np.log(eps):.1f} nats, which float64 holds "
        f"with ~700 nats to spare."
    )


def test_map_feasible_implies_draw_feasible__real_countdown_at_L256(countdown):
    """The same theorem on the production grammar, `L = 256`, `V = 262,144`,
    float64, `R = 0` (what `terminal_budget` yields on Countdown's only block).

    `p` is Gemma-shaped — softcapped at 30 and divided by the schedule's final
    temperature — with the model **confident about a reserved token** at one
    header position. Phase 0 measured real final-step entropies of 5e-5 to 4e-4
    nats, so a near-point-mass is the ordinary case, not an adversarial one.
    """
    aut = traced(countdown)
    lg = gemma_shaped_logits(L_PROD, V_PROD, seed=0)
    lg[1, :] = 0.0
    lg[1, 107] = 60.0          # `<channel|>`: reserved, i.e. NOT in the name class
    p = softmax64(lg)

    _t, ok_map = C.joint_map(p, aut, jnp.asarray(R_BLOCK0),
                             n_states=countdown.n_states_bucket,
                             n_classes=countdown.tables.n_classes)
    _t, ok_draw = C.joint_draw(p, aut, jnp.asarray(R_BLOCK0), jax.random.key(0),
                               n_states=countdown.n_states_bucket,
                               n_classes=countdown.tables.n_classes)
    assert bool(ok_map), "fixture broken: MAP must find a decode on this grammar"
    assert bool(ok_draw), (
        "the 73/250 arm, reproduced without the model: joint_map decodes and "
        "joint_draw reports Z == 0 on the identical automaton, budget and `p`."
    )


def test_the_draw_agrees_with_an_independent_log_space_partition(countdown):
    """The theorem again, against an oracle that does not involve MAP at all.

    `reference_log_partition` computes `log Z` in numpy, in log space, from the
    class definitions — sharing no code with `marginals`, `scans` or `tree`. It
    returns `-inf` **iff** the constrained language is empty, which is the only
    thing that may make the draw infeasible.
    """
    aut = traced(countdown)
    lg = gemma_shaped_logits(L_PROD, V_PROD, seed=0)
    lg[1, :] = 0.0
    lg[1, 107] = 60.0
    p = softmax64(lg)

    member = dense_membership(countdown.tables.sum_indices,
                              countdown.tables.sum_indptr,
                              countdown.tables.sum_is_neg,
                              countdown.tables.n_classes, V_PROD)
    logp = np.log(np.asarray(p, dtype=np.float64))
    log_W = reference_log_class_mass(logp, member)
    leaves = reference_log_leaves(log_W, countdown.edge_src, countdown.edge_dst,
                                 countdown.edge_class,
                                 countdown.n_states_bucket)
    logZ = reference_log_partition(leaves, np.asarray(countdown.start_vector),
                                   np.asarray(countdown.d), R_BLOCK0)
    assert np.isfinite(logZ), (
        "fixture broken: the reference says the language IS empty here, which "
        "would make this cause (b) rather than (c)"
    )

    _t, ok = C.joint_draw(p, aut, jnp.asarray(R_BLOCK0), jax.random.key(0),
                          n_states=countdown.n_states_bucket,
                          n_classes=countdown.tables.n_classes)
    assert bool(ok), (
        f"log Z = {logZ:.1f} nats by an independent float64 log-space "
        f"computation — finite, and {abs(logZ):.0f} nats is nowhere near "
        f"float64's ~709-nat exp range in the log domain — yet the kernel "
        f"reports Z == 0."
    )


def test_an_empty_language_is_infeasible_on_both_paths():
    """The converse, without which every test above is satisfied by a detector
    wired to `True`.

    A 3-cycle with `R = 0` accepts only lengths divisible by 3, and `8 % 3 = 2`,
    so the constrained language really is empty here — a genuine cause (a)/(b),
    which both emissions must report. `p` is benign (`eps = 0.5`), so nothing
    numerical is involved and the two flags are answering the structural
    question alone.
    """
    aut = minimal_automaton(n_cycle=3)
    p = minimal_p(0.5)
    _t, ok_map = C.joint_map(p, aut, jnp.asarray(0), n_states=MIN_S, n_classes=1)
    _t, ok_draw = C.joint_draw(p, aut, jnp.asarray(0), jax.random.key(0),
                               n_states=MIN_S, n_classes=1)
    assert not bool(ok_map), "MAP reported support on an empty language"
    assert not bool(ok_draw), "the draw reported support on an empty language"


def test_the_empty_language_control_is_a_language_property_not_a_p_property():
    """Keeps the control above honest: flip **only** the cycle length and the
    same `p`, the same shapes and the same budget must become feasible.

    Without this, `test_an_empty_language_is_infeasible_on_both_paths` could be
    passing because something unrelated (a shape, a bucket, `R = 0`) makes that
    fixture infeasible for a reason that has nothing to do with emptiness.
    """
    p = minimal_p(0.5)
    live = minimal_automaton(n_cycle=2)     # 8 % 2 == 0 -> non-empty
    _t, ok_map = C.joint_map(p, live, jnp.asarray(0), n_states=MIN_S, n_classes=1)
    _t, ok_draw = C.joint_draw(p, live, jnp.asarray(0), jax.random.key(0),
                               n_states=MIN_S, n_classes=1)
    assert bool(ok_map) and bool(ok_draw), (
        "the only difference from the empty control is the cycle length, so "
        "this must be feasible — otherwise that control proves nothing"
    )


# ===========================================================================
# 3. The mechanism — where the mass is destroyed
# SPEC §2.4: `W_c[c,i] = Σ_{v ∈ S_c} p_i(v)`, complement-aware.
# ===========================================================================

@pytest.mark.parametrize("eps", [1e-1, 1e-8, 1e-14, 1e-16, 1e-20, 1e-40, 1e-300])
def test_class_mass_matches_a_log_space_reference(eps):
    """`W_c` must be the class mass, to a relative accuracy, for every `eps`.

    SPEC §2.4 gives the complement-aware form
    `W_c = total − Σ_{v ∈ N_c} p_i(v)` for a negated class and reports it
    "verified against a direct scatter to 1.1e-16 **with mixed polarity**".
    That verification is an *absolute* one: at `|W_c| ~ 1`, agreeing to 1.1e-16
    says nothing about a class whose true mass is 1e-20, where the same
    absolute error is a 1e4 relative error and the subtraction has no correct
    digits left.

    The reference here is `logsumexp` over the class's members — the same real
    number, computed where cancellation cannot occur.
    """
    p = minimal_p(eps)
    logp = np.log(np.asarray(p))
    member = dense_membership([0], [0, 1], [True], 1, MIN_V)
    ref_log = reference_log_class_mass(logp, member)[0]        # [L]

    W = np.asarray(MG.class_weights(p.T, jnp.asarray([0], jnp.int32),
                                    jnp.zeros(1, jnp.int32),
                                    jnp.asarray([True]), 1))[0]
    assert np.isfinite(ref_log).all(), "fixture broken: the class is non-empty"
    assert (W > 0).all(), (
        f"eps = {eps:g}: the class mass is {np.exp(ref_log[0]):.3e} "
        f"(log {ref_log[0]:.2f}) but `class_weights` returned {W[0]:.3e}. A "
        f"zero here is not underflow — float64 holds 1e-300 — it is the "
        f"cancellation in `total - partial`."
    )
    # 1e-4 nats is a 0.01% relative error in a transition weight — far looser
    # than anything a log-space computation would produce, and deliberately so:
    # the point is not tightness, it is that `total − partial` loses *relative*
    # accuracy in proportion to how small the answer is (~1.1e-16/eps), so it
    # blows past any fixed relative bound once the class mass is small enough.
    rel = np.abs(np.log(W) - ref_log)
    assert rel.max() < 1e-4, (
        f"eps = {eps:g}: class mass off by {rel.max():.3g} nats "
        f"({np.expm1(rel.max()) * 100:.3g}% relative). This is the *survivor* "
        f"error — an edge that is still live but misweighted, which trips no "
        f"detector at all."
    )


def test_edge_weights_match_a_log_space_reference_on_the_real_grammar(countdown):
    """`W[i, e] = Σ_{v ∈ label(e)} p_i(v)` on the production grammar, at the
    production shapes, against `logsumexp` over the class's members.

    Separates the two things CLAUDE.md's cause (c) can mean. (c) is written as
    "fp32 underflow on an unnormalized path — expected, and SPEC §6.3
    deliberately tests for it", remedy "your scaling is missing" — a framing
    that assumes the lost quantity was **unrepresentable**. The failure message
    reports, for the worst edge, whether the true mass was inside float64's
    normal range. If it was, no wider float and no rescaling of the *tree*
    recovers it: a subtraction of two numbers equal in all 53 bits has already
    thrown the information away.
    """
    lg = gemma_shaped_logits(L_PROD, V_PROD, seed=0)
    lg[1, :] = 0.0
    lg[1, 107] = 60.0
    p = softmax64(lg)

    member = dense_membership(countdown.tables.sum_indices,
                              countdown.tables.sum_indptr,
                              countdown.tables.sum_is_neg,
                              countdown.tables.n_classes, V_PROD)
    ref_log = reference_log_class_mass(np.log(np.asarray(p)), member)  # [C, L]

    aut = traced(countdown)
    _pvl, W_e, _M = C._matrices(p, aut, countdown.n_states_bucket,  # noqa: SLF001
                                countdown.tables.n_classes)
    cls = np.asarray(countdown.edge_class)
    W_e = np.asarray(W_e)                                       # [E, L]
    ref_edge = ref_log[cls]                                     # [E, L]

    with np.errstate(divide="ignore", invalid="ignore"):
        got_log = np.where(W_e > 0, np.log(np.maximum(W_e, 1e-320)), -np.inf)
    err = np.where(np.isfinite(ref_edge), np.abs(got_log - ref_edge), 0.0)
    worst = np.unravel_index(np.nanargmax(np.where(np.isnan(err), -1.0, err)),
                             err.shape)
    e, i = int(worst[0]), int(worst[1])
    tiny_log = float(np.log(np.finfo(np.float64).tiny))
    assert float(np.nanmax(err)) < 1e-9, (
        f"edge {e} at position {i} (class {cls[e]}, "
        f"{'negated' if bool(np.asarray(countdown.tables.sum_is_neg)[cls[e]]) else 'positive'}"
        f") was given weight {W_e[e, i]:.6e}, but its true mass is "
        f"exp({ref_edge[e, i]:.2f}) = {np.exp(ref_edge[e, i]):.3e}. That is "
        f"{ref_edge[e, i] - tiny_log:.0f} nats above float64's smallest normal, "
        f"i.e. representable — so this is destroyed information, not underflow."
    )


def test_every_structurally_present_edge_is_live_in_the_leaf_log_matrices(countdown):
    """SPEC §2.6/§3.1b: the constrained support is the automaton's support.

    `M_i(s,s') = Σ_e Σ_{v ∈ label(e)} p_i(v)` is strictly positive whenever an
    edge exists and `p > 0` everywhere, which a softmax guarantees. So every
    `(i, s, s')` carrying an edge must come out of the leaf construction as a
    live log value, never as the "impossible" sentinel. An edge that dies here
    is a transition the sampler will never take, on an automaton that says it
    can — and on Countdown those edges are on the mandatory channel header, so
    losing them empties the whole language.
    """
    lg = gemma_shaped_logits(L_PROD, V_PROD, seed=0)
    lg[1, :] = 0.0
    lg[1, 107] = 60.0
    p = softmax64(lg)
    assert float(np.asarray(p).min()) > 0.0, (
        "fixture broken: `p` itself has exact zeros, which would make this a "
        "different (and legitimate) reason for a dead edge"
    )

    aut = traced(countdown)
    _pvl, _W, M = C._matrices(p, aut, countdown.n_states_bucket,  # noqa: SLF001
                              countdown.tables.n_classes)
    M = np.asarray(M)
    src = np.asarray(countdown.edge_src)
    dst = np.asarray(countdown.edge_dst)
    # `joint_draw` turns a non-positive `M` into the "impossible" sentinel, so
    # `M <= 0` on a real edge is exactly a transition erased from the automaton.
    dead = M[:, src, dst] <= 0.0                                # [L, E]
    n_dead = int(dead.sum())
    first = np.argwhere(dead)
    assert n_dead == 0, (
        f"{n_dead} of {dead.size} (position, edge) slots came back as "
        f"non-positive although the edge exists and `p > 0` everywhere, so "
        f"they become the impossible sentinel in the log tree; first at "
        f"position {int(first[0][0])}, edge {int(first[0][1])} "
        f"(class {int(np.asarray(countdown.edge_class)[first[0][1]])})."
    )


# ===========================================================================
# 4. The tree is not where the mass goes — isolate the blame
# SPEC §2.4/§2.6: normalize every node, sum the log-scale over all 2L−1 nodes.
# ===========================================================================

def test_the_log_scale_is_accumulated_over_all_2L_minus_1_nodes():
    """SPEC §2.6: "Accumulate `Σ log scale` over **all** `2L−1` nodes."

    Asserted as the property it exists for rather than by counting call sites:
    the normalised root plus its accumulated log-scale must equal the
    *unnormalised* product, computed independently. Dropping any node's
    normaliser — leaves included — breaks this by exactly that node's scale, so
    the assertion covers all `2L−1` of them at once and stays true of any
    correct implementation, including one that normalises differently.
    """
    L, S = 64, 6
    rng = np.random.default_rng(11)
    M = rng.random((L, S, S)) * np.exp(rng.normal(-4.0, 3.0, (L, S, S)))
    M[M < 1e-3] = 0.0                       # some structurally absent transitions

    tr = scans.up_sweep(jnp.asarray(M), normalize=True)
    assert tr.n_nodes == 2 * L - 1, (
        f"{tr.n_nodes} nodes, expected 2L-1 = {2 * L - 1}; the dyadic shape is "
        f"wrong and the scale sum cannot be over the node set SPEC names"
    )

    logM = np.where(M > 0, np.log(np.maximum(M, np.finfo(np.float64).tiny)), NEG)
    ref = sequential_log_fold(logM)
    got = np.where(np.asarray(tr.root) > 0,
                   np.log(np.maximum(np.asarray(tr.root), 1e-320))
                   + float(np.asarray(tr.root_log_scale)), NEG)

    live = ref > DEAD
    assert (np.asarray(tr.root)[live] > 0).all(), (
        "a live entry of the reference product came back as an exact zero from "
        "the normalised tree"
    )
    err = float(np.abs(got[live] - ref[live]).max())
    assert err < 1e-8, (
        f"root magnitude off by {err:.3g} nats once the accumulated log-scale "
        f"is added back — some node's normaliser is not in the sum"
    )


def test_the_log_tree_root_matches_a_sequential_fold_on_the_real_leaves(countdown):
    """The log-space tree, differential-tested against a plain sequential fold
    on the **real** Countdown leaves at `L = 256`.

    This is the control for everything above: if the tree were the lossy stage,
    it would show here. `up_sweep_log` carries no normalisation at all (log
    space removes the question, SPEC §2.6), so this test *is* the whole
    statement about scaling on the `joint_draw` path — and it is the leaves,
    built one stage earlier in linear space, that are outside it.
    """
    lg = gemma_shaped_logits(L_PROD, V_PROD, seed=3)
    p = softmax64(lg)
    member = dense_membership(countdown.tables.sum_indices,
                              countdown.tables.sum_indptr,
                              countdown.tables.sum_is_neg,
                              countdown.tables.n_classes, V_PROD)
    log_W = reference_log_class_mass(np.log(np.asarray(p)), member)
    leaves = reference_log_leaves(log_W, countdown.edge_src, countdown.edge_dst,
                                  countdown.edge_class,
                                  countdown.n_states_bucket)

    ref = sequential_log_fold(leaves)
    tr = scans.up_sweep_log(jnp.asarray(leaves))
    got = (np.asarray(tr.root, np.float64)
           + float(np.asarray(tr.root_log_scale)))

    lost = int(((ref > DEAD) & (got <= DEAD)).sum())
    spurious = int(((ref <= DEAD) & (got > DEAD)).sum())
    both = (ref > DEAD) & (got > DEAD)
    err = float(np.abs(ref[both] - got[both]).max()) if both.any() else 0.0
    assert lost == 0, f"{lost} live root entries lost by the tree"
    assert spurious == 0, f"{spurious} root entries invented by the tree"
    assert err < 1e-6, f"root values disagree by {err:.3g} nats"


# ===========================================================================
# 5. Dynamic range — where the transition happens, as a function of what
# ===========================================================================

@pytest.mark.parametrize("L,S_bucket", [(8, 4), (256, 4)])
@pytest.mark.parametrize("eps", [1e-12, 1e-18])
def test_the_driver_is_confidence_in_the_complement_not_L(L, S_bucket, eps):
    """Characterises the transition rather than asserting one threshold.

    CLAUDE.md's standing warning is that toy shapes validate nothing —
    "float32 was fine at `L=4`, `|S|<=8` and drove spurious `Z == 0` on >50% of
    records at `L=256`". Here the dependence runs the other way: the behaviour
    at `L = 8` and `L = 256` is *identical*, because the quantity that is lost
    is a per-position class mass and nothing about it accumulates over `L`.
    The controlling variable is how much of `p` sits in the class's
    **complement**: above `eps ≈ 1e-16` the subtraction still has digits, below
    it has none.

    Both parametrisations assert the same theorem, so this is a
    characterisation *and* a test.
    """
    aut = minimal_automaton()
    p = minimal_p(eps, L=L)
    _t, ok_map = C.joint_map(p, aut, jnp.asarray(0), n_states=S_bucket,
                             n_classes=1)
    _t, ok_draw = C.joint_draw(p, aut, jnp.asarray(0), jax.random.key(0),
                               n_states=S_bucket, n_classes=1)
    assert bool(ok_map)
    assert bool(ok_draw), (
        f"L = {L}, eps = {eps:g}: infeasible. If this fails identically at "
        f"L = 8 and L = 256 the defect is per-position and `L` is not the "
        f"variable to sweep."
    )


#: The smallest class mass the released model can produce. The final logits are
#: capped at `30·tanh(x/30)` (`gemma/diffusion/_transformer.py:182`) and the
#: sampler's schedule bottoms out at `min_temperature = 0.4`
#: (`gemma/diffusion/_sampler.py:236`), so the span of `p` is at most
#: `2·30/0.4 = 150` nats and a one-token class cannot fall below ~`e^-150`.
PRODUCTION_MASS_FLOOR = float(np.exp(-150.0))


def test_class_mass_is_resolved_down_to_the_floor_the_model_can_produce():
    """The resolution requirement, stated as a number rather than a vibe — and
    the measurement that says which mechanism is in the way.

    Two things could destroy a class mass computed in linear space:

      * **underflow**, at ~`1e-308`, which needs ~709 nats of dynamic range;
      * **cancellation** in `total − partial`, at ~`1e-16`, which needs ~37.

    Gemma's softcap bounds `p`'s span at ~150 nats, so production can reach the
    second and not the first. The bisection below reports the boundary, and its
    *location* is what distinguishes the two — and therefore what distinguishes
    "widen the float" from "do not compute a small number as a difference of
    large ones".
    """
    lo, hi = 1e-300, 1.0         # lo = smaller eps, hi = known-resolved
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if not np.isfinite(mid) or mid <= 0 or mid >= hi or mid <= lo:
            break
        W = float(np.asarray(MG.class_weights(
            minimal_p(mid).T, jnp.asarray([0], jnp.int32),
            jnp.zeros(1, jnp.int32), jnp.asarray([True]), 1))[0, 0])
        if W > 0:
            hi = mid
        else:
            lo = mid
    diagnosis = ("cancellation in `total - partial`" if hi > 1e-30
                 else "float64 underflow")
    assert hi <= PRODUCTION_MASS_FLOOR, (
        f"`class_weights` stops resolving a class mass below eps ~ {hi:.3g}, "
        f"but the model can produce masses down to {PRODUCTION_MASS_FLOOR:.3g} "
        f"(softcap 30, T >= 0.4). Measured boundary says: {diagnosis}."
    )


def test_a_wide_negated_class_cannot_exercise_the_hazard():
    """Why the existing suite is green: its fixtures cannot reach the regime.

    `tests/test_audit_numerics.py::_cycle_automaton` — the only place a negated
    class meets `joint_draw` — stores `N_c` = the even tokens, so the class is
    the odd tokens: **half the vocabulary**, ~0.5 of the mass under any
    remotely spread `p`. `total − partial` at `|W| ≈ 0.5` is exact to 1e-16
    relative, so no `p` that fixture can produce puts the subtraction anywhere
    near its cancellation point.

    This test states that as a measurement, so "the suite has a hole here" is
    checkable rather than an opinion. It must PASS: the wide-class fixture is
    genuinely safe, and that is precisely the problem.
    """
    V = 1024
    evens = np.arange(0, V, 2, dtype=np.int32)
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((32, V)) * 8.0
    p = softmax64(logits)
    W = np.asarray(MG.class_weights(
        p.T, jnp.asarray(evens), jnp.zeros(evens.size, jnp.int32),
        jnp.asarray([True]), 1))[0]
    assert (W > 0).all(), "even a wide negated class is losing mass"
    assert W.min() > 1e-6, (
        f"the wide class's minimum mass is {W.min():.3g}; the hazard needs it "
        f"below ~1e-16, so this fixture cannot reach it no matter how sharp "
        f"`p` gets"
    )


# ===========================================================================
# 6. A second floor picked without reference to the production dynamic range
# SPEC §2.7 / §6.1 test 5: `constrained_map == argmax_x p*(x)`, "error
# 0.000e+00". Found while establishing that the MAP arm is a valid witness —
# its *feasibility* is, but its decode is not always the argmax.
# ===========================================================================

def test_map_is_the_argmax_even_when_the_whole_class_sits_below_the_floor():
    """`joint_map` evaluates `log p` as `log(max(p, map_log_floor(dtype)))`.

    SPEC §2.4's clamp rule is about `q`, which has **exact zeros** where the
    automaton forbids a token, so there the clamp only ever multiplies a zero
    and its value is arbitrary. `p` is a softmax and has no exact zeros; a floor
    on it flattens every token beneath it to one value, and `argmax` then
    resolves the tie by lowest token id (SPEC §2.7's stated convention). Where a
    whole class sits under the floor, MAP emits the *smallest admissible token
    id* rather than the most probable one — silently, and with the feasibility
    flag still True.

    Fixture: the class's best member beats its siblings by 1e4, so there is no
    ambiguity about what the argmax is, and `eps` is swept from clearly above a
    1e-30-style floor to clearly below it. The threshold itself is read from
    `map_log_floor`, so this states the contract ("MAP is the argmax for every
    `p` the dtype can represent") rather than a particular constant.
    """
    aut = minimal_automaton()
    winner = 5
    floor = C.map_log_floor(np.float64)
    for eps in [1e-20, 1e-40]:
        p = np.full((MIN_L, MIN_V), eps * 0.1 / (MIN_V - 2))
        p[:, 0] = 1.0 - eps
        p[:, winner] = eps * 0.9
        assert p.min() > floor, (
            f"fixture broken at eps = {eps:g}: it must stay above "
            f"`map_log_floor` = {floor:.3g}, or MAP is entitled to lose the "
            f"ordering and this asserts nothing"
        )
        tok, ok = C.joint_map(jnp.asarray(p, jnp.float64), aut, jnp.asarray(0),
                              n_states=MIN_S, n_classes=1)
        assert bool(ok)
        got = np.asarray(tok)
        assert (got == winner).all(), (
            f"eps = {eps:g} (class max p = {p[0, winner]:.3e}, siblings "
            f"{p[0, 1]:.3e}): MAP emitted {got[:4].tolist()} where every "
            f"position's argmax over the class is token {winner}, which beats "
            f"its siblings by 1e4. Under a floor above these values every "
            f"member scores the same and the tie goes to the lowest id."
        )


def test_the_map_floor_is_below_the_marginals_the_model_actually_produces(countdown):
    """The same defect, measured on the production grammar rather than posed.

    Counts `(class, position)` pairs whose **entire** class lies under the floor
    `joint_map` puts beneath `p`, with `p` Gemma-shaped at the schedule's final
    temperature. Every such pair is a position where MAP's token choice is the
    lowest admissible id rather than the argmax.

    **The floor is read from `constrained.map_log_floor`, never written as a
    literal here.** An earlier revision compared against `1e-30` spelled out in
    the test body, which called no implementation function at all — it asserted
    a property of its own fixture. Two things exposed that, and both were
    already in the run log rather than in anyone's reading of the code: it went
    red under *all* nine registry entries including mutants that cannot reach
    it, and it stayed red under the `class_weight_direct_complement` repair that
    turned all 32 other tests green. **A repair that fails to move a test is a
    statement about the test.** Bound to the implementation, it now reverts with
    it.
    """
    floor = C.map_log_floor(np.float64)
    lg = gemma_shaped_logits(L_PROD, V_PROD, seed=0)
    p = np.asarray(softmax64(lg))
    member = dense_membership(countdown.tables.sum_indices,
                              countdown.tables.sum_indptr,
                              countdown.tables.sum_is_neg,
                              countdown.tables.n_classes, V_PROD)
    clamped = 0
    total = 0
    for c in range(countdown.tables.n_classes):
        if not member[c].any():
            continue
        total += L_PROD
        clamped += int((p[:, member[c]].max(axis=1) < floor).sum())
    assert clamped == 0, (
        f"{clamped} of {total} (class, position) pairs have every member below "
        f"`map_log_floor` = {floor:.3g}, so MAP's `class_max` is the floor for "
        f"all of them and the emitted token is the lowest admissible id. The "
        f"smallest marginal in this canvas is {p.min():.3g}; the floor has to "
        f"sit under everything the softcap allows (~e^-150), not at an "
        f"unrelated round number."
    )


# ===========================================================================
# 7. Blast radius — `_matrices` is not only on the `joint_draw` path
# ===========================================================================

def test_the_masking_baseline_keeps_a_non_empty_per_position_support(countdown):
    """SPEC §2.8's baseline, and SPEC §7.2's baseline 2, share the leaf stage.

    The `mask` variant enforces the product of the coordinate *projections*
    `∏_i π_i(C)`. `π_i(C)` is non-empty at every position whenever `C` is
    non-empty, by definition of a projection, so an empty per-position support
    is not a possible result of that baseline; it is the leaf stage failing.

    It matters more here than on the emission path, because `sampler.py`'s
    `mask` branch is **deliberately not flagged** ("an empty per-position
    support is this baseline's *result*, not an error"). With every support
    empty, `jnp.where(r > 0, logits, MASK_SENTINEL)` is all-sentinel and
    `jax.random.categorical` is shift-invariant — so the baseline draws
    uniformly over the full 262k vocabulary and reports nothing. SPEC §2.8's
    `CS` column is the headline evidence for the whole method; it must not be
    measured through that.

    The `mar` confidence path (`constrained_entropy_streamed`) reads the same
    `u`, so its `Z_i` collapses at the same positions.
    """
    lg = gemma_shaped_logits(L_PROD, V_PROD, seed=0)
    lg[1, :] = 0.0
    lg[1, 107] = 60.0
    p = softmax64(lg)
    aut = traced(countdown)

    _pvl, _W, M = C._matrices(p, aut, countdown.n_states_bucket,  # noqa: SLF001
                              countdown.tables.n_classes)
    tr = scans.up_sweep(M)
    a_v, b_v, _sa, _sb = scans.prefix_suffix(
        tr, aut.active.astype(p.dtype),
        C.budget_terminal_factor(aut.d, jnp.asarray(R_BLOCK0), dtype=p.dtype))
    u = a_v[:-1][:, aut.edge_src] * b_v[1:][:, aut.edge_dst]
    r = np.asarray(MG.scatter_edge_mass_to_tokens(
        u, aut.edge_class, aut.csr_indices, aut.csr_indptr, aut.is_neg,
        countdown.tables.n_classes, V_PROD))

    empty = int(((r > 0).sum(axis=1) == 0).sum())
    assert empty == 0, (
        f"{empty} of {L_PROD} positions have an empty support projection, so "
        f"the `mask` baseline samples from an all-sentinel logit vector — i.e. "
        f"uniformly over the whole vocabulary — and, unlike the emission path, "
        f"nothing reports it."
    )


# ===========================================================================
# 8. The same subtraction, one function down: `scatter_edge_mass_to_tokens`
# SPEC §2.4's complement-aware `r_i(v)`.
# ===========================================================================
#
# `r_i(v) = Σ_{c ∈ Neg} U_i(c) + Σ_{c ∈ Pos, v ∈ S_c} U_i(c)
#                              − Σ_{c ∈ Neg, v ∈ N_c} U_i(c)`
#
# is the identical shape as the class-mass defect: a small answer obtained by
# subtracting one large number from another. `neg_total` sums **every** negated
# class's mass and the scatter then removes the ones the token is stored by, so
# a token whose true `r` comes from a *small* negated class, while some *other*
# negated class carries most of `neg_total`, is a difference of near-equal
# doubles.
#
# It is not on the emission path, which is why the Countdown arm did not die of
# it — but it feeds `q_i` (eq 6), `--confidence=mar`'s entropy, the `mask`
# baseline's support projection, and `tree.sample_tokens`' eq (8) edge
# multiplicity. A wrong `r` there is a silently wrong *distribution*, with no
# `Z == 0` to announce it.

def scatter_reference(U: np.ndarray, member: np.ndarray) -> np.ndarray:
    """`r[i, v] = Σ_c U[i, c] · 1[v ∈ member_c]`, as a sum of non-negatives.

    The definition of eq (6)'s inner sum, grouped by class instead of by edge.
    No complement algebra, no subtraction, so it keeps full *relative* accuracy
    at any magnitude — which is the whole question here.
    """
    return U @ member.astype(np.float64)


@pytest.mark.parametrize("ratio", [1e-8, 1e-12, 1e-16, 1e-20, 1e-30])
def test_scatter_keeps_the_mass_of_a_small_negated_class(ratio):
    """Two negated classes, one carrying `ratio` times the other's mass.

    Token `t` is stored by the **large** class only, so its true `r` is exactly
    the small class's `U` — while the implementation reaches it as
    `(U_big + U_small) − U_big`. Below ~1e-16 those two doubles are equal and
    the mass is gone.

    The existing coverage — `test_audit_numerics.py::
    test_scatter_is_complement_aware_against_a_dense_float64_reference` — uses
    an **absolute** 1e-9 bound on an `O(1)` fixture, which is exactly the hole
    already documented for `class_weights` in
    `test_a_wide_negated_class_cannot_exercise_the_hazard`: an absolute
    agreement at `|r| ≈ 1` says nothing about an `r` of 1e-20.
    """
    V, L = 64, 4
    t, other = 3, 17
    indices = np.array([t, other], dtype=np.int32)     # N_0 = {t}, N_1 = {other}
    indptr = np.array([0, 1, 2], dtype=np.int32)
    is_neg = np.array([True, True])
    class_id = np.array([0, 1], dtype=np.int32)        # one edge per class

    edge_mass = np.zeros((L, 2), dtype=np.float64)
    edge_mass[:, 0] = 1.0                              # U_big
    edge_mass[:, 1] = ratio                            # U_small

    got = np.asarray(MG.scatter_edge_mass_to_tokens(
        jnp.asarray(edge_mass), jnp.asarray(class_id), jnp.asarray(indices),
        jnp.asarray(indptr), jnp.asarray(is_neg), 2, V))

    member = dense_membership(indices, indptr, is_neg, 2, V)
    ref = scatter_reference(edge_mass, member)

    assert ref[0, t] == pytest.approx(ratio, rel=1e-12), "fixture broken"
    rel = np.abs(got - ref) / np.where(ref > 0, ref, 1.0)
    assert rel.max() < 1e-9, (
        f"ratio = {ratio:g}: r[i, {t}] should be the small class's mass "
        f"{ref[0, t]:.6e} and came back {got[0, t]:.6e} (relative error "
        f"{rel.max():.3g}). `neg_total` sums both negated classes and the "
        f"scatter subtracts the large one back off, so the answer is a "
        f"difference of near-equal doubles — the same shape as the class-mass "
        f"defect, one function down."
    )


def test_countdown_cannot_exercise_the_scatter_hazard(countdown):
    """Why the production grammar's green `mask` numbers are not evidence.

    Countdown has two negated classes, and the cancellation needs two of them
    live **at the same position** — with one, `neg_total` has a single term and
    the subtraction it used to feed was exact by construction. Measured here
    rather than assumed, because "Countdown never showed this" is otherwise easy
    to read as "Countdown is safe".

    Measured: class 1 (SPEC §3.6's channel-header name, `|N| = 7`) is live at
    **8** of 256 positions and class 23 (§3.5's unscored `ACC --Σ--> ACC` tail,
    `|N| = 0`) at **246** — and the two supports are **disjoint**, so the count
    of positions carrying both is 0. The header occupies the front of the canvas
    and the tail the back; they cannot overlap in this grammar.

    This test PASSES, and that is the finding: the safety is a positional
    accident of one grammar, not a property of the kernel. BFCL carries two
    negated classes with non-empty stored sets (7 and 1,585) whose supports are
    not segregated that way, so it has no such luck — which is why the
    adversarial coverage above is synthetic rather than pointed at Countdown.
    """
    lg = gemma_shaped_logits(L_PROD, V_PROD, seed=0)
    p = softmax64(lg)
    aut = traced(countdown)
    _pvl, _W, M = C._matrices(p, aut, countdown.n_states_bucket,  # noqa: SLF001
                              countdown.tables.n_classes)
    tr = scans.up_sweep(M)
    a_v, b_v, _sa, _sb = scans.prefix_suffix(
        tr, aut.active.astype(p.dtype),
        C.budget_terminal_factor(aut.d, jnp.asarray(R_BLOCK0), dtype=p.dtype))
    u = a_v[:-1][:, aut.edge_src] * b_v[1:][:, aut.edge_dst]         # [L, E]
    U = np.asarray(jax.ops.segment_sum(
        u.T, aut.edge_class, num_segments=countdown.tables.n_classes)).T  # [L, C]

    is_neg = np.asarray(countdown.tables.sum_is_neg)[
        :countdown.tables.n_classes]
    U_neg = U[:, is_neg]                                             # [L, n_neg]
    live_per_class = (U_neg > 0).sum(axis=0)
    both = int(((U_neg > 0).sum(axis=1) >= 2).sum())
    assert both == 0, (
        f"{both} of {L_PROD} positions carry two or more live negated classes "
        f"on Countdown (live counts per negated class: "
        f"{live_per_class.tolist()}), so this grammar CAN reach the scatter "
        f"cancellation regime and the 'positional accident' note above is "
        f"stale — the adversarial coverage must be re-pointed at the real "
        f"grammar rather than left synthetic."
    )


# ===========================================================================
# 9. Duplicate token ids across classes
# SPEC §4.4: classes are interned label sets and nothing forbids two of them
# from containing the same token.
# ===========================================================================

def shared_token_tables(V: int = MIN_V):
    """Two negated classes whose stored sets **overlap**: `N_0 = {0}`,
    `N_1 = {0, 7}`.

    Token 0 therefore occupies two CSR slots. Any complement evaluated over the
    CSR support has to count it **once**; counting it per-slot inflates class 0's
    mass by the whole of `p(0)`, which in this fixture is essentially all of the
    probability. Real grammars have this: Countdown has 38 tokens at
    multiplicity ≤ 4, the compiled BFCL grammars 1,028 at ≤ 6.
    """
    indices = np.array([0, 0, 7], dtype=np.int32)
    indptr = np.array([0, 1, 3], dtype=np.int32)
    seg = np.array([0, 1, 1], dtype=np.int32)
    is_neg = np.array([True, True])
    member = dense_membership(indices, indptr, is_neg, 2, V)
    return indices, indptr, seg, is_neg, member


@pytest.mark.parametrize("eps", [1e-1, 1e-8, 1e-20, 1e-40])
def test_class_mass_with_a_token_shared_by_two_classes(eps):
    """The duplicate-collapse property, at the magnitudes that matter.

    `test_class_mass_matches_a_log_space_reference` uses a single class holding
    a single token and **cannot see this failure at all**: with no duplicates
    there is nothing to collapse, so the representative rule is unexercised and
    `min`, `max` and `set` are indistinguishable. Removing the collapse entirely
    is the silent-misweight mode — a class mass too large by a term that is not
    in it — and it needs a token in two classes to appear.
    """
    indices, indptr, seg, is_neg, member = shared_token_tables()
    p = minimal_p(eps)
    ref_log = reference_log_class_mass(np.log(np.asarray(p)), member)   # [2, L]

    W = np.asarray(MG.class_weights(p.T, jnp.asarray(indices),
                                    jnp.asarray(seg), jnp.asarray(is_neg), 2))
    assert np.isfinite(ref_log).all(), "fixture broken: both classes non-empty"
    assert (W > 0).all(), (
        f"eps = {eps:g}: class masses are {np.exp(ref_log[:, 0])} but "
        f"`class_weights` returned {W[:, 0]}"
    )
    err = np.abs(np.log(W) - ref_log).max()
    assert err < 1e-4, (
        f"eps = {eps:g}: class mass off by {err:.3g} nats "
        f"({np.expm1(err) * 100:.3g}% relative) with token 0 stored by both "
        f"classes. Counting a duplicated token once per CSR slot adds p(0) — "
        f"here {float(np.asarray(p)[0, 0]):.3g} of the mass — to a class that "
        f"does not contain it."
    )
