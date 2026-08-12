"""Audit of `infer/scans.prefix_suffix` and the quantity production reads off it.

Scope: SPEC §2.4 (forward–backward, eq (5)/(6), "scaling is mandatory"), §2.6
(normalisation, and the log-space correction [V-P4]), §2.8 (why the `mask`
baseline exists at all). The code under test is `infer/scans.prefix_suffix`
**as consumed** — `u_i(e) = a_{i−1}(src e) · b_i(dst e)` — by
`model/sampler.py`'s `mask` and `--confidence=mar` closures.

Written from the specification. The implementation was read only to locate the
call sites (§"consumers" below); no assertion here is derived from it.

---

**The oracles, and why they are not the thing under test.**

Four vacuous tests have been found in this repository in a week, every one of
them a test that re-implemented its subject inside itself. So:

1. **`_bool_reach` is integer arithmetic.** Every quantity in §2.4 is a sum of
   products of **non-negative** numbers, so `a_i(s) > 0` is a *reachability*
   question, not a numerical one, and it is decidable exactly by boolean
   matrix–vector products over `M > 0`. Nothing in it can underflow, round, or
   cancel, and it shares no line with `scans.py`. It answers "which entries of
   `u` are truly non-zero", which is precisely what SPEC §2.4 eq (5) defines.
2. **`_log_forward_backward` is a numpy float64 log-space pass.** It is used
   only to *classify* a loss — it reports how many nats below `exp`'s floor the
   lost mass sits, which separates CLAUDE.md's `Z == 0` cause (c) (underflow,
   the scaling is missing) from causes (a)/(b) (a genuinely empty language,
   which must be raised, not patched).
3. **The structural support is fed through the production scatter.** The
   expected per-position token support is computed by handing
   `marginals.scatter_edge_mass_to_tokens` a 0/1 edge-liveness indicator — the
   same kernel, the same complement algebra, sums of ones, no underflow
   possible. A difference between that and the real `u` therefore isolates
   `prefix_suffix` and cannot be blamed on §4.4's class tables.

`p` is built the way the released model builds it — `30·tanh(z/30)` softcapped
logits over `T = 0.4` — and every regime is reported with its **entropy**, so
"production sharpness" is a measured claim (Phase 0 measured 1.2e-5 – 4.7e-4
nats on the real checkpoint) and not an adjective.

---

**Consumers of `u = a·b`, established by grep, not by assumption**
(`grep -rn 'prefix_suffix' --include=*.py`):

    model/sampler.py:390   --confidence=mar  -> constrained_entropy_streamed
    model/sampler.py:456   variant=mask      -> scatter_edge_mass_to_tokens

and, through `infer/marginals`, `constrained_marginals`,
`constrained_marginals_and_partition` and eq (8)'s token draw share the same
scatter. The J0/J1 emission does **not**: `constrained.joint_draw` /
`joint_map` go through `up_sweep_log` / `up_sweep_maxplus`, which SPEC §2.6
[V-P4] moved to log space for this exact failure mode. `mask` (SPEC §7.2
baseline 2, the headline comparison) and `mar` were left behind.

---

Every mutant used to validate this file is registered in `_MUTANTS` and
re-runnable in-process:

    DGFA_MUT=<name> JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
        pytest tests/test_audit_prefix_suffix.py -x -q

> **Retracted.** An earlier version of this file argued that no fix was
> possible inside `prefix_suffix`'s 4-tuple signature, because the true
> `u_i(e)` spans more than float64's range within one position. That is wrong,
> and the coder was right: `a[i]` and `b[i+1]` are each consumed at **exactly
> one** position, so a per-position *pair* of scales is available where a
> per-vector max is not — anchor the pair on `log Z` and `u` becomes the edge
> posterior, bounded by 1. `_forward_backward` below forming `u` itself is what
> made that visible; the conclusion drawn from it was not.

One of them, `per_position_log_u`, is a **repair** rather than a break: it
replaces the linear edge marginal with the log-space one SPEC §2.6 [V-P4]
prescribes. Its job is to prove the assertions below are *satisfiable* — a test
that cannot pass on any implementation is as useless as one that cannot fail.
Under it 22 of 24 go green; the two that do not are named and explained in the
results block below.

---

**Blast radius on the PRE-FIX kernel** (`git show HEAD:diffgemma_fa/infer/
scans.py`, spliced in and driven by the oracles below), at the entropy Phase 0
measured on the checkpoint (mean 3.0e-4 nats), `L = 256`, `remaining = 0`,
float64:

    grammar                       legal (position, token)   positions with an
                                  pairs lost                EMPTY mask support
    sudoku          (|S| =  32)         2,097,192                 16 / 256
    countdown       (|S| = 124)                 0                  0 / 256
    BFCL stock  0-0-0 (|S| =  58)               2                  0 / 256
    BFCL stock  1-1-0 (|S| =  61)               0                  0 / 256
    BFCL stock  2-2-0 (|S| =  80)       2,359,296                  8 / 256
    BFCL pretty 0-0-0 (|S| = 111)              10                  0 / 256
    BFCL pretty 1-1-0 (|S| = 114)               0                  0 / 256
    BFCL pretty 2-2-0 (|S| = 170)       2,359,335                  8 / 256

At a *flat* `p` (mean entropy 4.1 nats) every row is 0 and 0/256: the defect is
switched on by sharpness, which is what made it invisible to any smoke test run
at temperature 1.

> **⚠ An earlier revision of this table reported 706 / 742 pairs and 0 / 256
> dead positions for the two `2-2-0` grammars, and was wrong by three orders of
> magnitude in the direction that matters — it was read as evidence that BFCL
> was largely spared.** The cause was measuring BFCL at `remaining = 1000`
> while measuring Sudoku and Countdown at `remaining = 0`, and tabulating the
> two as one regime. `remaining = 1000` makes `b_L = 1[d(s) ≤ 1000]`, i.e.
> "every live state is terminal" — SPEC §3.1b's `b_l_unbounded` error — which
> keeps `b` large and hides the loss. Reproduced deliberately: at
> `remaining = 1000` the pre-fix kernel returns exactly 706 and 742 with no
> dead positions, and at `remaining = 0` the same code and the same `p` return
> 2,359,296 and 2,359,335 with 8 dead positions each. **`remaining` is part of
> the regime. State it beside every number.**

Countdown loses no tokens even pre-fix, but its `Z_i` — the per-position
partition function, which must be constant in `i` — spanned 480 nats, so `q_i`
and hence `--confidence=mar` were distorted there while the support survived.
"""

from __future__ import annotations

import dataclasses
import os
import re

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402
from gemma.diffusion import _sampler as DS  # noqa: E402

from diffgemma_fa.compile import pipeline  # noqa: E402
from diffgemma_fa.compile.automaton import INF_DISTANCE  # noqa: E402
from diffgemma_fa.compile.tasks import sudoku as SD  # noqa: E402
from diffgemma_fa.compile.tasks.grammars import (  # noqa: E402
    countdown_regex, sudoku_regex)
from diffgemma_fa.infer import marginals as MG  # noqa: E402
from diffgemma_fa.infer import scans  # noqa: E402
from diffgemma_fa.model import constrained as C  # noqa: E402
from diffgemma_fa.model import sampler as S  # noqa: E402
from diffgemma_fa.model.state import Automaton  # noqa: E402


#: The released checkpoint's own bound on how sharp a single position can be.
#: `logit = SOFTCAP·tanh(z/SOFTCAP)/T` lands in `[-75, +75]` at `T = 0.4`, so
#: the largest log-ratio between two tokens at one position is `2·75 = 150`
#: nats. Every synthetic regime below stays inside it, so "this cannot happen
#: in production" is not available as an answer. (`_MIN_TEMP = 1e-12`,
#: CLAUDE.md, makes far sharper reachable by configuration alone.)
SOFTCAP = 30.0
T_PROD = 0.4
MAX_NATS_PER_POSITION = 2.0 * SOFTCAP / T_PROD

L_PROD = 256
V_PROD = 262_144


# ===========================================================================
# Reproducible mutation registry.  `DGFA_MUT=<name> pytest tests/…`
# ===========================================================================

_MUTANTS: dict[str, str] = {
    # --- the repair: this must turn every red test in this file GREEN -----
    "per_position_log_u":
        "REPAIR — every test in this file must pass",
    # --- breaks. The first three are applied ON TOP of the repair, because
    #     a mutant cannot be shown to kill a test that is already red -----
    "logu_no_anchor":
        "test_edge_marginal_survives_wherever_the_automaton_does "
        "(reinstates the defect exactly; must reproduce the full red set)",
    "logu_suffix_order":
        "test_mask_support_is_exactly_the_structural_projection[countdown] "
        "(and every other countdown case, which is what proves the currently "
        "green arm is not vacuous)",
    "logu_b_off_by_one":
        "test_the_per_position_partition_is_position_invariant",
    "mute_representation_guard":
        "test_a_distorted_scale_is_reported_rather_than_silently_reweighted "
        "(an IMPLEMENTATION mutant: the guard keeps reporting health while the "
        "scale is genuinely distorted)",
    "keep_nonviable":
        "test_a_branch_that_cannot_finish_in_the_canvas_contributes_no_token "
        "(an IMPLEMENTATION mutant: it patches `constrained.budget_terminal_"
        "factor`, so it bites the production path and the repair alike)",
    "logu_a_start_all":
        "test_mask_support_is_exactly_the_structural_projection "
        "(the `extra` half — a support wider than the automaton allows)",
}

#: ⚠ **This registry validates THIS FILE, not the implementation.** Every
#: entry below either replaces `scans.prefix_suffix` wholesale (the repair and
#: the `logu_*` breaks perturb the repair's own internals) or patches a
#: neighbouring function. None of them perturbs the shipped kernel *in place*,
#: so a green registry says the harness is self-consistent — it does not say
#: the kernel is right. Two exceptions, added after review: `keep_nonviable`
#: patches `constrained.budget_terminal_factor` and `mute_representation_guard`
#: wraps the real `prefix_suffix`, and both therefore bite production.
#:
#: The authoritative implementation-mutation evidence is the reviewer's
#: in-process plugin, which perturbs `scans.prefix_suffix` itself. Its results
#: against this file, recorded here because a claim nobody can check is not
#: evidence:
#:
#:   mutant                  tests killed
#:   suffix_operand_order         18
#:   b_shift_one                  18
#:   half_to_zero                 18
#:   no_floor                      9
#:   unrestricted_anchor           2      <- the scale construction rests here
#:   anchor_off_by_one             2      <-   "
#:   keep_nonviable                0      <- now covered, see below
#:   forget_leaf_scales            0      <- NOT covered by this file
#:
#: Two of those zeros were the review's findings and one is now closed:
#: `test_a_branch_that_cannot_finish_in_the_canvas_contributes_no_token` kills
#: `keep_nonviable`, and `test_a_distorted_scale_is_reported_rather_than_
#: silently_reweighted` adds a third test over the anchor. `forget_leaf_scales`
#: remains uncovered here; `tests/test_jax_differential.py` catches it, while
#: `test_audit_numerics.py::test_linear_prefix_suffix_...` does not, because
#: that test's fixture pins `max M_i = 1.0` and so `leaf_scales == 0` — a
#: fixture that cannot exercise the thing its name claims.
#:
#: **Measured 2026-08-12** against the shipped kernel, CPU, float64:
#:
#:   DGFA_MUT                    killed  note
#:   <none>                           0   29 green — the defect is fixed
#:   per_position_log_u               0   28 green + 1 skip; the repair and the
#:                                        shipped kernel now agree
#:   logu_no_anchor                   8   the two-lane set + the driver
#:   logu_suffix_order               20   every real-grammar case
#:   logu_b_off_by_one                5   the Z_i invariance set
#:   logu_a_start_all                 3   the support test on both grammars
#:   keep_nonviable                   1   the non-viable-branch test
#:   mute_representation_guard        1   the distorted-scale test at T = 0.1
#:
#: `logu_b_off_by_one` does **not** kill the support test: shifting `b` by one
#: boundary leaves the union over edges almost unchanged on these grammars. The
#: support test is blind to an off-by-one and the `Z_i` test is not; that
#: division of labour is real and is why both exist.
#: Replaced by the `per_position_log_u` repair. `None` = run production's own
#: sequence.
_EDGE_MARGINAL_HOOK = None
#: Knobs the `logu_*` breaks turn, inside the repair.
_LOGU = {"anchor": True, "suffix_order": True, "b_shift": 0,
         "a_start_all": False}


def _apply_mutation(spec: str) -> None:
    """`DGFA_MUT=a` or `DGFA_MUT=a,b` — applied left to right."""
    global _EDGE_MARGINAL_HOOK
    for name in spec.split(","):
        if name not in _MUTANTS:
            raise SystemExit(
                f"unknown DGFA_MUT={name!r}; known: {sorted(_MUTANTS)}")

        if name == "per_position_log_u":
            # SPEC §2.6 [V-P4], applied one layer up from the tree: run the
            # forward-backward in log space and anchor the exponentiation on
            # the PER-POSITION max of `log a(src) + log b(dst)`. Relative to
            # that anchor the dominant edge of every position is exp(0) = 1 by
            # construction, so nothing that matters can underflow at any
            # dynamic range — the identical argument `log_matmul`'s docstring
            # makes for the pairwise max. `u` is only ever consumed per
            # position (`q = p·r/Z_i`), so a per-position scale is invisible
            # downstream.
            _EDGE_MARGINAL_HOOK = _log_space_edge_marginal

        elif name == "logu_no_anchor":
            # The repair with its one load-bearing line removed: exponentiate
            # `log u` against a global anchor instead of a per-position one.
            # This is the present defect, restated in the repair's own terms.
            _EDGE_MARGINAL_HOOK = _log_space_edge_marginal
            _LOGU["anchor"] = False

        elif name == "logu_suffix_order":
            # SPEC §2.6(c): `reverse=True` yields `f(f(z,y),x)` — the suffix
            # composed in the wrong operand order. No shape error, just wrong
            # probabilities.
            _EDGE_MARGINAL_HOOK = _log_space_edge_marginal
            _LOGU["suffix_order"] = False

        elif name == "logu_b_off_by_one":
            # SPEC §2.4: "`a_{i−1}` is the state *before* position `i`, `b_i`
            # the state *after*. No off-by-one."
            _EDGE_MARGINAL_HOOK = _log_space_edge_marginal
            _LOGU["b_shift"] = 1

        elif name == "mute_representation_guard":
            # SPEC §2.6 [V-P5]: a degenerate result must not be
            # indistinguishable from a good one. Report health unconditionally.
            _orig_ps = scans.prefix_suffix

            def muted(tree, a_start, b_final, *, feasible_out=False):
                out = _orig_ps(tree, a_start, b_final, feasible_out=feasible_out)
                if not feasible_out:
                    return out
                return out[:-1] + (jnp.ones_like(out[-1]),)
            scans.prefix_suffix = muted

        elif name == "keep_nonviable":
            # SPEC §3.1b: `b_L(s) = 1[d(s) ≤ R]`. Return a tiny positive
            # weight for non-viable states instead of an exact zero — the
            # "it is negligible anyway" reasoning, which is false because the
            # consumers read `> 0` as membership.
            orig_btf = C.budget_terminal_factor

            def leaky(d, remaining, dtype=jnp.float64):
                return jnp.where(d <= remaining,
                                 jnp.ones((), dtype), jnp.asarray(1e-100, dtype))
            C.budget_terminal_factor = leaky

        elif name == "logu_a_start_all":
            # SPEC §5.7: "Start vectors are vectors" — but they are not the
            # all-ones vector. Starting from every state makes the support
            # strictly WIDER than the automaton allows, which is the failure
            # direction the loss under audit never produces and which nothing
            # else here would catch.
            _EDGE_MARGINAL_HOOK = _log_space_edge_marginal
            _LOGU["a_start_all"] = True


# ===========================================================================
# Oracles — integer / log-space, sharing no line with `scans.py`
# ===========================================================================

def _bool_reach(support: np.ndarray, a0: np.ndarray, bL: np.ndarray):
    """Exact liveness of §2.4's `a_i` / `b_i`, in boolean arithmetic.

    Every factor in eq (3)/(4) is non-negative, so `a_i(s) > 0` iff some path
    of positive-weight transitions reaches `s` from the start set, and likewise
    backwards for `b`. That is a reachability question and this decides it
    exactly — no float, no threshold, no dependence on `scans.py`.

    Args:
      support: `[L, S] bool`, `M_i(s, s') > 0`.
      a0: `[S] bool`, the start set. bL: `[S] bool`, the terminal factor.

    Returns:
      `(live_a [L+1, S], live_b [L+1, S])`.
    """
    L, S = support.shape[0], support.shape[1]
    la = np.zeros((L + 1, S), bool)
    la[0] = a0
    for i in range(L):
        la[i + 1] = (la[i][:, None] & support[i]).any(axis=0)
    lb = np.zeros((L + 1, S), bool)
    lb[L] = bL
    for i in range(L - 1, -1, -1):
        lb[i] = (support[i] & lb[i + 1][None, :]).any(axis=1)
    return la, lb


def _log_forward_backward(logM: np.ndarray, log_a0: np.ndarray,
                          log_bL: np.ndarray):
    """§2.4's recursions in float64 log space. Diagnostic only.

    Used to *classify* a zero: if the true `log u_i(e)` is finite but below
    `log(tiny) = -745`, the mass was destroyed by the representation and the
    scaling is missing — CLAUDE.md's `Z == 0` cause (c). If it is `-inf`, the
    language really is empty there and raising is correct.
    """
    L, S = logM.shape[0], logM.shape[1]
    A = np.full((L + 1, S), -np.inf)
    A[0] = log_a0
    for i in range(L):
        t = A[i][:, None] + logM[i]
        m = t.max(axis=0)
        ok = np.isfinite(m)
        A[i + 1] = np.where(ok, m + np.log(np.exp(t - np.where(ok, m, 0.0)).sum(0)),
                            -np.inf)
    B = np.full((L + 1, S), -np.inf)
    B[L] = log_bL
    for i in range(L - 1, -1, -1):
        t = logM[i] + B[i + 1][None, :]
        m = t.max(axis=1)
        ok = np.isfinite(m)
        B[i] = np.where(ok,
                        m + np.log(np.exp(t - np.where(ok, m, 0.0)[:, None]).sum(1)),
                        -np.inf)
    return A, B


# ===========================================================================
# The seam: exactly what `model/sampler.py` computes, and nothing more
# ===========================================================================

@dataclasses.dataclass(frozen=True)
class _FB:
    u: jnp.ndarray            # [L, E]
    W_e: jnp.ndarray          # [E, L]
    M: jnp.ndarray            # [L, S, S]
    a: jnp.ndarray | None     # [L+1, S]  linear, max-normalised per vector
    b: jnp.ndarray | None
    log_a: jnp.ndarray | None   # [L+1]    accumulated log-scale of `a`
    log_b: jnp.ndarray | None
    #: `[L+1, S]` log-space vectors, present only on the repaired path.
    log_a_vec: jnp.ndarray | None = None
    log_b_vec: jnp.ndarray | None = None
    #: `[L]` log of the per-position scale `u` carries, to be added back before
    #: any cross-position comparison. Zero on the production path.
    log_u_scale: jnp.ndarray | None = None

    def log_partition_per_boundary(self) -> np.ndarray:
        """SPEC §2.4's `[D]`: `log(a_i·b_i) + logscale_a[i] + logscale_b[i]`.

        In log space the same quantity is `logsumexp_s (log a_i(s) + log b_i(s))`
        — the scales are already inside. Both forms are `prefix_suffix`'s own
        output judged against itself for `i`-invariance; neither re-derives `Z`.
        """
        if self.a is not None:
            dot = (np.asarray(self.a) * np.asarray(self.b)).sum(axis=1)
            with np.errstate(divide="ignore"):
                return (np.log(dot) + np.asarray(self.log_a)
                        + np.asarray(self.log_b))
        t = np.asarray(self.log_a_vec) + np.asarray(self.log_b_vec)
        t = np.where(t > scans.NEG_SENTINEL / 2, t, -np.inf)
        m = t.max(axis=1)
        ok = np.isfinite(m)
        return np.where(ok,
                        m + np.log(np.exp(t - np.where(ok, m, 0.0)[:, None]).sum(1)),
                        -np.inf)


def _forward_backward(p_lv, aut, remaining, n_states, n_classes) -> _FB:
    """`sampler.py`'s `mask` / `mar` closures, verbatim in sequence.

    `test_this_harness_is_still_what_sampler_py_runs` pins the coupling, so
    this cannot silently drift from production.
    """
    p_vl, W_e, M = C._matrices(p_lv, aut, n_states, n_classes)  # noqa: SLF001
    b_final = C.budget_terminal_factor(aut.d, remaining, dtype=p_lv.dtype)
    if _EDGE_MARGINAL_HOOK is not None:
        return _EDGE_MARGINAL_HOOK(p_lv, aut, M, W_e, b_final)
    tr = scans.up_sweep(M)
    a_v, b_v, la, lb = scans.prefix_suffix(
        tr, aut.active.astype(p_lv.dtype), b_final)
    # NOTE: the `edge_valid` mask is this harness's, not production's --
    # `sampler.py` forms `u = a·b` unmasked. It is inert on every compiled
    # automaton here (`edge_valid` is all-True; `_matrices` has already zeroed
    # invalid edges in `W_e`, so `M` and hence `a`/`b` never see them), but it
    # is a divergence and it is unpinned, so it is written down rather than
    # left for the next reader to find.
    u = jnp.where(aut.edge_valid[None, :],
                  a_v[:-1][:, aut.edge_src] * b_v[1:][:, aut.edge_dst], 0.0)
    return _FB(u=u, W_e=W_e, M=M, a=a_v, b=b_v, log_a=la, log_b=lb,
               log_u_scale=jnp.zeros((M.shape[0],)))


def _log_space_edge_marginal(p_lv, aut, M, W_e, b_final) -> _FB:
    """The `per_position_log_u` repair (SPEC §2.6 [V-P4]) — see `_MUTANTS`."""
    neg = jnp.asarray(scans.NEG_SENTINEL, dtype=p_lv.dtype)
    tiny = jnp.finfo(p_lv.dtype).tiny
    logM = jnp.where(M > 0, jnp.log(jnp.maximum(M, tiny)), neg)
    la0 = (jnp.zeros_like(b_final) if _LOGU["a_start_all"]
           else jnp.where(aut.active, jnp.zeros((), p_lv.dtype), neg))
    lbL = jnp.where(b_final > 0, jnp.zeros((), p_lv.dtype), neg)
    L, S = logM.shape[0], logM.shape[1]
    A = [la0]
    for i in range(L):
        t = A[-1][:, None] + logM[i]
        m = jnp.max(t, axis=0)
        A.append(jnp.where(m > scans.NEG_SENTINEL / 2,
                           m + jnp.log(jnp.sum(jnp.exp(t - m[None, :]), axis=0)),
                           neg))
    B = [lbL]
    for i in range(L - 1, -1, -1):
        t = (logM[i] + B[-1][None, :] if _LOGU["suffix_order"]
             else logM[i].T + B[-1][None, :])
        m = jnp.max(t, axis=1)
        B.append(jnp.where(m > scans.NEG_SENTINEL / 2,
                           m + jnp.log(jnp.sum(jnp.exp(t - m[:, None]), axis=1)),
                           neg))
    A = jnp.stack(A)
    B = jnp.stack(B[::-1])
    if _LOGU["b_shift"]:
        B = jnp.roll(B, _LOGU["b_shift"], axis=0)
    log_u = A[:-1][:, aut.edge_src] + B[1:][:, aut.edge_dst]
    log_u = jnp.where(aut.edge_valid[None, :], log_u, neg)
    anchor = (jnp.max(log_u, axis=1, keepdims=True) if _LOGU["anchor"]
              else jnp.max(log_u))
    shifted = log_u - jnp.maximum(anchor, scans.NEG_SENTINEL / 2)
    if _LOGU["anchor"]:
        # The anchor bounds the largest edge at every position, but not the
        # spread *within* a position, which on a real grammar is thousands of
        # nats. Two of `u`'s consumers ask a **support** question — `r > 0` in
        # the §2.8 mask, and `q_i(v) > 0` — so the last representable decade
        # carries a bit that matters even where the value does not. Flooring
        # the exponent preserves that bit and perturbs nothing above `e^-700`,
        # which is 300 orders below anything `q_i` can express anyway.
        shifted = jnp.maximum(shifted, -700.0)
    u = jnp.where(log_u > scans.NEG_SENTINEL / 2, jnp.exp(shifted), 0.0)
    scale = jnp.where(jnp.squeeze(anchor) > scans.NEG_SENTINEL / 2,
                      jnp.broadcast_to(jnp.squeeze(anchor), (L,)),
                      jnp.zeros((L,)))
    return _FB(u=u, W_e=W_e, M=M, a=None, b=None, log_a=None, log_b=None,
               log_a_vec=A, log_b_vec=B, log_u_scale=scale)


def _structural(fb: _FB, aut, b_final) -> dict:
    """Everything the oracles say, from `M` and the production scatter."""
    Mn = np.asarray(fb.M)
    es, ed = np.asarray(aut.edge_src), np.asarray(aut.edge_dst)
    ev = np.asarray(aut.edge_valid)
    la, lb = _bool_reach(Mn > 0, np.asarray(aut.active).astype(bool),
                         np.asarray(b_final) > 0)
    edge_live = ev[None, :] & (np.asarray(fb.W_e).T > 0) & la[:-1][:, es] \
        & lb[1:][:, ed]
    return dict(live_a=la, live_b=lb, edge_live=edge_live, Mn=Mn, es=es, ed=ed)


def _support(edge_mass, aut, n_classes, vocab) -> np.ndarray:
    """`r_i(v) > 0` through the production scatter. `[L, V] bool`."""
    r = MG.scatter_edge_mass_to_tokens(
        jnp.asarray(edge_mass), aut.edge_class, aut.csr_indices,
        aut.csr_indptr, aut.is_neg, n_classes, vocab)
    return np.asarray(r) > 0


# ===========================================================================
# `p` at the model's own sharpness
# ===========================================================================

def _softmax64(lg):
    lg = np.asarray(lg, np.float64)
    m = lg.max(axis=-1, keepdims=True)
    e = np.exp(lg - m)
    return e / e.sum(axis=-1, keepdims=True)


def build_p(L, V, sigma, targets=None, seed=0, temperature=T_PROD):
    """The released model's own shape: `softmax(SOFTCAP·tanh(z/SOFTCAP)/T)`."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((L, V)) * sigma
    if targets is not None:
        z[np.arange(L), np.asarray(targets)] = 1e9
    return _softmax64(SOFTCAP * np.tanh(z / SOFTCAP) / temperature)


def entropy_of(p):
    return -(p * np.log(np.maximum(p, 1e-300))).sum(-1)


#: Phase 0 measured 1.2e-5 – 4.7e-4 nats on the real checkpoint's final steps.
PHASE0_ENTROPY_RANGE = (1.2e-5, 4.7e-4)


# ===========================================================================
# Synthetic automata — two lanes of unequal cost, which is what a compiled
# grammar with SPEC §3.5's unscored `ACC --Σ--> ACC` tail looks like
# ===========================================================================

def two_lane(L: int, nats: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`[L, 4, 4]` `M`, plus `a_start` and `b_final`.

    State 0 branches into a **cheap** lane (1) and an **expensive** lane (2),
    both of which can reach the accepting state 3. Every expensive step costs
    `nats` more than the cheap one, so the within-vector dynamic range of `a_i`
    grows by `nats` per position going forward and that of `b_i` by `nats` per
    position going backward — the structure SPEC §2.6 [V-P4] describes, where
    the unscored tail pins the max at 1.0 while a genuine grammar path decays.

    Both halves matter: the expensive lane's `a` entry sits `nats·i` below the
    max and its `b` entry `nats·(L−i)` below, so their **product** is
    `nats·L` below at every position while each factor is only ever half that
    far down. That is the shape that no per-vector normalisation can rescue.

    `nats` is asserted against `MAX_NATS_PER_POSITION` by the callers so the
    regime is always one the released model can actually produce.
    """
    S = 4
    M = np.zeros((L, S, S))
    w = np.exp(-nats)
    for i in range(L):
        M[i, 0, 1] = 1.0
        M[i, 0, 2] = w
        M[i, 1, 1] = 1.0
        M[i, 2, 2] = w
        M[i, 1, 3] = 1.0
        M[i, 2, 3] = w
        M[i, 3, 3] = 1.0          # unscored post-stop tail, SPEC §3.5
    a0 = np.zeros(S)
    a0[0] = 1.0
    bL = np.zeros(S)
    bL[3] = 1.0
    return M, a0, bL


#: The same two lanes as an actual `Automaton`, so every test below goes
#: through `_forward_backward` — the seam production uses — rather than poking
#: `scans.prefix_suffix` directly. That is what lets the `per_position_log_u`
#: repair be evaluated against these tests at all.
_TWO_LANE_EDGES = ((0, 1, 0), (0, 2, 1), (1, 1, 0), (2, 2, 1),
                   (1, 3, 0), (2, 3, 1), (3, 3, 2))
#: class 0 = the cheap token, class 1 = the expensive one, class 2 = `Σ`.
_TWO_LANE_MEMBERS = ((0,), (1,), (0, 1))


def _two_lane_automaton() -> Automaton:
    indices, indptr = [], [0]
    for c in _TWO_LANE_MEMBERS:
        indices.extend(sorted(c))
        indptr.append(len(indices))
    d = np.full(4, INF_DISTANCE, np.int32)
    for s, dist in _distance_to_final(
            4, [(e[0], e[1]) for e in _TWO_LANE_EDGES], {3}).items():
        d[s] = dist
    return Automaton(
        edge_src=jnp.asarray([e[0] for e in _TWO_LANE_EDGES], jnp.int32),
        edge_dst=jnp.asarray([e[1] for e in _TWO_LANE_EDGES], jnp.int32),
        edge_class=jnp.asarray([e[2] for e in _TWO_LANE_EDGES], jnp.int32),
        edge_valid=jnp.ones(len(_TWO_LANE_EDGES), bool),
        csr_indices=jnp.asarray(indices, jnp.int32),
        csr_indptr=jnp.asarray(indptr, jnp.int32),
        is_neg=jnp.zeros(len(_TWO_LANE_MEMBERS), bool),
        d=jnp.asarray(d),
        is_final=jnp.asarray([False, False, False, True]),
        active=jnp.asarray([True, False, False, False]))


def _run_two_lane(L, nats):
    """Drive the two-lane grammar at `nats` per position through the seam.

    `p` is a two-token softmax with logits `±nats/2`, i.e. exactly the class
    masses `two_lane` writes by hand, and inside the softcap window whenever
    `nats < MAX_NATS_PER_POSITION`.
    """
    aut = _two_lane_automaton()
    lg = np.tile(np.array([nats / 2.0, -nats / 2.0]), (L, 1))
    p = _softmax64(lg)
    fb = _forward_backward(jnp.asarray(p), aut, 0, 4, len(_TWO_LANE_MEMBERS))
    b_final = C.budget_terminal_factor(aut.d, 0, dtype=jnp.float64)
    st = _structural(fb, aut, b_final)
    u = np.asarray(fb.u)
    lost = st["edge_live"] & (u == 0)
    out = dict(fb=fb, st=st, u=u, edge_live=st["edge_live"], lost=lost,
               aut=aut, p=p)
    if fb.a is not None:
        a, b = np.asarray(fb.a), np.asarray(fb.b)
        out["a"], out["b"] = a, b
        out["log_a"], out["log_b"] = np.asarray(fb.log_a), np.asarray(fb.log_b)
        out["product_only"] = (lost & (a[:-1][:, st["es"]] > 0)
                               & (b[1:][:, st["ed"]] > 0))
    return out


# ===========================================================================
# 1. The invariant SPEC §2.4 actually states
# ===========================================================================

@pytest.mark.parametrize("L,nats", [(8, 120.0), (16, 85.0), (32, 40.0),
                                    (64, 20.0), (128, 10.0), (256, 5.0)])
def test_edge_marginal_survives_wherever_the_automaton_does(L, nats):
    """SPEC §2.4 eq (5): `u_i(e) = a_{i−1}(src e) · b_i(dst e)`.

    `u_i(e)` is exactly zero **iff** no accepted path uses edge `e` at position
    `i` — every factor in (3)/(4) is non-negative, so a zero can only come from
    the automaton, never from cancellation. SPEC's remedy for the magnitude is
    §2.4's "**scaling is mandatory**", and §2.6's normalisation rule exists so
    that the quantities production reads stay representable.

    Normalising `a_i` and `b_i` to max 1 *separately* bounds neither the
    within-vector dynamic range nor the product, so this asserts the property
    SPEC asks for rather than the one the code happens to provide: every
    structurally live edge must come back non-zero.

    `nats` is the per-position log-cost gap between the two lanes and is kept
    below the released model's own softcap ceiling of `2·30/0.4 = 150` nats, so
    no regime here is out of reach of the checkpoint.
    """
    assert nats < MAX_NATS_PER_POSITION, "regime is not reachable in production"
    r = _run_two_lane(L, nats)
    n_lost, n_live = int(r["lost"].sum()), int(r["edge_live"].sum())
    detail = ""
    if "a" in r and n_lost:
        detail = (f" smallest positive a = {r['a'][r['a'] > 0].min():.3e}, "
                  f"smallest positive b = {r['b'][r['b'] > 0].min():.3e}")
    assert n_lost == 0, (
        f"L={L}, {nats} nats/position: {n_lost} of {n_live} structurally live "
        f"(position, edge) pairs came back u == 0. Positions affected: "
        f"{np.nonzero(r['lost'].any(axis=1))[0][:8].tolist()}.{detail}"
    )


def test_both_factors_are_representable_yet_their_product_is_zero():
    """The precise defect, isolated: `a[i](s) > 0`, `b[i+1](s') > 0`, `a·b == 0`.

    This is the case no per-vector normalisation can reach, because each vector
    is *individually* well scaled — max exactly 1.0 — and the loss happens only
    when the two are multiplied. SPEC §2.4 makes `u` the object of eq (5) and
    (6), so it is `u`, not `a` and `b` separately, that the scaling rule has to
    keep representable.

    `L = 16` at 85 nats/position: both factors sit around `1e-260`, the product
    at `1e-520`.
    """
    r = _run_two_lane(16, 85.0)
    if "product_only" not in r:
        pytest.skip("no separate linear a/b to multiply — the interface that "
                    "made this possible is gone, which is the fix")
    n = int(r["product_only"].sum())
    assert n == 0, (
        f"{n} live edges lost with BOTH factors non-zero — the product "
        f"underflowed on its own. Example magnitudes: "
        f"a={np.asarray(r['a'])[:-1][:, r['st']['es']][r['product_only']].max():.3e}, "
        f"b={np.asarray(r['b'])[1:][:, r['st']['ed']][r['product_only']].max():.3e}"
    )


def test_the_aggregate_dot_survives_while_live_edges_die():
    """What the existing suite checks, and why it is not enough.

    `tests/test_audit_numerics.py::test_linear_prefix_suffix_recovers_the_log_
    space_partition_at_L256` asserts `(a·b).sum(axis=1) > 0` at every boundary
    and that the resulting `log Z` matches the log-space tree. Both hold here.
    But `Σ_s a_i(s)·b_i(s)` is dominated by its **largest** term — SPEC §3.5's
    unscored `ACC --Σ--> ACC` tail contributes exactly 1.0 — while eq (5)/(6)
    consume the individual products, which are hundreds of nats smaller.

    So the aggregate is a scalar that survives by construction and the vector
    of things production actually reads does not. This test asserts the
    implication the existing one is silently assumed to carry.
    """
    r = _run_two_lane(256, 5.0)
    logZ = r["fb"].log_partition_per_boundary()
    assert np.all(np.isfinite(logZ)), "precondition: the aggregate must survive"
    assert float(logZ.max() - logZ.min()) < 1e-6, "precondition: log Z is flat"
    n_lost = int(r["lost"].sum())
    assert n_lost == 0, (
        f"every boundary passes the aggregate `a_i·b_i > 0` check and log Z is "
        f"`i`-invariant, yet {n_lost} of {int(r['edge_live'].sum())} live edge "
        f"marginals are exactly zero. The aggregate check cannot see this."
    )


def test_the_driver_is_dynamic_range_not_canvas_length():
    """Characterisation: which axis to hold fixed when validating a fix.

    CLAUDE.md: "a numerical claim validated only on toy shapes is not
    validated". The dual error is to assume the axis is `L`. It is not: the
    quantity that matters is the accumulated **within-vector** log-range,
    `nats × distance from the boundary where the vector is seeded`. So the
    defect fires at `L = 8` and is absent at `L = 256` if the range is small,
    and the minimal reproducer is small in every dimension.

    The negative half is a statement about this implementation, not about the
    specification; it is here so that a fix cannot be declared validated on a
    regime that never exercised the failure.
    """
    mild = _run_two_lane(256, 1.0)         # 256 nats end to end
    assert int(mild["lost"].sum()) == 0, (
        "the 1-nat regime lost edges too — the reproducer below no longer "
        "isolates dynamic range from L"
    )
    # `L = 4` is *not* reachable: it would need 745/4 = 187 nats/position and
    # the softcap allows 150. So 8 is the floor at `T = 0.4`, and only because
    # `_MIN_TEMP = 1e-12` would take it lower.
    unreachable = _run_two_lane(4, MAX_NATS_PER_POSITION - 1.0)
    assert int(unreachable["lost"].sum()) == 0, (
        "L = 4 now fires; the softcap ceiling in MAX_NATS_PER_POSITION is wrong"
    )
    tiny = _run_two_lane(8, 120.0)         # 960 nats end to end, L = 8
    assert int(tiny["lost"].sum()) == 0, (
        f"L = 8, |S| = 4, 120 nats/position (the model's softcap allows 150): "
        f"{int(tiny['lost'].sum())} live edge marginals lost. `L` is not the "
        f"axis; accumulated log-range is."
    )


# ===========================================================================
# 2. Real compiled grammars, at the sharpness Phase 0 measured
# ===========================================================================

def _sudoku_automaton():
    rec = list(SD.generate(1, seed=0))[0]
    return pipeline.compile_regex(sudoku_regex(rec.puzzle, row_separator="\n"),
                                  name="sudoku", header_tokens=8).automaton


def _countdown_automaton():
    return pipeline.compile_regex(countdown_regex(max_steps=4, max_value=999),
                                  name="countdown", header_tokens=8).automaton


def _bfcl_automaton():
    """The first single-function `BFCL_v4_live_simple` schema, stock rendering.

    Compiled through `eval/run.py`'s own call site and `ALLOW` tuple. It is
    here for one reason: **Sudoku's `Z_i` does not move under temperature**
    (flat to 1.7e-12 even at `T = 0.1`), so it cannot pin the scale
    construction, whereas this grammar's spans 1.63 nats there.
    """
    from diffgemma_fa.compile import bfcl_data, schema as _schema
    from diffgemma_fa.eval.run import ALLOW
    rec = next(r for r in bfcl_data.iter_split("BFCL_v4_live_simple.json")
               if len(r.functions) == 1)
    sp = rec.functions[0]["parameters"]
    gate = _schema.synthesize_instance(_schema.normalize_bfcl_schema(sp))
    return pipeline.compile_json_schema(
        sp, name=rec.functions[0].get("name", ""), from_bfcl=True, allow=ALLOW,
        allow_wildcard=True, whitespace_pattern=None, fence=False,
        verify_renderings=gate, verify_strict=False,
        nonempty_required_strings=True).automaton


_GRAMMARS = {"sudoku": _sudoku_automaton, "countdown": _countdown_automaton,
             "bfcl": _bfcl_automaton}
_CACHE: dict = {}


def _grammar(name):
    if name not in _CACHE:
        _CACHE[name] = _GRAMMARS[name]()
    return _CACHE[name]


def _traced(a) -> Automaton:
    """The production conversion (`eval/run.to_traced`), un-batched."""
    return Automaton(
        edge_src=jnp.asarray(a.edge_src), edge_dst=jnp.asarray(a.edge_dst),
        edge_class=jnp.asarray(a.edge_class),
        edge_valid=jnp.ones(a.n_edges, bool),
        csr_indices=jnp.asarray(a.tables.sum_indices),
        csr_indptr=jnp.asarray(a.tables.sum_indptr),
        is_neg=jnp.asarray(a.tables.sum_is_neg),
        d=jnp.asarray(a.d), is_final=jnp.asarray(a.is_final),
        active=jnp.asarray(a.start_vector))


@dataclasses.dataclass(frozen=True)
class _Case:
    tag: str
    aut: Automaton
    n_states: int
    n_classes: int
    fb: _FB
    st: dict
    p_lv: jnp.ndarray
    H: np.ndarray
    remaining: int


def _production_case(name, *, sigma=8.0, confident=True, seed=0,
                     remaining=0, temperature=T_PROD) -> _Case:
    """A compiled grammar against a `p` at the checkpoint's own sharpness.

    `remaining = 0` is the production value for a single-block decode
    (`ConstrainedSamplingState.terminal_budget` subtracts the canvas), so
    `b_L = 1[d(s) ≤ 0] = 1[s ∈ F]` — SPEC §3.1b's tightest terminal factor.
    """
    a = _grammar(name)
    aut = _traced(a)
    ns, nc = a.n_states_bucket, a.tables.n_classes
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, V_PROD, L_PROD) if confident else None
    p = build_p(L_PROD, V_PROD, sigma, targets=targets, seed=seed,
                temperature=temperature)
    p_lv = jnp.asarray(p)
    fb = _forward_backward(p_lv, aut, remaining, ns, nc)
    b_final = C.budget_terminal_factor(aut.d, remaining, dtype=p_lv.dtype)
    return _Case(tag=f"{name}/sigma={sigma}/T={temperature}/confident={confident}",
                 aut=aut,
                 n_states=ns, n_classes=nc, fb=fb,
                 st=_structural(fb, aut, b_final), p_lv=p_lv,
                 H=entropy_of(p), remaining=remaining)


@pytest.fixture(scope="module")
def sudoku_prod():
    return _production_case("sudoku")


@pytest.fixture(scope="module")
def countdown_prod():
    return _production_case("countdown")


@pytest.fixture(scope="module")
def sudoku_sharp_prod():
    """Sudoku at `T = 0.2` — half the production temperature, still reachable.

    **Why a second temperature exists at all.** Every other assertion in this
    file is either support-based (`r > 0`) or scale-free, so the entire
    *scale* construction — whatever anchoring a fix chooses so that `u` is
    representable — rests on the `Z_i` invariance test alone. `T = 0.4` does
    not stress it: an anchoring bug that is off by a bounded per-position
    constant leaves `Z_i` visibly flat there.

    Halving the temperature doubles every log-ratio and puts the anchor under
    real load. It is not the production default and is **deliberately excluded
    from the Phase 0 entropy guard**; it is reachable by configuration alone,
    because `_MIN_TEMP = 1e-12` (CLAUDE.md, SPEC §3.9) admits temperatures far
    below this and `--temp greedy` is a shipped flag.
    """
    return _production_case("sudoku", temperature=0.2)


def test_the_regime_is_the_one_phase_0_measured(sudoku_prod):
    """Guard on every claim below: `p` must be production-sharp, not extreme.

    Without this the real-grammar results are unfalsifiable — anyone can make a
    forward-backward underflow with an absurd `p`. Phase 0 measured final-step
    entropies of 1.2e-5 to 4.7e-4 nats on the released checkpoint
    (`docs/PHASE0_FINDINGS.md`, quoted in `marginals.class_weights`), and the
    regime used below must land in that window.
    """
    lo, hi = PHASE0_ENTROPY_RANGE
    H = sudoku_prod.H
    assert lo <= float(H.mean()) <= hi, (
        f"mean entropy {H.mean():.3e} nats is outside Phase 0's measured "
        f"[{lo:.1e}, {hi:.1e}] — the regime is not production sharpness"
    )


@pytest.mark.parametrize("grammar", ["sudoku", "countdown"])
def test_mask_support_is_exactly_the_structural_projection(grammar, request):
    """SPEC §2.8 / §7.2 baseline 2. The `mask` variant's whole definition.

    The baseline masks to the per-position support projection `π_i(C)` and
    samples each position independently. SPEC's argument against it is that
    `∏_i π_i(C) ⊋ C` — it is *too permissive*. A support that is a strict
    **subset** of `π_i(C)` is a different baseline, and its `CS` column stops
    being evidence for anything.

    The expected support is computed by handing the same production scatter a
    0/1 edge-liveness indicator: identical kernel, identical complement
    algebra, sums of ones, nothing to underflow. So a difference isolates
    `prefix_suffix`.
    """
    case = request.getfixturevalue(f"{grammar}_prod")
    V = int(case.p_lv.shape[-1])
    got = _support(case.fb.u, case.aut, case.n_classes, V)
    want = _support(case.st["edge_live"].astype(np.float64), case.aut,
                    case.n_classes, V)
    missing = want & ~got
    extra = got & ~want
    assert not extra.any(), (
        f"{case.tag}: the mask support contains {int(extra.sum())} tokens the "
        f"automaton forbids")
    assert not missing.any(), (
        f"{case.tag}: mean entropy {case.H.mean():.2e} nats — the mask support "
        f"is missing {int(missing.sum())} of {int(want.sum())} legal "
        f"(position, token) pairs, at {int(missing.any(1).sum())} of "
        f"{missing.shape[0]} positions. `mask` is not sampling from "
        f"`π_i(C)`.")


@pytest.mark.parametrize("grammar", ["sudoku", "countdown"])
def test_no_live_position_loses_its_entire_support(grammar, request):
    """The severe form of the same loss, and SPEC §3.1's hole reopened.

    When `r_i(v) == 0` for every `v`, `sampler.py` computes
    `jnp.where(r > 0, logits, MASK_SENTINEL)` — every logit becomes the finite
    sentinel `-1e30`. `jax.random.categorical` on an all-equal row in float32
    swamps the Gumbel noise, so the position emits token id 0 deterministically
    and the canvas carries a token no grammar path allows.

    Empty support at a position is a legitimate *result* only when the language
    really has none there; here the oracle says otherwise, so it is CLAUDE.md's
    `Z == 0` cause (c) — "fp32 underflow on an unnormalized path — expected,
    and SPEC §6.3 deliberately tests for it", one representation up.
    """
    case = request.getfixturevalue(f"{grammar}_prod")
    V = int(case.p_lv.shape[-1])
    got = _support(case.fb.u, case.aut, case.n_classes, V)
    want = _support(case.st["edge_live"].astype(np.float64), case.aut,
                    case.n_classes, V)
    dead = np.nonzero(want.any(1) & ~got.any(1))[0]
    detail = ""
    if dead.size:
        logM = np.where(case.st["Mn"] > 0,
                        np.log(np.maximum(case.st["Mn"], 1e-320)), -np.inf)
        b_final = C.budget_terminal_factor(case.aut.d, case.remaining,
                                           dtype=case.p_lv.dtype)
        A, B = _log_forward_backward(
            logM, np.where(np.asarray(case.aut.active), 0.0, -np.inf),
            np.where(np.asarray(b_final) > 0, 0.0, -np.inf))
        i = int(dead[0])
        v = (A[i][case.st["es"]] + B[i + 1][case.st["ed"]])[case.st["edge_live"][i]]
        detail = (f" At position {i} the largest true log u is {v.max():.0f} "
                  f"nats — finite, so the language is NOT empty; it is "
                  f"{-745 - v.max():.0f} nats below float64's exp floor.")
    assert dead.size == 0, (
        f"{case.tag}: {dead.size} of {want.shape[0]} positions came back with "
        f"an EMPTY mask support although the automaton admits tokens there: "
        f"{dead[:14].tolist()}.{detail}")


@pytest.mark.parametrize("temperature", [0.4, 0.2, 0.1])
def test_a_distorted_scale_is_reported_rather_than_silently_reweighted(temperature):
    """The scale construction, and the guard that is supposed to police it.

    Every other assertion in this file is support-based or scale-free, so
    **whatever anchoring a fix chooses is pinned by the `Z_i` invariance test
    alone**, and `T = 0.4` does not stress it. Halving and quartering the
    temperature does: measured on the shipped kernel, `log Z_i` is flat to
    9e-13 at `T = 0.2` and spans **1.63 nats on BFCL `0-0-0` and 599 nats on
    `2-2-0` at `T = 0.1`**, while `H(q_i)` stays inside `[0, log V]` the whole
    way — so no other detector in this file or the repo fires on it.

    `_MIN_TEMP = 1e-12` (CLAUDE.md, SPEC §3.9) makes these reachable by
    configuration alone, so "production runs at 0.4" is not a defence.

    The assertion is the **implication**, for the reason SPEC §2.6 [V-P5]
    gives: a degenerate result must not be indistinguishable from a good one.
    Either the partition function is position-invariant, or the kernel says so.
    It passes when the scale holds and it passes when the damage is reported;
    it fails only in the state that costs a published number.
    """
    import inspect
    if "feasible_out" not in inspect.signature(scans.prefix_suffix).parameters:
        pytest.skip("kernel has no representation guard to pin")
    a = _grammar("bfcl")
    aut = _traced(a)
    ns, nc = a.n_states_bucket, a.tables.n_classes
    rng = np.random.default_rng(0)
    p = build_p(L_PROD, V_PROD, 8.0, targets=rng.integers(0, V_PROD, L_PROD),
                seed=0, temperature=temperature)
    p_lv = jnp.asarray(p)
    p_vl, W_e, M = C._matrices(p_lv, aut, ns, nc)  # noqa: SLF001
    b_final = C.budget_terminal_factor(aut.d, 0, dtype=p_lv.dtype)
    out = scans.prefix_suffix(scans.up_sweep(M), aut.active.astype(p_lv.dtype),
                              b_final, feasible_out=True)
    a_v, b_v, rep = out[0], out[1], np.asarray(out[-1])
    u = a_v[:-1][:, aut.edge_src] * b_v[1:][:, aut.edge_dst]
    r = MG.scatter_edge_mass_to_tokens(u, aut.edge_class, aut.csr_indices,
                                       aut.csr_indptr, aut.is_neg, nc, V_PROD)
    Z = np.asarray((p_lv * r).sum(axis=1))
    live = Z > 0
    spread = float(np.log(Z[live]).max() - np.log(Z[live]).min())
    reported = bool((~rep).any())
    assert spread < 1e-6 or reported, (
        f"T={temperature}: log Z_i spans {spread:.4g} nats across positions and "
        f"the representation guard reported no damage ({int(rep.sum())}/"
        f"{rep.size} positions healthy). `q_i` is reweighted by up to "
        f"e^{spread:.1f} with nothing raised and `H(q_i)` still inside "
        f"[0, log V] — SPEC §3.4's accept rule reads it as a confidence.")


def test_a_branch_that_cannot_finish_in_the_canvas_contributes_no_token():
    """The **exact-zero decision**, which nothing else in the repo pins.

    SPEC §2.4 eq (5)/(6) and §3.1b together say `u_i(e)` must be exactly `0.0`
    on any edge no accepted, within-budget path uses — and `mask`, `mar` and
    `q_i` all read that zero as a *set membership* answer, not as a small
    number. So "return 0.0 rather than something tiny" is a decision with a
    consumer-visible consequence, and it deserves a test that fails when the
    decision is reversed.

    The grammar: from the start state either one token into `ACC` (then SPEC
    §3.5's unscored tail), or into a 9-step chain that also accepts — but nine
    steps do not fit in an 8-token canvas. So the chain's token is live in the
    *graph* and dead in the *language at this length*, and the support at
    position 0 must be exactly `{A}`.

    `p` is uniform on purpose: at `L = 8` nothing here can underflow, so this
    isolates the zero decision from the loss the rest of the file is about. It
    is expected to be **green** today and to stay green through any fix.
    """
    V, bucket = 8, 16
    TOK_A, TOK_B = 2, 3
    chain = tuple((s, s + 1, 1) for s in range(2, 11))       # 2 -> ... -> 11
    edges = ((0, 1, 0), (1, 1, 2), (0, 2, 1)) + chain
    members = ((TOK_A,), (TOK_B,), tuple(range(V)))
    finals = {1, 11}
    indices, indptr = [], [0]
    for c in members:
        indices.extend(sorted(c))
        indptr.append(len(indices))
    d = np.full(bucket, INF_DISTANCE, np.int32)
    for s, dist in _distance_to_final(
            12, [(e[0], e[1]) for e in edges], finals).items():
        d[s] = dist
    is_final = np.zeros(bucket, bool)
    is_final[sorted(finals)] = True
    act = np.zeros(bucket, bool)
    act[0] = True
    aut = Automaton(
        edge_src=jnp.asarray([e[0] for e in edges], jnp.int32),
        edge_dst=jnp.asarray([e[1] for e in edges], jnp.int32),
        edge_class=jnp.asarray([e[2] for e in edges], jnp.int32),
        edge_valid=jnp.ones(len(edges), bool),
        csr_indices=jnp.asarray(indices, jnp.int32),
        csr_indptr=jnp.asarray(indptr, jnp.int32),
        is_neg=jnp.zeros(len(members), bool),
        d=jnp.asarray(d), is_final=jnp.asarray(is_final),
        active=jnp.asarray(act))
    assert int(d[2]) == 9, "fixture: the chain must not fit in the canvas"

    L = 8
    p = jnp.full((L, V), 1.0 / V, dtype=jnp.float64)
    fb = _forward_backward(p, aut, 0, bucket, len(members))
    b_final = C.budget_terminal_factor(aut.d, 0, dtype=jnp.float64)
    st = _structural(fb, aut, b_final)
    got = _support(fb.u, aut, len(members), V)
    want = _support(st["edge_live"].astype(np.float64), aut, len(members), V)

    assert not want[0, TOK_B], (
        "fixture is wrong: the oracle thinks the 9-step chain fits in 8 tokens")
    assert want[0, TOK_A], "fixture is wrong: the short branch must be legal"
    assert not got[0, TOK_B], (
        f"token {TOK_B} reached the support at position 0 through a branch "
        f"that cannot reach an accepting state inside the canvas. `u` on that "
        f"edge came back {float(np.asarray(fb.u)[0, 2]):.3e}, not exactly 0.0 "
        f"— every consumer reads that as membership.")
    assert np.array_equal(got, want), (
        f"support differs from the structural projection at "
        f"{int((got != want).sum())} (position, token) pairs on a case where "
        f"nothing can underflow")


@pytest.mark.parametrize("grammar", ["sudoku", "countdown"])
def test_mar_confidence_is_a_real_entropy(grammar, request):
    """SPEC §2.4/§3.4: `q_i` is the paper's **Mar** confidence signal.

    `H(q_i) = −Σ_v q_i(v) log q_i(v)` is the entropy of a probability
    distribution, so `0 ≤ H ≤ log V` at every position — the acceptance rule in
    §3.4 compares it against `entropy_bound` and a value outside that range is
    not a confidence, it is a defect leaking into the accept mask.

    `constrained_entropy_streamed` returns `log Z_i − (1/Z_i)Σ w log w` with
    `Z_i` floored at `finfo.tiny`, so a position whose `u` has been zeroed
    reports `log(tiny) = −708` nats: not `NaN`, not an exception, and *below*
    any `entropy_bound`, so the position is silently marked most-confident.
    """
    case = request.getfixturevalue(f"{grammar}_prod")
    V = int(case.p_lv.shape[-1])
    h = np.asarray(MG.constrained_entropy_streamed(
        case.p_lv.T, case.fb.u, case.aut.edge_class, case.aut.csr_indices,
        case.aut.csr_indptr, case.aut.is_neg, case.n_classes, V))
    # A position with exactly one legal token has `H = 0`, which rounds to
    # `-1e-17`; the tolerance is for that, and is 700 orders of magnitude away
    # from the `log(tiny) = -708` this test exists to catch.
    bad = np.nonzero(~np.isfinite(h) | (h < -1e-9) | (h > np.log(V) + 1e-9))[0]
    assert bad.size == 0, (
        f"{case.tag}: `--confidence=mar` returned a value outside [0, log V] "
        f"at {bad.size} of {h.size} positions, e.g. H[{int(bad[0])}] = "
        f"{h[bad[0]]:.1f} nats (median over all positions: "
        f"{np.median(h):.2e}). Anything below `entropy_bound` is treated as "
        f"maximal confidence by SPEC §3.4's accept rule.")


def test_the_log_partition_is_i_invariant_on_a_real_grammar(sudoku_prod):
    """SPEC §2.4, marked `[D]`, verbatim:

    > `log Z = log(a_i·b_i) + logscale_a[i] + logscale_b[i]` is **`i`-invariant**
    > — assert it across all `i`.

    This is `prefix_suffix`'s own self-consistency, over its own four return
    values — nothing here re-derives `Z`. A dot product that underflows to
    exactly `0.0` makes the expression `-inf` at some boundaries and finite at
    others, so the invariant SPEC asks to be asserted is the one the defect
    breaks. The existing suite asserts it at `|S| = 64` on a synthetic regime;
    this is the compiled grammar at the checkpoint's own sharpness.
    """
    logZ = sudoku_prod.fb.log_partition_per_boundary()
    bad = np.nonzero(~np.isfinite(logZ))[0]
    assert bad.size == 0, (
        f"log Z is not finite at {bad.size} of {logZ.size} boundaries "
        f"({bad[:10].tolist()}) at mean entropy {sudoku_prod.H.mean():.2e} "
        f"nats — `a_i · b_i` underflowed to exactly 0 there")
    spread = float(logZ.max() - logZ.min())
    assert spread < 1e-6, f"log Z varies by {spread:.4g} nats across boundaries"


@pytest.mark.parametrize("grammar", ["sudoku", "countdown", "sudoku_sharp"])
def test_the_per_position_partition_is_position_invariant(grammar, request):
    """SPEC §2.4, and the non-vacuous form of `Σ_v q_i(v) == 1`.

    `Σ_v p_i(v)·r_i(v) = Z` for **every** `i`, because both sides are the same
    sum over accepted length-`L` strings grouped by a different position. That
    equality is checkable without knowing `Z`, and unlike the row sums it is
    not satisfied by construction — `constrained_marginals_and_partition`
    divides by its own row sum, so `Σ_v q_i(v) == 1` holds bit-for-bit under an
    arbitrary scale error, and a mutation audit has already shown three
    different injected bugs leaving it at exactly 1.0.
    """
    case = request.getfixturevalue(f"{grammar}_prod")
    V = int(case.p_lv.shape[-1])
    if case.fb.a is not None:
        # The production consumer itself.
        _q, Z = MG.constrained_marginals_and_partition(
            case.p_lv.T, case.fb.a, case.fb.b, case.aut.edge_src,
            case.aut.edge_dst, case.aut.edge_class, case.aut.csr_indices,
            case.aut.csr_indptr, case.aut.is_neg, case.n_classes)
    else:
        # The repaired path does not expose linear `a`/`b`; `Z_i` is the same
        # `Σ_v p_i(v)·r_i(v)` through the same scatter. Only a per-position
        # scale differs, and this test is scale-free by construction.
        r = MG.scatter_edge_mass_to_tokens(
            case.fb.u, case.aut.edge_class, case.aut.csr_indices,
            case.aut.csr_indptr, case.aut.is_neg, case.n_classes, V)
        Z = (case.p_lv * r).sum(axis=1)
    Z = np.asarray(Z)
    assert np.all(Z > 0), (
        f"{case.tag}: Z_i == 0 at {int((Z <= 0).sum())} of {Z.size} positions "
        f"({np.nonzero(Z <= 0)[0][:10].tolist()}) while other positions carry "
        f"mass — the same string set summed two ways cannot be zero one way "
        f"and positive the other")
    # `u` may carry a per-position scale (the repair anchors it there); add it
    # back before comparing across positions, or the invariant is not about Z.
    logZ = np.log(Z) + np.asarray(case.fb.log_u_scale)
    rel = float(logZ.max() - logZ.min())
    assert rel < 1e-6, f"{case.tag}: log Z_i spans {rel:.4g} nats across positions"


@pytest.mark.parametrize("grammar", ["sudoku", "countdown"])
def test_a_lost_position_is_not_silent(grammar, request):
    """The property that makes this expensive: **nothing reports it.**

    SPEC §6.3 and CLAUDE.md require `Z == 0` to be classified and raised, and
    `[V-P5]` records that a root-mass predicate now rides
    `ConstrainedSamplingState.feasible` out of the block loop precisely because
    a degenerate draw is otherwise indistinguishable from a good one.

    This asserts the *implication*, not the defect: **if** live positions are
    lost, **then** at least one guard must fire. It passes when the loss is
    gone and it passes when a detector is added; it fails only in the state
    that costs a published number — loss present, every guard green. Written
    this way on purpose, because `Σ_v q_i(v) == 1` was asserted for weeks in a
    form mathematically incapable of failing.
    """
    case = request.getfixturevalue(f"{grammar}_prod")
    V = int(case.p_lv.shape[-1])
    got = _support(case.fb.u, case.aut, case.n_classes, V)
    want = _support(case.st["edge_live"].astype(np.float64), case.aut,
                    case.n_classes, V)
    lost = int((want & ~got).sum())

    # Every guard the pipeline actually has, on this exact input.
    guards = {}
    raised = None
    try:
        _tok, feasible = C.joint_draw(
            case.p_lv, case.aut, jnp.int32(case.remaining),
            jax.random.PRNGKey(0), case.n_states, case.n_classes)
        guards["joint_draw.feasible"] = bool(feasible)
    except Exception as exc:                                   # noqa: BLE001
        raised = f"joint_draw raised {type(exc).__name__}: {exc}"
    h = np.asarray(MG.constrained_entropy_streamed(
        case.p_lv.T, case.fb.u, case.aut.edge_class, case.aut.csr_indices,
        case.aut.csr_indptr, case.aut.is_neg, case.n_classes, V))
    guards["mar entropy is NaN"] = bool(np.isnan(h).any())
    guards["mask support empty everywhere"] = bool(not got.any())

    silent = (raised is None and guards.get("joint_draw.feasible", False)
              and not guards["mar entropy is NaN"]
              and not guards["mask support empty everywhere"])
    assert not (lost > 0 and silent), (
        f"{case.tag}: {lost} legal (position, token) pairs were dropped and "
        f"every guard reports health — {guards}, no exception. A published "
        f"`mask` or `--confidence=mar` column computed here would carry no "
        f"warning of any kind.")


# ===========================================================================
# 3. End to end, through the real denoising loop
# ===========================================================================

@dataclasses.dataclass(frozen=True)
class _Toy:
    """A 12-token mandatory prefix followed by SPEC §3.5's unscored tail.

    States `0 … 11` are the grammar, each step consuming token `3`; state `12`
    is `ACC` with the unscored `ACC --Σ--> ACC` self-loop at mass exactly 1.0.
    That is the shape §3.5 trap 4 describes and the one every compiled grammar
    in this repository has.

    It is the minimal structure that produces a **completely dead position**
    rather than merely a dead edge: the forward vector is one-hot on the chain
    (so max-normalisation rescues it), while the backward vector spans the
    whole chain — `b_i(s) = w^{11−s}` — so its far end underflows and takes the
    only live edge at the early positions with it.
    """
    vocab: int = 8
    bucket: int = 16
    n_states: int = 13
    #: (src, dst, class)
    edges: tuple = tuple((j, j + 1, 0) for j in range(12)) + ((12, 12, 1),)
    #: class -> member tokens
    members: tuple = ((3,), (0, 1, 2, 3, 4, 5, 6, 7))
    finals: frozenset = frozenset({12})
    start: frozenset = frozenset({0})

    def traced(self, batched=False) -> Automaton:
        indices, indptr = [], [0]
        for c in self.members:
            indices.extend(sorted(c))
            indptr.append(len(indices))
        d = np.full(self.bucket, INF_DISTANCE, np.int32)
        for s, dist in _distance_to_final(self.n_states,
                                          [(e[0], e[1]) for e in self.edges],
                                          self.finals).items():
            d[s] = dist
        is_final = np.zeros(self.bucket, bool)
        is_final[sorted(self.finals)] = True
        act = np.zeros(self.bucket, bool)
        for s in self.start:
            act[s] = True
        return Automaton(
            edge_src=jnp.asarray([e[0] for e in self.edges], jnp.int32),
            edge_dst=jnp.asarray([e[1] for e in self.edges], jnp.int32),
            edge_class=jnp.asarray([e[2] for e in self.edges], jnp.int32),
            edge_valid=jnp.ones(len(self.edges), bool),
            csr_indices=jnp.asarray(indices, jnp.int32),
            csr_indptr=jnp.asarray(indptr, jnp.int32),
            is_neg=jnp.zeros(len(self.members), bool),
            d=jnp.asarray(d), is_final=jnp.asarray(is_final),
            active=jnp.asarray(act[None, :] if batched else act))


def _distance_to_final(n_states, edges, finals) -> dict:
    """`d(s)`, SPEC §3.1b — reverse BFS, one token per edge. Re-derived here."""
    back: dict[int, list[int]] = {s: [] for s in range(n_states)}
    for s, t in edges:
        back[t].append(s)
    d = {s: INF_DISTANCE for s in range(n_states)}
    frontier = list(finals)
    for s in frontier:
        d[s] = 0
    while frontier:
        nxt = []
        for t in frontier:
            for s in back[t]:
                if d[s] > d[t] + 1:
                    d[s] = d[t] + 1
                    nxt.append(s)
        frontier = nxt
    return d


class _StubbedSampler(S.ConstrainedDiffusionSampler):
    """`ConstrainedDiffusionSampler` with only the transformer replaced."""

    def sample_step(self, *, canvas, sc_embeddings, cache, positions,
                    attention_mask, sliding_attention_mask,
                    current_noise_proportion, target_noise_proportion,
                    params, rng):
        return DS.SampleStepOutput(
            sc_embeddings=jnp.zeros_like(sc_embeddings),
            logits=jnp.broadcast_to(_STUB["logits"],
                                    (canvas.shape[0],) + _STUB["logits"].shape[-2:]),
            sampled_tokens=canvas,
            modified_tokens_mask=jnp.zeros_like(canvas, bool))


_STUB: dict = {}


class _FakeConfig:
    embed_dim = 4


class _FakeModel:
    config = _FakeConfig()


def _run_mask_variant(spec, logits, L, *, steps=2, seed=0, remaining=0):
    _STUB["logits"] = jnp.asarray(logits)
    sampler = _StubbedSampler(
        model=_FakeModel(), end_tokens=(1,), forbidden_tokens=None,
        sampling=None, cache_length=64, special_tokens=None, canvas_length=L,
        max_denoising_steps=steps, text_vocab_size=spec.vocab,
        n_states_bucket=spec.bucket, n_classes=len(spec.members),
        variant="mask", emission="sample", confidence="mf",
        sample_from_predictions=DS.SampleFromPredictions(
            entropy_bound=0.1, text_vocab_size=spec.vocab))
    cache = {"l0": {"k": jnp.zeros((1, 64, 1, 1)),
                    "end_index": jnp.zeros((1,), jnp.int32)}}
    emit, _feasible = sampler.sample_next_canvas_constrained(
        canvas_length=L, max_denoising_steps=steps, batch_size=1, cache=cache,
        params=None, rng=jax.random.PRNGKey(seed),
        full_attention_mask=jnp.ones((1, 64), bool),
        automaton=spec.traced(batched=True), remaining=jnp.int64(remaining))
    return [int(x) for x in emit[0]]


def test_the_mask_baseline_emits_only_tokens_in_the_per_position_projection():
    """SPEC §2.8 / §7.2 baseline 2, end to end through the real denoising loop.

    This is the acceptance test: it goes through
    `sample_next_canvas_constrained` unmodified, so no harness of this file's
    can stand in for the fix.

    The `mask` baseline is *defined* as sampling independently from `π_i(C)`.
    Whatever it emits at position `i` must therefore be a token some accepted
    string carries at `i` — that is the weakest possible correctness statement
    about it, and the one SPEC §2.8's argument (`∏_i π_i(C) ⊋ C`) presupposes.

    The logits are inside the checkpoint's own softcap window (`|logit| ≤ 75`
    at `T = 0.4`), so this is a configuration the released model can produce.
    """
    spec = _Toy()
    L = 16
    # The model is confident about token 2, which the grammar forbids at every
    # grammar position — the ordinary case, and the one `class_weights`'
    # docstring quotes Phase 0 for. `|logit| = 42.5` is inside the softcap
    # window `[-75, +75]` at `T = 0.4`, so the checkpoint can produce this.
    lg = np.full((1, L, spec.vocab), -42.5)
    lg[0, :, 2] = 42.5           # 85 nats above every grammar token
    assert np.abs(lg).max() <= SOFTCAP / T_PROD + 1e-9

    aut = spec.traced()
    p = _softmax64(lg[0])
    fb = _forward_backward(jnp.asarray(p), aut, 0, spec.bucket,
                           len(spec.members))
    b_final = C.budget_terminal_factor(aut.d, 0, dtype=jnp.float64)
    st = _structural(fb, aut, b_final)
    want = _support(st["edge_live"].astype(np.float64), aut,
                    len(spec.members), spec.vocab)

    toks = _run_mask_variant(spec, lg, L)
    bad = [(i, t) for i, t in enumerate(toks) if not want[i, t]]
    assert not bad, (
        f"the `mask` baseline emitted tokens outside the per-position "
        f"projection at {bad}; the full canvas was {toks}. Legal tokens at "
        f"those positions: "
        f"{[sorted(np.nonzero(want[i])[0].tolist()) for i, _ in bad]}")


# ===========================================================================
# 4. Guards on this file itself
# ===========================================================================

def test_this_harness_is_still_what_sampler_py_runs():
    """`_forward_backward` claims to be `sampler.py`'s sequence. Pin it.

    If the `mask`/`mar` closures stop computing `u` as
    `a[:-1][edge_src] * b[1:][edge_dst]` off `prefix_suffix` — which is exactly
    what fixing this defect should do — this test goes red and forces
    `_forward_backward` to be updated with it, instead of quietly testing a
    sequence nothing runs any more.
    """
    src = open(S.__file__).read()
    body = src[src.index("def sample_next_canvas_constrained"):]
    assert "scans.prefix_suffix(" in body, (
        "sampler.py no longer calls prefix_suffix — update `_forward_backward`")
    pat = re.compile(r"a_v\[:-1\]\[:,\s*aut\.edge_src\]\s*\*\s*"
                     r"b_v\[1:\]\[:,\s*aut\.edge_dst\]")
    n = len(pat.findall(body))
    assert n == 2, (
        f"expected the `mask` and `mar` closures to form u = a·b in the shape "
        f"this file replicates; found {n} occurrences")


def test_the_oracle_is_not_the_implementation():
    """A guard against this file becoming the fifth vacuous test.

    `_bool_reach` must not depend on the floating-point path at all: perturbing
    `p` by orders of magnitude, while keeping every support unchanged, must
    leave the oracle bit-identical. If it moved, the "oracle" would be reading
    the same numbers it is supposed to be judging.
    """
    M, a0, bL = two_lane(32, 1.0)
    M2, _, _ = two_lane(32, 100.0)
    assert np.array_equal(M > 0, M2 > 0), "supports must match by construction"
    la1, lb1 = _bool_reach(M > 0, a0 > 0, bL > 0)
    la2, lb2 = _bool_reach(M2 > 0, a0 > 0, bL > 0)
    assert np.array_equal(la1, la2) and np.array_equal(lb1, lb2)
    # ... and it must be capable of saying "dead": kill the accepting lane.
    M3 = M.copy()
    M3[:, :, 3] = 0.0
    la3, lb3 = _bool_reach(M3 > 0, a0 > 0, bL > 0)
    assert not lb3[0].any(), "the oracle cannot detect an empty language"


if os.environ.get("DGFA_MUT"):
    _apply_mutation(os.environ["DGFA_MUT"])
    for _n in os.environ["DGFA_MUT"].split(","):
        print(f"\n*** DGFA_MUT={_n} ACTIVE — {_MUTANTS[_n]} ***")
