"""The constrained joint draw, assembled from the Phase 3 kernels.

One function per emission mode, each taking the model's shaped logits plus a
traced `Automaton` and returning a whole `[B, L]` canvas drawn from the
constrained posterior. `sample_from_predictions` receives the entire `[B,L,V]`
logits and returns an entire `[B,L]` canvas, and nothing downstream assumes
per-position independence — **joint sampling is fully permitted** (SPEC §5.1).

SPEC §3.1's emission designs:

| | trajectory (fed back to the model) | emitted | guarantee |
|---|---|---|---|
| **J0-map** | stock: constrained sample at accepted, uniform random elsewhere | constrained MAP | unconditional |
| **J0-sample** | as above | constrained joint draw | unconditional |
| **J1** | single joint draw with **flattened** marginals at non-accepted | same tensor | unconditional, *and every intermediate canvas ∈ C* |
| **J2** | constrained sample at accepted, uniform random elsewhere | same tensor | only at full acceptance |

J1 is built first (SPEC §5.4): it needs no change to the denoising carry, so it
is the shortest path to end-to-end constrained generation and validates the
whole inference stack before the carry is touched.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from diffgemma_fa.infer import marginals as MG
from diffgemma_fa.infer import scans, tree

__all__ = [
    "budget_terminal_factor",
    "flatten_unaccepted",
    "map_log_floor",
    "joint_draw",
    "joint_map",
    "advance_states",
    "require_x64",
    "X64Required",
]


class X64Required(RuntimeError):
    """The sum-product sampling path needs float64. See `require_x64`."""


def require_x64() -> None:
    """**Required for `--emission=sample`. Raises when `jax_enable_x64` is off.**

    **[Corrected 2026-08-08 — the no-op is retracted.]** Between 2026-08-01 and
    now the body was `return`, on the rationale (kept below) that
    `log_matmul`'s pairwise-max anchor "removed the need" for float64. That
    rationale is true of the **tree** and false of everything **upstream** of
    it, so the function was silently permitting exactly the configuration its
    own docstring records as failing on 70/130 records.

    **Measured at production shapes**, `L = 256`, on the real compiled
    `BFCL_v4_live_simple` grammar (50 states, 83 classes, `V = 262,144`,
    30,976 live `M_i` entries, 1,168 live root entries). `M_lost32` is the
    float32 loss; `M_lostup` is the same run with `W`/`M` built in float64 —
    the fix this guard was proposed as an alternative to. Every row is a
    **single draw at seed 1**; see point 2 for how much they move across seeds:

        p regime                     M_lost32   M_lostup   root lost   err
        iid Gaussian x 8                    0          0     0/1168   8e-4
        iid Gaussian x 12                  76         76     0/1168   6e-4
        iid Gaussian x 16               2,761      2,761   168/1168    113
        iid Gaussian x 20               9,903      9,903   795/1168    556
        iid Gaussian x 30              20,660     20,660  1107/1168    6.2
        iid Gaussian x 40              25,118     25,118  1158/1168   3e-5
        softcap 30*tanh, T = 1.0            0          0     0/1168   4e-4
        softcap 30*tanh, T = 0.4        1,315      1,315    67/1168   27.6

    Three things follow, and none of them rests on a single point:

    1. **`M_lostup == M_lost32` at every scale.** The float64 upcast inside
       `_matrices` recovers *exactly zero* entries wherever anything is lost.
    2. **The driver is `p` sharpness × temperature, not grammar size.** The
       last two rows are a Gemma-shaped logit distribution: the softcap
       `30·tanh(x/30)` of `gemma/diffusion/_transformer.py:182`, divided by the
       sampler's own `min_temperature = 0.4` (`gemma/diffusion/_sampler.py:236`).
       At `T = 1.0` float32 is lossless; the production temperature alone takes
       it to 67 lost root entries and 27.6 nats of survivor error. The
       survivor error is the dangerous half: lost entries look like `Z == 0`,
       misvalued ones trip nothing at all.

       **Every figure in that row is one draw, and all of them move.** Across
       three seeds at `T = 0.4`: root loss 67 / 38 / 47 of 1,168, edge loss
       1,315 / 1,399 / 1,374 of 30,976, survivor error 27.6 / 27.7 / **77.8**
       nats. The honest ranges are **38–67 root entries, ~1,140–1,400 edges,
       and 27–78 nats** — quote those, not a single point. The `T = 0.408` run
       recorded elsewhere is *not* an independent replication: it is seed 1
       again at the schedule's true final temperature, and the 1,315-vs-1,141
       edge gap is that temperature difference, not a second sample.

       An earlier revision of this docstring read the matching 67/1,168 in two
       runs as "the root count is a property of the grammar". It is not — those
       two runs merely shared seed 1, and the count is 67/38/47 across seeds.
       That was an invariant inferred from n = 2 and written down as measured
       fact, inside the rewrite meant to remove exactly that failure. What *is*
       stable at a fixed seed is the root count across **temperature** (67 at
       both `T = 0.4` and `T = 0.408`), which is nearly the opposite claim.
    3. The iid-Gaussian rows are not a real logit distribution (a 458-nat span
       at scale 40) and should not be leaned on alone — but on *sharpness* they
       are conservative. Scale 40's mean entropy is 0.196 nats, while Phase 0
       measured real final-step entropies of 5e-5 – 4e-4 nats, i.e. sharper.

    **Why the upcast is not the fix, structurally.** Three reasons, in
    increasing order of finality:

    - *The loss is upstream of `_matrices`.* On the real grammar the float32
      `softmax` that produces `p` (`sampler.py`'s `p_real`) is where the
      entries die: at `T = 0.4` it flushes 7.6M of 67.1M entries to zero, and
      `W_lost32 == W_lostup` in every row above, so `class_weights` contributes
      nothing beyond it. A dtype patch inside `_matrices` cannot restore a zero
      it was handed.
    - *The synthetic fixture is the unrepresentative one.* On a `|S| = 64`,
      `V = 4096` toy the upcast **does** work (leaf loss 32 → 0, root loss
      32 → 0, error 7.6e-3 → 3.8e-5 nats) — because its two classes are 2,048
      tokens wide, so a class's mass is a sum over thousands of `p` entries and
      survives even when individual entries flush. The real grammar's 83
      classes over a 262k vocab have a **median of 2 true members** (min 1;
      62 of 83 have ≤ 8), so a class's whole mass *is* a handful of `p` entries
      and dies with them. The toy is where the patch looks like a fix.
    - *And it cannot work at all in the configuration this guard covers.* With
      `jax_enable_x64` **off**, `p.astype(jnp.float64)` is a silent no-op: JAX
      emits a truncation `UserWarning` and returns a float32 array (verified —
      `jnp.zeros(1, jnp.float64).dtype == float32`). The proposed fix is
      structurally incapable of addressing the guarded configuration, at any
      scale, on any fixture. That is the end of the argument.

    Taking the upcast instead of this guard would therefore have bought a green
    synthetic fixture and an unchanged production failure — the precise shape of
    mistake this file already made once.

    ---

    Historical note (**2026-08-01**), retained because the *measurement* stands
    even though the conclusion drawn from it did not:

    This docstring previously declared float64 obsolete on the strength of a
    toy-scale check — `L = 4`, `|S| <= 8`, 20k draws against brute-force
    enumeration, where float32 deviated 0.0025 / 0.0013 against float64's
    0.0033 / 0.0009. That check was real but far too small to generalise, and
    it did not.

    Measured at production scale on the E4 grammar (whitespace-tolerant, so
    roughly double `|S|`), `L = 256`, n = 130 records:

        float64 : Z == 0 on   0/130   (CS 1.000)
        float32 : Z == 0 on  70/130 and 55/130 across two seeds (CS 0.46/0.58)

    So float32 does not silently corrupt the draw — the detector catches it, and
    those records fail loudly rather than emitting garbage — but it makes the
    kernel spuriously infeasible on more than half of them. Whatever margin
    exists at `L = 4` is gone by `L = 256`: 8 tree levels of logsumexp in a
    format whose `exp` underflows at 87 nats cannot hold a grammar whose real
    paths span hundreds.

    The lesson is about the test, not the arithmetic: a numerical claim
    validated only on toy shapes is not validated. Any future attempt at
    float32 must be measured on a real grammar at `L = 256` before it is
    believed.

    ---

    Historical note on why the OLD failure was different:

    The pre-2026-07-31 kernel failed for a *different* reason, which the
    pairwise-max anchor did genuinely fix — it is worth keeping distinct from
    the dtype question above:

    - the old kernel exponentiated against `ra[i] + cb[j]`, a shift derived
      from row/column maxima that the unscored `ACC --Σ--> ACC` tail pins at
      0.0 while genuine grammar paths sit ~850 nats below. Terms landed at
      `exp(-423)·exp(-423) ≈ 1e-368` and underflowed *even in float64*;
    - `log_matmul` now anchors each entry on its own **pairwise max** (the
      max-plus product), so the dominant term of every entry is `exp(0) = 1`
      by construction. In float32 anything below `exp(-87)` is dropped —
      relative weight `1e-38` *of its own entry*, not of a foreign anchor.

    Measured on the record that exposed the whole problem
    (`live_simple_106-63-0`, 403 states, L = 256, adversarial per-position
    sharp `p`): feasible and simulator-accepted in float32 as well as float64.
    Distributionally, against brute-force enumeration on DFAs and NFAs, the
    float32 draw deviates by 0.0025 / 0.0013 against float64's 0.0033 / 0.0009
    — i.e. indistinguishable, both far inside the 0.02 threshold.

    **Consequences.** SPEC §5.6's memory table halves back: the tree is
    `(2L−1)·|S|²·4` bytes again, restoring the `|S|` ceiling from 2,211 to
    3,128 at 20 GB of headroom. It is also the likely cure for the
    intermittent `CUDA_ERROR_OUT_OF_MEMORY` that killed two n=130 arms — the
    exact kernel needs more headroom than the GEMM form did, and float32
    gives half of it straight back.

    That paragraph is what turned this function into a no-op. It is a claim
    about the *anchor*, and the anchor was never the only thing that underflows:
    it says nothing about `softmax`, `class_weights` or `transition_matrices`,
    all of which run before the first `log_matmul` and all of which lose entries
    in float32 at `L = 256` (numbers at the top).

    ---

    Historical rationale (**and it is again the operative one**):

    **Measured, on a real BFCL grammar at `L = 64`.** SPEC §2.4/§2.6 prescribe
    max-normalizing every tree node, which fixes the *overall* scale — but not
    the **dynamic range inside a single matrix**, and at the root that range is
    what kills fp32:

    - `ACC --Σ--> ACC`, the unscored post-stop tail (§3.5 trap 4), has emission
      mass **exactly 1.0** by construction, so the root's max is pinned at 1.0
      and per-node normalization divides by 1.0 and does nothing;
    - a genuine constrained path is a product of per-token probabilities around
      `4e-6`, so over 64 positions the root entry is `~1e-49` and the smallest
      positive entry measured was **1.2e-288**.

    float32's smallest subnormal is ~1e-45, so the entire joint underflows to
    exactly zero and the root draw degenerates. In float64 the identical code
    samples correctly and the draw is accepted.

    This is SPEC's `Z == 0` cause **(c)** — "fp32 underflow on an unnormalized
    path… means your scaling is missing" — except that the scaling is *present*
    and still insufficient, because the tail edge pins the normalizer.

    Two consequences:

    - **`--emission=map` is unaffected**, because §2.7 puts MAP in log space
      specifically so that "no scaling discussion, no underflow" applies. That
      choice is now empirically vindicated and is an argument for MAP being the
      default emission.
    - **SPEC §5.6's memory table doubles for the sampling path**: the tree is
      `(2L−1)·|S|²·8` bytes, so at the measured ~20 GB of headroom the `|S|`
      ceiling falls from 3,128 to **2,211** (and from 1,978 to 1,399 at 8 GB).

    Raises:
      X64Required: if float64 is disabled, in which case the sampler would
        silently draw from a degenerate distribution rather than fail.
    """
    # Read the flag off the live config, not off a cached value: the eval
    # entry points call `jax.config.update` at import time, and tests toggle it
    # inside a single process.
    if not jax.config.read("jax_enable_x64"):
        raise X64Required(
            "the constrained sum-product draw (`--emission=sample`) needs "
            "float64, but `jax_enable_x64` is off, so every `jnp.float64` "
            "here silently becomes float32. Measured at L = 256: float32 "
            "loses 32 of 64 live root entries on a synthetic |S|=64 grammar "
            "(deterministic, fixed-seed fixture) and the large majority of "
            "the live M_i entries on a real BFCL grammar with sharp p — 81% "
            "on one draw, but see the docstring: every such figure is "
            "draw-dependent and none of them is a bare fact. Either way it "
            "is a spurious `Z == 0` rather than a draw. Call "
            "`jax.config.update('jax_enable_x64', True)` before constructing "
            "the sampler, or use `--emission=map`, which SPEC §2.7 puts in "
            "log space and is unaffected."
        )


def map_log_floor(dtype) -> float:
    """The floor `joint_map` puts under `p` before taking `log`. SPEC §2.7.

    **This was `1e-30` and that was a second, independent defect.** SPEC §2.4's
    `1e-30` clamp is a rule about `q`, which has *exact zeros* wherever the
    automaton forbids a token; there the clamp only ever multiplies a zero, so
    its value is arbitrary. `p` is a softmax: it has no exact zeros, and the
    clamp is a **floor on live values**. Every token below it scores the same
    `log(1e-30)`, and `argmax` then resolves the tie by lowest id (SPEC §2.7's
    convention) — so MAP silently emitted the *smallest admissible token id*
    rather than the most probable one, with `feasible` still True.

    `1e-30` sits well *above* the range the released model produces. Measured on
    a Gemma-shaped canvas (softcap `30·tanh(x/30)`, schedule temperature 0.408)
    at the real `L = 256`, `V = 262,144`: the smallest marginal is `3.7e-58`,
    and on the compiled Countdown grammar **1,162 of 6,144 (class, position)
    pairs had their entire class under the floor**. So SPEC §2.7's "MAP ==
    exhaustive argmax, error 0.000e+00" did not hold as stated.

    **What this has not been shown to do is change a production emission.**
    Across 6 seeds on real Countdown, with 1,162–1,213 fully-clamped
    `(class, position)` pairs each time, swapping the floor changed **0 of 256
    tokens and 0.00 nats of score, every time**: the fully-clamped classes did
    not lie on the winning max-plus path. The fix is still correct and still
    necessary — a mutant kills a test on the minimal fixture, where the class
    that goes under the floor *is* on the path — but do not attribute the MAP
    Countdown arm's repeated `1 -1=1` bodies to this clamp. That attribution was
    made upstream of the measurement and the measurement does not support it;
    the cause of that repetition is open.

    The floor has to sit under everything the model can represent, not under an
    unrelated round number, so it is derived from the dtype: `finfo(dtype).tiny`
    is the smallest normal, `log` of it is about `-708` in float64 — finite,
    order-preserving over every normal `p`, and ~1e38 clear of
    `scans.NEG_SENTINEL`, so it can never be mistaken for "impossible". The
    §3.1b guarantee was never affected either way: feasibility is a max over a
    non-empty class of a finite `log p` and is a property of the automaton and
    the budget alone (`tests/test_audit_partition.py`).
    """
    return float(jnp.finfo(dtype).tiny)


def budget_terminal_factor(d: jnp.ndarray, remaining: jnp.ndarray,
                           dtype=jnp.float64) -> jnp.ndarray:
    """`b_L(s) = 1[d(s) ≤ R]`. SPEC §3.1b.

    **Not** `1[s ∈ F]`, and with **no final-block special case**: `d(s) ≤ 0 ⟺
    s ∈ F`, so the last block falls out automatically as `R` runs down. Using
    `1[s ∈ F]` instead would force the grammar to complete in exactly one
    canvas.
    """
    return (d <= remaining).astype(dtype)


def flatten_unaccepted(
    p: jnp.ndarray,
    accepted: jnp.ndarray,
    *,
    temperature: float = 1e6,
) -> jnp.ndarray:
    """J1's `p'`: the model's marginals where accepted, flattened elsewhere.

    Accepted positions keep the model's confident choice; the rest become
    near-uniform **but still grammar-consistent**, because the joint draw that
    follows is constrained. That is what makes every intermediate canvas a
    member of `C` — strictly stronger than the paper, whose intermediate
    canvases contain `[MASK]`s.

    The risk, which is why SPEC wants this ablated against J0: the model was
    trained to denoise **uniform random** noise, not grammar-valid noise.

    Args:
      p: `[L, V]` marginals.
      accepted: `[L] bool`.
      temperature: how flat the non-accepted rows become. Large ⇒ uniform.
    """
    flat = jnp.full_like(p, 1.0 / p.shape[-1])
    blended = jnp.where(accepted[:, None], p, flat)
    if temperature != 1e6:
        blended = jnp.where(accepted[:, None], p, p ** (1.0 / temperature))
        blended = blended / blended.sum(axis=-1, keepdims=True)
    return blended


def _matrices(p_lv, automaton, n_states, n_classes):
    """`W` and `M_i` from the class tables. Invalid (padding) edges are zeroed
    so a bucketed automaton behaves exactly like an unpadded one."""
    p_vl = p_lv.T
    seg = jnp.repeat(
        jnp.arange(n_classes, dtype=jnp.int32),
        jnp.diff(automaton.csr_indptr),
        total_repeat_length=automaton.csr_indices.shape[0],
    )
    W_c = MG.class_weights(p_vl, automaton.csr_indices, seg, automaton.is_neg,
                           n_classes)
    W_e = MG.edge_weights(W_c, automaton.edge_class)
    W_e = jnp.where(automaton.edge_valid[:, None], W_e, 0.0)
    M = MG.transition_matrices(W_e, automaton.edge_src, automaton.edge_dst,
                               n_states)
    return p_vl, W_e, M


@functools.partial(jax.jit, static_argnames=("n_states", "n_classes"))
def joint_draw(
    p_lv: jnp.ndarray,
    automaton,
    remaining: jnp.ndarray,
    key: jax.Array,
    n_states: int,
    n_classes: int,
) -> jnp.ndarray:
    """One draw from the constrained posterior over the whole canvas.

    `a_start = 1[s ∈ A_k]` — a **vector**, not a point mass (SPEC §5.7) — and
    `b_L = 1[d(s) ≤ R]`.

    Returns:
      `(tokens [L] int32, valid bool)`. **`valid` is False when the boundary
      draw degenerated**, which at `L = 256` happens even in float64 — see
      `require_x64` and `docs/PHASE4_FINDINGS.md`. Callers must check it; the
      degenerate draw is near-uniform over the full 262k vocab and looks like
      plausible multilingual text.
    """
    p_vl, W_e, M = _matrices(p_lv, automaton, n_states, n_classes)
    neg = jnp.asarray(scans.NEG_SENTINEL, dtype=p_lv.dtype)
    log_a = jnp.where(automaton.active, jnp.zeros((), p_lv.dtype), neg)
    log_b = jnp.where(automaton.d <= remaining, jnp.zeros((), p_lv.dtype), neg)

    # Log space, not linear: at L = 256 the linear form underflows to exactly
    # zero even in float64, because the unscored ACC --Sigma--> ACC tail pins
    # every node's max at 1.0 while real grammar paths sit below 1e-49.
    logM = jnp.where(M > 0, jnp.log(jnp.maximum(M, jnp.finfo(p_lv.dtype).tiny)),
                     neg)
    tr = scans.up_sweep_log(logM)
    k1, k2 = jax.random.split(key)
    states, feasible = tree.sample_states_log(tr, log_a, log_b, k1)
    tokens, valid = tree.sample_tokens(
        p_vl, states, automaton.edge_src, automaton.edge_dst,
        automaton.edge_class, automaton.csr_indices, automaton.csr_indptr,
        automaton.is_neg, n_classes, k2,
    )
    # `feasible` is the real Z == 0 signal; `valid` only says the drawn state
    # path traverses existing edges, which is near-powerless (measured: True on
    # 200/200 draws from a provably empty language). Both are returned, and
    # callers must check `ok`.
    return tokens, jnp.logical_and(feasible, valid)


@functools.partial(jax.jit, static_argnames=("n_states", "n_classes"))
def joint_map(
    p_lv: jnp.ndarray,
    automaton,
    remaining: jnp.ndarray,
    n_states: int,
    n_classes: int,
) -> jnp.ndarray:
    """Exact constrained MAP over the max-plus tree. SPEC §2.7, `--emission=map`.

    MAP is **exactly temperature-invariant**, so it is unaffected by the
    0.8 → 0.408 schedule — temperature enters only through sampling and the
    accept rule.

    Returns:
      `[L] int32`.
    """
    L, V = p_lv.shape
    logp = jnp.log(jnp.maximum(p_lv, map_log_floor(p_lv.dtype)))

    # Per-class max and argmax over the *true* member set, complement-aware.
    # A negated class is evaluated as "the first topk(p, K) entry not in N_c";
    # here the padded CSR makes a direct masked max simpler and exact, which is
    # what the reference does too.
    seg = jnp.repeat(
        jnp.arange(n_classes, dtype=jnp.int32),
        jnp.diff(automaton.csr_indptr),
        total_repeat_length=automaton.csr_indices.shape[0],
    )
    onehot = jnp.zeros((n_classes, V), dtype=bool).at[seg, automaton.csr_indices].set(True)
    member = jnp.where(automaton.is_neg[:, None], ~onehot, onehot)   # [C, V]

    scored = jnp.where(member[:, None, :], logp[None, :, :],
                       jnp.asarray(scans.NEG_SENTINEL, dtype=logp.dtype))
    class_max = jnp.max(scored, axis=-1)                              # [C, L]
    class_arg = jnp.argmax(scored, axis=-1).astype(jnp.int32)         # ties -> lowest id

    per_edge = class_max[automaton.edge_class, :]                     # [E, L]
    per_edge = jnp.where(automaton.edge_valid[:, None], per_edge,
                         scans.NEG_SENTINEL)
    Mt = jnp.full((L, n_states, n_states), scans.NEG_SENTINEL,
                  dtype=logp.dtype)
    Mt = Mt.at[:, automaton.edge_src, automaton.edge_dst].max(per_edge.T)

    neg = jnp.asarray(scans.NEG_SENTINEL, dtype=logp.dtype)
    a_log = jnp.where(automaton.active, jnp.zeros((), logp.dtype), neg)
    b_log = jnp.where(automaton.d <= remaining, jnp.zeros((), logp.dtype), neg)

    tr = scans.up_sweep_maxplus(Mt)
    tokens, _, _, feasible = tree.map_states_and_tokens(
        logp.T, tr, automaton.edge_src, automaton.edge_dst,
        automaton.edge_class, class_max, class_arg, a_log, b_log,
    )
    # MAP's own score is a perfect Z == 0 detector and used to be discarded.
    return tokens, feasible


@functools.partial(jax.jit, static_argnames=("n_states", "n_classes", "vocab_size"))
def advance_states(
    automaton,
    tokens: jnp.ndarray,
    n_states: int,
    n_classes: int,
    vocab_size: int,
) -> jnp.ndarray:
    """`A_{k+1} = δ*(A_k, canvas_k)`. SPEC §3.5.

    Computed from the **truncated** canvas, and as traced ops on fixed-shape
    arrays — there is no Python between blocks, so `A_k` is a `[S]` array and
    never a Python set.

    Membership of each token in each class is recovered complement-aware from
    the CSR, so this is correct for negated classes.

    Returns:
      `(active [S] bool, ok bool)` — `ok` is False if the state set was ever
      emptied. **It must be checked**; see the no-fallback comment below.
    """
    seg = jnp.repeat(
        jnp.arange(n_classes, dtype=jnp.int32),
        jnp.diff(automaton.csr_indptr),
        total_repeat_length=automaton.csr_indices.shape[0],
    )
    onehot = jnp.zeros((n_classes, vocab_size), dtype=bool).at[
        seg, automaton.csr_indices].set(True)
    member = jnp.where(automaton.is_neg[:, None], ~onehot, onehot)  # [C, V]

    def step(carry, tok):
        active, ever_empty = carry
        edge_ok = member[automaton.edge_class, tok] & automaton.edge_valid
        live = active[automaton.edge_src] & edge_ok
        nxt = jnp.zeros((n_states,), dtype=bool).at[automaton.edge_dst].max(live)
        # NO FALLBACK. This previously did `where(nxt.any(), nxt, active)`,
        # justified by PAD after a stop token — but that justification is FALSE
        # for the compiled automata: `ACC --Σ--> ACC` spans range(vocab_size),
        # so PAD and every end token are already absorbed by the unscored tail.
        #
        # Substituting a stale carry for an empty set makes SPEC §3.1b closure 2
        # UNSOUND rather than merely unenforced: the conjunct
        # `done ∧ (A_{k+1} ∩ F ≠ ∅)` can then be TRUE for a string not in L(M),
        # i.e. the system affirmatively reports acceptance of a rejected string.
        return (nxt, ever_empty | ~nxt.any()), None

    (final, ever_empty), _ = jax.lax.scan(
        step, (automaton.active, jnp.bool_(False)), tokens)
    return final, ~ever_empty
