"""Constrained `DiffusionSampler`. SPEC §5.1–§5.4.

**Fork surface, as Phase 0 corrected it.** CLAUDE.md and SPEC §5.3(c) originally
said `_sample_step` need not be forked. It must be:

- `sample_next_canvas` **never receives `state`** — its stock signature is
  `(canvas_length, max_denoising_steps, batch_size, cache, params, rng,
  full_attention_mask)`, so neither `A_k` nor `predicted_tokens` is reachable
  from the one place the constrained sampler has to live;
- `max_new_tokens` is **not a field of `SamplingState`**, only a parameter of
  `_sample_loop` captured in its `cond_fn` closure, so `R` is not computable in
  `_sample_step` either.

Both `A_k` and `R` are therefore threaded explicitly through a widened
`SamplingState` (`model/state.py`), and `_sample_step` is overridden to advance
`A_k` across the block boundary.

**Everything on `self` must stay static and hashable** — it is a
`static_argname` on both loops, and the class docstring of `SamplerLoop` says so
outright. Config ints live here; **automaton arrays never do**, or every
distinct grammar triggers a full recompile.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import override

import flax.struct
import jax
import numpy as np
import jax.numpy as jnp
from gemma.diffusion import _sampler as _diffusion_sampler
from gemma.gm.text import _sampler_loop

from diffgemma_fa.infer import marginals as _marginals
from diffgemma_fa.infer import scans
from diffgemma_fa.model import constrained as _constrained
from diffgemma_fa.model.state import Automaton, ConstrainedSamplingState, widen

__all__ = ["ConstrainedDiffusionSampler", "_ConstrainedCarry", "DIAGNOSTICS"]


class _Diagnostics:
    """Host-side sink for per-denoising-step traces.

    Module state rather than an attribute, because `self` is a
    `static_argname` and anything mutable on it would be baked into the trace.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.enabled = False

    def reset(self) -> None:
        self.rows = []

    def record(self, step, n_accepted, mean_entropy, emitted, trajectory):
        self.rows.append({
            "step": int(step),
            "n_accepted": int(n_accepted),
            "mean_entropy": float(mean_entropy),
            "emitted": [int(x) for x in emitted],
            "trajectory": [int(x) for x in trajectory],
        })


DIAGNOSTICS = _Diagnostics()
_DIAG = DIAGNOSTICS


@flax.struct.dataclass
class _ConstrainedCarry:
    """SPEC §5.4's widened denoising carry.

    `canvas` is the **trajectory** — under J0 it keeps stock uniform renoising,
    so the model's inputs stay on its training distribution. `emit_canvas` is
    the constrained MAP or joint draw, and is what actually gets emitted. That
    decoupling is what makes J0's guarantee unconditional and independent of
    whether the accept prefix ever covers the canvas (SPEC §3.1).
    """

    step: jnp.ndarray
    canvas: jnp.ndarray
    emit_canvas: jnp.ndarray
    sc_embeddings: jnp.ndarray
    rng: jnp.ndarray
    done: jnp.ndarray
    #: `[B] bool` -- SPEC §6.3's `Z == 0` detector for the constrained draw,
    #: ANDed over every denoising step that actually contributed an emission.
    feasible: jnp.ndarray


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstrainedDiffusionSampler(_diffusion_sampler.DiffusionSampler):
    """Diffusion sampler with a constrained joint emission.

    Attributes:
      n_states_bucket: padded `|S|`. **Static** — array shapes and the tree's
        unrolling depend on it, and it is bucketed to a power of two so XLA
        compiles once per bucket rather than once per grammar (SPEC §5.5).
      n_classes: padded class count. **Static** — `segment_sum`'s
        `num_segments` raises on a traced value.
      variant: `j0` (decoupled trajectory/emission), `j1` (single joint draw
        with flattened marginals at non-accepted), `j2` (SPEC §7.2 baseline:
        the constrained draw kept only at accepted positions, stock uniform
        renoise elsewhere), `mask` (§2.8's naive per-position masking) or
        `unconstrained` (the stock sampler).
      emission: `map` or `sample`.
      constrained_dtype: the sampling path needs float64
        (`constrained.require_x64` explains why in detail); MAP is fine in
        float32 because §2.7 puts it in log space.
    """

    n_states_bucket: int
    n_classes: int
    variant: str = "j0"
    emission: str = "map"
    constrained_dtype: str = "float64"
    #: `mf` = the stock rule, entropy of the **unconstrained** shaped logits.
    #: `mar` = entropy of the **constrained** marginal `q_i` (SPEC §3.4, the
    #: paper's remasking confidence). Static: it changes the traced graph.
    confidence: str = "mf"
    #: Emit a per-step trace to `DIAGNOSTICS`. Static, so turning it on
    #: recompiles — which is fine, it is a debugging path.
    diagnose: bool = False

    def __post_init__(self) -> None:
        if self.variant not in ("j0", "j1", "j2", "mask", "unconstrained"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.emission not in ("map", "sample"):
            raise ValueError(f"unknown emission {self.emission!r}")
        if self.confidence not in ("mf", "mar"):
            raise ValueError(f"unknown confidence {self.confidence!r}")
        # SPEC §3.9's flag table: `--emission={map,sample}` selects "MAP or
        # joint draw (**J0 only**; J1/J2 are always a draw)". J1/J2 define the
        # emission to *be* the draw, so a MAP emission is not one of their
        # variants and is rejected rather than silently reinterpreted.
        if self.variant in ("j1", "j2", "mask") and self.emission == "map":
            raise ValueError(
                f"variant={self.variant!r} is always a draw (SPEC §3.9); "
                "emission='map' is J0-only. With J1's flattened marginals a "
                "joint MAP degenerates to the shortest string in the language."
            )
        if self.emission == "sample":
            # Called from a **production** path, not just documented. Asking
            # for `constrained_dtype='float64'` is not the same as *getting*
            # float64: without `jax_enable_x64` every `jnp.float64` silently
            # becomes float32, `_matrices` builds the tree in fp32, and the root
            # product underflows to exactly zero — a degenerate draw that still
            # returns plausible-looking tokens. Fail at construction instead.
            _constrained.require_x64()
        if self.constrained_dtype not in ("float32", "float64"):
            # float32 became admissible with the pairwise-max `log_matmul`
            # (see `constrained.require_x64`): the old ban existed because the
            # kernel exponentiated against a FOREIGN anchor, so entries
            # underflowed even in float64. Anchored per entry, float32 is
            # measured indistinguishable from float64 against brute-force
            # enumeration, and halves the tree.
            raise ValueError(
                f"constrained_dtype must be float32 or float64, got "
                f"{self.constrained_dtype!r}")

    # -- entry point: widen the prefilled state --------------------------
    @override
    def sample(self, *, params, init_state, max_new_tokens, stream=False):
        """Lift `init_state` before handing it to the stock loop.

        `init_state` is built by `_prefill.prefill` inside
        `gm.text.Sampler.sample`, so widening here avoids forking that method.
        """
        raise NotImplementedError(
            "call sample_constrained(...) — the automaton has to come from the "
            "caller, and the stock signature has nowhere to put it"
        )

    def sample_constrained(self, *, params, init_state, max_new_tokens,
                           automaton: Automaton, stream: bool = False):
        state = widen(init_state, automaton=automaton,
                      max_new_tokens=jnp.asarray(max_new_tokens),
                      cache_length=self.cache_length)
        out = _sampler_loop.SamplerLoop.sample(
            self, params=params, init_state=state,
            max_new_tokens=jnp.asarray(max_new_tokens), stream=stream,
        )
        if not stream:
            _check_feasible(out)
        return out

    # -- per block --------------------------------------------------------
    @functools.partial(jax.jit, static_argnames=("self",))
    @override
    def _sample_step(self, state, *, params):
        """One block. Forked to thread `A_k` and `R` into the canvas sampler
        and to advance `A_{k+1}` across the boundary (SPEC §3.5)."""
        next_rng, sample_rng = jax.random.split(state.rng)
        cache = state.cache
        batch_size = list(cache.values())[0]["end_index"].shape[0]

        canvas, canvas_feasible = self.sample_next_canvas_constrained(
            canvas_length=self.canvas_length,
            max_denoising_steps=self.max_denoising_steps,
            batch_size=batch_size,
            cache=cache,
            params=params,
            rng=sample_rng,
            full_attention_mask=state.full_attention_mask,
            automaton=state.automaton,
            # Budget left AFTER this canvas -- see state.terminal_budget.
            remaining=state.terminal_budget(self.canvas_length),
        )

        canvas, batch_has_stop_token = _diffusion_sampler._truncate_canvas_at_stop_tokens(  # noqa: SLF001
            canvas, end_tokens=self.end_tokens,
            canvas_length=self.canvas_length, done=state.done,
        )

        # A_{k+1} = delta*(A_k, TRUNCATED canvas) -- from the truncated canvas,
        # not from a MAP backtrace (SPEC §3.5 trap 2).
        active, advance_ok = jax.vmap(
            lambda act, toks: _constrained.advance_states(
                state.automaton.with_active(act), toks,
                self.n_states_bucket, self.n_classes, self.text_vocab_size)
        )(state.automaton.active, canvas)

        cache = self.append_tokens_to_cache(tokens=canvas, cache=cache,
                                            params=params)
        done = state.done | batch_has_stop_token
        # int32 explicitly: under jax_enable_x64 `arange` is int64 and the
        # scatter into an int32 buffer warns (and will become an error).
        indices = (jnp.arange(self.canvas_length, dtype=jnp.int32)
                   + state.step.astype(jnp.int32))
        predicted_tokens = state.predicted_tokens.at[:, indices].set(
            canvas.astype(state.predicted_tokens.dtype))

        return ConstrainedSamplingState(
            step=state.step + self.canvas_length,
            done=done,
            last_token=canvas[:, -1],
            last_token_pos=state.last_token_pos + self.canvas_length,
            predicted_tokens=predicted_tokens,
            cache=cache,
            rng=next_rng,
            init_cache_length=state.init_cache_length,
            full_attention_mask=state.full_attention_mask,
            automaton=state.automaton.with_active(active),
            max_new_tokens=state.max_new_tokens,
            cache_length=state.cache_length,
            # Sticky AND: once a block was infeasible the whole generation is
            # suspect, and `advance_ok` catches the state set emptying at a
            # block boundary -- which used to be papered over by a stale-carry
            # fallback inside `advance_states`.
            #
            # `advance_ok` is folded in only for the variants that actually
            # promise the guarantee. `j2`, `mask` and `unconstrained` emit
            # tokens the automaton rejects **by construction** -- that is what
            # they are measuring -- so an empty state set there is the result,
            # not a bug, and raising on it would make the baselines unrunnable.
            feasible=(state.feasible & canvas_feasible & advance_ok
                      if self.variant in ("j0", "j1")
                      else state.feasible & canvas_feasible),
        )

    # -- the denoising loop ----------------------------------------------
    def sample_next_canvas_constrained(
        self, *, canvas_length, max_denoising_steps, batch_size, cache, params,
        rng, full_attention_mask, automaton, remaining,
    ):
        """The stock denoising loop with a constrained emission.

        Under **J1** the trajectory and the emission are the same tensor, so the
        stock `_WhileLoopCarry` suffices and no carry widening is needed — which
        is why SPEC §5.4 says to build J1 first.
        """
        initial_canvas_rng, step_rng = jax.random.split(rng)
        cache_layer = list(cache.values())[0]
        cache_len = cache_layer["k"].shape[1]
        samples_in_cache = cache_layer["end_index"]
        positions = samples_in_cache[:, None] + jnp.arange(canvas_length)[None, :]

        attention_mask = _diffusion_sampler._make_global_attention_mask(  # noqa: SLF001
            batch_size=batch_size, canvas_length=canvas_length,
            cache_length=cache_len, num_valid_tokens=samples_in_cache,
            full_attention_mask=full_attention_mask)
        block_local = _diffusion_sampler._make_block_local_attention_mask(  # noqa: SLF001
            batch_size=batch_size, canvas_length=canvas_length,
            sliding_window_size=self.sliding_window_size,
            cache_length=cache_len, num_valid_tokens=samples_in_cache)

        initial = self.diffusion_process.get_initial_sample(
            rng=initial_canvas_rng, batch_size=batch_size,
            canvas_length=canvas_length, text_vocab_size=self.text_vocab_size)

        noise_proportions = 1.0 - jnp.arange(max_denoising_steps + 1) / max_denoising_steps
        embed_dim = self.model.config.embed_dim
        dt = jnp.float64 if self.constrained_dtype == "float64" else jnp.float32

        def cond_fn(carry):
            return jnp.logical_and(~jnp.all(carry.done),
                                   carry.step < max_denoising_steps)

        def body_fn(carry):
            step = carry.step
            next_rng_, sample_rng_ = jax.random.split(carry.rng)
            cur_np = jnp.full((batch_size,), noise_proportions[step])
            tgt_np = jnp.full((batch_size,), noise_proportions[step + 1])

            out = self.sample_step(
                canvas=carry.canvas, sc_embeddings=carry.sc_embeddings,
                cache=cache, positions=positions, attention_mask=attention_mask,
                sliding_attention_mask=block_local,
                current_noise_proportion=cur_np, target_noise_proportion=tgt_np,
                params=params, rng=sample_rng_)

            p_real = jax.nn.softmax(out.logits.astype(dt), axis=-1)  # [B, L, V]

            if self.confidence == "mar":
                # SPEC §3.4 / the paper's remasking confidence. `r_i(v)` is the
                # same quantity the `mask` variant already builds; `q = p*r/Z`
                # and its entropy then replace the unconstrained entropy in the
                # accept rule. One forward-backward per step -- the cost the
                # `mask` arm already pays and which measured at 44.4 s/record
                # against 37.4 unconstrained.
                def _q_entropy(pi, act):
                    aut = automaton.with_active(act)
                    p_vl, W_e, M = _constrained._matrices(  # noqa: SLF001
                        pi, aut, self.n_states_bucket, self.n_classes)
                    tr = scans.up_sweep(M)
                    a_v, b_v, _, _ = scans.prefix_suffix(
                        tr, aut.active.astype(pi.dtype),
                        _constrained.budget_terminal_factor(
                            aut.d, remaining, dtype=pi.dtype))
                    u = (a_v[:-1][:, aut.edge_src]
                         * b_v[1:][:, aut.edge_dst])             # [L, E]
                    # Streamed: `H(q)` from the CSR without building `q`,
                    # `row`, `lq` and `q*lq` at [L, V] = [256, 262144]. The
                    # dense chain deadlocked on three records (CPU frozen at
                    # 5:21 over 39 minutes). Verified equal to the dense form
                    # to 8.9e-16.
                    return _marginals.constrained_entropy_streamed(
                        p_vl, u, aut.edge_class, aut.csr_indices,
                        aut.csr_indptr, aut.is_neg, self.n_classes,
                        pi.shape[-1])                            # [L]
                h_q = jax.vmap(_q_entropy)(p_real, automaton.active)
                accepted = _accept_from_entropy(
                    h_q.astype(jnp.float32),
                    self.sample_from_predictions.entropy_bound)
            else:
                accepted = _accept_mask(
                    out.logits, self.sample_from_predictions.entropy_bound)

            # The EMISSION always uses the model's real marginals. Only J1's
            # *trajectory* is flattened — that is the whole point of J0's
            # decoupling (SPEC §3.1), and it is why J0's guarantee is
            # unconditional and independent of whether the accept prefix ever
            # covers the canvas.
            p = p_real
            if self.variant == "j1":
                p = jax.vmap(_constrained.flatten_unaccepted)(p_real, accepted)

            if self.variant == "unconstrained":
                # SPEC §7.2 baseline 1: the stock sampler, untouched.
                sampled = self.sample_from_predictions(
                    rng=sample_rng_, denoiser_logits=out.logits,
                    canvas=carry.canvas, current_noise_proportion=cur_np,
                    target_noise_proportion=tgt_np)
                new_done = jnp.logical_or(
                    carry.done,
                    self.early_stop_fn.should_stop(
                        step=step, canvas=sampled,
                        previous_canvas=carry.canvas, logits=out.logits))
                canvas = jnp.where(carry.done[:, None], carry.canvas, sampled)
                return _ConstrainedCarry(
                    step=step + 1, canvas=canvas, emit_canvas=canvas,
                    sc_embeddings=out.sc_embeddings.astype(
                        carry.sc_embeddings.dtype),
                    rng=next_rng_, done=new_done, feasible=carry.feasible)

            if self.variant == "mask":
                # SPEC §7.2 baseline 2 and §2.8's target: **naive per-position
                # masking**. Mask to the support projection pi_i(C) and sample
                # each position INDEPENDENTLY.
                #
                # This enforces the product of coordinate projections, which
                # strictly contains C whenever C is not a product set — the
                # paper's own example being that a grammar accepting real
                # numbers admits "1." and ".1" and a factorized sampler then
                # emits "..". Its CS column is the headline evidence that the
                # joint method is needed at all.
                def support_of(pi, act):
                    aut = automaton.with_active(act)
                    p_vl, W_e, M = _constrained._matrices(  # noqa: SLF001
                        pi, aut, self.n_states_bucket, self.n_classes)
                    tr = scans.up_sweep(M)
                    a_v, b_v, _, _ = scans.prefix_suffix(
                        tr, aut.active.astype(pi.dtype),
                        _constrained.budget_terminal_factor(
                            aut.d, remaining, dtype=pi.dtype))
                    u = a_v[:-1][:, aut.edge_src] * b_v[1:][:, aut.edge_dst]
                    return _marginals.scatter_edge_mass_to_tokens(
                        u, aut.edge_class, aut.csr_indices, aut.csr_indptr,
                        aut.is_neg, self.n_classes, pi.shape[-1])

                r = jax.vmap(support_of)(p_real, automaton.active)   # [B, L, V]
                masked = jnp.where(r > 0, out.logits.astype(jnp.float32),
                                   _marginals.MASK_SENTINEL)
                sampled = jax.random.categorical(sample_rng_, masked)
                new_done = jnp.logical_or(
                    carry.done,
                    self.early_stop_fn.should_stop(
                        step=step, canvas=sampled,
                        previous_canvas=carry.canvas, logits=masked))
                canvas = jnp.where(carry.done[:, None], carry.canvas, sampled)
                return _ConstrainedCarry(
                    step=step + 1, canvas=canvas, emit_canvas=canvas,
                    sc_embeddings=out.sc_embeddings.astype(
                        carry.sc_embeddings.dtype),
                    # Deliberately NOT flagged. An empty per-position support is
                    # this baseline's *result*, not an error -- SPEC §2.8 exists
                    # because factorized masking gets the joint wrong.
                    rng=next_rng_, done=new_done, feasible=carry.feasible)

            def per_example(pi, act, key):
                aut = automaton.with_active(act)
                if self.emission == "map":
                    return _constrained.joint_map(
                        pi, aut, remaining, self.n_states_bucket, self.n_classes)
                return _constrained.joint_draw(
                    pi, aut, remaining, key, self.n_states_bucket,
                    self.n_classes)

            keys = jax.random.split(sample_rng_, batch_size)
            emitted, ok = jax.vmap(per_example)(p, automaton.active, keys)

            if self.variant == "j2":
                # SPEC §7.2 baseline 3, and until now NOT implemented: `j2` ran
                # the same fully-constrained emission as J0 and therefore
                # measured J0 twice under two names.
                #
                # J2 keeps the constrained draw only at **accepted** positions
                # and leaves the rest as the stock uniform renoise over the full
                # 262k vocab. That is precisely SPEC §3.1's finding — the
                # emitted canvas is the sample, and non-accepted positions reach
                # the output as random tokens — so its CS column is the evidence
                # that constraining the sampler alone is necessary but **not
                # sufficient**, and that the guarantee needs the emission itself
                # to be a constrained object.
                emitted = jnp.where(
                    accepted, emitted,
                    jax.random.randint(jax.random.fold_in(sample_rng_, 3),
                                       emitted.shape, minval=0,
                                       maxval=self.text_vocab_size))
                # The joint draw was still feasible; J2 corrupts it deliberately
                # afterwards, so `ok` stays the Z == 0 signal and does not become
                # a report on the baseline's (expected) constraint violations.

            if self.diagnose:
                # Per-step visibility inside the jitted `while_loop`. There is
                # no Python between denoising steps (SPEC §5.3), so a callback
                # is the only way to see the emission converge — or not.
                lp = jax.nn.log_softmax(out.logits.astype(jnp.float32))
                pr = jnp.exp(lp)
                h = -jnp.sum(jnp.where(pr == 0, 0.0, lp) * pr, axis=-1)
                jax.debug.callback(
                    _DIAG.record, step, jnp.sum(accepted[0]), jnp.mean(h[0]),
                    emitted[0], carry.canvas[0])

            if self.variant == "j0":
                # J0: the TRAJECTORY keeps stock uniform renoising, so the
                # model's inputs stay on its training distribution, while the
                # emission is constrained. Cost: the carry gains an
                # `emit_canvas` field (SPEC §5.4) -- here the two are tracked
                # as `canvas` (trajectory) and the returned `emitted`.
                denoiser = jax.random.categorical(
                    jax.random.fold_in(sample_rng_, 1),
                    out.logits.astype(jnp.float32))
                noise = jax.random.randint(
                    jax.random.fold_in(sample_rng_, 2), carry.canvas.shape,
                    minval=0, maxval=self.text_vocab_size)
                trajectory = jnp.where(accepted, denoiser, noise)
            else:
                trajectory = emitted

            new_done = jnp.logical_or(
                carry.done,
                self.early_stop_fn.should_stop(
                    step=step, canvas=emitted, previous_canvas=carry.canvas,
                    logits=out.logits))
            canvas = jnp.where(carry.done[:, None], carry.canvas, trajectory)
            emit = jnp.where(carry.done[:, None], carry.emit_canvas, emitted)

            return _ConstrainedCarry(
                step=step + 1, canvas=canvas, emit_canvas=emit,
                sc_embeddings=out.sc_embeddings.astype(carry.sc_embeddings.dtype),
                rng=next_rng_, done=new_done,
                # `| carry.done` for the same reason `emit` is gated on it: a
                # finished example's emission is frozen, so a later step's
                # infeasibility never reaches the output.
                feasible=carry.feasible & (ok | carry.done))

        init_carry = _ConstrainedCarry(
            step=jnp.int32(0), canvas=initial, emit_canvas=initial,
            sc_embeddings=jnp.zeros((batch_size, canvas_length, embed_dim),
                                    dtype=jnp.bfloat16),
            rng=step_rng, done=jnp.zeros(batch_size, dtype=jnp.bool_),
            feasible=jnp.ones(batch_size, dtype=jnp.bool_))

        final = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        # `sample_next_canvas` returns the EMITTED canvas; `_sample_step`
        # truncates that and writes it to the cache and `predicted_tokens`.
        return final.emit_canvas, final.feasible


def _check_feasible(state) -> None:
    """Raise SPEC §6.3's `Z == 0` outside the jit.

    The block loop is a `lax.while_loop` under `jit` with a traced
    `max_new_tokens`, so there is no Python between blocks (SPEC §5.3) and this
    cannot be raised where it is detected. It rides `ConstrainedSamplingState`
    out instead.

    Deliberately NOT caught anywhere. CLAUDE.md: `Z == 0` has three causes and
    only (c), fp32 underflow, is benign -- and the constrained paths run in log
    space precisely so (c) cannot occur. What is left is (a) an empty automaton
    or (b) no live continuation within budget, and both are bugs that must be
    seen, not smoothed over.
    """
    feasible = getattr(state, "feasible", None)
    if feasible is None:
        return
    bad = np.flatnonzero(~np.asarray(feasible))
    if bad.size:
        raise ZeroPartitionError(
            f"Z == 0 for batch element(s) {bad.tolist()}: no accepted string of "
            f"the canvas length exists from A_k within the remaining budget "
            f"(SPEC §3.1b / §6.3 cause (a) or (b)). The emitted canvas for "
            f"those elements is meaningless -- an all-sentinel root makes "
            f"`categorical` and `argmax` return a confident-looking draw."
        )


class ZeroPartitionError(RuntimeError):
    """`Z == 0`: the constrained posterior has no support. SPEC §6.3."""


def _accept_mask(logits: jnp.ndarray, entropy_bound: float) -> jnp.ndarray:
    """The stock accept rule, recomputed rather than reached into.

    Ascending-entropy `argsort`, `cumsum − sorted ≤ entropy_bound`; always
    accepts ≥1. Entropy in **nats**, from `log_softmax` on the **shaped**
    logits, with the stock code's own `0·log 0` guard.
    """
    lp = jax.nn.log_softmax(logits.astype(jnp.float32))
    p = jnp.exp(lp)
    h = -jnp.sum(jnp.where(p == 0, 0.0, lp) * p, axis=-1)      # [B, L]
    return _accept_from_entropy(h, entropy_bound)


def _accept_from_entropy(h: jnp.ndarray, entropy_bound: float) -> jnp.ndarray:
    """The accept rule's *selection*, given per-position entropy `[B, L]`.

    Split out so `--confidence=mar` can feed it the entropy of the
    **constrained** marginal `q_i` instead of the unconstrained `p_i`. The
    paper's own ablation puts that swap at 68.4 -> 76.4, the larger half of its
    accuracy gain; here it was built (`infer/marginals.py`) and then never
    called on the production path.
    """
    order = jnp.argsort(h, axis=-1)
    srt = jnp.take_along_axis(h, order, axis=-1)
    keep = (jnp.cumsum(srt, axis=-1) - srt) <= entropy_bound
    b = jnp.arange(h.shape[0])[:, None]
    return jnp.zeros_like(order, dtype=jnp.bool_).at[b, order].set(keep)
