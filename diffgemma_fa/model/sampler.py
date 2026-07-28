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

import jax
import jax.numpy as jnp
from gemma.diffusion import _sampler as _diffusion_sampler
from gemma.gm.text import _sampler_loop

from diffgemma_fa.model import constrained as _constrained
from diffgemma_fa.model.state import Automaton, ConstrainedSamplingState, widen

__all__ = ["ConstrainedDiffusionSampler"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstrainedDiffusionSampler(_diffusion_sampler.DiffusionSampler):
    """Diffusion sampler with a constrained joint emission.

    Attributes:
      n_states_bucket: padded `|S|`. **Static** — array shapes and the tree's
        unrolling depend on it, and it is bucketed to a power of two so XLA
        compiles once per bucket rather than once per grammar (SPEC §5.5).
      n_classes: padded class count. **Static** — `segment_sum`'s
        `num_segments` raises on a traced value.
      variant: `j1` (single joint draw, flattened marginals at non-accepted) or
        `j2` (baseline: constrained draw at accepted, uniform random elsewhere).
      emission: `map` or `sample`.
      constrained_dtype: the sampling path needs float64
        (`constrained.require_x64` explains why in detail); MAP is fine in
        float32 because §2.7 puts it in log space.
    """

    n_states_bucket: int
    n_classes: int
    variant: str = "j1"
    emission: str = "map"
    constrained_dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.variant not in ("j1", "j2"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.emission not in ("map", "sample"):
            raise ValueError(f"unknown emission {self.emission!r}")
        if self.emission == "sample" and self.constrained_dtype != "float64":
            raise ValueError(
                "emission='sample' requires constrained_dtype='float64'; see "
                "model.constrained.require_x64 — in fp32 the root product "
                "underflows to exactly zero and the draw degenerates silently"
            )

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
        return _sampler_loop.SamplerLoop.sample(
            self, params=params, init_state=state,
            max_new_tokens=jnp.asarray(max_new_tokens), stream=stream,
        )

    # -- per block --------------------------------------------------------
    @functools.partial(jax.jit, static_argnames=("self",))
    @override
    def _sample_step(self, state, *, params):
        """One block. Forked to thread `A_k` and `R` into the canvas sampler
        and to advance `A_{k+1}` across the boundary (SPEC §3.5)."""
        next_rng, sample_rng = jax.random.split(state.rng)
        cache = state.cache
        batch_size = list(cache.values())[0]["end_index"].shape[0]

        canvas = self.sample_next_canvas_constrained(
            canvas_length=self.canvas_length,
            max_denoising_steps=self.max_denoising_steps,
            batch_size=batch_size,
            cache=cache,
            params=params,
            rng=sample_rng,
            full_attention_mask=state.full_attention_mask,
            automaton=state.automaton,
            remaining=state.remaining_budget,
        )

        canvas, batch_has_stop_token = _diffusion_sampler._truncate_canvas_at_stop_tokens(  # noqa: SLF001
            canvas, end_tokens=self.end_tokens,
            canvas_length=self.canvas_length, done=state.done,
        )

        # A_{k+1} = delta*(A_k, TRUNCATED canvas) -- from the truncated canvas,
        # not from a MAP backtrace (SPEC §3.5 trap 2).
        active = jax.vmap(
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

            p = jax.nn.softmax(out.logits.astype(dt), axis=-1)   # [B, L, V]

            if self.variant == "j1":
                # J1: flatten the marginals wherever the stock accept rule
                # would have renoised, then draw ONE constrained joint sample.
                accepted = _accept_mask(out.logits,
                                        self.sample_from_predictions.entropy_bound)
                p = jax.vmap(_constrained.flatten_unaccepted)(p, accepted)

            def per_example(pi, act, key):
                aut = automaton.with_active(act)
                if self.emission == "map":
                    return _constrained.joint_map(
                        pi, aut, remaining, self.n_states_bucket, self.n_classes)
                toks, _valid = _constrained.joint_draw(
                    pi, aut, remaining, key, self.n_states_bucket,
                    self.n_classes)
                return toks

            keys = jax.random.split(sample_rng_, batch_size)
            sampled = jax.vmap(per_example)(p, automaton.active, keys)

            new_done = jnp.logical_or(
                carry.done,
                self.early_stop_fn.should_stop(
                    step=step, canvas=sampled, previous_canvas=carry.canvas,
                    logits=out.logits))
            canvas = jnp.where(carry.done[:, None], carry.canvas, sampled)

            return _diffusion_sampler._WhileLoopCarry(  # noqa: SLF001
                step=step + 1, canvas=canvas,
                sc_embeddings=out.sc_embeddings.astype(carry.sc_embeddings.dtype),
                rng=next_rng_, done=new_done)

        init_carry = _diffusion_sampler._WhileLoopCarry(  # noqa: SLF001
            step=jnp.int32(0), canvas=initial,
            sc_embeddings=jnp.zeros((batch_size, canvas_length, embed_dim),
                                    dtype=jnp.bfloat16),
            rng=step_rng, done=jnp.zeros(batch_size, dtype=jnp.bool_))

        return jax.lax.while_loop(cond_fn, body_fn, init_carry).canvas


def _accept_mask(logits: jnp.ndarray, entropy_bound: float) -> jnp.ndarray:
    """The stock accept rule, recomputed rather than reached into.

    Ascending-entropy `argsort`, `cumsum − sorted ≤ entropy_bound`; always
    accepts ≥1. Entropy in **nats**, from `log_softmax` on the **shaped**
    logits, with the stock code's own `0·log 0` guard.
    """
    lp = jax.nn.log_softmax(logits.astype(jnp.float32))
    p = jnp.exp(lp)
    h = -jnp.sum(jnp.where(p == 0, 0.0, lp) * p, axis=-1)      # [B, L]
    order = jnp.argsort(h, axis=-1)
    srt = jnp.take_along_axis(h, order, axis=-1)
    keep = (jnp.cumsum(srt, axis=-1) - srt) <= entropy_bound
    b = jnp.arange(h.shape[0])[:, None]
    return jnp.zeros_like(order, dtype=jnp.bool_).at[b, order].set(keep)
