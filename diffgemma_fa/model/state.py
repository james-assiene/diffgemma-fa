"""Widened `SamplingState` carrying the automaton. SPEC §5.3.

**Why the automaton lives here and not on the sampler object.** `self` is a
`static_argname` on both `_sample_loop` and `_sample_step`, so the whole sampler
— including any custom `sample_from_predictions` or `logit_shaper` — is hashed
as a compile-time constant. Per-request automaton arrays held as attributes
would trigger a **full recompile per distinct grammar**: at 4,549 BFCL-Live
schemas that is hours of pure compilation, and it defeats the compilation cache
entirely. **Automaton values must be traced; only shapes may be static.**

`SamplingState` is a `flax.struct.dataclass(kw_only=True)`, i.e. pytree-
registered, so adding fixed-shape fields is mechanical and they thread through
`lax.while_loop` for free.

Two things Phase 0 established that force this design:

- `sample_next_canvas` **never receives `state`** — its signature is
  `(canvas_length, max_denoising_steps, batch_size, cache, params, rng,
  full_attention_mask)`. Neither `A_k` nor `predicted_tokens` is reachable from
  the one place the constrained sampler has to live.
- `max_new_tokens` is **not a field of `SamplingState`** at all; it is a
  parameter of `_sample_loop`, captured only in that function's `cond_fn`
  closure. So `R = max_new_tokens − step` is not computable in `_sample_step`
  either, and must be carried explicitly.
"""

from __future__ import annotations

import dataclasses

import flax.struct
import jax.numpy as jnp
from gemma.gm.text import _sampler_loop

__all__ = ["Automaton", "ConstrainedSamplingState", "widen"]


@flax.struct.dataclass
class Automaton:
    """The compiled automaton, as traced arrays with bucketed shapes.

    Every field is a `jnp` array so the whole object is one pytree leaf-set that
    threads through `lax.while_loop`. Shapes are padded to buckets; **sizes**
    (`n_states_bucket`, `n_classes_bucket`) live on the sampler as Python ints
    because `segment_sum`'s `num_segments` and array shapes must be static.

    Shapes:
      edge_src, edge_dst, edge_class  [E_bucket] int32
      edge_valid                      [E_bucket] bool   padding mask
      csr_indices                     [nnz_bucket] int32
      csr_indptr                      [C_bucket + 1] int32
      is_neg                          [C_bucket] bool
      d                               [S_bucket] int32   min tokens to F
      is_final                        [S_bucket] bool
      active                          [B, S_bucket] bool  A_k — a VECTOR (§5.7)
    """

    edge_src: jnp.ndarray
    edge_dst: jnp.ndarray
    edge_class: jnp.ndarray
    edge_valid: jnp.ndarray
    csr_indices: jnp.ndarray
    csr_indptr: jnp.ndarray
    is_neg: jnp.ndarray
    d: jnp.ndarray
    is_final: jnp.ndarray
    active: jnp.ndarray

    @property
    def n_states(self) -> int:
        return int(self.d.shape[-1])

    def with_active(self, active: jnp.ndarray) -> "Automaton":
        return dataclasses.replace(self, active=active)


@flax.struct.dataclass(kw_only=True)
class ConstrainedSamplingState(_sampler_loop.SamplingState):
    """`SamplingState` plus the automaton and the remaining-budget inputs.

    `cache_info` is inherited as a **property**, not a field (Phase 0 resolved
    this), so nothing extra has to be supplied for it.
    """

    automaton: Automaton
    #: Traced, because `max_new_tokens` is traced in the stock loop and a static
    #: copy here would recompile per requested length.
    max_new_tokens: jnp.ndarray
    #: Cache capacity, so the budget can account for SPEC §3.1b's **fourth**
    #: termination path — `_sample_loop`'s `cond_fn` also exits on
    #: `state.cache_info.is_full`, which truncates mid-grammar exactly as the
    #: token budget does.
    cache_length: jnp.ndarray

    @property
    def remaining_budget(self) -> jnp.ndarray:
        """Tokens still available **including** the block about to be emitted.

        Bounded by **both** the token budget and cache capacity — SPEC §3.1b's
        fourth termination path is cache exhaustion, which truncates
        mid-grammar exactly as the token budget does.
        """
        by_tokens = self.max_new_tokens - self.step
        by_cache = self.cache_length - (self.init_cache_length + self.step)
        return jnp.minimum(by_tokens, by_cache)

    def terminal_budget(self, canvas_length: int) -> jnp.ndarray:
        """`R` for `b_L(s) = 1[d(s) ≤ R]`: the budget left **after** this canvas.

        SPEC §3.5 writes `R ← max_new_tokens − state.step` and applies
        `1[d(s) ≤ R]` as the terminal factor, but `state.step` counts tokens
        committed *before* the block while `b_L` is evaluated at the state
        reached *after* its `L` tokens. Using `R` unadjusted is therefore too
        permissive by exactly `canvas_length`: it admits states that cannot
        actually finish, and generation then fails to terminate because nothing
        ever forces completion. `tests/test_guarantee.py` catches this.
        """
        return self.remaining_budget - canvas_length


def widen(
    state: _sampler_loop.SamplingState,
    *,
    automaton: Automaton,
    max_new_tokens: jnp.ndarray,
    cache_length: int,
) -> ConstrainedSamplingState:
    """Lift a prefilled `SamplingState` into the constrained one.

    Used at the entry point: `init_state` is built by `_prefill.prefill` inside
    `gm.text.Sampler.sample`, so the widening happens in the sampler's own
    `sample()` override rather than by forking that method.
    """
    fields = {f.name: getattr(state, f.name)
              for f in dataclasses.fields(_sampler_loop.SamplingState)}
    return ConstrainedSamplingState(
        **fields,
        automaton=automaton,
        max_new_tokens=jnp.asarray(max_new_tokens),
        cache_length=jnp.asarray(cache_length),
    )
