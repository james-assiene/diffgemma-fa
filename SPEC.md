# Constrained Decoding for DiffusionGemma via Finite Automata — JAX

**Implementation specification for Claude Code.**

Target paper: Meihua Dang & Stefano Ermon, *Constrained Decoding for Diffusion Language Models
via Efficient Inference over Finite Automata*, [arXiv:2607.07026](https://arxiv.org/abs/2607.07026).

Target implementation: **`google-deepmind/gemma`, subpackage `gemma/diffusion/`** — the official
JAX/Flax DiffusionGemma. Checkpoint: `gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it`
(Orbax OCDBT, **37.63 GiB = 40.4 GB on disk**, publicly readable). [V, measured Phase 0]

Status: **Phase 0 complete (2026-07-27).** Originally written from source reading over the web plus
numerical verification of §2's formulas on CPU; §1.2/§1.3 have now been verified against the
installed `gemma` 4.1.0 and the real checkpoint on an H100 80GB. Corrections are marked
**[V-P0]** and collected in `docs/PHASE0_FINDINGS.md`. Markers:

- **[V]** — read verbatim from source, or verified numerically in this document's own test harness.
- **[R]** — reported by docs/blog. Likely right, verify cheaply.
- **[D]** — a design decision made here. Argued, not asserted.
- **[?]** — unknown. Resolve empirically.

---

## Hard invariant: inference-only, sampler-only

**No training. No fine-tuning. No weight modification. No backward pass.** This is a *sampler*
modification against the released checkpoint.

The paper's algorithm consumes exactly one thing from the model: the per-position mean-field
marginals `p_θ(x_i⁰ | xᵗ)`, i.e. the `[B, 256, V]` logits array already produced at every denoising
step. It never needs gradients and is agnostic to how the model was trained.

**Minimum fork surface: `DiffusionSampler.sample_next_canvas`, plus a widened `SamplingState`.**
The Flax module and the params are untouched. §5.3 derives this from the library's actual loop
structure; it is *not* what an early reading suggests, so read §5.3 before designing anything.

Two places an implementer might drift and must not: §3.3's renoising variants and §3.7's
self-conditioning fork shift the model's *inputs* off its training distribution. They are
inference-time flags to be ablated and possibly rejected, never a reason to train. And §7.5 —
reasoning-bound tasks will show near-zero gains; that is the expected result, not something to
train away.

---

## 0. Why JAX is the right target, not a compromise

**The paper's core contribution is a stock JAX primitive.** The `O(L)` → `O(log L)` depth reduction
is an associative scan over transition matrices. `jax.lax.associative_scan` is stable, public,
works eager and under `jit`, and accepts an arbitrary associative combine over pytrees — verified
here for both `jnp.matmul` (sum-product) and max-plus. PyTorch's `torch.associative_scan` is
documented as *"a prototype feature… does not support autograd and you may run into miscompiles"*,
its fast `combine_mode='pointwise'` **forbids matmul**, and it requires `torch.compile` (no eager).
[V]

**The paper's performance story is a launch-overhead story, and XLA fixes it by default.** Table 3
reports +114% wall-clock for the sequential chain versus +4% for the tree. Compiling both and
reading the optimized HLO shows why:

| | HLO lines | fusions | dot/custom-call | **while loops** |
|---|---|---|---|---|
| `lax.scan` (sequential chain) | 110 | 6 | 1 | **1** |
| `associative_scan` (tree) | 471 | 59 | 15 | **0** |

XLA:GPU command buffers *are* CUDA graphs, and `xla_gpu_enable_command_buffer` defaults to
`{FUSION, CUBLAS, CUBLASLT, CUDNN, CUSTOM_CALL, CONDITIONAL, DYNAMIC_SLICE_FUSION}` — **`WHILE` is
absent**. So the sequential form becomes a device while-loop that escapes graph capture and pays
256 per-kernel launches; the tree becomes unrolled straight-line code whose every op type is in the
default capture set. [V, `xla/debug_options_flags.cc`]

**The extension surface is better.** `Sampler`/`ChatSampler` take `logit_shaper`,
`sample_from_predictions` and `early_stop_fn` as constructor arguments [V].

**Measured costs, so you know what you're signing up for.** `associative_scan` at `L=256` invokes
the combine **16 times** (12 at `L=64`, 8 at `L=16`), i.e. `2·log₂L` invocations, with **502 total
elementwise combines ≈ 2L** — work-efficient. An unrolled 8-level tree over `[256,S,S]` compiles in
**~5 s at `S=1024`**, and the HLO is *identical in size* across `S`, so compile cost scales with the
number of shape buckets, not with `S`.

**But do not over-read the "+4%".** That is a *small-automaton* result. The tree's own arithmetic
is `≈2L · 2|S|³` FLOPs per denoising step. Against a 4B-active-parameter forward over 256 tokens
(`2·4e9·256 = 2.05e12` FLOPs), measured:

| `\|S\|` | tree FLOPs/step | as % of one model forward |
|---|---|---|
| 385 | 5.8e10 | **2.9%** |
| 512 | 1.4e11 | 6.7% |
| 1024 | 1.1e12 | **53.7%** |
| 1976 | 7.9e12 | 386% |
| 2459 | 1.5e13 | 743% |

So you are **launch-bound below `|S| ≈ 512` and compute-bound above it**. §0's argument and the
paper's +4% apply to the former. Report the crossover you measure. [V]

---

## 1. Phase 0 — environment and fact verification

**Do this first. Write no implementation code until `docs/PHASE0_FINDINGS.md` exists.** Correct
this spec in place where it is wrong.

### 1.1 Install and load

```bash
# PyPI is 4.0.1 (2026-05-20), BEFORE DiffusionGemma shipped. gemma/diffusion/ is NOT in it. [V]
pip install "git+https://github.com/google-deepmind/gemma.git"   # declares __version__ 4.1.0
python -c "import jax; print(jax.__version__, jax.devices())"
gsutil -m cp -r gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it .   # 37.6 GiB, anonymous [V]
```

```python
from gemma import gm, diffusion            # NOT gm.diffusion — top-level sibling package [V]
model   = diffusion.DiffusionGemma_26B_A4B()          # = Gemma4_26B_A4B + DiffusionMixin [V]
params  = gm.ckpts.load_params(diffusion.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT)
sampler = diffusion.ChatSampler(model=model, params=params)
```

`load_params` also accepts a **local directory path** — no need to re-read from `gs://` once the
checkpoint is downloaded. [V-P0]

Python ≥3.12. The subpackage is **entirely undocumented** — no `gm.diffusion` namespace, nothing on
readthedocs, not in the README. **Source is the only spec.** [V]

Record in `docs/ENV.md`: accelerator and memory, JAX/jaxlib versions, `nproc`, disk.

> **[V-P0] Measured memory, which is what the §5.6 budget must be built on.** The checkpoint is
> 37.63 GiB on disk but **51.65 GB resident in HBM** once loaded, and generation peaks at
> **56.5 GB**. On an 80 GB H100 at `XLA_PYTHON_CLIENT_MEM_FRACTION=.90` (limit 76.52 GB) that
> leaves **~20 GB** free for the tree, not the 8 GB §5.6 assumes. Do not size the `|S|` ladder
> from the on-disk figure. Below ~64 GB HBM you are fighting for room and should shrink the `|S|`
> budget (§5.6).

Set on day one:

```python
jax.config.update("jax_compilation_cache_dir", "~/.cache/jax")   # highest-leverage single line
```

### 1.2 Verify the sampler source

Read `gemma/diffusion/{_sampler,_early_stopping,_transformer,_chat_sampler}.py` and
`gemma/gm/text/_sampler_loop.py` locally and tick each box.

> **Phase 0 status: all boxes verified against `gemma` 4.1.0 (git main, installed 2026-07-27).**
> Everything below is confirmed **[V-P0]** unless flagged **CORRECTED**. `main` moves — re-verify
> if you upgrade `gemma`. Full evidence in `docs/PHASE0_FINDINGS.md`.

- [x] `SampleFromPredictions.__call__(*, rng, denoiser_logits, canvas, current_noise_proportion,
      target_noise_proportion) -> Tokens` — **all keyword-only**, returns a bare `[B, L]` token
      array, not a `(canvas, mask)` tuple. [V-P0]
- [x] It returns `jnp.where(selection_mask, denoiser_tokens, random_tokens)` with
      `denoiser_tokens = jax.random.categorical(...)` and
      `random_tokens = jax.random.randint(0, text_vocab_size)`. [V-P0]
- [x] `selection_mask` is built from `jnp.zeros_like(...)` **every call** — non-monotone, nothing
      mask-shaped in the carry. [V-P0] (It is `zeros_like(sorted_index, dtype=bool)`, scattered
      through `.at[arange(B)[:,None], sorted_index].set(sorted_selection_mask)`.)
- [x] Accept predicate: ascending `argsort` of entropy, `cumsum − sorted ≤ entropy_bound`. Default
      `entropy_bound = 0.1`. Always accepts ≥1 (the first sorted element gives `0 ≤ bound`). [V-P0]
- [x] Entropy is `-Σ p log p` from `jax.nn.log_softmax(denoiser_logits.astype(float32))` — **nats**,
      over the vocab axis, on the **shaped** logits. [V-P0]
- [x] `_WhileLoopCarry` is exactly `(step, canvas, sc_embeddings, rng, done)` — no extension slot.
      [V-P0]
- [x] `sample_next_canvas` returns `final_carry.canvas`; `_sample_step` writes *that same tensor* to
      the KV cache and to `predicted_tokens`. **No argmax anywhere in the emission path.** The only
      `argmax` in `_sampler.py` is `first_stop_idx` at line 305 — verified by grep, exactly one hit.
      **§3.1.** [V-P0]
- [x] `logit_shaper`'s output (`shaped_prediction`) is the single tensor consumed by **both**
      `sample_from_predictions` **and** `embedder.encode_logits` for self-conditioning. **§5.2.**
      [V-P0]
- [x] Softcap is applied inside `call_with_self_conditioning`, before the shaper — the last three
      lines of `DiffusionMixin.call_with_self_conditioning`, `tanh(logits/cap)*cap`. [V-P0]
- [x] `_truncate_canvas_at_stop_tokens` keeps the first stop token and PADs after it;
      `& ~done` makes a finished sequence emit an all-PAD block. `PAD_TOKEN = 0`.
      `end_tokens = (EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE, *stop_tokens)`. [V-P0]
      Measured ids: **`EOS=1, END_OF_TURN=106, BEGIN_OF_TOOL_RESPONSE=50`**; `PAD=0`.
- [x] `_sample_step` advances by a fixed `canvas_length`; the last block does **not** shrink. [V-P0]
- [x] `canvas_length = 256`, `max_denoising_steps = 48` — defaults on the diffusion
      `Sampler`/`ChatSampler`. Note they are **required fields with no default** on
      `DiffusionSampler` itself. [V-P0]

**Resolved since the first draft:**

- [x] **`ChainedEarlyStop` is AND** — `jnp.all(jnp.stack([...]), axis=0)`. [V-P0] Combined with
      `TokenStabilityEarlyStop.should_stop = jnp.all(argmax(logits) == previous_canvas, axis=-1)`
      (no fields, no patience), early stop **can only fire on a step whose predecessor achieved
      full entropy-acceptance** — a uniform random token will essentially never equal an argmax.
      The reasoning is confirmed: `previous_canvas` is the canvas that *entered* this step, so
      stability requires the previous step to have accepted all 256 positions.
      **But see §1.4 — the empirical consequence is the opposite of what this spec predicted.**
- [x] **`DiffusionSampler`'s own default is `NoEarlyStop`** — only `Sampler`/`ChatSampler` inject
      the chain. A directly-constructed `DiffusionSampler` **never early-stops** and therefore
      *always* exits via the budget path. If you construct one directly in tests, you will see the
      random-token failure mode every time. [V-P0]
- [x] **`_MIN_TEMP = 1e-12`.** [V-P0] `AnnealingTemperatureShaperConfig.__post_init__` rejects
      `min_temperature < _MIN_TEMP` and `max < min`, so `min = max = 1e-12` is legal and gives
      `logits / 1e-12` → the categorical collapses to argmax and all entropies collapse to 0.
      **A near-greedy path is reachable by configuration alone, with no code change.** §3.9 depends
      on this. Config defaults are `exponent=1.0, max_temperature=0.8, min_temperature=0.4`; with
      48 steps the last *executed* step sits at `noise_proportion = 1/48`, giving `T = 0.4083` —
      hence the "0.8 → 0.408" schedule quoted throughout. [V-P0]
- [x] **`forbidden_tokens` and `sampling` are definitively inert on the diffusion path.** They are
      read only in the base `SamplerLoop._sample_step` (which does
      `einops.rearrange(logits, 'B 1 V -> B V')` — strictly single-token autoregressive), and
      `DiffusionSampler._sample_step` is an `@override` that references neither. [V-P0] **Do not use
      them for constraints; they will silently do nothing.**
- [x] **The outer block loop is `jax.lax.while_loop` inside `jax.jit`**, with a **traced**
      `max_new_tokens`. Structure: `jit _sample_loop → lax.while_loop (blocks) → _sample_step →
      sample_next_canvas → lax.while_loop (denoising steps)`. **There is no Python between blocks.**
      [V-P0] **§5.3 is built entirely on this.**
- [x] `SamplingState` is a `flax.struct.dataclass(kw_only=True)` with
      `predicted_tokens: Int['B max_out_length']` holding everything committed so far, plus
      `step, done, last_token, last_token_pos, cache, rng, init_cache_length,
      full_attention_mask` — **exactly nine fields**. [V-P0]
- [x] **`SamplingState.cache_info` RESOLVED: it is a `@property`, not a field** —
      `return _cache_helper.Cache(self.cache)`, defined in `_sampler_loop.py`. It is therefore
      *derived*, and widening `SamplingState` does not have to supply it. [V-P0]

**CORRECTED and newly found — these change the design, see §5.3:**

- [x] **CORRECTED. `max_new_tokens` is not a field of `SamplingState` at all.** It is a parameter
      of `_sample_loop`, captured only in that function's `cond_fn` closure. So
      `R = max_new_tokens − state.step` is **not computable inside `_sample_step`**, contrary to
      what §5.3 and §3.5 previously asserted. [V-P0]
- [x] **CORRECTED. `sample_next_canvas` never receives `state`.** Its signature is
      `(*, canvas_length, max_denoising_steps, batch_size, cache, params, rng,
      full_attention_mask)`. Neither `state.step`, nor `predicted_tokens`, nor any per-block carry
      is reachable from inside it. The only per-block quantity available there is
      `cache_layer['end_index']` (tokens committed so far). **This is why `_sample_step` must be
      forked after all — see §5.3(c).** [V-P0]
- [x] **NEW. The block loop has a *third* exit condition.** `_sample_loop`'s `cond_fn` is
      `(state.step < max_new_tokens) & ~all(state.done) & ~state.cache_info.is_full`.
      §3.1b lists only budget truncation and early stopping; **cache exhaustion is a third way to
      terminate mid-grammar** and must be closed the same way. [V-P0]
- [x] **NEW. The stock entropy computation is already `0·log 0`-safe.** Both
      `SampleFromPredictions` and `EntropyEarlyStop` do
      `log_probs = jnp.where(probs == 0, 0.0, log_probs)` before the sum. So §2.4's NaN hazard does
      **not** apply to the stock entropy path even with `-1e30`/`-inf` logits. The hazard is real
      and unguarded in exactly one place: **`embedder.encode_logits(shaped_prediction)`**, the
      self-conditioning tap. Keep §2.4's finite-sentinel rule, but know where it actually bites.
      [V-P0]
- [x] **NEW. `_sample_loop` runs a second, global truncation pass after the block loop.**
      `_mask_tokens_after_end_tokens` zeroes everything strictly after the first `end_token` across
      the whole `predicted_tokens` buffer. Per-block truncation is not the last word. [V-P0]

### 1.3 Verify the tokenizer

```python
tok = sampler.tokenizer
print(tok.vocab_size, tok.special_tokens)   # expect 262144 = 2^18
```

**All four resolved in Phase 0. Two were wrong.**

1. **CORRECTED — there is no lm_head padding.** `vocab_size = 262,144 = 2^18` [V-P0], but the
   SentencePiece model has **262,144 real pieces** and the highest non-empty id is **262,143**.
   Exactly **three** ids decode to `""`: `PAD=0`, `EOS=1`, `BOS=2` — control tokens, not padding.
   The contiguous empty tail is **length 0**. So there are no dead slots to exclude; what must be
   excluded from `FREE` is the control/special set, which §3.6 already requires. [V-P0]
2. **CONFIRMED. A `<mask>` token exists (`MASK = 4`, a single token id) but is NOT the diffusion
   mechanism.** DiffusionGemma corrupts by replacing tokens with uniform random tokens
   (`jax.random.randint(0, text_vocab_size)`). [V-P0]
3. **CORRECTED — `Vocabulary.from_pretrained` is not the path.** It fails on
   `google/gemma-3-4b-it` with **HTTP 401**: the repo is gated, so this is an auth failure, not the
   `UnsupportedByTokenProcessor` rejection this spec anticipated. **Build the `Vocabulary`
   directly from Gemma's own SentencePiece model instead** — it is the ~30 lines described here and
   needs no HF access at all:

   ```python
   vocab = outlines_core.Vocabulary(tok.special_tokens.EOS, {})
   sp = tok._sp
   for i in range(sp.GetPieceSize()):
       if sp.IsControl(i) or sp.IsUnknown(i):
           continue                                    # emit no bytes; keep out of the alphabet
       piece = sp.IdToPiece(i)
       if len(piece) == 6 and piece.startswith('<0x'): # byte fallback
           b = bytes([int(piece[3:5], 16)])
       else:
           b = piece.replace('▁', ' ').encode()
       if b:
           vocab.insert(b, i)
   ```

   Measured: **262,140 pieces mapped** (256 byte-fallback, 4 control/unknown skipped),
   `len(vocab) = 262,141`, built in **2.9 s**. `outlines_core.Index(regex, vocab)` and
   `.get_transitions()` both work on it. See `scripts/phase0_outlines.py`. [V-P0]
4. **RESOLVED, and better than expected — but the marker is not what this spec assumed.**
   There is **no** `END_OF_THOUGHT` token and nothing thought-related in `special_tokens`. The
   released model emits a **channel-tagged** format, and the channel delimiters *are* single
   dedicated token ids:

   | literal | token id(s) | single dedicated? |
   |---|---|---|
   | `<\|channel>` | **100** | ✅ yes |
   | `<channel\|>` | **101** | ✅ yes |
   | `thought` | 45518 | (an ordinary word token) |
   | `<\|channel>thought\n<channel\|>` | `[100, 45518, 107, 101]` | 4 tokens |

   So §3.6's two-state construction **does hold**, keyed on ids **100/101**, not on a hypothetical
   `END_OF_THOUGHT`. No Aho–Corasick is needed and there is no BPE-ambiguity problem. Rewrite §3.6
   against the real format. [V-P0]

`special_tokens` in full, measured: `PAD=0, EOS=1, BOS=2, UNK=3, MASK=4,
BEGIN_OF_TOOL_RESPONSE=50, START_OF_TURN=105, END_OF_TURN=106, START_OF_IMAGE=255999,
START_OF_AUDIO=256000, IMAGE_PLACEHOLDER=258880, AUDIO_PLACEHOLDER=258881, END_OF_IMAGE=258882,
END_OF_AUDIO=258883`. Note `<|channel>`/`<channel|>` (100/101) are **not** in this enum despite
being dedicated ids — do not enumerate the special set from `special_tokens` alone. [V-P0]

### 1.4 Smoke test and baseline instrumentation

Generate 20 prompts unconstrained. Capture per step: `n_accepted`, `mean_entropy`, `max_entropy`,
`Σ H − max H`; per block: **the number of non-accepted positions on the final executed step**, and
**whether the block exited via early stop or via the 48-step budget** (§1.2's invariant). Time a
step, record peak HBM. This is the baseline §3.4 recalibrates against.

#### [V-P0] Measured baseline — H100 80GB, `ChatSampler` defaults, 2026-07-27

Scripts: `scripts/phase0_smoke.py` (20 prompts, `max_new_tokens=256`) and
`scripts/phase0_multiblock.py` (3 long prompts, `max_new_tokens=1024`). Raw data in
`artifacts/phase0_baseline.json` and `artifacts/phase0_multiblock.json`.

| Quantity | Measured |
|---|---|
| Blocks observed | **31** (20 single-block + 11 across 3 multi-block runs, up to 4 blocks deep) |
| Exited via **early stop** | **31 / 31** |
| Exited via the **48-step budget** | **0 / 31** |
| Denoising steps actually executed per block | **3 – 31** of 48 (median ≈ 12) |
| Non-accepted positions on the final executed step | **0** in 30 blocks, **1** in one block |
| Params resident / peak HBM | 51.65 GB / **56.5 GB** of a 76.52 GB limit |
| Checkpoint load | 35–36 s from local disk |
| First generation (cold compile) | 47.3 s → 11.9 s with a warm `jax_compilation_cache_dir` |
| Steady-state generation, 256 tokens | **1.2 – 4.2 s** (≈ 0.21 s per denoising step) |

**Three consequences, all of which change decisions elsewhere in this spec:**

1. **The stock model never exhausts its step budget on ordinary prompts.** It converges in 3–31 of
   48 steps. §7.4's `48 → 24 → 12 → 6` ablation is therefore measuring something different from
   what was intended below ~24 steps: the model already stops well short of 48, so the first two
   rungs may be no-ops. Report *executed* steps, not the cap.
2. **§3.1's invariant is false as written — see the correction there.** Early stopping does *not*
   keep uniform random tokens out of the emission.
3. **The emitted format is channel-tagged.** Every one of the 11 long generations began with
   exactly `<|channel>thought\n<channel|>` = ids `[100, 45518, 107, 101]` at positions 0–3, with no
   second channel marker anywhere in the output. §3.6 must be written against this.

---

## 2. The method, restated precisely

`L = 256` (canvas), `V = 262144` (vocab), `B` = batch.

### 2.1 Objects

The model produces logits, softcapped inside the transformer and temperature-scaled by
`logit_shaper` into `shaped_prediction`. Define the **mean-field marginals**

```
p_i(v) = softmax(shaped_prediction_i)_v ,   i ∈ [L], v ∈ [V]
```

The implied joint is `p(x) = ∏_i p_i(x_i)` — fully factorized. This is the *only* thing the method
needs from the model, which is why it is agnostic to the forward process.

An automaton `M = (S, V, E, s₀, F)` with edges `e ∈ E` carrying `src(e), dst(e) ∈ S` and
`label(e) ⊆ V`. `C = L(M) ∩ V^L`.

### 2.2 The chain graphical model

Latents are **edges**, not states — the paper's design choice, and what makes the emission factor
local. [V, Eq. 3]

```
p_M(z₁ = e)               = 1[src(e) = s₀]
p_M(z_i = e' | z_{i−1}=e) = 1[dst(e) = src(e')]
p_M(x_i = v | z_i = e)    = 1[v ∈ label(e)]
terminal factor           = 1[dst(z_L) ∈ F]
```

For a **DFA** each accepted string has exactly one accepting path, so `p_M` is uniform on `C` and
the product below is *exactly* the constrained posterior. For an **NFA** it weights strings by
their number of accepting paths — a soft proxy. Support (hence constraint satisfaction) is
unaffected. [V]

> **The path-weighting is not just a caveat about NFAs — it is a correctness requirement on the
> sampler.** Eq (2) sums over parallel edges, so the state-path distribution *is* path-weighted.
> If the token draw is not weighted the same way you get a *third* distribution, neither uniform on
> `C` nor path-weighted. See §2.6 eq (8) and §6.1 test 4.

### 2.3 The tractable product

Reweight **emissions only**; transitions are untouched. [V, Eq. 5] Define the **edge emission
mass**:

```
W[i, e] = Σ_{v ∈ label(e)} p_i(v)                                        (1)
M_i(s, s') = Σ_{e : src(e)=s, dst(e)=s'} W[i, e]  ∈ R_{≥0}^{|S|×|S|}     (2)
```

### 2.4 Forward–backward

**Reference semantics.** These define *what the values are*; §2.6 computes them at `O(log L)`
depth. **Never run these as a Python or `lax.scan` loop** — that reintroduces a device `while` and
costs the CUDA-graph capture that is the reason for choosing JAX (§0).

```
a_0(s)     = a_start(s)     # 1[s=s₀] for a DFA at block 0; a vector in general — §5.7
a_i(s')    = Σ_s a_{i−1}(s) · M_i(s, s')                                 (3)

b_L(s)     = terminal factor    # NOT 1[s ∈ F]; budget-aware form in §3.1b
b_{i−1}(s) = Σ_{s'} M_i(s, s') · b_i(s')                                 (4)

Z = a_0 M_1 M_2 ⋯ M_L b_L = Σ_{x ∈ C} ∏_i p_i(x_i)
```

**Scaling is mandatory.** `a_i` is a product of `i` sub-unit matrices; at `L=256` it underflows
fp32 to exactly zero. Store `a_i`, `b_i` max-normalized with accumulated log-scales; then
`log Z = log(a_i·b_i) + logscale_a[i] + logscale_b[i]` is **`i`-invariant** — assert it across all
`i`. [D]

**Per-position constrained marginals.**

```
u_i(e) = a_{i−1}(src e) · b_i(dst e)                                     (5)
q_i(v) = p_i(v) · ( Σ_{e : v ∈ label(e)} u_i(e) ) / Z                    (6)
```

Indexing verified against brute-force enumeration: `a_{i−1}` is the state *before* position `i`,
`b_i` the state *after*. No off-by-one. `Σ_v q_i(v) = (1/Z)Σ_e u_i(e)W[i,e] = a_i·b_i/Z = 1` —
make it a unit test. [V]

**Complement-aware evaluation of the inner sum.** Under §4.4's class representation the scatter in
(6) is *not* a plain sparse scatter — negated classes contribute everywhere. Since
`1[v ∈ S_c] = 1 − 1[v ∈ N_c]` for a negated class:

```
r_i(v) = Σ_{c ∈ Neg} U_i(c) + Σ_{c ∈ Pos, v ∈ S_c} U_i(c) − Σ_{c ∈ Neg, v ∈ N_c} U_i(c)
q_i(v) = p_i(v) · r_i(v) / Z          where  U_i(c) = Σ_{e ∈ class c} u_i(e)
```

Verified against a direct scatter to 1.1e-16 with mixed polarity. [V] Getting this wrong silently
produces a *plausible* wrong distribution that only §6.1 test 11 will catch.

> **`q_i` has exact zeros wherever the automaton forbids a token. Never compute `log q_i`
> directly.** For entropy use `lq = jnp.log(jnp.maximum(q, 1e-30))` and `H = -(q*lq).sum(-1)`,
> which gives exactly `0·(−69) = 0` on forbidden tokens. For any write-back into a logits slot use
> a finite sentinel: `jnp.where(support, shaped, -1e30)`, **never `-inf`** — `-inf` logits produce
> `NaN` entropies and poison the self-conditioning embedding matmul. Verified: naive
> `-Σ q log q` → `nan`; `(-inf logits) @ embeddings` → `-inf`. [V] This rule applies at every use
> site: §3.4, §3.7, §5.2.

`q_i` is the paper's **Mar** confidence signal, which beat the unconstrained **Mf** on every
Python-format benchmark (Table 2: Dream xLAM-Python 68.4 → 76.4). [V]

### 2.5 Ancestral (chain) sampling — `O(L)` depth

```
s ← s₀
for i = 1..L:
    e_i ~ P(e) ∝ 1[src(e)=s] · W[i,e] · b_i(dst e)
    x_i ~ P(v) ∝ p_i(v) · 1[v ∈ label(e_i)]
    s ← dst(e_i)
```

Exact, and **path-weighted** — it draws the edge first. This is the **reference implementation**
(§6.2) and the fallback for large `|S|` (§5.6), but it is 2.1× slower than unconstrained decoding
end to end (Table 3: +114%) and is not the shippable path. In JAX it is a `lax.scan`, which
compiles to a device `while` and escapes command buffers — the mechanism behind that +114%.

### 2.6 Tree (log-depth) sampling — Algorithm 1, and how it maps to JAX

**Indexing — 0-based throughout this subsection.** Positions `0..L−1`; boundary states `s_0..s_L`
where `s_i` is the state **before** position `i`. Position `i` sits between `s_i` and `s_{i+1}`.
`M_i` here is §2.4's `M_{i+1}`. (§2.4 is 1-indexed, §2.6 is 0-indexed. Use 0-based in code; the
exactness harness catches you if you mix them.)

`P_{[ℓ,r)} = M_ℓ M_{ℓ+1} ⋯ M_{r−1}`, the transfer from `s_ℓ` to `s_r`. Associativity is the trick.

```
P(x_{ℓ:r−1} | s_ℓ, s_r) ∝ Σ_{s_m} P(s_m|s_ℓ,s_r) · P(x_{ℓ:m−1}|s_ℓ,s_m) · P(x_{m:r−1}|s_m,s_r)
P(s_m | s_ℓ, s_r)       ∝ P_{[ℓ,m)}(s_ℓ, s_m) · P_{[m,r)}(s_m, s_r)      (7)
```

**Top-down.** Draw the boundary pair jointly — the start is a *vector*, not a point mass, past
block 0 or under an NFA (§5.7):

```
(s_0, s_L) ~ P ∝ a_start(s_0) · P_{[0,L)}(s_0, s_L) · b_L(s_L)
```

then recurse by (7). Once every `s_i` is fixed, all tokens are drawn independently and in parallel
— **but the draw must be edge-multiplicity-weighted, not an `∃` indicator**:

```
e_i ~ P(e) ∝ 1[src(e)=s_i, dst(e)=s_{i+1}] · W[i, e]                     (8a)
x_i ~ P(v) ∝ p_i(v) · 1[v ∈ label(e_i)]                                  (8b)
```

equivalently, in one step,
`x_i ~ P(v) ∝ p_i(v) · |{e : src(e)=s_i, dst(e)=s_{i+1}, v ∈ label(e)}|`.

> **This is a real bug if you write the `∃` form.** Measured on a 3-state NFA with parallel,
> overlapping-label edges at `L=4`: the `∃` form deviates from the exact path-weighted posterior by
> **1.7e-2**; the multiplicity-weighted form by **2.1e-17**. [V] On a **DFA** the two coincide (one
> edge per `(s,s')`, disjoint labels), so gate the cheap `∃` path on an `is_dfa` flag. Note also
> that §2.5's chain sampler *is* path-weighted, so with the `∃` form §6.1 test 4 (tree == chain)
> fails on every NFA — which is how you would find this, if you tested NFAs.

#### Implementation in JAX — four facts that matter

**(a) `lax.associative_scan` computes exactly the tree you need, and throws it away.** Its
`reduced_elems` at recursion depth `k` *are* the `P_{[ℓ,r)}` over aligned dyadic blocks of size
`2^k` — verified against hand-computed block products. There is no API to retain them. **Hand-roll
an ~8-line bottom-up loop over `log₂(256) = 8` levels**, Python-unrolled at trace time. Do *not*
use `lax.fori_loop`: level shapes differ (`L, L/2, L/4, …`) so it cannot be a real loop, and
unrolling is what buys the CUDA-graph capture. [V]

**(b) Build the Blelloch / Brent–Kung shape.** An 8-level **up-sweep** producing the `L−1` internal
aligned dyadic products (plus `L` leaves = `2L−1` nodes, matching §5.6's memory table), then an
8-level **down-sweep** for the exclusive prefix/suffix vectors `a`/`b`. Measured on
`associative_scan` at `L=256`: **16 combine invocations** (`2·log₂L`) and **502 elementwise
combines ≈ 2L`** — work-efficient. [V]

> **Do not build it Kogge–Stone-shaped.** A shallower Hillis–Steele scan does **3.6× the work**
> (1793 combines at `L=256`), does not naturally produce an aligned dyadic node set — which eq (7)
> requires — and materializing its levels is `L·log₂L = 2048` nodes versus `2L−1 = 511`, a **4×**
> memory blow-up that invalidates §5.6's table and flips the dispatch decision. And you need the
> down-sweep regardless, because that is where `a` and `b` come from. The often-cited
> shallow-scan win is a benchmark on 500 *small* matrices; at `S ≥ 512` a single combine is a
> multi-GFLOP GEMM (§0) and the work-efficient shape strictly wins.

**(c) `reverse=True` reverses the operand order — a silent correctness trap.** The docstring is
explicit: it yields `f(f(z,y),x)`. For non-commutative matmul that is wrong for the suffix pass.
Verified: `lax.associative_scan(jnp.matmul, M, reverse=True)[0]` equals `M[L-1] @ … @ M[0]`, while
`lambda a, b: b @ a` gives `M[k] @ … @ M[L-1]` at every `k`. No shape error, just wrong
probabilities. Put this in a comment at the call site. [V]

**(d) `L` must be a power of two, or pad with identities.** The 8-level dyadic decomposition
assumes `L = 2^k`. `canvas_length = 256` always [V], so this is normally moot — but if you ever run
a shorter canvas, pad to the next power of two with `M_i = I` for `i ≥ L`, and place `b_L` at the
**true** `L`, not the padded one. §6.1 test 7 exercises this.

**Numerical scaling.** Normalize every tree node's matrix by its max entry on construction.
*Provably free*: in (7) the scale factors of `P_{[ℓ,m)}` and `P_{[m,r)}` are scalars, constant
w.r.t. `s_m`, and cancel; likewise for the root draw. Accumulate `Σ log scale` over **all `2L−1`
nodes** to recover `Z`. [D]

**PRNG.** `keys = jax.random.split(jax.random.fold_in(key, level), n_nodes)`, then `vmap` the
categorical across nodes. `n_nodes` is static because the tree is unrolled, satisfying `split`'s
static-count requirement for free. Reproducibility across levels is *exact* — `fold_in` is a pure
function of `(key, index)`, so unrolling or reordering cannot change results. **Gotcha:** with
midpoint recursion a position is sampled at a level depending on `L`, so draws are not stable per
position across different `L`; if you want per-position stability for A/B-ing against the reference,
address by position (`fold_in(key, i)`). [V]

#### A cheaper alternative for the sampling phase only

`dynamax`'s `parallel_inference.hmm_posterior_sample` does `O(log L)` ancestral sampling without
the midpoint matrices, via a Särkkä-style FFBS whose associative operator composes index maps:

```python
@vmap
def _operator(E_jk, E_ij):
    return jnp.take(E_ij, E_jk)     # composition of functions S -> S
```

scanned in reverse over `[L, S]` **int** arrays — `O(S)` per combine instead of `O(S²)`. Worth
prototyping. [V]

> **But it does not remove the wall.** It makes the *backward sampling* pass `O(S)`; the parallel
> *filtering* pass still composes `S×S` elements, and §2.4's `a`/`b` are a prefix/suffix product of
> matrices with no `O(S)` associative form. At `|S| = 19,509` the `M_i` leaves alone are
> `L·S²·4 = 389 GB`. Spider remains structurally infeasible in any dense formulation. Do not let
> §9's open question 0a rest on this.

Theory for both semirings: Hassan, Särkkä & García-Fernández,
[arXiv:2102.05743](https://arxiv.org/abs/2102.05743). Note dynamax's *sequential* `inference.py` is
a plain `lax.scan` — you want `parallel_inference.py`, and it has **no parallel Viterbi**; write
that yourself (§2.7).

**Complexity.** The paper states these in edge space; §4.4 supersedes the `|E||V|` term with
`O(nnz)`, and §2.4's state-space recursions make the chain `O(L|S|²)`.

| | work | depth |
|---|---|---|
| chain, state-space (§2.4/§2.5) | `O(L(\|S\|² + nnz))` | `O(L)` |
| **tree** | `O(L(\|S\|³ + nnz))` | **`O(log L)`** |

### 2.7 Constrained MAP (max-plus)

The paper claims greedy support and calls itself "a strict generalization of DINGO, supporting
greedy decoding *and sampling*", where DINGO does **MAP inference on DFAs**. But there is no greedy
algorithm box and no occurrence of argmax/MAP/Viterbi in the method section. [?]

[D] **This spec implements exact constrained MAP via the max-plus semiring in log space.**

```
M̃_i(s, s') = max_{e : src(e)=s, dst(e)=s'} max_{v ∈ label(e)} log p_i(v)     (9)
```

Replace `(+, ×)` with `(max, +)` over log-probabilities in (3)/(4) and take `argmax` instead of
sampling in the tree recursion. `(max, +)` is a semiring, so the identical tree gives `O(log L)`
exact MAP. Verified in JAX against exhaustive `argmax` on both DFAs and NFAs with parallel
overlapping edges: error `0.000e+00`. The combine is
`jnp.max(a[..., :, :, None] + b[..., None, :, :], axis=-2)`. [V]

Use **log space, not `(max, ×)`**: exact, no scaling discussion, no underflow. Represent impossible
transitions with a finite sentinel (`-3e38`), not `-inf`, so fused kernels don't produce `NaN` from
`-inf + -inf`. [D]

**MAP is exact on NFAs too** — path multiplicity cannot change a `max` over strings. So §2.2's
soft-proxy caveat applies to *sampling only*, and MAP on Spider is exact. [V]

**MAP is exactly temperature-invariant**, since
`argmax_x ∏_i softmax(ℓ_i/T)_{x_i} = argmax_x Σ_i ℓ_{i,x_i}` for any `T > 0`. So a MAP emission is
unaffected by the 0.8 → 0.408 schedule; temperature enters only through sampling and the accept
rule. Removes an ablation axis. [D]

#### Token recovery — needs a `(s,s') → class` table, not just a class table

**Do not store an `O(L|S|²)` backtrace table** — at `|S|=2459` that is 12.4 GB. But "recover the
token from the per-class argmax table" is under-specified on its own: once the tree fixes
`(s_i, s_{i+1})` you still need to know *which* class realized the max on that transition. [D]

```
argclass[i, pair]  = argmax over edges e with (src,dst)=pair  of  W_c^max[class_id[e], i]
x_i                = argmax_token[ argclass[i, pair(s_i, s_{i+1})], i ]
```

`argclass` is `[L, |E_pairs|] int32` (one entry per *distinct* `(src,dst)` pair present in the
automaton, not per state pair), and `argmax_token` is the `[C, L] int32` per-class argmax table
from §4.4 Layer 2b — `500·256·4 = 512 KB`, not the ~100 KB figure, which belongs to Layer 3's CSR.

**Deterministic tie-breaking — lowest token id, then lowest state id.** `test_guarantee` and the
unconstrained-equivalence test depend on it. [D] **JAX has no `jax.ops.segment_argmax`** (it exports
exactly `{segment_max, segment_min, segment_prod, segment_sum}`), and the obvious two-pass
workaround gets the tie-break **backwards**:

```python
# WRONG — segment_max over indices resolves ties to the LARGEST index:
#   vals=[5,7,7,2,7,1] seg=[0,0,0,1,1,1] -> [2, 4]
mx = jax.ops.segment_max(vals, seg, num_segments=n)
am = jax.ops.segment_max(jnp.where(vals == mx[seg], idx, -1), seg, num_segments=n)

# CORRECT — lowest index wins ties, matching the stated convention:  -> [1, 4]
mx = jax.ops.segment_max(vals, seg, num_segments=n)
am = jax.ops.segment_min(jnp.where(vals == mx[seg], idx, idx.shape[0]),
                         seg, num_segments=n)
# Empty segments: segment_max's identity is -inf (NOT -1), and am == idx.shape[0].
# Test for am == N, never am == -1.
```

Float equality is safe here — `segment_max` returns an element of `vals` exactly. Both behaviours
verified. [V] PyTorch's `torch_scatter.scatter_max` returns argmax natively; a minor real edge to
PyTorch.

Also provide `--emission=marginal-argmax` (per-position `argmax_v q_i(v)`) as an ablation. It is
*not* guaranteed to lie in `C` for `|S| > 1` and must be validated post-hoc; include it only to
measure what the joint MAP buys.

### 2.8 Why per-position masking is wrong — keep this in the README

Per-position masking enforces `∏_i π_i(C)`, the product of coordinate projections, which strictly
contains `C` whenever `C` is not a product set. The paper's example [V]:

> if `C` only accepts valid real numbers, both "1." and ".1" are valid length-2 sequences, but a
> factorized sampler that draws "." at position 1 (consistent with ".1") and "." at position 2
> (consistent with "1.") produces the invalid sequence "..".

Table 1's `CS` column quantifies the damage: unconstrained constraint satisfaction under `T=1`
sampling falls to **4.4%** (Dream, BFCL-Live Python) and **7.6%** (Sudoku). [V]

---

## 3. Reconciling the paper with the JAX DiffusionGemma sampler

### 3.1 The emission is the sample — and it can contain uniform random tokens

The stock loop, verbatim [V]:

```python
denoiser_tokens = jax.random.categorical(categorical_rng, denoiser_logits.astype(jnp.float32))
random_tokens   = jax.random.randint(noise_rng, canvas.shape, minval=0, maxval=text_vocab_size)
output_tokens   = jnp.where(selection_mask, denoiser_tokens, random_tokens)
```

and `sample_next_canvas` returns `final_carry.canvas`, which `_sample_step` writes **unchanged** to
both the KV cache and `predicted_tokens`.

**Good news.** Unlike the HF PyTorch implementation — which emits `argmax(processed_logits)`, i.e.
exactly the factorized decode §2.8 proves unsound — the JAX implementation emits *the sample*. That
restores the paper's setting: replace the per-position categorical with §2.6's constrained joint
sampler and the emitted canvas is a draw from the constrained posterior, hence in `C`.

**The residual hole.** Positions not accepted on the **final executed step** are emitted as uniform
random tokens over the whole 262k vocab, with no cleanup pass.

> **⚠ CORRECTED [V-P0]. The invariant this spec previously stated here is FALSE, and the error was
> in the direction that matters.** The old claim was:
>
> > ~~Random tokens reach the output **only when the 48-step budget is exhausted** without reaching
> > a stability fixed point.~~
>
> **Early stopping does not protect the emission.** Read `body_fn` carefully:
>
> ```python
> out      = self.sample_step(...)                 # out.sampled_tokens is the fresh sample
> new_done = carry.done | self.early_stop_fn.should_stop(
>     step=step, canvas=out.sampled_tokens,
>     previous_canvas=carry.canvas, logits=out.logits)
> canvas   = jnp.where(carry.done[:, None], carry.canvas, out.sampled_tokens)   # OLD done
> ```
>
> `should_stop` is evaluated on `previous_canvas` and `logits` — it certifies convergence of the
> step's **input**, not of its **output**. The gate on the emitted canvas is `carry.done`, the
> *old* flag, so on the step where early stop first fires the canvas still becomes
> `out.sampled_tokens`, **including that step's own unaccepted positions**. The loop then exits.
> A block can therefore early-stop *and* emit uniform random tokens in the same step.
>
> **Measured (§1.4): 31/31 blocks exited via early stop, 0/31 via the budget — and one block still
> emitted a random token** (1 unaccepted position on its final step). So the failure mode is real,
> is *not* gated by the budget path, and is simply **rare** on ordinary prompts: ~1 block in 31,
> ~1 position in 7,936. It is still unconditional-guarantee-breaking, and it is exactly what J0's
> decoupled `emit_canvas` closes.

Two things remain true and matter for testing: a directly-constructed `DiffusionSampler` defaults
to `NoEarlyStop` and so *always* runs the full 48 steps [V-P0]; and the frequency above is a fact
about the released model that nobody has published — report it.

#### [D] Emission designs

| | Trajectory (fed back to model) | Emitted | Guarantee | Fork |
|---|---|---|---|---|
| **J0-map** *(default)* | stock: constrained sample at accepted, uniform random elsewhere | **constrained MAP** (§2.7) | ✅ unconditional | `sample_next_canvas` + widened carry |
| **J0-sample** | as above | **constrained joint draw** (§2.6) | ✅ unconditional | same |
| **J1** | single constrained joint draw with **flattened marginals** at non-accepted positions | same tensor | ✅ unconditional, *and every intermediate canvas ∈ C* | `sample_next_canvas` only |
| **J2** *(baseline only)* | constrained sample at accepted, uniform random elsewhere | same tensor | ❌ only at full acceptance | `sample_next_canvas` only |

**J0 — ship this.** Decouple trajectory from emission. The trajectory keeps stock uniform
renoising, so the model's *inputs* stay on its training distribution; the emission is constrained,
so the guarantee is unconditional and independent of whether the accept prefix ever covers the
canvas. Cost: the denoising carry gains an `emit_canvas` field (§5.4).

**J1 — the elegant one, and a genuine contribution.** Keep the single-tensor contract, but make the
whole canvas one joint draw: build `p'_i = p_i` at accepted positions and a flattened /
high-temperature `p_i` at non-accepted positions, then draw **one** joint sample from the
constrained posterior with marginals `p'`. Accepted positions get the model's confident choice,
non-accepted get near-uniform *but grammar-consistent* noise, and the whole canvas is in `C` by
construction. Strictly stronger than the paper, whose intermediate canvases contain `[MASK]`s.
Risk: the model was trained to denoise uniform random noise, not grammar-valid noise. **Ablate
against J0.**

**J2 — the honest baseline.** The naive port; its CS column is the evidence that J0/J1 are needed.

> **All four still require §5.3's automaton threading.** "J1 needs no fork" refers only to the
> denoising carry — the automaton arrays must be *traced* regardless, or you recompile per schema
> (~2 h across BFCL). Do not read the table's "Fork" column as "no changes needed".

### 3.1b The guarantee, stated correctly

> **Proposition (port).** If `emit_canvas` is either the max-plus MAP (§2.7) **or** a joint draw
> (§2.6) from the constrained posterior, with `a_start = 1[s ∈ A_k]` and `b_L = 1[d(s) ≤ R]`, then
> for every block `k`: `δ*(A_k, canvas_k) ≠ ∅` and every state in it can still reach `F` within the
> remaining budget. Hence `canvas_0 ⋯ canvas_k` is a **viable prefix** of `L(M)`. It lies in `L(M)`
> **iff** generation terminates at a block boundary with `A_{k+1} ∩ F ≠ ∅`.

The argument uses only the *support* of the constrained posterior, not maximality, which is why it
covers both emission modes. Membership in `L(M)` is **not** implied by a constrained emission
alone. **Four** failure modes, all of which must be closed:

1. **Budget truncation** — the block cap is reached with `A_k ∩ F = ∅`. `Live = {s : d(s) < ∞}` is
   an *unbounded-horizon* predicate and does not close this.
2. **Early stopping** — the early-stop functions fire on the model's own uncertainty with zero
   automaton awareness, halting mid-grammar.
3. **Stop token from a non-accepting state** — if the free-text region (§3.6) admits any of
   `end_tokens`, the emission can terminate inside the thinking channel.
4. **[V-P0] Cache exhaustion — a third loop exit this spec previously missed.** `_sample_loop`'s
   `cond_fn` is `(state.step < max_new_tokens) & ~all(state.done) & ~state.cache_info.is_full`.
   The cache filling up halts generation mid-grammar exactly as budget truncation does, and is
   *not* covered by the `R = max_new_tokens − step` accounting. Bound `R` by the remaining **cache**
   capacity as well: `R = min(max_new_tokens − step, cache_length − used_cache_length)`.

**The three closures.**

- **Budget-aware terminal factor.** Precompute `d(s) = min tokens from s to F` by BFS on the
  reversed automaton, and use `b_L(s) = 1[d(s) ≤ R]` with `R` the remaining budget.
  `d(s) ≤ 0 ⟺ s ∈ F`, so the final-block behaviour falls out automatically as `R` runs down — do
  **not** special-case `1[s ∈ F]` on the last block (§3.5 trap 1). **`d` must be computed *after*
  the stop-token augmentation of §3.5 trap 4 and *after* the `FA_grammar | FA_refusal` union of
  §3.8**, or it is a completely different function.
- **Automaton-aware stopping.** Conjoin `A_{k+1} ∩ F ≠ ∅` into the block-level done flag.
  **This needs care under J0** — see below.
- **`FREE` must exclude every `end_token` and `PAD`**, not just the marker.

> **Closure 2 is not free under J0.** §5.1's protocol is
> `should_stop(*, step, canvas, previous_canvas, logits)`, and under J0 that `canvas` is the
> *trajectory* — mostly uniform random tokens. `δ*(A_k, trajectory_canvas)` is `∅` on essentially
> every step, so a naive conjunct is permanently false and the loop never early-stops (always 48
> steps). **The forked `sample_next_canvas` must pass `emit_canvas` to a widened `should_stop`.**
> That makes `EarlyStopFn` a third protocol change, and it must compose as AND with the stock chain
> (which §1.2 confirms is already AND).

Also note the block-level done flag is **distinct from the carry's `done`**, which is monotone
(`& ~done` makes a finished sequence emit an all-PAD block [V]). Compute
`done_k = done_stock_k ∧ (A_{k+1} ∩ F ≠ ∅)` at block level; never flip the carry's `done` back to
False after a canvas has been truncated at a stop token.

### 3.2 The accept set is not monotone — this simplifies things

`selection_mask` is rebuilt from `jnp.zeros_like` every call; nothing mask-shaped is in the carry.
A position accepted at step `t` can be renoised at `t+1`. [V]

The paper's `U` (committed set) has **no analogue inside a canvas**. Therefore:

- **No delta-clamping is needed within a block.** The constrained posterior is over the full
  256-position canvas, unconditioned, every step. The canvas does carry information across steps,
  but only through the model's own `p_i`, which the method treats as given — nothing to condition
  on.
- Conditioning happens **only at block boundaries** (§3.5).
- **This is also why automaton state can be recomputed statelessly each step** (§5.3) — there is no
  accumulated within-block state to thread.

A consequence to internalize: the trajectory and the emission are two *different* strings drawn
from the same `p`. Under J0, `accept_canvas`, renoising and the stock stopping criteria all measure
convergence of a canvas that is never emitted.

Do not add a monotone commit set to make it look like LLaDA.

### 3.3 Uniform-state noise, not absorbing-state masking

Google is explicit [R]: DiffusionGemma noises "by replacing original words with entirely random
tokens from the vocabulary". Confirmed in code (`jax.random.randint(0, text_vocab_size)`). [V]

The method is unaffected — it only consumes `p_i(·)` — but **renoising is a live fork [?]**, and in
JAX it is entangled with the emission choice:

| Variant | Renoise rule | Argument |
|---|---|---|
| **R0** (default, = J0's trajectory) | uniform over `[0, V)` | Matches the training corruption exactly. Zero shift. |
| **R1** | uniform over `supp(q_i)` | Coherent context; at most positions `supp(q_i)` is large so the shift is small. Needs `q_i` (§5.6's memory gate). |
| **R2** (= J1) | flattened-marginal constrained joint draw | Every canvas is a valid string. Most elegant, largest shift, highest risk. |

Ship R0 and ablate R1/R2. The paper cannot answer this — Dream and LLaDA have no analogue.

### 3.4 The entropy bound must be recalibrated — do not skip this

`SampleFromPredictions` accepts the largest ascending-entropy prefix with
`Σ_{i≤k} Hᵢ − max_{i≤k} Hᵢ ≤ 0.1` nats; `EntropyEarlyStop` fires at mean entropy `< 0.005`. Both on
`shaped_prediction`. [V]

Constrained marginals `q_i` have restricted support and are far sharper. Substituting `q_i` (the
**Mar** variant) collapses entropies, accepts far more positions per step, and fires the stopping
criterion almost immediately. **The defaults are calibrated for unconstrained entropies and will
silently produce garbage.** [D] Use the clamp rule of §2.4 — `log q` is `-inf` on forbidden tokens
and the naive entropy is `NaN`.

Note the interaction with §3.1: more acceptance is not purely bad, since it is what keeps random
tokens out of the emitted canvas under J1/J2. Under J0 the emission is constrained regardless,
which is another argument for J0.

Required: a calibration sweep on a 100-example dev slice over
`entropy_bound ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}` × `entropy_threshold ∈ {5e-5, 5e-4, 5e-3}`,
selecting for accuracy at fixed step budget. Report chosen values, the baseline's, and
accepted-count-per-step curves for both.

`--confidence={mf,mar}` selects the entropy source. **Note it changes stopping as well as
acceptance** (the same tensor feeds both), so hold it fixed while sweeping the two thresholds, or
split the flag.

### 3.5 Blocks: the automaton must be threaded across canvases

Generation is semi-autoregressive: within a 256-token canvas all positions denoise in parallel;
across canvases it is strictly left-to-right, and committed canvases are frozen into the KV cache
and never revisited. [V]

```
A_0 ← {s₀}
per block k:
    R      ← max_new_tokens − state.step        # remaining budget, available in the loop [V]
    b_L(s) ← 1[d(s) ≤ R]                        # §3.1b — no final-block special case
    run the constrained denoising loop with a_start = 1[s ∈ A_k]
    emit canvas_k                               # J0-map: MAP; J0-sample/J1: joint draw
    canvas_k ← _truncate_canvas_at_stop_tokens(canvas_k)
    A_{k+1} ← δ*(A_k, canvas_k)                 # from the TRUNCATED canvas
    done_k  ← done_stock_k ∧ (A_{k+1} ∩ F ≠ ∅)  # block-level, not the carry's monotone `done`
```

**`R` is measured in tokens and blocks are whole**: `_sample_step` advances
`step ← step + canvas_length`, and `max_new_tokens` is effectively rounded up to a multiple of 256
[V]. `max_new_tokens` is **traced**, so the block count is not a compile-time constant — you cannot
unroll per block or size an array by it (§5.3).

Four traps:

1. **`b_L` is not `1[s ∈ F]`.** You do not know whether a block is the last one, and `1[s∈F]`
   forces the grammar to complete in exactly 256 tokens. The budget-aware form handles the final
   block automatically.
2. **Recompute `A_{k+1}` from the truncated canvas**, not from a MAP backtrace.
   `_truncate_canvas_at_stop_tokens` rewrites tokens after the first stop token to `PAD_TOKEN = 0`
   [V], and the PAD-truncated canvas is what enters the KV cache.
3. **The automaton must accept a stop-token-terminated tail**, or step 2 walks off the machine and
   `A_{k+1} = ∅`. Handle **all** of `end_tokens`, not just EOS. The single most likely source of an
   "empty state set" bug.
4. **The tail must be UNSCORED — `ACC --Σ--> ACC`, not `ACC --PAD--> ACC`.** [D]

   A joint decode maximizes/samples over **all** 256 positions. If the tail is constrained to PAD,
   terminating at position `j` costs `log p(EOS) + (255−j)·log p(PAD)`. With `p(PAD) ≈ 1e-3` and any
   grammar continuation at `p ≈ 0.3`, ending at `j=100` scores ≈ −1080 against ≈ −188 for
   continuing. **The MAP places the stop token at position 255 or never**, the model blows through
   blocks, and you land in §3.1b's truncation failure. Since `_truncate_canvas_at_stop_tokens`
   overwrites everything after the first stop token anyway, letting the tail accept any token is
   *semantically free*, and it removes the corresponding `∏ p(PAD)` bias from `Z` and `q_i` too.

   Diagnostic: log the stop-token position per block. Clustering at 255 means this is still open.

### 3.6 Thinking mode and the free-text region

> **[V-P0] Rewritten against the real format.** There is no `END_OF_THOUGHT` token. The released
> model emits a **channel-tagged** response whose delimiters *are* single dedicated token ids:
> **`<|channel>` = 100** and **`<channel|>` = 101**. Every one of the 11 long generations in §1.4
> opened with exactly `[100, 45518, 107, 101]` (`<|channel>thought\n<channel|>`) at positions 0–3
> and contained **no further channel marker** — the header is a prefix, not a thought/answer split.

```
FA_total  =  HEADER  ·  FA_grammar  ·  STOP  ·  Σ*
HEADER    =  100 · Σ_name+ · 107 · 101          # <|channel> <name> \n <channel|>
FREE      =  any token except 100, 101, every end_token, and PAD
```

Because the header is a fixed 4-token prefix in every observed sample, the `FREE*` free-text region
of the original construction is **not needed for this model** — anchor `FA_grammar` immediately
after token **101**. Keep a `FREE*` variant behind a flag in case a prompt elicits a longer or
repeated header; the two-state "no occurrence of `w`" construction is cheap and correct either way
now that `w` is a single id.

**The two-state construction only holds if the marker is a single dedicated token id** — which it
is here, so no Aho–Corasick and no BPE-segmentation enumeration is required. (For a multi-token
marker you would need `|w|+1` states plus failure links **and** every tokenization of the marker;
that risk is retired.) [V-P0]

**Caveat on coverage.** Only `thought` was observed as a channel name across 11 long generations;
`<|channel>final` and `<|channel>answer` tokenize fine (`[100, 10218]`, `[100, 14433]`) but were
never emitted. Do not hardcode `thought` — accept `Σ_name+` between 100 and 107.

**Joint decoding over a segmented automaton has its own failure mode** [?]: the decoder chooses the
split point *jointly*, so tiny probability differences can place the marker at position 0 or 255.
The model was never trained for that. Minimum: log the marker position per block. Likely needed:
freeze the thinking/answer boundary once the marker is emitted and re-anchor `A_k` there.

### 3.7 Self-conditioning — and where the constraint mask must go

`shaped_prediction` (the output of `logit_shaper`) is the **single** tensor consumed by *both*
`sample_from_predictions` *and* `embedder.encode_logits`. The SC tap is strictly **upstream** of the
sampler. [V]

> **A constraint mask installed in `sample_from_predictions` will NOT reach self-conditioning.**
> To get feedback into the next step's input embeddings, the mask must go in **`logit_shaper`**.

Subclass `AnnealingTemperatureShaper`, call `super().__call__`, then apply the support mask. You get
(a) the constraint fed into the next step's embeddings and (b) the entropy accept rule automatically
computed on the masked distribution, since §1.2 confirms entropy is taken on that same tensor.

**Use a finite sentinel, never `-inf`** (§2.4): `jnp.where(support, shaped, -1e30)`. An `-inf`
logit propagates through `softmax → @ embed_tokens.weight` and poisons the embedding. [V]

Two options to ablate [?]:

- `--self-cond=unconstrained` — mask only inside the sampler. Matches training. **Default.**
- `--self-cond=constrained` — mask in `logit_shaper`. Should sharpen convergence; off-distribution.

The support-projection mask is *sound* but *insufficient*: it enforces `π_i(C)`, which never removes
probability from a valid sequence, so it does not change the constrained posterior at all (the
per-position normalizer cancels in `Z`). Its entire value is the self-conditioning feedback and the
entropy signal. Do not mistake it for the method.

Also `is_zero_sc = jnp.all(sc_embeddings == 0.0)` is a **global** reduction over the whole batch,
not per-sequence [V] — a latent batching quirk.

### 3.8 The BFCL irrelevance trap

BFCL scores `irrelevance` (240) + `live_irrelevance` (882) = **1,122 of 2,501 single-turn entries**
by treating a decode *exception* as "no call". A grammar that forces a well-formed call scores **0**
on all of them. [R] The paper's handling is unknown [?].

[D] Make the answer region `FA_grammar | FA_refusal`, where `FA_refusal` is a free-text loop that
cannot start with `[` or `{`. Report both with and without the escape hatch and state which splits
each number covers.

**State the caveat next to every `CS = 100%` claim:** with a free-text branch (and `FREE*` before
the marker), the model can spend its whole budget outside `FA_grammar`, making CS trivially 100%
while measuring nothing. Report **CS conditional on the grammar branch being taken**, plus the
branch-taken rate.

### 3.9 Where "greedy" and "sampling" map to

The paper's two columns are `T=0` and `T=1`. DiffusionGemma's default schedule runs 0.8 → ~0.408 —
but **`_MIN_TEMP = 1e-12`** [V], so `AnnealingTemperatureShaperConfig(min_temperature=1e-12,
max_temperature=1e-12)` is legal and gives `logits / 1e-12`: the categorical collapses to argmax,
every entropy collapses to 0, and all 256 positions accept on step 1. **A near-greedy path is
reachable by configuration alone.** Use it as the closest analogue of the paper's `T=0` column, and
say in the write-up that it is a temperature limit, not a true `T=0`.

`forbidden_tokens` and `sampling` (greedy/top-k) are **definitively inert** on the diffusion path
[V] — do not reach for them.

| Flag | Meaning | Analogue |
|---|---|---|
| `--variant={j0,j1,j2}` | which emission design (§3.1) | — |
| `--emission={map,sample}` | MAP or joint draw (J0 only; J1/J2 are always a draw) | paper's greedy / sampling |
| `--trajectory={R0,R1,R2}` | what the model sees next step | no analogue |
| `--temp-schedule={default,near-greedy}` | 0.8→0.408, or `1e-12` flat | paper's T=1 / T=0 |

Default `--variant=j0 --emission=map --trajectory=R0 --temp-schedule=default`. Report the sampling
column too — that is where the paper's most dramatic gains live (Dream BFCL-L JSON: baseline
collapses 63.9 → 22.3 under sampling; constrained holds 71.5 → 69.0). [V]

### 3.10 Summary of divergences

| Aspect | Paper (Dream/LLaDA) | JAX DiffusionGemma | Handled |
|---|---|---|---|
| Forward process | absorbing `[MASK]` | uniform random tokens | §3.3 |
| Commit set | monotone `U` | rebuilt every step | §3.2 |
| **Emission** | the constrained sample | the sample + uniform random at non-accepted | **§3.1** |
| Confidence | `low_confidence`/`entropy` | `Σ H − max H ≤ bound` | §3.4 |
| Block size | 32 / none | 256, frozen; loop is traced | §3.5, §5.3 |
| Self-conditioning | none | on `shaped_prediction`, **upstream of the sampler** | §3.7 |
| Temperature | `T ∈ {0,1}` | 0.8→0.408, but `_MIN_TEMP=1e-12` allows near-greedy | §3.9 |
| Steps | 64–128 | 48 | §3.4, §7.4 |
| Vocab | 32k–126k | 262,144 | §4 |
| Constraint hook | — | `logit_shaper` + `sample_from_predictions` | §5.2 |

---

## 4. Constraint compilation

Framework-independent — ordinary Python producing arrays.

### 4.1 Pipeline

```
grammar spec (JSON Schema | regex | task DSL)
      ↓  §4.2      →  byte-level regex
      ↓  regex-automata dense::DFA (anchored, via outlines_core)
   byte DFA        →  minimize (Hopcroft over ByteClasses, ≤257 symbols, ms)
      ↓  §4.3         token lift
   token DFA/NFA   →  minimize (Valmari, §4.5)
      ↓  §4.4         label interning → TokenSet classes → CSR ×2 (sum and max tables)
   Automaton artifact (serialized, cached, padded to a power-of-two |S| bucket)
```

### 4.2 Schema → regex

`outlines_core.json_schema.build_regex_from_schema` works and is the fastest path to something
running. [V] **Its limitations are severe and silent** — put them in Phase 1's tests:

- **First-match-wins keyword precedence**
  (`empty-object > properties > allOf > anyOf > oneOf > prefixItems > enum > const > $ref > type`),
  and **sibling keywords are silently dropped**. If `properties` is present, everything else is
  ignored. [V, maintainer-confirmed]
- **All numeric bounds silently ignored** (`minimum`, `maximum`, `exclusiveMin/Max`, `multipleOf`),
  as are `patternProperties`, `propertyNames`, `uniqueItems`, `not`, `if/then/else`. [V]
- `additionalProperties: false` not honored. Only 6 `format` values. [V]
- `INTEGER` permits `-0`; `NUMBER` requires a **signed** exponent, so `1e10` is rejected. [V]
- `properties` emitted in **map order only** — an over-constraint that rejects valid documents with
  reordered keys. State the key order in the prompt.
- `{}` / missing `type` / `additionalProperties: true` expand to a 7-way alternation over all JSON
  types — the origin of "regex too large" and of §4.4's wildcard classes.

[D] **Fail loud.** Wrap the converter with a pre-pass that **raises** on any keyword outlines will
silently drop. Port llguidance's `parser/src/json/numeric.rs` (`rx_int_range`, `rx_float_range`) for
real numeric bounds. Anchor: JSONSchemaBench measured Outlines at **0.47 declared / 0.03 empirical
coverage** on Github-Hard [R].

### 4.3 Token lift

`outlines_core` is the **only** shipped library that materializes an explicit
`state → {token_id → state}` table, via `Index(regex, vocabulary).get_transitions()`. [V] XGrammar
and llguidance are per-step bitmask engines with no automaton export.

Traps [V]: `get_transitions()` deep-clones the whole nested map into Python dicts on every call
(once, at compile time, never in a loop); state ids are raw `regex-automata` dense ids — byte
offsets, **multiples of 64 as measured here, not 8** (observed range 256…2,816 with a uniform
stride of 64 [V-P0]) — not dense, renumber immediately; matching is **anchored**; a token is
allowed only if the DFA consumes **all** its bytes; `guide.advance(eos)` raises even though
`T[final][eos]` exists, so handle stop tokens out of band (you do anyway, §3.5); import from the
`outlines_core` top level, not `outlines_core.guide`. Wire
`IncompatibleVocabulary{regex, error_state, missing_tokens}` (≥0.2.14) into validation.

### 4.4 Token-set representation

Every step needs `W[i,e] = Σ_{v ∈ label(e)} p_i(v)` for all `i ∈ [256]`, `e ∈ [E]`, `E` up to
~170k [V, Table 7]. That is `W = A pᵀ` with `A ∈ {0,1}^{E×V}`, `V = 262,144`. Naive: dense bitsets
= 5.5 GB; dense float `A` = 178 GB fp32; plain CSR has `nnz = Σ_e |label(e)|` with JSON string-body
edges carrying ~260k members each. All dead.

**[D] Four layers.**

**Layer 1 — intern labels into equivalence classes.** Hash each edge's canonical label set; group
identical ones. The same `STRING_INNER` class appears at every string position. Expect **170k edges
→ O(10²–10³) classes.** Edges carry `class_id: int32`. *This ratio has never been published —
measure and report it.*

**Layer 2 — polarity, for the SUM table.** If `|S_c| > V/2`, store the complement `N_c` and a flag.
Maintain `total[i] = Σ_v p_i(v)`, then `W_c^Σ[c,i] = total[i] − Σ_{v ∈ N_c} p_i(v)`. Turns a
260k-member class into a ~20-member one. **Without this the scheme fails.** Same trick as
XGrammar's `{kAccepted, kRejected, kAcceptedBitset}` → **160 MB → 0.46 MB, 348×** [R].

**Layer 2b — the max semiring has no complement trick, and it is the guarantee path.** [D] `max`
has no inverse. Eq (9) needs `max_{v ∈ S_c} p_i(v)` and its argmax, and `--emission=map` is the
default. Fix:

> Compute `topk(p_i, K)` once per position. Then `max_{v ∈ S_c} p_i(v)` for a negated class is
> **the first top-`K` entry not in `N_c`**, and its index is the argmax token. Positive classes use
> a segmented max over the CSR (with §2.7's `segment_min` tie-break).

Verified: 0/200 failures at `K = 1 + max_c |N_c|`, and an adversarial `p` exists where `K = |N_c|`
fails — so the bound is tight. [V]

> **But `K` is unbounded under Layer 2's `|S_c| > V/2` rule** — worst case `|N_c| = 131,072`, giving
> `K = 131,073`, a 134 MB topk buffer plus a 262k-wide partial sort per position. **The max table
> must use its own polarity rule:** store the complement only when `|N_c| ≤ K_max`, otherwise store
> the class positively for the max table even if it is stored negatively for the sum table. Then
> `K = K_max + 1` is a compile-time constant. **The two tables have independent `Pos`/`Neg`
> partitions — do not share the polarity flag array**, and note §2.4's `r_i(v)` uses the *sum*
> table's partition.

> **[V-P0] `K_max = 256` is too small — use ~1,100.** Measured on the eight schemas of §4.7, the
> largest label class in every one covers **261,077–261,153 of 262,141 tokens** (~99.6% of the
> vocab) — emphatic confirmation that Layer 2's complement trick is load-bearing. But that puts
> `|N_c|` at **988–1,064**, comfortably above a `K_max` of 256. With `K_max = 256` those classes
> would all fall back to positive storage for the max table, and Layer 3's "mean post-complement
> `|S_c| ≈ 50`, `nnz ≈ 25,000`" budget would be violated by ~four orders of magnitude — the max CSR
> would carry ~261k entries per class.
>
> The fix is free: `K_max = 1100` gives `K = 1101`, a topk buffer of `1101 × 256 × 4 B ≈ 1.1 MB`.
> **Set `K_max` from the compiled grammar's measured `max_c |N_c|`, with 1,100 as the default**,
> and assert `K > max_c |N_c|` at build time (§4.4's verified bound is tight at `K = |N_c|`).

**Layer 3 — CSR over classes.** `C ≈ 500`, mean post-complement `|S_c| ≈ 50` → `nnz ≈ 25,000`.
int32 indices, **no values array** (all ones). ~100 KB, cache-resident. (The per-class argmax table
of Layer 2b is separately `[C, L] int32` = 512 KB — §2.7.)

**Layer 4 — gather to edges.** `W[i,e] = W_c[class_id[e], i]`. A gather, not a matmul.

Budget for this layer: `2·nnz·L ≈ 1.3e7` FLOPs/step versus `E·V·L ≈ 1.1e13` naive — **~9e5×**.
~1.5 MB memory, ~25.6 MB traffic fp32 → ~9 µs at 3 TB/s. All four figures verified. [V]

> **This budget covers the class layer only.** The tree itself dominates above `|S| ≈ 512` — see
> §0's table. You are launch-bound below that and compute-bound above it, so §7.3's "10% overhead
> means launch-bound" heuristic applies only in the small-automaton regime.

**Do not use interval/trie decomposition over token ids** — Gemma token ids have no useful ordering.
**Do not use bitset-AND-popcount** — you need a *weighted sum*, not set membership.

#### JAX specifics

**Do not use `jax.experimental.sparse`.** Its module docstring, verbatim: *"experimental reference
implementations, and not recommended for use in performance-critical applications. The submodule is
no longer being actively developed."* [V]

The dense toolbox is sufficient and idiomatic. **Store `p` as `[V, L]` column-major** so each
token's 256 marginals are contiguous — this layout decision matters more than the kernel algorithm,
and the gather below assumes it:

```python
g  = p[indices, :]                                       # [V,L] -> [nnz, L]
Wc = jax.ops.segment_sum(g, seg_ids, num_segments=C)     # [C, L]   static num_segments
W  = Wc[class_id]                                        # [E, L]   Layer 4 gather
M  = jnp.zeros((L, S, S)).at[:, si, sj].add(W.T)         # [L,S,S]  scatter to transitions
```

`num_segments` must be **static** under `jit` (a traced value raises `ConcretizationTypeError`) —
bucket it like `|S|`. [V]

**Pallas: not yet.** *"Pallas is experimental and is changing frequently"*, you probably don't need
it (`gather + segment_sum` already fuses; the tree matmuls go to cuBLAS), and there is a specific
hazard: Pallas Triton kernels are not captured into command buffers by default
([jax#27988](https://github.com/jax-ml/jax/issues/27988)), so dropping to Pallas can **cost** you
the CUDA-graph capture that is your reason for choosing JAX. **Only after profiling.** [V]

### 4.5 Minimization

**Hopcroft is the wrong algorithm at `|Σ| = 262,144`** — its bound is `O(|Σ|·n·log n)` because the
splitter worklist holds `(symbol, block)` pairs. At `n = 20k`: dense table = 5.24e9 cells → **21
GB**; Hopcroft **7.5e10** ops; **Valmari `n + m·log₂m` = 3.0e6**. ~25,000×.

Implement **Valmari 2012, IPL 112(6):213–217**
([free PDF](https://www.cs.cmu.edu/~cdm/resources/Valmari2012.pdf)) — ~130 lines, `|Σ|` never
appears in the bound, handles partial DFAs natively, includes trimming. **Pure Python is fine**: at
3.0e6 operations it runs in seconds. Reference port:
`WalkerCodeRanger/dfaMinimizationComparison`. `pynini` (`.determinize().minimize()`, raw token ids
+1) is an alternative if you are on manylinux x86-64.

> **THE COMPLETION TRAP.** Never complete the partial DFA with a sink state — it converts
> `m = 170k` real transitions into `m = n·|Σ| = 5.24e9`. Keep it partial.

Sort transitions by label first (the paper notes this lets you drop a line for `O(n + m log n)`).
Never use `pyformlang` (`np.zeros((n, 262144))` = 5.2 GB), `automata-lib` (5.24e9 empty dicts),
`greenery`, `dk.brics` (char-capped at 65,536), `re2`. `interegular`'s *model* is right but
`crawl()` dedupes with a linear list scan — 22k states over a **39-symbol** alphabet took >30
minutes. The objection is to those libraries' algorithms, not to Python.

**Minimize twice** [D]: on the byte DFA (≤257 symbols, ms, and it directly cuts lift time) and on
the token DFA. Nobody in the literature does the second pass, but the lift is a non-injective
relabeling with path compression: the byte DFA has a state per byte inside every literal, the token
DFA needs states only at token boundaries. `outlines_core` prunes dead states but never merges
equivalent ones. **Measure the reduction; it is unpublished.**

### 4.6 NFA fallback

Determinization blows up on counted repetition (`.*a.{k}` → `2^(k+1)` DFA states [R]), which maps
onto JSON Schema `minLength`/`maxLength`/`minItems`/`maxItems`. Determinize eagerly; fall back to
NFA above a state budget. The paper does exactly this — DFAs everywhere except Spider
(8,796–19,509 states / 85k–172k edges as an NFA) [V].

**There is no "active state set" in this method — do not build one.** That framing is a
per-position projection `π_i(C)`, i.e. the naive masking baseline §2.8 proves wrong. The method
carries `|S|`-dimensional `a`/`b` vectors and full `M_i` regardless of determinism.

NFA-specific costs are exactly three: a larger `|S|`/`|E|`; a start *vector* rather than a point
mass (§5.7); and **the edge-multiplicity-weighted token draw of eq (8)**, which the `∃` fast path
gets wrong (§2.6). §2.2's soft-proxy caveat applies to **sampling only** — `--emission=map` is exact
on NFAs (§2.7).

### 4.7 Compilation throughput — [V-P0] measured, and **not** a scheduling problem

> **The original estimate is retracted.** It extrapolated one published measurement (132–261 s per
> schema at a **151k** vocab [R]) to 262k, giving ~~4–8 minutes per schema~~ and ~~4–7 days
> serially for BFCL-Live's 1,351 instances~~. **Measured on this stack
> (`outlines-core` 0.2.14, Gemma 262k vocab, `scripts/phase0_lift_timing.py`):**
>
> | schema shape | regex→DFA→lift, total | states | nnz |
> |---|---|---|---|
> | 1 string property | **0.126 s** | 25 | 268,397 |
> | typical 3-property call (enum + int) | **0.271 s** | 75 | 268,750 |
> | 8 properties, mixed types | **0.652 s** | 182 | 537,953 |
> | nested object, depth 2 | **0.364 s** | 91 | 537,205 |
> | array of strings | **0.262 s** | 56 | 537,077 |
> | 20-way enum | **0.250 s** | 70 | 268,614 |
> | array of objects | **0.390 s** | 95 | 537,315 |
> | **`{}` wildcard (§4.2's 7-way alternation)** | **16.7 s** | 3,261 | 36,210,539 |
>
> Realistic function-call schemas: **0.13–0.65 s each.** Mean over all eight including the
> wildcard: 2.38 s. **BFCL-Live's 1,351 instances project to ~0.9 hours serially** — minutes on
> this box's 26 cores. Building the `Vocabulary` itself costs a one-off 2.9 s.

Consequences:

1. **The vocabulary pre-filter is unnecessary.** It was billed as the "largest single payoff" and
   ~20 lines; at 0.3 s per schema there is nothing left to win. **Do not build it.** If a future
   grammar set turns out to be wildcard-dominated, revisit — that is the only case with headroom.
2. **Open question 5 is resolved: compilation throughput is not a project risk.**
3. **The one thing to actually watch is `{}` / missing `type` / `additionalProperties: true`**
   (§4.2). It is 26–130× slower than every other shape, produces 3,261 states and 36.2M
   transitions, and is the realistic route to "regex too large". §4.2's fail-loud pre-pass should
   flag it explicitly rather than let it through silently.

Still worth keeping, cheaply:

4. **Parallelize across cores** — embarrassingly parallel, and now the difference between minutes
   and seconds rather than days and hours.
5. **Cache** on `(schema_hash, tokenizer_hash, compiler_version)`. Do not reuse `~/.cache/outlines`
   — its key ignores the tokenizer.
6. Minimize the byte DFA before lifting.
7. Cap `maxLength`/`maxItems` at build time.

### 4.8 Per-task grammars

| Task | Constraint | FA | Paper's states [V] |
|---|---|---|---|
| xLAM / BFCL, JSON | function-signature JSON schema | DFA | 122–385 median, ≤2,459 |
| xLAM / BFCL, Python | `[func(k=v, ...), ...]`, **keyword args only** | DFA | 90–301 median |
| Sudoku 4×4 | format **+ preserves prefilled cells** | DFA | 21 (fixed) |
| Countdown | each step matches `A op B=C` | DFA | 47–77 |
| GSM-Symbolic | symbolic expressions only inside `«…»` | DFA | 56 (single automaton) |
| Spider | SQL grammar restricted to the schema's tables/columns | **NFA** | 8,796–19,509 |

**BFCL's Python format — no canonical grammar exists; you author it.** Parsing is Python's `ast`,
and **only `elem.keywords` are read — positional arguments are silently discarded**, guaranteeing a
`missing_required` failure. **Your grammar must forbid positional args.** [R] Brackets are
auto-added if missing; dotted names supported; scoring lowercases and strips `",./-_*^"`. It is
**not regular** (nested `[] {} ()`, plus `BinOp`/`Lambda` via `eval`). [D] Constrain to a
**bounded-depth** subset: dotted name, keyword-only args, values ∈ {string, number, bool, null,
list, dict}, nesting depth ≤ `d` (start `d=2`). Report the depth bound's coverage against the
ground-truth answers.

---

## 5. Integration with `gemma/diffusion`

### 5.1 The hook surface

`Sampler`/`ChatSampler` are frozen kw-only dataclasses forwarding `diffusion_process`,
`logit_shaper`, `sample_from_predictions`, `early_stop_fn`, `canvas_length`, `max_denoising_steps`
straight to `DiffusionSampler`. Required protocols [V]:

| Hook | Signature |
|---|---|
| `logit_shaper` | `__call__(self, logits: Float['*B L V'], noise_proportion: Float['*B']) -> Float['*B L V']` |
| `sample_from_predictions` | `__call__(self, *, rng, denoiser_logits, canvas, current_noise_proportion, target_noise_proportion) -> Int['*B L']` — **all kw-only** |
| `early_stop_fn` | `should_stop(self, *, step, canvas, previous_canvas, logits) -> Bool['*B']` |

**Only `EarlyStopFn` is a real `typing.Protocol`.** The other two are annotated with *concrete
classes*, so **subclass** them rather than duck-typing — this also keeps
`dataclasses.replace(..., text_vocab_size=...)` working (`SampleFromPredictions.text_vocab_size`
defaults to `0` and is repaired by a `replace` keyed on `== 0`; a custom class lacking the field
raises). [V]

`DiffusionSampler` is `@dataclasses.dataclass(frozen=True, kw_only=True)` and its parent
`SamplerLoop`'s docstring states the contract: *"This class only has static hashable attributes, so
it can be passed to `jax.jit`."* **Your subclass must be frozen, kw-only, and hashable — tuples not
lists, and no `jnp` arrays as attributes.** [V]

`forbidden_tokens` and `sampling` are **inert** on the diffusion path (§1.2). Do not use them.

**Joint sampling is fully permitted.** `sample_from_predictions` receives the entire `[B,L,V]`
logits and `[B,L]` canvas and returns an entire `[B,L]` canvas; nothing downstream assumes
per-position independence.

### 5.2 Where the constraint goes

1. **`logit_shaper`** — output is `shaped_prediction`, consumed by **both** the sampler **and**
   self-conditioning. Put the support mask here if you want `--self-cond=constrained` (§3.7). Use a
   finite sentinel, never `-inf` (§2.4).
2. **`sample_from_predictions`** — where the joint constrained sampler lives. This is the method.
3. **`early_stop_fn.should_stop`** — read-only, and where §3.1b's `A_{k+1} ∩ F ≠ ∅` conjunct goes.
   Under J0 it must be given `emit_canvas`, not the trajectory canvas (§3.1b).

### 5.3 The loop structure that shapes everything

```
jit _sample_loop  (static_argnames=('self',),  max_new_tokens is TRACED)
└── lax.while_loop   (blocks)            carry = SamplingState
    └── _sample_step   (also jit, static self)
        └── sample_next_canvas
            └── lax.while_loop  (denoising steps)   carry = _WhileLoopCarry
```

[V] Three consequences, and they determine the whole design:

**(a) There is no Python between blocks.** §3.5's per-block protocol — advancing `A_k`, recomputing
`b_L` from the remaining budget, running `δ*` over the truncated canvas — **must be jit-traceable
ops on fixed-shape padded arrays.** Ordinary Python `for`/`if` over block index or over automaton
states will not work. `max_new_tokens` is traced, so the block count is not a compile-time constant:
you cannot unroll per block or size an array by it.

**(b) `self` is a `static_argname` on both loops.** The whole sampler — *including your custom
`sample_from_predictions` and `logit_shaper`* — is hashed as a compile-time constant.

> **Anything stored on those objects is baked into the trace.** Per-request automaton arrays held
> as attributes would trigger a **full recompile per distinct automaton**: at 1,351 BFCL schemas ×
> ~5 s that is ~2 hours of pure compilation, and it defeats the compilation cache. **Automaton
> values must be traced; only shapes may be static.** Store static config on the hook objects
> (entropy bound, vocab size, bucket sizes) and nothing else.

**(c) ⚠ CORRECTED [V-P0] — `_sample_step` DOES need forking, and so does the entry point.**

The original claim was that the denoising `while_loop` lives entirely inside `sample_next_canvas`,
so a subclass could widen `_WhileLoopCarry`, replace `body_fn` and add joint constrained sampling
**without touching `_sample_step`**. That half is true and still holds — the *within-block*
machinery is fully encapsulated. What is false is that this suffices for the **per-block** protocol
of §3.5. Two facts, both read from the installed source:

```python
# gemma/diffusion/_sampler.py — the real signature. No `state`.
def sample_next_canvas(self, *, canvas_length, max_denoising_steps, batch_size,
                       cache, params, rng, full_attention_mask) -> Tokens: ...

# gemma/gm/text/_sampler_loop.py — max_new_tokens is a parameter of _sample_loop,
# captured only in cond_fn's closure. It is NOT a field of SamplingState.
def _sample_loop(self, *, params, state, max_new_tokens): ...
def _sample_step(self, state, *, params): ...          # never sees max_new_tokens
```

1. **`sample_next_canvas` receives no `state`.** `A_k`, `state.step` and `predicted_tokens` are all
   unreachable from the place the constrained sampler has to live. The only per-block quantity
   visible there is `cache_layer['end_index']`.
2. **`max_new_tokens` is not in `SamplingState`**, so `R = max_new_tokens − state.step` is not
   computable even inside `_sample_step`.

Since `b_L(s) = 1[d(s) ≤ R]` needs `R`, and `a_start = 1[s ∈ A_k]` needs `A_k`, **both must be
threaded in explicitly**. CLAUDE.md's "minimum fork surface is `sample_next_canvas` plus a widened
`SamplingState`, `_sample_step` does *not* need forking" is therefore **wrong** and should be read
as the corrected version below.

[D] **Recommended structure (revised):**

- Subclass `DiffusionSampler` (frozen, kw-only, hashable fields only).
- **Widen `SamplingState`** with `automaton_state: Bool['B S_bucket']` (`A_k`) and
  `max_new_tokens: Int['']`. It is a `flax.struct.dataclass(kw_only=True)` with nine fields and a
  derived `cache_info` **property** (not a field [V-P0]), so subclassing is mechanical.
- **Fork `_sample_step`** — unavoidable. It must (i) read `A_k` and `R` off the widened state,
  (ii) pass them into a widened `sample_next_canvas`, and (iii) recompute
  `A_{k+1} = δ*(A_k, truncated_canvas)` and write it back. It is ~40 lines and still inherits
  nothing structural from the parent beyond shape.
- **Initialize the widened state at the entry point.** `init_state` is built by
  `_prefill.prefill(...)` inside `gm.text.Sampler.sample`, so either override `sample` to widen the
  prefilled state before calling `sampler.sample(...)`, or — cleaner — override
  `_initialize_sampler_loop` *and* widen in a thin `sample` wrapper. `max_new_tokens` is a
  per-request constant, so it can equally ride in the `params` pytree; `A_k` cannot, because it
  changes every block.
- Override **`sample_next_canvas`** — widened carry (§5.4), constrained sampler, constrained
  emission, `emit_canvas` passed to `should_stop`.
- Thread the automaton itself as **traced arrays**, either as extra `SamplingState` fields or in
  the `params` pytree under a reserved key. **Do not** put them on `self` (§5.3b).
- `A_k` is a traced `[S_bucket]` boolean/float array, **not** a Python set.
- The stateless alternative — recomputing `A_k` from `state.predicted_tokens` each block — **does
  not avoid the fork**, since `predicted_tokens` is also unreachable from `sample_next_canvas`. It
  only avoids the extra *field*. Measure before choosing it over a carried `A_k`.

Minor hazard: `predicted_tokens.at[:, jnp.arange(256) + state.step].set(canvas)` can index past
`max_out_length` on the final block; JAX silently drops out-of-bounds scatter indices, so overflow
truncates rather than errors. [V]

### 5.4 The J0 carry

```python
@flax.struct.dataclass
class _ConstrainedCarry:
    step: Int['']
    canvas: Tokens            # trajectory — stock uniform renoising, feeds the model
    emit_canvas: Tokens       # constrained MAP or draw — what actually gets emitted
    sc_embeddings: Embeddings
    rng: PRNGKey
    done: Bool['B']
```

`sample_next_canvas` returns `final_carry.emit_canvas`; `_sample_step` (unmodified) truncates that
and writes it to the cache and `predicted_tokens`.

**Build J1 first.** It needs no carry change — only a replacement `sample_from_predictions` plus
§5.3's automaton threading — and it is the fastest path to an end-to-end constrained generation,
validating the whole inference stack before you touch the carry.

### 5.5 Shape bucketing

`|S|` varies from 21 (Sudoku) to 19,509 (Spider), and JAX needs static shapes. Measured: an
unrolled 8-level tree over `[256,S,S]` compiles in 3.2–5.4 s for `S ∈ {64,256,1024}`, and **the HLO
is 2,878 lines in all three cases** — the program structure is fixed, only shapes change. You pay
per *bucket*, not per size. [V]

[D] Bucket `|S|` to powers of two, padding with an absorbing dead state. **Derive the ladder from
the tree memory budget, do not hardcode it** — at 8 GB the top usable bucket is 1024 (2048 needs
8.59 GB and can never be dispatched to; 4096 needs 34.3 GB, and AOT-warming it would try to compile
and allocate a 34 GB program). Anything above the top bucket goes to the chain path (§5.6). Warm all
usable buckets at startup with AOT `jax.jit(f).lower(...).compile()`, then they are free via
`jax_compilation_cache_dir`.

**Shape polymorphism does not help.** `jax.export.symbolic_shape` traces and lowers once but is
*"re-compiled on demand for each concrete input shape"* — confirmed empirically. It saves tracing
and lowering, not XLA compilation. Also the scan axis `L` **cannot** be symbolic
(`NotImplementedError: associative scan over axis of non-constant size`) — moot at `L=256`. [V]

**`donate_argnums` is narrower than it looks.** Input/output aliasing requires an output with
**identical shape, dtype and layout**. The tree builder is `[L,S,S] → [2L−1,S,S]`, so donation is
silently ignored (XLA warns "Some donated buffers were not usable" and allocates anyway). It helps
only where shapes match — e.g. an in-place `M_i` normalization `[L,S,S] → [L,S,S]`. And **never
donate a buffer you feed to the AOT warm-up**: the warm-up call deletes it
(`ValueError: Buffer has been deleted or donated`). Warm with a throwaway `jnp.zeros` of the bucket
shape. [V]

### 5.6 Memory and dispatch

| Tensor | Size at `B=1` |
|---|---|
| logits / `shaped_prediction` `[1,256,262144]` fp32 | 268 MB (already allocated by stock) |
| `p` (softmax) | 268 MB (stock) |
| `q` constrained marginals | 268 MB (**new**; needed under `--confidence=mar`, `--self-cond=constrained`, **and `--trajectory=R1`**) |
| tree nodes `(2L−1, S, S)` fp32 | `\|S\|=385` → **303 MB** · `1000` → **2.044 GB** · `2459` → **12.359 GB** · `19509` → **777.9 GB** |
| class tables (`W_c^Σ`, `W_c^max`, argmax) + `class_id[E]` | ~2 MB |

(Tree row is `(2·256−1)·|S|²·4 B`. **Do not drop the factor 2** — it flips the dispatch decision.
All cells verified. [V])

[D] **Dispatch rule.** Evaluate `B·(2L−1)·S_max²·4 B` — **per batch, not per request**, since padded
batching makes it `B·S_max²`. At an 8 GB budget and `B=1` the threshold is **`|S| = 1,978`**
(decimal GB; **2,050** if you mean GiB — state which). So the paper's largest BFCL DFA (2,459
states, 12.36 GB) **falls back to the chain sampler**, i.e. the +114% path.

> **[V-P0] The 8 GB figure is not what this machine has — it has more.** Measured on the H100 80GB
> at `MEM_FRACTION=.90`: allocator limit **76.52 GB**, params resident **51.65 GB**, generation
> peak **56.5 GB** ⇒ **~20 GB genuinely free** for the tree at `B=1`. That lifts the memory-side
> threshold from `|S| = 1,978` to **`|S| ≈ 3,127`**, which would bring the paper's largest BFCL DFA
> (2,459 states, 12.36 GB) **inside** the tree path rather than falling back to the chain.
> Recompute the ladder from the measured free HBM, not from the 8 GB placeholder — but see the
> compute caveat immediately below, which is the binding constraint long before this one.

> **But memory is the wrong binding constraint at that size.** Per §0's table, the tree's arithmetic
> at the memory threshold (`|S| ≈ 2,000`) is already **~390% of a full model forward per denoising
> step** — you hit the compute wall roughly 4× over before you hit the 8 GB memory wall. Set the
> dispatch threshold from *both*, and expect the compute bound to dominate above `|S| ≈ 512`.

Spider's 19,509 states need **778 GB** for the tree and **389 GB** for the `M_i` leaves alone: the
dense formulation is structurally infeasible there, and the dynamax operator (§2.6) does **not**
rescue it because the forward–backward pass is still `S×S`.

**The paper never discusses this crossover** [?], and never says how it handled `|S| ≈ 20,000`.
**Resolving that is the highest-value open question in this spec.** Log which path was taken per
request and report the split.

### 5.7 Start vectors are vectors, not point masses

`a_start = 1[s = s₀]` only holds for a DFA on block 0. After the first block boundary — and always
for an NFA — the start is a *set* `A_k`, so `a_start` is a general vector and the tree's root draw
must be joint over `(s_0, s_L)`. Verified exact with a general `a_start` and a budget-aware `b_L`
(error 8.3e-17). [V] Hardcoding a point mass is a bug that first appears on block 2 or on Spider.

### 5.8 Do not build a PyTorch/JAX hybrid

Same-GPU DLPack is zero-copy so the 268 MB is not the issue; three other things are: (1) allocator
war — JAX preallocates ~75% of VRAM, disabling it puts JAX on its slower on-demand allocator and
fragments against PyTorch's caching allocator; (2) **it destroys the thing you came for** — a
framework boundary forces a sync per denoising step, and you would insert 48 hard syncs into a
program whose bottleneck is sync overhead; (3) two compilers, two profilers, two debuggers.

API notes: `jax.dlpack.to_dlpack` **no longer exists**; use `jnp.from_dlpack(x)`. [V]

---

## 6. Testing — correctness before speed

**The exactness harness is the most valuable artifact in this repo.** Build it in Phase 2, before
any GPU work. A sampler that is subtly wrong still produces *valid strings*, so nothing else will
catch it.

### 6.1 Brute-force exactness (the bedrock)

```
V = 4, L ∈ {4, 6, 8}, random small DFAs AND NFAs (3–8 states, incl. parallel overlapping edges)
  → enumerate all V^L sequences, filter, compute the exact path-weighted posterior in float64
```

**Fixed seeds per instance, and a Bonferroni-corrected threshold on every χ² test.** This suite has
dozens of independent statistical tests per run; an uncorrected `p > 0.001` will go red on its own,
and CLAUDE.md says never to let it. Document a re-run-with-a-different-seed protocol.

1. `Z` from forward–backward == `Z` from enumeration (rtol 1e-10).
2. `q_i` from eq (6) == the exact marginal (rtol 1e-10); `Σ_v q_i(v) == 1`; `a_i·b_i` `i`-invariant.
3. Chain sampler: χ² against the exact posterior.
4. **Tree sampler == chain sampler in distribution**, on NFAs as well as DFAs. χ² only — they draw a
   different number of variates, so do *not* assert seed-identity. **This is the test that catches
   an `∃`-form eq (8)** (§2.6).
5. `constrained_map` == `argmax_x p*(x)` by exhaustive search, 1,000 instances, **NFAs included**.
6. Every sample and every MAP output accepted by an independent simulator.
7. Degenerate cases: `|C| = 1`; `C = V^L` (must recover unconstrained sampling exactly); `C = ∅`
   (must raise); unreachable and non-co-accessible states; `L=1`; **`L` not a power of two — assert
   the identity-padding path of §2.6(d), with `b_L` at the true `L`**.
8. Non-point-mass start vectors and NFA start sets.
9. Budget-aware terminal factors `1[d(s) ≤ R]` for several `R`, including `R` too small (must yield
   `Z = 0` and raise, not silently emit).
10. **`reverse=True` operand order** — assert the suffix scan equals `M[k] @ … @ M[L-1]`. Fails
    silently otherwise (§2.6(c)).
11. **The complement-aware path** — the one thing §2.4 says only this suite can catch. Force
    negated classes by setting the polarity threshold to `|S_c| > 2` at `V=4`; assert `q_i` from
    `r_i(v)` equals `q_i` from a direct scatter *and* from enumeration, with mixed Pos/Neg classes
    in one automaton, an empty `N_c` (`S_c = V`), and an empty `S_c`. Do the same for `W_c^max` /
    the topk path (including `K = |N_c|` failing and `K = |N_c|+1` succeeding), and for §2.7's
    `segment_min` tie-break including ties and empty segments.

### 6.2 Reference vs optimized

`infer/reference.py` in plain float64 numpy, no cleverness. Every optimized path (segment-sum class
SpMM, tree scan, max-plus, NFA, dynamax operator) differential-tested against it on random automata
up to `|S|=64, |E|=512, L=32`.

### 6.3 Numerical stability

No NaN/Inf and `Z > 0` at `L = 256` with `p ~ softmax(N(0,1)·10)`. Deliberately test the
unnormalized path to *confirm it underflows* — this documents why the scaling exists. Separately,
assert no `NaN` anywhere with a `q` containing exact zeros (§2.4's clamp rule).

### 6.4 Integration

- **`test_guarantee.py` — two distinct assertions; conflating them is a trap.** A non-final block's
  canvas ends in a live-but-not-accepting state and is **not** accepted on its own.

  ```
  per block k:  δ*(A_k, canvas_k) ≠ ∅   and   ⊆ {s : d(s) ≤ R}
  at end:       simulator.accepts(concat(canvas_0 .. canvas_K))     # this is CS
  ```

- **No-uniform-random assertion.** Under J1/J2, assert the emitted canvas contains no position from
  the `random_tokens` branch. Test it with a directly-constructed `DiffusionSampler` (whose
  `early_stop_fn` default is `NoEarlyStop`, so it *always* exits via the budget path and *always*
  exposes this) as well as with `ChatSampler`.
- Unconstrained-equivalence: with a trivial `Σ*` automaton, the constrained sampler must match
  stock. **Token-identity needs an explicit RNG contract** — when `|S| = 1`, the constrained sampler
  must reduce to a single `jax.random.categorical` on the same tensor with the same key, or it
  consumes the key stream differently and every subsequent step diverges. Honour that or downgrade
  to "step-1 emission identical + distributional equivalence thereafter".
- Cross-block: a grammar that cannot complete in 256 tokens threads across ≥3 canvases with
  `A_k ≠ ∅` at every boundary.
- Stop-token/pad: assert the automaton accepts `_truncate_canvas_at_stop_tokens`'s output for **all**
  of `end_tokens`.
- **No recompilation per request.** Assert the jit cache size is constant across 50 different
  automata in the same `|S|` bucket. This is §5.3(b) and it is invisible otherwise.
- **Kernel-launch count per denoising step must be `O(log L)`, not `O(L)`.** Profile it.

---

## 7. Evaluation

### 7.1 Datasets

| Benchmark | Source | Notes |
|---|---|---|
| BFCL v4 | `github.com/ShishirPatil/gorilla`, `berkeley-function-call-leaderboard`, `pip install -e .` | **Data in git, not HF.** HF mirror is v3 only. **v4 renamed `simple`→`simple_python`.** Non-Live 1,150 / Live 1,351. |
| xLAM | `Salesforce/xlam-function-calling-60k` | **GATED** — needs `HF_TOKEN` + form. `tools`/`answers` are JSON-encoded strings. |
| Spider | `xlangai/spider` | **HF is parquet only — no SQLite DBs, no `tables.json`.** Pull the Yale archive; use `taoyds/test-suite-sql-eval` (`--etype exec --plug_value --keep_distinct`). |
| GSM-Symbolic | `apple/GSM-Symbolic`, config `main` | 5,000 rows. **cc-by-nc-nd-4.0** — NoDerivatives. No shipped scorer; 8-shot, regex the final number. |
| Countdown | `Jiayi-Pan/Countdown-Tasks-3to4` | TinyZero's actual source. Test slice `ds.select(range(327680, 328704))` (1,024). **License unspecified.** |
| Sudoku 4×4 | **no canonical source** | Sakana's `Sudoku-Bench` is 401/discontinued. **Generate synthetically** — 288 completed grids, mask cells, verify uniqueness by exhaustive solve. Fully reproducible. |

Paper's counts [V]: xLAM 1,000 · BFCL-NL 1,000 · BFCL-Live 1,351 · Sudoku 900 · Countdown 993 ·
Spider 20 schemas.

### 7.2 Baselines

1. Stock DiffusionGemma, unconstrained. Accuracy **and CS**, plus the §1.4 budget-path frequency.
2. **Naive per-position masking** — mask `shaped_prediction` to `π_i(C)` and sample per position.
   The wrong-but-obvious baseline §2.8 targets; its CS is the headline evidence.
3. **J2** — constrained joint sample at accepted positions, uniform random elsewhere. The gap
   between J2 and J0 quantifies the §3.1 finding.
4. **J0-map** (paper's greedy analogue, with `--temp-schedule=near-greedy` for the closest `T=0`).
5. **J0-sample** and **J1** (paper's sampling analogue).
6. Chain vs tree — identical accuracy, different latency (§7.3).

### 7.3 Overhead measurement

Mirror Table 3: 200 BFCL examples, `B=1` latency, throughput at `B ∈ {1,2,4}`, mean ± std over 3
seeds. **Report compile time separately from steady-state latency**, and state whether the
compilation cache was warm — otherwise the JAX numbers are not comparable to the paper's.

Paper's numbers, A6000 + Dream-7B: unconstrained 24.26 s, chain 51.92 s (**+114%**), tree 25.35 s
(**+4%**). [V]

> **Read the +4% as a small-`|S|` result.** Per §0, the tree is 2.9% of a model forward at
> `|S|=385` but 54% at `|S|=1024`. High overhead means launch-bound **only below `|S| ≈ 512`** —
> above it you are compute-bound and chasing kernel counts is wasted effort. Report the crossover
> you measure, and bucket the overhead table by `|S|`.

### 7.4 Ablations

| Ablation | Values | Paper's finding |
|---|---|---|
| Confidence signal | `mf` / `mar` | **Mar > Mf**, largest gap on Python format [V] |
| Sampler | chain / tree / dynamax-operator | equal accuracy; 114% vs 4% [V]; third is novel |
| Variant × emission | J0-map / J0-sample / J1 / J2 | **novel — §3.1 is the port's core finding** |
| Renoising | R0 / R1 / R2 | **no analogue — novel** |
| Self-conditioning | unconstrained / constrained | **no analogue — novel** |
| Temperature schedule | default / near-greedy (`1e-12`) | closest analogue of the paper's T=0/T=1 |
| Entropy bound | calibration sweep | **novel and necessary** (§3.4) |
| Denoising steps | 48 → 24 → 12 → 6 | Figure 4: baseline degrades sharply, constrained stays flat [V]. **DiffusionGemma already runs at 48 vs the paper's 64–128, so this is the most relevant scaling result for this port.** |
| DFA vs NFA | Spider only | sampling is a soft proxy; MAP exact |

### 7.5 Expected result shape

The paper's gains split cleanly; expect the same [V]:

- **Format-bound tasks** (function calling, Sudoku, Countdown): huge. Dream BFCL-Live Python greedy
  22.4 → 69.7; Sudoku 36.4 → 92.4; Countdown 53.6 → **100.0**.
- **Reasoning-bound tasks** (Spider, GSM-Symbolic): near-zero greedy gain (52.7 → 53.0,
  39.9 → 40.2). The constraint fixes format, not reasoning. **Do not expect otherwise and do not
  tune toward it.**
- **The sampling regime is where everything happens.** Unconstrained dLLMs collapse under `T=1`
  (Dream BFCL-Live JSON 63.9 → 22.3); constrained holds (71.5 → 69.0).

DiffusionGemma is far stronger and more recent, so its unconstrained baseline will be higher and
*relative* gains smaller. **The CS column is what transfers. Lead with it** — and with the J2-vs-J0
gap, which is specific to this model family and is the port's own contribution.

---

## 8. Phase plan

Each phase ends with a commit, a green suite, and a note in `docs/`. Do not start `n+1` until `n` is
green.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0** | `docs/ENV.md`, `docs/PHASE0_FINDINGS.md`, this spec corrected in place | Every §1.2 box ticked or corrected. Model loads from Orbax; smoke test passes; §1.4 baseline captured including per-block **early-stop-vs-budget exit** and non-accepted count on the final step. |
| **1** | `compile/` — schema→regex→byte DFA→token FA→minimize→classes (**two tables, independent polarity**)→bucketed artifact | Compiles all 6 task grammars. Round-trip: 1,000 sampled strings per FA accepted by an independent validator, **and** 1,000 ground-truth answers per dataset accepted by their FA. Report class-dedup ratio, post-lift minimization ratio, and the §4.7 vocab-prefilter speedup. |
| **2** | `infer/reference.py` — float64 numpy chain + FB + Viterbi + a reference (non-parallel) tree | **§6.1 fully green**, including tests 4 (NFA tree==chain), 7 (padding), 10 (reverse order) and 11 (complement + topk + tie-break). The bedrock. |
| **3** | `infer/` JAX — `segment_sum` class SpMM, **Blelloch** up-sweep retaining dyadic products + down-sweep for `a`/`b`, max-plus tree, dynamax-operator prototype, NFA path, bucket dispatcher | Differential-tested vs Phase 2. Microbenchmarks: latency vs `\|S\|`, **kernel launches per step (must be `O(log L)`)**, compile time per bucket, and the measured **compute/launch crossover** as well as the memory crossover. |
| **4** | `model/` — J1 first, then J0 (`_ConstrainedCarry` + forked `sample_next_canvas`), custom `logit_shaper` and `EarlyStopFn`, traced automaton threading | §6.4 green, **including the no-recompilation-per-request assertion**. End-to-end constrained generation on 10 prompts. Diagnostics: stop-token position per block, marker position per block, non-accepted count on the final step. |
| **5** | `eval/` — 6 benchmarks × 6 baselines × ablations | Full tables. **CS = 100% on every J0/J1 run, asserted not assumed.** J2's CS reported as the contrast. |
| **6** | `docs/RESULTS.md` | Every table filled; every `[?]` resolved or explicitly left open with a reason. |

```
diffgemma_fa/
  compile/   schema.py lift.py minimize.py classes.py automaton.py tasks/
  infer/     reference.py tree.py scans.py viterbi.py marginals.py dynamax_op.py
             dispatch.py buckets.py
  model/     sampler.py shaper.py early_stop.py carry.py blocks.py
  eval/      run.py ablate.py runners/ metrics.py
  tests/
  docs/      ENV.md PHASE0_FINDINGS.md LOG.md RESULTS.md
```

---

## 9. Open questions, ranked by risk

0a. **§5.6 — how does the paper handle `|S| ≈ 20,000`?** A dense tree is 778 GB and the `M_i` leaves
   alone are 389 GB. The dynamax operator does **not** rescue this (it only makes backward *sampling*
   `O(S)`; forward–backward is still `S×S`). The paper never says. This may invalidate the dense
   design regardless of framework. **Highest value to resolve.**

0b. **§0 / §5.6 — where is the compute/launch crossover on your hardware?** The tree is 2.9% of a
   model forward at `|S|=385` and 54% at `|S|=1024`. The paper's +4% is a small-automaton result,
   and the dispatch threshold should be set from compute, not just memory.

0c. **§3.5 trap 4 — joint decode termination bias.** If the post-stop tail is scored, the model may
   never terminate. The `ACC --Σ--> ACC` fix is believed correct but untested. Diagnose by logging
   stop-token position per block from the first end-to-end run.

1. **§3.4 entropy recalibration.** Constrained marginals are far sharper; the stock defaults are
   calibrated for unconstrained entropies.
2. **§3.1 J0 vs J1.** J1 gives a stronger invariant (every canvas ∈ C) but shifts the model's inputs
   off-distribution. Genuinely open.
3. **§3.3 renoising R0/R1/R2** and **§3.7 self-conditioning feedback** — both novel, both entangled
   with (2).
4. **§5.3 — carried `A_k` vs stateless recompute from `predicted_tokens`.** The stateless route
   avoids widening `SamplingState` but costs a `lax.scan` over the committed prefix per block.
   Measure.
5. ~~**§4.7 compilation throughput.**~~ **RESOLVED [V-P0], not a risk.** Measured 0.13–0.65 s per
   realistic schema (16.7 s for the `{}` wildcard); BFCL-Live projects to **~0.9 h serially**, not
   4–7 days. The vocabulary pre-filter is unnecessary and should not be built. The residual watch
   item is the wildcard schema shape, not throughput.
6. **§2.7 greedy semantics.** The paper is ambiguous; this spec commits to max-plus MAP. Compare
   against per-position constrained argmax and report the gap.
7. ~~**§1.2 — how often does the stock model exit via the budget path?**~~ **RESOLVED [V-P0], and
   the question was mis-framed.** Measured over 31 blocks with `ChatSampler` defaults: **0/31 exit
   via the budget, 31/31 via early stop**, converging in 3–31 of the 48 available steps. But the
   budget path is *not* the gate on random tokens — §3.1's corrected reading shows early stopping
   emits the fresh sample including its own unaccepted positions, and **1 of 31 blocks emitted a
   random token anyway**. The unpublished facts to report are both of those, plus the median ~12
   executed steps, which undercuts §7.4's `48→24` ablation rungs.
8. **§3.8 BFCL irrelevance** — 45% of the single-turn benchmark; the paper's handling is unknown.
9. **§4.5** post-lift minimization ratio and **§4.4** label-class dedup ratio — unpublished, cheap,
   citable.

---

## 10. Primary sources

**Implementation** — [`google-deepmind/gemma`](https://github.com/google-deepmind/gemma) ·
[`gemma/diffusion/_sampler.py`](https://raw.githubusercontent.com/google-deepmind/gemma/main/gemma/diffusion/_sampler.py) ·
[`_transformer.py`](https://raw.githubusercontent.com/google-deepmind/gemma/main/gemma/diffusion/_transformer.py) ·
[`_early_stopping.py`](https://raw.githubusercontent.com/google-deepmind/gemma/main/gemma/diffusion/_early_stopping.py) ·
[`_chat_sampler.py`](https://raw.githubusercontent.com/google-deepmind/gemma/main/gemma/diffusion/_chat_sampler.py) ·
`gemma/gm/text/_sampler_loop.py` ·
checkpoint `gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it`

**Paper & model** — [arXiv:2607.07026](https://arxiv.org/abs/2607.07026) ·
[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) ·
[Google: Diffusion Explained](https://ai.google.dev/gemma/docs/diffusiongemma/explained) ·
[vLLM blog](https://vllm-project.github.io/2026/06/10/diffusion-gemma.html) ·
[developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/)

**JAX** — [`lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html) ·
[jax#10599 (scan depth/work tradeoff)](https://github.com/jax-ml/jax/discussions/10599) ·
[jax#27988 (Pallas + command buffers)](https://github.com/jax-ml/jax/issues/27988) ·
[`jax.experimental.sparse` (unmaintained)](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) ·
[shape polymorphism](https://docs.jax.dev/en/latest/export/shape_poly.html) ·
[XLA `debug_options_flags.cc`](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc) ·
[NVIDIA JAX-Toolbox GPU perf](https://docs.nvidia.com/jax-toolbox/performance-profiling/gpu-performance) ·
[dynamax `parallel_inference.py`](https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/parallel_inference.py) ·
[Hassan, Särkkä & García-Fernández, arXiv:2102.05743](https://arxiv.org/abs/2102.05743)

**Compilation** — [outlines-core](https://github.com/dottxt-ai/outlines-core) ·
[XGrammar](https://github.com/mlc-ai/xgrammar) ·
[llguidance](https://github.com/guidance-ai/llguidance) ·
[Valmari 2012](https://www.cs.cmu.edu/~cdm/resources/Valmari2012.pdf) ·
[Valmari–Lehtinen 2008](https://arxiv.org/abs/0802.2826) ·
[JSONSchemaBench](https://arxiv.org/abs/2501.10868) ·
[Kuchnik et al.](https://arxiv.org/abs/2407.08103)

**Benchmarks** — [gorilla/BFCL](https://github.com/ShishirPatil/gorilla) ·
[xlangai/spider](https://huggingface.co/datasets/xlangai/spider) ·
[apple/GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) ·
[Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
