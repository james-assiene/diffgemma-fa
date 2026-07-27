# docs/LOG.md — running record

Newest last. One entry per working session.

---

## 2026-07-27 — Phase 0

Started from a bare repo: `SPEC.md` + `CLAUDE.md` + `env.sh`, empty `docs/`, empty venv.

### Environment

- Machine is **dedicated** (`DGFA_DEDICATED=1`, `DGFA_MAX_PHASE=6`): H100 80GB HBM3, idle at start,
  26 threads, 221 GB RAM, 2.7 TB free. CLAUDE.md's shared-box section does not apply here and
  `env.sh` correctly sets preallocation **on** — recorded in `docs/ENV.md`.
- Installed into the project venv: `jax[cuda12]` 0.11.0, `gemma` 4.1.0 from git main,
  `outlines-core` 0.2.14. SPEC §1.1 confirmed — the git install ships `gemma/diffusion/`, PyPI
  4.0.1 does not.
- **New dependency note (per CLAUDE.md's rule):** no dependencies were added beyond the intended
  stack. `gemma` drags in a very large transitive tree including two TensorFlow distributions
  (`tensorflow` 2.20.0 and `tensorflow-cpu` 2.21.0) side by side. Nothing in the diffusion path
  imports TF and nothing has broken; noted in `docs/ENV.md` in case it does.
- Checkpoint pulled to `artifacts/ckpt/` — **37.63 GiB**, not SPEC's ~47.4 GB. ~2.5 min at
  415 MiB/s, `commit_success.txt` present, no errors. `logs/ckpt_download.log`, PID 6522 (finished).

### Work done

Read `gemma/diffusion/{_sampler,_early_stopping,_transformer,_chat_sampler,_models,_paths}.py`,
`gemma/gm/text/{_sampler_loop,_sampler,_template}.py` in the installed tree. Ran five scripts under
`scripts/` (see `docs/PHASE0_FINDINGS.md` for the index). Background runs:

| PID | job | log |
|---|---|---|
| 6522 | checkpoint download | `logs/ckpt_download.log` |
| 11699 | 2-prompt smoke probe | `logs/phase0_smoke_probe.log` |
| 13610 | 20-prompt §1.4 baseline | `logs/phase0_smoke.log` |
| 14391 | 3-prompt multi-block probe | `logs/phase0_multiblock.log` |

All finished; none left running.

### `[?]` resolved by measurement

- **Open question 7 (budget-path frequency) — CLOSED, and mis-framed.** 31/31 blocks exit via early
  stop, 0/31 via the 48-step budget; median ~12 executed steps. But random tokens still reached the
  emission in 1/31 blocks, because early stop does not gate the emitted canvas. See below.
- **Open question 5 (compilation throughput) — CLOSED.** 0.13–0.65 s per realistic schema, not
  4–8 min. BFCL-Live projects to ~0.9 h serially. The vocabulary pre-filter is unnecessary; do not
  build it.
- **§1.3(4) thought marker — RESOLVED.** `<|channel>` = 100 and `<channel|>` = 101 *are* single
  dedicated token ids. §3.6's two-state construction survives on different tokens than assumed.
- **§1.2 `cache_info` — RESOLVED.** It is a `@property`, not a field.
- **§1.3(1) vocab padding — RESOLVED, and it does not exist.** 262,144 real pieces, no dead tail.

### Corrections made to SPEC.md (in place, marked `[V-P0]`)

§1.1 checkpoint size and HBM residency · §1.2 whole checklist ticked, plus four new findings ·
§1.3 all four items (two were wrong) · §1.4 measured baseline table · §3.1 **the emission invariant
was false** · §3.1b a fourth termination path (cache exhaustion) · §3.6 rewritten against the real
channel format · §4.3 state-id stride 64 not 8 · §4.4 `K_max` 256 → ~1,100 · §4.7 **retracted, wrong
by ~1,000×** · §5.3(c) **`_sample_step` does need forking** · §5.6 measured memory headroom ·
§9 open questions 5 and 7 closed.

### The two that change design decisions

1. **§3.1's invariant is false.** `should_stop` is computed from `previous_canvas` and `logits`, so
   it certifies the step's *input*; the emitted canvas is gated on the *old* `carry.done`, so the
   step where early stop fires still emits its own fresh sample including unaccepted positions.
   Measured: 1 of 31 blocks emitted a uniform random token despite early-stopping. J0 is still the
   right answer, but the J2→J0 gap on this model will be **small** — do not oversell it.
2. **`_sample_step` must be forked.** `sample_next_canvas` never receives `state`, and
   `max_new_tokens` is not a field of `SamplingState` at all — it lives only in `_sample_loop`'s
   closure. So neither `A_k` nor `R` can reach the constrained sampler without forking
   `_sample_step` and widening the state at the entry point.

**`CLAUDE.md` was amended for (2)** — its "Hard invariant" paragraph asserted the opposite, and a
future session following it would design into a dead end. The inference-only invariant itself is
untouched; only the fork-surface sentence changed, marked with a Phase-0 note.

### Not done — do not assume coverage

No benchmark data downloaded (BFCL/xLAM/Spider/GSM-Symbolic/Countdown/Sudoku all untouched; the
§4.7 timings use hand-written schemas). Everything ran at `B=1`. `near-greedy` temperature never
executed. Only 11 multi-block blocks observed. No automaton compiled, no tree built, no constrained
generation. §4.2's outlines limitations were not re-verified.

### Next

Phase 1 — `compile/`. First task should be downloading real BFCL schemas and re-measuring §4.7's
projection against them before quoting the 0.9 h figure.
