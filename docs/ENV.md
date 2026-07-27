# docs/ENV.md — machine, stack, and configuration

Recorded 2026-07-27, Phase 0. Required by SPEC §1.1.

## Hardware

| | |
|---|---|
| Accelerator | 1 × **NVIDIA H100 80GB HBM3** (81,559 MiB), driver 580.105.08, CUDA 13.0 |
| GPU visible to JAX | `CudaDevice(id=0)` — pinned by `CUDA_VISIBLE_DEVICES=0` |
| CPU | Intel Xeon Platinum 8480+, 13 cores / 26 threads (`nproc` = 26) |
| System RAM | 221 GB |
| Disk | 2.7 TB total, 2.6 TB free after the checkpoint |

**The box is dedicated.** `nvidia-smi` at Phase 0 start showed 0 MiB used and no
processes. `env.sh` sets `DGFA_DEDICATED=1`.

## This machine is NOT the shared box CLAUDE.md describes

CLAUDE.md's *"Running alongside other experiments"* section is explicitly scoped
to `DGFA_DEDICATED=0`. Here `DGFA_DEDICATED=1`, and `env.sh` accordingly sets the
**opposite** of what that section mandates:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=true     # CLAUDE.md's shared-box rule says never do this
XLA_PYTHON_CLIENT_MEM_FRACTION=.90     # shared-box rule caps at .60
```

This is correct and intentional: `env.sh` states preallocation is *"the only
configuration in which SPEC §7.3's overhead numbers are publishable"*.
Consequently:

- **SPEC §1.1's dedicated-machine assumption holds. There is no override to record.**
- §7.3 latency numbers measured here are **not** subject to CLAUDE.md's
  "shared box, indicative only" labelling requirement. They run on the fast
  preallocating allocator with no fragmentation against other tenants.
- `DGFA_MAX_PHASE=6`, so all phases including eval are in scope on this machine.

Measured allocator limit: **76.52 GB** (`bytes_limit` from
`device.memory_stats()`), i.e. 0.90 × 81,559 MiB, as configured.

## Software stack

| Package | Version | Note |
|---|---|---|
| Python | 3.12.13 | SPEC requires ≥3.12 |
| `jax` / `jaxlib` | 0.11.0 / 0.11.0 | installed as `jax[cuda12]` |
| `gemma` | **4.1.0**, from git main | ✅ SPEC §1.1 confirmed: PyPI is 4.0.1 and lacks `gemma/diffusion/`; the git install declares 4.1.0 and **does** ship `gemma/diffusion/` |
| `flax` | 0.12.8 | |
| `orbax-checkpoint` | 0.12.1 | |
| `numpy` | 2.5.1 | |
| `outlines-core` | 0.2.14 | SPEC §4.3 requires ≥0.2.14 for `IncompatibleVocabulary` |
| `sentencepiece` | 0.2.2 | pulled in by `gemma` |

Everything is in the project venv at `$PROJECT_DIR/.venv`. Nothing was installed
system-wide and no `sudo pip` was used.

`gemma` pulls a large transitive tree (TensorFlow, Keras, kauldron, xmanager,
google-cloud-*). Two TensorFlow distributions land side by side
(`tensorflow 2.20.0` and `tensorflow-cpu 2.21.0`); harmless so far — nothing in
the diffusion path imports TF — but noted in case it bites later.

## Compilation cache

`env.sh` exports `JAX_COMPILATION_CACHE_DIR=$PROJECT_DIR/.jax_cache`. Scripts
additionally set it in-process, as CLAUDE.md's day-one line requires — but
pointed at the **project** directory, not the shared `~/.cache/jax`:

```python
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")
```

## Checkpoint

```
gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it
  -> artifacts/ckpt/diffusiongemma-26B-A4B-it
```

- Publicly readable anonymously ✅ (no credentials needed; `gsutil` warns about
  missing Google auth and proceeds).
- **Size on disk: 37.63 GiB (40.4 GB), not SPEC §1.1's "~47.4 GB".**
  Downloaded in ~2.5 min at ~415 MiB/s over 32 objects; `commit_success.txt`
  present, no errors in `logs/ckpt_download.log`.
- Orbax OCDBT layout as described (`manifest.ocdbt`, `ocdbt.process_0/`, `d/`).
- **Resident size once loaded: 51.65 GB of HBM**, notably larger than the
  on-disk figure. Peak during generation: **56.5 GB** of the 76.52 GB limit.
- `gm.ckpts.load_params(<local path>)` works directly on the local directory;
  the `diffusion.CheckpointPath` enum value (a `gs://` URL) is not required.
- Load time: **36.2 s** from local disk.

### Headroom for the tree (SPEC §5.6)

76.52 GB limit − 56.5 GB peak ⇒ **~20 GB** genuinely free for the automaton
tree at `B=1`. That is *more* than SPEC §5.6's assumed 8 GB budget, which moves
the memory-side dispatch threshold up:

| budget | `\|S\|` threshold from `(2L−1)·S²·4 B` |
|---|---|
| SPEC's assumed 8 GB | 1,978 |
| **measured ~20 GB here** | **~3,127** |

SPEC §5.6 already warns that memory is the *wrong* binding constraint at that
size — the compute wall (§0) arrives near `|S| ≈ 512`. Phase 3 measures the
real crossover on this H100; do not set the dispatch threshold from this table
alone.

## Reproducing

```sh
source env.sh          # venv, CUDA_VISIBLE_DEVICES, prealloc, cache dirs
pytest tests/ -x -q
```
