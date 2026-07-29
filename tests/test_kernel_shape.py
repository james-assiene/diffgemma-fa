"""Compiled-program shape assertions. SPEC §0, §5.5, §6.4.

SPEC §6.4 lists this as a Phase 3 exit criterion:

> **Kernel-launch count per denoising step must be `O(log L)`, not `O(L)`.**
> Profile it.

and CLAUDE.md is emphatic that the detection must be by *counting*, not by
reading code — a `lax.scan` looks perfectly reasonable in source and compiles to
a device `while` that escapes CUDA-graph capture. `WHILE` is **absent** from
XLA:GPU's default `xla_gpu_enable_command_buffer` set
(`{FUSION, CUBLAS, CUBLASLT, CUDNN, CUSTOM_CALL, CONDITIONAL, DYNAMIC_SLICE_FUSION}`),
which is the mechanism behind the paper's +114%.

These assertions run on the optimized HLO, so they are deterministic and hold on
CPU as well as GPU — no profiler, no flakiness.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from diffgemma_fa.infer import scans  # noqa: E402


def optimized_hlo(fn, *args) -> list[str]:
    return jax.jit(fn).lower(*args).compile().as_text().splitlines()


def counts(fn, *args) -> dict[str, int]:
    lines = optimized_hlo(fn, *args)
    return {
        "lines": len(lines),
        "while": sum(1 for l in lines if " while(" in l),
        "dot": sum(1 for l in lines if " custom-call(" in l or " dot(" in l),
        "fusion": sum(1 for l in lines if " fusion(" in l),
    }


def tree_root(M):
    return scans.up_sweep(M, normalize=True).root


def sequential_chain(M):
    def step(carry, m):
        return carry @ m, None
    out, _ = lax.scan(step, jnp.eye(M.shape[1], dtype=M.dtype), M)
    return out


def mats(L, S=32):
    return jnp.asarray(
        np.abs(np.random.default_rng(0).random((L, S, S))).astype(np.float32) * 0.5
    )


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [16, 32, 64, 128, 256])
def test_tree_has_no_device_while_loop(L):
    """The whole reason for choosing JAX. A single `while` here means the tree
    is not being captured into a command buffer."""
    assert counts(tree_root, mats(L))["while"] == 0


@pytest.mark.parametrize("L", [16, 32, 64, 128, 256])
def test_tree_matmul_count_is_exactly_log2_L(L):
    """`O(log L)`, asserted as the exact identity rather than a bound — the
    Blelloch up-sweep emits one batched GEMM per level and there are `log₂ L`
    levels. Measured on an H100: 4, 5, 6, 7, 8 for L = 16…256.
    """
    assert counts(tree_root, mats(L))["dot"] == int(np.log2(L))


@pytest.mark.parametrize("L", [16, 64, 256])
def test_sequential_chain_does_compile_to_a_while_loop(L):
    """The contrast that gives the assertions above their meaning. If this ever
    stops holding, the detection method — not the tree — needs revisiting."""
    c = counts(sequential_chain, mats(L))
    assert c["while"] == 1, "lax.scan is supposed to become a device while-loop"
    assert c["dot"] == 1, "and to hide all L matmuls inside that one loop body"


def test_tree_hlo_grows_with_L_but_chain_hlo_does_not():
    """Unrolling is visible in the program size, and it is what buys the
    capture: the tree's HLO grows with `L`, the chain's is constant because the
    work is hidden in a loop body."""
    t16, t256 = counts(tree_root, mats(16)), counts(tree_root, mats(256))
    c16, c256 = counts(sequential_chain, mats(16)), counts(sequential_chain, mats(256))
    assert t256["lines"] > t16["lines"]
    assert c256["lines"] == c16["lines"]


@pytest.mark.parametrize("S", [64, 128, 256])
def test_hlo_size_is_flat_in_S(S):
    """SPEC §5.5: "the HLO is 2,878 lines in all three cases — the program
    structure is fixed, only shapes change. **You pay per bucket, not per
    size.**" Measured here: 403–485 lines across S = 64…2048, i.e. flat.

    This is what makes power-of-two `|S|` bucketing worth doing at all.
    """
    base = counts(tree_root, mats(256, 64))["lines"]
    got = counts(tree_root, mats(256, S))["lines"]
    assert abs(got - base) < 0.3 * base, (
        f"HLO size should be ~independent of S: {base} at S=64 vs {got} at S={S}"
    )


@pytest.mark.parametrize("L", [16, 64, 256])
def test_maxplus_tree_also_avoids_a_while_loop(L):
    """The MAP path is the default emission (`--emission=map`), so it needs the
    same guarantee as the sampling path."""
    logM = jnp.asarray(
        np.random.default_rng(1).standard_normal((L, 32, 32)).astype(np.float32)
    )
    c = counts(lambda m: scans.up_sweep_maxplus(m).root, logM)
    assert c["while"] == 0


def test_prefix_suffix_has_no_while_loop():
    """`a` and `b` must come from a down-sweep, never a sequential loop — SPEC
    §2.4 says a `lax.scan` here is precisely the paper's +114%."""
    M = mats(64, 16)

    def f(m, a0, bf):
        tr = scans.up_sweep(m)
        return scans.prefix_suffix(tr, a0, bf)[0]

    a0 = jnp.ones((16,), dtype=jnp.float32)
    bf = jnp.ones((16,), dtype=jnp.float32)
    assert counts(f, M, a0, bf)["while"] == 0


@pytest.mark.parametrize("L", [16, 64, 256])
def test_log_space_tree_also_avoids_a_while_loop_and_is_log_depth(L):
    """The log-space sum-product tree is the emission=sample path after Phase
    4's underflow finding, so it needs the same `O(log L)` guarantee as the
    linear one — and it keeps it, because `log_matmul` shifts by row/column
    maxima and hands cuBLAS ordinary GEMMs instead of materialising `[S,S,S]`.

    **No GEMM count is pinned any more — deliberately.** Two GEMM-shaped
    forms of `log_matmul` were measured wrong on real grammars (single-shift,
    then two-band; see its docstring), and the exact streaming form has no
    `dot` at all. What this test still guards is the load-bearing property:
    zero device `while` loops, so the whole tree stays CUDA-graph-capturable.
    """
    logM = jnp.asarray(
        np.random.default_rng(2).standard_normal((L, 32, 32)).astype(np.float32)
    )
    c = counts(lambda m: scans.up_sweep_log(m).root, logM)
    assert c["while"] == 0, (
        "the streaming chunk loop must unroll at trace time — a device while "
        "escapes CUDA-graph capture, which is the paper's +114%"
    )
