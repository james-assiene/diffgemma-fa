"""Evaluation for Countdown and Sudoku. SPEC §7.1.

    python -m diffgemma_fa.eval.run_tasks --task countdown --variant j0 \
        --emission map --n 200

**Why these two, separately from `eval/run.py`.** Every accuracy number in this
project so far comes from BFCL, whose grammars are large permissive JSON
schemas: `|S|` in the hundreds to 1,024, where §7.3 measures the constrained
tree at up to 82.6% of a model forward, and where the automaton barely narrows
the output space. Countdown and Sudoku are the opposite regime — tight
combinatorial grammars over a handful of symbols, small `|S|`, where the
constraint eliminates almost everything. SPEC §2.8 cites the paper measuring
unconstrained constraint satisfaction at **7.6%** on Sudoku.

That difference is the point. If the whitespace finding (the compiled grammar
forbidding the model's own separators, worth 0.328 -> 0.628 on BFCL) was an
artifact of JSON rendering rather than something general about
grammar-tokenizer alignment, these tasks are where it fails to appear.

The scorers are deliberately **not** format checks. A grammar guarantees the
shape of an answer and says nothing about whether it is right: `countdown`
admits `1+1=3`, and a format-only Sudoku grammar would let a model overwrite
the givens and solve a different puzzle. Both are re-derived in
`compile/tasks/{countdown,sudoku}.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/home/ubuntu/diffgemma_fa/.jax_cache")

import jax.numpy as jnp  # noqa: E402
from gemma import diffusion, gm  # noqa: E402
from gemma.diffusion import _sampler as ds  # noqa: E402
from gemma.gm.text import _prefill  # noqa: E402

from diffgemma_fa.compile import pipeline  # noqa: E402
from diffgemma_fa.compile import tasks as _tasks  # noqa: E402
from diffgemma_fa.compile.tasks import countdown as CD  # noqa: E402
from diffgemma_fa.compile.tasks import sudoku as SD  # noqa: E402
from diffgemma_fa.compile.tasks.grammars import (  # noqa: E402
    countdown_regex, sudoku_regex)
from diffgemma_fa.compile.validate import Simulator  # noqa: E402
from diffgemma_fa.model.sampler import (  # noqa: E402
    ConstrainedDiffusionSampler, ZeroPartitionError)
from diffgemma_fa.model.state import Automaton  # noqa: E402

CKPT = "/home/ubuntu/diffgemma_fa/artifacts/ckpt/diffusiongemma-26B-A4B-it"
COUNTDOWN_DATA = "/home/ubuntu/diffgemma_fa/data/countdown_test.jsonl"


def to_traced(a, batch: int) -> Automaton:
    return Automaton(
        edge_src=jnp.asarray(a.edge_src), edge_dst=jnp.asarray(a.edge_dst),
        edge_class=jnp.asarray(a.edge_class),
        edge_valid=jnp.ones(a.n_edges, bool),
        csr_indices=jnp.asarray(a.tables.sum_indices),
        csr_indptr=jnp.asarray(a.tables.sum_indptr),
        is_neg=jnp.asarray(a.tables.sum_is_neg),
        d=jnp.asarray(a.d), is_final=jnp.asarray(a.is_final),
        active=jnp.broadcast_to(jnp.asarray(a.start_vector),
                                (batch, a.n_states_bucket)),
    )


#: A bounded free-text region before the answer, terminated by a literal
#: marker. SPEC §3.6's channel header gives the model only 8 tokens of free
#: space, which is a *label*, not a scratchpad.
#:
#: The measurement that motivates this: unconstrained, the model solves 86.4%
#: of 4x4 Sudokus (while never matching the grammar); constrained to emit only
#: the grid, it solves 0.0% with CS 1.000. The grammar is provably correct --
#: the failures are row/column violations, not overwritten givens -- so the
#: constraint is not corrupting the answer, it is removing the space the model
#: reasons in. `gsm_symbolic_regex` already uses this shape (free prose,
#: constrained arithmetic); this applies it to the other two tasks.
#: **A regex prefix does not work here and the failure is instructive.** The
#: first attempt was `[^\n]{0,400}(?:\n[^\n]{0,400}){0,12}\nANSWER:\n`, which
#: is a near-Σ character class repeated thousands of times; lifted over a
#: 262,144-token vocabulary it OOM-killed the host on both tasks. That is
#: exactly SPEC §4.2's "regex too large" explosion.
#:
#: The right mechanism already exists. SPEC §3.6's channel header is a
#: **bounded token chain** — `100 · Σ_name{1,n} · 107 · 101` — deliberately not
#: a Σ* self-loop, because a self-loop has emission mass ~1.0 and a joint
#: decode sits in it forever. Widening `n` from 8 to `THINK_TOKENS` costs that
#: many states and gives the model a real scratchpad, and the scorers already
#: discard everything before `<channel|>`.
THINK_TOKENS = 64


def load_task(task: str, n: int, seed: int, think: bool = False):
    """`[(record, prompt, regex, scorer)]`."""
    hint = ("\nThink briefly in the channel header first, then give the "
            "answer.") if think else ""
    if task == "countdown":
        recs = list(CD.iter_records(COUNTDOWN_DATA, limit=n))
        return [(r, CD.build_prompt(r) + hint,
                 countdown_regex(max_steps=4, max_value=999,
                                 step_separator=r"\n"),
                 CD.score_solution) for r in recs]
    recs = SD.generate(n or 100, seed=seed)
    return [(r, SD.build_prompt(r) + hint,
             sudoku_regex(r.puzzle, row_separator="\n"),
             SD.score_solution) for r in recs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="countdown", choices=["countdown", "sudoku"])
    ap.add_argument("--variant", default="j0",
                    choices=["unconstrained", "mask", "j0", "j1", "j2"])
    ap.add_argument("--emission", default="map", choices=["map", "sample"])
    ap.add_argument("--entropy-bound", type=float, default=0.1)
    ap.add_argument("--confidence", default="mf", choices=["mf", "mar"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", action="store_true",
                    help="admit a bounded free-text scratchpad before the "
                         "answer, terminated by `ANSWER:`. Tests whether the "
                         "constraint's cost on reasoning tasks is the loss of "
                         "room to reason")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model = diffusion.DiffusionGemma_26B_A4B()
    params = gm.ckpts.load_params(CKPT)
    base = diffusion.Sampler(model=model, params=params)
    tok = base.tokenizer
    print(f"[eval] task={args.task} variant={args.variant} "
          f"emission={args.emission} n={args.n}", flush=True)

    items = load_task(args.task, args.n, args.seed, args.think)
    n_ok = n_cs = n_parsed = 0
    rows, skipped, zero_partition, oom = [], {}, [], []
    # Which records the compile-skip path removed from `n`. `skipped_by_reason`
    # stays a reason->count histogram because `n` below is literally
    # `len(items) - sum(skipped.values())` and because phase5_report prints it
    # into docs/RESULTS.md; the identities go in this ADDITIONAL field.
    skipped_records: list[dict] = []
    reasons: dict[str, int] = {}
    t_start = time.perf_counter()

    for idx, (rec, prompt, regex, scorer) in enumerate(items):
        try:
            a = pipeline.compile_regex(
                regex, name=args.task,
                header_tokens=THINK_TOKENS if args.think else 8).automaton
        except Exception as e:  # noqa: BLE001
            k = f"compile:{type(e).__name__}"
            skipped[k] = skipped.get(k, 0) + 1
            # ADDITIVE ONLY: nothing is scored, nothing is appended to `rows`
            # (phase5_report pairs arms on the ids it finds there), no counter
            # moves. The record leaves `n` exactly as before -- it is now
            # merely NAMED, so the difference set against an arm with a
            # different `n` is computable from the artifact alone. It matters
            # more here than in eval/run.py, because `n` below is derived from
            # the histogram. Strings only: `e` and a `set` both die in
            # `json.dump` at the end of the run.
            skipped_records.append({"id": rec.id, "reason": k,
                                    "detail": str(e)[:200]})
            continue

        sampler = ConstrainedDiffusionSampler(
            model=model,
            end_tokens=(tok.special_tokens.EOS, tok.special_tokens.END_OF_TURN,
                        tok.special_tokens.BEGIN_OF_TOOL_RESPONSE),
            forbidden_tokens=None, sampling=base.sampling,
            cache_length=base.cache_length, special_tokens=tok.special_tokens,
            canvas_length=256, max_denoising_steps=48,
            text_vocab_size=tok.vocab_size,
            sliding_window_size=getattr(model.config, "sliding_window_size", None),
            n_states_bucket=a.n_states_bucket, n_classes=a.tables.n_classes,
            variant=args.variant, emission=args.emission,
            confidence=args.confidence,
            sample_from_predictions=ds.SampleFromPredictions(
                entropy_bound=args.entropy_bound,
                text_vocab_size=tok.vocab_size),
        )

        inputs = base._get_inputs(prompt=prompt, images=None, add_bos=True,  # noqa: SLF001
                                  has_batch_dim=False, sharding=None)
        init = _prefill.prefill(
            model=model, params=params, input=inputs, last_state=None,
            cache_length=base.cache_length, pad_length=base.pad_length,
            rng=jax.random.PRNGKey(args.seed * 10_000 + idx), sharding=None,
            max_out_length=base.max_out_length)

        try:
            state = sampler.sample_constrained(
                params=params, init_state=init,
                max_new_tokens=args.max_new_tokens,
                automaton=to_traced(a, batch=init.predicted_tokens.shape[0]))
        except jax.errors.JaxRuntimeError as e:
            if "RESOURCE_EXHAUSTED" not in str(e):
                raise
            oom.append(rec.id)
            rows.append({"id": rec.id, "text": "", "ok": False, "oom": True})
            continue
        except ZeroPartitionError:
            zero_partition.append(rec.id)
            rows.append({"id": rec.id, "text": "", "ok": False,
                         "zero_partition": True})
            continue
        jax.block_until_ready(state.predicted_tokens)

        toks = [int(x) for x in np.asarray(state.predicted_tokens)[0][
            : args.max_new_tokens]]
        while toks and toks[-1] == 0:
            toks.pop()
        text = tok.decode(toks)

        accepted = Simulator(a).accepts(toks)
        ok, why = scorer(text, rec)
        n_cs += int(accepted)
        n_ok += int(ok)
        # [AUDIT-E] Was `int(why != "empty")`, and Sudoku's no-output reason was
        # `"only 0 digits"` -- so the predicate was true for every Sudoku record
        # including the empty ones and the column was 100% by construction. The
        # two scorers now agree on `"empty"`, and the predicate is named
        # (`tasks.produced_an_answer`) rather than open-coded here.
        n_parsed += int(_tasks.produced_an_answer(why))
        reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1
        rows.append({"id": rec.id, "text": text[:300], "ok": ok,
                     "why": why, "accepted": accepted})

        if (idx + 1) % 10 == 0:
            m = idx + 1
            print(f"  [{m}/{len(items)}] CS={n_cs/m:.3f} solved={n_ok/m:.3f}",
                  flush=True)

    # THE DENOMINATOR, stated rather than implied. `eval/run.py` and this file
    # used to disagree: `run.py` dropped a compile-skipped record out of `n`
    # entirely, this file kept it in `n` and scored it as a failure, so the same
    # column meant different things in two artifacts that get compared. They now
    # agree on `run.py`'s rule, which is also the one every published BFCL
    # number in docs/RESULTS.md was computed under:
    #
    #   `n` is the records the model was actually **asked**. A compile failure
    #   happens before the model is invoked, so the record leaves `n` and is
    #   reported in `skipped_by_reason`. An OOM or a zero-partition happens
    #   during generation, so the record stays in `n` as a failure and is
    #   reported in its own column beside it.
    n_skipped = sum(skipped.values())
    n = len(items) - n_skipped
    out = {
        "task": args.task, "variant": args.variant, "emission": args.emission,
        "confidence": args.confidence, "entropy_bound": args.entropy_bound,
        "think": args.think,
        "seed": args.seed, "n": n,
        "records_available": len(items),
        "denominator_policy": (
            "n = records attempted: compile-skipped records are excluded from "
            "n and reported in skipped_by_reason; oom and zero_partition "
            "records stay in n as failures. Matches eval/run.py. The "
            "compile-skipped records are named individually in "
            "skipped_records, so the difference set against another arm's n "
            "is computable from this file alone."
        ),
        "cs": n_cs, "cs_rate": round(n_cs / max(1, n), 4),
        "solved": n_ok, "solve_rate": round(n_ok / max(1, n), 4),
        # [AUDIT-E] Now actually written out. It was computed and dropped, which
        # is the only reason the vacuous predicate above never reached a table.
        "parsed": n_parsed, "parse_rate": round(n_parsed / max(1, n), 4),
        "zero_partition": len(zero_partition), "oom": len(oom),
        "skipped_by_reason": skipped,
        # Which records left n, and why. The histogram counts them; this
        # names them. Additive -- see the handler.
        "skipped_records": skipped_records,
        # Why the unsolved ones failed -- a format failure and a wrong answer
        # are different diagnoses and must not be merged.
        "failure_reasons": reasons,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        "rows": rows,
    }
    path = args.out or (f"/home/ubuntu/diffgemma_fa/artifacts/"
                        f"task_{args.task}_{args.variant}_{args.emission}.json")
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n===== RESULT =====")
    for k in ("task", "variant", "emission", "n", "records_available",
              "cs_rate", "solve_rate", "parsed", "parse_rate",
              "zero_partition", "oom", "skipped_by_reason", "failure_reasons",
              "elapsed_seconds"):
        print(f"{k}: {out[k]}")
    # Ids only -- see eval/run.py. `n` here is derived from the histogram, so
    # the operator seeing `n` shrink should see which records did it.
    if skipped_records:
        print(f"skipped_records: {[r['id'] for r in skipped_records]}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
