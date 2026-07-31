"""Evaluation runner. SPEC §7, CLAUDE.md's documented CLI.

    python -m diffgemma_fa.eval.run --task bfcl_live --variant j0 --emission map

SPEC §7.2's baselines, all reachable through `--variant`:

| `--variant` | what it is |
|---|---|
| `unconstrained` | stock DiffusionGemma. Accuracy **and CS** |
| `mask` | **naive per-position masking** — the wrong-but-obvious baseline §2.8 targets; its CS is the headline evidence |
| `j2` | constrained draw at accepted positions, uniform random elsewhere |
| `j0` | decoupled trajectory/emission (`--emission=map` or `sample`) |
| `j1` | single joint draw with flattened marginals at non-accepted |

**Every run reports CS beside the content columns**, never alone: SPEC §3.8
warns that CS can be trivially 100% while measuring nothing, and Phase 4
measured exactly that (`{"location":""}` for every prompt).
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

from diffgemma_fa.compile import bfcl_data, pipeline, schema as _schema  # noqa: E402
from diffgemma_fa.compile.validate import Simulator  # noqa: E402
from diffgemma_fa.compile.vocab import END_TOKENS  # noqa: E402
from diffgemma_fa.eval import metrics  # noqa: E402
from diffgemma_fa.model.sampler import (  # noqa: E402
    ConstrainedDiffusionSampler, ZeroPartitionError)
from diffgemma_fa.model.state import Automaton  # noqa: E402

CKPT = "/home/ubuntu/diffgemma_fa/artifacts/ckpt/diffusiongemma-26B-A4B-it"
ALLOW = ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength",
         "first_match_wins", "additionalProperties")

#: E4/P1. The model's observed separators: indent-2 pretty printing to depth 3,
#: whitespace runs of 1/3/5/7 characters across all 130 unconstrained outputs.
#: Structured rather than `[ \n\t]{0,6}` so it cannot admit blank lines or bare
#: 6-space runs; measured at near-identical state count.
PRETTY_WS = r"( |\n {0,6})?"


def _case_insensitive_enums(schema):
    """E4/P4: expand each string enum literal to its case variants.

    BFCL's scorer lowercases and strips, so every variant scores identically —
    this cannot manufacture a wrong answer. Without it, a model that writes
    `"pizza"` against an enum of `PIZZA` has its preferred spelling forbidden
    and the renormalised draw lands on a *different* enum member (`SALAD`),
    which is strictly worse than the unconstrained arm. Measured on 7/130.
    """
    if isinstance(schema, list):
        return [_case_insensitive_enums(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == "enum" and isinstance(v, list):
            seen, variants = set(), []
            for e in v:
                for cand in ((e, e.lower(), e.upper(), e.capitalize())
                             if isinstance(e, str) else (e,)):
                    if cand not in seen:
                        seen.add(cand)
                        variants.append(cand)
            out[k] = variants
        else:
            out[k] = _case_insensitive_enums(v)
    return out


TASKS = {
    "bfcl_live": bfcl_data.LIVE_SPLITS,
    "bfcl_live_simple": ("BFCL_v4_live_simple.json",),
    "bfcl_nonlive": bfcl_data.NON_LIVE_SPLITS,
}


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


#: Experiment E2. The stock prompt says nothing about rendering, and the model
#: fills the vacuum with fenced, pretty-printed multi-line JSON — 126/130 of the
#: unconstrained outputs. The compiled grammar admits none of that (0/130
#: verbatim acceptance), so every constrained decode is forced off the model's
#: plan at each value boundary. `compact` tells the model to render the way the
#: grammar reads, which is the zero-code test of that whole hypothesis; the
#: "every key" clause additionally targets the shortest-member collapse, since
#: the unconstrained arm already emits every key in 125/126 outputs.
PROMPT_STYLES = {
    "stock": "",
    "compact": (
        " Output compact single-line JSON with no newlines and no code "
        "fences, and include every listed key."
    ),
}


def build_prompt(rec, fn, style: str = "stock") -> str:
    question = (rec.question[0][0]["content"] if rec.question
                else "Call the function.")
    props = list((fn.get("parameters") or {}).get("properties", {}))
    return (f"{question}\n\nRespond with a JSON object of arguments for "
            f"`{fn.get('name')}`, keys in this order: {props}."
            + PROMPT_STYLES[style])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="bfcl_live_simple", choices=sorted(TASKS))
    ap.add_argument("--variant", default="j1",
                    choices=["unconstrained", "mask", "j0", "j1", "j2"])
    ap.add_argument("--emission", default="sample", choices=["map", "sample"])
    ap.add_argument("--entropy-bound", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--nonempty", action="store_true", default=True)
    ap.add_argument("--no-nonempty", dest="nonempty", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--prompt-style", default="stock",
                    choices=sorted(PROMPT_STYLES),
                    help="E2: 'compact' asks for the rendering the grammar "
                         "actually admits")
    ap.add_argument("--whitespace", default="stock",
                    choices=["stock", "pretty"],
                    help="E4/P1: 'pretty' admits the model's newline+indent "
                         "separators (measured: 0/130 -> 74/130 verbatim "
                         "acceptance of unconstrained outputs, with --fence)")
    ap.add_argument("--fence", action="store_true",
                    help="E4/P2: allow an optional ```json fence around the "
                         "object, which 126/130 unconstrained outputs use")
    ap.add_argument("--ci-enums", action="store_true",
                    help="E4/P4: accept enum literals in any case. BFCL's own "
                         "scorer lowercases, so this cannot create a wrong "
                         "answer; without it the grammar forces a DIFFERENT "
                         "enum member (measured on 7/130)")
    args = ap.parse_args()

    model = diffusion.DiffusionGemma_26B_A4B()
    params = gm.ckpts.load_params(CKPT)
    base = diffusion.Sampler(model=model, params=params)
    tok = base.tokenizer
    print(f"[eval] task={args.task} variant={args.variant} "
          f"emission={args.emission} bound={args.entropy_bound}", flush=True)

    splits = TASKS[args.task]
    records = [r for s in splits for r in bfcl_data.iter_split(s)
               if len(r.functions) == 1]
    if args.n:
        records = records[: args.n]
    truth = {}
    for s in splits:
        truth.update(bfcl_data.load_possible_answers(s))

    sc = metrics.Scores()
    rows, skipped, zero_partition = [], {}, []
    t_start = time.perf_counter()

    for idx, rec in enumerate(records):
        fn = rec.functions[0]
        try:
            # Normalise the schema the GRAMMAR was compiled from, not the raw
            # one. With `--ci-enums` the grammar admits `"pizza"` for an enum
            # of `PIZZA`; validating against the un-expanded schema then marks
            # it invalid, while BFCL's own scorer (which lowercases) counts it
            # CORRECT. Measured: 5 of E4's 7 "schema-invalid" records were
            # exactly this, so the column was penalising the flag for doing
            # what it was designed to do.
            # NOT `params` -- that name holds the model weights in this scope,
            # and shadowing it fed a schema dict to `_prefill.prefill`, which
            # died in gemma's own `_dtype(params)` with "'str' object has no
            # attribute 'dtype'". Every arm queued after that edit failed the
            # same way; the checkpoint was fine all along.
            schema_params = fn["parameters"]
            if args.ci_enums:
                schema_params = _case_insensitive_enums(schema_params)
            norm = _schema.normalize_bfcl_schema(schema_params)
            a = pipeline.compile_json_schema(
                schema_params, name=fn.get("name", ""), from_bfcl=True,
                allow=ALLOW, allow_wildcard=True,
                whitespace_pattern=(PRETTY_WS if args.whitespace == "pretty"
                                    else None),
                fence=args.fence,
                nonempty_required_strings=args.nonempty).automaton
        except Exception as e:  # noqa: BLE001
            k = f"compile:{type(e).__name__}"
            skipped[k] = skipped.get(k, 0) + 1
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
            sample_from_predictions=ds.SampleFromPredictions(
                entropy_bound=args.entropy_bound,
                text_vocab_size=tok.vocab_size),
        )

        inputs = base._get_inputs(prompt=build_prompt(rec, fn,  # noqa: SLF001
                                                      args.prompt_style),
                                  images=None,
                                  add_bos=True, has_batch_dim=False,
                                  sharding=None)
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
        except ZeroPartitionError as e:
            # SPEC §6.3 cause (a) or (b): no accepted string of the canvas
            # length exists from A_k within budget. The **library** must raise
            # -- CLAUDE.md is explicit that only cause (c) is benign, and the
            # constrained paths run in log space so (c) cannot occur.
            #
            # But a benchmark harness aborting all 130 records because one
            # grammar has no in-budget completion is the wrong granularity: it
            # destroys the other 129 measurements and tells you nothing about
            # which record failed. So it is caught HERE, at the record level,
            # counted in its own column, and the record scores as a failure.
            # It is never swallowed: `zero_partition` is reported beside every
            # rate and the offending ids are written to the artifact.
            zero_partition.append({"id": rec.id, "fn": fn.get("name"),
                                   "detail": str(e)[:200]})
            sc.add(accepted=False, parsed_obj=None, want=None, schema_ok=False)
            rows.append({"id": rec.id, "fn": fn.get("name"), "text": "",
                         "parsed": None, "accepted": False, "schema_ok": False,
                         "zero_partition": True})
            continue
        jax.block_until_ready(state.predicted_tokens)

        toks = [int(x) for x in np.asarray(state.predicted_tokens)[0][
            : args.max_new_tokens]]
        while toks and toks[-1] == 0:
            toks.pop()
        text = tok.decode(toks)

        accepted = Simulator(a).accepts(toks)
        parsed = metrics.extract_json(text)
        # SPEC §7.2: `cs_rate` is the guarantee the sampler enforces, but the
        # automaton carries OUR channel header, so it scores the stock model at
        # ~0 partly for a convention it was never asked to follow.
        # `schema_valid_rate` is the header- and tokenizer-independent
        # cross-arm question. Both are reported; neither substitutes.
        schema_ok = metrics.schema_valid(parsed, norm)
        gt = truth.get(rec.id, [])
        want = (bfcl_data.materialize_ground_truth(
            gt[0].get(fn.get("name"), {}), fn["parameters"]) if gt else None)

        sc.add(accepted=accepted, parsed_obj=parsed, want=want,
               schema_ok=schema_ok)
        # Full text, not truncated: it is the evidence behind every column, and
        # a 220-character cap silently discards the tail of a long call.
        rows.append({"id": rec.id, "fn": fn.get("name"), "text": text,
                     "parsed": parsed, "accepted": accepted,
                     "schema_ok": schema_ok})

        if (idx + 1) % 10 == 0:
            d = sc.as_dict()
            print(f"  [{idx+1}/{len(records)}] CS={d['cs_rate']:.3f} "
                  f"schema={d['schema_valid_rate']:.3f} "
                  f"acc={d['arg_accuracy']:.3f} "
                  f"nonempty={d['nonempty_rate']:.3f}", flush=True)

    out = {
        "task": args.task, "variant": args.variant, "emission": args.emission,
        "entropy_bound": args.entropy_bound, "nonempty_strings": args.nonempty,
        "seed": args.seed,
        "prompt_style": args.prompt_style,
        "whitespace": args.whitespace,
        "fence": args.fence,
        "ci_enums": args.ci_enums,
        "records_available": len(records),
        "skipped_by_reason": skipped,
        # SPEC §6.3 causes (a)/(b) hit at run time, per record. Reported, never
        # folded into the other columns.
        "zero_partition": len(zero_partition),
        "zero_partition_records": zero_partition,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        **sc.as_dict(),
        "per_key": sc.detail,
        "rows": rows,
    }
    path = args.out or (
        f"/home/ubuntu/diffgemma_fa/artifacts/eval_{args.task}_"
        f"{args.variant}_{args.emission}_s{args.seed}.json")
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n===== RESULT =====")
    for k in ("task", "variant", "emission", "n", "cs", "cs_rate",
              "schema_ok", "schema_valid_rate", "parsed",
              "nonempty", "nonempty_rate", "arg_correct", "arg_total",
              "arg_accuracy", "exact_calls", "exact_call_rate",
              "skipped_by_reason", "zero_partition", "elapsed_seconds"):
        print(f"{k}: {out[k]}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
