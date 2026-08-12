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
#:
#: **Historical. Accepts 4 of the 5 standard renderings — it misses tabs** —
#: because it was reverse-engineered from observed outputs rather than from RFC
#: 8259. Kept selectable only so the arms in `docs/RESULTS.md` that were
#: measured with it can be reproduced; `--whitespace=json` (the default) is the
#: one to use.
PRETTY_WS = r"( |\n {0,6})?"

#: **[AUDIT-D2] `--whitespace` -> the pattern actually handed to the compiler.**
#:
#: This mapping used to be inline as `PRETTY_WS if args.whitespace == "pretty"
#: else None`, and `None` is **not** "use the pipeline default": `build_regex`
#: omits the kwarg entirely, so outlines' own `[ ]?` is used, which accepts 2 of
#: the 5 standard renderings. The repaired default (`schema.JSON_WS`) was
#: therefore unreachable from the CLI, and every eval arm to date ran a grammar
#: with a known over-constraint — the exact 0/130 failure family, whose symptom
#: is not a crash but a silently extended value (`600` -> `6000`).
#:
#: `None` is left meaning "outlines' own" on purpose (`test_schema.py` pins that
#: as the negative control), so the repair is made here, at the call site, where
#: the choice is visible in the results JSON.
WHITESPACE_PATTERNS: dict[str, str | None] = {
    #: RFC 8259's own definition. 5/5 renderings at every nesting depth.
    "json": _schema.JSON_WS,
    #: outlines' default, `[ ]?`. 2/5. What every arm before this ran.
    "stock": None,
    #: the hand-fitted pattern of E4/P1. 4/5, misses tabs.
    "pretty": PRETTY_WS,
}


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
    #: Drops the key-order hint entirely. The stock prompt ends with "keys in
    #: this order: [...]", and 98/98 unconstrained outputs obeyed it -- so the
    #: grammar's key-order over-constraint currently costs nothing. But that
    #: measurement cannot tell whether the model would reorder WITHOUT the
    #: hint, and anyone reusing the grammar without it inherits the risk. This
    #: style removes the hint so the question can be answered.
    "noorder": "",
    "compact": (
        " Output compact single-line JSON with no newlines and no code "
        "fences, and include every listed key."
    ),
}


def build_prompt(rec, fn, style: str = "stock") -> str:
    question = (rec.question[0][0]["content"] if rec.question
                else "Call the function.")
    props = list((fn.get("parameters") or {}).get("properties", {}))
    if style == "noorder":
        return (f"{question}\n\nRespond with a JSON object of arguments for "
                f"`{fn.get('name')}` using these keys: {props}.")
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
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many records before taking --n; lets the "
                         "second half of a split be run without redoing the "
                         "first")
    ap.add_argument("--out", default="")
    ap.add_argument("--prompt-style", default="stock",
                    choices=sorted(PROMPT_STYLES),
                    help="E2: 'compact' asks for the rendering the grammar "
                         "actually admits")
    ap.add_argument("--whitespace", default="json",
                    choices=["json", "pretty", "stock"],
                    help="whitespace policy for the grammar. 'json' is RFC "
                         "8259's own definition and accepts all five standard "
                         "renderings at every nesting depth -- use it. "
                         "'pretty' (E4/P1, 4/5, misses tabs) and 'stock' "
                         "(outlines' [ ]?, 2/5) are the historical patterns, "
                         "kept only to reproduce the arms in docs/RESULTS.md "
                         "that were measured with them. Under either, the "
                         "build gate REPORTS the renderings the narrow pattern "
                         "costs instead of refusing the compile -- and only "
                         "those: a schema whose grammar is defective for any "
                         "other reason is still refused, because the same "
                         "schema is re-checked under the wide pattern first")
    ap.add_argument("--expand-wildcard", action="store_true", default=True,
                    help="rewrite typeless sub-schemas (BFCL's `type: any`, a "
                         "literal `{}`) as an explicit anyOf over every JSON "
                         "type. ON BY DEFAULT and you want it: outlines-core "
                         "0.2.14 emits the typeless expansion as an "
                         "UNPARENTHESISED 7-way alternation, so the enclosing "
                         "object's braces attach to its first and last branch "
                         "only -- the grammar then rejects `{\"input_value\": "
                         "\"say hi\"}` and accepts a bare `1` and a stray `}`. "
                         "Measured on 11 of BFCL-Live's 4,549 schemas, 2 of "
                         "them inside the live_simple 130 cut")
    ap.add_argument("--no-expand-wildcard", dest="expand_wildcard",
                    action="store_false",
                    help="reproduce the pre-fix (broken) wildcard language. "
                         "The build gate still REFUSES those schemas, so this "
                         "reproduces the n=128 denominator, not the n=130 one")
    ap.add_argument("--fence", action="store_true",
                    help="E4/P2: allow an optional ```json fence around the "
                         "object, which 126/130 unconstrained outputs use")
    ap.add_argument("--dtype", default="float64",
                    choices=["float32", "float64"],
                    help="tree dtype for --emission=sample. **float64.** "
                         "float32 was briefly the default on the strength of a "
                         "toy-scale check (L=4, |S|<=8) and is WRONG at "
                         "production scale: measured on the E4 grammar at "
                         "L=256, it drives the Z==0 detector on 70/130 and "
                         "55/130 records, against 0/130 in float64 on the same "
                         "grammar. See model.constrained.require_x64")
    ap.add_argument("--temp", default="stock", choices=["stock", "greedy"],
                    help="'greedy' pins min=max=_MIN_TEMP (1e-12), the closest "
                         "reachable analogue of the paper's T=0 column. The "
                         "paper's headline 63.9 -> 71.5 is GREEDY; every arm "
                         "here so far used the stock 0.8 -> 0.408 anneal, so "
                         "we may be competing against a stronger baseline than "
                         "the paper did. Never executed before")
    ap.add_argument("--confidence", default="mf", choices=["mf", "mar"],
                    help="'mar' computes the accept rule's entropy from the "
                         "CONSTRAINED marginal q_i instead of the raw logits "
                         "(SPEC §3.4). The paper's ablation puts this at "
                         "68.4 -> 76.4, the larger half of its accuracy gain")
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
    # `--offset` exists so the SECOND half of a split can be run without
    # redoing the first. Every number reported so far is "the first 130 of
    # 258 by id", and a prefix is not a random sample -- BFCL ids cluster by
    # schema family, so the untouched half may not look like the measured one.
    records = records[args.offset:]
    if args.n:
        records = records[: args.n]
    truth = {}
    for s in splits:
        truth.update(bfcl_data.load_possible_answers(s))

    sc = metrics.Scores()
    rows, skipped, zero_partition, oom = [], {}, [], []
    # The compile-skip path's `oom_records` twin. `skipped_by_reason` keeps its
    # reason->count shape (two live consumers depend on it: this file's twin
    # `eval/run_tasks.py` derives `n` from `sum(skipped.values())`, and
    # `scripts/phase5_report.py` prints it verbatim into docs/RESULTS.md), so
    # the identities go in an ADDITIONAL field rather than into the histogram.
    skipped_records: list[dict] = []
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
            norm = _schema.normalize_bfcl_schema(
                schema_params, expand_wildcard=args.expand_wildcard)
            # [AUDIT-D3] THE BUILD GATE, wired. An unwired gate is not a
            # mitigation: `verify_renderings` existed and neither call site
            # passed it, so the check that turns "the grammar must accept how
            # the model writes" into a build failure sat unused beside the
            # defect it was written for. The instance comes from the schema
            # itself (no model, no data, milliseconds), so there is nothing to
            # opt into.
            gate_instance = _schema.synthesize_instance(norm)
            a = pipeline.compile_json_schema(
                schema_params, name=fn.get("name", ""), from_bfcl=True,
                allow=ALLOW, allow_wildcard=True,
                expand_wildcard=args.expand_wildcard,
                whitespace_pattern=WHITESPACE_PATTERNS[args.whitespace],
                fence=args.fence,
                verify_renderings=gate_instance,
                # The historical patterns are known-narrow by construction, so
                # under them the gate reports instead of raising -- otherwise
                # `--whitespace=pretty` could not reproduce its own arm at all.
                # Never silent: it prints, and `whitespace` is in the output.
                verify_strict=(args.whitespace == "json"),
                nonempty_required_strings=args.nonempty).automaton
        except Exception as e:  # noqa: BLE001
            k = f"compile:{type(e).__name__}"
            skipped[k] = skipped.get(k, 0) + 1
            # ADDITIVE ONLY: no `sc.add`, no `rows.append`, no counter. The
            # record never reached the model, so it stays out of `n` exactly as
            # `denominator_policy` says. What was missing was its *identity*.
            # `{"compile:ValueError": 2}` says two records left and not WHICH,
            # so a reader holding this artifact and one of the published n=130
            # rows in docs/RESULTS.md cannot compute the difference set -- and
            # the [AUDIT-D3] build gate refuses `live_simple_117-73-0` and
            # `live_simple_122-78-0`, both inside that cut, so the next arm is
            # an n=128 that must be shown to be comparable. Same shape as
            # `oom.append` / `zero_partition.append` below.
            #
            # Strings and plain dicts only: `e` itself and a `set` of ids both
            # raise in `json.dump` at the END of the run, after the GPU time is
            # spent.
            skipped_records.append({"id": rec.id, "fn": fn.get("name"),
                                    "reason": k, "detail": str(e)[:200]})
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
            constrained_dtype=args.dtype,
            logit_shaper=(
                ds.AnnealingTemperatureShaper(
                    ds.AnnealingTemperatureShaperConfig(
                        exponent=1.0, min_temperature=1e-12,
                        max_temperature=1e-12))
                if args.temp == "greedy" else base.logit_shaper),
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

        # [AUDIT-B2] Hoisted ABOVE the generation `try` so the OOM and
        # zero-partition handlers can pass it. It used to be computed only on
        # the success path, and both handlers passed `want=None` -- which drops
        # the record out of `arg_total` while leaving it in `n`. That is the
        # same emission-dependent denominator as [AUDIT-B], one level up, and on
        # this corpus it is *larger*: `exp_e5_grammar130_j0.json` has 70
        # zero-partition records of 130 (54%) and `exp_e5_grammar130_s1.json`
        # has 55 (42%), against `mask`'s 63 unparsed. Scored as shipped, the
        # first reports arg_accuracy 0.7321 over a denominator of 112 -- best in
        # the table, while having failed to generate on more than half its
        # records. Over the full 291 it is 0.2818.
        #
        # The exception handlers' own comments already said this is what they
        # meant: "the record scores as a failure", "never swallowed". `want=None`
        # is reserved for a record with no ground truth at all.
        gt = truth.get(rec.id, [])
        want = (bfcl_data.materialize_ground_truth(
            gt[0].get(fn.get("name"), {}), fn["parameters"]) if gt else None)

        try:
            state = sampler.sample_constrained(
                params=params, init_state=init,
                max_new_tokens=args.max_new_tokens,
                automaton=to_traced(a, batch=init.predicted_tokens.shape[0]))
        except jax.errors.JaxRuntimeError as e:
            # RESOURCE_EXHAUSTED on ONE record must not destroy the other 129.
            # Same lesson as the Z == 0 catch below: the library is right to
            # fail loudly, but the harness owns the granularity. The
            # sum-product tree is twice the size of MAP's max-plus tree, so a
            # 1024-bucket record wants ~8 GiB on top of the model's 51 GB.
            # Counted in its own column and the ids recorded -- never folded
            # into the accuracy numbers as if the model had answered badly.
            #
            # **[AUDIT-D1] This used to add "the whitespace-tolerant grammar
            # roughly doubles |S|". That is now backwards.** It was true of
            # `PRETTY_WS` and of the old `JSON_WS = [ \t\n\r]{0,8}`, both of
            # which count a bounded whitespace run. `JSON_WS` is now RFC 8259's
            # unbounded `[ \t\n\r]*`, one DFA state rather than an eight-state
            # counter: measured 34/45/59 states on the flat / one-array / 3-key
            # BFCL shapes against 42/55/71 for outlines' own `[ ]?`. The shipped
            # default moves records DOWN a bucket, not up, so this path should
            # be rarer than the numbers in docs/RESULTS.md suggest.
            if "RESOURCE_EXHAUSTED" not in str(e):
                raise
            oom.append({"id": rec.id, "fn": fn.get("name"),
                        "n_states_bucket": int(a.n_states_bucket)})
            # `want`, not None -- see the hoist above. The record failed to
            # produce a call; its arguments are still arguments the benchmark
            # asked for, and `oom` beside it says *why* it failed.
            sc.add(accepted=False, parsed_obj=None, want=want, schema_ok=False)
            rows.append({"id": rec.id, "fn": fn.get("name"), "text": "",
                         "parsed": None, "accepted": False, "schema_ok": False,
                         "oom": True})
            continue
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
            # `want`, not None -- see the hoist above. This is the handler the
            # 70/130 artifact went through.
            sc.add(accepted=False, parsed_obj=None, want=want, schema_ok=False)
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
        "offset": args.offset,
        "prompt_style": args.prompt_style,
        "confidence": args.confidence,
        "temp": args.temp,
        "dtype": args.dtype,
        "whitespace": args.whitespace,
        # A grammar-shaping boolean that never lands in the artifact is
        # invisible in exactly the way `--whitespace` was. It also moves the
        # DENOMINATOR: with it on, the two live_simple records the build gate
        # refused (live_simple_117-73-0, live_simple_122-78-0) compile, so a
        # `live_simple` arm reports n=130 where the published ones report 128.
        "expand_wildcard": args.expand_wildcard,
        "fence": args.fence,
        "ci_enums": args.ci_enums,
        "records_available": len(records),
        # THE DENOMINATOR, stated rather than implied -- `eval/run_tasks.py`
        # used to keep compile-skipped records in `n` while this file dropped
        # them, so the same column meant different things in two artifacts that
        # get compared. Both now use this rule.
        #
        # [AUDIT-B2] The previous wording of this string said "oom and
        # zero_partition records stay in n as failures" while the code passed
        # them `want=None`, so they left `arg_total`. An artifact asserting a
        # policy it does not implement is worse than one that asserts nothing.
        # The code now matches.
        #
        # NOT COVERED BY A TEST. `tests/test_audit_measurement.py` pins the
        # denominator rule on `Scores.add`, which is where [AUDIT-B] lived; the
        # three call sites in this loop are what [AUDIT-B2] was, and reaching
        # them needs the model. If you touch them, re-check this string by hand.
        "denominator_policy": (
            "n = records attempted. Compile-skipped records never reached the "
            "model: they are excluded from n and from arg_total, and reported "
            "in skipped_by_reason. Oom and zero_partition records DID reach the "
            "model and failed: they stay in n AND in arg_total as failures, and "
            "are reported in their own columns. arg_total therefore depends "
            "only on the benchmark, never on what the model emitted. Matches "
            "eval/run_tasks.py. The skipped records are named individually in "
            "skipped_records, so the difference set against another arm's n is "
            "computable from this file alone."
        ),
        "skipped_by_reason": skipped,
        # Which records left n, and why. `skipped_by_reason` counts them;
        # this names them. Additive -- see the handler.
        "skipped_records": skipped_records,
        # SPEC §6.3 causes (a)/(b) hit at run time, per record. Reported, never
        # folded into the other columns.
        "zero_partition": len(zero_partition),
        # Records the device could not fit. A capacity fact, not a model fact.
        "oom": len(oom),
        "oom_records": oom,
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
              "skipped_by_reason", "zero_partition", "oom",
              "elapsed_seconds"):
        print(f"{k}: {out[k]}")
    # Ids only -- an operator watching an 80-minute log should be able to name
    # the records that left `n` without opening the JSON, and the `detail`
    # strings are 200 chars each. The full entries are in the artifact.
    if skipped_records:
        print(f"skipped_records: {[r['id'] for r in skipped_records]}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
