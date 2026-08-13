# diffgemma_fa — constrained decoding for DiffusionGemma via finite automata

A JAX implementation of [arXiv:2607.07026](https://arxiv.org/abs/2607.07026),
built against `google-deepmind/gemma`'s `gemma/diffusion/` and the Orbax
checkpoint `gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it`.

**Inference only.** No training, no fine-tuning, no weight modification, no
backward pass. The method consumes exactly one thing from the model — the
`[B, 256, V]` logits already produced at every denoising step — and reshapes the
sampler around a compiled automaton. The Flax module and the params are never
touched.

---

## What this does

A diffusion language model denoises a whole 256-token canvas at once, so the
usual autoregressive trick of masking the next-token logits does not apply: any
position may be resampled at any step. This implements the paper's alternative —
treat the canvas as a chain, compile the target language into a finite
automaton, and draw the **entire canvas jointly** from the constrained
posterior via a forward–backward over the automaton's transition matrices.

The guarantee is exact rather than statistical: every emitted token sequence is
accepted by the compiled automaton, by construction, not by rejection sampling.

## Does it work?

**For JSON, yes.** On BFCL-Live `simple` (function-calling, so the schema is the
task):

| | unconstrained | constrained |
|---|---|---|
| **schema valid** | 0.631 | **1.000** (130/130) |
| constraint satisfaction | 0.000 | **1.000** |
| argument accuracy | 0.639 | 0.663 |
| exact call | 0.500 | 0.531 |

Schema validity 63% → **100%** at no accuracy cost. Every record emits a valid
instance of its schema, with zero partition failures and zero OOMs.

Measured at `4f9a6e3` (code identity `1aa8ccf`), n = 130 of 130 available, no
records skipped at compile time. The constrained row reached 1.000 only after
the `type: any` fix (`15cdc47`) — before it, the single failure was
`live_simple_122-78-0`, whose grammar could not emit a well-formed object; see
"A worked example" below. The unconstrained row was re-verified at HEAD in the
same run and reproduces verbatim, so it is a current number rather than a stale
one. Read the two content
columns carefully, though: **the accuracy claim is parity, not a gain.**
Restricted to records both arms answered, argument accuracy *reverses*
(0.6703 unconstrained vs 0.6667), and the exact-call gap is 7 vs 4 discordant
records at exact-binomial p = 0.55. The supported statement is "costs nothing",
and `docs/RESULTS.md` argues against its own most flattering reading.

**Not everywhere.** On Sudoku the same machinery takes an 86%-accurate solver to
near zero. The honest generalisation is in `docs/RESULTS.md`:

> The guarantee is free when the grammar encodes the task's correctness
> condition, and can be catastrophic when it only encodes the output's shape.

For BFCL the JSON schema largely *is* the correctness condition. For Sudoku the
grammar pins the givens and the digit alphabet and says nothing about rows,
columns or boxes — so it removes the model's freedom without adding information
about what makes an answer right.

### Caveats on the JSON result

- 130 of 258 records in one split, one seed. A prefix is not a random sample.
- **The accuracy movement is not attributable.** Schema validity 0.992 → 1.000
  is: the failing record is named and its mechanism measured. The argument
  accuracy move (0.650 → 0.663, four arguments of 291) is **not** — four commits
  landed on the `j0/map` path between the two measurements (`da1294a` can move a
  MAP argmax, `b58fd84` gates eq (8)'s fast path), so this is not a
  one-variable delta and no cause should be assigned to it.
- **The constrained block straddles two grammars.** This row is post-`15cdc47`;
  `j0-sample`, `j1-sample`, `j2-sample`, `mask-sample` and `j0-map` are not, and
  they decoded records 117 and 122 under a grammar that rejects the correct
  object. This row pairs legitimately against the unconstrained baseline, which
  is grammar-invariant and was re-verified at HEAD — but **not** against the
  other constrained rows without restricting to common records, which the
  per-record `rows` lists in each artifact permit on CPU.
- **All 4,549 BFCL-Live schemas now compile** (was 4,538). BFCL's `"type": "any"`
  used to become an *unparenthesised* alternation, so the grammar rejected the
  valid object and accepted bare scalars; those 11 were refused at compile time
  rather than mis-decoded. Fixed in `15cdc47` — see "A worked example" below,
  which is retained because it is the clearest case in this project of the
  guarantee locating a fault.
- **The compiled grammar is narrower than RFC 8259**, inherited from
  `outlines-core` rather than introduced here: `\uXXXX` escapes, unsigned
  exponents (`1e5`), and nesting deeper than 4 are all rejected despite being
  valid JSON. Measured, with root causes and fixability, in
  `docs/LIMITATIONS.md` — the first two are one-line RFC deviations, the third
  is a real bound on what a finite automaton can express.
- SPEC §3.7 (`--self-cond=constrained`) and §3.3 (`--trajectory=R1/R2`) do not
  exist in the tree. Five of six SPEC §7.1 datasets have never been run.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install git+https://github.com/google-deepmind/gemma.git   # PyPI 4.0.1 predates gemma/diffusion/
pip install -e .
source env.sh          # pins CUDA_VISIBLE_DEVICES, caps JAX memory, sets the compile cache

pytest tests/ -q                      # ~1,100 tests, CPU-only, ~9 min
pytest tests/test_exactness.py -x      # the bedrock: brute-force posterior on V=4, L<=8
pytest tests/test_guarantee.py -x      # the two-part guarantee

python -m diffgemma_fa.compile.tasks.bfcl --out artifacts/fa/bfcl/ --jobs $(nproc)
python -m diffgemma_fa.eval.run --task bfcl_live_simple --variant j0 --emission map
```

`env.sh` matters: JAX preallocates ~75% of VRAM on first use by default, and
`pytest tests/` inherits it. Several hours were lost to exactly that.

## Layout

```
diffgemma_fa/
  compile/    JSON Schema -> regex -> token DFA. Valmari minimisation, class
              tables with complement-aware storage, the build gate.
  infer/      the math core. Blelloch up-sweep/down-sweep, log-space
              forward-backward, constrained marginals, eq (8)'s token draw.
  model/      the sampler fork: widened SamplingState, constrained emission,
              the budget-aware terminal factor b_L = 1[d(s) <= R].
  eval/       BFCL / Countdown / Sudoku runners, scorers, Wilson + McNemar.
tests/        26 files, 15,067 lines against 8,916 lines of implementation.
docs/         SPEC.md is the design; the rest is the working record.
```

## Reading the documentation

| file | what it is |
|---|---|
| **`docs/HANDOFF.md`** | **start here if you are picking this up cold.** What works, what is not implemented, what is broken and why, which results are invalid, and the cheapest next move for each. |
| `SPEC.md` | the design document. Read this first; it is the reference the tests derive from. |
| `docs/RESULTS.md` | every measured number, with its regime and its caveats. Contains corrections to its own earlier claims. |
| `docs/LOG.md` | the chronological working record. |
| `docs/PHASE*_FINDINGS.md` | what each phase established, including where SPEC.md was wrong. |
| `docs/METHOD_REFERENCE.md` | the paper's method, derived. |
| `CLAUDE.md` | the operating manual — how to work in this repo, and the landmines that cost a day each. |

**`docs/LOG.md` is a working record, not a summary.** It contains claims that
were later retracted, and it says so at the point of retraction rather than
deleting them. That is deliberate: the retractions are the most useful part,
because they show what a measurement actually supported versus what was inferred
from it. Do not quote a number from it without checking whether a later entry
withdrew it. `docs/RESULTS.md` is the place to take numbers from.

## A worked example: when the guarantee is doing its job

`live_simple_122-78-0`, the single schema-invalid record in the JSON result.
BFCL declares one property as `"type": "any"`, which normalises to an omitted
type, which outlines expands into an alternation it does not parenthesise:

```
\{ws"input_value"ws:ws((true|false))|(null)|(number)|(string)|(array)|(object)ws\}
```

Only the first branch carries the `{`, only the last the `}`. Measured:

| probe | in the language? |
|---|---|
| `{"input_value": 1}` | **no** — the correct answer is rejected |
| `1`, `null`, `"x"` | **yes** — bare scalars leak |

Under constraint the model emitted `1`. Unconstrained, on the same record, it
emitted a complete and correct object. The grammar structurally prevented the
right answer.

The useful part is what the guarantee lets you conclude. That record has
`CS = 1.000` and `schema_valid = False` — and since CS *proves* the emission was
in the compiled language, a string that is in the language and violates the
schema means the language is not the schema. **The compiler is wrong, provably,
without inspecting the model.** The guarantee turns an ambiguous failure into
a located one.

## How this was built

Every feature and non-trivial fix goes through **tester → coder → reviewer**,
as three separate agents, described in `CLAUDE.md`. The tester writes tests from
`SPEC.md` before the implementation exists and may not read it; the coder
implements against those tests and may not edit them; the reviewer runs both,
mutation-tests the tests, and decides whether "the tests pass" means anything.
Nothing is committed without the reviewer's explicit approval.

This is not ceremony. It exists because the expensive failures here were not
hard bugs — they were plausible code, and plausible *measurements*, that nobody
checked against an independent expectation:

- A scorer that read digits out of the raw emission, so a channel header shifted
  the grid and it reported "overwrote the given" on 250/250 records **while the
  automaton had already proved the givens intact.** A number contradicting a
  proof is the measurement's fault.
- An arm reporting the best accuracy in the table over a denominator that had
  silently dropped **54% of its own records** — the ones it failed on.
- A blast-radius table wrong by four orders of magnitude because two grammars
  were measured at `remaining=0` and a third at `remaining=1000`, which makes
  `b_L = 1[s live]` and *hides the very loss being measured*.
- Six vacuous tests, each of which re-implemented the thing under test inside
  the test and asserted against its own re-implementation. One had a docstring
  prescribing the defective formula as the specification.

Two rules earned the hard way and enforced in `CLAUDE.md`: a reviewer reads
every GPU- or artifact-writing command **before** it runs (the first application
caught a 13.5-hour queue configured against a grammar no artifact in this repo
had ever used), and no measurement is taken against a working tree that an agent
is editing.

## Status

Working: the compiler, the inference kernels, the sampler fork, the guarantee,
and the BFCL evaluation. ~1,100 tests green.

Open, tracked in `docs/LOG.md`: `infer/reference.py` — the float64 oracle every
optimised path is differential-tested against — has itself never been audited;
`up_sweep`'s linear normalisation underflows below T ≈ 0.1 at *unflagged*
positions; SPEC §7.3's overhead table needs re-measuring under a whitespace
grammar that turned out to be *smaller* than the stock one; and the `type: any`
normalisation above.

## Licence and provenance

Not yet licensed — add one before distributing. Depends on
`google-deepmind/gemma` (Apache-2.0) and `outlines-core` (Apache-2.0), and
evaluates against BFCL, which is not vendored here. Model weights are not
included and are not redistributable from this repo.
