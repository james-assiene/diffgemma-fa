# Phase 1 findings — the constraint compiler

Date: **2026-07-27**. Stack as `docs/ENV.md`. Deliverable: `diffgemma_fa/compile/`. 270 tests green.

Phase 1 builds the pipeline SPEC §4.1 specifies:

```
schema -> regex -> byte DFA (outlines) -> token lift -> Valmari -> label classes -> artifact
```

Everything below is measured on the **real** BFCL v4 checkout, not on hand-written
approximations. Scripts are re-runnable; artifacts are in `artifacts/phase1_*.json` and
`artifacts/bfcl_compile_report.json`.

| script / entry point | what it establishes |
|---|---|
| `scripts/phase1_bfcl_timing.py` | §4.7 — per-schema compile cost on real BFCL-Live |
| `scripts/phase1_roundtrip.py` | §8 Phase 1 exit — FA→validator and ground-truth→FA |
| `scripts/phase1_diagnose_gt.py` | classifies every ground-truth rejection by cause |
| `python -m diffgemma_fa.compile.tasks.bfcl` | the parallel compile CLI from CLAUDE.md |

---

## 1. Three correctness bugs found by building it

None of these announce themselves. Each produces an automaton that still looks fine.

### 1.1 Reserved tokens leak into the grammar alphabet

`<eos>` (1) and `<pad>` (0) **are** SentencePiece control tokens, so they drop out of the
vocabulary automatically. **`<turn|>` (106) and `<|tool_response>` (50) are not** — they are
ordinary pieces whose bytes are plain ASCII (`<turn|>`, `<|tool_response>`), so `outlines_core`
happily lets a JSON string body match them.

Consequence: the grammar can emit a stop token *inside a string value*.
`_truncate_canvas_at_stop_tokens` then cuts the canvas mid-grammar and `A_{k+1}` goes empty. That
is SPEC §3.1b failure mode 3 — but reached from the **grammar** region, whereas §3.6 only
discusses it for the free-text region. `compile/vocab.py: RESERVED_TOKENS` excludes
`{PAD} ∪ end_tokens ∪ {100, 101, 105}`.

### 1.2 `T[final][eos]` must be stripped, not just ignored

SPEC §4.3 records that `guide.advance(eos)` raises "even though `T[final][eos]` exists" and
concludes stop tokens should be handled out of band. They are — but the injected edge still has to
be **removed**. Left in place, the grammar-final state has two destinations on the EOS label once
§3.5's augmentation adds `final --eos--> ACC`.

The automaton is then **spuriously nondeterministic**, which silently drops eq (8) off its `∃`
fast path onto the multiplicity-weighted one, and makes `is_dfa` false for *every* grammar. Nothing
fails; the sampler just gets slower and the `is_dfa` gate becomes useless. With the edge stripped
(`lift.py: drop_labels`), real BFCL schemas lift to genuine DFAs — measured **4,549 / 4,549 DFA**
across the whole of BFCL-Live.

### 1.3 BFCL's `any` cannot be passed through

BFCL's wildcard has no JSON Schema spelling; the equivalent is *omitting* `type`. Passing the
literal string reaches outlines as `ValueError: Unsupported type: any` and killed **11 of 4,549**
BFCL-Live schemas. Now translated to a typeless schema, still refused unless
`allow_wildcard=True`.

---

## 2. BFCL is not JSON Schema — a translation layer SPEC never mentions

`function[].parameters` is a Python-flavoured dialect, with Java and JavaScript type names leaking
in from the `simple_java` / `simple_javascript` splits. Measured occurrences across all v4 splits:

| type | n | type | n | type | n |
|---|---|---|---|---|---|
| `string` | 21,854 | `dict` | **9,464** | `integer` | 4,757 |
| `boolean` | 3,115 | `float` | **1,690** | `array` | 959 |
| `any` | 199 | `String` | 115 | `tuple` | 66 |
| `Array` | 13 | `HashMap` | 7 | `long` | 7 |
| `ArrayList` | 6 | `Boolean` | 4 | `double` | 1 |
| `char` | 1 | | | | |

**`object` and `number` occur exactly zero times.** `compile/schema.py: BFCL_TYPE_MAP` handles the
translation and **raises** on an unrecognised name rather than guessing — a new dialect leaking in
should be a loud failure, not a silently mistyped grammar.

Rare keywords that outlines silently drops do occur: `maximum` (2), `minItems` (1), `maxItems` (1),
`format` (4). The §4.2 fail-loud pre-pass fires on exactly these, which is cheap to tolerate
explicitly.

### 2.1 Ground truth is wrapped at every level

`possible_answer` wraps **every leaf, at every nesting level**, in a list of acceptable values, and
uses `null` / `""` for "omitted":

```json
{"body": {"airConJobMode": ["AIR_CLEAN"], "enabled": [true]}}
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ still wrapped, one level down
```

Unwrapping only the top level leaves `["AIR_CLEAN"]` as the value of a parameter declared `string`,
which reads as grammar over-constraint when it is nothing of the sort.

There is a second, sharper trap in the same encoding. `""` among a parameter's acceptable values
means **"this parameter may be omitted"**, and omission is then always an acceptable answer — so it
must be *preferred*, not skipped past:

```json
{"unit": ["", "N/A"]}     against     enum ["seconds", "milliseconds"]
```

reads as "omitted, or N/A", and since `"N/A"` is not in the enum, **omission is the only valid
reading**. Taking the second entry looks like the grammar rejecting a correct answer.

**Measured impact of getting both right, on `live_simple`:**

| harness | ground truth accepted |
|---|---|
| unwrap top level only | 82.5% |
| + unwrap recursively | 93.8% |
| + prefer omission on `""` | **98.8%** |

**Three times in Phase 1 an apparent over-constraint turned out to be the harness misreading BFCL's
ground-truth encoding.** `bfcl_data.materialize_ground_truth` now handles all three and
`tests/test_bfcl_data.py` pins them. The lesson generalises: do not report an over-constraint rate
without first auditing the harness — a wrong one looks exactly like a real grammar bug.

---

## 3. §4.7 re-measured on real BFCL-Live

SPEC projected 4–8 min per schema and **4–7 days serially** for BFCL-Live's 1,351 records. Both
numbers below are far smaller, but they measure different things and the distinction matters.

### 3.1 Lift only (regex → DFA → `get_transitions`)

1,351 records → **4,549 schemas** (records carry 1–8+ functions).

| | |
|---|---|
| compiled | 4,538 / 4,549 (the 11 are §1.3's `any`, since fixed) |
| median | **0.42 s** |
| p90 / p99 | 0.78 s / 1.66 s |
| max | 15.8 s — the `{}` / `any` wildcard shape |
| total, serial | **2,272 s = 37.9 min** |
| states median / p90 / max | 114 / 249 / **3,573** (raw, pre-minimization) |

### 3.2 Full pipeline, which is the number to quote

`python -m diffgemma_fa.compile.tasks.bfcl --splits live --jobs 12` — regex, lift, **Valmari**, and
class tables:

| | |
|---|---|
| compiled | **4,549 / 4,549, zero failures** |
| all deterministic | **4,549 DFA / 0 NFA** |
| wall, 12 workers | **2,562 s = 42.7 min** |
| CPU, serial equivalent | **26,908 s = 7.5 h** |
| per schema: median / p90 / max | **4.28 s / 9.44 s / 313.8 s** |
| `\|S\|` median / p90 / **max** | 97 / 205 / **595** |
| bucket histogram | 16:68 · 32:239 · 64:831 · **128:2,292** · 256:797 · 512:308 · 1024:14 |
| edges median / max | 270 / 3,068 |
| largest tree | **2.14 GB** |

**Pure-Python Valmari dominates** — ~1 s per schema against ~0.15 s for the lift. SPEC §4.5's
"pure Python is fine" holds (43 min across 12 cores), but the honest full-pipeline figure is
**hours serially, not 38 minutes**. Still ~250× better than the original projection, and **the
vocabulary pre-filter is unnecessary and has not been built.**

The one shape worth watching remains `{}` / missing `type` / `additionalProperties: true` / BFCL
`any`: the 313.8 s worst case, and the realistic route to "regex too large".

### 3.3 The chain-path fallback is not needed for BFCL

The `states_max = 3,573` in §3.1 is the **raw lifted** count and is misleading for dispatch. After
Valmari and stop-token augmentation the largest BFCL-Live automaton is **595 states**. Every
grammar lands in the 1024 bucket or below, the largest tree is **2.14 GB** against ~20 GB of
measured headroom, and `needs_chain_path` never fires.

So SPEC §5.6's chain fallback — and the paper's quoted 2,459-state worst case — do **not** bind on
BFCL. The fallback remains necessary only for Spider (open question 0a). The mechanism is
implemented and flagged (`bucket_size(..., allow_oversize=True)` →
`CompiledAutomaton.needs_chain_path`) so Phase 3's dispatcher can honour it, but on this benchmark
it is dead code.

---

## 4. Numbers SPEC asks for that are unpublished

All measured over the full 4,549-schema BFCL-Live compile.

### 4.1 Label-class dedup ratio (§4.4 Layer 1, open question 9)

Measured **per edge**, which is the quantity §4.4 actually defines — Phase 0 measured distinct
label sets per *state*, a different and less useful number.

- **1.69 median, 1.79 mean, 14.69 max** edges per class
- **151 classes median, 532 max**

SPEC hoped "170k edges → O(10²–10³) classes". The class *count* lands squarely in that range
(151–532), so the layer behaves as designed. The *ratio* is low (1.69) only because these grammars
are small — 270 edges median — so there is little repetition to exploit. Re-measure on Spider,
where the class layer is load-bearing.

### 4.2 Post-lift minimization ratio (§4.5, unpublished)

SPEC notes nobody in the literature minimizes *after* the token lift, and asks for the number.

- **1.09× mean state reduction, 9.95× max**
- Only **21 of 4,549** schemas (0.5%) reduce by more than 1.5×

So the second pass is marginal for almost everything and dramatic for a handful. It also **costs
~1 s per schema and is the dominant term in the compile budget** (§3.2). Recommendation: keep it —
43 min for the whole benchmark is cheap, and a 9.95× reduction on the largest grammars is exactly
where tree memory matters — but make it a flag, since 99.5% of schemas gain almost nothing.

### 4.3 Complement sizes and `K_max` (§4.4 Layer 2b)

The largest label class in every real BFCL schema covers **~99.6% of the vocabulary**
(max observed 261,154 of 262,141), giving `|N_c|` of **987–1,591**.

**SPEC's `K_max = 256` default is too small by ~6×.** With it, every one of these classes falls
back to positive storage in the max table and Layer 3's `nnz ≈ 25,000` budget is violated by ~four
orders of magnitude. `classes.py` defaults to **1,100** and additionally *derives* `K_max` from the
grammar (`auto_k_max`), asserting `K > max_c |N_c|` at build time.

### 4.4 The class-layer memory budget holds

SPEC §4.4 Layer 3 predicts `nnz ≈ 25,000` and ~100 KB of tables (plus a separate 512 KB argmax
table). Measured across all 4,549 schemas:

- `nnz_sum` **9,125 median, 51,200 max**
- total class tables **76.2 KB median, 425.2 KB max**

Both sit within the predicted order. The four-layer scheme delivers what §4.4 claims.

---

## 5. Phase 1 exit criteria (SPEC §8)

From `artifacts/phase1_roundtrip.json` and `artifacts/bfcl_compile_report.json`.

### 5.1 "Compiles all 6 task grammars"

BFCL/xLAM JSON ✅ (4,549/4,549), BFCL/xLAM Python ✅, Sudoku ✅, Countdown ✅, GSM-Symbolic ✅,
refusal branch ✅. **Spider ✗** — deferred, see §5.3.

### 5.2 Round-trip, `BFCL_v4_live_simple` (258 records, 258 automata)

| direction | result |
|---|---|
| FA → random walk → `json.loads` | **5,157 / 5,160 (99.94%)** |
| ground truth → FA | **255 / 258 (98.8%)** |
| all deterministic | 258 DFA / 0 NFA |

**The three remaining ground-truth rejections are both of the shapes SPEC already flags as
hazardous**, not general grammar defects:

| record | cause |
|---|---|
| `live_simple_117-73-0` | `input_value: {"type": "any"}` — the §4.2 wildcard; viable prefix, not accepting |
| `live_simple_122-78-0` | `model: {"type": "any"}` — same |
| `live_simple_189-114-0` | array of objects (`data: array of dict with properties`); rejected |

Left open deliberately. Both shapes are opt-in (`allow_wildcard=True`) and the wildcard is already
documented as 26–130× the median compile cost; chasing outlines' internal expansion for two records
is not the best use of Phase 2. **Re-check once Phase 2's exactness harness exists**, which is a
better instrument for it.

**The 3 invalid-JSON samples out of 5,160 are a *sampler* artifact, not a grammar defect.** The
random walk picks tokens from classes with ~260k members; a token sequence the FA legitimately
accepts can decode to invalid UTF-8, which `json.loads` then rejects. The FA accepted all 5,160.

### 5.3 Coverage — stated, not silently capped

Per CLAUDE.md's rule, everything not covered:

- **Round-trip is measured on `live_simple` only** (258 records, single-function). The
  `multiple` / `parallel` splits need the union-of-functions and call-list grammars, which exist
  (`tasks/bfcl_python.py: build_calls_regex`) but are not yet round-tripped.
- **The Python-format grammar is unit-tested but not round-tripped against BFCL ground truth.**
- **Nesting depth is bounded at `d = 2`** in the Python grammar; the coverage cost of that bound
  against ground-truth answers is not yet measured.
- **Spider is not implemented.** It is the NFA case, and SPEC §5.6/open question 0a says the dense
  formulation is structurally infeasible there (778 GB tree). Deferring until Phase 3 establishes
  what the chain path can actually carry.
- **xLAM, GSM-Symbolic, Countdown and Sudoku grammars exist and are unit-tested, but no dataset has
  been downloaded**, so none has been round-tripped against real ground truth. xLAM is gated on HF.
- **Sudoku is 9×9 in the shipped `hackable_diffusion_adapter` eval, not SPEC §4.8's 4×4.**
  `sudoku_regex` is size-generic; which one to evaluate is still open.

---

## 6. What is green

`pytest tests/ -q` → **270 passed**, covering:

- Valmari: language preservation on 100 random partial DFAs, idempotence, determinism of the
  output, state-map residual-language agreement, trimming, the empty language, the completion trap,
  and a 262k-scale alphabet.
- Class tables: both polarities round-trip to the true member set; SUM/MAX flag arrays are distinct
  objects and are allowed to disagree; `K > max|N_c|` enforced; the full-vocab class costs nothing.
- Automaton: all `end_tokens` wired; the post-stop tail is Σ and has **unit emission mass**;
  `d(s)` computed after augmentation (grammar-final `d = 1`, not 0); budget semantics; dead padding
  states infinite; start vector is a vector; `(2L−1)` factor kept; oversize buckets flagged;
  save/load bit-exact.
- Schema: every silently-dropped keyword raises; `enum`+`type` and `properties`+`type` are benign;
  the BFCL dialect and the Java/JS leakage map correctly; `any` handling.
- BFCL Python grammar: **positional arguments are impossible**, cross-checked with `ast.parse` on
  every accepted string.
- Task grammars: Sudoku preserves givens (and rejects overwriting each one individually);
  Countdown; GSM-Symbolic guillemets; the refusal branch cannot start with `[` or `{`.
