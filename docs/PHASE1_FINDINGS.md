# Phase 1 findings — the constraint compiler

Date: **2026-07-27**. Stack as `docs/ENV.md`. Deliverable: `diffgemma_fa/compile/`.

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
(`lift.py: drop_labels`), real BFCL schemas lift to genuine DFAs — measured **256/256 DFA**.

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
which reads as grammar over-constraint when it is nothing of the sort. **Measured impact: ground
truth acceptance went 82.5% → 93.8% on `live_simple` purely from unwrapping recursively.**
`bfcl_data.materialize_ground_truth` does it properly. Worth stating loudly because the wrong
number looks exactly like a real bug.

---

## 3. §4.7 re-measured on real BFCL-Live — 38 minutes, not 4–7 days

1,351 records → **4,549 schemas** (records carry 1–8+ functions).

| | |
|---|---|
| compiled | **4,538 / 4,549** (the 11 are §1.3's `any`, since fixed) |
| median | **0.42 s** |
| p90 / p99 | 0.78 s / 1.66 s |
| max | **15.8 s** — the `{}` / `any` wildcard shape |
| **total, serial** | **2,272 s = 37.9 min** |
| states median / p90 / max | **114 / 249 / 3,573** |

SPEC originally projected 4–8 min per schema and **4–7 days serially**; Phase 0's hand-written
estimate of ~0.9 h was the right order and the real figure is 0.63 h. **The vocabulary pre-filter
is unnecessary and has not been built.**

The one shape worth watching remains `{}` / missing `type` / `additionalProperties: true` / BFCL
`any`: 26–130× the median, and the realistic route to "regex too large".

### 3.1 `states_max = 3,573` exceeds every usable tree bucket

Above the 2,459 SPEC quotes as the paper's largest BFCL DFA, and above the top bucket the measured
~20 GB of free HBM admits (2048 → 8.59 GB; 4096 → 34.3 GB). **SPEC §5.6's chain-path fallback is
not hypothetical.** `bucket_size(..., allow_oversize=True)` flags these through
`CompiledAutomaton.needs_chain_path` instead of aborting the run; Phase 3's dispatcher must honour
it, and the eval must report the tree/chain split per request.

(That figure is the *raw lifted* count. After Valmari and augmentation the `live_simple` maximum is
much smaller — see §5.)

---

## 4. Numbers SPEC asks for that are unpublished

### 4.1 Label-class dedup ratio (§4.4 Layer 1, open question 9)

Measured **per edge**, which is the quantity §4.4 actually defines — Phase 0 measured distinct
label sets per *state*, which is a different and less useful number.

- `live_simple` mean: **1.68 edges per class**

Lower than SPEC's hoped "170k edges → O(10²–10³) classes", because these grammars are small
(83 states mean) and their edges are genuinely diverse. The ratio should be re-measured on the
large `|S| ≈ 2,000+` grammars where the class layer actually matters.

### 4.2 Post-lift minimization ratio (§4.5, unpublished)

SPEC notes nobody in the literature minimizes *after* the token lift, and asks for the number.

- `live_simple` mean: **1.11× state reduction**
- **max: 9.29×**

So the second pass is usually marginal but occasionally dramatic. It costs ~1 s per schema in pure
Python — the dominant term in the per-schema budget at this grammar size — so it is worth keeping
for the tail but is a candidate to make optional.

### 4.3 Complement sizes and `K_max` (§4.4 Layer 2b)

The largest label class in every real BFCL schema covers **~99.6% of the vocabulary**
(max observed 261,154 of 262,141), giving `|N_c|` of **987–1,585**.

**SPEC's `K_max = 256` default is too small by ~6×.** With it, every one of these classes falls
back to positive storage in the max table and Layer 3's `nnz ≈ 25,000` budget is violated by ~four
orders of magnitude. `classes.py` defaults to **1,100** and additionally *derives* `K_max` from the
grammar (`auto_k_max`), asserting `K > max_c |N_c|` at build time.

---

## 5. Phase 1 exit criteria (SPEC §8)

*(Filled from `artifacts/phase1_roundtrip.json` and `artifacts/bfcl_compile_report.json`.)*

### 5.1 Round-trip, `BFCL_v4_live_simple` — see §6 for coverage caveats

| direction | result |
|---|---|
| FA → random walk → `json.loads` | **pending final run** |
| ground truth → FA | **pending final run** |

### 5.2 Coverage — stated, not silently capped

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

`pytest tests/ -q` → **256 passed**, covering:

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
