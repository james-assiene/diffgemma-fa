# Phase 5 findings — entropy calibration, J0 vs J1, and where the accuracy goes

Date: **2026-07-28**. SPEC §3.4's calibration, plus the diagnosis Phase 4 left open.

**Headline: constraint satisfaction is 12/12 everywhere and is insensitive to every knob tried.
Argument accuracy is 5–10% and the entropy bound does not explain it.** The cause is now
characterised in three steps, only the first two of which are fixed.

---

## 1. SPEC §3.4 calibration — a negative result

Sweep exactly as SPEC prescribes: `entropy_bound ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}` (stock
default `0.1`) at a fixed 48-step budget, `--variant=j1 --emission=sample`, `live_simple`.

| bound | CS | parsed | non-empty args | arg accuracy |
|---|---|---|---|---|
| 0.003 | **12/12** | 12/12 | 7/12 | 1/19 = 0.053 |
| 0.01 | **12/12** | 12/12 | 6/12 | 1/19 = 0.053 |
| 0.03 | **12/12** | 12/12 | 8/12 | 1/19 = 0.053 |
| **0.1 (stock)** | **12/12** | 12/12 | 7/12 | 2/19 = 0.105 |
| 0.3 | **12/12** | 12/12 | 4/12 | 1/19 = 0.053 |
| 1.0 | **12/12** | 12/12 | 5/12 | 2/19 = 0.105 |

**Read this as "no effect", not as "0.1 and 1.0 are best".** The whole accuracy column is 1 or 2
successes out of 19; the 95% interval on 1/19 is roughly [0.001, 0.26] and every row overlaps every
other. The non-empty column moves between 4 and 8 with no monotone trend. At `n = 12` this table
cannot distinguish the bounds and does not claim to.

**What it does establish:**

- **CS is completely insensitive to the entropy bound** — 12/12 at every value, which is what you
  want: the guarantee is structural, not statistical.
- **SPEC §3.4's premise is not confirmed on this slice.** It warns the stock defaults "will
  silently produce garbage" under constrained marginals and that recalibration is *required*. The
  stock `0.1` is as good as anything tried, and recalibration is **not** where the accuracy is.

**Coverage, stated:** 12 records of one split, not SPEC's 100-example dev slice; one split, not a
dev set spanning tasks; and `entropy_threshold` (the `EntropyEarlyStop` half of §3.4's 2-D grid)
**was not swept at all**. Enough to rule the bound out as the dominant factor, not enough to select
a production value.

Accuracy uses **BFCL's own normalisation** — lowercase and strip `",./-_*^` (SPEC §4.8) — not exact
equality. An earlier version used strict equality and understated accuracy; those numbers are
superseded. Every prediction is saved in `artifacts/phase5_*.json` so the metric can be re-derived
offline without another GPU run.

---

## 2. Open question 2 resolved: J0 starves the emission, J1 does not

SPEC lists J0 vs J1 as "genuinely open" and picks J0 (*"J0 — ship this"*), with J1 carrying the
risk of feeding the model grammar-valid noise it was never trained to denoise.

**Measured, J0 is the one that fails.** Per-denoising-step trace, same prompt, same grammar, same
bound:

| step | J0-map | J1-sample |
|---|---|---|
| 12 | `{"user_id":0}` | `{"user_id":17890}` |
| 14 | `{"user_id":1}` | `{"user_id":77890,"special":":black"}` |
| 21–47 | flips 0/1/6/8 **at H = 0.0000** | stable |

Ground truth `7890` / `black`.

The tell is steps 21–47: mean entropy is **0.0000** — maximally confident — and the constrained MAP
is still arbitrary. A confident model whose MAP is arbitrary is confident about something else, and
it is: under J0 the trajectory is stock uniform renoising, so the model never sees JSON, converges
on a *natural-language* answer, and leaves its marginals peaked on tokens the grammar forbids. The
emission then has no signal but length and takes the shortest member of the language.

**J0 buys an unconditional guarantee at the cost of all feedback** — precisely what §3.7 means by
"its entire value is the self-conditioning feedback and the entropy signal".

Ship **J1**, or J0 **plus `--self-cond=constrained`** (§3.7's mask in `logit_shaper`, still
unimplemented) — the other channel by which the constraint can reach the model's inputs. Testable
as a 2×2; only one cell is built.

This also **retires the Phase 4 hypothesis** that MAP's length bias was the cause. J1 uses the same
joint machinery on the same grammar and does not collapse.

---

## 3. SPEC §4.2 is wrong about four keywords

SPEC lists `minLength`, `maxLength`, `minItems`, `maxItems` among the keywords outlines "silently
ignores". Measured against outlines-core 0.2.14:

| keyword | verdict |
|---|---|
| `minLength`, `maxLength`, `minItems`, `maxItems` | **ENFORCED** (the regex rejects a too-short string / too-long array) |
| `minimum`, `maximum`, `multipleOf` | ignored — SPEC correct |

Not a cosmetic correction. With the wrong four in the fail-loud list, callers must `allow`-list
constraints that actually work, which desensitises the very signal the pre-pass exists to give for
the bounds that really are dropped. `_SILENTLY_DROPPED` corrected; tests updated to the measured
behaviour.

---

## 4. Where the accuracy actually goes

The per-argument detail is far more informative than the sweep. At the stock bound:

| argument kind | outcome |
|---|---|
| **enum** (`type: comfort`, `type: plus`) | ✅ **correct** |
| free-form string (`location`, `loc`, `repos`) | `""` or a single junk character |
| integer (`user_id`, `time`) | present but wrong (`77890` for `7890`, `22` for `600`) |
| optional (`aligned`, `unit`) | omitted |

**Enums succeed and free strings collapse.** That is a very specific signature: where the automaton
narrows the choice to a handful of literals, the model picks the right one; where it admits ~260k
tokens per position, nothing wins against closing the string.

### 4.1 Fixed: the empty string was a self-consistent fixed point

BFCL schemas never carry `minLength`, so the grammar faithfully permits `""` — and under J1 the
trajectory *is* the constrained draw, so the model sees `{"location":""}` in its own canvas from
step 0 and confirms it. Entropy drops to zero and it is done. `require_nonempty_strings` gives every
**required, non-enum** string property `minLength: 1` (a compiler-side decision, applied only at the
top level so it cannot silently over-constrain optional or nested fields).

| | non-empty args |
|---|---|
| baseline | 7/12 |
| **with `minLength: 1`** | **12/12** |

### 4.2 Root cause found and fixed: the grammar forbade the model's own format

The `221B Baker Street` case — ground truth verbatim, wrapped in junk — said the content was there
and the decode was placing it badly. Two concrete causes, both mine, both fixed:

**(a) The channel header was never wired in.** Phase 0 measured that *every* generation opens with
`<|channel>NAME\n<channel|>` (ids `[100, 45518, 107, 101]`), and I rewrote SPEC §3.6 to say
`FA_total = HEADER · FA_grammar · STOP · Σ*` — then never implemented it. Measured at canvas
position 0 the grammar admitted **3 tokens, all variants of `{`, with token 100 forbidden**. The
model's habitual first token was impossible, so the entire canvas was decoded from an
off-distribution prefix. `prepend_channel_header` fixes it.

**(b) `whitespace_pattern=""` forbade the model's tokenisation.** I set it deliberately, reasoning
it "shrinks the automaton and costs nothing the benchmark scores". **That reasoning was wrong.**
Gemma puts the space *inside* the separator token (`": "`), so forbidding whitespace makes the
model's own rendering `{"user_id": 7890, "special": "black"}` **unacceptable to the grammar**. The
symptom was a leading `:` on essentially every string value — `":Divinópolis, MG"` against a ground
truth of `Divinópolis, MG`, a one-character defect BFCL's normalisation does not strip. The cost of
allowing whitespace is **8 states** (43 → 51) against ~20 GB of tree headroom.

**Measured effect, same 12 records, same bound, J1-sample:**

| configuration | non-empty | argument accuracy |
|---|---|---|
| baseline | 7/12 | 1/19 = **5.3%** |
| + `minLength: 1` | 12/12 | 1/19 = 5.3% |
| + channel header | 12/12 | 1/19 = 5.3% |
| **+ whitespace allowed** | **12/12** | **7–8/19 = 37–42%** |

`user_id: 7890` and `special: black` are now **exact** (they were `77890` and `:black`), as are
`2020 Addison Street, Berkeley, CA, USA`-class values, `comfort`, `plus`, `celsius`,
`Divinópolis, MG`, `Riga, Latvia`, `London, UK`.

**A trap this exposed, worth recording separately.** The first header implementation used a `Σ*`
self-loop for the channel name. That loop has emission mass ~1.0 — exactly like the unscored
`ACC --Σ--> ACC` tail — so it is *free* to stay in, and the joint MAP consumed all 64 canvas
positions inside it without ever closing the header. **Any Σ-labelled self-loop in a grammar is an
attractor for a joint decode.** The name repetition is now bounded (`Σ_name{1,8}`), and
`test_channel_header_name_repetition_is_BOUNDED` asserts no self-loops exist.

### 4.3 Residual: string boundary placement

The remaining misses at 7/19 fall into two classes, both about *where the string ends*:

| class | example | count |
|---|---|---|
| trailing junk | `2020 Addison Street, Berkeley, CA, USA1  ` (want `…USA`) | 2 |
| truncated | `Tel` (want `Tel Aviv, Israel`), `Hyderabad`, `Naples,`, `San Francisco` | 4 |
| model judgement | `Yosemite National Park, Mariposa, CA` (want `Mariposa, CA`) | 1 |

Not a format failure — every one is a valid member of the language. This is the fixed-canvas /
variable-length tension narrowed to its last component: the decode chooses the closing quote's
position, and gets it slightly wrong in both directions.

### 4.4 Previously: the content is *misplaced*, not missing

Accuracy did **not** improve (1/19). The model mostly emits one junk character to satisfy the
minimum. But one output is decisive:

```
{"loc":":{{    loc_\":221B Baker Street, Berkeley, CA, USA1  ","type":"plus","time":66000}
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ground truth, verbatim
```

**The model has the right answer and the decode misplaces it.** So this is not a knowledge failure
and not a grammar failure — the emitted string is a valid member of the language, and the content is
in it.

The likely mechanism, stated as a hypothesis and **not yet tested**: the marginals are positional
over a fixed 256-wide canvas, while the grammar's value fields start wherever the preceding tokens
end. Under J1 the constrained draw changes field lengths between denoising steps, so every
subsequent position shifts and the model can never lock onto an alignment. That would be a genuine
tension between variable-length grammars and fixed-canvas diffusion, not an implementation bug — and
it is the obvious next thing to test (e.g. by measuring whether field start positions are stable
across steps).

---

## 5. State of the numbers

**Nothing here is an accuracy claim.** CS = 12/12 is real and asserted by an independent simulator,
but SPEC §3.8 already warns that CS alone "measur[es] nothing" if the output is not content-bearing —
and §4 above shows it largely is not. The honest summary is:

- format: solved (12/12 CS, 12/12 parsed, 12/12 non-empty with `minLength`);
- content: 5–10% argument accuracy, cause characterised in §4.2, unfixed;
- the entropy bound: ruled out as the dominant factor.

A BFCL leaderboard number should not be quoted from this work until §4.2 is resolved.
