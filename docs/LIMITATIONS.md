# Known limitations of the compiled grammar

**What this document is for.** The guarantee this project makes is exact: every
emitted token sequence is accepted by the compiled automaton. That says nothing
about whether the compiled automaton is the language you meant. Where the two
differ, the model is *structurally prevented* from emitting a correct answer —
and because constraint satisfaction still reads 1.000, the failure looks like a
model error rather than a grammar error.

Each entry below states the regime, the measurement, the root cause, and whether
it is fixable. Measured on `outlines-core 0.2.14`, `JSON_WS`, at commit
`1aa8ccf`.

---

## 1. `\uXXXX` escapes are not in the language

```
{"v": "é"}     REJECTED     (valid JSON)
{"v": "café"}       accepted
```

**Root cause.** outlines' string production is

```
"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt])*"
```

The escape alternation is `\\["\\/bfnrt]` — the eight two-character escapes.
RFC 8259 §7 defines `escape = ["\\/bfnrt]` **or** `u` followed by four hex
digits. **The `u` branch is simply absent.**

Note the character class also excludes `\x7F-\x9F` (DEL and the C1 controls),
which RFC 8259 permits unescaped; only `\x00-\x1F` must be escaped. A second,
smaller narrowing from the same production.

**Fixable: yes, straightforwardly.** Add `|\\u[0-9a-fA-F]{4}` to the
alternation. The clean fix is upstream in outlines-core; locally it is a
targeted rewrite of the generated regex, which is the same kind of
post-processing this repo already does (`lift.py` wraps every regex as
`(?:…)\z`, and `schema.py` expands typeless nodes). Cost is one extra branch
per string position — negligible against `|S|`.

**Why it has not been done.** No arm has needed it: BFCL ground truth contains
no `\uXXXX`, and models emit literal UTF-8 rather than escapes. It is latent
until a schema requires escaped output.

---

## 2. Unsigned exponents are not in the language

```
{"v": 1e5}      REJECTED     (valid JSON)
{"v": 1e+5}     accepted
```

**Root cause.** outlines' number production ends

```
([eE][+-][0-9]+)?
```

RFC 8259 §6 is `exp = ("e" / "E") [ minus / plus ] 1*DIGIT` — **the sign is
optional.** outlines requires it.

**Fixable: yes, trivially.** `[+-]` → `[+-]?`. One character. Same delivery
options as (1): upstream, or a local rewrite.

**Why it has not been done.** Same reason — nothing has needed it yet. Worth
noting this one is more likely to bite than (1), since `1e5` is a natural way
for a model to write a large number.

---

## 3. Nesting deeper than 4 is not in the language

```
{"v": {"a":{"b":{"c":1}}}}       accepted   (depth 4)
{"v": {"a":{"b":{"c":{"d":1}}}}} REJECTED   (depth 5, valid JSON)
```

**Root cause.** JSON is recursive; a finite automaton is not. outlines resolves
this by **unrolling the recursion to a fixed depth**. Beyond it, the production
stops.

**Fixable: only by paying for it, and the price is steep.** Each extra level
multiplies the regex — SPEC §4.2 warns that a too-large regex lifted over a
262,144-token vocabulary is how this host was OOM-killed once, and
`UnsupportedSchemaError` already quotes a "~26–130× compile cost" for the
wildcard expansion for the same structural reason. Depth is the axis where that
cost is genuinely exponential rather than an artifact of spelling — unlike the
`type: any` case, where a claimed 212× blowup turned out to be a measurement
error and the correct grammar was *twelve characters shorter* than the broken
one (see `docs/LOG.md`, 2026-08-12).

**The honest position:** this is a real bound on what the method can express,
not an oversight. A depth-5 schema is compilable in principle; whether it fits
in the `|S|` budget is an empirical question per schema, and the build gate
(`b58fd84`) will refuse it loudly rather than mis-decode it.

---

## What is *not* a limitation

Checked and confirmed accepted, because an earlier draft of this list was wrong
on two of them:

| construct | status |
|---|---|
| whitespace inside arrays/objects (`[ 1 , 2 ]`) | accepted |
| signed exponent (`1e+5`) | accepted |
| literal non-ASCII (`"café"`) | accepted |
| two-character escapes (`\t`, `\n`, `\"`) | accepted |
| nesting to depth 4 | accepted |

That correction matters: a limitations list nobody has measured is worse than
none, because it gets quoted.

---

## Why these exist at all

All three come from `outlines-core`, not from this project's compiler, and they
are **identical before and after** the `type: any` fix — pinned by
`tests/test_audit_wildcard.py::test_the_wildcard_narrowings_are_inherited_not_introduced`,
which asserts each is rejected by *both* the pre-fix wildcard and the seven-way
`anyOf`. That test exists so the distinction cannot rot: it is the difference
between "our fix narrowed the language" and "the layer beneath us was always
this narrow".

This repo has already found and fixed two genuine defects in the same layer:

- **an unparenthesised alternation** for typeless nodes, which made the grammar
  reject the valid object and accept bare scalars (fixed in `15cdc47`);
- **dropped edges** in the token-DFA index, where `regex-automata` reports
  matches one byte late so an edge leaving an accepting state was silently
  discarded — which made repeated groups compile to their minimum count and
  broke Countdown entirely (root-caused in `e2be58c`).

So the pattern is established: the layer is fixable, and defects in it are
findable by differential testing the compiled automaton against `re.fullmatch`
rather than trusting the generator.

## Priority, if someone picks this up

1. **Unsigned exponent** — one character, most likely of the three to be hit.
2. **`\uXXXX`** — one alternation branch, needed the moment a schema wants
   escaped output.
3. **Nesting depth** — not a bug; measure the `|S|` cost per schema and let the
   build gate refuse what does not fit.

For (1) and (2), prefer a fix upstream in outlines-core over local regex
surgery. Both are RFC deviations that anyone using the library hits, and this
repo has already had to work around two more.
