#!/usr/bin/env bash
# =====================================================================
# THE JSON HEADLINE, RE-MEASURED AT THE FIXED WILDCARD GRAMMAR.
# REVIEWED, NOT YET LAUNCHED.
#
# Supersedes scratchpad/rerun_1aa8ccf_LOCKED.sh, which ran two smokes and
# task_sudoku_mask and then CORRECTLY self-aborted at its own hard stop C
# (cs = 1/250 against a prediction of >= 2). Sudoku is out of scope here. Its
# guard design -- preflight A1/A2/A3, the .py fingerprint re-checked before
# every arm, condition-naming aborts, archive-with-logs, [skip]-requires-parse,
# per-arm config readback out of the artifact, require_n -- is reused verbatim
# below, because it was reviewed and it was right.
#
# ONE QUESTION: does constrained JSON decoding work, at 15cdc47's grammar?
#
# ---------------------------------------------------------------------
# 1. WHAT RUNS, AND WHAT DOES NOT
# ---------------------------------------------------------------------
# docs/RESULTS.md's headline is two rows:
#
#                        unconstrained   constrained (ws-tolerant)
#     schema valid           0.631            0.992
#     CS                     0.000            1.000
#     arg acc                0.639            0.650
#     exact call             0.500            0.523
#
# Both are PRE-GATE `records_available = 130`, no skips, so both INCLUDE
# live_simple_117-73-0 and live_simple_122-78-0 scored under the broken
# `type: any` grammar. The constrained row's single schema-validity failure IS
# live_simple_122-78-0, which emitted a bare `1` at CS = 1.000 because the
# grammar rejected the valid object and accepted bare scalars. 15cdc47 repairs
# exactly that.
#
#   ARM 1 (103 min)  exp_e5_grammar130_map -- the constrained row, re-measured.
#   ARM 0 (~4 min)   an n=4 unconstrained reproduction probe. Not the baseline;
#                    see section 3 for why the 81-minute baseline arm is NOT in
#                    this queue by default.
#
# THE UNCONSTRAINED BASELINE IS NOT RE-RUN. Section 3 is that ruling and its
# evidence. It is reachable without editing this file: `DGFA_RUN_UNC=1`.
#
# ---------------------------------------------------------------------
# 2. CONFIG FIDELITY -- EVERY FLAG, READ OUT OF THE STORED ARTIFACT
# ---------------------------------------------------------------------
# artifacts/exp_e5_grammar130_map.json, written 2026-07-31 23:44:
#
#   field in artifact        value    flag passed here
#   --------------------------------------------------------------
#   task                     bfcl_live_simple   --task bfcl_live_simple
#   variant                  j0                 --variant j0
#   emission                 map                --emission map
#   entropy_bound            0.1                --entropy-bound 0.1
#   whitespace               pretty             --whitespace pretty
#   fence                    True               --fence
#   ci_enums                 True               --ci-enums
#   nonempty_strings         True               --nonempty
#   seed                     0                  --seed 0
#   prompt_style             stock              --prompt-style stock
#   n                        130                --n 130
#   records_available        130                (asserted, hard stop 2)
#
# FIVE FIELDS THE ARTIFACT DOES NOT RECORD, and how each was resolved --
# against git, not by assumption. Absence means "the flag did not exist yet",
# and every one of them defaulted to the pre-existing behaviour:
#
#   --dtype    added 19a3eca 2026-08-01 00:31, i.e. 47 minutes AFTER this
#              artifact was written. Before it, eval/run.py passed no
#              constrained_dtype and model/sampler.py's field defaulted to
#              "float64" (`git show 19a3eca^:diffgemma_fa/model/sampler.py`,
#              line 120). So -> --dtype float64. Note 19a3eca made the CLI
#              default float32 for four days; 1f99542 retracted it and today's
#              default is float64 again -- passing it explicitly is not
#              ceremony, it is the difference between reproducing this arm and
#              reproducing the retracted one.
#   --offset   added 8c3be4c 2026-08-02 16:32. Before it the split was always
#              taken from the front -> --offset 0. records_available=130 with
#              n=130 is the same statement.
#   --confidence added 66042d9 2026-08-02 19:40. `mf` was the only accept rule
#              that existed -> --confidence mf.
#   --temp     added e4e7144 2026-08-02 21:43. The stock anneal was the only
#              schedule -> --temp stock.
#   --expand-wildcard  added 15cdc47 2026-08-12. THE CHANGE UNDER TEST. It
#              defaults to True today; it is passed explicitly because an
#              omitted flag is not the default you assume, and it is read back
#              OUT OF THE ARTIFACT below (hard stop 2) because a grammar-shaping
#              boolean invisible in the console log is precisely the defect the
#              readback exists to catch.
#
# --max-new-tokens is recorded by NO artifact in this repo. Resolved by the CLI
# default, which has been 256 since eval/run.py was created (9399e7b, line 87)
# and has never been changed. Passed explicitly anyway.
#
# TWO NON-FLAG DEPENDENCIES CHECKED FOR DRIFT rather than assumed:
#   * PROMPT_STYLES["stock"] == "" today, and build_prompt's stock branch is
#     byte-identical to 9399e7b's. ae144f3 (2026-08-05) added a `noorder`
#     branch above it and did not touch stock.
#   * PRETTY_WS == r"( |\n {0,6})?" at e12ed56, 19a3eca, e2be58c and HEAD.
#     e2be58c moved it into the WHITESPACE_PATTERNS dict without changing the
#     string. So `--whitespace pretty` still compiles the grammar this arm was
#     published at.
#
# ---------------------------------------------------------------------
# 3. THE RULING ON THE UNCONSTRAINED ARM: NOT WORTH ITS 81 MINUTES
# ---------------------------------------------------------------------
# The question is not "is unconstrained affected by the wildcard fix" in the
# abstract -- it is "would re-running it change any number in the headline".
# Measured on CPU from the artifact's own `rows`, not reasoned:
#
# (a) THE TWO AFFECTED RECORDS DO NOT MOVE. For live_simple_117-73-0 and
#     live_simple_122-78-0 the unconstrained arm emitted
#       {"input_value": "hi say Reverse"}
#       {"image_path": ..., "question": ..., "model": "vikhyatk/moondream2"}
#     and `metrics.schema_valid(parsed, normalize_bfcl_schema(p, expand))` is
#     True for BOTH records under BOTH expand_wildcard=False and True; both have
#     accepted=False under both. The expansion is `anyOf` over all seven JSON
#     types, which is a semantic no-op for jsonschema validation -- it is a
#     REGEX fix, and the unconstrained arm is not scored through the regex
#     except via `accepted`, which is 0/130 either way.
# (b) THE DENOMINATOR DOES NOT MOVE. That artifact is category 3: PRE-GATE
#     records_available=130, skipped_by_reason={}. Both records were already
#     inside its n. Post-fix it is 130 again, over the same 130 records.
# (c) `want` -- and therefore arg_accuracy and exact_call_rate -- is
#     materialised from `fn["parameters"]`, the RAW schema, never from `norm`.
#     expand_wildcard cannot reach it.
# (d) THE SCORER GAP IS ALREADY CLOSED ON CPU. The stored fields are the
#     pre-e2be58c scorer (arg 0.6413, exact 0.4692); RESULTS.md's headline is
#     0.639 / 0.500. Re-scoring the stored `rows` with today's metrics module
#     reproduces 0.6392 / 0.5000 EXACTLY -- and 0.6495 / 0.5231 for the
#     constrained arm, which is RESULTS.md's 0.650 / 0.523. So RESULTS.md's
#     headline already IS the current-scorer number for both rows; 81 GPU
#     minutes would buy a number that a two-second CPU rescore already has.
#
# WHAT THE 81 MINUTES *WOULD* BUY, stated honestly: the unconstrained decode
# itself has not been executed since 2026-07-28, and fifteen commits have
# touched diffgemma_fa/model|infer|compile since. None of them is on the
# unconstrained code path (sampler.py:467 returns before every constrained
# branch; nothing above it consumes randomness beyond `split(carry.rng)` and
# `sample_step`, both untouched), so the emission SHOULD be bit-identical. But
# "should" is reasoning, and this repo's rule is measurement. Hence ARM 0: an
# n=4 unconstrained run whose emitted text is diffed against the stored
# artifact's rows for the same four records. Four minutes to convert the
# exemption into a measurement, which is what "state the exemption's evidence"
# asks for.
#
# ARM 0 IS REPORT-ONLY, DELIBERATELY. A mismatch has two candidate causes --
# real code drift, or XLA recompilation nondeterminism -- and it cannot tell
# them apart, so it must not be allowed to abort the arm that answers the actual
# question. It also cannot see a per-record rarity: 4 records detects a
# systematic change, not a rare one. Read its line; if it reports a mismatch,
# the published baseline is not exactly reproducible at HEAD and re-running it
# (DGFA_RUN_UNC=1, 81 min) becomes worth the time. That call is a human's.
#
# ---------------------------------------------------------------------
# 4. THE FALSIFIABLE PREDICTION, AND WHAT A FAILURE WOULD MEAN
# ---------------------------------------------------------------------
# HARD STOP 1: exp_e5_grammar130_map must report schema_ok == n, i.e.
# schema_valid_rate == 1.0, against a published 129/130.
#
# Its one failure was live_simple_122-78-0, whose grammar could not emit a
# well-formed object and could emit a bare `1`; 15cdc47 repairs exactly that,
# and preflight A5 re-confirms on CPU that both wildcard records now compile
# clean AND still fail without the flag.
#
# BUT THE MECHANISM BEING REAL IS NOT EVIDENCE THAT IT IS THE ONLY CAUSE.
# The precedent is one day old: task_sudoku_mask was predicted to improve on
# exactly this shape of argument ("one mechanical cause, the fix removes it")
# and reproduced its published value to the record. So the stop is strict, and
# a failure here has at least THREE readings, which the artifact's own `rows`
# discriminate by naming the failing record:
#   (i)   the failing record is still live_simple_122-78-0 -> the wildcard fix
#         did not deliver what 15cdc47 measured it to deliver, and nothing
#         downstream of it should be believed;
#   (ii)  the failing record is a DIFFERENT one -> the wildcard fix worked and
#         something else moved. Four commits have touched the j0/map emission
#         path since 2026-07-31 (da1294a's complement-algebra fix is the one
#         that can change a MAP argmax; b58fd84's is_dfa polarity gates eq (8)'s
#         fast path; a41bf74 and 1aa8ccf are guards and a tie-break that is a
#         no-op on the compiled path). This re-run is therefore NOT a pure
#         wildcard-fix delta and must not be reported as one.
#   (iii) more than one record fails -> both.
# In every reading the answer is stop, not continue: the second arm exists only
# to be compared against this one.
#
# HARD STOP 2: records_available == 130, n == 130, skipped_by_reason == {}, and
# expand_wildcard == True in the artifact. A 128 here means the wildcard fix is
# not actually active in the run and n=130 elsewhere would be a coincidence.
#
# HARD STOP 3 (zp_gate): zero_partition 0 is clean; below 10% of n prints and
# continues; >= 10% stops. j0/map/mf does not run prefix_suffix, so 0 is the
# expectation and the published value.
#
# ---------------------------------------------------------------------
# 5. COMPARABILITY -- WHAT MAY AND MAY NOT BE CLAIMED AFTERWARDS
# ---------------------------------------------------------------------
# MAY be claimed:
#   * The headline PAIR remains legitimate. The constrained row moves to the
#     fixed grammar; the unconstrained row is grammar-INVARIANT by section 3
#     (a)-(c), i.e. it is the same number under either grammar rather than a
#     stale one. RESULTS.md's McNemar line for `constrained, ws-tolerant` vs
#     `unconstrained-sample` (b=7, c=4, p=0.55) is recomputable on CPU from the
#     two `rows` lists over the same 130 records afterwards.
#   * "The constrained decode is schema-valid on 130/130 at CS 1.000" -- if the
#     hard stop passes -- as a statement about the CURRENT code and grammar.
#
# MAY NOT be claimed without more work:
#   * ANY pairing of this row against the other constrained rows in RESULTS.md.
#     `j0-sample`, `j1-sample`, `j2-sample`, `j0-map` and `mask-sample` are all
#     pre-15cdc47, so on records 117 and 122 they decoded under a grammar that
#     rejects the correct object. After this queue the constrained side of the
#     table straddles two grammars. Do not average them.
#   * `mask-sample` additionally has NO artifact in artifacts/ right now: the
#     aborted queue archived it to artifacts/stale_pre_1aa8ccf/ and stopped
#     before re-running it. Eleven other tags are in the same state. Any
#     regeneration of RESULTS.md must account for that or it will silently drop
#     rows.
# THE CHEAP WAY OUT IS CPU-ONLY AND SUFFICIENT. Every artifact stores a
# per-record `rows` list carrying id / parsed / accepted / schema_ok, and the
# ground truth is regenerable from bfcl_data, so any paired comparison can be
# restricted to the 128 records common to both grammars -- or, better here,
# recomputed over all 130 with the two wildcard records flagged, since section 3
# shows they are grammar-invariant on the unconstrained side. No further GPU
# time is required to make the headline table honest. ~6.4 GPU-hours would be
# required to make the whole constrained BLOCK single-grammar; that is a
# separate, deliberate decision.
#
# ---------------------------------------------------------------------
# 6. WALL CLOCK
# ---------------------------------------------------------------------
#   arm 0  unconstrained repro probe, n=4      ~4 min
#   arm 1  exp_e5_grammar130_map, n=130       ~103 min (its own elapsed_seconds
#                                              is 6189.6 s; 15cdc47 compiles the
#                                              two wildcard schemas ~1.3x faster
#                                              and no records are added, since
#                                              this arm was already pre-gate
#                                              n=130, so call it 100-105)
#   = ~1 h 50 m. With DGFA_RUN_UNC=1: + ~81 min (4859.5 s) = ~3 h 12 m.
#
# ---------------------------------------------------------------------
# 7. LAUNCH
# ---------------------------------------------------------------------
#   D=/tmp/claude-1000/-home-ubuntu-diffgemma-fa/29da7bd9-0534-4f90-a9d4-d456e1f98acd/scratchpad
#   cp $D/json_headline.sh $D/json_headline_LOCKED.sh
#   chmod 444 $D/json_headline_LOCKED.sh
#   md5sum $D/json_headline_LOCKED.sh
#   nohup $D/json_headline_LOCKED.sh \
#         > /home/ubuntu/diffgemma_fa/logs/json_headline.log 2>&1 &
#   then record the PID, the md5 and the log path in docs/LOG.md.
# The script refuses to run if it is writable: editing a live bash script
# corrupts the running instance, and that has already happened here.
# =====================================================================
set -u

# ---------- preflight A1: immutability ----------
# RESOLVE $0 BEFORE THE cd. With this check placed after the cd, a relative
# invocation leaves $0 pointing at a path that no longer resolves, `[ -w ]` is
# false on a nonexistent file, and the guard silently passes on a WRITABLE
# script. It has to be absolute, and it has to be here.
SELF=$(readlink -f "$0" 2>/dev/null || echo "")
if [ -z "$SELF" ] || [ ! -f "$SELF" ]; then
  echo "!! cannot resolve this script's own path from '$0'. The immutability"
  echo "!! check cannot be performed, so it is not being skipped. STOPPING."
  exit 1
fi
if [ -w "$SELF" ]; then
  echo "!! $SELF is writable. Launch from a chmod 444 copy -- editing a live"
  echo "!! bash script corrupts the running instance, and that has already"
  echo "!! happened on this box."
  exit 1
fi

cd /home/ubuntu/diffgemma_fa || exit 1

# `source env.sh` under `set -u` is safe: the venv activate script guards every
# optional variable, and three previous queues ran ten arms this way. It sets
# PREALLOCATE=true / MEM_FRACTION=.90 on this dedicated box, so the queue must
# stay SERIAL -- two of these processes cannot coexist.
source env.sh

EXPECT_COMMIT=1aa8ccf634934dbbe4b5ad9820b24e77f2baaa19
QUEUE=json_headline
ARCHIVE=artifacts/stale_pre_15cdc47_json
RUN_UNC=${DGFA_RUN_UNC:-0}

# ---------- preflight A2/A3: code identity and clean tree ----------
# NOT `HEAD == EXPECT_COMMIT`. What every arm below is attributed to is the
# CONTENT of diffgemma_fa/, and a commit that does not touch it cannot change a
# number. The predecessor queue died on a docs/LOG.md commit made one minute
# after launch; HEAD is currently 4f9a6e3 ("Document the grammar's RFC 8259
# gaps"), which touches README.md and docs/ only, and
# `git diff --name-only 1aa8ccf 4f9a6e3 -- diffgemma_fa/` is empty. A HEAD
# equality check would refuse to launch on that. A code commit still stops the
# queue -- it changes a tracked file under diffgemma_fa/, so the diff below is
# non-empty (and, mid-queue, the edit changes an mtime before the commit even
# exists: check_tree).
HEAD_AT_LAUNCH=$(git rev-parse HEAD 2>/dev/null || echo none)
if ! git diff --quiet "$EXPECT_COMMIT" HEAD -- diffgemma_fa/ 2>/dev/null; then
  echo "!! STOP (preflight A2): diffgemma_fa/ at HEAD ($HEAD_AT_LAUNCH)"
  echo "!! differs from $EXPECT_COMMIT, which is the commit both arms below are"
  echo "!! written against and whose config resolutions were traced through the"
  echo "!! code:"
  git diff --name-only "$EXPECT_COMMIT" HEAD -- diffgemma_fa/ | sed 's/^/   /'
  echo "!! Re-review section 2 and section 4 against the new code first."
  exit 1
fi
if [ "$HEAD_AT_LAUNCH" != "$EXPECT_COMMIT" ]; then
  echo "[note] HEAD is $HEAD_AT_LAUNCH, not $EXPECT_COMMIT, but diffgemma_fa/"
  echo "       is identical between them -- commits since are docs/tests only."
  echo "       Recorded in every artifact as git_commit; the code identity is"
  echo "       recorded as code_base_commit."
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "!! STOP (preflight A3): the working tree is dirty. An unattributable"
  echo "!! number is worse than no number, because it looks like evidence."
  git status --porcelain | sed 's/^/   /'
  exit 1
fi
HEAD_SEEN="$HEAD_AT_LAUNCH"

tree_fp () { find diffgemma_fa -name '*.py' -printf '%T@ %s %p\n' | sort | md5sum | cut -d' ' -f1; }
tree_cfp () { find diffgemma_fa -name '*.py' -print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1; }
FP0=$(tree_fp)
CFP0=$(tree_cfp)

# ---------- the mid-queue guard ----------
# Named conditions. An earlier queue tested a three-way OR and printed only the
# fingerprint pair, so a HEAD change rendered as `fingerprint X -> X`, which
# reads like a bug in the guard and cost real diagnosis time.
check_tree () {
  local now cnow dirty code_dirty head_now stop=0
  now=$(tree_fp)
  dirty=$(git status --porcelain)
  head_now=$(git rev-parse HEAD 2>/dev/null || echo none)

  if [ "$now" != "$FP0" ]; then
    cnow=$(tree_cfp)
    echo "!! STOP -- CONDITION 1 of 2: THE CODE TREE MOVED UNDER THE QUEUE."
    echo "!!   test:  md5 of (mtime, size, path) over diffgemma_fa/**/*.py"
    echo "!!   launch: $FP0"
    echo "!!   now:    $now"
    if [ "$cnow" != "$CFP0" ]; then
      echo "!!   CONTENT also changed ($CFP0 -> $cnow): an agent edited"
      echo "!!   diffgemma_fa/ while this queue was measuring it."
    else
      echo "!!   content is unchanged ($cnow): the files were rewritten"
      echo "!!   byte-identically (checkout, formatter, touch). Still a stop --"
      echo "!!   a measurement must not straddle a write to the tree, and the"
      echo "!!   window between the write and the read is not auditable."
    fi
    stop=1
  fi

  # Scoped deliberately. The preflight already demanded a fully clean tree;
  # mid-queue, only dirt that can change what is being measured is fatal. A
  # docs/LOG.md write is exactly what killed an earlier queue and it cannot
  # move a number.
  code_dirty=$(printf '%s\n' "$dirty" | grep -E '(diffgemma_fa/|env\.sh)' || true)
  if [ -n "$code_dirty" ]; then
    echo "!! STOP -- CONDITION 2 of 2: UNCOMMITTED CHANGES UNDER diffgemma_fa/"
    echo "!!   test:  git status --porcelain, filtered to diffgemma_fa/ and env.sh"
    printf '%s\n' "$code_dirty" | sed 's/^/     /'
    stop=1
  elif [ -n "$dirty" ]; then
    echo "[note] tree is dirty OUTSIDE diffgemma_fa/ -- not a stop, recorded:"
    printf '%s\n' "$dirty" | sed 's/^/       /'
  fi

  if [ "$head_now" != "$HEAD_SEEN" ]; then
    echo "[note] HEAD moved $HEAD_SEEN -> $head_now. NOT a stop: attribution"
    echo "       rides the .py fingerprint above, which a code commit cannot"
    echo "       avoid tripping (the edit changes an mtime before the commit"
    echo "       exists). Recorded into every artifact as git_head_at_write."
    HEAD_SEEN="$head_now"
  fi

  if [ "$stop" -ne 0 ]; then
    echo "!! Everything measured from here would be unattributable. STOPPING."
    exit 1
  fi
}

echo "=== $QUEUE at $(git rev-parse --short HEAD)  $(date -u '+%Y-%m-%d %H:%M') UTC"
echo "=== tree fingerprint $FP0  (content $CFP0)"
echo "=== DGFA_DEDICATED=${DGFA_DEDICATED:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=== DGFA_RUN_UNC=$RUN_UNC  (1 = also re-run the 81-min unconstrained arm;"
echo "===   section 3 argues it buys nothing. Default 0.)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv || { echo "!! nvidia-smi failed"; exit 1; }
echo

# ---------- preflight A6: the automaton cache hazard ----------
if [ -d artifacts/fa/bfcl ] && [ -n "$(ls -A artifacts/fa/bfcl 2>/dev/null)" ]; then
  echo "!! STOP (preflight A6): artifacts/fa/bfcl/ is non-empty."
  echo "!! Those .npz files are keyed by FILENAME ({rec_id}__{fn_index}.npz),"
  echo "!! not by schema_hash, so a pre-15cdc47 automaton for a wildcard schema"
  echo "!! would be reused verbatim under the fixed grammar. Delete the"
  echo "!! directory by hand (it is not deleted here -- silently discarding"
  echo "!! hours of compiled automata is the worse failure) and relaunch."
  ls -la artifacts/fa/bfcl | sed 's/^/   /'
  exit 1
fi
echo "[ok] artifacts/fa/bfcl/ absent or empty -- no stale automaton can be reused"

# ---------- preflight A5: the denominator premise, on CPU ----------
# 15cdc47's claim, tested rather than trusted, in milliseconds, before any GPU
# time. The expand_wildcard=False leg is the anti-vacuity control: it is the
# assertion that FAILED before the fix, so a probe that cannot fail is a probe
# that has been broken by a refactor.
python3 - <<'PY' || exit 1
import sys
from diffgemma_fa.compile import bfcl_data, schema as S
from diffgemma_fa.compile.tasks.bfcl import ALLOW

WANT = {117: "live_simple_117-73-0", 122: "live_simple_122-78-0"}

def gate_bad(params, expand):
    """The build gate's own predicate, as pipeline.compile_json_schema runs it
    for eval/run.py's arms: --nonempty and --whitespace stock|pretty, so
    verify_strict is False and the deciding recompile is under JSON_WS with the
    caller's expand_wildcard. 'reordered-keys' is not a build failure on an
    ordered grammar (pipeline.py) and is filtered there too."""
    norm = S.normalize_bfcl_schema(params, expand_wildcard=expand)
    inst = S.synthesize_instance(norm)
    prepared = S.require_nonempty_strings(norm)
    rgx = S.build_regex(prepared, from_bfcl=False, allow=ALLOW,
                        allow_wildcard=True, whitespace_pattern=S.JSON_WS,
                        expand_wildcard=expand)
    _, bad = S.accepts_all_renderings(rgx, inst)
    return [b for b in bad if b != "reordered-keys"]

recs = list(bfcl_data.load(("BFCL_v4_live_simple.json",)))
fail = False
if len(recs) != 258:
    print(f"!! live_simple has {len(recs)} records, expected 258"); fail = True
for i, want in WANT.items():
    if i >= len(recs) or recs[i].id != want:
        got = recs[i].id if i < len(recs) else "<out of range>"
        print(f"!! index {i} is {got}, expected {want} -- the split moved, so"
              f" every offset/n in this queue means something else")
        fail = True
        continue
    p = recs[i].functions[0].get("parameters") or {}
    fixed, broken = gate_bad(p, True), gate_bad(p, False)
    print(f"[gate] {want}: expand=True bad={fixed}  expand=False bad={broken}")
    if fixed:
        print(f"!! {want} STILL fails the build gate with the wildcard fix on."
              f" The arm would report n=128 and hard stop 2 would fire after"
              f" 103 GPU-minutes instead of now.")
        fail = True
    if not broken:
        print(f"!! {want} passes the gate even WITHOUT the fix. The probe"
              f" cannot fail, so it is proving nothing -- do not read its"
              f" green as evidence.")
        fail = True
if fail:
    print("!! STOP (preflight A5): the n=130 premise is not established.")
    sys.exit(1)
print("[ok] preflight A5: both wildcard records compile clean at this commit"
      " and still fail without --expand-wildcard (control non-vacuous)")
PY
echo

mkdir -p logs "$ARCHIVE"

# ---------- archiving ----------
# `logs/` is gitignored and `run` opens logs/<tag>.log with `>`, so an
# unarchived log is destroyed unrecoverably by its own re-run. Idempotent: on a
# relaunch the archive copy exists and the original is left alone.
#
# ORDER MATTERS: `run` [skip]s a tag whose artifact already exists, so the
# archive of a tag MUST happen before that tag runs, not after.
archive_to () {                # archive_to <dir> <tag>...
  local dir="$1"; shift
  local t
  for t in "$@"; do
    if [ -f "artifacts/$t.json" ] && [ ! -e "$dir/$t.json" ]; then
      mv "artifacts/$t.json" "$dir/$t.json" && echo "[archived] $t.json -> $dir"
    fi
    if [ -f "logs/$t.log" ] && [ ! -e "$dir/$t.log" ]; then
      mv "logs/$t.log" "$dir/$t.log" && echo "[archived] $t.log -> $dir"
    fi
  done
}

write_archive_readme () {
  [ -e "$ARCHIVE/README.txt" ] && return 0
  cat > "$ARCHIVE/README.txt" <<'TXT'
Artifacts and console logs replaced by the json_headline queue.

Stale for ONE reason: 15cdc47 (`type: any` -> a parenthesised anyOf). Both of
these arms are docs/RESULTS.md headline rows and both were measured PRE-GATE at
records_available=130 with no skips, so both INCLUDE live_simple_117-73-0 and
live_simple_122-78-0 scored under a grammar that rejects the correct object and
accepts a bare `1`. The constrained arm's single schema-validity failure IS
live_simple_122-78-0.

Their replacements in artifacts/ carry git_commit, code_base_commit,
git_tree_fingerprint and queue fields; these do not.

Two further reasons not to cite the numbers stored in these files even as a
baseline:
  * the stored accuracy fields predate the 2026-08-08 scorer (e2be58c).
    docs/RESULTS.md carries the rescore, and the rescore is exactly reproducible
    on CPU from each file's own `rows` list.
  * the replacement ran at HEAD, and four commits have touched the j0/map
    emission path since 2026-07-31 (da1294a, b58fd84, a41bf74, 1aa8ccf). The
    delta is therefore not attributable to the wildcard fix alone.
TXT
  echo "[archived] wrote $ARCHIVE/README.txt"
}

# ---------- the runner ----------
# The config is printed back OUT OF THE ARTIFACT, on the [skip] path too: on a
# relaunch the readback is still the only record of what actually ran, and a
# wrong flag that is invisible in the console log is exactly the defect this is
# here to catch.
show_cfg () {                  # show_cfg <artifact path>
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
g = d.get
print("    cfg: variant=%s emission=%s conf=%s bound=%s ws=%s fence=%s ci=%s "
      "expand_wildcard=%s dtype=%s seed=%s offset=%s temp=%s prompt=%s "
      "nonempty=%s commit=%s code_base=%s" % (
          g("variant"), g("emission"), g("confidence"), g("entropy_bound"),
          g("whitespace"), g("fence"), g("ci_enums"), g("expand_wildcard"),
          g("dtype"), g("seed"), g("offset"), g("temp"), g("prompt_style"),
          g("nonempty_strings"), (g("git_commit") or "?")[:7],
          (g("code_base_commit") or "?")[:7]))
print("    n=%s avail=%s skips=%s cs=%s zp=%s oom=%s schema=%s (%s/%s) "
      "arg=%s (%s/%s) exact=%s  %.1f min" % (
          g("n"), g("records_available"), g("skipped_by_reason"), g("cs_rate"),
          g("zero_partition"), g("oom"),
          g("schema_valid_rate"), g("schema_ok"), g("n"),
          g("arg_accuracy"), g("arg_correct"), g("arg_total"),
          g("exact_call_rate"), (g("elapsed_seconds") or 0) / 60))
for r in (g("skipped_records") or []):
    print("      excluded: %s (%s)" % (r.get("id"), r.get("reason")))
for r in (g("zero_partition_records") or []):
    print("      Z==0: %s" % (r.get("id") if isinstance(r, dict) else r))
for r in (g("oom_records") or []):
    print("      oom: %s (bucket %s)" % (r.get("id"), r.get("n_states_bucket")))
# Name every schema-invalid record. Hard stop 1 rides on this column, and
# WHICH record fails is what discriminates its three readings (section 4).
bad = [r.get("id") for r in (g("rows") or []) if not r.get("schema_ok")]
if bad:
    print("      schema-invalid: %s" % bad)
PY
  return 0
}

run () {                       # run <module> <tag> <flags...>
  local mod="$1" tag="$2"; shift 2
  local out="artifacts/${tag}.json"
  check_tree
  # [skip] requires the artifact to PARSE, not merely to exist: a run killed
  # while writing leaves a truncated file, and skipping that would publish a
  # half-artifact as a measurement.
  if [ -f "$out" ] && python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \
       "$out" 2>/dev/null; then
    echo "[skip] $tag (exists and parses)"
    show_cfg "$out"
    return 0
  fi
  echo "=== $(date -u +%H:%M) $tag"
  echo "    $mod $*"
  python -m "$mod" --out "$out" "$@" > "logs/${tag}.log" 2>&1
  if [ ! -f "$out" ]; then
    echo "  !! $tag DIED -- no artifact"
    tail -20 "logs/${tag}.log" | sed 's/^/     /'
    return 1
  fi
  # Stamp attribution INTO the artifact so nothing here is ever unattributable.
  # Extra keys only; scripts/phase5_report.py reads named fields. Written to a
  # temp file and renamed: an interrupted in-place rewrite would truncate an arm
  # that already cost 100 GPU-minutes, and os.replace is atomic on the same
  # filesystem.
  QUEUE="$QUEUE" BASE="$EXPECT_COMMIT" LAUNCH="$HEAD_AT_LAUNCH" \
  FP="$FP0" CFP="$CFP0" \
  HEADNOW="$(git rev-parse HEAD 2>/dev/null || echo none)" \
  python3 - "$out" <<'PY'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
d["git_commit"] = os.environ["LAUNCH"]          # HEAD when the queue started
d["code_base_commit"] = os.environ["BASE"]      # diffgemma_fa/ is identical to this
d["git_head_at_write"] = os.environ["HEADNOW"]  # may differ: a docs commit
d["git_tree_fingerprint"] = os.environ["FP"]
d["git_tree_content_hash"] = os.environ["CFP"]
d["queue"] = os.environ["QUEUE"]
tmp = p + ".stamping"
with open(tmp, "w") as f:
    json.dump(d, f)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, p)
PY
  show_cfg "$out"
  return 0
}

# ---------- gates ----------
field () {                     # field <tag> <key>  -> value or ERR
  python3 -c 'import json,sys
try:
    v = json.load(open(sys.argv[1]))[sys.argv[2]]
    print("ERR" if v is None else v)
except Exception:
    print("ERR")' "artifacts/$1.json" "$2" 2>/dev/null || echo ERR
}

# HARD STOP 2: the denominator, per arm.
require_n () {                 # require_n <tag> <expect_avail> <expect_n>
  local tag="$1" ea="$2" en="$3" avail n skips ew
  avail=$(field "$tag" records_available)
  n=$(field "$tag" n)
  ew=$(field "$tag" expand_wildcard)
  skips=$(field "$tag" skipped_by_reason)
  if [ "$avail" = ERR ] && [ "$n" = ERR ]; then
    echo "!! $tag: no readable artifact -- the arm died before writing one."
    echo "!! See logs/$tag.log (its tail is above). STOPPING."
    exit 1
  fi
  if [ "$avail" != "$ea" ] || [ "$n" != "$en" ]; then
    echo "!! $tag: records_available=$avail n=$n, expected $ea and $en."
    echo "!! skipped_by_reason=$skips"
    if [ "$avail" = "$ea" ]; then
      echo "!! The cut is right and the BUILD GATE refused records. 15cdc47"
      echo "!! predicts 0 refusals over all 4,549 live schemas, and preflight"
      echo "!! A5 confirmed it on records 117 and 122 minutes ago -- so this is"
      echo "!! a THIRD record, or the fix is not reaching eval/run.py. Either"
      echo "!! way the n=130 denominator this queue is built on is wrong."
    else
      echo "!! records_available is the size of the CUT, so --offset or --n is"
      echo "!! not what this arm was published at. Every comparison behind this"
      echo "!! would be against a different set of records."
    fi
    echo "!! STOPPING."
    exit 1
  fi
  # skipped_by_reason must be EMPTY, asserted rather than inferred from n. It
  # cannot currently disagree with (avail, n), but it is the field a reader of
  # the artifact will look at, and asserting it costs nothing.
  if [ "$skips" != "{}" ]; then
    echo "!! $tag: skipped_by_reason=$skips, expected {}. Records left the"
    echo "!! denominator. STOPPING."
    exit 1
  fi
  if [ "$ew" != "True" ]; then
    echo "!! $tag: expand_wildcard=$ew in the artifact, expected True. The"
    echo "!! grammar is the pre-15cdc47 one and n=$n is a coincidence."
    echo "!! STOPPING."
    exit 1
  fi
  echo "[ok] $tag avail=$avail n=$n skips={} expand_wildcard=True"
}

# HARD STOP 3: zero_partition. 0 is clean; below 10% of n prints and continues;
# >= 10% stops. j0/map/mf runs neither prefix_suffix nor the linear up_sweep, so
# 0 is both the prediction and the published value.
zp_gate () {
  local tag="$1" zp n lim
  zp=$(field "$tag" zero_partition)
  n=$(field "$tag" n)
  if [ "$zp" = ERR ] || [ "$n" = ERR ] || ! [ "$n" -gt 0 ] 2>/dev/null; then
    echo "!! $tag: zero_partition or n unreadable (zp=$zp n=$n). STOPPING."
    exit 1
  fi
  if [ "$zp" -eq 0 ]; then
    echo "[ok] $tag zero_partition=0"
    return 0
  fi
  lim=$(( (n + 9) / 10 ))
  if [ "$zp" -ge "$lim" ]; then
    echo "!! $tag zero_partition=$zp of n=$n, at or above the 10% systematic"
    echo "!! threshold ($lim). STOPPING."
    exit 1
  fi
  echo "[!!] $tag zero_partition=$zp of n=$n, predicted 0. Below the 10%"
  echo "     systematic threshold, so the queue continues and the artifact"
  echo "     keeps its own column. Report the ids above in docs/RESULTS.md;"
  echo "     do not average them away."
}

# ================= ARM 0: the unconstrained reproduction probe, ~4 min ======
# Section 3. Report-only by construction: a mismatch cannot distinguish code
# drift from XLA recompilation nondeterminism, and must not abort the arm that
# answers the actual question.
run diffgemma_fa.eval.run smoke_unc_repro \
    --task bfcl_live_simple --variant unconstrained --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 0 \
    --n 4 --max-new-tokens 256 --nonempty

python3 - <<'PY'
import json, pathlib
new = pathlib.Path("artifacts/smoke_unc_repro.json")
old = pathlib.Path("artifacts/eval_bfcl_live_simple_unconstrained_sample.json")
if not new.exists():
    print("[!!] arm 0 wrote no artifact -- nothing to compare. Not a stop.")
    raise SystemExit(0)
if not old.exists():
    print("[!!] the published unconstrained artifact is not in artifacts/ --"
          " it may have been archived. Comparison skipped, not a stop.")
    raise SystemExit(0)
ref = {r["id"]: r for r in json.load(open(old))["rows"]}
rows = json.load(open(new))["rows"]
same = [r["id"] for r in rows if r["id"] in ref and r["text"] == ref[r["id"]]["text"]]
diff = [r["id"] for r in rows if r["id"] in ref and r["text"] != ref[r["id"]]["text"]]
miss = [r["id"] for r in rows if r["id"] not in ref]
print(f"[arm 0] unconstrained text reproduction: {len(same)} identical, "
      f"{len(diff)} different, {len(miss)} not in the published artifact")
for i in diff:
    print(f"    DIFFERS: {i}")
    print(f"      published: {ref[i]['text'][:160]!r}")
    print(f"      now      : {[r for r in rows if r['id']==i][0]['text'][:160]!r}")
if diff:
    print("[!!] The published unconstrained baseline is NOT reproducing "
          "verbatim at HEAD. Two candidate causes and this probe cannot tell "
          "them apart: (i) a commit since 2026-07-28 moved the unconstrained "
          "decode, (ii) XLA recompilation nondeterminism. If (i), the "
          "baseline row of docs/RESULTS.md is at old code while the "
          "constrained row about to be measured is at HEAD -- re-run it "
          "(DGFA_RUN_UNC=1, 81 min) before publishing the pair. NOT a stop: "
          "read this, then decide.")
else:
    print("[ok] arm 0: the unconstrained decode reproduces verbatim on 4 of 4 "
          "records, so the published baseline row is a HEAD number and the "
          "81-minute re-run buys nothing. 4 records detects a systematic "
          "change, not a rare one -- say so if you quote this.")
PY
echo

# ================= ARM 1: exp_e5_grammar130_map, ~103 min ==================
# THE HEADLINE. Section 2 is every flag below and where it came from; the five
# that the stored artifact does not record are resolved there against git.
# Archived FIRST: `run` [skip]s a tag whose artifact still exists.
write_archive_readme
archive_to "$ARCHIVE" exp_e5_grammar130_map

run diffgemma_fa.eval.run exp_e5_grammar130_map \
    --task bfcl_live_simple --variant j0 --emission map \
    --confidence mf --entropy-bound 0.1 \
    --whitespace pretty --fence --ci-enums --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 0 \
    --n 130 --max-new-tokens 256 --nonempty

# ---- HARD STOP 2 (denominator) ----
require_n exp_e5_grammar130_map 130 130

# ---- HARD STOP 1 (the falsifiable prediction) ----
# schema_ok == n rather than a float comparison against 1.0: integers, and the
# same statement.
sm_ok=$(field exp_e5_grammar130_map schema_ok)
sm_n=$(field exp_e5_grammar130_map n)
sm_rate=$(field exp_e5_grammar130_map schema_valid_rate)
case "$sm_ok$sm_n" in
  ''|*[!0-9]*)
    echo "!! exp_e5_grammar130_map schema_ok/n unreadable ($sm_ok/$sm_n)."
    echo "!! STOPPING."; exit 1 ;;
esac
if [ "$sm_ok" -ne "$sm_n" ]; then
  echo "!! HARD STOP 1: exp_e5_grammar130_map schema_ok=$sm_ok of n=$sm_n"
  echo "!! (schema_valid_rate=$sm_rate), predicted $sm_n of $sm_n."
  echo "!!"
  echo "!! The prediction was that this arm's ONLY schema-validity failure --"
  echo "!! live_simple_122-78-0, which emitted a bare \`1\` at CS = 1.000"
  echo "!! because the broken wildcard grammar rejected the valid object and"
  echo "!! accepted bare scalars -- is exactly what 15cdc47 repairs."
  echo "!!"
  echo "!! WHAT A FAILURE MEANS. Read the 'schema-invalid:' line printed above,"
  echo "!! which names the failing records:"
  echo "!!   * still live_simple_122-78-0  -> the wildcard fix did not deliver"
  echo "!!     what 15cdc47 was measured to deliver. Nothing downstream of that"
  echo "!!     fix should be believed, including preflight A5's green, which"
  echo "!!     tests the BUILD GATE's predicate and not the decode."
  echo "!!   * a different record -> the wildcard fix worked and something else"
  echo "!!     moved. Four commits have touched the j0/map emission path since"
  echo "!!     this arm was published on 2026-07-31: da1294a (complement"
  echo "!!     algebra -- can change a MAP argmax), b58fd84 (is_dfa polarity --"
  echo "!!     gates eq (8)'s fast path), a41bf74, 1aa8ccf. This re-run is then"
  echo "!!     not a pure wildcard-fix delta and must not be reported as one."
  echo "!!   * the failing record also appears on an 'oom:' or 'Z==0:' line"
  echo "!!     above -> it failed for capacity or for an empty support, not for"
  echo "!!     the grammar. schema_ok is False on both of those paths by"
  echo "!!     construction (the handlers score the record as a failure), so"
  echo "!!     this column cannot tell them apart on its own."
  echo "!!   * more than one -> more than one of the above."
  echo "!!"
  echo "!! Precedent for taking this seriously: task_sudoku_mask was predicted"
  echo "!! to improve on the same shape of argument ('one mechanical cause, the"
  echo "!! fix removes it') one day ago and reproduced its published value"
  echo "!! exactly. A mechanism being real is not evidence that it is the only"
  echo "!! cause."
  echo "!!"
  echo "!! The unconstrained arm behind this exists only to be compared against"
  echo "!! this row, so it is NOT run. STOPPING."
  exit 1
fi
echo "[ok] HARD STOP 1 PASSED: exp_e5_grammar130_map schema_ok=$sm_ok of"
echo "     n=$sm_n (schema_valid_rate=$sm_rate), against a published 129/130."
echo "     The wildcard record live_simple_122-78-0 now decodes to a"
echo "     well-formed object. Check cs_rate on the cfg line above: the claim"
echo "     is schema validity AND CS 1.000, not either alone."
zp_gate exp_e5_grammar130_map
echo

# ================= ARM 2 (OPT-IN): the unconstrained baseline, ~81 min =====
# NOT RUN BY DEFAULT. Section 3 is the ruling and its CPU evidence: the wildcard
# fix is provably inert on this arm's denominator and on both of its
# grammar-sensitive columns, and RESULTS.md's baseline row is already today's
# scorer applied to this artifact's own `rows`. Launch with DGFA_RUN_UNC=1 only
# if arm 0 reported a text mismatch, or if a decision has been taken to put both
# headline rows on identical model code regardless.
if [ "$RUN_UNC" = "1" ]; then
  echo "=== DGFA_RUN_UNC=1: re-running the unconstrained baseline."
  echo "=== Flags below reproduce a 2026-07-28 artifact that records only"
  echo "=== variant/emission/entropy_bound/nonempty_strings/seed. Everything"
  echo "=== else postdates it and is set to the pre-flag behaviour:"
  echo "===   --whitespace stock  = WHITESPACE_PATTERNS['stock'] is None, i.e."
  echo "===     outlines' own default, which is what compile_json_schema was"
  echo "===     called with before e12ed56. It also keeps verify_strict False,"
  echo "===     so the build gate reports instead of refusing."
  echo "===   no --fence, no --ci-enums (both store_true, added e12ed56)"
  echo "===   --prompt-style stock (PROMPT_STYLES['stock'] == '')"
  echo "===   --dtype float64, --confidence mf, --temp stock, --offset 0"
  echo "=== The grammar reaches this arm ONLY through the build gate and"
  echo "=== through cs (0/130); the emission never touches the automaton."
  archive_to "$ARCHIVE" eval_bfcl_live_simple_unconstrained_sample
  run diffgemma_fa.eval.run eval_bfcl_live_simple_unconstrained_sample \
      --task bfcl_live_simple --variant unconstrained --emission sample \
      --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
      --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 0 \
      --n 130 --max-new-tokens 256 --nonempty
  require_n eval_bfcl_live_simple_unconstrained_sample 130 130
  echo "[note] cs is expected to stay 0/130: the automaton carries OUR channel"
  echo "       header, so the stock model scores ~0 partly for a convention it"
  echo "       was never asked to follow (SPEC 7.2)."
  echo
else
  echo "=== ARM 2 (unconstrained baseline, 81 min) NOT RUN -- section 3."
  echo "=== docs/RESULTS.md's baseline row (0.631 / 0.000 / 0.639 / 0.500) is"
  echo "=== today's scorer applied to artifacts/"
  echo "===   eval_bfcl_live_simple_unconstrained_sample.json,"
  echo "=== and that artifact is grammar-invariant: both wildcard records are"
  echo "=== schema-valid and unaccepted under BOTH grammars, and both were"
  echo "=== already inside its pre-gate n=130. Set DGFA_RUN_UNC=1 to override."
  echo
fi

check_tree
echo
echo "ALL DONE $(date -u '+%Y-%m-%d %H:%M') UTC"
echo "code base $EXPECT_COMMIT (diffgemma_fa/ identical); HEAD at launch"
echo "$HEAD_AT_LAUNCH, HEAD now $(git rev-parse HEAD)"
echo "tree fingerprint still $FP0"
echo
echo 'WHAT MAY BE CLAIMED FROM THIS:'
echo '  * The headline PAIR is legitimate. The constrained row is now post-fix'
echo '    at n=130; the unconstrained row is grammar-INVARIANT (both wildcard'
echo '    records are schema-valid and unaccepted under either grammar, and'
echo '    both were already inside its pre-gate n=130), so it is the same'
echo '    number under the fixed grammar rather than a stale one.'
echo '  * McNemar for constrained vs unconstrained is recomputable on CPU from'
echo '    the two per-record `rows` lists over the same 130 records. No GPU.'
echo
echo 'WHAT MAY NOT:'
echo '  * Any pairing of this row against the OTHER constrained rows.'
echo '    j0-sample, j1-sample, j2-sample, j0-map and mask-sample are all'
echo '    pre-15cdc47 and decoded records 117 and 122 under a grammar that'
echo '    rejects the correct object. The constrained block of the table now'
echo '    straddles two grammars. Restrict those pairs to the 128 common'
echo '    records using the `rows` list each artifact already stores -- CPU'
echo '    only -- or re-run the partners (~6.4 GPU-hours). Do not average
    two grammars.'
echo '  * That this re-run isolates the wildcard fix. Four commits touched the'
echo '    j0/map emission path since 2026-07-31 (da1294a, b58fd84, a41bf74,'
echo '    1aa8ccf). It is a HEAD measurement, not a one-variable delta.'
echo
echo 'STATE OF artifacts/ THAT A RESULTS.md REGENERATION MUST HANDLE:'
echo '  * Twelve tags were archived to artifacts/stale_pre_1aa8ccf/ by the'
echo '    aborted queue and only task_sudoku_mask was re-run, so'
echo '    eval_bfcl_live_simple_mask_sample, task_countdown_mask,'
echo '    task_sudoku_j1_sample, exp_marmap_b{0.01,0.1,0.5}, exp_h2_marmap,'
echo '    exp_h2_{stock,grammar}_j1, exp_oomfix_j0 and exp_f64_ws_only_j1 have'
echo '    NO artifact in artifacts/ at all. scripts/phase5_report.py will drop'
echo '    those rows silently.'
echo '  * scripts/phase5_report.py reads the STORED accuracy fields of'
echo '    each artifact and does not apply the 2026-08-08 rescoring, so any'
echo '    surviving pre-e2be58c row is on the old scorer while the row this'
echo '    queue wrote is on the new one. Rescore from `rows` first, or the'
echo '    regenerated table silently mixes two scorers.'
