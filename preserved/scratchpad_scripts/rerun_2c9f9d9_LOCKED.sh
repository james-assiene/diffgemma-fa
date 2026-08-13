#!/usr/bin/env bash
# =====================================================================
# Re-run queue for the arms invalidated by 2c9f9d9 (prefix_suffix) and
# the residue of da1294a (complement algebra).  REVIEWED, NOT YET LAUNCHED.
#
# Runs at commit 2c9f9d9 "Fix prefix_suffix: log-space forward-backward with
# a widening floor".  The queue refuses to start unless HEAD is that commit,
# the tree is clean, and it is being run from a read-only copy; and it
# re-verifies the tree fingerprint before EVERY arm, because the last two
# incidents on this box were both "the files moved underneath a running
# measurement" (CLAUDE.md, "Never measure a tree that an agent is editing").
#
# ---------------------------------------------------------------------
# 1. WHAT 2c9f9d9 INVALIDATED, AND WHAT IT DID NOT
# ---------------------------------------------------------------------
# infer/scans.prefix_suffix returned linear a, b normalised PER VECTOR, which
# bounds neither factor's within-vector range nor their product.  At the
# shipped sharpness it erased structurally live edges: 16 of 256 Sudoku
# positions and 7 of 256 uber.ride positions had EMPTY support.
#
# Its callers are exactly two, established by grep and by reading
# model/sampler.py rather than by assumption:
#
#   sampler.py:433  the `--confidence mar` branch  -> constrained_entropy_streamed
#   sampler.py:502  the `--variant mask` branch    -> scatter_edge_mass_to_tokens
#
# Everything else reaches the automaton through tree.sample_states_log /
# map_states_and_tokens (log-space / max-plus) and never touches it.
#
# The damage was directional and silent:
#   mask: an all -1e30 float32 row, Gumbel noise swamped, categorical returns
#         token 0 DETERMINISTICALLY.  No exception, no Z==0.
#   mar : H = -708.4 nats against a live median of 1.6e-4, i.e. below every
#         entropy_bound, so _accept_from_entropy sorts it first and ALWAYS
#         accepts it.  The accept mask itself moved -> fresh baseline, not a
#         rescore.  A rescore cannot recover a decode that never happened.
#
# NOT re-run, on a traced code path rather than caution:
#
#   (a) every j0 / j1 / j2 / unconstrained arm at --confidence mf.
#       _accept_mask reads the UNCONSTRAINED shaped logits (sampler.py:659,
#       jax.nn.softmax of out.logits) and never calls prefix_suffix.
#       joint_map calls neither class_weights nor scatter_edge_mass_to_tokens
#       nor prefix_suffix.  The one 2c9f9d9 change that does touch the mf path
#       is _accept_from_entropy's |H| < 1e-12 snap, and that is a MEASURED
#       null there: 0 of 2048 positions snap-eligible at production shape,
#       0 acceptance changes at every bound >= 0.003, and no published arm ran
#       at bound 0.0.  Every arm below is at 0.003 or above.
#   (b) SPEC 7.3's overhead table.  prefix_suffix is off the J0/J1 hot path,
#       so the tree cost it measures is unchanged; the fix's own cost is
#       +2.3/+5.6/+14.8% of the MASK closure at S=64/128/256, which is not in
#       that table, and `while == 0` / `scan == 0` still hold.
#
# ---------------------------------------------------------------------
# 2. THE THREE QUARANTINED ARTIFACTS ARE REGENERATED, NEVER RESTORED
# ---------------------------------------------------------------------
# artifacts/unattributable_dirty_tree/{exp_marmap_b0.003, smoke_b0.003,
# smoke_post_da1294a_mar}.json were written while a coder was rewriting
# infer/scans.py (mtime landed mid-run) and are attributable to no commit.
# They are NOT copied back and are NOT used as a [skip] source: the tags
# below either differ (the smokes) or the artifact simply does not exist in
# artifacts/ (exp_marmap_b0.003), so `run` cannot see them.  exp_marmap_b1.0
# never wrote an artifact at all -- it was killed mid-run.
# Their two console logs DO still sit in logs/ and are archived beside the
# quarantined artifacts before anything overwrites them.
#
# ---------------------------------------------------------------------
# 3. PER-ARM CONFIGURATION, READ OUT OF EACH EXISTING ARTIFACT
# ---------------------------------------------------------------------
# Every measurement-affecting flag is passed EXPLICITLY, including the ones
# that equal today's default.  eval/run.py defaults to --variant j1
# --emission sample (contradicting SPEC 3.9 and eval/run_tasks.py) and, worse,
# to --whitespace json -- and NO artifact in this repo has ever been measured
# at json.  An omitted flag is not the default you assume.
#
#   tag                                variant emission conf  bound  ws     fence ci  dtype seed offset n
#   eval_bfcl_live_simple_mask_sample   mask   sample   mf    0.1    stock  no    no  f64   0    0      130
#   exp_marmap_b0.003 / .01 / .1 / .5   j0     map      mar   ...    pretty yes   yes f64   0    0      130
#   exp_marmap_b1.0                     j0     map      mar   1.0    pretty yes   yes f64   0    0      130
#   exp_h2_marmap                       j0     map      mar   0.1    pretty yes   yes f64   0    130    128
#   exp_h2_stock_j1                     j1     sample   mf*   0.1    stock  no    no  f64   0    130    128
#   exp_h2_grammar_j1                   j1     sample   mf*   0.1    pretty yes   yes f64   0    130    128
#   exp_oomfix_j0                       j0     sample   mf*   0.1    pretty yes   yes f64   0    0*     130
#   exp_f64_ws_only_j1                  j1     sample   mf*   0.1    pretty no    no  f64   0    0*     130
#   task_sudoku_mask                    mask   sample   mf    0.1    (run_tasks has no grammar flags)   250
#   task_countdown_mask                 mask   sample   mf    0.1    (run_tasks has no grammar flags)   250
#   task_sudoku_j1_sample               j1     sample   mf    0.1    (run_tasks has no grammar flags)   250
#
# --entropy-bound for exp_marmap_b0.003 is taken from the QUARANTINED
# artifact's stored fields, which is legitimate: the quarantine is about the
# NUMBERS being unattributable, not about the invocation, and the invocation
# is independently fixed by the tag.  exp_marmap_b1.0 has no artifact at all;
# its bundle is the sweep's, its bound is its name.
#
# WHAT AN ABSENT FIELD MEANS (the four starred rows).  exp_f64_ws_only_j1,
# exp_oomfix_j0, exp_h2_grammar_j1 and exp_h2_stock_j1 all ran on 2026-08-02
# and store NO `confidence` and NO `temp` key, and the first two store no
# `offset` either.  Absence here is "the flag did not exist yet", not "it ran
# at None" -- verified against git rather than guessed:
#     --confidence  added 66042d9  2026-08-02 19:40:31
#     --temp        added e4e7144  2026-08-02 21:43:14
#     --offset      added 8c3be4c  2026-08-02 16:32:48
# and the four artifacts have mtimes 11:57, 14:40, 18:46 and 19:52, every one
# of them BEFORE the flag that is missing from it.  In each case the new flag
# defaulted to the pre-existing behaviour (`mf` was the only accept rule,
# `stock` the only temperature schedule, offset 0 the only cut), so
# `--confidence mf --temp stock --offset 0` reproduces them exactly.  The
# artifacts written after each landing do carry the key -- exp_mar_ctrl_mf
# (20:03) has `confidence`, exp_h2_mfmap (2026-08-03) has `temp` -- which is
# what makes "the field did not exist" the right reading and not a guess.
#
# --fence and --ci-enums are store_true with NO negated form, so False is
# expressible only by omission.  That is why every arm prints its fence/ci
# flags back OUT OF THE ARTIFACT below: the readback is the only place an
# omission would show up as wrong.
#
# TWO CONFOUNDS THAT CANNOT BE FLAGGED AWAY, and that the results table must
# state rather than the queue pretend to fix:
#   * The 2026-08-08 build gate (b58fd84) refuses live_simple_117-73-0 and
#     live_simple_122-78-0.  exp_f64_ws_only_j1 and exp_oomfix_j0 published
#     n=130; re-run they will report n=128, and 122-78-0 was also the single
#     `oom` record in the pretty+fence+ci arms, so it leaves the denominator
#     rather than being counted as a failure.  The three offset-130 arms are
#     unaffected (the gate refuses 0 of records 130..257) and stay at n=128.
#   * The 2026-08-08 scorer changes (e2be58c) postdate all four, and postdate
#     every Sudoku and Countdown artifact.  In particular the Sudoku
#     `solved` / `solve_rate` FIELDS in the existing artifacts are the OLD
#     scorer (task_sudoku_mask stores solved=51, 0.204) while RESULTS.md's
#     row is the rescore (0.012).  Compare a new Sudoku arm against
#     RESULTS.md, never against the artifact it replaces.  `cs` is computed
#     by compile.validate.Simulator and is scorer-independent, which is why
#     every gate below is written on `cs`.
#
# ---------------------------------------------------------------------
# 4. WHY --n 130 AND NOT --n 128
# ---------------------------------------------------------------------
# --n selects a PREFIX before the build gate runs.  --n 130 pins the same
# 130-record prefix every published arm used; the gate then refuses 2 and the
# artifact reports n=128.  --n 128 would select a DIFFERENT 128-record
# prefix, lose records 128 and 129, and still be gated down to n=126.  So
# --n 130 is correct and n=128 in the output is expected, not a bug.
# exp_h2_marmap is --offset 130 --n 128 and its 128 means something else
# entirely -- "all 128 of records 130..257 ran", 0 refused.  The two senses
# of 128 must not be flattened; its mf control exp_h2_mfmap is also 128 and
# that pair is denominator-clean.
#
# ---------------------------------------------------------------------
# 5. ORDER, BY HOW BADLY THE CURRENT NUMBER MISLEADS
# ---------------------------------------------------------------------
#  1 task_sudoku_mask       The only row 2c9f9d9's own message names as
#                           INVALIDATED, and the failure is mechanical, not
#                           statistical: 249 of 250 emissions are missing the
#                           whole channel-header region because positions
#                           0..15 were dead and token 0 decodes to nothing.
#                           It is also the queue's hard stop (section 6).
#  2 eval_..._mask_sample   The mask row of RESULTS.md's headline table and
#                           the left side of five significant McNemar pairs.
#                           Its CS column is SPEC 2.8's headline evidence.
#  3 task_countdown_mask    Confirm-don't-assume.  Countdown lost 5,432 live
#                           edges but 0 legal (position,token) pairs at the
#                           one measured p -- and sharpness is per record, so
#                           0 on one p is not 0 on 250.  45 min to convert an
#                           assumption into a measurement.
#  4 task_sudoku_j1_sample  The da1294a decider.  Front-loaded out of order
#                           on purpose: its outcome decides whether
#                           task_sudoku_j0_sample and task_sudoku_j2 need
#                           90 more GPU-minutes, and that is worth knowing at
#                           hour 4 rather than hour 20.  It is not a low-
#                           stakes arm either -- "constrained to the grid the
#                           model solves 1.2%" is the whole 7.1 cross-task
#                           story.
#  5 the five mar bounds    A sweep is misleading as a SET.  Pre-fix, a dead
#                           position returned H = -708 and was always
#                           accepted first, which makes acceptance nearly
#                           bound-INDEPENDENT -- so the published flatness
#                           (0.6655 / 0.6899 / 0.6725 across 1.7 decades)
#                           is exactly what a corrupted rule would produce,
#                           and is the single most likely artifact in
#                           RESULTS.md.  The endpoints 0.003 and 1.0 are
#                           where signal would appear if there is any.
#  6 exp_h2_marmap          Same defect, second-half replication.
#  7 the four da1294a arms  Stale for the older reason and already carrying a
#                           "do not cite" in docs/LOG.md, so they mislead
#                           least; each has provable Z==0 records to recover
#                           (5, 1, 1, 2 of 128/128/130/130).
#
# SPEC 3.4's grid is {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}.  After this queue
# the mar sweep covers 0.003, 0.01, 0.1, 1.0 plus an off-ladder 0.5 -- 0.03
# and 0.3 are still missing, and entropy_threshold has never been swept at
# all.  Say so in the table; do not call this "the SPEC 3.4 sweep".
#
# WALL CLOCK, from these arms' own elapsed_seconds (b1.0 has none; estimated
# from b0.5):
#   smokes 9m | sudoku_mask 45 | bfcl mask 83 | countdown_mask 45
#   | sudoku_j1 45 | marmap 0.003 123 + 0.01 102 + 0.1 102 + 0.5 102
#   + 1.0 ~102 | h2_marmap 81 | h2_stock_j1 66 | h2_grammar_j1 72
#   | oomfix_j0 76 | f64_ws_only_j1 105
#   = 1,158 min ~= 19.3 h, call it 19.5-21 h with the fix's own +2.3-14.8%
#   of the mask/mar closure and the usual compile jitter.  If the Sudoku
#   decider fails, +1.5 h for j0_sample and j2, queued separately.
#
# THE ONE ARM TO KILL IF THE GPU IS NEEDED: exp_f64_ws_only_j1, which is last
# for that reason.  It is the most expensive of the four da1294a leftovers
# (105 min) and recovers the fewest facts: 2 Z==0 records.  It published at
# n=130 pre-gate and will return n=128 on a scorer that also changed, so
# neither its accuracy nor its denominator is comparable to the row it
# replaces; its only citation in RESULTS.md is a single s/record latency
# figure (48.3 s, +29.1%) in a table CLAUDE.md already brands indicative; and
# its configuration (pretty whitespace, no fence, no ci-enums, j1, sample) has
# no partner arm at the current gate, so it cannot enter a paired comparison
# with anything.  Dropping it costs one uncitable singleton and saves 1.75 h.
#
# ---------------------------------------------------------------------
# 6. THE HARD STOPS -- CODE, NOT PROSE
# ---------------------------------------------------------------------
#  A. Preflight: HEAD == 2c9f9d9, `git status --porcelain` empty, this file
#     not writable, GPU idle.  Then a fingerprint of every .py under
#     diffgemma_fa/ re-checked before EVERY arm: if an agent edits the tree
#     mid-queue the queue stops instead of producing another unattributable
#     number.
#  B. Both smokes must report zero_partition == 0.  Post-fix, `rep` (the
#     widening floor's damage report) rides feasible -> ZeroPartitionError,
#     so this is a NEW failure mode that did not exist pre-fix.  2c9f9d9
#     predicts it cannot fire here: the viable-restricted misalignment is
#     gamma <= 474 nats across 8 real grammars at the shipped T=0.4 (404
#     Sudoku, 415 Countdown, 473.6 on BFCL 0-0-0 pretty, the worst of six
#     BFCL variants), against a 623.4-nat widening threshold.
#     On the LONG mask/mar arms this is deliberately NOT an all-or-nothing
#     stop.  Those 8 grammars are not 130 records, and one record above the
#     bound is a finding to report -- it is already counted in
#     zero_partition, its own column -- not a reason to discard the other
#     127 and the 14 hours behind them.  So: 0 is [ok], anything under 10%
#     of n prints [!!] and continues, and >= 10% of n stops the queue,
#     because that is unambiguously systematic rather than a tail record.
#  C. task_sudoku_mask must report cs >= 2, against a published 1 of 250.
#     This is the queue's falsifiable prediction and its cheapest possible
#     form.  Pre-fix, 249 of 250 emissions look like
#     '<|channel>124\n4321\n2413\n12422\n\n1' -- the entire header name region
#     absent, because positions 0..15 had empty support and categorical
#     returned token 0 for all of them; the single accepted record instead
#     reads '<|channel>thought\n<channel|>4132\n2314\n3241\n1423<turn|>'.
#     That failure has ONE mechanical cause and the fix removes it, so a
#     correct mask closure cannot reproduce 1/250.  The threshold is 2, the
#     weakest strictly-directional claim available, so a false stop needs the
#     fix to have changed nothing at all.  If it fires, the remaining ~17 h
#     are measuring something other than this fix.
#  D. task_sudoku_j1_sample must satisfy cs + zero_partition == n.  On J1 a
#     record either satisfies the automaton or died with Z==0; published
#     250 + 0 = 250.  Anything else is a guarantee violation, which is a
#     stop-everything event, and it is written this way rather than as
#     `cs == 250` precisely so that a legitimate Z==0 does not false-stop it.
#
# Every gate is idempotent: on a relaunch the arm is [skip]ped and the gate
# re-reads the artifact and passes.
#
# ---------------------------------------------------------------------
# 7. LAUNCH
# ---------------------------------------------------------------------
#   D=/tmp/claude-1000/-home-ubuntu-diffgemma-fa/29da7bd9-0534-4f90-a9d4-d456e1f98acd/scratchpad
#   cp $D/rerun_2c9f9d9.sh $D/rerun_2c9f9d9_LOCKED.sh
#   chmod 444 $D/rerun_2c9f9d9_LOCKED.sh
#   md5sum $D/rerun_2c9f9d9_LOCKED.sh
#   nohup $D/rerun_2c9f9d9_LOCKED.sh \
#         > /home/ubuntu/diffgemma_fa/logs/rerun_2c9f9d9.log 2>&1 &
#   then record the PID, the md5 and the log path in docs/LOG.md.
# The script refuses to run if it is writable: editing a live bash script
# corrupts the running instance, and that has already happened here.
# =====================================================================
set -u

# ---------- preflight (hard stop A) ----------
# RESOLVE $0 BEFORE THE cd.  Caught by testing this block rather than reading
# it: with the immutability check placed after `cd /home/ubuntu/diffgemma_fa`,
# a relative invocation (`nohup ./rerun_..._LOCKED.sh`, or any launch from a
# directory other than the repo) leaves $0 pointing at a path that no longer
# resolves, `[ -w ]` is false on a nonexistent file, and the guard silently
# passes on a WRITABLE script.  It has to be absolute, and it has to be here.
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

# `source env.sh` under `set -u` is safe: the venv activate script guards
# every optional variable, and the v3 queue (PID 2528151) ran eight arms this
# way.  It sets PREALLOCATE=true / MEM_FRACTION=.90 on this dedicated box, so
# the queue must stay SERIAL -- two of these processes cannot coexist.
source env.sh

EXPECT_COMMIT=2c9f9d97b0f15584bf3b7fc8f7ee12d5bc65480c
QUEUE=rerun_2c9f9d9

HEAD_NOW=$(git rev-parse HEAD 2>/dev/null || echo none)
if [ "$HEAD_NOW" != "$EXPECT_COMMIT" ]; then
  echo "!! HEAD is $HEAD_NOW, expected $EXPECT_COMMIT. STOPPING."
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "!! working tree is dirty. An unattributable number is worse than no"
  echo "!! number, because it looks like evidence. STOPPING."
  git status --porcelain | sed 's/^/   /'
  exit 1
fi

tree_fp () { find diffgemma_fa -name '*.py' -printf '%T@ %s %p\n' | sort | md5sum | cut -d' ' -f1; }
FP0=$(tree_fp)

check_tree () {
  local now
  now=$(tree_fp)
  if [ "$now" != "$FP0" ] || [ -n "$(git status --porcelain)" ] \
     || [ "$(git rev-parse HEAD)" != "$EXPECT_COMMIT" ]; then
    echo "!! THE TREE MOVED UNDER THE QUEUE (fingerprint $FP0 -> $now)."
    echo "!! Everything measured from here would be unattributable. STOPPING."
    exit 1
  fi
}

echo "=== $QUEUE at $(git rev-parse --short HEAD)  $(date -u '+%Y-%m-%d %H:%M') UTC"
echo "=== tree fingerprint $FP0"
echo "=== DGFA_DEDICATED=${DGFA_DEDICATED:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv || { echo "!! nvidia-smi failed"; exit 1; }
echo

mkdir -p logs artifacts/stale_pre_2c9f9d9 artifacts/stale_pre_da1294a \
         artifacts/unattributable_dirty_tree

# ---------- archiving ----------
# `logs/` is gitignored and `run` opens logs/<tag>.log with `>`, so an
# unarchived log is destroyed unrecoverably by its own re-run.  Every tag
# below has an existing log; all of them are moved.  Idempotent: on a
# relaunch the archive copy exists and the original is left alone.
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

archive_invalidated () {
  # Post-da1294a, pre-2c9f9d9: the mask baseline and the mar sweep.
  archive_to artifacts/stale_pre_2c9f9d9 \
      eval_bfcl_live_simple_mask_sample task_sudoku_mask task_countdown_mask \
      exp_marmap_b0.01 exp_marmap_b0.1 exp_marmap_b0.5 exp_h2_marmap
  # Pre-da1294a (2026-08-02/03).  A different archive dir because they are
  # stale for a different, older reason -- filing them under
  # stale_pre_2c9f9d9 would mislabel them, and that directory already exists
  # holding the arms the v3 queue replaced.
  archive_to artifacts/stale_pre_da1294a \
      task_sudoku_j1_sample exp_h2_stock_j1 exp_h2_grammar_j1 \
      exp_oomfix_j0 exp_f64_ws_only_j1
  # The two dirty-tree console logs.  Their artifacts are already quarantined
  # (or were never written); the logs are the only surviving record of what
  # those runs did, including b1.0 being killed mid-run, and the tags below
  # would overwrite them.
  local t
  for t in exp_marmap_b0.003 exp_marmap_b1.0; do
    if [ -f "logs/$t.log" ] && [ ! -e "artifacts/unattributable_dirty_tree/$t.log" ]; then
      mv "logs/$t.log" "artifacts/unattributable_dirty_tree/$t.log" \
        && echo "[archived] $t.log -> unattributable_dirty_tree"
    fi
  done
}

# ---------- the runner ----------
# The config is printed back OUT OF THE ARTIFACT, on the [skip] path too: on a
# relaunch the readback is still the only record of what actually ran, and a
# wrong flag that is invisible in the console log is exactly the defect this
# is here to catch.
show_cfg () {                  # show_cfg <artifact path>
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
g = d.get
# run_tasks.py stores none of ws/fence/ci/dtype/offset/temp/prompt/nonempty,
# so the Sudoku and Countdown lines print None for all of them.  That is "the
# field was never written", not "the arm ran at None".
print("    cfg: variant=%s emission=%s conf=%s bound=%s ws=%s fence=%s ci=%s "
      "dtype=%s seed=%s offset=%s temp=%s prompt=%s nonempty=%s think=%s "
      "commit=%s" % (
          g("variant"), g("emission"), g("confidence"), g("entropy_bound"),
          g("whitespace"), g("fence"), g("ci_enums"), g("dtype"), g("seed"),
          g("offset"), g("temp"), g("prompt_style"), g("nonempty_strings"),
          g("think"), (g("git_commit") or "?")[:7]))
print("    n=%s avail=%s cs=%s solved=%s zp=%s oom=%s schema=%s arg=%s (%s/%s) "
      "exact=%s skipped=%d  %.1f min" % (
          g("n"), g("records_available"), g("cs_rate"), g("solve_rate"),
          g("zero_partition"), g("oom"), g("schema_valid_rate"),
          g("arg_accuracy"), g("arg_correct"), g("arg_total"),
          g("exact_call_rate"), len(g("skipped_records") or []),
          (g("elapsed_seconds") or 0) / 60))
for r in (g("skipped_records") or []):
    print("      excluded: %s (%s)" % (r.get("id"), r.get("reason")))
for r in (g("zero_partition_records") or []):
    print("      Z==0: %s" % (r.get("id") if isinstance(r, dict) else r))
for r in (g("oom_records") or []):
    print("      oom: %s (bucket %s)" % (r.get("id"), r.get("n_states_bucket")))
PY
  return 0
}

run () {                       # run <module> <tag> <flags...>
  local mod="$1" tag="$2"; shift 2
  local out="artifacts/${tag}.json"
  check_tree
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
  # Stamp the commit INTO the artifact so nothing here is ever unattributable
  # again.  Extra keys only; phase5_report.py reads named fields.  Written to
  # a temp file and renamed: an interrupted in-place rewrite would truncate an
  # arm that already cost 80 GPU-minutes, and os.replace is atomic on the same
  # filesystem.
  QUEUE="$QUEUE" COMMIT="$EXPECT_COMMIT" FP="$FP0" python3 - "$out" <<'PY'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
d["git_commit"] = os.environ["COMMIT"]
d["git_tree_fingerprint"] = os.environ["FP"]
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

# hard stop B, strict form: any nonzero stops.  Used on the two n=4 smokes,
# where a single affected record is >= 25% of the arm and therefore already
# systematic.
require_zp0 () {
  local tag="$1" zp
  zp=$(field "$tag" zero_partition)
  if [ "$zp" != "0" ]; then
    echo "!! $tag zero_partition=$zp, predicted 0."
    echo "!! The viable-restricted misalignment is <= 474 nats across 8 real"
    echo "!! grammars at the shipped T=0.4, against a 623.4-nat widening"
    echo "!! threshold, so the floor must not fire here. At n=4 a nonzero is"
    echo "!! systematic: the support is degraded and every mask/mar arm behind"
    echo "!! this would be measuring damaged marginals. STOPPING."
    exit 1
  fi
  echo "[ok] $tag zero_partition=0"
}

# hard stop B, graded form: 0 is clean, a tail record is reported and kept,
# 10% or more of n stops the queue.  See section 6B for why the long arms are
# not all-or-nothing.
zp_gate () {
  local tag="$1" zp n lim
  zp=$(field "$tag" zero_partition)
  n=$(field "$tag" n)
  if [ "$zp" = ERR ] || [ "$n" = ERR ] || [ "$n" -le 0 ] 2>/dev/null; then
    echo "!! $tag: zero_partition or n unreadable (zp=$zp n=$n). STOPPING."
    exit 1
  fi
  if [ "$zp" -eq 0 ]; then
    echo "[ok] $tag zero_partition=0 -- the widening floor did not fire"
    return 0
  fi
  lim=$(( (n + 9) / 10 ))
  if [ "$zp" -ge "$lim" ]; then
    echo "!! $tag zero_partition=$zp of n=$n, at or above the 10% systematic"
    echo "!! threshold ($lim). This is not a tail record: the two-factor"
    echo "!! validity bound is being exceeded across the corpus and every"
    echo "!! mask/mar arm behind this is measuring damaged support. STOPPING."
    exit 1
  fi
  echo "[!!] $tag zero_partition=$zp of n=$n, predicted 0. Below the 10%"
  echo "     systematic threshold, so the queue continues and the artifact"
  echo "     keeps its own column. These are records whose gamma_i exceeded"
  echo "     the 623.4-nat two-factor bound: report them by id in"
  echo "     docs/RESULTS.md, do not average them away."
}

# ================= 0. the smokes, ~9 min =================
# Both repaired consumers, before the 19-hour tail.  0a is the mask branch
# (prefix_suffix -> scatter_edge_mass_to_tokens, sampler.py:502) at the stock
# grammar; 0b is the mar branch (prefix_suffix -> constrained_entropy_streamed,
# sampler.py:433) at the pretty+fence+ci grammar, whose larger |S| is where
# the 2048 bucket and the one historical OOM live.  A different call site each
# -- the previous queue's single smoke missed the second, whose arms run last.
run diffgemma_fa.eval.run smoke_2c9f9d9_mask \
    --task bfcl_live_simple --variant mask --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 0 --n 4 \
    --max-new-tokens 256 --nonempty
require_zp0 smoke_2c9f9d9_mask

run diffgemma_fa.eval.run smoke_2c9f9d9_mar \
    --task bfcl_live_simple --variant j0 --emission map \
    --confidence mar --entropy-bound 0.1 \
    --whitespace pretty --fence --ci-enums --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 0 --n 4 \
    --max-new-tokens 256 --nonempty
require_zp0 smoke_2c9f9d9_mar

# Only now displace what is being replaced: a smoke failure must not leave
# twelve artifacts missing and phase5_report.py emitting an incomplete table.
archive_invalidated
echo

# ================= 1. task_sudoku_mask, 45 min =================
# The row 2c9f9d9 names as INVALIDATED, and the queue's hard stop.
run diffgemma_fa.eval.run_tasks task_sudoku_mask \
    --task sudoku --variant mask --emission sample \
    --confidence mf --entropy-bound 0.1 --n 250 --seed 0 --max-new-tokens 256

# ---- HARD STOP C ----
sk_cs=$(field task_sudoku_mask cs)
case "$sk_cs" in
  ''|*[!0-9]*)
    echo "!! task_sudoku_mask cs unreadable ($sk_cs). STOPPING."; exit 1 ;;
esac
if [ "$sk_cs" -lt 2 ]; then
  echo "!! task_sudoku_mask cs=$sk_cs of 250, predicted >= 2 (published 1)."
  echo "!! The 249 pre-fix failures had ONE mechanical cause -- positions"
  echo "!! 0..15 empty, categorical returning token 0 -- and the fix removes"
  echo "!! it. Reproducing 1/250 means the mask closure is still degenerate"
  echo "!! and the remaining ~17 GPU-hours would measure something else."
  exit 1
fi
echo "[ok] task_sudoku_mask cs=$sk_cs of 250 (published 1) -- the dead"
echo "     front-of-canvas positions are gone."
echo "     NOTE: compare solve_rate against docs/RESULTS.md's rescored 0.012,"
echo "     NOT against this artifact's predecessor (0.204, pre-e2be58c scorer)."
echo

# ================= 2. eval_bfcl_live_simple_mask_sample, 83 min =================
# --whitespace stock is not a preference: it is what the other rows of that
# table were measured at, and they are correctly exempt.  Mixing grammars
# across rows of one table breaks every paired comparison in it.
run diffgemma_fa.eval.run eval_bfcl_live_simple_mask_sample \
    --task bfcl_live_simple --variant mask --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 0 --n 130 \
    --max-new-tokens 256 --nonempty
zp_gate eval_bfcl_live_simple_mask_sample
echo

# ================= 3. task_countdown_mask, 45 min =================
run diffgemma_fa.eval.run_tasks task_countdown_mask \
    --task countdown --variant mask --emission sample \
    --confidence mf --entropy-bound 0.1 --n 250 --seed 0 --max-new-tokens 256
zp_gate task_countdown_mask
echo

# ================= 4. task_sudoku_j1_sample, 45 min -- THE DECIDER ==========
# da1294a, not 2c9f9d9: j1+sample goes joint_draw -> up_sweep_log, which reads
# class_weights.  Published cs 250/250, solved 3/250, zero_partition 0.
run diffgemma_fa.eval.run_tasks task_sudoku_j1_sample \
    --task sudoku --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 --n 250 --seed 0 --max-new-tokens 256

# ---- HARD STOP D ----
sj_cs=$(field task_sudoku_j1_sample cs)
sj_zp=$(field task_sudoku_j1_sample zero_partition)
sj_n=$(field task_sudoku_j1_sample n)
if [ "$sj_cs" = ERR ] || [ "$sj_zp" = ERR ] || [ "$sj_n" = ERR ]; then
  echo "!! task_sudoku_j1_sample fields unreadable. STOPPING."; exit 1
fi
if [ $((sj_cs + sj_zp)) -ne "$sj_n" ]; then
  echo "!! task_sudoku_j1_sample cs=$sj_cs zp=$sj_zp n=$sj_n."
  echo "!! On J1 a record either satisfies the automaton or died with Z==0."
  echo "!! cs + zero_partition != n is a GUARANTEE VIOLATION. STOPPING."
  exit 1
fi
echo "[ok] task_sudoku_j1_sample cs=$sj_cs zp=$sj_zp n=$sj_n"
echo "     DECIDER: published cs 250/250, solved 3/250 (0.012, and that one"
echo "     agrees with RESULTS.md's rescore). If solved is within seed noise"
echo "     of 3/250, task_sudoku_j0_sample and task_sudoku_j2 are CLEARED and"
echo "     need no GPU time. If it is not, queue all three -- roughly 1.5 h."
echo "     That call is a human one; this queue does not make it."
echo

# ================= 5. the mar sweep, ~8.5 h =================
# Emission unaffected (joint_map), but acceptance AND stopping route through
# prefix_suffix -> constrained_entropy_streamed, where a dead position
# returned H = -708.4 and was sorted first and always accepted.  Fresh
# baseline, not a rescore: the accept mask itself moved.
# --whitespace pretty --fence --ci-enums reproduces the published bundle,
# which is also the bundle of the mf control these arms exist to be compared
# against (exp_e5_grammar130_map, exempt).
MAR="--task bfcl_live_simple --variant j0 --emission map --confidence mar \
--whitespace pretty --fence --ci-enums --dtype float64 \
--prompt-style stock --temp stock --seed 0 --max-new-tokens 256 --nonempty"

# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b0.003 $MAR --entropy-bound 0.003 --offset 0   --n 130
zp_gate exp_marmap_b0.003
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b0.01  $MAR --entropy-bound 0.01  --offset 0   --n 130
zp_gate exp_marmap_b0.01
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b0.1   $MAR --entropy-bound 0.1   --offset 0   --n 130
zp_gate exp_marmap_b0.1
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b0.5   $MAR --entropy-bound 0.5   --offset 0   --n 130
zp_gate exp_marmap_b0.5
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b1.0   $MAR --entropy-bound 1.0   --offset 0   --n 130
zp_gate exp_marmap_b1.0
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_h2_marmap     $MAR --entropy-bound 0.1   --offset 130 --n 128
zp_gate exp_h2_marmap
echo

# ================= 6. the da1294a residue, ~5.3 h =================
# NOT affected by 2c9f9d9 -- these are j0/j1 sample arms whose emission is
# joint_draw -> up_sweep_log, and whose accept rule is mf.  They are here
# because da1294a's complement-algebra fix is not in them: each has provable
# pre-fix zero_partition records to recover.  Predicted 0 for all four.
# No zp gate: these run last, so stopping here buys nothing, and a
# residual Z==0 is a finding to write up rather than a reason to abandon
# artifacts already on disk.  The check prints instead.
zp_report () {                 # zp_report <tag> <published zp>
  local tag="$1" pub="$2" zp
  zp=$(field "$tag" zero_partition)
  if [ "$zp" = "0" ]; then
    echo "[ok] $tag zero_partition=0 (published $pub) -- da1294a confirmed"
  else
    echo "[!!] $tag zero_partition=$zp, published $pub, predicted 0."
    echo "     da1294a did not close it. Write this up in docs/LOG.md."
  fi
}

run diffgemma_fa.eval.run exp_h2_stock_j1 \
    --task bfcl_live_simple --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 130 --n 128 \
    --max-new-tokens 256 --nonempty
zp_report exp_h2_stock_j1 5

run diffgemma_fa.eval.run exp_h2_grammar_j1 \
    --task bfcl_live_simple --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 \
    --whitespace pretty --fence --ci-enums --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 130 --n 128 \
    --max-new-tokens 256 --nonempty
zp_report exp_h2_grammar_j1 1

run diffgemma_fa.eval.run exp_oomfix_j0 \
    --task bfcl_live_simple --variant j0 --emission sample \
    --confidence mf --entropy-bound 0.1 \
    --whitespace pretty --fence --ci-enums --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 0 --n 130 \
    --max-new-tokens 256 --nonempty
zp_report exp_oomfix_j0 1

# LAST ON PURPOSE. See "THE ONE ARM TO KILL" above: 105 min to recover two
# Z==0 records on an arm that cannot be compared to the row it replaces and
# has no paired partner. Kill this one first if the GPU is wanted back.
run diffgemma_fa.eval.run exp_f64_ws_only_j1 \
    --task bfcl_live_simple --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace pretty --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 0 --n 130 \
    --max-new-tokens 256 --nonempty
zp_report exp_f64_ws_only_j1 2

check_tree
echo
echo "ALL DONE $(date -u '+%Y-%m-%d %H:%M') UTC at $EXPECT_COMMIT"
echo "tree fingerprint still $FP0"
echo
echo 'STILL INVALID AFTER THIS QUEUE -- do not cite:'
echo '  task_sudoku_j0_sample, task_sudoku_j2   (da1294a; gated on the'
echo '    task_sudoku_j1_sample decider above -- read its [ok] line)'
echo '  exp_oomfix_j1, exp_f64_grammar130_s1    (da1294a, j1/sample; not in'
echo '    this queue)'
echo '  exp_e5_grammar130_j0, exp_e5_grammar130_s1  (float32; NOT reproducible'
echo '    -- --dtype float32 now raises for sample, mask and mar. Withdraw'
echo '    them rather than re-running.)'
echo '  smoke_after_zdetector, smoke_post_da1294a  (n=4/8 smokes, never cited)'
echo '  exp_e{0,2,4,5,6}_*, exp_mar_ctrl_mf (n=30), phase4_e2e_sample,'
echo '    phase5_{calibrate_j1,sweep_fixed,final,header,nonempty,ws,diag_j1}'
echo '    -- the phase5 set is what calibrated entropy_bound=0.1, which every'
echo '    arm above inherits.'
echo
echo 'CPU FOLLOW-UP, before regenerating docs/RESULTS.md:'
echo '  * scripts/phase5_report.py reads each artifact STORED accuracy fields'
echo '    and does not apply the 2026-08-08 rescoring, so the exempt rows'
echo '    still hold pre-audit numbers while every row above holds post-audit'
echo '    ones. Rescore the exempt artifacts from their own rows first, or the'
echo '    regenerated table silently mixes two scorers.'
echo '  * The Sudoku and Countdown solve_rate FIELDS in every pre-2026-08-08'
echo '    artifact are the old scorer. RESULTS.md carries the rescore. Compare'
echo '    new arms against RESULTS.md, not against the artifact they replace.'
echo '  * The mar sweep now covers 0.003 / 0.01 / 0.1 / 0.5 / 1.0. SPEC 3.4'
echo '    asks for 0.003 / 0.01 / 0.03 / 0.1 / 0.3 / 1.0 crossed with three'
echo '    entropy_threshold values on a 100-example dev slice. 0.03, 0.3 and'
echo '    the whole threshold axis are still unmeasured, and 0.5 is not on'
echo '    the ladder. Say so in the table rather than calling it the sweep.'
echo '  * n=128 means two different things: 130 minus the two gate-refused'
echo '    records for every offset-0 arm, and all 128 ran for the offset-130'
echo '    ones. Do not pool them.'
