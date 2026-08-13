#!/usr/bin/env bash
# Re-run every arm invalidated by the complement-algebra defect (da1294a).
# REVIEWED VERSION -- supersedes rerun_affected.sh, which was NOT safe to launch.
#
# WHAT CHANGED AND WHY (the v1 defects, in order of how badly they hurt):
#
# 1. v1 passed NO grammar flags, so every arm would have inherited
#    `--whitespace json --no-fence --no-ci-enums`. **No artifact in the repo
#    was ever measured at `--whitespace json`.** The four table arms it
#    replaces ran at `stock` (outlines' `[ ]?`); the three `marmap` arms ran at
#    `pretty --fence --ci-enums`. Whitespace alone is worth 0.327 -> 0.643
#    arg acc (`exp_abl_ws_only`), i.e. twenty times the effect being measured.
#    v1 would have produced seven numbers that differ from the ones they
#    replace for two reasons at once, with no way to separate them, and the
#    `mar`-vs-`mf` contrast would have had no control at all (its `mf` control,
#    `exp_e5_grammar130_map`, is correctly exempt and is at pretty+fence+ci).
#    Every measurement-affecting flag is now passed EXPLICITLY, including the
#    ones that happen to equal today's default. An omitted flag is not the
#    default you assume -- and `eval/run.py`'s defaults (`j1`, `sample`)
#    already contradict SPEC 3.9 and `eval/run_tasks.py`.
#
# 2. v1 wrote to `artifacts/rerun_*.json`. `scripts/phase5_report.py` globs
#    exactly `artifacts/eval_<task>_<variant>_<emission>.json`, so it would
#    have regenerated docs/RESULTS.md from the OLD defective artifacts and said
#    nothing. The invalidated files are archived to
#    `artifacts/stale_pre_da1294a/` (repo convention: four `stale_*` dirs
#    already exist) and the re-runs take the canonical names.
#
# 3. `exp_h2_marmap` was missing. da1294a's own message says FOUR `mar` arms
#    are affected; v1 re-ran three. It is the offset-130 half.
#
# 4. No end-to-end GPU run has happened since the fix landed (last artifact
#    2026-08-08 23:56, fix 2026-08-10 04:31). A 4-record smoke costs 5 minutes
#    and turns 12 hours of hope into 12 hours of measurement.
#
# 5. The `[skip]` guard honoured a truncated JSON from a killed run. It now
#    requires the file to parse.
#
# 6. The per-arm summary did not echo the arm's own configuration, so defect 1
#    would have been invisible in the console log. It now prints the config
#    back OUT OF THE ARTIFACT -- which is the only record of what actually ran.
#
# DENOMINATOR, MEASURED (not assumed) 2026-08-10 by replaying the build gate
# over all 130 records under all three whitespace policies:
#
#     ws=json   strict=True   refused 2 -> n=128
#     ws=stock  strict=False  refused 2 -> n=128
#     ws=pretty strict=False  refused 2 -> n=128
#
# The same two records every time -- `live_simple_117-73-0` and
# `live_simple_122-78-0` -- because their defect is an unparenthesised
# top-level alternation from a BFCL `any` property, not whitespace, so the
# `verify_strict=False` hatch rebuilds them under JSON_WS and still refuses.
# So `--n 130` is correct (it pins the SAME 130-record prefix as every
# published arm) and the four table arms and the three offset-0 `marmap` arms
# will report n=128. To compare against a published n=130 row, drop those two
# ids from the published `rows` and rescore -- do NOT compare 128 against 130.
# Note `live_simple_122-78-0` was also the single `oom` record in the
# pretty+fence+ci arms, so it leaves the published denominator as a counted
# failure and leaves the new one entirely.
#
# `exp_h2_marmap` is DIFFERENT and its n=128 means something else. The gate was
# replayed over records 130..257 under the same pretty+fence+ci bundle:
# 128 records, refused 0. Neither offending id is in that half (they sit at
# indices 117 and 122). So its 128 is "all 128 ran", and its `mf` control
# `exp_h2_mfmap` is also 128 -- that pair is denominator-clean and needs no
# rescore. Only the offset-0 pair does. Flattening the two into one sentence
# is the same species of error v1 was condemned for.
#
# NOT re-run, by traced code path rather than caution -- `--emission=map` with
# `--confidence=mf`: `joint_map` builds its own dense `member` and takes a max;
# it calls neither `class_weights` nor `scatter_edge_mass_to_tokens` (grep:
# the only callers are `constrained._matrices`, `tree.sample_tokens` and
# `sampler.py`'s mask branch). `advance_states` likewise builds its own
# `member`. The only da1294a change that touches this path is `map_log_floor`,
# a measured null there (0/256 tokens, 0.00 nats, 6 seeds). Covers
# `eval_..._j0_map`, `exp_abl_*`, `exp_e5_grammar130_map`, `exp_h2_mfmap`,
# `exp_seed{1,2,3}_map`, the Sudoku/Countdown `j0map` arms, and every
# `unconstrained` arm (which returns from `body_fn` before any automaton code).
#
# Wall clock from the published `elapsed_seconds` of the same arms:
#   smoke 5m + countdown 46m + mask 96m + j1 61m + j0 92m + j2 95m
#   + 3x marmap 112m + h2_marmap 79m  ~= 13.5 h.
#
# This file must not be edited while running -- editing a live bash script
# corrupts the running instance (see docs/LOG.md). Launch from a chmod 444 copy.
set -u
cd /home/ubuntu/diffgemma_fa
source env.sh

ARCHIVE=artifacts/stale_pre_da1294a
mkdir -p "$ARCHIVE" logs

archive_invalidated () {
  # Idempotent: on a relaunch the archive copy already exists, so the original
  # is left alone and `run`'s [skip] guard handles the rest.
  # Runs AFTER the smokes -- a smoke failure must not leave nine artifacts
  # missing and `scripts/phase5_report.py` emitting "Incomplete table".
  local t
  for t in eval_bfcl_live_simple_mask_sample eval_bfcl_live_simple_j1_sample \
           eval_bfcl_live_simple_j0_sample   eval_bfcl_live_simple_j2_sample \
           exp_marmap_b0.1 exp_marmap_b0.01 exp_marmap_b0.5 exp_h2_marmap \
           task_countdown_j0sample_refixed; do
    if [ -f "artifacts/$t.json" ] && [ ! -e "$ARCHIVE/$t.json" ]; then
      mv "artifacts/$t.json" "$ARCHIVE/$t.json" && echo "[archived] $t"
      # `logs/` is gitignored and `run` opens logs/$t.log with `>`. Eight of
      # these nine logs already exist from the published runs, so without this
      # the re-run destroys the only console record of the arm it replaces --
      # and unlike the artifact there is no copy anywhere.
      if [ -f "logs/$t.log" ]; then
        mv "logs/$t.log" "$ARCHIVE/$t.log" && echo "[archived] $t.log"
      fi
    fi
  done
}

run () {                       # run <module> <tag> <flags...>
  local mod="$1" tag="$2"; shift 2
  local out="artifacts/${tag}.json"
  if [ -f "$out" ] && python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \
       "$out" 2>/dev/null; then
    echo "[skip] $tag (exists and parses)"; return
  fi
  echo "=== $(date -u +%H:%M) $tag"
  echo "    $mod $*"
  python -m "$mod" --out "$out" "$@" > "logs/${tag}.log" 2>&1
  if [ ! -f "$out" ]; then
    echo "  !! $tag DIED"; tail -20 "logs/${tag}.log" | sed 's/^/     /'; return
  fi
  python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
g = d.get
# NOTE: run_tasks.py stores none of ws/fence/ci/dtype/offset/temp/prompt/
# nonempty, so the Countdown line prints None for all of them. That is
# "the field was never written", not "the arm ran at None".
# The CONFIG, read back out of the artifact -- the only record of what ran.
print("    cfg: variant=%s emission=%s conf=%s bound=%s ws=%s fence=%s ci=%s "
      "dtype=%s seed=%s offset=%s temp=%s prompt=%s nonempty=%s" % (
          g("variant"), g("emission"), g("confidence"), g("entropy_bound"),
          g("whitespace"), g("fence"), g("ci_enums"), g("dtype"), g("seed"),
          g("offset"), g("temp"), g("prompt_style"), g("nonempty_strings")))
print("    n=%s avail=%s cs=%s schema=%s arg=%s (%s/%s) exact=%s zp=%s oom=%s "
      "skipped=%d  %.1f min" % (
          g("n"), g("records_available"), g("cs_rate"), g("schema_valid_rate"),
          g("arg_accuracy"), g("arg_correct"), g("arg_total"),
          g("exact_call_rate"), g("zero_partition"), g("oom"),
          len(g("skipped_records") or []), (g("elapsed_seconds") or 0) / 60))
for r in (g("skipped_records") or []):
    print("      excluded: %s (%s)" % (r.get("id"), r.get("reason")))
for r in (g("zero_partition_records") or []):
    print("      Z==0: %s" % r.get("id"))
for r in (g("oom_records") or []):
    print("      oom: %s (bucket %s)" % (r.get("id"), r.get("n_states_bucket")))
PY
}

E=diffgemma_fa.eval.run
T=diffgemma_fa.eval.run_tasks

# --- 0a. smoke, 5 min. Nothing has run end to end on the GPU since da1294a. ---
run $E smoke_post_da1294a --task bfcl_live_simple --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 0 --n 4 \
    --max-new-tokens 256 --nonempty

# --- 0b. smoke the OTHER repaired call site, ~3 min. `--confidence mar` routes
# acceptance and stopping through `constrained_entropy_streamed` -> the scatter
# (marginals.py:445), a DIFFERENT call site from the one 0a exercises
# (tree.sample_tokens -> marginals.py's scatter). It also builds the
# pretty+fence+ci grammar, whose larger |S| is where the 2048 bucket and the one
# historical OOM live. The four `mar` arms are ~5 h and run LAST; without this
# their first failure surfaces 8.5 h in. ---
run $E smoke_post_da1294a_mar --task bfcl_live_simple --variant j0 \
    --emission map --confidence mar --entropy-bound 0.1 \
    --whitespace pretty --fence --ci-enums --dtype float64 \
    --prompt-style stock --temp stock --seed 0 --offset 0 --n 4 \
    --max-new-tokens 256 --nonempty

# Only now displace the artifacts being replaced (D5): a smoke failure must not
# leave nine of them missing.
archive_invalidated

# --- 1. Countdown j0+sample, 46 min. THE sharp check on the fix: this is the
# arm that exposed the defect (Z==0 on 73/250, against 0/250 on MAP over the
# identical grammar and budget). class_weights was the cause and is fixed, so
# the prediction is 0/250. If this arm still shows Z==0, STOP -- the remaining
# 12 hours are measuring something else. run_tasks has no grammar flags (the
# Countdown grammar is hand-built), so there is no confound here. ---
run $T task_countdown_j0sample_refixed --task countdown --variant j0 \
    --emission sample --confidence mf --entropy-bound 0.1 --n 250 --seed 0 \
    --max-new-tokens 256

# --- 1b. THE HARD STOP. The comment above is not a safeguard; this is.
# `run_tasks.py` writes `zero_partition` as an int. Absent file / unparsable /
# non-zero all stop the queue rather than spending the remaining ~12.5 GPU-hours
# measuring something other than the fix. Idempotent: on a relaunch the arm is
# [skip]ped and this re-reads the artifact and passes. ---
zp=$(python3 -c 'import json,sys
try:
    print(json.load(open(sys.argv[1]))["zero_partition"])
except Exception:
    print("ERR")' artifacts/task_countdown_j0sample_refixed.json 2>/dev/null || echo ERR)
if [ "$zp" != "0" ]; then
  echo "!! Countdown zero_partition=$zp, predicted 0 (published: 73/250)."
  echo "!! STOPPING. da1294a is not what the remaining 12.5 h would measure."
  exit 1
fi
echo "[ok] Countdown zero_partition=0 -- da1294a confirmed end to end"

# --- 2. The four affected rows of docs/RESULTS.md's headline table.
# `--whitespace stock` is not a preference, it is what the other two rows of
# that table (`unconstrained-sample`, `j0-map`) were measured at and they are
# correctly exempt from re-running. Mixing grammars across rows of one table
# would break every paired comparison in it. `mask` first: its CS column is
# meaningless rather than noisy (255/256 positions had zero support, and
# `categorical` is shift-invariant so nothing reported it). ---
WSARM="--whitespace stock --dtype float64 --confidence mf --entropy-bound 0.1 \
--prompt-style stock --temp stock --seed 0 --offset 0 --n 130 \
--max-new-tokens 256 --nonempty"
run $E eval_bfcl_live_simple_mask_sample --task bfcl_live_simple --variant mask --emission sample $WSARM
run $E eval_bfcl_live_simple_j1_sample   --task bfcl_live_simple --variant j1   --emission sample $WSARM
run $E eval_bfcl_live_simple_j0_sample   --task bfcl_live_simple --variant j0   --emission sample $WSARM
run $E eval_bfcl_live_simple_j2_sample   --task bfcl_live_simple --variant j2   --emission sample $WSARM

# --- 3. The four `mar` arms. Emission unaffected, but acceptance AND stopping
# route through class_weights and constrained_entropy_streamed -> the scatter,
# which lost 5,013 entries on github_star and 4,033 on uber.ride at relative
# error 1.0. `--whitespace pretty --fence --ci-enums` reproduces the published
# configuration, which is also the configuration of the `mf` control these arms
# exist to be compared against (`exp_e5_grammar130_map`, exempt). ---
BUNDLE="--whitespace pretty --fence --ci-enums --dtype float64 \
--variant j0 --emission map --confidence mar --prompt-style stock --temp stock \
--seed 0 --max-new-tokens 256 --nonempty"
run $E exp_marmap_b0.1  --task bfcl_live_simple $BUNDLE --entropy-bound 0.1  --offset 0 --n 130
run $E exp_marmap_b0.01 --task bfcl_live_simple $BUNDLE --entropy-bound 0.01 --offset 0 --n 130
run $E exp_marmap_b0.5  --task bfcl_live_simple $BUNDLE --entropy-bound 0.5  --offset 0 --n 130
run $E exp_h2_marmap    --task bfcl_live_simple $BUNDLE --entropy-bound 0.1  --offset 130 --n 128

echo "ALL DONE $(date -u +%H:%M)"
echo
echo "STILL INVALID AFTER THIS QUEUE -- affected, not re-run, do not cite:"
echo "  exp_oomfix_j0, exp_oomfix_j1, exp_f64_grammar130_s1, exp_f64_ws_only_j1,"
echo "  exp_h2_grammar_j1, exp_h2_stock_j1        (BFCL sample, pretty bundle)"
echo "  task_sudoku_{j0_sample,j1_sample,j2,mask}, task_countdown_mask"
echo "  exp_e5_grammar130_{j0,s1} (float32; NOT reproducible -- eval/run.py"
echo "    cannot pass allow_unsafe_float32, so --dtype float32 now raises."
echo "    Withdraw them rather than re-running.)"
echo "  smoke_after_zdetector (j1/sample, n=8)"
echo "  exp_e{0,2,4,5,6}_* and exp_mar_ctrl_mf (n=30), phase4_e2e_sample,"
echo "    phase5_{calibrate_j1,sweep_fixed,final,header,nonempty,ws,diag_j1}"
echo "    -- the phase5 set is what calibrated entropy_bound=0.1, which every"
echo "    arm above inherits."
echo
echo "CPU follow-up before regenerating docs/RESULTS.md:"
echo "  scripts/phase5_report.py reads each artifact's STORED accuracy fields"
echo "  and does not apply the 2026-08-08 rescoring. The two exempt rows"
echo "  (unconstrained-sample, j0-map) still hold pre-audit numbers, the new"
echo "  rows hold post-audit ones. Rescore the exempt artifacts from their"
echo '  own `rows` first, or the regenerated table mixes two scorers.'
