#!/usr/bin/env bash
# SPEC 3.4, finished: the two entropy-bound extremes, post-da1294a, on the
# production configuration.
#
# THIS IS NOT A RE-RUN OF THE PHASE5 ARTIFACTS. Read this block before launching.
#
# The task was "re-run artifacts/phase5_*.json (8 arms), the set that calibrated
# entropy_bound=0.1". All eight are EXCLUDED, and 0.1 is CONFIRMED, on the
# following record. The queue below is a different, smaller, and strictly more
# informative measurement, and it is labelled as such so nobody later mistakes
# these artifacts for reproductions of the phase5 ones.
#
# WHY NONE OF THE EIGHT IS RE-RUNNABLE
#
# (a) They were not produced by diffgemma_fa/eval/run.py. Every one comes from
#     scripts/phase5_calibrate.py, whose CLI is only
#         --n --bounds --emission --variant --nonempty --diagnose --out
#     It has no --whitespace, no --fence, no --ci-enums, no --dtype, no
#     --confidence, no --seed, no --offset, no --prompt-style, no --temp.
#
# (b) Its whitespace policy therefore cannot be pinned, and it CHANGED under it.
#     phase5_calibrate.py calls compile_json_schema WITHOUT whitespace_pattern.
#     At b152d07 (the commit that shipped the last of these artifacts) that
#     parameter defaulted to None -- outlines' own [ ]?, i.e. today's
#     --whitespace stock. At HEAD it defaults to schema.JSON_WS ([ \t\n\r]*),
#     i.e. --whitespace json. NO ARTIFACT IN THIS REPO HAS EVER BEEN MEASURED AT
#     json. So re-running the script today silently changes the one axis worth
#     0.327 -> 0.622 arg acc (exp_abl_ws_only), roughly twenty times the entropy
#     bound effect being measured, with no flag available to stop it. This is
#     defect 1 of the previous two launch reviews, in a form no flag list can fix.
#
# (c) Three of the eight are rungs of b4554aa's root-cause ladder and their
#     "before" states are LIBRARY states, not flags: phase5_header.json needs
#     channel_header off, phase5_ws.json and phase5_final.json need
#     whitespace_pattern="". Both are now unconditional in compile/pipeline.py.
#     Reproducing them means reverting library code, which is not a launch.
#
# (d) They also predate fe01dbb + 288cb59 (the two log_matmul corrections, Jul
#     29 -- on the sum-product path every one of these j1/sample arms uses),
#     fd830d9 ("a corpus-affecting schema bug"), b58fd84, 9115229 and ae144f3.
#     da1294a is not the only thing that changed; it is the sixth.
#
# PER-ARM EXCLUSION, with the reason for each
#
#   phase5_calibrate_j1.json  07:59  j1/sample, 6 bounds, n=12, nonempty absent
#   phase5_nonempty.json      08:06  j1/sample, bound 0.1, n=12, nonempty=True
#       RETRACTED, not merely stale. b152d07 and docs/PHASE5_FINDINGS.md 1 say
#       in terms: measured on the broken grammar, "the earlier numbers should
#       not be quoted". Re-running a retracted table produces nothing citable.
#
#   phase5_header.json        08:46  j1/sample, bound 0.1, n=12
#   phase5_ws.json            09:01  j1/sample, bound 0.1, n=12
#   phase5_final.json         09:24  j1/sample, bound 0.1, n=12
#       Structurally unreproducible -- see (c). Superseded anyway at n=130 by
#       exp_abl_ws_only / exp_abl_ws_fence / exp_abl_fence_only, which measure
#       the same ablation on the production harness.
#
#   phase5_diag.json          07:19  j0/map, bound 0.1, n=2, --diagnose
#       NOT AFFECTED by da1294a, by the same traced exemption the v3 queue used
#       and a reviewer already checked: joint_map builds its own dense `member`
#       and takes a max; it calls neither class_weights nor
#       scatter_edge_mass_to_tokens. --confidence did not exist on 2026-07-28,
#       so this arm is mf, and map_log_floor is a measured null there (0/256
#       tokens, 0.00 nats, 6 seeds).
#
#   phase5_diag_j1.json       07:21  j1/sample, bound 0.1, n=2, --diagnose
#       Affected in principle, worthless in practice: it is a per-step trace of
#       ONE record on the pre-b4554aa grammar -- no channel header, whitespace
#       forbidden -- the grammar b4554aa proved was itself the cause of the 5%
#       accuracy. Its only claim (J0 starves the emission, J1 does not) has
#       already been re-measured post-fix at n=128:
#       eval_bfcl_live_simple_j0_sample 0.3240 vs _j1_sample 0.3868.
#
#   phase5_sweep_fixed.json   14:07  j1/sample, 6 bounds, n=12, nonempty=True
#       The only one carrying a live claim, and the one blocked by (b). It is
#       also the one whose conclusion has already been retested -- see below.
#
# WHY entropy_bound = 0.1 STANDS, and why the premise of the task is wrong
#
# 1. Nothing in the phase5 set ever selected 0.1. b152d07: "nothing here selects
#    a value". PHASE5_FINDINGS 1: "the stock 0.1 sits mid-range and no value is
#    distinguishable from it". 0.1 is gemma's own SampleFromPredictions default,
#    inherited; the phase5 sweep is the negative result that failed to displace
#    it. There is no calibration here to invalidate.
#
# 2. It has ALREADY been recalibrated post-da1294a, at ten times the records, on
#    the production configuration. artifacts/exp_marmap_b{0.01,0.1,0.5}.json
#    were written 2026-08-10 22:00, 23:43 and 2026-08-11 01:27 -- all after the
#    fix (2026-08-10 04:31) -- by the reviewed v3 queue, at
#    j0/map/mar/pretty+fence+ci/float64/seed 0/offset 0, n=128, arg_total=287:
#        0.01  191/287 = 0.6655
#        0.1   198/287 = 0.6899   <- stock, and nominally best
#        0.5   193/287 = 0.6725
#    A 7-of-287 spread across a 50x range of the bound. 0.1 survives.
#
# 3. Traced, for the mf arms: da1294a CANNOT change which positions a given
#    bound accepts. sampler.py line 411 computes the accept mask as
#    _accept_mask(out.logits, entropy_bound) -- the entropy of the UNCONSTRAINED
#    shaped logits, with no automaton term anywhere in it. Every phase5 arm is
#    mf (--confidence did not exist until 66042d9, Aug 2). The fix can only
#    change which TOKEN is drawn at an already-accepted position. Under mar
#    (lines 378-408) acceptance does route through
#    constrained_entropy_streamed -> the scatter -- and the mar sweep is exactly
#    the one already re-run in point 2.
#
# WHAT IS ACTUALLY STILL OPEN, AND WHAT THIS QUEUE MEASURES
#
# SPEC 3.4 prescribes six bounds {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}. The post-fix
# sweep covers three, {0.01, 0.1, 0.5}, and NEITHER EXTREME. If the bound has any
# effect it shows at the ends. Two arms at 0.003 and 1.0, on the byte-identical
# configuration of exp_marmap_b0.1, complete SPEC's grid at its two most
# informative points, post-fix, at n=128 -- and pool directly with the three
# existing arms because every other flag matches. That is the whole queue.
#
# NOT INCLUDED, deliberately: a j1/sample bound sweep at n=128. By point 3 the
# mf accept rule is automaton-free, so that is not a da1294a question at all; it
# is a separate open question (the sample path's bound has never been swept
# above n=12 on any grammar) and costs a further ~2.7 h. Named here so it is a
# decision, not an omission.
#
# DENOMINATOR. --n 130 --offset 0 pins the same 130-record prefix as every
# published arm; the build gate refuses live_simple_117-73-0 and
# live_simple_122-78-0 under all three whitespace policies, so both arms MUST
# report n=128 and arg_total=287. The gate below enforces exactly that: an arm
# that reports anything else is not poolable with exp_marmap_b{0.01,0.1,0.5} and
# the queue stops rather than producing a fourth number that looks comparable.
#
# ONE FLAG THE READBACK CANNOT CONFIRM. eval/run.py's output dict does not
# contain max_new_tokens (grep: 0 hits in exp_marmap_b0.1.json), and it is
# measurement-affecting -- it is the R in SPEC 3.1b's budget-aware terminal
# factor b_L = 1[d(s) <= R]. It is passed explicitly as --max-new-tokens 256,
# matching every published arm, and the argv line run() echoes is the only
# record of it. Adding it to the artifact is a code change for another day; do
# not let its absence from the cfg line read as "not passed".
#
# ARCHIVING. Nothing is replaced. artifacts/exp_marmap_b0.003.json,
# artifacts/exp_marmap_b1.0.json, logs/exp_marmap_b0.003.log and
# logs/exp_marmap_b1.0.log all do not exist (checked 2026-08-11). The only
# destructive path left is overwriting an artifact that exists but does NOT
# parse -- run() archives that one, with its log, before touching it.
#
# WALL CLOCK, from the elapsed_seconds of the same arms:
#   exp_marmap_b0.01 6137.4 s, b0.1 6120.8 s, b0.5 6148.0 s -> 102.3 min each.
#   elapsed_seconds excludes model load (t_start is set after it); each run() is
#   a separate interpreter, so add ~4 min per arm.
#   smoke 189.5 s + load ~4 min = ~7 min. Total ~3.7 h.
#
# LAUNCH. Do not edit this file while it runs -- editing a live bash script
# corrupts the running instance (docs/LOG.md, 5d30083). Copy it, chmod 444 the
# copy, and launch the copy:
#   cp phase5_recal.sh phase5_recal_LOCKED.sh && chmod 444 phase5_recal_LOCKED.sh
#   nohup bash phase5_recal_LOCKED.sh > logs/phase5_recal.log 2>&1 &
# Record the PID and the log path in docs/LOG.md.
set -u
cd /home/ubuntu/diffgemma_fa || exit 1
source env.sh

ARCHIVE=artifacts/stale_pre_da1294a
mkdir -p "$ARCHIVE" logs

# CLAUDE.md: "Check before you allocate. nvidia-smi before any long run." This
# box sets XLA_PYTHON_CLIENT_PREALLOCATE=true at MEM_FRACTION=.90, so a second
# process holding the device does not degrade this run -- it kills it at BLASLT
# init, after the model load, once per arm. Cheaper to refuse up front.
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits \
       2>/dev/null || echo ERR)
case "$used" in                      # empty or non-numeric, not just "ERR":
  ''|*[!0-9]*)                       # nvidia-smi can exit 0 with no output.
    echo "!! nvidia-smi gave '${used}'; refusing to launch blind."; exit 1;;
esac
if [ "$used" -gt 2048 ]; then
  echo "!! GPU 0 already has ${used} MiB in use. PREALLOCATE=true at .90 will"
  echo "!! collide with it. Stop and log this rather than racing for memory."
  exit 1
fi
echo "[preflight] GPU 0 free (${used} MiB used)"

run () {                       # run <tag> <flags...>
  local tag="$1"; shift
  local out="artifacts/${tag}.json"
  if [ -f "$out" ] && python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \
       "$out" 2>/dev/null; then
    echo "[skip] $tag (exists and parses)"; return
  fi
  # The only destructive case: a file that exists but does not parse, i.e. a
  # killed run. logs/ is gitignored and the redirect below is `>`, so without
  # this the truncated artifact AND the only console record of it are both gone.
  if [ -e "$out" ] || [ -e "logs/${tag}.log" ]; then
    local stamp; stamp=$(date -u +%Y%m%dT%H%M%SZ)
    [ -e "$out" ] && mv "$out" "$ARCHIVE/${tag}.unparsable.${stamp}.json" \
      && echo "[archived] ${tag}.json (did not parse)"
    [ -e "logs/${tag}.log" ] && mv "logs/${tag}.log" \
      "$ARCHIVE/${tag}.${stamp}.log" && echo "[archived] ${tag}.log"
  fi
  echo "=== $(date -u +%H:%M) $tag"
  echo "    diffgemma_fa.eval.run $*"
  python -m diffgemma_fa.eval.run --out "$out" "$@" > "logs/${tag}.log" 2>&1
  if [ ! -f "$out" ]; then
    echo "  !! $tag DIED"; tail -20 "logs/${tag}.log" | sed 's/^/     /'; return
  fi
  python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); g = d.get
# The CONFIG, read back OUT OF THE ARTIFACT -- the only record of what ran.
print("    cfg: variant=%s emission=%s conf=%s bound=%s ws=%s fence=%s ci=%s "
      "dtype=%s seed=%s offset=%s temp=%s prompt=%s nonempty=%s task=%s" % (
          g("variant"), g("emission"), g("confidence"), g("entropy_bound"),
          g("whitespace"), g("fence"), g("ci_enums"), g("dtype"), g("seed"),
          g("offset"), g("temp"), g("prompt_style"), g("nonempty_strings"),
          g("task")))
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

# THE HARD STOP. Code, not a comment. An arm that does not match the record set
# and the guarantee of the three arms it is pooled with is not comparable to
# them, and the next arm would spend 1.7 h producing a second such number.
# Idempotent: on a relaunch the arm is [skip]ped and this re-reads the artifact.
# A gate failure therefore persists across relaunches ON PURPOSE -- it needs a
# human, not a retry.
#
# It also RE-ASSERTS the whole configuration out of the artifact, not just
# prints it. Printing is not enough on its own: run()'s [skip] guard accepts any
# artifact that parses, so an file left at the wrong bound by someone else would
# be skipped, gated on its numbers alone, and pooled. The bundle below is
# exp_marmap_b0.1.json's config field verbatim.
gate () {                # gate <artifact> <want_n> <want_arg_total|-> <want_bound>
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
path, want_n, want_at, want_b = sys.argv[1:5]
try:
    d = json.load(open(path))
except Exception as e:                                        # noqa: BLE001
    print("!! GATE: %s missing or unreadable (%s)" % (path, e)); sys.exit(1)
bad = []
BUNDLE = {"task": "bfcl_live_simple", "variant": "j0", "emission": "map",
          "confidence": "mar", "whitespace": "pretty", "fence": True,
          "ci_enums": True, "dtype": "float64", "seed": 0, "offset": 0,
          "prompt_style": "stock", "temp": "stock", "nonempty_strings": True}
for k, v in sorted(BUNDLE.items()):
    if d.get(k) != v:
        bad.append("%s=%r want %r" % (k, d.get(k), v))
if float(d.get("entropy_bound", -1)) != float(want_b):
    bad.append("entropy_bound=%s want %s" % (d.get("entropy_bound"), want_b))
if str(d.get("n")) != want_n:
    bad.append("n=%s want %s" % (d.get("n"), want_n))
if want_at != "-" and str(d.get("arg_total")) != want_at:
    bad.append("arg_total=%s want %s" % (d.get("arg_total"), want_at))
if d.get("cs_rate") != 1.0:
    bad.append("cs_rate=%s want 1.0" % d.get("cs_rate"))
if d.get("zero_partition") != 0:
    bad.append("zero_partition=%s want 0" % d.get("zero_partition"))
if d.get("oom") != 0:
    bad.append("oom=%s want 0" % d.get("oom"))
if bad:
    print("!! GATE FAILED on %s: %s" % (path, "; ".join(bad)))
    print("!! STOPPING. This arm is not poolable with exp_marmap_b"
          "{0.01,0.1,0.5}; a human decides before more GPU time is spent.")
    sys.exit(1)
print("[gate ok] %s" % path)
PY
}

# The exact configuration of exp_marmap_b0.1.json, every measurement-affecting
# flag passed EXPLICITLY including the ones equal to today's default --
# eval/run.py defaults to --variant j1 --emission sample --whitespace json, all
# three of which are wrong here and none of which is asserted anywhere.
BUNDLE="--task bfcl_live_simple --variant j0 --emission map --confidence mar \
--whitespace pretty --fence --ci-enums --dtype float64 --prompt-style stock \
--temp stock --seed 0 --offset 0 --max-new-tokens 256 --nonempty"

# --- 0. The cheap early check, ~7 min. Runs the SMALLER of the two bounds,
# because 0.003 accepts the fewest positions per denoising step and so is the
# one that can leave positions unaccepted at the 48-step budget. It exercises
# the mar path (class_weights and constrained_entropy_streamed -> the scatter,
# both da1294a call sites) on the pretty+fence+ci grammar, which is where the
# larger |S| buckets live. Without it the first failure surfaces 1.7 h in. ---
run smoke_b0.003 $BUNDLE --entropy-bound 0.003 --n 4
gate artifacts/smoke_b0.003.json 4 - 0.003 || exit 1

# --- 1. bound 0.003, SPEC 3.4's lower extreme, ~1.8 h. ---
run exp_marmap_b0.003 $BUNDLE --entropy-bound 0.003 --n 130
gate artifacts/exp_marmap_b0.003.json 128 287 0.003 || exit 1

# --- 2. bound 1.0, SPEC 3.4's upper extreme, ~1.8 h. ---
run exp_marmap_b1.0 $BUNDLE --entropy-bound 1.0 --n 130
gate artifacts/exp_marmap_b1.0.json 128 287 1.0 || exit 1

# --- 3. The pooled table, CPU only. Reads all five bounds back out of the
# artifacts. Every row must show the same cfg except bound; a row that does not
# is the defect this queue exists to avoid, and it will be visible here. ---
echo
echo "===== SPEC 3.4, post-da1294a, j0/map/mar/pretty+fence+ci, n=128 ====="
python3 - <<'PY'
import json, pathlib
rows = []
for b in ("0.003", "0.01", "0.1", "0.5", "1.0"):
    p = pathlib.Path("artifacts/exp_marmap_b%s.json" % b)
    if not p.exists():
        rows.append((b, "MISSING", "", "", "", "", "")); continue
    d = json.load(open(p))
    rows.append((b, d["n"], "%s/%s" % (d["arg_correct"], d["arg_total"]),
                 "%.4f" % d["arg_accuracy"], "%.4f" % d["exact_call_rate"],
                 "%.4f" % d["cs_rate"],
                 "%s/%s/%s/%s/%s/%s" % (d.get("variant"), d.get("emission"),
                                        d.get("confidence"), d.get("whitespace"),
                                        d.get("fence"), d.get("ci_enums"))))
print("%-8s %-5s %-9s %-8s %-8s %-7s %s" % (
    "bound", "n", "arg", "arg_acc", "exact", "cs", "cfg"))
for r in rows:
    print("%-8s %-5s %-9s %-8s %-8s %-7s %s" % r)
PY
echo
echo "ALL DONE $(date -u +%H:%M)"
echo
echo "READ BEFORE CITING THESE TWO ARTIFACTS:"
echo "  They are NOT reproductions of artifacts/phase5_*.json. All eight phase5"
echo "  arms are excluded -- see the header of this script for the per-arm"
echo "  reason. Two are retracted by b152d07, three need library states that no"
echo "  longer exist, one is da1294a-exempt (j0/map/mf), one is a two-record"
echo "  trace on a grammar b4554aa deleted, and the last cannot pin its own"
echo "  whitespace policy because scripts/phase5_calibrate.py has no flag for"
echo "  it and the library default moved from outlines to RFC 8259 underneath."
echo
echo "  entropy_bound = 0.1 was NOT calibrated by the phase5 set -- b152d07 says"
echo "  'nothing here selects a value'. It is gemma's own default, and it was"
echo "  re-confirmed post-da1294a at n=128 by exp_marmap_b{0.01,0.1,0.5}."
echo "  These two arms complete SPEC 3.4's six-point grid at its extremes."
echo
echo "STILL OPEN, not measured by this queue:"
echo "  - the bound on the j1/sample path above n=12 (~2.7 h; by sampler.py:411"
echo "    the mf accept rule is automaton-free, so this is not a da1294a"
echo "    question -- it is a separate one)"
echo "  - entropy_threshold, the other half of SPEC 3.4's 2-D grid. Structurally"
echo "    unswept: early_stop_fn is passed at zero production call sites, so"
echo "    every arm inherits NoEarlyStop. Sweeping it is a code change, not a"
echo "    launch (docs/RESULTS.md, the entropy_threshold bullet)."
