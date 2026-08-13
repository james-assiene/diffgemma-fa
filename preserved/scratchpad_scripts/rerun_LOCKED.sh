#!/usr/bin/env bash
# Re-run every arm invalidated by the complement-algebra defect (da1294a).
#
# NOT re-run, on measured evidence rather than caution: --emission=map with
# --confidence=mf. joint_map calls neither class_weights nor the scatter (it
# builds its own dense `member` and takes a max), and J0's trajectory is stock
# uniform renoising. The map_log_floor change is a measured null on those arms
# (0/256 tokens, 0.00 nats across 6 seeds).
#
# Order is by how badly the old number misleads:
#   1 mask     -- 255/256 positions had ZERO support, so `categorical` drew
#                 uniformly over 262k tokens and reported nothing. SPEC 2.8's
#                 CS column is the headline evidence for the whole method and
#                 is currently meaningless rather than noisy.
#   2 sample   -- the arms that raised Z==0 (Countdown 73/250) or silently lost
#                 thousands of r entries (github_star 5,013; uber.ride 4,033).
#   3 mar      -- emission unaffected, but acceptance and stopping route
#                 through class_weights AND the scatter. The reviewer withdrew
#                 its own blanket MAP exemption on tracing this.
#
# Expect n=128, not 130: the build gate (b58fd84) refuses live_simple_117-73-0
# and live_simple_122-78-0, and 012660c now names them in the artifact. Do not
# compare against the published n=130 rows without accounting for that.
#
# This file must not be edited while running -- editing a live bash script
# corrupts the running instance (see docs/LOG.md).
set -u
cd /home/ubuntu/diffgemma_fa
source env.sh

run () {
  local tag="$1"; shift
  local out="artifacts/${tag}.json"
  if [ -f "$out" ]; then echo "[skip] $tag (exists)"; return; fi
  echo "=== $(date -u +%H:%M) $tag ==="
  python -m diffgemma_fa.eval.run --out "$out" "$@" > "logs/${tag}.log" 2>&1
  if [ ! -f "$out" ]; then
    echo "  !! $tag DIED"; tail -5 "logs/${tag}.log" | sed 's/^/     /'; return
  fi
  python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sk = d.get("skipped_records") or []
print("    n=%s cs=%s schema=%s arg=%s exact=%s zp=%s oom=%s skipped=%d" % (
    d.get("n"), d.get("cs_rate"), d.get("schema_valid_rate"),
    d.get("arg_accuracy"), d.get("exact_call_rate"),
    d.get("zero_partition"), d.get("oom"), len(sk)))
for r in sk:
    print("      excluded: %s (%s)" % (r.get("id"), r.get("reason")))
PY
}

N=130
run rerun_mask_sample      --task bfcl_live_simple --variant mask --emission sample --n $N
run rerun_j1_sample        --task bfcl_live_simple --variant j1   --emission sample --n $N
run rerun_j0_sample        --task bfcl_live_simple --variant j0   --emission sample --n $N
run rerun_j2_sample        --task bfcl_live_simple --variant j2   --emission sample --n $N
run rerun_marmap_b0.1      --task bfcl_live_simple --variant j0   --emission map --confidence mar --entropy-bound 0.1 --n $N
run rerun_marmap_b0.01     --task bfcl_live_simple --variant j0   --emission map --confidence mar --entropy-bound 0.01 --n $N
run rerun_marmap_b0.5      --task bfcl_live_simple --variant j0   --emission map --confidence mar --entropy-bound 0.5 --n $N
echo "ALL DONE $(date -u +%H:%M)"
