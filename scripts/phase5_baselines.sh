#!/usr/bin/env bash
# SPEC §7.2's baseline table, run sequentially.
#
# Sequential is not a stylistic choice: each run holds ~51 GB of params and
# preallocates 0.90 of an 80 GB GPU, so two at once die on
# "Failed to initialize BLASLT support". Learned the hard way.
#
# Usage: scripts/phase5_baselines.sh [N_RECORDS] [TASK]
set -u
cd /home/ubuntu/diffgemma_fa
source env.sh

N="${1:-40}"
TASK="${2:-bfcl_live_simple}"

run () {   # run <variant> <emission>
  local v="$1" e="$2" tag="$1_$2"
  local out="artifacts/eval_${TASK}_${tag}.json"
  if [ -f "$out" ]; then
    echo "[skip] $tag (already have $out)"
    return
  fi
  echo "=== $tag (n=$N) ==="
  python -m diffgemma_fa.eval.run --task "$TASK" --variant "$v" \
      --emission "$e" --n "$N" --out "$out" \
      > "logs/eval_${TASK}_${tag}.log" 2>&1
  grep -E "^(cs_rate|arg_accuracy|nonempty_rate|exact_call_rate|n):" \
      "logs/eval_${TASK}_${tag}.log" | sed 's/^/    /'
}

# SPEC §7.2, in the order the argument is made:
run unconstrained sample     # 1. stock model — accuracy AND CS
run mask          sample     # 2. naive per-position masking — §2.8's target
run j2            sample     # 3. constrained at accepted, random elsewhere
run j1            sample     # 5. single joint draw, flattened non-accepted
run j0            map        # 4. decoupled emission, constrained MAP
run j0            sample     # 5. decoupled emission, joint draw

echo
echo "=== SUMMARY ==="
python - "$TASK" <<'PY'
import json, pathlib, sys
task = sys.argv[1]
print(f"{'variant':22} {'n':>4} {'CS':>7} {'parsed':>7} {'nonempty':>9} "
      f"{'arg_acc':>8} {'exact':>7}")
for p in sorted(pathlib.Path("artifacts").glob(f"eval_{task}_*.json")):
    d = json.load(open(p))
    print(f"{d['variant'] + '-' + d['emission']:22} {d['n']:>4} "
          f"{d['cs_rate']:>7.3f} {d['parsed']:>7} {d['nonempty_rate']:>9.3f} "
          f"{d['arg_accuracy']:>8.3f} {d['exact_call_rate']:>7.3f}")
PY
