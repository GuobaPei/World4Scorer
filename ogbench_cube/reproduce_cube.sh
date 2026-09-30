#!/usr/bin/env bash
# Outcome-based scoring inside the LeWM planner on OGBench-Cube.
# CEM with 10 iterations, as in the LeWM evaluation protocol; everything else at the LeWM defaults.
#   GPU=0 bash ogbench_cube/reproduce_cube.sh
# 1. anchor: released LeWM cost on the 11 report seeds and the 4 tuning seeds
# 2. collect simulator-labelled candidates on 8 disjoint seeds (planner driven by LeWM)
# 3. train the outcome head, excluding every episode used for tuning or reporting
# 4. choose the blend coefficient in closed loop on the tuning seeds only
# 5. evaluate the selected blend on the report seeds
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export CUBE_RUNS=${CUBE_RUNS:-$HERE/runs}
REPORT_SEEDS="42 0 1 2 3 4 5 6 7 8 9"
TUNE_SEEDS="200 201 202 203"
COLLECT_SEEDS="100 101 102 103 104 105 106 107"
COEFFS="0.1 0.2 0.3 0.5 1.0"
HEAD=$HERE/heads/n10_r1

one() {  # tag mode overrides...
  local T=$1 M=$2; shift 2
  [ -f "$CUBE_RUNS/$T/result.json" ] && return 0
  bash "$HERE/run.sh" "$M" solver.n_steps=10 tag="$T" "$@"
}

for s in $REPORT_SEEDS; do one "n10_cube_lewm_s$s" lewm seed="$s"; done
for s in $TUNE_SEEDS; do one "n10_cube_tuneanchor_s$s" lewm seed="$s"; done
for s in $COLLECT_SEEDS; do one "n10_cube_collect_s$s" collect seed="$s"; done

[ -f "$HEAD/head.pt" ] || python "$HERE/train_head.py" \
  --banks "$CUBE_RUNS"/n10_cube_collect_s1[0-9][0-9] \
  --block-from "$CUBE_RUNS"/n10_cube_lewm_s*/result.json "$CUBE_RUNS"/n10_cube_tuneanchor_s*/result.json \
  --out "$HEAD"

for c in $COEFFS; do for s in $TUNE_SEEDS; do
  one "n10_cube_tunen10_r1c${c}_s$s" blend seed="$s" cost.head_path="$HEAD" cost.coefficient="$c"
done; done
read -r C D <<< "$(python "$HERE/select_c.py" "n10_cube_tuneanchor_s*" "n10_cube_tunen10_r1c" "$COEFFS" 4)"
echo "selected coefficient c=$C (tuning-seed delta $D; the released head was run with c=0.3)"
[ "$C" != NONE ] || exit 1

for s in $REPORT_SEEDS; do
  one "n10_cube_oursn10_r1c${C}_s$s" blend seed="$s" cost.head_path="$HEAD" cost.coefficient="$C"
done
python - "$CUBE_RUNS" "$C" <<'PY'
import glob, json, os, sys
runs, c = sys.argv[1], sys.argv[2]
def rate(pat):
    r = {json.load(open(p))["seed"]: json.load(open(p))["success_rate"]
         for p in glob.glob(os.path.join(runs, pat, "result.json"))}
    return sum(r.values()) / max(len(r), 1), len(r)
a, na = rate("n10_cube_lewm_s*"); o, no = rate(f"n10_cube_oursn10_r1c{c}_s*")
print(f"LeWM cost {a:.2f} ({na} seeds)  ->  + outcome head {o:.2f} ({no} seeds)")
PY
