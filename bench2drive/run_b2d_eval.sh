#!/bin/bash
# Closed-loop Bench2Drive evaluation of World4Scorer.
#   bash bench2drive/run_b2d_eval.sh <routes_xml> <subset|-> <agent_cfg_json> <result_json> [gpu]
# Agent config: b2d_agent/config.json.
# Requires CARLA 0.9.15 and a Bench2Drive checkout (leaderboard + scenario_runner).
set -euo pipefail
: "${BENCH2DRIVE_ROOT:?set BENCH2DRIVE_ROOT to your Bench2Drive checkout}"
: "${CARLA_ROOT:?set CARLA_ROOT to CARLA 0.9.15}"
ROUTES=$1; SUBSET=$2; CFG=$(realpath "$3"); RESULT=$(realpath -m "$4"); GPU=${5:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$BENCH2DRIVE_ROOT/leaderboard:$BENCH2DRIVE_ROOT/scenario_runner:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI:$HERE:$(dirname "$HERE"):${PYTHONPATH:-}
export SCENARIO_RUNNER_ROOT=$BENCH2DRIVE_ROOT/scenario_runner LEADERBOARD_ROOT=$BENCH2DRIVE_ROOT/leaderboard
export IS_BENCH2DRIVE=True HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
SUBARG=""
[ "$SUBSET" != "-" ] && SUBARG="--routes-subset=$SUBSET"
# the evaluator resolves weather.xml relative to cwd; a relative "ckpt" in the agent
# config is resolved against the repository root
cd "$BENCH2DRIVE_ROOT" && exec env CUDA_VISIBLE_DEVICES=$GPU python \
  leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes="$ROUTES" $SUBARG --repetitions=1 --track=SENSORS \
  --checkpoint="$RESULT" \
  --agent="$HERE/b2d_agent/world4scorer_agent.py" --agent-config="$CFG" \
  --resume=True --port="${PORT:-2000}" --traffic-manager-port="${TM_PORT:-8000}" --timeout=600 \
  --gpu-rank="$GPU"
