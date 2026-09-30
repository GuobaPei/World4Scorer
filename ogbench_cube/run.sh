#!/usr/bin/env bash
# One LeWM-protocol evaluation on OGBench-Cube with a swappable candidate scorer.
#   bash ogbench_cube/run.sh <lewm|collect|blend> [hydra overrides...]
#   lewm    released LeWM planning cost (anchor)
#   collect LeWM cost drives the planner; every scored candidate is replayed in the
#           simulator and stored for head training
#   blend   LeWM cost + coefficient x outcome head (cost.head_path, cost.coefficient)
# Results: $CUBE_RUNS/<tag>/result.json (default ogbench_cube/runs).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:?"usage: run.sh lewm|collect|blend [hydra overrides...]"}; shift
export STABLEWM_HOME=${STABLEWM_HOME:-$HOME/.stable_worldmodel}
export CUBE_RUNS=${CUBE_RUNS:-$HERE/runs}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH="$HERE"
POLICY=quentinll/lewm-cube
CKPT=$STABLEWM_HOME/checkpoints/models--${POLICY/\//--}
[ -f "$CKPT/weights.pt" ] && [ -f "$CKPT/config.json" ] || {
  echo "LeWM-Cube weights missing at $CKPT (download $POLICY from the Hugging Face Hub)"; exit 1; }
exec python "$HERE/eval_protocol.py" --config-name cube policy=$POLICY cost="$MODE" "$@"
