#!/bin/bash
# NAVSIM-v2 (EPDMS, one-stage navtest, non-reactive) with inertial re-ranking.
#   bash tools/inertial_reranking/run_navsim_v2.sh <checkpoint.ckpt> <work_dir> [n_shards]
# 1. candidate pool of the checkpoint on navtest (this repository, GPU)
# 2. LQR rollouts of every candidate (NAVSIM v2 devkit, CPU, sharded)
# 3. inertial re-ranking, lambda in {0, 1}
# 4. official one-stage EPDMS of each submission (NAVSIM v2 devkit, CPU, sharded)
# Selection weights (noc, dac, ddc, ttc, ep, comfort) = (10, 13, 6, 14, 15, 2.1), those of the
# released DrivoR NAVSIM-v2 model. lam0 = World4Scorer, lam1 = + inertial re-ranking.
# NAVSIM_V2_ROOT is a checkout of the official NAVSIM v2 devkit (our copy reports 2.0.0);
# copy score_one_stage_{shard,aggregate}.py into $NAVSIM_V2_ROOT/navsim/planning/script/.
# V2_METRIC_CACHE is its navtest metric cache (run_metric_caching.py in that devkit).
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_V2_ROOT:?}" "${V2_METRIC_CACHE:?}" "${OPENSCENE_DATA_ROOT:?}"
: "${NUPLAN_MAPS_ROOT:?}" "${DINO_WEIGHTS:?}"
CKPT=$(realpath "${1:?checkpoint}"); W=$(realpath -m "${2:?work_dir}"); NS=${3:-16}
HERE=$(cd "$(dirname "$0")" && pwd)
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$W/sims" "$W/subs"

echo "[1/4] pool dump"
( cd "$NAVSIM_DEVKIT_ROOT/navsim/planning/script" && \
  export PYTHONPATH=$NAVSIM_DEVKIT_ROOT NAVSIM_EXP_ROOT=$W SUBSCORE_PATH=$W && \
  python run_pool_dump.py \
    train_test_split=navtest experiment_name=w4s_pool agent=drivoR "agent.checkpoint_path='$CKPT'" \
    "+trainer.params.default_root_dir=$W" \
    agent.batch_size=64 agent.progress_bar=false \
    "agent.config.image_backbone.model_weights=$DINO_WEIGHTS" \
    agent.config.tf_d_model=256 agent.config.tf_d_ffn=1024 agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true agent.config.refiner_num_heads=1 \
    agent.config.area_pred=false agent.config.agent_pred=false agent.config.ref_num=4 \
    agent.config.long_trajectory_additional_poses=2 \
    agent.config.noc=10 agent.config.dac=13 agent.config.ddc=6 \
    agent.config.ttc=14 agent.config.ep=15 agent.config.comfort=2.1 \
    agent.loss.prev_weight=0.0 \
    "team_name= " authors=anonymous email=anonymous institution=anonymous country=anonymous \
    "+pool_out=$W/pool.pkl" )

export PYTHONPATH=$NAVSIM_V2_ROOT NAVSIM_DEVKIT_ROOT=$NAVSIM_V2_ROOT/navsim CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[2/4] LQR rollouts ($NS shards)"
for i in $(seq 0 $((NS-1))); do
  python "$HERE/simulate_candidates.py" --pool "$W/pool.pkl" --metric-cache "$V2_METRIC_CACHE" \
    --out-dir "$W/sims" --shard-idx "$i" --shard-count "$NS" > "$W/sims/sim_$i.log" 2>&1 &
done
wait
[ "$(ls "$W"/sims/shard_*.pkl | wc -l)" -eq "$NS" ] || { echo "rollout shards missing, see $W/sims/*.log"; exit 1; }

echo "[3/4] inertial re-ranking"
python "$HERE/rerank.py" --pool "$W/pool.pkl" --sims-dir "$W/sims" --out-dir "$W/subs"

echo "[4/4] EPDMS"
cd "$NAVSIM_V2_ROOT/navsim/planning/script"
for tag in lam0 lam1; do
  SD=$W/score_$tag; rm -rf "$SD"; mkdir -p "$SD"
  for i in $(seq 0 $((NS-1))); do
    NAVSIM_EXP_ROOT=$SD SUBSCORE_PATH=$SD python score_one_stage_shard.py \
      experiment_name=w4s_$tag traffic_agents=non_reactive metric_cache_path="$V2_METRIC_CACHE" \
      output_dir="$SD" "+submission_file_path=$W/subs/sub_$tag.pkl" \
      "+shard_idx=$i" "+shard_count=$NS" > "$SD/shard_$i.log" 2>&1 &
  done
  wait
  NAVSIM_EXP_ROOT=$SD SUBSCORE_PATH=$SD python score_one_stage_aggregate.py \
    experiment_name=w4s_${tag}_agg traffic_agents=non_reactive metric_cache_path="$V2_METRIC_CACHE" \
    output_dir="$SD" "+rows_glob=$SD/shard_*.pkl" > "$SD/aggregate.log" 2>&1
  CSV=$(ls -t "$SD"/*.csv | head -1)
  python - "$CSV" "$tag" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
avg = [r for r in rows[1:] if r[1] == "average_all_frames"][0]
print(f"{sys.argv[2]:10s} EPDMS {float(avg[12]):.4f}  ({sys.argv[1]})")
PY
done
