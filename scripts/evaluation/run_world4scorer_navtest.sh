#!/bin/bash
# NAVSIM-v1 evaluation on navtest: write a submission pickle with the V1 selection
# weights (noc, dac, ddc, ttc, ep, comfort) = (1, 1, 0, 5, 5, 2), then score it with
# the NAVSIM v1.1 PDM scorer in this repository.
#   bash scripts/evaluation/run_world4scorer_navtest.sh <checkpoint.ckpt> [out_dir]
# Needs the navtest metric cache: navsim/planning/script/run_metric_caching.py
#   train_test_split=navtest cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache
# NAVSIM-v2 (EPDMS) and inertial re-ranking: tools/inertial_reranking/run_navsim_v2.sh.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"
: "${DINO_WEIGHTS:?}"
CKPT=$(realpath "${1:?checkpoint}"); OUT=$(realpath -m "${2:-$NAVSIM_EXP_ROOT/eval_navtest}")
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export SUBSCORE_PATH=$OUT
mkdir -p "$OUT"
cd "$NAVSIM_DEVKIT_ROOT/navsim/planning/script"
python run_create_submission_pickle_warmup_gpu.py \
  train_test_split=navtest experiment_name=w4s_navtest output_dir="$OUT" \
  agent=drivoR "agent.checkpoint_path='$CKPT'" \
  "agent.config.image_backbone.model_weights=$DINO_WEIGHTS" \
  agent.config.refiner_ls_values=0.0 agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true agent.config.proposal_num=64 \
  agent.config.refiner_num_heads=1 agent.config.tf_d_model=256 agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false agent.config.agent_pred=false agent.config.ref_num=4 \
  agent.config.long_trajectory_additional_poses=2 agent.loss.prev_weight=0.0 \
  agent.config.noc=1 agent.config.dac=1 agent.config.ddc=0.0 \
  agent.config.ttc=5 agent.config.ep=5 agent.config.comfort=2 \
  ++trainer.params.logger=false \
  "team_name= " authors=anonymous email=anonymous institution=anonymous country=anonymous
SUB=$(ls -t "$OUT"/submission.pkl "$OUT"/*/submission.pkl 2>/dev/null | head -1)
[ -n "$SUB" ] || { echo "no submission.pkl under $OUT"; exit 1; }
python run_pdm_score_from_submission.py \
  train_test_split=navtest "submission_file_path=$SUB" \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" output_dir="$OUT/v1_score"
