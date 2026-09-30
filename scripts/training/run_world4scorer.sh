#!/bin/bash
# Train World4Scorer on navtrain:
# 4 GPUs x 16 samples, AdamW 2e-4, 25 epochs, seed 3; lambda_fut = 1, lambda_bank = 0.5.
#   bash scripts/training/run_world4scorer.sh [experiment_name] [seed]
# Required environment (see README):
#   NAVSIM_DEVKIT_ROOT  this repository          OPENSCENE_DATA_ROOT  NAVSIM/OpenScene data
#   NUPLAN_MAPS_ROOT    nuPlan maps               NAVSIM_EXP_ROOT      output root; must contain
#                                                                      train_metric_cache/
#   FUTURE_BANK   output of tools/precompute_future_embeddings.py (+2 s front-camera targets)
#   BANK_PACK     candidate-bank pack directory (tokens/bank_candidates/bank_subscores/bank_mask .npy)
#   DINO_WEIGHTS  vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors
# Stage the data on local disk or tmpfs first: training is bound by random reads.
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"
: "${FUTURE_BANK:?}" "${BANK_PACK:?}" "${DINO_WEIGHTS:?}"
EXP=${1:-world4scorer}; SEED=${2:-3}
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export SUBSCORE_PATH=${SUBSCORE_PATH:-$NAVSIM_EXP_ROOT/subscore}
[ -d "$NAVSIM_EXP_ROOT/train_metric_cache" ] || { echo "run navsim/planning/script/run_train_metric_caching.py first"; exit 1; }
for f in tokens bank_candidates bank_subscores bank_mask; do
  [ -f "$BANK_PACK/$f.npy" ] || { echo "missing $BANK_PACK/$f.npy"; exit 1; }
done
# run_training_full.py resumes from the newest checkpoint under the PARENT of
# output_dir, so give every run its own parent directory.
OUT=$NAVSIM_EXP_ROOT/runs/$EXP/run
mkdir -p "$OUT"
cd "$NAVSIM_DEVKIT_ROOT/navsim/planning/script"
python run_training_full.py \
  agent=drivoR experiment_name="$EXP" output_dir="$OUT" train_test_split=navtrain \
  use_cache_without_dataset=false cache_path=null force_cache_computation=false \
  "agent.config.realized_future_emb_path=$FUTURE_BANK" agent.config.realized_weight=1.0 \
  agent.config.realized_lambda=1.0 \
  "agent.config.clover_pack_path=$BANK_PACK" agent.config.clover_k=16 \
  agent.config.clover_weight=0.5 agent.config.clover_safety_frac=0.5 \
  agent.lr_args.name=AdamW agent.lr_args.base_lr=0.0002 \
  agent.scheduler_args.dataset_size=103288 agent.num_gpus=4 agent.progress_bar=false \
  agent.config.refiner_ls_values=0.0 agent.config.image_backbone.focus_front_cam=false \
  "agent.config.image_backbone.model_weights=$DINO_WEIGHTS" \
  agent.config.one_token_per_traj=true agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false agent.config.agent_pred=false agent.config.ref_num=4 \
  agent.loss.prev_weight=0.0 agent.config.long_trajectory_additional_poses=2 \
  dataloader.params.batch_size=16 dataloader.params.num_workers=8 \
  dataloader.params.pin_memory=false dataloader.params.prefetch_factor=2 \
  ++trainer.params.devices=4 ++trainer.params.strategy=ddp \
  ++trainer.params.max_epochs=25 ++trainer.params.accumulate_grad_batches=1 \
  ++trainer.params.limit_train_batches=1.0 ++trainer.params.limit_val_batches=1.0 \
  ++trainer.params.val_check_interval=1.0 ++trainer.params.check_val_every_n_epoch=1 \
  ++trainer.params.num_sanity_val_steps=0 seed="$SEED"
