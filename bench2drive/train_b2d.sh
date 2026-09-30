#!/bin/bash
# Bench2Drive training of World4Scorer.
#   B2D_DATA=/path/to/bench2drive_base  bash bench2drive/train_b2d.sh [gpus] [stage]
# B2D_DATA holds the official 1000-clip Bench2Drive-base release (one directory per clip).
# Stages (default: all):
#   prep   index, ego/label sidecars, agent boxes, drivable raster, future-target bank, route point
#   blind  route-blind model, 15 epochs            -> $OUT/route_blind/ep14.ckpt
#   route  add the 20 m route point to the ego input (two zero-initialized columns),
#          warm start from the route-blind model, same data/loss/schedule
#                                                   -> $OUT/route_tp/ep14.ckpt (released checkpoint)
# The drivable raster needs a running CARLA 0.9.15 server (dump_drivable.py, port $CARLA_PORT).
set -euo pipefail
: "${B2D_DATA:?set B2D_DATA to the Bench2Drive-base directory}"
GPUS=${1:-0,1,2,3}; STAGE=${2:-all}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
OUT=${OUT:-$HERE/runs}
N=$(awk -F, '{print NF}' <<<"$GPUS")
D=$B2D_DATA
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1

train() {  # out sidecar [extra...]
  local out=$1 sidecar=$2; shift 2
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.run --nproc_per_node="$N" \
    b2d_train_v2ddp.py --out "$out" \
    --data "$D" --index "$D/samples.jsonl" --sidecar "$sidecar" \
    --boxes "$D/boxes.npz" --drivable "$D/drivable_sidecar.npy" --bank "$D/b2d_bank_f2s.pt" \
    --bs 24 --workers 4 --epochs 15 "$@" 2>&1 | tee -a "$out/train.log"
}

if [ "$STAGE" = all ] || [ "$STAGE" = prep ]; then
  python b2d_dataset.py "$D" "$D/samples.jsonl"
  python b2d_sidecar.py "$D" "$D/samples.jsonl" "$D/sidecar.npz"
  python b2d_boxes_sidecar.py "$D" "$D/samples.jsonl" "$D/boxes.npz"
  python dump_drivable.py "${CARLA_PORT:-2000}" "$D/drivable"
  python build_drivable_sidecar.py "$D/drivable_sidecar.npy"
  python b2d_bank.py "$D" "$D/samples.jsonl" "$D/b2d_bank_f2s.pt"
  python b2d_route_cache.py "$D" "$D/routes.npz"
  python b2d_route_tp_sidecar.py "$D" "$D/samples.jsonl" "$D/sidecar.npz" "$D/routes.npz" "$D/sidecar_tp.npz"
fi
# b2d_shapes.npz (trajectory vocabulary) ships with the code; it is k-means with
# random_state=0 on sidecar.npz and can be rebuilt with: python b2d_vocab.py $D/sidecar.npz

if [ "$STAGE" = all ] || [ "$STAGE" = blind ]; then
  train "$OUT/route_blind" "$D/sidecar.npz"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = route ]; then
  mkdir -p "$OUT/route_tp"
  [ -f "$OUT/route_tp/init13.ckpt" ] || \
    python tools/warm_start_route_tp.py "$OUT/route_blind/ep14.ckpt" "$OUT/route_tp/init13.ckpt" --ego-dim 13
  train "$OUT/route_tp" "$D/sidecar_tp.npz" --init "$OUT/route_tp/init13.ckpt"
fi
