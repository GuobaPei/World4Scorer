#!/usr/bin/env python3
"""Precompute the future-feature targets of World4Scorer.

For every navtrain token, take the front camera two seconds ahead (frame index 7,
the current frame being 3), encode it with the frozen pretrained DINOv2 ViT-S/14 with
registers (the same initial weights as the planner's encoder, never LoRA-updated),
mean-pool the patch tokens to a 384-d vector, and store {token: fp16 tensor}.

These vectors are the targets of the visual readout for the executed (logged)
trajectory; generated and bank candidates are supervised by simulator outcomes.

Run once (a GPU is recommended) and point agent.config.realized_future_emb_path
at the output (about 103k tokens x 384 x fp16, roughly 80 MB). Future frames
must be read from the full OpenScene sensor data, not a current-frame-only cache.

Example:
  python tools/precompute_future_embeddings.py \
    --repo $NAVSIM_DEVKIT_ROOT \
    --openscene $OPENSCENE_DATA_ROOT \
    --split trainval --scene-filter navtrain \
    --enc weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors \
    --batch 64 \
    --out $NAVSIM_EXP_ROOT/realized_future_emb_navtrain_f7_camf0.pt
"""
import argparse, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--openscene", required=True)
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--scene-filter", default="navtrain")
    ap.add_argument("--enc", required=True, help="dinov2 vits14 reg safetensors")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import torch, timm
    from PIL import Image
    import numpy as np
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig

    root = Path(args.openscene)
    meta, blob = root / "meta_datas" / args.split, root / "sensor_blobs" / args.split
    sf_yaml = (Path(args.repo) / "navsim/planning/script/config/common/train_test_split"
               / "scene_filter" / f"{args.scene_filter}.yaml")
    sl = SceneLoader(meta, blob, instantiate(OmegaConf.load(sf_yaml)),
                     SensorConfig.build_no_sensors())
    print(f"tokens={len(sl.scene_frames_dicts)}", flush=True)

    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True,
        pretrained_cfg_overlay=dict(file=args.enc), num_classes=0)
    model.eval().to(args.device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)

    bank, paths, toks, miss = {}, [], [], 0

    @torch.no_grad()
    def flush():
        if not paths:
            return
        ims = torch.stack([tf(Image.open(blob / p).convert("RGB")) for p in paths]).to(args.device)
        feats = model.forward_features(ims)          # [B, n_tokens, 384]
        emb = feats.mean(dim=1) if feats.dim() == 3 else feats
        for t, e in zip(toks, emb):
            bank[t] = e.half().cpu()
        paths.clear(); toks.clear()

    fut = 7   # frame 3 is the current frame; +4 frames at 2 Hz = 2.0 s ahead
    for i, (tok, frames) in enumerate(sl.scene_frames_dicts.items()):
        if len(frames) <= fut:
            miss += 1
            continue
        paths.append(frames[fut]["cams"]["CAM_F0"]["data_path"])
        toks.append(tok)
        if len(paths) >= args.batch:
            flush()
        if i % 5000 == 0:
            print(f"{i} done, bank={len(bank)}, miss={miss}", flush=True)
    flush()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(bank, args.out)
    print(f"SAVED {len(bank)} embeddings (miss={miss}) -> {args.out}")

if __name__ == "__main__":
    main()
