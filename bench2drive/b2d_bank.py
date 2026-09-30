#!/usr/bin/env python3
"""b2d_bank.py — future targets for Bench2Drive: the frozen DINOv2 feature of the front
camera 2 s ahead of each sample (JPEGs decoded on the GPU)."""
import json, os, sys
_B2D = os.path.dirname(os.path.abspath(__file__))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DATA = os.environ.get("B2D_DATA", os.path.join(_B2D, "data"))
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
import numpy as np
import torch
import timm
from torchvision.io import decode_jpeg
import torch.nn.functional as F

DATA = sys.argv[1] if len(sys.argv) > 1 else _DATA
INDEX = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_DATA, "samples.jsonl")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_DATA, "b2d_bank_f2s.pt")
FUT = 20

m = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=False, num_classes=0)
import safetensors.torch as st
sd = st.load_file(_DINO)
mi, un = m.load_state_dict(sd, strict=False)
print("bank encoder load: missing", len(mi), "unexpected", len(un), flush=True)
m = m.eval().cuda()
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

rows = [json.loads(l) for l in open(INDEX)]
bank, miss = {}, 0
bufs, keys = [], []


def flush():
    global bufs, keys
    if not bufs:
        return
    try:
        imgs = decode_jpeg(bufs, device="cuda")
    except Exception:
        imgs = [decode_jpeg(b).cuda() for b in bufs]
    x = torch.stack(imgs).float() / 255.0
    x = F.interpolate(x, size=(518, 518), mode="bicubic", align_corners=False, antialias=True)
    x = (x - MEAN) / STD
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        feats = m.forward_features(x)
        emb = feats[:, 5:, :].mean(1)          # skip cls+4 reg, mean-pool patches
    for k, e in zip(keys, emb.float().cpu()):
        bank[k] = e
    bufs, keys = [], []


for i, r in enumerate(rows):
    p = os.path.join(DATA, r["clip"], "camera", "rgb_front", f"{r['t'] + FUT:05d}.jpg")
    if not os.path.exists(p):
        miss += 1
        continue
    with open(p, "rb") as f:
        bufs.append(torch.frombuffer(bytearray(f.read()), dtype=torch.uint8))
    keys.append(f"{r['clip']}_{r['t']:05d}")
    if len(bufs) == 96:
        flush()
    if i % 4000 == 0:
        print(f"[bank] {i}/{len(rows)} miss {miss}", flush=True)
flush()
torch.save(bank, OUT)
print(f"BANK_DONE {len(bank)} tokens, {miss} missing -> {OUT}", flush=True)
