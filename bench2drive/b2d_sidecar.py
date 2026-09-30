#!/usr/bin/env python3
"""b2d_sidecar.py — one-time pass: parse all annos -> single npz of
ego(11)/fut(8,3)/cmd/token per sample + finite mask, so training reads no gzip/json
at runtime. b2d_route_tp_sidecar.py appends the route target point to it.

  python3 b2d_sidecar.py <data> <index.jsonl> <out.npz>
"""
import os, sys, json
_B2D = os.path.dirname(os.path.abspath(__file__))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DATA = os.environ.get("B2D_DATA", os.path.join(_B2D, "data"))
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
import numpy as np

sys.path.insert(0, _B2D)
from b2d_dataset import B2DDataset

DATA = sys.argv[1] if len(sys.argv) > 1 else _DATA
INDEX = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_DATA, "samples.jsonl")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_DATA, "sidecar.npz")

rows = [json.loads(l) for l in open(INDEX)]
n = len(rows)

import math
from b2d_dataset import load_anno, wrap, FUT_STRIDE, N_POSES
from model_config import build_ego_status

EGO_DIM = 11
ego = np.zeros((n, EGO_DIM), np.float32)
fut = np.zeros((n, 8, 3), np.float32)
cmd = np.zeros(n, np.int16)
ok = np.zeros(n, bool)
tokens = []

ds = B2DDataset.__new__(B2DDataset)   # reuse loading code paths without split logic
ds.root, ds.rows, ds.val = DATA, rows, False

for i, r in enumerate(rows):
    clip, t = r["clip"], r["t"]
    tokens.append(f"{clip}_{t:05d}")
    try:
        a0 = load_anno(os.path.join(DATA, clip, "anno", f"{t:05d}.json.gz"))
        th_m = math.pi / 2 - float(a0["theta"])
        spd = float(a0["speed"])
        vel_w = [spd * math.cos(th_m), spd * math.sin(th_m)]
        acc = a0.get("acceleration", 0.0)
        if isinstance(acc, (list, tuple)):
            acc_w = [float(acc[0]), -float(acc[1])]
        else:
            acc_w = [float(acc) * math.cos(th_m), float(acc) * math.sin(th_m)]
        vel_w = [max(-45.0, min(45.0, vel_w[0])), max(-45.0, min(45.0, vel_w[1]))]
        acc_w = [max(-15.0, min(15.0, acc_w[0])), max(-15.0, min(15.0, acc_w[1]))]
        cb = min(max(int(a0.get("next_command", 4)), 1), 6) - 1
        e = build_ego_status({"velocity": vel_w, "heading": th_m,
                              "acceleration": acc_w}, cb).numpy()
        x0, y0 = float(a0["x"]), -float(a0["y"])
        c, s = math.cos(th_m), math.sin(th_m)
        f = np.zeros((N_POSES, 3), np.float32)
        for k in range(1, N_POSES + 1):
            ak = load_anno(os.path.join(DATA, clip, "anno", f"{t + FUT_STRIDE * k:05d}.json.gz"))
            dx, dy = float(ak["x"]) - x0, -float(ak["y"]) - y0
            f[k - 1] = [c * dx + s * dy, -s * dx + c * dy,
                        wrap((math.pi / 2 - float(ak["theta"])) - th_m)]
        if np.isfinite(e).all() and np.isfinite(f).all():
            ego[i], fut[i], cmd[i], ok[i] = e, np.clip(f, -150, 150), cb, True
    except Exception:
        pass
    if i % 4000 == 0:
        print(f"[sidecar] {i}/{n} ok={ok.sum()}", flush=True)

np.savez(OUT, ego=ego, fut=fut, cmd=cmd, ok=ok, tokens=np.array(tokens))
print(f"SIDECAR_DONE {ok.sum()}/{n} finite ego_dim={EGO_DIM} -> {OUT}", flush=True)
