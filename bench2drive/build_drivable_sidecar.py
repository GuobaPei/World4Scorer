#!/usr/bin/env python3
"""build_drivable_sidecar.py — per-sample ego-frame drivable-area patch.

The drivable_area_compliance label of each candidate is read from this patch
(b2d_metric_labels.py), not from a corridor around the expert path, which would
penalise long candidates rather than illegal ones.

Rasters come from `dump_drivable.py` (CARLA `Map.generate_waypoints`, every
driving lane of all 12 towns the clips use). Here each sample gets the patch
resampled into its own ego frame, so the label stays a pure tensor lookup.

Patch: forward [-10, +50) m, lateral [-25, +25) m at 0.5 m -> (120, 100) uint8.
"""
import glob
import gzip
import json
import math
import os
_B2D = os.path.dirname(os.path.abspath(__file__))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DATA = os.environ.get("B2D_DATA", os.path.join(_B2D, "data"))
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
import sys
from multiprocessing import Pool

import numpy as np

DATA = _DATA
RAST = f"{DATA}/drivable"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/drivable_sidecar.npy"
F0, F1, L0, L1, RES = -10.0, 50.0, -25.0, 25.0, 0.5
NF, NL = int((F1 - F0) / RES), int((L1 - L0) / RES)          # 120, 100

rows = [json.loads(l) for l in open(f"{DATA}/samples.jsonl")]
by_clip = {}
for i, r in enumerate(rows):
    by_clip.setdefault(r["clip"], []).append((i, r["t"]))

# ego-frame sample lattice, shared by every sample
ff = F0 + RES * (np.arange(NF) + 0.5)
ll = L0 + RES * (np.arange(NL) + 0.5)
FG, LG = np.meshgrid(ff, ll, indexing="ij")                   # (NF, NL)

_cache = {}


def raster(town):
    if town not in _cache:
        p = f"{RAST}/{town}.npz"
        if not os.path.exists(p):
            _cache[town] = None
        else:
            d = np.load(p)
            _cache[town] = (d["grid"], float(d["x0"]), float(d["y0"]), float(d["res"]))
    return _cache[town]


def do_clip(clip):
    town = clip.split("_")[1]
    R = raster(town)
    out = []
    if R is None:
        for i, _ in by_clip[clip]:
            out.append((i, np.ones((NF, NL), np.uint8)))       # unknown -> permissive
        return out
    grid, x0, y0, res = R
    H, W = grid.shape
    for i, t in by_clip[clip]:
        try:
            with gzip.open(f"{DATA}/{clip}/anno/{t:05d}.json.gz") as f:
                a = json.load(f)
            eb = [b for b in a.get("bounding_boxes", ()) if b.get("class") == "ego_vehicle"]
            if eb:
                cx, cy = float(eb[0]["center"][0]), float(eb[0]["center"][1])
            else:
                cx, cy = float(a["x"]), float(a["y"])
            th = math.pi / 2 - float(a["theta"])                # math-frame heading
            c, s = math.cos(th), math.sin(th)
            # ego math frame (x fwd, y LEFT) -> world math -> CARLA world (y negated)
            wx = cx + c * FG - s * LG
            wy_math = (-cy) + s * FG + c * LG
            wy = -wy_math
            gi = ((wy - y0) / res).astype(np.int32)
            gj = ((wx - x0) / res).astype(np.int32)
            ok = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
            patch = np.zeros((NF, NL), np.uint8)
            patch[ok] = grid[gi[ok], gj[ok]]
            out.append((i, patch))
        except Exception:
            out.append((i, np.ones((NF, NL), np.uint8)))
    return out


if __name__ == "__main__":
    n = len(rows)
    arr = np.zeros((n, NF, NL), np.uint8)
    clips = sorted(by_clip)
    miss = sorted({c.split("_")[1] for c in clips if raster(c.split("_")[1]) is None})
    if miss:
        print(f"[drivable] NO RASTER for {miss} -> those samples are permissive")
    with Pool(48) as p:
        for k, res in enumerate(p.imap_unordered(do_clip, clips, chunksize=4)):
            for i, patch in res:
                arr[i] = patch
            if k % 200 == 0:
                print(f"[drivable] {k}/{len(clips)} clips", flush=True)
    np.save(OUT, arr)
    print(f"[drivable] saved {OUT} shape={arr.shape} "
          f"({arr.nbytes / 1e6:.0f} MB, {100 * arr.mean():.1f}% drivable)")
