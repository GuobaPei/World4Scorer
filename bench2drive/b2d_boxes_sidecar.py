#!/usr/bin/env python3
"""b2d_boxes_sidecar.py — per-sample other-agent box trajectories, ego-frame
at sample time t. Output aligned row-for-row with samples.jsonl:
  boxes (n, 9, 24, 5) f16  [x, y, yaw, half_len, half_wid], step k = t + 5k
  bmask (n, 9, 24) bool
Parallel over clips (each clip's annos read once)."""
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

DATA = sys.argv[1] if len(sys.argv) > 1 else _DATA
INDEX = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_DATA, "samples.jsonl")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_DATA, "boxes.npz")
STRIDE, STEPS, MAXA, RADIUS = 5, 9, 24, 45.0
GHOST_STEPS = 2                      # carry a dropped agent for 1.0 s

rows = [json.loads(l) for l in open(INDEX)]
by_clip = {}
for i, r in enumerate(rows):
    by_clip.setdefault(r["clip"], []).append((i, r["t"]))


def _ego_xy(anno):
    """Ego position in math frame from the ego box centre; anno x,y is GNSS."""
    for b in anno.get("bounding_boxes", ()):
        if b.get("class") == "ego_vehicle":
            c = b["center"]
            return float(c[0]), -float(c[1])
    return float(anno["x"]), -float(anno["y"])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load(clip, t):
    with gzip.open(os.path.join(DATA, clip, "anno", f"{t:05d}.json.gz")) as f:
        return json.load(f)


def do_clip(clip):
    items = by_clip[clip]
    out = []
    cache = {}
    for i, t in items:
        try:
            a0 = cache.get(t) or cache.setdefault(t, load(clip, t))
            th = math.pi / 2 - float(a0["theta"])
            x0, y0 = _ego_xy(a0)
            c, s = math.cos(th), math.sin(th)
            B = np.zeros((STEPS, MAXA, 5), np.float16)
            M = np.zeros((STEPS, MAXA), bool)
            ghosts = {}     # id -> (world x, world y, world yaw rad, speed, ext)
            for k in range(STEPS):
                tk = t + STRIDE * k
                ak = cache.get(tk) or cache.setdefault(tk, load(clip, tk))
                cand = []
                live = set()
                for b in ak.get("bounding_boxes", []):
                    # allow-list: only vehicles and walkers are obstacles (not signs or lights)
                    if b.get("class") not in ("vehicle", "walker"):
                        continue
                    ext = b.get("extent") or [0, 0, 0]
                    if ext[0] <= 0.0 or ext[1] <= 0.0:
                        continue
                    wx, wy = float(b["center"][0]), -float(b["center"][1])
                    dx, dy = wx - x0, wy - y0
                    rx, ry = c * dx + s * dy, -s * dx + c * dy
                    d2 = rx * rx + ry * ry
                    if d2 > RADIUS * RADIUS:
                        continue
                    # rotation[2] is the raw CARLA yaw, NOT the ego's `theta`
                    # compass: the two differ by exactly +90 deg.
                    byaw = wrap(-math.radians(float(b["rotation"][2])) - th)
                    cand.append((d2, rx, ry, byaw, ext[0], ext[1]))
                    aid = b.get("id")
                    if aid is not None:
                        live.add(aid)
                        ghosts[aid] = (wx, wy,
                                       math.radians(float(b["rotation"][2])),
                                       float(b.get("speed", 0.0) or 0.0),
                                       (ext[0], ext[1]), k)
                # carry forward agents the annotation dropped: they did not
                # teleport away, and a trajectory aimed at one must not be
                # labelled safe. Constant-velocity along last observed heading.
                for aid, (gx, gy, gyaw, gspd, gext, gk) in ghosts.items():
                    # only carry a dropped agent for GHOST_STEPS: constant
                    # velocity past ~1 s invents phantoms for cars that turn
                    # or brake.
                    if aid in live or k <= gk or k - gk > GHOST_STEPS:
                        continue
                    dt = (k - gk) * STRIDE * 0.1          # 10 Hz annotations
                    px = gx + gspd * dt * math.cos(gyaw)
                    py = gy - gspd * dt * math.sin(gyaw)  # world y is negated
                    dx, dy = px - x0, py - y0
                    rx, ry = c * dx + s * dy, -s * dx + c * dy
                    if rx * rx + ry * ry > RADIUS * RADIUS:
                        continue
                    cand.append((rx * rx + ry * ry, rx, ry,
                                 wrap(-gyaw - th), gext[0], gext[1]))
                cand.sort(key=lambda z: z[0])
                for j, (_, rx, ry, byaw, ex, ey) in enumerate(cand[:MAXA]):
                    B[k, j] = (rx, ry, byaw, ex, ey)
                    M[k, j] = True
            out.append((i, B, M))
        except Exception:
            out.append((i, np.zeros((STEPS, MAXA, 5), np.float16),
                        np.zeros((STEPS, MAXA), bool)))
    return out


if __name__ == "__main__":
    n = len(rows)
    boxes = np.zeros((n, STEPS, MAXA, 5), np.float16)
    bmask = np.zeros((n, STEPS, MAXA), bool)
    clips = sorted(by_clip)
    with Pool(48) as p:
        for ci, res in enumerate(p.imap_unordered(do_clip, clips, chunksize=4)):
            for i, B, M in res:
                boxes[i], bmask[i] = B, M
            if ci % 100 == 0:
                print(f"[boxes] {ci}/{len(clips)} clips", flush=True)
    np.savez(OUT, boxes=boxes, bmask=bmask)
    print(f"[boxes] saved {OUT} n={n} filled={bmask.any(axis=(1,2)).sum()}", flush=True)
