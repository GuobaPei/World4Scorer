#!/usr/bin/env python3
"""b2d_vocab.py — factorized trajectory dictionary from the training expert:
256 path shapes (k-means on arclength-normalized 8-pt paths) x 16 speed
profiles (trapezoidal ramps to target speeds 1..16 m/s). Composition and the
two-round coarse->fine sampler live here too (imported by trainer and agent).

Vocab entry (shape s, speed v): walk shape s's unit path at the cumulative
distances of profile v -> (8,3) [x, y, heading].
"""
import os
_B2D = os.path.dirname(os.path.abspath(__file__))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DATA = os.environ.get("B2D_DATA", os.path.join(_B2D, "data"))
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
import numpy as np

N_SHAPES, N_SPEEDS, HORIZON, DT = 256, 16, 8, 0.5
# starts at 0.0: without a stop cell the hardest brake from 10 m/s still
# travels 15.25 m and ends at 1 m/s, so red lights and cut-ins are unstoppable
TARGET_V = np.linspace(0.0, 16.0, N_SPEEDS)          # m/s
ACCEL = 2.0                                          # m/s^2 ramp from v0


def speed_profile(v0, vt):
    """8 step distances for a trapezoid v0 -> vt (accel-limited)."""
    v, out = v0, []
    for _ in range(HORIZON):
        dv = np.clip(vt - v, -4.0 * DT, ACCEL * DT)   # allow -8 m/s^2 braking
        v = max(0.0, v + dv)
        out.append(v * DT)
    return np.array(out)                              # (8,) segment lengths


def build_vocab(fut, ok, out_npz):
    """fut (n,8,3) expert futures -> shapes.npz {shapes (256, S, 2), lens}."""
    F = fut[ok]
    pts = np.concatenate([np.zeros((len(F), 1, 2)), F[:, :, :2]], 1)  # (n,9,2)
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    L = seg.sum(1)
    keep = L > 1.0                                    # drop stationary
    pts, L = pts[keep], L[keep]
    # arclength-resample every path to S points on [0,1]
    S = 32
    cum = np.concatenate([np.zeros((len(pts), 1)), np.cumsum(
        np.linalg.norm(np.diff(pts, axis=1), axis=2), axis=1)], 1)
    cum = cum / cum[:, -1:]
    grid = np.linspace(0, 1, S)
    shp = np.stack([np.stack([np.interp(grid, c, p[:, d]) for d in (0, 1)], 1)
                    for c, p in zip(cum, pts)])       # (n,S,2)
    shp = shp / np.maximum(L[:, None, None], 1e-3)    # unit-length shapes
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(N_SHAPES, batch_size=4096, n_init=3,
                         random_state=0).fit(shp.reshape(len(shp), -1))
    shapes = km.cluster_centers_.reshape(N_SHAPES, S, 2)
    np.savez(out_npz, shapes=shapes.astype(np.float32))
    print(f"[vocab] {N_SHAPES} shapes from {len(shp)} paths -> {out_npz}")
    return shapes


class Vocab:
    def __init__(self, npz_path):
        sh = np.load(npz_path)["shapes"]               # (256, S, 2)
        # k-means centres of unit paths are NOT unit length:
        # renormalise, and parameterise each shape by its own arclength rather
        # than assuming the S samples are equally spaced along it.
        seg = np.linalg.norm(np.diff(sh, axis=1), axis=2)          # (N,S-1)
        L = seg.sum(1)                                             # (N,)
        self.shapes = sh / np.maximum(L[:, None, None], 1e-6)
        c = np.concatenate([np.zeros((len(sh), 1)), np.cumsum(seg, axis=1)], 1)
        self.cum = c / np.maximum(c[:, -1:], 1e-6)                 # (N,S)

    def compose(self, sid, v0, vt):
        """shape sid at trapezoid v0->vt -> (8,3)."""
        segs = speed_profile(v0, vt)
        dist = np.cumsum(segs)
        total = max(dist[-1], 1e-3)
        sh = self.shapes[sid] * total                  # scale unit path
        xy = np.stack([np.interp(dist / total, self.cum[sid], sh[:, d])
                       for d in (0, 1)], 1)            # (8,2)
        d = np.diff(np.concatenate([np.zeros((1, 2)), xy]), axis=0)
        hd = np.arctan2(d[:, 1], d[:, 0])
        hd[np.linalg.norm(d, axis=1) < 0.05] = 0.0
        return np.concatenate([xy, hd[:, None]], 1).astype(np.float32)

    def round1(self, v0):
        """uniform 64: every 16th shape x 4 coarse speed bands -> (64,8,3)."""
        out, meta = [], []
        for si in range(0, N_SHAPES, N_SHAPES // 16):          # 16 shapes
            for vi in range(0, N_SPEEDS, N_SPEEDS // 4):       # 4 speeds
                out.append(self.compose(si, v0, TARGET_V[vi]))
                meta.append((si, vi))
        return np.stack(out), meta                             # (64,8,3)

    def round2(self, winners, v0, budget=64):
        """expand winner (si,vi) cells: neighbor shapes x all speeds."""
        # split the budget over shapes AND speeds. The old code let the inner
        # speed loop exhaust `per` at ds=0, so no winner ever reached a
        # different shape and 240/256 shapes were unreachable.
        n_sp = 4                                       # local speed window
        n_sh = max(1, budget // max(len(winners), 1) // n_sp)
        out, meta, seen = [], [], set()
        for si, vi in winners:
            for ds in range(n_sh):
                s2 = (si + ds - n_sh // 2) % N_SHAPES
                for dv in range(-(n_sp // 2), n_sp - n_sp // 2):
                    v2 = min(max(vi + dv, 0), N_SPEEDS - 1)
                    if (s2, v2) in seen:
                        continue
                    seen.add((s2, v2))
                    out.append(self.compose(s2, v0, TARGET_V[v2]))
                    meta.append((s2, v2))
        return np.stack(out), meta


if __name__ == "__main__":
    import sys
    sc = np.load(sys.argv[1])                          # sidecar.npz
    build_vocab(sc["fut"], sc["ok"].astype(bool),
                sys.argv[2] if len(sys.argv) > 2 else os.path.join(_B2D, "b2d_shapes.npz"))
