#!/usr/bin/env python3
"""b2d_route_tp_sidecar.py — add the route target point to an existing sidecar.

Reads one anno per sample, copies the 11 dims of sidecar.npz (b2d_sidecar.py) through
untouched and appends the route target point (2 dims). The first 11 columns are
asserted equal to the source.

  OMP_NUM_THREADS=4 python3 b2d_route_tp_sidecar.py \
      <anno_root> <samples.jsonl> <sidecar.npz> <routes.npz> <out.npz>
"""
import json, math, os, sys, time
_B2D = os.path.dirname(os.path.abspath(__file__))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DATA = os.environ.get("B2D_DATA", os.path.join(_B2D, "data"))
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
import numpy as np

sys.path.insert(0, _B2D)
from b2d_dataset import load_anno, _ego_xy                          # noqa: E402
from b2d_route_cache import load_cache                              # noqa: E402
from route_target import RouteCursor, route_target, TARGET_DIST, TARGET_DIM  # noqa: E402


def main():
    RAW, INDEX, SRC, ROUTES, OUT = sys.argv[1:6]
    rows = [json.loads(l) for l in open(INDEX)]
    z = np.load(SRC, allow_pickle=False)
    ego0 = z["ego"]
    n = len(rows)
    assert ego0.shape == (n, 11), f"source sidecar is {ego0.shape}, expected ({n}, 11)"
    routes = load_cache(ROUTES)
    print(f"[tp] {n} samples, {len(routes)} clip routes, D={TARGET_DIST} m", flush=True)

    tgt = np.zeros((n, TARGET_DIM), np.float32)
    tgt_end = np.zeros(n, bool)
    cmd_near = np.full(n, -1, np.int16)
    cmd_far = np.full(n, -1, np.int16)
    ok = z["ok"].copy()
    cur_clip, cursor, route = None, None, None
    n_noroute = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        clip, t = r["clip"], r["t"]
        if clip != cur_clip:
            cur_clip, cursor = clip, RouteCursor()
            route = routes.get(clip)
        if route is None or len(route) < 2:
            if ok[i]:
                n_noroute += 1
            ok[i] = False
            continue
        try:
            a0 = load_anno(os.path.join(RAW, clip, "anno", f"{t:05d}.json.gz"))
            th_m = math.pi / 2 - float(a0["theta"])
            ex, ey = _ego_xy(a0)
            fwd, lat, he = route_target(route, cursor, ex, ey, th_m)
            if not (np.isfinite(fwd) and np.isfinite(lat)):
                ok[i] = False
                continue
            tgt[i] = (fwd, lat)
            tgt_end[i] = he
            cmd_near[i] = int(a0.get("command_near", -1))
            cmd_far[i] = int(a0.get("command_far", -1))
        except Exception:
            ok[i] = False
        if i % 4000 == 0:
            print(f"[tp] {i}/{n} ok={ok.sum()} ({time.time()-t0:.0f}s)", flush=True)

    ego = np.concatenate([ego0, tgt], axis=1).astype(np.float32)
    assert np.array_equal(ego[:, :11], ego0), "the original 11 dims changed"
    out = {k: z[k] for k in z.files}
    out["ego"] = ego
    out["ok"] = ok            # a sample whose target point could not be built is out
    out["tgt_end"] = tgt_end
    out["cmd_near"] = cmd_near
    out["cmd_far"] = cmd_far
    out["target_dist"] = np.float32(TARGET_DIST)
    tmp = OUT + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, OUT)
    print(f"ROUTE_TP_SIDECAR_DONE ego{ego.shape} ok={ok.sum()}/{n} "
          f"(dropped {int(z['ok'].sum()-ok.sum())}, {n_noroute} with no route) "
          f"saturated={100*tgt_end[ok].mean():.1f}% -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
