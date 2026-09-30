#!/usr/bin/env python3
"""b2d_route_cache.py — one route polyline per B2D clip, so the sidecar and the
dataset can compute the route target point without re-reading every anno.

The clips ship no route file. Every anno frame does store the node the expert's
dense planner is heading for (`x_command_near`), and that planner walks the
leaderboard's 1 m interpolated route, so the ordered unique near nodes ARE the
route. The stream starts a few metres in and stops a few metres short of the
end; reconstruct_route_from_anno restores both from the spawn pose and the final
far node.

Writes <out>.npz: one flat float32 (M,2) array of all clip routes concatenated,
plus offsets and clip names.

  OMP_NUM_THREADS=4 nice -n 15 python3 b2d_route_cache.py <anno_root> <out.npz> \
      [shard_i] [n_shard]
"""
import gzip, json, os, sys, time
_B2D = os.path.dirname(os.path.abspath(__file__))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DATA = os.environ.get("B2D_DATA", os.path.join(_B2D, "data"))
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
import numpy as np

sys.path.insert(0, _B2D)
from route_target import reconstruct_route_from_anno   # noqa: E402


def load(p):
    with gzip.open(p) as f:
        return json.load(f)


def _ego_xy(anno):
    """Ego position in the math frame, from the ego bounding-box centre --
    the same accurate source b2d_dataset._ego_xy uses. anno['x']/['y'] is a
    noisy GNSS read."""
    for b in anno.get("bounding_boxes", ()):
        if b.get("class") == "ego_vehicle":
            c = b["center"]
            return float(c[0]), -float(c[1])
    return float(anno["x"]), -float(anno["y"])


def clip_route(anno_dir):
    frames = sorted(int(f[:5]) for f in os.listdir(anno_dir) if f.endswith(".json.gz"))
    near, first, last_far = [], None, None
    for t in frames:
        try:
            a = load(os.path.join(anno_dir, f"{t:05d}.json.gz"))
        except Exception:
            continue
        if first is None:
            first = _ego_xy(a)
        near.append((float(a["x_command_near"]), -float(a["y_command_near"])))
        last_far = (float(a["x_command_far"]), -float(a["y_command_far"]))
    if first is None or len(near) < 2:
        return None
    return reconstruct_route_from_anno(near, first, last_far)


def merge(out, parts):
    names, offs, pts = [], [0], []
    for p in parts:
        z = np.load(p, allow_pickle=False)
        xy, off, clip = z["xy"], z["off"], z["clip"]
        for i, c in enumerate(clip):
            r = xy[off[i]:off[i + 1]]
            names.append(str(c)); pts.append(r); offs.append(offs[-1] + len(r))
    assert len(names) == len(set(names)), "duplicate clip across shards"
    np.savez(out, xy=np.concatenate(pts), off=np.asarray(offs, np.int64),
             clip=np.asarray(names))
    print(f"ROUTE_CACHE_MERGED {len(names)} clips -> {out}")


def main():
    if sys.argv[1] == "--merge":
        return merge(sys.argv[2], sys.argv[3:])
    root, out = sys.argv[1], sys.argv[2]
    si = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ns = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    clips = sorted(c for c in os.listdir(root)
                   if os.path.isdir(os.path.join(root, c, "anno")))
    clips = clips[si::ns]
    names, offs, pts = [], [0], []
    t0 = time.time()
    for i, c in enumerate(clips):
        r = clip_route(os.path.join(root, c, "anno"))
        if r is None or len(r) < 2:
            print(f"[route] SKIP {c} (no usable near-node stream)", flush=True)
            continue
        names.append(c); pts.append(r.astype(np.float32)); offs.append(offs[-1] + len(r))
        if i % 50 == 0:
            print(f"[route] {i}/{len(clips)} {c} n={len(r)} ({time.time()-t0:.0f}s)",
                  flush=True)
    np.savez(out, xy=np.concatenate(pts) if pts else np.zeros((0, 2), np.float32),
             off=np.asarray(offs, np.int64), clip=np.asarray(names))
    print(f"ROUTE_CACHE_DONE {len(names)}/{len(clips)} clips -> {out}", flush=True)


def load_cache(path):
    """{clip: (N,2) float64 route} — float64 so route_target's hypot matches
    the agent, which reads carla float64 locations."""
    z = np.load(path, allow_pickle=False)
    xy, off, clip = z["xy"], z["off"], z["clip"]
    return {str(c): xy[off[i]:off[i + 1]].astype(np.float64)
            for i, c in enumerate(clip)}


if __name__ == "__main__":
    main()
