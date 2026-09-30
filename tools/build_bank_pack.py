#!/usr/bin/env python3
"""Convert a CLOVER-format candidate file into the bank pack read during training.

Input: a pickle holding a list of scene dicts (or a dict with a "scenes", "data" or
"items" list). Each scene has
    token                  navtrain scene token
    valid                  scene-level flag; invalid scenes are skipped
    trajectories_relative  (M, T>=8, 3) candidates in the ego frame, (x, y, heading), 0.5 s steps
    scores                 PDM sub-scores of every candidate, either a dict of (M,) arrays
                           or a list of M dicts, with the keys in SCORE_KEYS
Output directory (see drivor_model._load_clover_pack):
    tokens.npy (N,) bytes, bank_candidates.npy (N, K, 8, 3) float32,
    bank_subscores.npy (N, K, 7) float32, bank_mask.npy (N, K) bool, manifest.json

  python tools/build_bank_pack.py candidates.pkl out_dir [--k 64]
  python tools/build_bank_pack.py --selftest
"""
import argparse, json, os, pickle
import numpy as np

SCORE_KEYS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
              "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "pdm_score"]


def scene_list(obj):
    if isinstance(obj, dict):
        for key in ("scenes", "data", "items"):
            if key in obj:
                return obj[key]
        raise ValueError("dict input needs a 'scenes', 'data' or 'items' list")
    return obj


def score_matrix(scores, m):
    if isinstance(scores, dict):
        return np.stack([np.asarray(scores[k], np.float32).reshape(m) for k in SCORE_KEYS], 1)
    return np.asarray([[float(s[k]) for k in SCORE_KEYS] for s in scores], np.float32).reshape(m, len(SCORE_KEYS))


def build(scenes, k=64):
    toks, cands, subs, mask = [], [], [], []
    for sc in scenes:
        if not bool(np.all(sc.get("valid", True))):
            continue
        traj = np.asarray(sc["trajectories_relative"], np.float32)[:, :8, :3]
        s = score_matrix(sc["scores"], len(traj))
        keep = np.flatnonzero(np.isfinite(traj).all((1, 2)) & np.isfinite(s).all(1))
        # First k finite candidates in file order. The pack behind the released checkpoint
        # ordered candidate families before truncating; this differs only for scenes with
        # more than k candidates.
        keep = keep[:k]
        c = np.zeros((k, 8, 3), np.float32); c[:len(keep)] = traj[keep]
        y = np.zeros((k, len(SCORE_KEYS)), np.float32); y[:len(keep)] = s[keep]
        m = np.zeros(k, bool); m[:len(keep)] = True
        toks.append(str(sc["token"])); cands.append(c); subs.append(y); mask.append(m)
    return (np.asarray(toks, dtype="S64"), np.stack(cands), np.stack(subs), np.stack(mask))


def write(out, arrays, source):
    os.makedirs(out, exist_ok=True)
    for name, a in zip(["tokens", "bank_candidates", "bank_subscores", "bank_mask"], arrays):
        np.save(os.path.join(out, f"{name}.npy"), a)
    json.dump({"source": os.path.basename(source), "num_tokens": int(len(arrays[0])),
               "bank_k": int(arrays[3].shape[1]), "bank_subscore_keys": SCORE_KEYS,
               "mean_valid_per_scene": float(arrays[3].sum(1).mean())},
              open(os.path.join(out, "manifest.json"), "w"), indent=1)


def selftest():
    rng = np.random.default_rng(0)
    def scene(tok, m, valid=True, nan=False, as_list=False):
        traj = rng.normal(size=(m, 10, 3)).astype(np.float32)
        if nan:
            traj[1, 0, 0] = np.nan
        sc = {k: rng.uniform(size=m) for k in SCORE_KEYS}
        if as_list:
            sc = [{k: sc[k][i] for k in SCORE_KEYS} for i in range(m)]
        return {"token": tok, "valid": valid, "trajectories_relative": traj, "scores": sc}
    scenes = {"scenes": [scene("a", 5), scene("b", 70), scene("c", 3, valid=False), scene("d", 4, nan=True, as_list=True)]}
    t, c, s, m = build(scene_list(scenes), k=64)
    assert list(t) == [b"a", b"b", b"d"], t
    assert c.shape == (3, 64, 8, 3) and s.shape == (3, 64, 7) and m.shape == (3, 64)
    assert m.sum(1).tolist() == [5, 64, 3]
    assert np.allclose(c[0, :5], scenes["scenes"][0]["trajectories_relative"][:, :8])
    assert np.allclose(s[0, :5, 6], scenes["scenes"][0]["scores"]["pdm_score"])
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    with open(a.src, "rb") as f:
        arrays = build(scene_list(pickle.load(f)), a.k)
    write(a.out, arrays, a.src)
    print(f"wrote {len(arrays[0])} scenes, mean valid candidates {arrays[3].sum(1).mean():.1f} -> {a.out}")


if __name__ == "__main__":
    main()
