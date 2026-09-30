"""Pick the blend coefficient by CLOSED-LOOP success on the tuning seeds.

    select_c.py <anchor_glob> <tag_prefix> "<coeffs>" [need_seeds]

Tags are <tag_prefix><c>_s<seed>.  Prints the best c on stdout (or NONE);
per-coefficient paired deltas go to stderr.
"""
import glob
import json
import os
import sys

import numpy as np

RUNS = os.environ.get("CUBE_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))


def load(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, pat))):
        f = os.path.join(p, "result.json")
        if os.path.exists(f):
            d = json.load(open(f))
            out[d["seed"]] = d
    return out


def main():
    anchor_glob, prefix, coeffs = sys.argv[1], sys.argv[2], sys.argv[3].split()
    need = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    anchor = load(anchor_glob)
    best = None
    for c in coeffs:
        g = load(f"{prefix}{c}_s*")
        da, db = [], []
        for s, ra in anchor.items():
            rb = g.get(s)
            if rb is None:
                continue
            assert ra["episodes_idx"] == rb["episodes_idx"], f"episode set differs, seed {s}"
            da.append(np.array(ra["episode_successes"], float))
            db.append(np.array(rb["episode_successes"], float))
        if len(da) < need:
            print(f"  c={c}: only {len(da)}/{need} seeds, skipped", file=sys.stderr)
            continue
        diff = np.concatenate(db) - np.concatenate(da)
        d = float(diff.mean() * 100)
        rng = np.random.default_rng(0)
        boot = diff[rng.integers(0, len(diff), size=(20000, len(diff)))].mean(axis=1) * 100
        lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
        print(f"  c={c}: delta={d:+.2f} [{lo:+.2f}, {hi:+.2f}] n={len(diff)} "
              f"flips +{int((diff > 0).sum())}/-{int((diff < 0).sum())}", file=sys.stderr)
        if best is None or d > best[1]:
            best = (c, d)
    print(f"{best[0]} {best[1]:.4f}" if best else "NONE 0")


if __name__ == "__main__":
    main()
