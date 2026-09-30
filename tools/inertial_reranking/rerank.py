"""Inertial re-ranking of a pool dump, causal in time within each log.

For every token T with an adjacent previous token P (the official NAVSIM v2 pairing,
infer_start_adjacent_mapping), compute the official two-frame extended-comfort check
ec_j of each candidate j of T against the plan already selected at P, then select
    argmax_j  s_j + lambda * log(0.99 * ec_j + 0.01)
where s_j is the model's selection score in the pool dump. lambda = 0 keeps the
model's own argmax; its submission must reproduce the raw EPDMS, which is the check
that the pairing and the pool are consistent.

Writes sub_lam0.pkl and sub_lam1.pkl (one-stage submission format) to --out-dir.
Needs the NAVSIM v2 devkit on PYTHONPATH.

  python rerank.py --pool pool.pkl --sims-dir sims --out-dir out
"""
import argparse, glob, json, os, pickle, sys, time
import numpy as np
import pandas as pd
import navsim
from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_two_frame_extended_comfort,
)

sys.path.insert(0, os.path.join(os.path.dirname(navsim.__file__), "planning", "script"))
from run_pdm_score_one_stage import infer_start_adjacent_mapping  # official pairing  # noqa: E402

INTERVAL = 0.1
LAMBDAS = [0.0, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--sims-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    sims = {}
    for p in sorted(glob.glob(os.path.join(a.sims_dir, "shard_*.pkl"))):
        with open(p, "rb") as f:
            sims.update(pickle.load(f))
    with open(a.pool, "rb") as f:
        dump = pickle.load(f)
    df = pd.DataFrame({
        "token": list(sims.keys()),
        "log_name": [v["log_name"] for v in sims.values()],
        "start_time": [v["time_s"] for v in sims.values()],
        "frame_type": [SceneFrameType.ORIGINAL if v["orig"] else SceneFrameType.SYNTHETIC
                       for v in sims.values()],
    })
    mapping = infer_start_adjacent_mapping(df)
    print(f"loaded: sims={len(sims)} dump={len(dump)} pairs={len(mapping)} in {time.time()-t0:.0f}s", flush=True)
    toks = sorted(sims.keys(), key=lambda t: (sims[t]["log_name"], sims[t]["time_s"]))
    samp = TrajectorySampling(num_poses=8, time_horizon=4.0, interval_length=0.5)
    report = {"n_tokens": len(toks), "n_pairs": len(mapping), "runs": {}}
    chosen = {t: int(dump[t]["chosen_idx"]) for t in toks}
    pdm = {t: np.asarray(dump[t]["pdm_score"], dtype=np.float32).astype(np.float64) for t in toks}
    for lam in LAMBDAS:
        sel = dict(chosen)
        nchg = 0
        ec_sel = []
        for T in toks:
            P = mapping.get(T)
            if P is None:
                continue
            dt = sims[T]["time_s"] - sims[P]["time_s"]
            k = round(dt / INTERVAL)
            prev_states = sims[P]["cand_sims"][sel[P]][k:]
            if lam != 0.0:
                cands = sims[T]["cand_sims"][:, :-k]
                prev_o = np.tile(prev_states[None], (cands.shape[0], 1, 1))
                t_axis = np.arange(cands.shape[1]) * INTERVAL
                ec = ego_is_two_frame_extended_comfort(cands, prev_o, t_axis).astype(np.float64)
                total = pdm[T] + lam * np.log(0.99 * ec + 0.01)
                j = int(np.argmax(total))
                if j != chosen[T]:
                    nchg += 1
                sel[T] = j
                ec_sel.append(ec[j])
            else:
                cur = sims[T]["cand_sims"][sel[T]][:-k]
                t_axis = np.arange(cur.shape[0]) * INTERVAL
                ec_sel.append(float(ego_is_two_frame_extended_comfort(
                    cur[None], prev_states[None], t_axis)[0]))
        tag = f"lam{'0' if lam == 0.0 else '1'}"
        d = {t: Trajectory(poses=np.asarray(dump[t]["proposals"][sel[t]], np.float32),
                           trajectory_sampling=samp) for t in toks}
        out = {"predictions": [d], "first_stage_predictions": [d]}
        path = os.path.join(a.out_dir, f"sub_{tag}.pkl")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(out, f, protocol=4)
        os.replace(tmp, path)
        report["runs"][tag] = dict(n_changed=nchg, n_pairs=len(ec_sel),
                                   pred_mean_ec=float(np.mean(ec_sel)) if ec_sel else None,
                                   path=path)
        print(f"{tag}: changed={nchg}/{len(ec_sel)} pred_mean_EC={np.mean(ec_sel):.4f} -> {path}", flush=True)
    with open(os.path.join(a.out_dir, "rerank_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"RERANK_DONE dt={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
