"""LQR rollouts of every candidate in a pool dump, for inertial re-ranking.

Runs on CPU with the NAVSIM v2 devkit on PYTHONPATH (its PDMSimulator and metric
cache). Rows simulated per token: [pdm_traj, *64 candidates]. On the first tokens of
every shard the model's own choice is also rolled out in the two-row batch the official
scorer uses, [pdm_traj, trajectory], and must equal its candidate row.

  python simulate_candidates.py --pool pool.pkl --metric-cache <navtest_v2_metric_cache> \
      --out-dir sims --shard-idx I --shard-count N
"""
import argparse, os, pickle, time
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--metric-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--shard-count", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = Path(a.out_dir) / f"shard_{a.shard_idx}.pkl"
    if out_path.exists():
        print(f"SKIP existing {out_path}"); return
    with open(a.pool, "rb") as f:
        dump = pickle.load(f)
    loader = MetricCacheLoader(Path(a.metric_cache))
    tokens = sorted(set(dump) & set(loader.tokens))
    tokens = tokens[a.shard_idx::a.shard_count]
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    # trajectory sampling of the submission poses (8, 4.0, 0.5)
    samp = TrajectorySampling(num_poses=8, time_horizon=4.0, interval_length=0.5)
    res = {}
    t0 = time.time()
    checked = 0
    for n, tok in enumerate(tokens):
        mc = loader.get_from_token(tok)
        ego = mc.ego_state
        pdm_states = get_trajectory_as_array(mc.trajectory, proposal_sampling, ego.time_point)
        e = dump[tok]
        rows = [pdm_states]
        for p in e["proposals"]:
            tr = Trajectory(poses=np.asarray(p, np.float32), trajectory_sampling=samp)
            rows.append(get_trajectory_as_array(transform_trajectory(tr, ego), proposal_sampling, ego.time_point))
        states = np.stack(rows, axis=0)
        sims = simulator.simulate_proposals(states, ego)
        if checked < 3:
            sel = Trajectory(poses=np.asarray(e["trajectory"], np.float32), trajectory_sampling=samp)
            two = np.stack([pdm_states, get_trajectory_as_array(
                transform_trajectory(sel, ego), proposal_sampling, ego.time_point)], axis=0)
            s2 = simulator.simulate_proposals(two, ego)
            d = float(np.abs(s2[1] - sims[1 + int(e["chosen_idx"])]).max())
            print(f"CONSISTENCY tok={tok} maxdiff={d:.3e}", flush=True)
            if d > 1e-9:
                raise SystemExit(f"FATAL: candidate rollout differs from the two-row rollout ({d:.3e})")
            checked += 1
        res[tok] = dict(log_name=mc.log_name, time_s=float(mc.timepoint.time_s),
                        orig=bool(mc.scene_type == SceneFrameType.ORIGINAL),
                        cand_sims=np.asarray(sims[1:], dtype=np.float64))
        if (n + 1) % 200 == 0:
            print(f"shard {a.shard_idx}: {n+1}/{len(tokens)} {(time.time()-t0)/(n+1):.3f}s/tok", flush=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=4)
    os.rename(tmp, out_path)
    print(f"SHARD_DONE idx={a.shard_idx} ntok={len(res)} dt={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
