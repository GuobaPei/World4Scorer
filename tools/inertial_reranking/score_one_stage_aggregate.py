"""Aggregate one-stage shard rows into the final navtest EPDMS CSV.
Monkeypatches the worker fan-out of run_pdm_score_one_stage, then runs its ORIGINAL
main() so adjacent-frame two-frame-EC aggregation + final CSV are byte-identical to
upstream. Run from navsim/planning/script/. Pass +rows_glob=<glob of shard_*.pkl>."""
import glob
import pickle

import run_pdm_score_one_stage as M


def _load_rows(worker, fn, data_points):
    cfg = data_points[0]["cfg"]
    rows = []
    paths = sorted(glob.glob(cfg.rows_glob))
    for p in paths:
        with open(p, "rb") as f:
            rows.extend(pickle.load(f))
    print(f"AGG_LOADED shards={len(paths)} rows={len(rows)} glob={cfg.rows_glob}")
    return rows


M.worker_map = _load_rows
M.build_worker = lambda cfg: None

if __name__ == "__main__":
    M.main()
