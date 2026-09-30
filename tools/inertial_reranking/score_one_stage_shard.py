"""Per-token EPDMS rows for ONE shard of a one-stage (navtest) submission.
Trajectory comes from submission.pkl first_stage_predictions instead of an agent;
row construction is copied verbatim from run_pdm_score_one_stage.run_pdm_score.
Run from navsim/planning/script/. Overrides: +submission_file_path=... +shard_idx=I
+shard_count=N traffic_agents=non_reactive|reactive metric_cache_path=... output_dir=..."""
import pickle
import traceback
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from omegaconf import DictConfig

from navsim.common.dataclasses import PDMResults
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    shard_idx = int(cfg.shard_idx)
    shard_count = int(cfg.shard_count)
    with open(cfg.submission_file_path, "rb") as f:
        sub = pickle.load(f)
    fs = sub["first_stage_predictions"][0]

    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling
    if cfg.traffic_agents == "non_reactive":
        traffic_agents_policy = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    elif cfg.traffic_agents == "reactive":
        traffic_agents_policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    else:
        raise ValueError(str(cfg.traffic_agents))

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tokens_all = sorted(set(fs.keys()) & set(metric_cache_loader.tokens))
    missing = len(fs) - len(tokens_all)
    tokens = tokens_all[shard_idx::shard_count]

    pdm_results = []
    for token in tokens:
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = fs[token]
            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row["endpoint_x"] = absolute_endpoint.x
            score_row["endpoint_y"] = absolute_endpoint.y
            score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row["ego_simulated_states"] = [ego_simulated_states]
        except Exception:
            traceback.print_exc()
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False
        score_row["token"] = token
        pdm_results.append(score_row)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"shard_{shard_idx}.pkl", "wb") as f:
        pickle.dump(pdm_results, f)
    print(f"SHARD_DONE idx={shard_idx} count={shard_count} rows={len(pdm_results)} missing_cache={missing}")


if __name__ == "__main__":
    main()
