"""Dump the candidate pool of World4Scorer on navtest for inertial re-ranking.

One forward pass per batch gives the 64 candidates, their sub-score logits and the
model's selection score under the agent's selection weights (run_navsim_v2.sh passes
the NAVSIM-v2 weights of the released DrivoR model). The argmax is the model's own.

Run from navsim/planning/script with the agent overrides of
tools/inertial_reranking/run_navsim_v2.sh plus +pool_out=<path.pkl>.
"""
import logging
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle_gpu"

SUB_KEYS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
            "time_to_collision_within_bound", "ego_progress", "comfort"]


class PoolDumpModule(AgentLightningModule):
    def predict_step(self, batch, batch_idx):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            out = self.agent.forward(features)
            chosen = torch.argmax(out["pdm_score"], dim=1)
        proposals = out["proposals"].detach().cpu().float().numpy()
        subs = torch.stack([out["pred_logit"][k] for k in SUB_KEYS], dim=-1).detach().cpu().float().numpy()
        pdm = out["pdm_score"].detach().cpu().float().numpy()
        traj = out["trajectory"].detach().cpu().float().numpy()
        chosen = chosen.cpu().numpy()
        return {tok: {"proposals": proposals[i].astype(np.float32),
                      "subscores": subs[i].astype(np.float32),
                      "pdm_score": pdm[i].astype(np.float32),
                      "chosen_idx": int(chosen[i]),
                      "trajectory": traj[i].astype(np.float32)}
                for i, tok in enumerate(tokens)}


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pool_out = cfg.pool_out
    ac = cfg.agent.config
    print(f"selection weights (noc, dac, ddc, ttc, ep, comfort) = "
          f"{(ac.noc, ac.dac, ac.ddc, ac.ttc, ac.ep, ac.comfort)}", flush=True)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    input_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=input_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    mod = PoolDumpModule(agent=agent)
    predictions = trainer.predict(mod, dataloader, return_predictions=True)
    merged = {}
    for d in predictions:
        merged.update(d)
    e = next(iter(merged.values()))
    print("POOL_DUMP_SHAPES proposals", e["proposals"].shape, "subscores", e["subscores"].shape,
          "pdm", e["pdm_score"].shape, flush=True)
    os.makedirs(os.path.dirname(pool_out), exist_ok=True)
    tmp = pool_out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(merged, f, protocol=4)
    os.replace(tmp, pool_out)
    print(f"POOL_DUMP_DONE tokens={len(merged)} out={pool_out}", flush=True)


if __name__ == "__main__":
    main()
