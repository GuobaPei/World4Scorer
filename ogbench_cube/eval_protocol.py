"""LeWM protocol eval with a pluggable candidate scorer.

This is repo/eval.py with two additions and no other change:
  1. ``cost.mode`` selects which scorer the CEM loop calls
     (``lewm`` = the released hand-written cost, i.e. the anchor).
  2. per-episode successes are written to a json next to the log so two
     scorers can be compared pairwise on the same episodes.

Everything the protocol touches -- solver, CEM iterations/population, MPC
receding horizon, the eval episode set, the success rule -- comes from the
untouched config files under repo/config/eval.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


@hydra.main(version_base=None, config_path="./conf", config_name="cube")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy_name = cfg.get("policy", "random")

    scorer = None
    if policy_name != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        # ---- scorer swap (the only thing that differs between runs) ----
        from scorers import build_scorer

        scorer = build_scorer(cfg, model=model, world=world, process=process)
        scorer.attach(model)
        # ---------------------------------------------------------------

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
        scorer.attach_policy(policy)
    else:
        policy = swm.policy.RandomPolicy()

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.", flush=True)

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=None,
    )
    end_time = time.time()

    print(metrics, flush=True)

    succ = np.asarray(metrics["episode_successes"], dtype=bool)
    payload = {
        "tag": cfg.tag,
        "cost_mode": cfg.cost.mode,
        "env": cfg.eval.dataset_name,
        "seed": int(cfg.seed),
        "num_eval": int(cfg.eval.num_eval),
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": succ.astype(int).tolist(),
        "episodes_idx": [int(x) for x in eval_episodes.tolist()],
        "start_steps": [int(x) for x in eval_start_idx.tolist()],
        "plan_config": OmegaConf.to_container(cfg.plan_config, resolve=True),
        "solver": OmegaConf.to_container(cfg.solver, resolve=True),
        "eval_budget": int(cfg.eval.eval_budget),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "wall_seconds": end_time - start_time,
    }
    if scorer is not None:
        payload.update(scorer.summary())
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    if scorer is not None:
        scorer.finalize(out_dir)
    print("WROTE", out_dir / "result.json", flush=True)
    print("FINAL success_rate=%.2f" % metrics["success_rate"], flush=True)


if __name__ == "__main__":
    run()
