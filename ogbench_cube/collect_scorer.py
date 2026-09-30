"""Collect in-loop training data for the utility head.

Planning is driven by the *released* cost, so the candidates we record are the
ones the deployed planner actually scores -- including the way the CEM
population narrows over its iterations.  Each candidate additionally gets a
ground-truth label from a real-simulator replay: on-distribution inputs, true
outcomes.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from sim_worker import snapshot, worker_init, worker_rollout


class CollectScorer:
    def __init__(self, cfg, model, world, process):
        self.cfg = cfg
        self.model = model
        self.world = world
        self.process = process
        self.n_calls = 0
        self.pool = None
        self.workers = int(cfg.cost.get("workers", 16))
        self.buf = []
        self._iter_of_call = {}

    def attach(self, model):
        self.base = model.get_cost  # released cost: feature + reference
        model.get_cost = self.get_cost

    def attach_policy(self, policy):
        self.policy = policy
        base = policy.get_action
        n_envs = self.world.num_envs

        def tagged(info_dict, **kw):
            info_dict = dict(info_dict)
            info_dict["_env_id"] = np.arange(n_envs, dtype=np.int64)[:, None]
            return base(info_dict, **kw)

        policy.get_action = tagged

    def _ensure_pool(self):
        if self.pool is None:
            ctx = mp.get_context("spawn")
            kw = OmegaConf.to_container(self.cfg.world, resolve=True)
            kw.pop("num_envs", None)
            self.pool = ctx.Pool(self.workers, initializer=worker_init, initargs=(kw,))

    def get_cost(self, info_dict, action_candidates):
        self._ensure_pool()
        self.n_calls += 1

        released = self.base(info_dict, action_candidates)  # fills predicted_emb; drives planning

        env_ids = info_dict["_env_id"]
        env_ids = env_ids.detach().cpu().numpy() if torch.is_tensor(env_ids) else np.asarray(env_ids)
        env_ids = env_ids.reshape(env_ids.shape[0], -1)[:, 0].astype(int)

        # CEM iteration index: consecutive calls on the same env within one solve
        key = tuple(env_ids.tolist())
        self._iter_of_call[key] = self._iter_of_call.get(key, -1) + 1
        it = self._iter_of_call[key]
        if it >= int(self.cfg.solver.n_steps):
            self._iter_of_call[key] = it = 0

        pred = info_dict["predicted_emb"]  # (B,S,T,D)
        term = pred[:, :, -1, :].float()
        goal = info_dict["goal_emb"].float()  # (B,1,D)
        goal = goal[:, -1, :]  # (B,D)

        cand = action_candidates.detach().cpu().numpy()
        B, S = cand.shape[0], cand.shape[1]
        act_dim = self.policy.env.single_action_space.shape[-1]
        steps = (cand.shape[2] * cand.shape[3]) // act_dim
        cand_r = cand.reshape(B, S, steps, act_dim)
        raw = self.process["action"].inverse_transform(
            cand_r.reshape(-1, act_dim).astype(np.float64)
        ).reshape(B, S, steps, act_dim)

        for b in range(B):
            env = self.world.envs.envs[int(env_ids[b])].unwrapped
            snap = snapshot(env)
            target = np.array(env._data.mocap_pos[env._cube_target_mocap_ids[0]], copy=True)
            chunks = [c for c in np.array_split(np.arange(S), self.workers) if len(c)]
            task_cost = np.concatenate(self.pool.starmap(
                worker_rollout, [(snap, target, raw[b][idx]) for idx in chunks]))
            self.buf.append(
                dict(
                    call=self.n_calls,
                    env=int(env_ids[b]),
                    cem_iter=it,
                    terminal=term[b].half().cpu(),
                    goal=goal[b].half().cpu(),
                    base_cost=released[b].float().cpu(),
                    actions=torch.from_numpy(cand[b].astype(np.float16)),
                    task_cost=torch.from_numpy(task_cost.astype(np.float32)),
                )
            )

        return released

    def summary(self):
        return {"scorer_calls": self.n_calls, "collected_groups": len(self.buf)}

    def finalize(self, out_dir):
        torch.save(self.buf, Path(out_dir, "bank.pt"))
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
