"""The outcome head inside the LeWM planning loop.

Candidates are ranked by standardised(-released cost) + c * standardised(head), with c
chosen in closed loop on tuning seeds (select_c.py). Nothing else changes: same solver,
same CEM iteration count and population, same MPC schedule, same evaluation episodes,
same success rule.
"""

from __future__ import annotations

from pathlib import Path

import torch

from head_model import OutcomeHead, combined_utility_score, compose_candidate_features


class HeadScorer:
    def __init__(self, cfg, model, world, process):
        self.cfg = cfg
        self.world = world
        self.process = process
        self.n_calls = 0
        ck = torch.load(Path(cfg.cost.head_path, "head.pt"), map_location="cpu", weights_only=False)
        c = ck["config"]
        self.head = OutcomeHead(c["input_dim"], state_dim=c["state_dim"],
                                hidden_dim=c["hidden_dim"], dropout=c["dropout"]).eval()
        self.head.load_state_dict(ck["state_dict"])
        self.head = self.head.to("cuda")
        self.coefficient = float(cfg.cost.coefficient)
        self.head_path = str(cfg.cost.head_path)

    def attach(self, model):
        self.base = model.get_cost
        model.get_cost = self.get_cost

    def attach_policy(self, policy):
        self.policy = policy

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        self.n_calls += 1
        base_cost = self.base(info_dict, action_candidates)  # (B,S)
        term = info_dict["predicted_emb"][:, :, -1, :].float()
        goal = info_dict["goal_emb"].float()[:, -1, :]
        goal = goal.unsqueeze(1).expand(-1, term.shape[1], -1)
        feats = compose_candidate_features(term, goal, base_cost, action_candidates.float())
        utility = self.head(feats)
        score = combined_utility_score(base_cost, utility, self.coefficient)
        return (-score).to(dtype=action_candidates.dtype)

    def summary(self):
        return {
            "scorer_calls": self.n_calls,
            "head_path": self.head_path,
            "coefficient": self.coefficient,
        }

    def finalize(self, out_dir):
        pass
