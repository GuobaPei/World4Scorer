"""The outcome head: candidate features, the head, the blend with the released cost,
and the training objective."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def compose_candidate_features(
    terminal_latent: torch.Tensor,
    goal_latent: torch.Tensor,
    base_cost: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Per-candidate input: predicted terminal latent, goal latent, their difference and
    product, the released cost, and the mean and std of the candidate's actions."""
    latent_dim = terminal_latent.shape[-1]
    terminal = F.layer_norm(terminal_latent.float(), (latent_dim,))
    goal = F.layer_norm(goal_latent.float(), (latent_dim,))
    difference = terminal - goal
    interaction = terminal * goal
    base_signal = torch.log1p(base_cost.float().clamp_min(0)).unsqueeze(-1)
    action_features = torch.cat(
        [
            candidates.float().mean(dim=-2),
            candidates.float().std(dim=-2, unbiased=False),
        ],
        dim=-1,
    )
    return torch.cat(
        [terminal, goal, difference, interaction, base_signal, action_features],
        dim=-1,
    )


class OutcomeHead(nn.Module):
    def __init__(self, input_dim: int, *, state_dim: int = 128, hidden_dim: int = 256,
                 dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(state_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.utility_head(self.state_encoder(features)).squeeze(-1)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


def standardize_within_candidates(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=-1, keepdim=True)
    scale = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (values - mean) / scale


def combined_utility_score(base_cost: torch.Tensor, predicted_utility: torch.Tensor,
                           coefficient: float) -> torch.Tensor:
    """standardised(-released cost) + coefficient * standardised(head)."""
    return (standardize_within_candidates(-base_cost.float())
            + coefficient * standardize_within_candidates(predicted_utility.float()))


def objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    regression = F.smooth_l1_loss(prediction, target)
    target_prob = F.softmax(target / 0.5, dim=-1)
    listwise = -(target_prob * F.log_softmax(prediction / 0.5, dim=-1)).sum(-1).mean()
    return regression + 0.25 * listwise


@torch.no_grad()
def pairwise_accuracy(score: torch.Tensor, utility: torch.Tensor) -> float:
    score_diff = score.unsqueeze(-1) - score.unsqueeze(-2)
    utility_diff = utility.unsqueeze(-1) - utility.unsqueeze(-2)
    mask = torch.triu(torch.ones_like(score_diff, dtype=torch.bool), diagonal=1)
    mask &= utility_diff.abs() > 1e-7
    if not torch.any(mask):
        return float("nan")
    correct = (score_diff[mask] * utility_diff[mask]) > 0
    return float(correct.float().mean())
