"""The World4Scorer scoring trunk.

Every candidate trajectory token cross-attends the scene features and becomes a
predicted state; the score heads read the candidate's sub-scores from that state.
During training, `realized_head` reads out the state of the executed (logged)
trajectory and matches it to the frozen DINOv2 feature of the front camera 2 s ahead
(DrivoRModel.realized_future_loss).
"""

import torch
import torch.nn as nn

from .transformer_decoder import TransformerDecoderScorer


class SharedFuturePredictor(nn.Module):
    """forward(candidate_tokens [B, N, D], scene_features [B, M, D]) -> states [B, N, D]"""

    def __init__(self, config):
        super().__init__()
        d_model = config.tf_d_model
        # self-attention across candidates, cross-attention to the scene, FFN
        self.predictor = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num,
            d_model=d_model,
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.realized_head = nn.Linear(d_model, int(config.get("realized_emb_dim", 384)))

    def forward(self, proposal_tokens: torch.Tensor, scene_features: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.predictor(proposal_tokens, scene_features))
