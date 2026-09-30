from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder
from .shared_future_predictor import SharedFuturePredictor
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)


# future targets: navtrain token -> frozen DINOv2 feature of the front camera 2 s ahead
# (tools/precompute_future_embeddings.py)
_REALIZED_BANK = None

def _load_realized_bank(config):
    global _REALIZED_BANK
    if _REALIZED_BANK is None:
        path = config.get("realized_future_emb_path", "")
        assert path, "realized_future_emb_path is empty"
        _REALIZED_BANK = torch.load(path, map_location="cpu")
        log.info(f"realized-future bank loaded: {len(_REALIZED_BANK)} tokens from {path}")
    return _REALIZED_BANK


# candidate bank: navtrain token -> 64 candidates + precomputed PDM sub-scores
# (tools/build_bank_pack.py), memory-mapped
_CLOVER_PACK = None

def _load_clover_pack(config):
    global _CLOVER_PACK
    if _CLOVER_PACK is None:
        base = config.get("clover_pack_path", "")
        assert base, "clover_pack_path is empty"
        toks = np.load(f"{base}/tokens.npy")
        idx = {(t.decode() if isinstance(t, bytes) else str(t)): i for i, t in enumerate(toks)}
        _CLOVER_PACK = dict(
            idx=idx,
            cands=np.load(f"{base}/bank_candidates.npy", mmap_mode="r"),
            subs=np.load(f"{base}/bank_subscores.npy", mmap_mode="r"),
            mask=np.load(f"{base}/bank_mask.npy", mmap_mode="r"),
        )
        log.info(f"candidate bank loaded: {len(idx)} tokens from {base}")
    return _CLOVER_PACK

class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder. 11 = [pose(3), velocity(2), acceleration(2),
        # driving_command one-hot(4)]; the Bench2Drive route-point model appends the
        # route target point (13, bench2drive/route_target.py).
        ego_dim = int(config.get("ego_status_dim", 11))
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(ego_dim*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(ego_dim, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        # one trajectory head, on the last decoder layer: only the final proposals are
        # scored and executed
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size)])

        # scorer
        self.scorer = Scorer(config)

        # scoring trunk: one shared predictor gives every candidate a state
        self.future_predictor = SharedFuturePredictor(config)

        self.b2d=config.b2d


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]



        scene_features = []
        # image features
        if self.num_cams > 0:
            
            if "image" in features :
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")

        # decode the trajectories; the head reads the last decoder layer
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        tokens = token_list[-1]
        proposals = self.traj_head[-1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]

        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]
        

        output={}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        # scoring
        B,N,_,_=proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        # each candidate's predicted state; the score heads read the sub-scores from it
        tr_out = self.future_predictor(embedded_traj, scene_features)  # (B, N, d_model)
        if self.training:
            # kept for the two auxiliary losses of the step (drivor_agent.compute_loss)
            self._realized_scene_feats = scene_features
            self._clover_ctx = (scene_features, ego_token)

        tr_out = tr_out+ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        pdm_score = (
        self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
        self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
        self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +    
        (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
        self._config.ep * pred_logit['ego_progress'].sigmoid()  
        + self._config.comfort * pred_logit['comfort'].sigmoid()).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output

    def realized_future_loss(self, targets):
        """Future loss: 1 - cos between the readout of the executed (logged)
        trajectory's predicted state and the frozen DINOv2 feature of the front camera
        2 s ahead. Scene features are detached; realized_lambda scales the gradient
        into the shared predictor. None outside training."""
        sf = getattr(self, "_realized_scene_feats", None)
        if sf is None:
            return None
        self._realized_scene_feats = None
        bank = _load_realized_bank(self._config)
        gt = targets["trajectory"].to(sf.device).float()
        B, P = gt.shape[0], self.poses_num
        if gt.shape[1] < P:
            gt = torch.cat([gt, gt[:, -1:, :].expand(B, P - gt.shape[1], gt.shape[2])], 1)
        elif gt.shape[1] > P:
            gt = gt[:, :P]
        q = self.pos_embed(gt.reshape(B, 1, -1))
        lat = self.future_predictor(q, sf.detach())
        lam = float(self._config.get("realized_lambda", 1.0))
        lat = lam * lat + (1.0 - lam) * lat.detach()
        pred_emb = self.future_predictor.realized_head(lat[:, 0])
        tgt, mask = [], []
        E = pred_emb.shape[-1]
        for t in targets["token"]:
            v = bank.get(t if isinstance(t, str) else str(t))
            if v is None:
                tgt.append(torch.zeros(E)); mask.append(0.0)
            else:
                tgt.append(v.float()); mask.append(1.0)
        tgt = torch.stack(tgt).to(pred_emb.device)
        mask = torch.tensor(mask, device=pred_emb.device)
        cos = torch.nn.functional.cosine_similarity(pred_emb, tgt, dim=-1)
        return ((1.0 - cos) * mask).sum() / mask.sum().clamp(min=1.0)

    def clover_aux_loss(self, targets):
        """Bank loss: pointwise BCE/L1 on clover_k bank candidates per scene against
        their precomputed PDM sub-scores, scored by the same predictor and heads as the
        model's own candidates. Columns: [noc, dac, ep, ttc, comfort, ddc, pdm].
        None outside training."""
        ctx = getattr(self, "_clover_ctx", None)
        if ctx is None:
            return None
        self._clover_ctx = None
        sf, ego = ctx
        pack = _load_clover_pack(self._config)
        k = int(self._config.get("clover_k", 16))
        sfrac = float(self._config.get("clover_safety_frac", 0.5))
        B = sf.shape[0]
        cands_l, labs_l, rowmask = [], [], []
        for t in targets["token"]:
            tok = t if isinstance(t, str) else str(t)
            ri = pack["idx"].get(tok)
            if ri is None:
                rowmask.append(0); cands_l.append(np.zeros((k, 8, 3), np.float32)); labs_l.append(np.zeros((k, 7), np.float32)); continue
            valid = np.where(np.array(pack["mask"][ri]))[0]
            if len(valid) == 0:
                rowmask.append(0); cands_l.append(np.zeros((k, 8, 3), np.float32)); labs_l.append(np.zeros((k, 7), np.float32)); continue
            subs = np.array(pack["subs"][ri, valid], np.float32)
            unsafe = valid[(subs[:, 0] < 0.5) | (subs[:, 1] < 0.5)]
            ns = min(int(k * sfrac), len(unsafe))
            pick_s = np.random.choice(unsafe, ns, replace=False) if ns > 0 else np.array([], np.int64)
            rest = np.setdiff1d(valid, pick_s)
            nr = min(k - ns, len(rest))
            pick_r = np.random.choice(rest, nr, replace=False) if nr > 0 else np.array([], np.int64)
            pick = np.concatenate([pick_s, pick_r]).astype(np.int64)
            if len(pick) < k:
                pick = np.concatenate([pick, np.random.choice(valid, k - len(pick), replace=True)])
            cands_l.append(np.array(pack["cands"][ri, pick], np.float32))
            labs_l.append(np.array(pack["subs"][ri, pick], np.float32))
            rowmask.append(1)
        rm = torch.tensor(rowmask, device=sf.device, dtype=torch.float32)
        if rm.sum() == 0:
            return None
        c = torch.from_numpy(np.stack(cands_l)).to(sf.device)          # [B,k,8,3]
        # pad 8 -> poses_num by constant-velocity extrapolation of the last segment
        P = self.poses_num
        if c.shape[2] < P:
            step = c[:, :, -1:, :] - c[:, :, -2:-1, :]
            extra = c[:, :, -1:, :] + step * torch.arange(1, P - c.shape[2] + 1, device=c.device, dtype=c.dtype).view(1, 1, -1, 1)
            c = torch.cat([c, extra], dim=2)
        lab = torch.from_numpy(np.stack(labs_l)).to(sf.device)          # [B,k,7]
        # bank coordinates are detached; the loss reaches the predictor and the heads
        emb = self.pos_embed(c.reshape(B, c.shape[1], -1).detach())
        tr = self.future_predictor(emb, sf)
        tr = tr + ego
        pl = self.scorer(c, tr)[0]
        from .layers.losses.drivor_loss import three_to_two_classes
        m = rm.view(B, 1)
        def bce(logit, y):
            per = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
            return (per * m).sum() / (m.sum() * per.shape[1])
        noc_y = three_to_two_classes(lab[..., 0].clone()); ddc_y = three_to_two_classes(lab[..., 5].clone())
        loss = (bce(pl["no_at_fault_collisions"], noc_y)
                + bce(pl["drivable_area_compliance"], lab[..., 1])
                + bce(pl["time_to_collision_within_bound"], lab[..., 3])
                + bce(pl["driving_direction_compliance"], ddc_y)
                + bce(pl["comfort"], lab[..., 4])) / 5.0
        ep_l1 = (torch.abs(torch.sigmoid(pl["ego_progress"]) - lab[..., 2]) * m).sum() / (m.sum() * lab.shape[1])
        loss = loss + 0.1 * ep_l1
        return loss
