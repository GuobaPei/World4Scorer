"""World4Scorer model configuration and ego-status input for Bench2Drive.

build_b2d_config() is the configuration of the Bench2Drive models (the NAVSIM
architecture with its 4-camera, 1148x672 input). build_ego_status() builds the ego
vector exactly as the NAVSIM feature builder does, optionally with the route target
point appended.
"""
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf


def build_b2d_config(dino_weights_path: str, realized_future_emb_path: str = "",
                     ego_status_dim: int = 11) -> OmegaConf:
    """ego_status_dim 11 = route-blind model, 13 = route-point model."""
    cfg = {
        # ---- sensor channels (4-cam, no lidar) ----
        "cam_f0": [3],
        "cam_l0": [3],
        "cam_l1": [],
        "cam_l2": [],
        "cam_r0": [3],
        "cam_r1": [],
        "cam_r2": [],
        "cam_b0": [3],
        "lidar_pc": [],

        # ---- image ----
        "image_size": [1148, 672],   # (width, height) — must match ckpt training size
        "lidar_image_size": [518, 518],

        # ---- model arch ----
        "num_scene_tokens": 16,
        "tf_d_model": 256,
        "tf_d_ffn": 1024,
        "num_poses": 8,
        "proposal_num": 64,
        "ref_num": 4,
        "scorer_ref_num": 4,

        # ---- trajectory flags ----
        "one_token_per_traj": True,
        "full_history_status": False,
        "long_trajectory_additional_poses": 2,   # training-only target; does not touch proposals

        # ---- image backbone ----
        "image_backbone": {
            "model_name": "timm/vit_small_patch14_reg4_dinov2.lvd142m",
            "model_weights": dino_weights_path,
            "use_lora": True,
            "lora_rank": 32,
            "finetune": False,
            "use_feature_pooling": False,
            "focus_front_cam": False,
            "compress_fc": False,
        },
        "lidar_backbone": {
            "model_name": "timm/vit_small_patch14_reg4_dinov2.lvd142m",
            "model_weights": dino_weights_path,
            "use_lora": False,
            "lora_rank": 0,
            "finetune": False,
            "use_feature_pooling": False,
            "focus_front_cam": False,
            "compress_fc": False,
        },

        # ---- lidar geometry (unused but required by config schema) ----
        "lidar_min_x": -35, "lidar_max_x": 35,
        "lidar_min_y": -35, "lidar_max_y": 35,
        "lidar_max_height": 100.0, "lidar_split_height": 0.2,
        "lidar_use_ground_plane": True, "lidar_hist_max_per_pixel": 5,

        # ---- selection weights ----
        "noc": 1.0, "dac": 1.0, "ddc": 0.0,
        "ttc": 5.0, "ep": 5.0, "comfort": 2.0,

        # ---- scorer aux (all off) ----
        "double_score": False,
        "agent_pred": False,
        "area_pred": False,
        "bev_map": False,
        "bev_agent": False,
        "refiner_num_heads": 1,
        "refiner_ls_values": 0.0,

        # ---- DrivoR flags (not used in forward) ----
        "b2d": False,

        # ---- future loss (training only): weight and gradient scale into the predictor ----
        "realized_future_emb_path": realized_future_emb_path,
        "realized_emb_dim": 384,
        "realized_weight": 0.1,
        "realized_lambda": 0.05,
        "ego_status_dim": int(ego_status_dim),

        # ---- trajectory sampling ----
        "trajectory_sampling": {
            "num_poses": 8,
            "time_horizon": 4.0,
            "interval_length": 0.5,
        },
    }
    return OmegaConf.create(cfg)


def build_ego_status(ego_state: Dict[str, Any], command: int,
                     target_pt=None) -> torch.Tensor:
    """
    Build the ego status vector that matches native DrivoRFeatureBuilder output.

    Native format (drivor_features.py line 55-63):
      [ego_pose(3), ego_velocity(2), ego_acceleration(2), driving_command(4)] = 11 dims
    The model's forward takes features["ego_status"][:, -1] (last timestep).

    `target_pt` appends the route target point, (forward, lateral) in metres in
    the ego frame, lateral positive left, unnormalised -- 13 dims. It is the one
    piece of route geometry the model gets; see route_target.py for the single
    definition both the sidecar and the agent compute it with. With None the vector
    has the 11 dims of the route-blind model.
    """
    # pose: (0,0,0) — current frame is always origin in ego-centric rep
    pose = np.zeros(3, dtype=np.float32)

    velocity = np.asarray(ego_state.get("velocity", [0.0, 0.0]), dtype=np.float32)[:2]
    heading  = float(ego_state.get("heading", 0.0))
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, s], [-s, c]], dtype=np.float32)
    vel_local = R @ velocity

    if "acceleration" in ego_state:
        acc = np.asarray(ego_state["acceleration"], dtype=np.float32)[:2]
        acc_local = R @ acc
    else:
        acc_local = np.zeros(2, dtype=np.float32)

    # Command one-hot (matches NAVSIM_CMD_MAPPING convention)
    cmd_vec = _command_to_onehot(command)

    parts = [pose, vel_local, acc_local, cmd_vec]
    if target_pt is not None:
        tp = np.asarray(target_pt, dtype=np.float32).reshape(-1)[:2]
        assert tp.shape == (2,), f"target_pt must be (fwd, lat), got {target_pt!r}"
        parts.append(tp)

    ego_status = np.concatenate(parts).astype(np.float32)
    return torch.from_numpy(ego_status)


def _command_to_onehot(command: int) -> np.ndarray:
    """Map the integer command (CARLA road option - 1) to the NAVSIM 4-dim one-hot.

        road option - 1:       0=LEFT 1=RIGHT 2=STRAIGHT 3=LANEFOLLOW
        NAVSIM one-hot slot:   0=LEFT 1=STRAIGHT 2=RIGHT 3=UNKNOWN
    LANEFOLLOW carries no turn information, which in the NAVSIM taxonomy is
    STRAIGHT (slot 1); slot 3 has no training support.
    """
    mapping = {
        0: [1, 0, 0, 0],   # LEFT       -> LEFT      (slot 0)
        1: [0, 0, 1, 0],   # RIGHT      -> RIGHT     (slot 2)
        2: [0, 1, 0, 0],   # STRAIGHT   -> STRAIGHT  (slot 1)
        3: [0, 1, 0, 0],   # LANEFOLLOW -> STRAIGHT  (slot 1)
    }
    onehot = np.array(mapping.get(command, [0, 1, 0, 0]), dtype=np.float32)
    assert onehot[3] == 0.0, "native driving_command index 3 has no training support"
    return onehot
