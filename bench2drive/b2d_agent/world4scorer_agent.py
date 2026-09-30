#!/usr/bin/env python3
"""World4Scorer agent for the Bench2Drive leaderboard.

Every 0.5 s (10 ticks at 20 Hz) the model proposes 64 trajectories of 4 s and scores
them; the route point 20 m ahead is part of its ego input. Two deployment rules:
route re-ranking adds a route-agreement bonus to the scores, and command retention
retires a route node only once the ego has passed it. A PID pair tracks the selected
trajectory every tick. Cameras: the Bench2Drive training rig (4 x 1600x900).

Config json: {"ckpt": "<route-point checkpoint>", "save": "<optional dir for overlay frames>"}
"""
import json
import math
import os
import sys
from collections import deque

import numpy as np

_B2D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
for p in (_B2D, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import carla  # noqa
import torch  # noqa
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track  # noqa
import model_config  # noqa
from route_target import RouteCursor, route_from_plan, route_target, TARGET_DIM  # noqa


def get_entry_point():
    return "World4ScorerAgent"


# B2D training rig, CARLA-native (y RIGHT, yaw CW), from anno["sensors"] cam2ego
B2D_CAMS = {  # id -> carla sensor spec; stack order fixed below = training order
    "rgb_front":       dict(x=0.80, y=0.00, z=1.60, yaw=0.0,   fov=70.0),
    "rgb_back":        dict(x=-2.00, y=0.00, z=1.60, yaw=180.0, fov=110.0),
    "rgb_front_left":  dict(x=0.27, y=-0.55, z=1.60, yaw=-55.0, fov=70.0),
    "rgb_front_right": dict(x=0.27, y=0.55, z=1.60, yaw=55.0,  fov=70.0),
}
B2D_ORDER = ["rgb_front", "rgb_back", "rgb_front_left", "rgb_front_right"]
B2D_W, B2D_H = 1600, 900
MODEL_W, MODEL_H = 1148, 672

REPLAN_TICKS = 10          # 0.5 s at the leaderboard's fixed 20 Hz
DT = 0.05
# distance-based look-ahead, scheduled on measured speed, as in the published
# Bench2Drive agents
AIM_NEAR, AIM_FAR, AIM_SWITCH = 4.0, 10.0, 6.5
# route re-ranking: bonus gain (in units of the pool's score spread) and horizon (m)
ROUTE_ALIGN, ROUTE_ALIGN_H = 4.0, 30.0
DESIRED_GAIN, CLIP_DELTA, MAX_THROTTLE = 2.0, 0.25, 0.75
BRAKE_MIN, BRAKE_RATIO = 0.15, 1.1
# creep: after this long stationary, a bounded forward nudge (brake -> frozen scene ->
# same plan -> brake is otherwise a latch)
CREEP_AFTER, CREEP_THROTTLE = 12.0, 0.4

_MODEL_CACHE = {}          # survives per-route re-instantiation (220-route runs)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class PID:
    def __init__(self, kp, ki, kd, n=40):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.win = deque([0.0] * n, maxlen=n)

    def step(self, err):
        self.win.append(err)
        return self.kp * err + self.ki * float(np.mean(self.win)) \
            + self.kd * (self.win[-1] - self.win[-2])


class World4ScorerAgent(AutonomousAgent):
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        # the base class keeps only downsample_route(..., 50); route re-ranking needs
        # the dense polyline to tell a turning candidate from a straight one.
        self._dense_plan_world = list(global_plan_world_coord)
        # route target point fed to the model: same polyline, cursor reset for the
        # new route.
        self._tp_route = None
        self._tp_cursor = RouteCursor()
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        # the leaderboard passes "<cfg>+<tag>" and APPENDS another +<tag> every route --
        # take the last segment only, truncated (Errno 36 after a few routes otherwise)
        segs = path_to_conf_file.split("+")
        cfg = json.load(open(segs[0]))
        if not os.path.isabs(cfg["ckpt"]):  # relative paths are repo-root relative
            cfg["ckpt"] = os.path.join(_REPO, cfg["ckpt"])
        route_tag = (segs[-1] if len(segs) > 1 else "route")[:80]
        self.save_dir = os.path.join(cfg["save"], route_tag) if cfg.get("save") else ""
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        if cfg["ckpt"] not in _MODEL_CACHE:
            _MODEL_CACHE[cfg["ckpt"]] = self._load(cfg["ckpt"])
        self.model = _MODEL_CACHE[cfg["ckpt"]]
        self.dev = "cuda"
        self.tick = 0
        self.plan = None           # (9,3) world math-frame knots @0.5s, [0]=pose at plan time
        self.plan_t0 = 0.0
        self.node_idx = 0
        self._tp_route = None
        self._tp_cursor = RouteCursor()
        self.turn_pid = PID(1.25, 0.75, 0.3)
        self.speed_pid = PID(5.0, 0.5, 1.0)
        self.still_ticks = 0
        self.creep_left = 0
        self.creep_cool = 0

    @staticmethod
    def _load(ck):
        from navsim.agents.drivoR.drivor_model import DrivoRModel
        cfg = model_config.build_b2d_config(_DINO, ego_status_dim=11 + TARGET_DIM)
        m = DrivoRModel(cfg)
        # weights_only=False: the trainer's checkpoints carry an omegaconf cfg blob
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        m.load_state_dict(sd, strict=True)
        m.to("cuda").eval()
        print(f"[agent] checkpoint loaded: {ck}", flush=True)
        return m

    def sensors(self):
        out = []
        for name in B2D_ORDER:
            c = B2D_CAMS[name]
            out.append({"type": "sensor.camera.rgb", "x": c["x"], "y": c["y"],
                        "z": c["z"], "roll": 0.0, "pitch": 0.0, "yaw": c["yaw"],
                        "width": B2D_W, "height": B2D_H, "fov": c["fov"], "id": name})
        return out

    # ------------------------------------------------------------------ helpers
    def _pose(self):
        tf = self.hero_actor.get_transform()
        return np.array([tf.location.x, -tf.location.y, -math.radians(tf.rotation.yaw)])

    def _command(self, pose):
        """RoadOption of the upcoming plan node within 35 m -> int (option - 1).

        Command retention: a node is retired once the ego is past it, not when it is
        merely within 15 m (that retired a single turn node before it was used)."""
        plan = self._global_plan_world_coord
        while self.node_idx < len(plan) - 1:
            loc = plan[self.node_idx][0].location
            dx, dy = loc.x - pose[0], -loc.y - pose[1]
            # heading is +x forward in the ego frame used throughout here
            ahead = dx * math.cos(pose[2]) + dy * math.sin(pose[2])
            if ahead < -2.0 or math.hypot(dx, dy) < 3.0:
                self.node_idx += 1
            else:
                break
        dist = 0.0
        px, py = pose[0], pose[1]
        for j in range(self.node_idx, len(plan)):
            loc = plan[j][0].location
            dist += math.hypot(loc.x - px, -loc.y - py)
            px, py = loc.x, -loc.y
            if dist > 35.0:
                break
            v = plan[j][1].value
            if v in (1, 2, 3):
                return v - 1
        return 3  # LANEFOLLOW

    def _images_tensor(self, input_data):
        arrs = [input_data[n][1][:, :, [2, 1, 0]].copy() for n in B2D_ORDER]
        x = torch.from_numpy(np.stack(arrs)).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(MODEL_H, MODEL_W), mode="bicubic",
                                            align_corners=False, antialias=True)
        x = ((x.unsqueeze(0) - _MEAN.to(self.dev)) / _STD.to(self.dev)).clamp(-5, 5)
        return x  # (1,4,3,672,1148)

    def _replan(self, input_data, timestamp, pose):
        v = self.hero_actor.get_velocity()
        a = self.hero_actor.get_acceleration()
        cmd = self._command(pose)
        ego_state = {"velocity": [v.x, -v.y], "heading": float(pose[2]),
                     "acceleration": [a.x, -a.y], "command": cmd}
        img_t = self._images_tensor(input_data)
        ego_t = model_config.build_ego_status(
            ego_state, cmd, target_pt=self._route_tp(pose)
        ).unsqueeze(0).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            out = self.model({"image": img_t, "ego_status": ego_t})
        props = out["proposals"][0].float().cpu().numpy()
        score = out["pdm_score"][0].float().cpu().numpy()
        if not (np.isfinite(props).all() and np.isfinite(score).all()):
            raise RuntimeError(f"MODEL_OUTPUT_BUG: NaN at tick {self.tick}")
        # route re-ranking: the model never sees the route polyline, and the command is
        # LANEFOLLOW for most of a turn route, so at junctions it can rank the turning
        # candidates last although they are in the pool. The route is available at
        # deployment; add its agreement bonus, scaled by the pool's score spread.
        sfin = np.isfinite(score)
        if sfin.any():
            ssd = float(score[sfin].std())
            if ssd > 1e-6:
                score = score + ROUTE_ALIGN * ssd * self._route_align_bonus(props, pose)
        bi = int(np.argmax(score))
        traj = props[bi]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        R = np.array([[c, -s], [s, c]])
        k = np.zeros((9, 3))
        k[0] = pose
        for i in range(8):
            k[i + 1, :2] = pose[:2] + R @ traj[i, :2]
            k[i + 1, 2] = pose[2] + traj[i, 2]
        self.plan, self.plan_t0 = k, timestamp
        if self.save_dir:   # every replan (0.5 s) — dense enough for demo video
            self._save_overlay(input_data, traj, bi, float(score[bi]), cmd)

    def _route_tp(self, pose):
        """The route target point handed to the model.

        Same polyline the leaderboard interpolates at 1 m (`_dense_plan_world`),
        same route_target() the sidecar calls when it builds the training
        vector, same math frame -- `pose` is already [x, -y, -yaw] here, which
        is the frame b2d_sidecar computes in.
        """
        if self._tp_route is None:
            plan = getattr(self, "_dense_plan_world", None) or self._global_plan_world_coord
            self._tp_route = route_from_plan(plan)
            self._tp_cursor = RouteCursor()
        fwd, lat, _ = route_target(self._tp_route, self._tp_cursor,
                                   float(pose[0]), float(pose[1]), float(pose[2]))
        return (fwd, lat)

    def _advance_cursor(self, pose, plan):
        """Nearest node to the ego, searched forward from a monotone cursor so a
        loop in the route cannot snap us back to an earlier leg."""
        i0 = getattr(self, "_ra_cursor", 0)
        best, bestd = i0, float("inf")
        for j in range(i0, min(i0 + 400, len(plan))):
            loc = plan[j][0].location
            d = math.hypot(loc.x - pose[0], -loc.y - pose[1])
            if d < bestd:
                bestd, best = d, j
        self._ra_cursor = best
        return best

    def _route_ahead(self, pose, horizon):
        """Dense route polyline from the ego forward `horizon` m, ego frame."""
        plan = getattr(self, "_dense_plan_world", None) or self._global_plan_world_coord
        best = self._advance_cursor(pose, plan)
        pts, dist, px, py = [], 0.0, pose[0], pose[1]
        c, s = math.cos(-pose[2]), math.sin(-pose[2])
        for j in range(best, len(plan)):
            loc = plan[j][0].location
            wx, wy = loc.x, -loc.y
            dist += math.hypot(wx - px, wy - py)
            px, py = wx, wy
            dx, dy = wx - pose[0], wy - pose[1]
            pts.append((dx * c - dy * s, dx * s + dy * c))
            if dist > horizon:
                break
        return pts

    def _route_align_bonus(self, props, pose):
        """Per-candidate agreement with the route ahead, standardised in the pool.

        Distance from each candidate waypoint to the nearest route point, mapped
        through a soft kernel and standardised across the pool so the term is
        scale-free and cannot swamp the model score on straight roads (where
        every candidate agrees with the route and the spread collapses).
        """
        pts = self._route_ahead(pose, ROUTE_ALIGN_H)
        if len(pts) < 2:
            return np.zeros(len(props), dtype=np.float32)
        R = np.asarray(pts, dtype=np.float32)
        w = np.asarray(props[:, :, :2], dtype=np.float32)          # (N, T, 2)
        d = np.linalg.norm(w[:, :, None, :] - R[None, None, :, :], axis=3).min(axis=2)
        tw = np.linspace(0.5, 1.5, d.shape[1], dtype=np.float32)   # later points weigh more
        b = np.exp(-(d * tw).sum(axis=1) / tw.sum() / 4.0)
        sd = float(b.std())
        if sd < 1e-3:
            return np.zeros(len(props), dtype=np.float32)
        return ((b - b.mean()) / sd).astype(np.float32)

    def _plan_at_dist(self, tau, dist, pose):
        """First point on the plan at or beyond `dist` metres from the ego,
        walked forward from tau. Falls back to the plan's end, which bounds the
        look-ahead by the 4 s horizon rather than letting it run off."""
        t = tau
        while t < 3.99:
            p = self._plan_at(t)
            if math.hypot(p[0] - pose[0], p[1] - pose[1]) >= dist:
                return p
            t += 0.05
        return self._plan_at(3.99)

    def _plan_at(self, tau):
        tau = max(0.0, min(tau, 3.99))
        seg = min(int(tau / 0.5), 7)
        f = (tau - seg * 0.5) / 0.5
        p0, p1 = self.plan[seg], self.plan[seg + 1]
        return p0[:2] + (p1[:2] - p0[:2]) * f

    def _track(self, timestamp, pose):
        ctrl = carla.VehicleControl()
        if self.plan is None:
            ctrl.brake = 1.0
            return ctrl
        tau = timestamp - self.plan_t0
        v = self.hero_actor.get_velocity()
        speed = math.hypot(v.x, v.y)
        p_now, p_next = self._plan_at(tau), self._plan_at(tau + 0.5)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        # forward component only: a lateral offset is not speed. Using the 2-D
        # magnitude let a standing-still candidate read as ~1 m/s.
        _dp = p_next - p_now
        desired = max(0.0, c * _dp[0] + s * _dp[1]) * DESIRED_GAIN
        aim_d = AIM_FAR if speed > AIM_SWITCH else AIM_NEAR
        aim = self._plan_at_dist(tau, aim_d, pose)
        d = aim - pose[:2]
        fwd, lat = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        if fwd < 0.5:
            # aim point is beside or behind us: atan2 against a 0.05 m floor
            # yields +/-84 deg and locks the wheel. Hold straight instead.
            angle = 0.0
        else:
            angle = math.degrees(math.atan2(lat, fwd)) / 90.0
        steer = float(np.clip(-self.turn_pid.step(angle), -1.0, 1.0))
        brake = desired < BRAKE_MIN or (speed / max(desired, 0.1)) > BRAKE_RATIO
        delta = float(np.clip(desired - speed, 0.0, CLIP_DELTA))
        throttle = float(np.clip(self.speed_pid.step(delta), 0.0, MAX_THROTTLE))
        ctrl.steer = steer
        ctrl.throttle = 0.0 if brake else throttle
        ctrl.brake = 1.0 if brake else 0.0
        # creep: model wants stop for too long -> nudge forward, bounded + cooldown
        self.still_ticks = self.still_ticks + 1 if speed < 0.5 else 0
        if self.creep_left > 0:
            ctrl.throttle, ctrl.brake = CREEP_THROTTLE, 0.0
            self.creep_left -= 1
            if self.creep_left == 0:
                self.creep_cool = 40
        elif self.creep_cool > 0:
            self.creep_cool -= 1
        elif self.still_ticks > CREEP_AFTER / DT:
            self.creep_left = 60
        return ctrl

    def _save_overlay(self, input_data, traj, bi, sc, cmd):
        import cv2
        fid, fx, cx, cy = "rgb_front", 1142.518, 800.0, 450.0
        cam = dict(x=0.80, y=0.0, z=1.60, yaw=0.0)
        f = input_data[fid][1][:, :, :3].copy()  # BGRA -> BGR
        cyw, syw = math.cos(math.radians(cam["yaw"])), math.sin(math.radians(cam["yaw"]))
        pts = []
        for i in range(8):
            lx, ly = traj[i, 0] - cam["x"], traj[i, 1] - cam["y"]
            xc, yc = cyw * lx + syw * ly, -syw * lx + cyw * ly
            if xc < 1.0:
                continue
            pts.append((int(cx - fx * yc / xc), int(cy + fx * (cam["z"] - 0.2) / xc)))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(f, p1, p2, (0, 255, 0), 3)
        vv = self.hero_actor.get_velocity()
        cv2.putText(f, f"t={self.tick * DT:6.1f}s v={math.hypot(vv.x, vv.y) * 3.6:4.1f}km/h "
                       f"cmd={cmd} cand#{bi} s={sc:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(f"{self.save_dir}/{self.tick:06d}.jpg",
                    cv2.resize(f, (800, 450)), [cv2.IMWRITE_JPEG_QUALITY, 80])

    # ------------------------------------------------------------------ main
    def run_step(self, input_data, timestamp):
        pose = self._pose()
        if self.tick % REPLAN_TICKS == 0 or self.plan is None:
            self._replan(input_data, timestamp, pose)
        ctrl = self._track(timestamp, pose)
        self.tick += 1
        return ctrl

    def destroy(self):
        self.plan = None
