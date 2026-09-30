"""b2d_dataset.py — Bench2Drive-Base -> DrivoR features/targets.

Frames @10Hz, 8 future poses @0.5s (=frames t+5..t+40). Sample stride 5.
Coordinates: B2D anno x/y/theta are CARLA world (y-right). Math frame = y-left:
y_m=-y, th_m=-theta. Ego-frame future = R(-th_t) @ (p-p_t), yaw wrapped.
"""
import gzip, json, os, sys, glob, math
_B2D = os.path.dirname(os.path.abspath(__file__))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DATA = os.environ.get("B2D_DATA", os.path.join(_B2D, "data"))
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, _B2D)
from model_config import build_ego_status, CAM_ORDER_4, MODEL_IMAGE_SIZE_WH, PIL_BICUBIC
from route_target import RouteCursor, route_target
import PIL.Image as PImage


def preprocess_images_u8(images_dict):
    """Resize-only uint8 images (9 MB/sample instead of 37 MB as float32).
    Normalization happens on GPU in the trainer."""
    w, h = MODEL_IMAGE_SIZE_WH
    out = []
    for cam in CAM_ORDER_4:
        im = PImage.fromarray(images_dict[cam]).resize((w, h), resample=PIL_BICUBIC)
        out.append(np.asarray(im, np.uint8).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out))          # (4,3,H,W) uint8

CAM_MAP = {  # DrivoR slot -> B2D camera dir
    "CAM_F0": "rgb_front", "CAM_B0": "rgb_back",
    "CAM_L0": "rgb_front_left", "CAM_R0": "rgb_front_right",
}
FUT_STRIDE, N_POSES, SAMPLE_STRIDE = 5, 8, 5


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ego_xy(anno):
    """Ego position in math frame, from the ego bounding-box centre.

    anno["x"],["y"] is a noisy GNSS read (path length 2.79x the speed
    integral, implied accel 64 m/s^2). The ego box centre matches the speed
    integral to 1.000x at 2.19 m/s^2. Falls back to the GNSS read only if the
    frame carries no ego box.
    """
    for b in anno.get("bounding_boxes", ()):
        if b.get("class") == "ego_vehicle":
            c = b["center"]
            return float(c[0]), -float(c[1])
    return float(anno["x"]), -float(anno["y"])


def load_anno(p):
    with gzip.open(p) as f:
        return json.load(f)


def build_index(data_root, out_jsonl):
    """Scan extracted clips -> one line per usable sample (needs t+40 future)."""
    n = 0
    with open(out_jsonl, "w") as w:
        for clip in sorted(os.listdir(data_root)):
            adir = os.path.join(data_root, clip, "anno")
            if not os.path.isdir(adir):
                continue
            frames = sorted(int(f[:5]) for f in os.listdir(adir) if f.endswith(".json.gz"))
            if not frames:
                continue
            last = frames[-1]
            for t in range(frames[0], last - FUT_STRIDE * N_POSES + 1, SAMPLE_STRIDE):
                w.write(json.dumps({"clip": clip, "t": t}) + "\n")
                n += 1
    print(f"[index] {n} samples -> {out_jsonl}")
    return n


class B2DDataset(Dataset):
    def __init__(self, data_root, index_jsonl, val=False, val_every=33, routes_npz=""):
        self.root = data_root
        rows = [json.loads(l) for l in open(index_jsonl)]
        clips = sorted({r["clip"] for r in rows})
        val_clips = set(clips[::val_every])
        keep = val_clips if val else (set(clips) - val_clips)
        self.rows = [r for r in rows if r["clip"] in keep]
        self.val = val
        # routes_npz (b2d_route_cache.py) switches the ego vector from 11 to 13
        # dims by appending the route target point. Empty = the 11-dim vector
        # this dataset has always produced.
        self.routes = {}
        if routes_npz:
            from b2d_route_cache import load_cache
            self.routes = load_cache(routes_npz)
        self._cursor_clip, self._cursor = None, None

    def __len__(self):
        return len(self.rows)

    def _anno(self, clip, t):
        return load_anno(os.path.join(self.root, clip, "anno", f"{t:05d}.json.gz"))

    def __getitem__(self, i):
        r = self.rows[i]
        clip, t = r["clip"], r["t"]
        a0 = self._anno(clip, t)

        # images (uint8; GPU-side normalization in trainer)
        imgs = {}
        for slot, cam in CAM_MAP.items():
            p = os.path.join(self.root, clip, "camera", cam, f"{t:05d}.jpg")
            imgs[slot] = np.asarray(PImage.open(p).convert("RGB"))
        image = preprocess_images_u8(imgs)                      # (4,3,672,1148) uint8

        # ego status (math frame). Empirical convention fit (sign_diag over 1209 samples):
        # B2D theta is measured from +Y (carla_garage lineage) -> heading_m = pi/2 - theta,
        # with y_m = -y. Straight fwd +24.7m, LEFT lat +1.85, RIGHT lat -2.85 under this fit.
        th_m = math.pi / 2 - float(a0["theta"])
        spd = float(a0["speed"])
        vel_w = [spd * math.cos(th_m), spd * math.sin(th_m)]
        acc = a0.get("acceleration", 0.0)
        if isinstance(acc, (list, tuple)):
            acc_w = [float(acc[0]), -float(acc[1])]
        else:
            acc_w = [float(acc) * math.cos(th_m), float(acc) * math.sin(th_m)]
        cmd_raw = int(a0.get("next_command", 4))
        cmd_bs = min(max(cmd_raw, 1), 6) - 1                    # RoadOption-1
        vel_w = [max(-45.0, min(45.0, vel_w[0])), max(-45.0, min(45.0, vel_w[1]))]
        acc_w = [max(-15.0, min(15.0, acc_w[0])), max(-15.0, min(15.0, acc_w[1]))]
        ego_state = {"velocity": vel_w, "heading": th_m, "acceleration": acc_w}
        # route target point -- word for word what b2d_sidecar.py computes:
        # same route cache, same route_target(), same ego box centre, same
        # heading. The two must not drift apart, so both call one function.
        tp = None
        if self.routes:
            route = self.routes.get(clip)
            if route is None or len(route) < 2:
                raise KeyError(f"no route for clip {clip}")
            if clip != self._cursor_clip:
                self._cursor_clip, self._cursor = clip, RouteCursor()
            ex, ey = _ego_xy(a0)
            fwd, lat, _ = route_target(route, self._cursor, ex, ey, th_m)
            tp = (fwd, lat)
        ego = build_ego_status(ego_state, cmd_bs, target_pt=tp)  # (11,) or (13,)

        # expert future in ego math frame, from the ego box centre (see _ego_xy)
        x0, y0 = _ego_xy(a0)
        c, s = math.cos(th_m), math.sin(th_m)
        fut = np.zeros((N_POSES, 3), np.float32)
        for k in range(1, N_POSES + 1):
            ak = self._anno(clip, t + FUT_STRIDE * k)
            axk, ayk = _ego_xy(ak)
            dx, dy = axk - x0, ayk - y0
            fut[k - 1, 0] = c * dx + s * dy                     # fwd
            fut[k - 1, 1] = -s * dx + c * dy                    # lat (left+)
            fut[k - 1, 2] = wrap((math.pi / 2 - float(ak["theta"])) - th_m)
        # B2D annos contain NaN pose frames (respawn glitches) — np.clip passes NaN.
        # Deterministically substitute a different sample instead of poisoning targets.
        if not (np.isfinite(fut).all() and torch.isfinite(ego).all()):
            return self.__getitem__((i + 9173) % len(self.rows))
        fut = np.clip(fut, -150.0, 150.0)
        token = f"{clip}_{t:05d}"
        return {"image": image, "ego_status": ego[None, :],     # (1,D) -> model takes [:, -1]
                "trajectory": torch.from_numpy(fut), "token": token,
                "cmd": cmd_bs}


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "ego_status": torch.stack([b["ego_status"] for b in batch]),
        "trajectory": torch.stack([b["trajectory"] for b in batch]),
        "token": [b["token"] for b in batch],
        "cmd": torch.tensor([b["cmd"] for b in batch]),
    }


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    build_index(root, out)
