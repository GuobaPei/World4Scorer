"""b2d_dataset_gpu_v2.py — training dataset: the CPU only reads raw JPEG bytes.
Targets come from the precomputed sidecar npz; jpg decode+resize happen on GPU
in the trainer (torchvision nvJPEG). Corrupt (non-finite) samples are dropped
at init via the sidecar `ok` mask.
"""
import json, os
import numpy as np
import torch
from torch.utils.data import Dataset

CAMS = ["rgb_front", "rgb_back", "rgb_front_left", "rgb_front_right"]  # CAM_F0/B0/L0/R0 order


class B2DGpuDataset(Dataset):
    def __init__(self, data_root, index_jsonl, sidecar_npz, val=False, val_every=33):
        self.root = data_root
        rows = [json.loads(l) for l in open(index_jsonl)]
        sc = np.load(sidecar_npz, allow_pickle=False)
        self.ego, self.fut = sc["ego"], sc["fut"]
        self.cmd, self.ok, self.tokens = sc["cmd"], sc["ok"], sc["tokens"]
        clips = sorted({r["clip"] for r in rows})
        val_clips = set(clips[::val_every])
        keep = val_clips if val else (set(clips) - val_clips)
        self.idx = [i for i, r in enumerate(rows)
                    if r["clip"] in keep and self.ok[i]]
        self.rows = rows

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        r = self.rows[i]
        clip, t = r["clip"], r["t"]
        bufs = []
        for cam in CAMS:
            with open(os.path.join(self.root, clip, "camera", cam, f"{t:05d}.jpg"), "rb") as f:
                bufs.append(torch.frombuffer(bytearray(f.read()), dtype=torch.uint8))
        return {"jpgs": bufs,
                "ego_status": torch.from_numpy(self.ego[i]).unsqueeze(0),
                "trajectory": torch.from_numpy(self.fut[i]),
                "token": str(self.tokens[i]), "cmd": int(self.cmd[i]), "bi": i}


def collate_gpu(batch):
    return {
        "jpgs": [b["jpgs"] for b in batch],            # list[list[ByteTensor x4]]
        "ego_status": torch.stack([b["ego_status"] for b in batch]),
        "trajectory": torch.stack([b["trajectory"] for b in batch]),
        "token": [b["token"] for b in batch],
        "cmd": torch.tensor([b["cmd"] for b in batch]),
        "bi": torch.tensor([b["bi"] for b in batch]),
    }


def gpu_decode_batch(jpgs, dev, out_wh=(1148, 672)):
    """list[list[bytes x4]] -> (B,4,3,H,W) normalized float on dev."""
    from torchvision.io import decode_jpeg
    import torch.nn.functional as F
    flat = [c for s in jpgs for c in s]
    try:
        imgs = decode_jpeg(flat, device=dev)            # list of (3,h,w) uint8 on GPU
    except Exception:
        imgs = [decode_jpeg(b).to(dev) for b in flat]   # CPU fallback
    x = torch.stack(imgs).float() / 255.0               # (B*4,3,h,w)
    x = F.interpolate(x, size=(out_wh[1], out_wh[0]), mode="bicubic",
                      align_corners=False, antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x = (x - mean) / std
    B = len(jpgs)
    return x.view(B, 4, 3, out_wh[1], out_wh[0]).clamp_(-5, 5)
