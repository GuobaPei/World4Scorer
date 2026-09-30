#!/usr/bin/env python3
"""Check a download against the released files, and the model size.

  python tools/verify_release.py --artifact navsim_ckpt=weights/w4s_navsim_ep23.ckpt ...
  python tools/verify_release.py --count-params   # builds the model (torch, timm, omegaconf)
"""
import argparse, hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS = {  # sha256 of the released files
    "navsim_ckpt": "22604f83dda8a2bd2dd05356fca8457f0c7dc78439f47db396e5c85d74971a44",  # best epoch 23
    "b2d_ckpt": "ed3f61a9ac850627c07bbf1d37cc0a9e123d5b41a9c1cecefe820644e14aff5b",     # route point, epoch 14
    "b2d_route_blind_ckpt": "640dd42934b57d9305c336db7234747533977ad916c18723a9bb97b385407d31",
    "dino": "dca70548ecd7b03ffba6172c4db403014511b5ee6073f9fca72ba9e6e602a25d",
    "future_bank": "2d64716d8f07a0dbc94e3f240108167acf9c9ef5093635eacb857725a13f0d82",
    "cube_head": "fce54aa53576587f002a8a515db393bc346b1daee2716630ecf14352fb08bf0e",  # OGBench-Cube head n10_r1
    "pack_tokens": "37b445adac5c5d4674824d7fd148549abf5687054af6b492bba6d4ccc5c3dca6",
    "pack_candidates": "823765d80d44897d37741ffeea95bc3217e433c5df6600828c8cef8b651300fc",
    "pack_subscores": "5639a9f27b744b207add8400b9753d2e4216dc078de27ab408ce11998e19f58c",
    "pack_mask": "742517e558f7cf041a8f481d0690517ef3f552dca51997f73ec90e2451e4a453",
}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def count_params():
    sys.path.insert(0, ROOT)
    import timm
    from omegaconf import OmegaConf
    create = timm.create_model
    timm.create_model = lambda *a, **k: create(*a, **{**{x: y for x, y in k.items()
                                                          if x != "pretrained_cfg_overlay"},
                                                       "pretrained": False})
    cfg = OmegaConf.load(os.path.join(ROOT, "navsim/planning/script/config/common/agent/drivoR.yaml")).config
    for k, v in dict(proposal_num=64, refiner_ls_values=0.0, one_token_per_traj=True,
                     refiner_num_heads=1, tf_d_model=256, tf_d_ffn=1024, area_pred=False, agent_pred=False,
                     ref_num=4, long_trajectory_additional_poses=2).items():
        OmegaConf.update(cfg, k, v, force_add=True)
    OmegaConf.update(cfg, "image_backbone.focus_front_cam", False, force_add=True)
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    n = sum(p.numel() for p in DrivoRModel(cfg).parameters())
    print(f"[{'ok' if n == 35_520_926 else 'FAIL'}] parameters {n:,} (expected 35,520,926)")
    return n == 35_520_926


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-params", action="store_true")
    ap.add_argument("--artifact", nargs="*", default=[], metavar="KIND=PATH",
                    help="kinds: " + ", ".join(ARTIFACTS))
    a = ap.parse_args()
    if not (a.artifact or a.count_params):
        ap.error("nothing to check: pass --artifact and/or --count-params")
    ok = True
    for item in a.artifact:
        kind, path = item.split("=", 1)
        got = sha256(path)
        print(f"[{'ok' if got == ARTIFACTS[kind] else 'FAIL'}] {kind}: {path}")
        ok &= got == ARTIFACTS[kind]
    if a.count_params:
        ok &= count_params()
    print("ALL OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
