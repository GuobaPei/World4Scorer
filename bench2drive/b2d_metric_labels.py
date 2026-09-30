#!/usr/bin/env python3
"""b2d_metric_labels.py — consequence labels for arbitrary candidate pools,
computed against logged other-agent boxes. All torch, no grad, GPU-fast.

label_candidates(pool, boxes, bmask, gt, drive) ->
  dict of (B, M) float targets keyed by the six DrivoR sub-score heads.
pool  (B, M, 8, 3)  candidates, ego frame at t   [x, y, heading]
boxes (B, 9, 24, 5) agents at t+0.5k             [x, y, yaw, hlen, hwid]
bmask (B, 9, 24)    agent validity
gt    (B, 8, 3)     expert future (progress direction)
drive (B, 120, 100) ego-frame drivable patch (build_drivable_sidecar.py)
"""
import math

import torch

EGO_HL, EGO_HW = 2.45, 0.92          # true ego half-extents (logged box)
EGO_R2C = 1.35                       # rear axle -> box centre
DT = 0.5


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rect_overlap(c1, yaw1, e1, c2, yaw2, e2):
    """Oriented-rect SAT overlap. c* (...,2), yaw* (...), e* (hl, hw) tuples.
    Returns bool (...)."""
    d = c2 - c1
    out = None
    for yaw, (hl_a, hw_a), (hl_b, hw_b) in (
            (yaw1, e1, e2), (yaw2, e2, e1)):
        ca, sa = torch.cos(yaw), torch.sin(yaw)
        for ax, ay, half_a in ((ca, sa, hl_a), (-sa, ca, hw_a)):
            proj_d = (d[..., 0] * ax + d[..., 1] * ay).abs()
            oth = yaw2 if yaw is yaw1 else yaw1
            hb_l = (hl_b if yaw is yaw1 else hl_b)
            co, so = torch.cos(oth), torch.sin(oth)
            r_oth = (co * ax + so * ay).abs() * hl_b + \
                    (-so * ax + co * ay).abs() * hw_b
            sep = proj_d > (half_a + r_oth)
            out = sep if out is None else (out | sep)
    return ~out


# ego-frame drivable patch geometry, must match build_drivable_sidecar.py
DRIVE_F0, DRIVE_L0, DRIVE_RES, DRIVE_NF, DRIVE_NL = -10.0, -25.0, 0.5, 120, 100


def _on_drivable(patch, xy):
    """patch (B,NF,NL) uint8, xy (B,M,8,2) ego metres -> (B,M,8) float in {0,1}.
    Points outside the patch are treated as drivable: the patch covers the
    reachable envelope, and a hard 0 out there would re-introduce a length
    penalty through the back door."""
    fi = ((xy[..., 0] - DRIVE_F0) / DRIVE_RES).long()
    li = ((xy[..., 1] - DRIVE_L0) / DRIVE_RES).long()
    ok = (fi >= 0) & (fi < DRIVE_NF) & (li >= 0) & (li < DRIVE_NL)
    fi, li = fi.clamp(0, DRIVE_NF - 1), li.clamp(0, DRIVE_NL - 1)
    b = torch.arange(patch.shape[0], device=patch.device)
    b = b[:, None, None].expand_as(fi)
    v = patch[b, fi, li].float()
    return torch.where(ok, v, torch.ones_like(v))


def label_candidates(pool, boxes, bmask, gt, drive):
    B, M = pool.shape[:2]
    dev = pool.device
    boxes = boxes.to(dev).float()
    # a NaN box has no separating axis and would "collide" with everything
    bmask = bmask.to(dev) & torch.isfinite(boxes).all(-1)
    gt = gt.to(dev)

    ego_y = pool[..., 2]
    # candidate poses are rear-axle; agent boxes are centres. Shift forward.
    ego_c = pool[..., :2] + EGO_R2C * torch.stack(
        [torch.cos(ego_y), torch.sin(ego_y)], dim=-1)
    a_c = boxes[:, 1:, :, :2]                          # (B,8,24,2) steps 1..8
    a_y = boxes[:, 1:, :, 2]
    a_e = boxes[:, 1:, :, 3:5]
    a_m = bmask[:, 1:]                                 # (B,8,24)

    def coll(ec, ey):
        """ec (B,M,8,2) vs agents -> (B,M) any-step collision."""
        c1 = ec.unsqueeze(3)                           # (B,M,8,1,2)
        y1 = ey.unsqueeze(3)
        c2 = a_c.unsqueeze(1)                          # (B,1,8,24,2)
        y2 = a_y.unsqueeze(1)
        hb = a_e.unsqueeze(1)
        ov = _rect_overlap(c1, y1, (torch.full_like(y1, EGO_HL),
                                    torch.full_like(y1, EGO_HW)),
                           c2, y2, (hb[..., 0], hb[..., 1]))
        ov = ov & a_m.unsqueeze(1)
        # at-fault only: ignore agents struck from behind. The logged agents
        # are replayed non-reactively, so a braking candidate gets rear-ended
        # by a car that would in reality have braked as well.
        rel = c2 - c1                                  # (B,M,8,24,2)
        fwd = rel[..., 0] * torch.cos(y1) + rel[..., 1] * torch.sin(y1)
        ov = ov & (fwd > -0.5)
        return ov.any(-1).any(-1)                      # (B,M)

    hit = coll(ego_c, ego_y)
    noc = (~hit).float()

    # ttc: ego shifted one step ahead vs agents at the same step (lead margin)
    ec2 = torch.cat([ego_c[:, :, 1:], ego_c[:, :, -1:]], 2)
    ey2 = torch.cat([ego_y[:, :, 1:], ego_y[:, :, -1:]], 2)
    ttc = (~coll(ec2, ey2)).float()

    # progress along the expert direction, ratio to pool best
    gdir = gt[:, -1, :2]
    gdir = gdir / gdir.norm(dim=-1, keepdim=True).clamp_min(1e-3)   # (B,2)
    proj = (pool[:, :, -1, :2] * gdir[:, None]).sum(-1).clamp_min(0.0)
    # reference is the EXPERT's own progress, not the pool max: a pool-relative
    # target drifts for a bit-identical trajectory and rewards overshoot.
    ref = (gt[:, -1, :2] * gdir).sum(-1)[:, None]                   # (B,1)
    prog = torch.where(ref > 5.0, proj / ref.clamp_min(1e-3),
                       torch.ones_like(proj)).clamp(0.0, 1.0)

    # comfort: accel + yaw-rate bounds
    seg = torch.diff(torch.cat([torch.zeros(B, M, 1, 2, device=dev),
                                ego_c], 2), dim=2).norm(dim=-1)      # (B,M,8)
    v = seg / DT
    # central difference over 1.0 s: the forward difference at 0.5 s is a 2 Hz
    # differentiator and too noisy for a comfort bound.
    acc = ((v[:, :, 2:] - v[:, :, :-2]) / (2 * DT)).abs()
    yr = _wrap(torch.diff(torch.cat([torch.zeros(B, M, 1, device=dev), ego_y], 2),
                          dim=2)).abs() / DT
    comfort = ((acc.amax(-1) < 9.0) & (yr.amax(-1) < 1.5)).float()

    # drivable area: fraction of the candidate's waypoints on a lane
    dac = _on_drivable(drive.to(dev), ego_c).mean(-1)          # (B,M)

    ddc = (_wrap(pool[:, :, -1, 2] - gt[:, None, -1, 2]).abs()
           < math.pi / 2).float()

    # driving_direction_compliance is omitted: it is constant on this data and its
    # selection weight is 0.
    del ddc
    return {"no_at_fault_collisions": noc,
            "time_to_collision_within_bound": ttc,
            "ego_progress": prog,
            "comfort": comfort,
            "drivable_area_compliance": dac}


if __name__ == "__main__":
    B, M = 2, 6
    pool = torch.randn(B, M, 8, 3) * 3
    boxes = torch.randn(B, 9, 24, 5).abs()
    bmask = torch.rand(B, 9, 24) > 0.5
    gt = torch.randn(B, 8, 3) * 3
    drive = (torch.rand(B, DRIVE_NF, DRIVE_NL) > 0.3).to(torch.uint8)
    lb = label_candidates(pool, boxes, bmask, gt, drive)
    for k, v in lb.items():
        assert v.shape == (B, M) and torch.isfinite(v).all(), k
    print("self-check OK:", {k: round(float(v.mean()), 3) for k, v in lb.items()})
