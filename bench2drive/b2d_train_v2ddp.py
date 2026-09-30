#!/usr/bin/env python3
"""b2d_train_v2ddp.py — Bench2Drive trainer for the World4Scorer architecture (DDP).

Losses (pointwise; no ranking or listwise terms):
  traj   : winner-take-all L1 over the 64 proposals vs the expert future
  select : BCE of the sub-score heads against consequence labels of each candidate
           (b2d_metric_labels.label_candidates: logged agent boxes, drivable raster,
           expert progress), on the model's own proposals and on a vocabulary pool
  future : 1 - cos between the readout of the expert trajectory and the frozen DINOv2
           feature of the front camera 2 s ahead (b2d_bank.py); weight 0.1, gradient
           scale 0.05 into the shared predictor (model_config.py; NAVSIM uses 1.0 and 1.0)
"""
import argparse, json, math, os, sys, time
_B2D = os.path.dirname(os.path.abspath(__file__))  # bench2drive/
_REPO = os.path.dirname(_B2D)
_DATA = os.environ.get("B2D_DATA", os.path.join(_B2D, "data"))
_DINO = os.environ.get("DINO_WEIGHTS", os.path.join(
    _REPO, "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"))
from torch.nn.attention import SDPBackend, sdpa_kernel
from b2d_metric_labels import label_candidates
from b2d_vocab import Vocab, TARGET_V, N_SHAPES, N_SPEEDS


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from datetime import timedelta

sys.path.insert(0, _REPO)
sys.path.insert(0, _B2D)
from model_config import build_b2d_config  # noqa
from b2d_dataset_gpu_v2 import B2DGpuDataset, collate_gpu, gpu_decode_batch

# driving_direction_compliance dropped: constant 1.0, zero within-scene
# variance, and its inference weight is 0.0 — it only diluted the informative
# heads' share of the averaged BCE.
SUB_KEYS = ["no_at_fault_collisions", "drivable_area_compliance",
            "time_to_collision_within_bound", "ego_progress", "comfort"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=_DATA)
    ap.add_argument("--index", default=os.path.join(_DATA, "samples.jsonl"))
    ap.add_argument("--sidecar", default=os.path.join(_DATA, "sidecar.npz"))
    ap.add_argument("--boxes", default=os.path.join(_DATA, "boxes.npz"))
    ap.add_argument("--drivable", default=os.path.join(_DATA, "drivable_sidecar.npy"))
    ap.add_argument("--bank", default=os.path.join(_DATA, "b2d_bank_f2s.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=-1)
    ap.add_argument("--resume", action="store_true", help="resume from <out>/last.ckpt if present")
    ap.add_argument("--init", default="",
                    help="warm start: load this state_dict before training. Must "
                         "already have the right ego_status width -- expand an "
                         "11-dim checkpoint with tools/warm_start_route_tp.py first.")
    ap.add_argument("--save_every", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    RANK = int(os.environ.get("RANK", -1))
    DDP_ON = RANK >= 0
    R0 = (not DDP_ON) or RANK == 0
    if DDP_ON:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    dev = "cuda"

    # ego width comes from the sidecar, never from a flag: 11 = the original
    # vector, 13 = plus the route target point. Reading it here makes a
    # sidecar/model mismatch impossible instead of silent.
    EGO_DIM = int(np.load(args.sidecar, allow_pickle=False)["ego"].shape[1])
    print(f"[b2d] sidecar ego_status_dim = {EGO_DIM}", flush=True)
    cfg = build_b2d_config(_DINO, realized_future_emb_path=args.bank, ego_status_dim=EGO_DIM)

    from navsim.agents.drivoR.drivor_model import DrivoRModel
    # torch2.5: GridMask does .view on a non-contiguous permute -> feed it contiguous memory
    from navsim.agents.drivoR.layers.image_encoder import grid_mask as _gm
    _ogf = _gm.GridMask.forward
    _gm.GridMask.forward = lambda self, x: _ogf(self, x.contiguous())
    model = DrivoRModel(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[b2d] params {n_par/1e6:.2f}M", flush=True)
    if args.init:
        # Warm start: missing parameters raise (assert below); a shape mismatch
        # raises in load_state_dict regardless of strict.
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        sd = {k: v for k, v in sd.items() if not k.startswith("realized_bank")}
        mi, un = model.load_state_dict(sd, strict=False)
        assert not mi, f"--init checkpoint missing params: {mi[:5]}"
        w = model.hist_encoding.weight
        print(f"[b2d] warm start {args.init}: hist_encoding {tuple(w.shape)}, "
              f"{len(un)} unexpected keys", flush=True)
    if DDP_ON:
        for p_ in model.parameters():
            dist.broadcast(p_.data, 0)

    tr = B2DGpuDataset(args.data, args.index, args.sidecar, val=False)
    va = B2DGpuDataset(args.data, args.index, args.sidecar, val=True)
    print(f"[b2d] train {len(tr)} val {len(va)}", flush=True)
    _samp = DistributedSampler(tr, shuffle=True) if DDP_ON else None
    dl = DataLoader(tr, batch_size=args.bs, shuffle=(_samp is None), sampler=_samp, num_workers=args.workers,
                    collate_fn=collate_gpu, pin_memory=False, drop_last=True,
                    prefetch_factor=1 if args.workers > 0 else None,
                    persistent_workers=args.workers > 0)
    dv = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=4,
                    collate_fn=collate_gpu, pin_memory=False)

    lr = args.lr if args.lr > 0 else 2e-4 * args.bs / 64.0
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    model._cap = {}
    def _hpre(m, a): model._cap["scene"] = a[1]
    def _hpost(m, a, o): model._cap["pre"] = o
    def _hspre(m, a): model._cap["post"] = a[1]
    _trunk = model.future_predictor
    _trunk.register_forward_pre_hook(_hpre)
    _trunk.register_forward_hook(_hpost)
    model.scorer.register_forward_pre_hook(_hspre)
    BOXNPZ = np.load(args.boxes, mmap_mode="r")
    BOXES_ALL, BMASK_ALL = BOXNPZ["boxes"], BOXNPZ["bmask"]
    # ego-frame drivable-area patch, one per sample (see build_drivable_sidecar.py)
    DRIVE_ALL = np.load(args.drivable, mmap_mode="r")
    VOCAB = Vocab(os.path.join(_B2D, "b2d_shapes.npz"))
    _shape_flat = torch.tensor(VOCAB.shapes.reshape(N_SHAPES, -1))

    steps_total = args.epochs * len(dl)
    warm = min(500, steps_total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps_total - warm, 1))))
    AC = dict(device_type="cuda", dtype=torch.bfloat16)

    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)

    def losses(batch):
        img = gpu_decode_batch(batch["jpgs"], dev)
        feats = {"image": img,
                 "ego_status": batch["ego_status"].to(dev, non_blocking=True)}
        gt = batch["trajectory"].to(dev)                        # (B,8,3)
        out = model(feats)
        props = out["proposals"]                                # (B,64,8,3)
        # WTA imitation
        l1 = (props - gt[:, None]).abs().mean((-1, -2))         # (B,64)
        wta = l1.min(1).values.mean()
        # consequence labels on an expanded pool: own 64 + vocab uniform 64
        # + expert-neighborhood 64, scored by the same trunk+scorer tail.
        Bn = props.shape[0]
        with torch.no_grad():
            ade = torch.linalg.norm(props[..., :2] - gt[:, None, :, :2], dim=-1).mean(-1)
            # clamp: teleport/reset frames give absurd finite v0 (60 m/s) whose
            # vocab compositions blow up attention logits downstream
            v0 = (torch.linalg.norm(gt[:, 0, :2], dim=-1) / 0.5).clamp(0.0, 20.0)
            vpool = []
            for bi_ in range(Bn):
                r1, _ = VOCAB.round1(float(v0[bi_]))
                gp = gt[bi_, :, :2].cpu().numpy()
                pts = np.concatenate([np.zeros((1, 2)), gp], 0)
                L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
                if L > 1.0:
                    cum = np.linalg.norm(np.diff(pts, axis=0), axis=1).cumsum()
                    cum = np.concatenate([[0], cum]) / cum[-1]
                    gs = np.stack([np.interp(np.linspace(0, 1, VOCAB.shapes.shape[1]),
                                             cum, pts[:, d]) for d in (0, 1)], 1) / L
                    sid = int(torch.cdist(torch.tensor(gs.reshape(1, -1), dtype=torch.float32),
                                          _shape_flat).argmin())
                else:
                    sid = 0
                r2, _ = VOCAB.round2([(sid, 0)], float(v0[bi_]), budget=64)
                vpool.append(np.concatenate([r1, r2], 0))
            vpool_t = torch.tensor(np.stack(vpool), device=props.device,
                                   dtype=props.dtype).clamp(-80.0, 80.0)  # (B,128,8,3)
        cap = model._cap
        ego_tok = (cap["post"] - cap["pre"])[:, :1, :]
        trunk_mod = model.future_predictor
        bx = torch.from_numpy(np.ascontiguousarray(BOXES_ALL[batch["bi"].numpy()]))
        bm = torch.from_numpy(np.ascontiguousarray(BMASK_ALL[batch["bi"].numpy()]))
        dv = torch.from_numpy(np.ascontiguousarray(DRIVE_ALL[batch["bi"].numpy()]))

        def _bce_on(cand, logit_):
            with torch.no_grad():
                lb = label_candidates(cand.detach().float(), bx, bm, gt, drive=dv)
            return sum(F.binary_cross_entropy_with_logits(logit_[k],
                                                          lb[k].to(props.dtype))
                       for k in SUB_KEYS) / len(SUB_KEYS)

        # (a) the model's own 64, in a 64-token trunk call — the exact function
        #     deployed at inference. out["pred_logit"] already is that call.
        bce_own = _bce_on(props, out["pred_logit"])
        # (b) the vocab pool as separate negatives, in its own call, so it can
        #     never change what the deployed 64-token function computes.
        emb_v = model.pos_embed(vpool_t.reshape(Bn, vpool_t.shape[1], -1).detach())
        # math SDP only for this call: the flash-attention backward gives NaN on this shape
        with sdpa_kernel([SDPBackend.MATH]):
            tr_v = trunk_mod(emb_v, cap["scene"])
            logit_v = model.scorer(vpool_t, tr_v + ego_tok)[0]
        bce = 0.5 * (bce_own + _bce_on(vpool_t, logit_v))
        loss = wta + bce
        if not torch.isfinite(loss):
            print("DIAG img_fin", torch.isfinite(img).all().item(),
                  "img_rng", float(img.min()), float(img.max()),
                  "props_fin", torch.isfinite(props).all().item(),
                  "logits_fin", {k: torch.isfinite(out["pred_logit"][k]).all().item() for k in SUB_KEYS},
                  "wta", float(wta), "bce", float(bce), flush=True)
        fut = torch.tensor(0.0, device=dev)
        fl = model.realized_future_loss({"trajectory": gt, "token": batch["token"]})
        if fl is not None:
            fut = fl
            loss = loss + cfg.realized_weight * fut
        return loss, wta.detach(), bce.detach(), fut.detach(), ade.min(1).values.mean().detach()

    def atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    step, t0, n_skip, start_ep = 0, time.time(), 0, 0
    if args.resume:
        cands = [p for p in (f"{args.out}/last.ckpt", f"{args.out}/step_last.ckpt")
                 if os.path.exists(p)]
        # a resume would silently overwrite the warm start; make the collision loud.
        assert not (cands and args.init), (
            f"--resume found {cands[0]} and --init {args.init} was also given; "
            "the resume would overwrite the warm start. Pick one.")
        if cands:
            path = max(cands, key=os.path.getmtime)   # newest state wins
            ck = torch.load(path, map_location="cpu")
            model.load_state_dict(ck["state_dict"])
            if "opt" in ck:
                opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
                step = ck.get("step", 0)
            start_ep = ck.get("epoch", -1) + 1
            if "opt" not in ck:
                step = start_ep * len(dl)
                for _ in range(step):
                    sched.step()
                print(f"[resume] no opt in ckpt: cold Adam, sched fast-forwarded to {step}", flush=True)
            print(f"[resume] {os.path.basename(path)} -> epoch {start_ep} step {step}", flush=True)
    n_badloss = 0
    for ep in range(start_ep, args.epochs):
        if DDP_ON:
            dl.sampler.set_epoch(ep)
        model.train()
        for b in dl:
            with torch.amp.autocast(**AC):
                loss, wta, bce, fut, adem = losses(b)
            if not torch.isfinite(loss):
                n_badloss += 1
                print("BAD_BATCH tokens:", b["token"][:6],
                      "gt_absmax", float(b["trajectory"].abs().max()),
                      "ego_absmax", float(b["ego_status"].abs().max()), flush=True)
                if not DDP_ON:
                    if n_badloss > 100:
                        raise RuntimeError(f"NONFINITE_LOSS x{n_badloss} step {step}")
                    sched.step(); step += 1
                    continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if DDP_ON:
                _grads = [p_.grad for p_ in model.parameters() if p_.grad is not None]
                _flat = _flatten_dense_tensors(_grads)
                dist.all_reduce(_flat, op=dist.ReduceOp.AVG)
                for _g, _ng in zip(_grads, _unflatten_dense_tensors(_flat, _grads)):
                    _g.copy_(_ng)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gnorm):
                # bf16 has no GradScaler: a non-finite grad batch would poison every
                # weight through the clip coefficient — skip the step instead.
                opt.zero_grad(set_to_none=True)
                n_skip += 1
                if n_skip % 10 == 1:
                    print(f"[skip] nonfinite grad batch #{n_skip} at step {step}", flush=True)
                if n_skip > 200:
                    raise RuntimeError(f"SKIP_FUSE x{n_skip} step {step}")
                sched.step(); step += 1
                continue
            opt.step(); sched.step()
            if R0 and step % 50 == 0:
                print(f"[ep{ep} s{step}] loss {loss.item():.4f} wta {wta:.3f} bce {bce:.3f} "
                      f"fut {fut:.3f} minADE {adem:.2f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time()-t0)/max(step,1):.2f}s/it", flush=True)
            if R0 and step % args.save_every == 0 and step > 0:
                atomic_save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                             "sched": sched.state_dict(), "epoch": ep - 1, "step": step},
                            f"{args.out}/step_last.ckpt")
            step += 1
        # val
        model.eval()
        vl, vade, vsel = [], [], []
        with torch.no_grad():
            for b in dv:
                vimg = gpu_decode_batch(b["jpgs"], dev)
                feats = {"image": vimg, "ego_status": b["ego_status"].to(dev)}
                gt = b["trajectory"].to(dev)
                out = model(feats)
                props = out["proposals"]
                ade = torch.linalg.norm(props[..., :2] - gt[:, None, :, :2], dim=-1).mean(-1)
                pick = out["pdm_score"].argmax(1)
                sel_ade = ade[torch.arange(len(pick)), pick]
                vade.append(ade.min(1).values.mean().item())
                vsel.append(sel_ade.mean().item())
        if R0:
            print(f"[VAL ep{ep}] minADE {np.mean(vade):.3f} selADE {np.mean(vsel):.3f}", flush=True)
            atomic_save({"state_dict": model.state_dict(), "cfg": dict(cfg), "epoch": ep,
                         "val_minade": float(np.mean(vade)), "val_selade": float(np.mean(vsel))},
                        f"{args.out}/ep{ep:02d}.ckpt")
            atomic_save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "epoch": ep, "step": step},
                        f"{args.out}/last.ckpt")
    print("===TRAIN_DONE===", flush=True)


if __name__ == "__main__":
    main()
