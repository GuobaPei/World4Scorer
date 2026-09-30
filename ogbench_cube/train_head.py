"""Train the utility head on in-loop candidates labelled by the simulator.

The bank comes from a ``cost=collect`` run: the candidate distribution is the
one the deployed planner actually produces, CEM narrowing included.

Two disciplines that decide whether this works at all:

* Episodes used by any evaluation run are dropped -- the head never sees an
  episode it will be scored on.
* Planning calls in which every candidate has the same true outcome carry no
  ranking information.  Standardising such a group divides by a clamped
  near-zero spread and turns the target into noise, so they are dropped from
  the fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from head_model import (
    OutcomeHead,
    compose_candidate_features,
    objective,
    pairwise_accuracy,
    standardize_within_candidates,
)


def load_banks(run_dirs, blocked_episodes):
    groups, dropped = [], 0
    for rd in run_dirs:
        rd = Path(rd)
        meta = json.loads((rd / "result.json").read_text())
        ep_of_env = meta["episodes_idx"]
        for g in torch.load(rd / "bank.pt", map_location="cpu", weights_only=False):
            ep = ep_of_env[g["env"]]
            if ep in blocked_episodes:
                dropped += 1
                continue
            g = dict(g)
            g["episode"] = int(ep)
            groups.append(g)
    return groups, dropped


def stack(groups):
    return dict(
        terminal=torch.stack([g["terminal"] for g in groups]).float(),
        goal=torch.stack([g["goal"] for g in groups]).float(),
        base_cost=torch.stack([g["base_cost"] for g in groups]).float(),
        actions=torch.stack([g["actions"] for g in groups]).float(),
        utility=torch.stack([-g["task_cost"] for g in groups]).float(),
        episode=torch.tensor([g["episode"] for g in groups]),
        cem_iter=torch.tensor([g["cem_iter"] for g in groups]),
    )


def features_of(d):
    goal = d["goal"].unsqueeze(1).expand(-1, d["terminal"].shape[1], -1)
    return compose_candidate_features(d["terminal"], goal, d["base_cost"], d["actions"])


def evaluate(model, feats, dev, bs=256):
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [model(feats[i : i + bs].to(dev)).cpu() for i in range(0, feats.shape[0], bs)]
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", required=True)
    ap.add_argument("--block-from", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-utility-std", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    blocked = set()
    for f in args.block_from:
        blocked |= set(json.loads(Path(f).read_text())["episodes_idx"])
    print("blocked episodes: %d" % len(blocked))

    groups, dropped_overlap = load_banks(args.banks, blocked)
    n_all = len(groups)
    spreads = np.array([float((-g["task_cost"]).std()) for g in groups])
    keep = spreads >= args.min_utility_std
    groups = [g for g, k in zip(groups, keep) if k]
    print("groups: loaded=%d  dropped(eval overlap)=%d  dropped(flat label)=%d  kept=%d"
          % (n_all, dropped_overlap, int((~keep).sum()), len(groups)))
    assert len(groups) > 0, "no training groups left"

    eps = sorted({g["episode"] for g in groups})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * args.val_frac))
    val_eps = set(eps[:n_val])
    tr = [g for g in groups if g["episode"] not in val_eps]
    va = [g for g in groups if g["episode"] in val_eps]
    print("train groups=%d (%d episodes)   val groups=%d (%d episodes)"
          % (len(tr), len(eps) - n_val, len(va), n_val))
    assert not ({g["episode"] for g in tr} & {g["episode"] for g in va})

    d_tr, d_va = stack(tr), stack(va)
    f_tr, f_va = features_of(d_tr), features_of(d_va)
    y_tr = standardize_within_candidates(d_tr["utility"])

    dev = args.device
    model = OutcomeHead(f_tr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = f_tr.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    base_acc = pairwise_accuracy(-d_va["base_cost"], d_va["utility"])
    best = {"val_pairwise": -1.0}
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            pred = model(f_tr[idx].to(dev))
            loss = objective(pred, y_tr[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        pv = evaluate(model, f_va, dev)
        acc = pairwise_accuracy(pv, d_va["utility"])
        if acc > best["val_pairwise"]:
            best = {
                "epoch": ep,
                "val_pairwise": float(acc),
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if ep % 10 == 0 or ep == args.epochs - 1:
            print("ep%3d loss=%.4f val_pairwise=%.4f (released cost %.4f)"
                  % (ep, tot / n, acc, base_acc), flush=True)

    model.load_state_dict(best["state"])
    pv = evaluate(model, f_va, dev)
    ptr = evaluate(model, f_tr, dev)
    tr_acc = pairwise_accuracy(ptr, d_tr["utility"])
    print("\nbest epoch %d: train_pairwise=%.4f  val_pairwise=%.4f  released=%.4f"
          % (best["epoch"], tr_acc, best["val_pairwise"], base_acc))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best["state"], "config": model.config(), "input_dim": f_tr.shape[-1]},
        out / "head.pt",
    )
    (out / "train_report.json").write_text(
        json.dumps(
            {
                "banks": [str(b) for b in args.banks],
                "min_utility_std": args.min_utility_std,
                "groups_loaded": n_all,
                "groups_dropped_eval_overlap": dropped_overlap,
                "groups_dropped_flat_label": int((~keep).sum()),
                "n_train_groups": len(tr),
                "n_val_groups": len(va),
                "n_train_episodes": len(eps) - n_val,
                "n_val_episodes": n_val,
                "best_epoch": best["epoch"],
                "train_pairwise": float(tr_acc),
                "val_pairwise": best["val_pairwise"],
                "released_cost_pairwise": float(base_acc),
            },
            indent=2,
        )
    )
    print("\nWROTE %s" % (out / "head.pt"))


if __name__ == "__main__":
    main()
