#!/usr/bin/env python3
"""warm_start_route_tp.py — widen an 11-dim checkpoint to 13 dims.

The only tensor whose shape changes when the route target point joins
ego_status is hist_encoding.weight, (d_model, 11) -> (d_model, 13). The old
columns are copied verbatim and the two new ones start at zero, so at step 0 the
new model computes hist_encoding(x) exactly as the old one did for any input
whose target-point dims are zero -- the route feature grows from nothing instead
of the checkpoint being thrown away. The bias is copied unchanged.

  python3 warm_start_route_tp.py <in.ckpt> <out.ckpt> [--ego-dim 13]

Everything else in the checkpoint (optimizer state excluded on purpose -- Adam
moments for the old-width tensor are meaningless) is carried through.
"""
import argparse, os
import torch

KEY = "hist_encoding.weight"
BKEY = "hist_encoding.bias"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ego-dim", type=int, default=13)
    a = ap.parse_args()

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    assert KEY in sd, f"{KEY} not in {a.src}"
    w = sd[KEY]
    d_model, old = w.shape
    new = a.ego_dim
    assert new > old, f"{KEY} is already {old} wide, cannot widen to {new}"

    nw = torch.zeros(d_model, new, dtype=w.dtype)
    nw[:, :old] = w
    zeroed = list(range(old, new))
    sd[KEY] = nw
    assert torch.equal(nw[:, :old], w)
    assert float(nw[:, zeroed].abs().max()) == 0.0

    out = dict(ck) if isinstance(ck, dict) and "state_dict" in ck else {"state_dict": sd}
    if "state_dict" in out:
        out["state_dict"] = sd
    for k in ("opt", "sched", "step", "epoch"):
        out.pop(k, None)          # a warm start is not a resume
    out["warm_start_from"] = os.path.abspath(a.src)
    out["ego_status_dim"] = a.ego_dim
    tmp = a.dst + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, a.dst)
    print(f"WARM_START_OK {a.src} -> {a.dst}")
    print(f"  {KEY}: {tuple(w.shape)} -> {tuple(nw.shape)}, "
          f"columns {zeroed} zero-initialised, {BKEY} copied")


if __name__ == "__main__":
    main()
