"""b2d_dataset.py — Bench2Drive-Base sample index and anno helpers.

Builds samples.jsonl (one line per sample with a full future) and holds the anno
helpers the sidecar builders share. Frames @10Hz, 8 future poses @0.5s
(=frames t+5..t+40). Sample stride 5.
Coordinates: B2D anno x/y/theta are CARLA world (y-right). Math frame = y-left:
y_m = -y, heading_m = pi/2 - theta (B2D theta is measured from +Y).
Ego-frame future = R(-heading_t) @ (p-p_t), yaw wrapped.
"""
import gzip, json, os, sys, math

FUT_STRIDE, N_POSES, SAMPLE_STRIDE = 5, 8, 5


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ego_xy(anno):
    """Ego position in the math frame from the ego bounding-box centre (anno x/y is
    a noisy GNSS read); falls back to anno x/y when the frame has no ego box.
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


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    build_index(root, out)
