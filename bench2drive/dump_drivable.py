#!/usr/bin/env python3
"""dump_drivable.py — export a drivable-area raster per CARLA town.

The drivable_area_compliance label (b2d_metric_labels.py) asks whether each
candidate waypoint lies on a driving lane, so a legal lane change is not penalised.

CARLA has the geometry: `Map.generate_waypoints(d)` walks every driving lane of
the loaded town. Rasterising those into an occupancy grid gives an offline
lookup that answers "is this point on a driving lane" for any candidate
waypoint, at any position, with no map server at label time.

Writes <out>/<Town>.npz with {grid (H,W) uint8, x0, y0, res} in CARLA world
coordinates (note: y is NOT negated here; the label side negates).
"""
import glob
import os
import sys
import time

import numpy as np

_CARLA = os.environ.get("CARLA_ROOT", "/opt/carla")
sys.path.insert(0, f"{_CARLA}/PythonAPI/carla")
sys.path.insert(0, f"{_CARLA}/PythonAPI")
_DATA = os.environ.get("B2D_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
import carla  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 2900
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_DATA, "drivable")
RES = 0.5            # m per cell
HALF_W = 2.0         # half lane width painted around each waypoint, metres

os.makedirs(OUT, exist_ok=True)
towns = sorted({p.split("/")[-1].split("_")[1]
                for p in glob.glob(os.path.join(_DATA, "*_Town*"))})
print(f"[dump] towns needed by the training clips: {towns}")

client = carla.Client("127.0.0.1", PORT)
client.set_timeout(600.0)
avail = [m.split("/")[-1] for m in client.get_available_maps()]
print(f"[dump] server offers {len(avail)} maps")

for town in towns:
    dst = f"{OUT}/{town}.npz"
    if os.path.exists(dst):
        print(f"[dump] {town}: already done")
        continue
    match = [m for m in avail if m == town] or [m for m in avail if m.startswith(town)]
    if not match:
        print(f"[dump] {town}: NOT OFFERED by this server, skipped")
        continue
    t0 = time.time()
    world = client.load_world(match[0])
    time.sleep(3.0)
    cmap = world.get_map()
    wps = cmap.generate_waypoints(1.0)
    if not wps:
        print(f"[dump] {town}: no waypoints, skipped")
        continue
    xs = np.array([w.transform.location.x for w in wps], np.float32)
    ys = np.array([w.transform.location.y for w in wps], np.float32)
    ws = np.array([max(w.lane_width, 2.0) for w in wps], np.float32)
    pad = HALF_W + 5.0
    x0, y0 = xs.min() - pad, ys.min() - pad
    W = int((xs.max() + pad - x0) / RES) + 1
    H = int((ys.max() + pad - y0) / RES) + 1
    grid = np.zeros((H, W), np.uint8)
    # paint a disc of the lane's own half-width at each waypoint
    r_max = int(np.ceil(ws.max() / 2 / RES))
    dy, dx = np.mgrid[-r_max:r_max + 1, -r_max:r_max + 1]
    d2 = (dx * RES) ** 2 + (dy * RES) ** 2
    ci = ((xs - x0) / RES).astype(np.int32)
    ri = ((ys - y0) / RES).astype(np.int32)
    for k in range(len(wps)):
        m = d2 <= (ws[k] / 2) ** 2
        r, c = ri[k] + dy[m], ci[k] + dx[m]
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        grid[r[ok], c[ok]] = 1
    np.savez_compressed(dst, grid=grid, x0=np.float32(x0), y0=np.float32(y0),
                        res=np.float32(RES))
    print(f"[dump] {town}: {len(wps)} waypoints -> grid {H}x{W} "
          f"({100 * grid.mean():.1f}% drivable, {os.path.getsize(dst) / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s)", flush=True)
print("DUMP_DONE")
