"""route_target.py — the single definition of the route target point.

The model's ego_status carried no route geometry at all: 11 dims of
[pose(3), velocity(2), acceleration(2), driving_command one-hot(4)]. At a
junction the command is LANEFOLLOW for most of the approach, so nothing in the
input says which way the road goes, and the scorer ranks the turning candidates
last even though they sit in the pool. This module supplies the missing two
dims.

ONE definition, imported by both sides:

  * training   — b2d_route_tp_sidecar.py, over the route reconstructed from the
    anno's dense near-node stream (b2d_route_cache.py)
  * deployment — b2d_agent/world4scorer_agent.py, over `_dense_plan_world`,
    the leaderboard's 1 m interpolation of the route

A point at a fixed distance along the route depends only on the route curve and
the ego pose, so it is the same at training and at deployment; the anno's sparse
route node (x_command_far) would depend on how each side downsamples the route.

Frame: the sidecar "math frame" used everywhere in this codebase — X = CARLA x,
Y = -CARLA y, heading measured CCW from +X. The returned point is ego-relative,
+x forward, +y left, in metres, NOT normalised (carla_garage feeds raw metres
too).
"""
import math
import numpy as np

# look-ahead along the route, metres
TARGET_DIST = 20.0
TARGET_DIM = 2
# window sizes for the monotone cursor; the route is interpolated at 1 m
_NEAR_WIN = 400
_FWD_WIN = 600


class RouteCursor:
    """Monotone position along the route, so a route that loops back on itself
    cannot snap the target onto an earlier leg."""
    __slots__ = ("i",)

    def __init__(self, i=0):
        self.i = int(i)

    def reset(self):
        self.i = 0


def prepare_route(pts):
    """(N,2) float64 array in the math frame from a sequence of (x, y)."""
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(a) < 2:
        return a
    keep = np.concatenate([[True], np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1])) > 1e-6])
    return np.ascontiguousarray(a[keep])


def route_from_plan(plan):
    """Math-frame route from the leaderboard's [(carla.Transform, RoadOption)]."""
    return prepare_route([(p[0].location.x, -p[0].location.y) for p in plan])


def route_target(route, cursor, ex, ey, heading, dist=TARGET_DIST):
    """Target point in the ego frame: (forward, lateral), lateral positive left.

    The point on `route` at Euclidean distance `dist` from (ex, ey), found by
    walking forward from `cursor` and interpolating the straddling segment. If
    the route ends before `dist` the last route point is returned, so the target
    saturates towards the goal on a short route instead of jumping.

    Returns (fwd, lat, hit_end).
    """
    n = len(route)
    if n == 0:
        return 0.0, 0.0, True
    i0 = cursor.i if cursor is not None else 0
    i0 = max(0, min(i0, n - 1))
    # monotone cursor: nearest route point at or after the current cursor
    hi = min(i0 + _NEAR_WIN, n)
    seg = route[i0:hi]
    k = i0 + int(np.argmin(np.hypot(seg[:, 0] - ex, seg[:, 1] - ey)))
    if cursor is not None:
        cursor.i = k
    # first point at or beyond `dist`, walking forward from there
    hj = min(k + _FWD_WIN, n)
    fwdseg = route[k:hj]
    d = np.hypot(fwdseg[:, 0] - ex, fwdseg[:, 1] - ey)
    over = np.flatnonzero(d >= dist)
    if len(over) == 0:
        if hj < n:                       # window too short: fall back to a full scan
            d = np.hypot(route[k:, 0] - ex, route[k:, 1] - ey)
            over = np.flatnonzero(d >= dist)
            fwdseg = route[k:]
        if len(over) == 0:               # the route really does end before dist
            px, py = float(route[-1, 0]), float(route[-1, 1])
            return _to_ego(px, py, ex, ey, heading) + (True,)
    j = int(over[0])
    if j == 0:
        px, py = float(fwdseg[0, 0]), float(fwdseg[0, 1])
    else:
        a, b = fwdseg[j - 1], fwdseg[j]
        da, db = float(d[j - 1]), float(d[j])
        u = 0.0 if db - da < 1e-9 else min(max((dist - da) / (db - da), 0.0), 1.0)
        px = float(a[0] + u * (b[0] - a[0]))
        py = float(a[1] + u * (b[1] - a[1]))
    return _to_ego(px, py, ex, ey, heading) + (False,)


def _to_ego(px, py, ex, ey, heading):
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = px - ex, py - ey
    return c * dx + s * dy, -s * dx + c * dy


def reconstruct_route_from_anno(near_xy, first_pose_xy, last_far_xy, step=1.0):
    """Training-side route for one B2D clip.

    The clips carry no route file, but every anno frame stores the node the
    expert's dense planner is heading for (`x_command_near`), and that planner
    walks the leaderboard's 1 m route. The unique near nodes are therefore the
    route itself, minus the first ~3.5 m (already popped when logging starts)
    and the last ~3.5 m (never reached). The head is restored from the ego's
    spawn pose -- the route starts under the ego -- and the tail from the final
    far node, which is the route's last node.
    """
    D = []
    for q in near_xy:
        if not D or math.hypot(q[0] - D[-1][0], q[1] - D[-1][1]) > 1e-3:
            D.append((float(q[0]), float(q[1])))
    if len(D) < 2:
        return prepare_route([first_pose_xy, last_far_xy])
    out = [tuple(first_pose_xy)] + _fill(first_pose_xy, D[0], step) + D
    out += _fill(D[-1], last_far_xy, step) + [tuple(last_far_xy)]
    return prepare_route(out)


def _fill(a, b, step=1.0):
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    n = int(d // step)
    return [(a[0] + (b[0] - a[0]) * i / (n + 1), a[1] + (b[1] - a[1]) * i / (n + 1))
            for i in range(1, n + 1)]
