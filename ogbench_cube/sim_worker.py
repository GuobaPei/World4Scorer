"""Simulator worker: one head-less cube env per process.

The env is built with exactly the kwargs the evaluation world uses, minus rendering.
A candidate's label is the minimum over its rollout of the cube-to-target distance,
the quantity the episode success rule thresholds at 0.04 m.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

PHYS_KEYS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart", "time")

_ENV = None


def snapshot(env):
    """Full physics state of a live mujoco env (bitwise reproducible)."""
    d = env._data
    out = {}
    for k in PHYS_KEYS:
        v = getattr(d, k)
        out[k] = float(v) if k == "time" else np.array(v, copy=True)
    return out


def worker_init(env_kwargs):
    global _ENV
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        os.nice(15)
    except Exception:
        pass
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  registers swm/* envs

    kw = dict(env_kwargs)
    name = kw.pop("env_name")
    max_steps = kw.pop("max_episode_steps")
    env = gym.make(name, max_episode_steps=max_steps, render_mode=None, **kw)
    env.reset(seed=0)
    _ENV = env.unwrapped


def _set_phys(env, s):
    import mujoco

    d = env._data
    for k in PHYS_KEYS:
        if k == "time":
            d.time = s[k]
        else:
            getattr(d, k)[:] = s[k]
    mujoco.mj_forward(env._model, d)


def worker_rollout(snap, target, candidates):
    """Replay each candidate from *snap*; return its minimum cube-to-target distance."""
    env = _ENV
    n = candidates.shape[0]
    task = np.empty(n, dtype=np.float32)
    jid = env._model.joint("object_joint_0").id
    adr = env._model.jnt_qposadr[jid]
    for i in range(n):
        _set_phys(env, snap)
        best_t = np.inf
        for a in candidates[i]:
            env.step(a)
            q = env._data.qpos
            best_t = min(best_t, float(np.linalg.norm(q[adr : adr + 3] - target)))
        task[i] = best_t
    return task
