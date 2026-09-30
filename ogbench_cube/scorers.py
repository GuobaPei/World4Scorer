"""Candidate scorers plugged into the LeWM CEM loop.

A scorer replaces ``model.get_cost``.  Everything else in the planner --
solver, iterations, population, MPC schedule, eval set, success rule --
is untouched, so two scorers are directly comparable.

``lewm``    the released cost (no swap at all; the anchor)
``collect`` the released cost drives planning; every scored candidate is replayed in
            the simulator and stored for head training
``blend``   released cost + coefficient x outcome head
"""

from __future__ import annotations

import numpy as np
import torch


class Scorer:
    """Base: does nothing, so the released cost runs unmodified."""

    def __init__(self, cfg, model, world, process):
        self.cfg = cfg
        self.model = model
        self.world = world
        self.process = process
        self.n_calls = 0

    def attach(self, model):
        pass

    def attach_policy(self, policy):
        self.policy = policy

    def summary(self) -> dict:
        return {"scorer_calls": self.n_calls}

    def finalize(self, out_dir):
        pass


class LewmScorer(Scorer):
    """The anchor.  ``get_cost`` is left exactly as released; we only count calls."""

    def attach(self, model):
        base = model.get_cost

        def counted(info_dict, action_candidates):
            self.n_calls += 1
            return base(info_dict, action_candidates)

        model.get_cost = counted


def build_scorer(cfg, model, world, process):
    mode = cfg.cost.mode
    if mode == "lewm":
        return LewmScorer(cfg, model, world, process)
    if mode == "collect":
        from collect_scorer import CollectScorer

        return CollectScorer(cfg, model, world, process)
    if mode == "blend":
        from head_scorer import HeadScorer

        return HeadScorer(cfg, model, world, process)
    raise ValueError(f"unknown cost.mode={mode!r}")
