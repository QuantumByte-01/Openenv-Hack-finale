"""SpeedupRubric — Roofline-grounded reward (per plan §10).

reward = log2(1 + speedup / roofline_peak(hw)) / LOG_NORM

This is physically interpretable: the agent's reward maxes out at exactly the
hardware's theoretical ceiling. An agent that hits the Roofline gets 1.0;
an agent at half the ceiling gets ~0.6; no reward grows unbounded.

Why log not linear: a 100x speedup is not 10x more impressive than a 10x
speedup once you've blown past the Roofline; you've hit a different bottleneck
and the marginal reward should plateau.
"""

from __future__ import annotations

import math

from .rubrics import Rubric
from server.tools.hardware_profiler import roofline_bound


# Normalize so that hitting the Roofline ceiling yields ~1.0 reward
# log2(1 + 1.0) = 1.0, so LOG_NORM = 1.0 means speedup == roofline_peak yields exactly 1.0.
# We allow the agent to slightly exceed the ceiling (up to ~2x) which gives ~1.6 reward,
# clamped to 1.0 by WeightedSum.
LOG_NORM = 1.0


class SpeedupRubric(Rubric):
    name = "speedup"

    def score(self, state, submission: dict[str, Any]) -> float:  # type: ignore[override]
        speedup = float(submission.get("speedup", 0.0))
        if speedup <= 0:
            self.last_breakdown = {"speedup": 0.0, "reward": 0.0}
            return 0.0

        peak = roofline_bound(state.hardware_profile)
        normalized = speedup / max(peak, 1e-6)
        reward = math.log2(1 + normalized) / LOG_NORM

        # Clamp to [0, 1]
        reward = max(0.0, min(1.0, reward))

        self.last_breakdown = {
            "speedup": speedup,
            "roofline_peak": peak,
            "normalized": normalized,
            "reward": reward,
        }
        return reward


# Re-import after definition
from typing import Any  # noqa: E402


__all__ = ["SpeedupRubric"]
