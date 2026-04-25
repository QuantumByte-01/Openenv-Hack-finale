"""PortabilityRubric — bonus for code that works across hardware profiles.

Only contributes when state.difficulty_axes['portability_required'] is on.
If the axis is off, returns 0 (i.e., this component contributes nothing to the
weighted sum, freeing the 10% weight to be implicit-zero).

Score = n_profiles_passing / n_other_profiles, clamped [0, 1]. Eligible only if
n_profiles_passing ≥ 3 (per plan §3 axis 4).
"""

from __future__ import annotations

from typing import Any

from .rubrics import Rubric


class PortabilityRubric(Rubric):
    name = "portability"

    def score(self, state, submission: dict[str, Any]) -> float:
        # If the axis is off, this rubric contributes 0 (it's still in the weighted sum,
        # but it neutralizes the 0.10 weight automatically).
        axis_on = state.difficulty_axes.get("portability_required", 0) >= 1
        portability = submission.get("portability", {}) or {}
        n_passing = int(portability.get("n_profiles_passing", 0))

        if not axis_on:
            self.last_breakdown = {"axis_on": False, "score": 0.0}
            return 0.0

        # Need at least 3 to count
        if n_passing < 3:
            self.last_breakdown = {"axis_on": True, "n_passing": n_passing, "score": 0.0,
                                   "reason": "below_3_profile_threshold"}
            return 0.0

        # Normalize against other-profile count (7 = total profiles minus the home one)
        denom = max(7, 1)
        score = min(1.0, n_passing / denom)
        self.last_breakdown = {"axis_on": True, "n_passing": n_passing, "denom": denom, "score": score}
        return score


__all__ = ["PortabilityRubric"]
