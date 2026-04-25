"""SelfCorrectionRubric — rewards improvement from R1 to R3.

Per plan §10 anti-gaming rule: agent could deliberately submit a bad R1 to
maximize R1→R3 delta. Defense: R1 must compile (CompilationRubric pass)
or this rubric returns 0. That makes a deliberately-broken R1 a net loss.

Score = clamp((R3_speedup - R1_speedup) / R1_speedup, 0, 1)
        but only if R1.compile_status == "success".
"""

from __future__ import annotations

from typing import Any

from .rubrics import Rubric


class SelfCorrectionRubric(Rubric):
    name = "self_correction"

    def score(self, state, submission: dict[str, Any]) -> float:
        # Only meaningful at round 3
        if state.round_number != 3:
            self.last_breakdown = {"score": 0.0, "reason": "not_round_3"}
            return 0.0

        # Find R1 result
        r1_result = next((r for r in state.round_results if r["round"] == 1), None)
        if r1_result is None:
            self.last_breakdown = {"score": 0.0, "reason": "no_r1_result"}
            return 0.0

        r1_submission = r1_result.get("submission", {})
        r1_compile = r1_submission.get("compile_status")

        # Floor: R1 must have at least compiled (defeats deliberate-bad-R1 cheating)
        if r1_compile != "success":
            self.last_breakdown = {"score": 0.0, "reason": "r1_did_not_compile",
                                   "r1_compile": r1_compile}
            return 0.0

        r1_speedup = float(r1_submission.get("speedup", 0.0))
        r3_speedup = float(submission.get("speedup", 0.0))

        if r1_speedup <= 0:
            self.last_breakdown = {"score": 0.0, "reason": "r1_speedup_zero"}
            return 0.0

        delta = (r3_speedup - r1_speedup) / r1_speedup
        score = max(0.0, min(1.0, delta))

        self.last_breakdown = {
            "r1_speedup": r1_speedup,
            "r3_speedup": r3_speedup,
            "delta_pct": delta,
            "score": score,
        }
        return score


__all__ = ["SelfCorrectionRubric"]
