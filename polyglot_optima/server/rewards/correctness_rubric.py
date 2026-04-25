"""CorrectnessRubric + CompilationRubric (binary).

CorrectnessRubric returns the fuzzer pass_rate directly (∈ [0,1]). Used both as
a Gate target and as a weighted component.

CompilationRubric is binary: 1.0 if compile succeeded, 0.0 otherwise. Used only
as a Gate (a compile failure is a hard reward = 0).
"""

from __future__ import annotations

from typing import Any

from .rubrics import Rubric


class CorrectnessRubric(Rubric):
    name = "correctness"

    def score(self, state, submission: dict[str, Any]) -> float:
        pass_rate = float(submission.get("correctness_pass_rate", 0.0))
        adv_pass_rate = float(submission.get("adversarial_pass_rate", 0.0))

        # Hard penalty if adversarial sub-pool is below 0.9 (per plan §10b)
        if adv_pass_rate < 0.9:
            penalty = 0.5  # halve the score if adversarial cases are failing
            pass_rate *= penalty

        self.last_breakdown = {
            "raw_pass_rate": float(submission.get("correctness_pass_rate", 0.0)),
            "adversarial_pass_rate": adv_pass_rate,
            "adversarial_penalty_applied": adv_pass_rate < 0.9,
            "score": pass_rate,
        }
        return max(0.0, min(1.0, pass_rate))


class CompilationRubric(Rubric):
    """1.0 if the submission compiled; 0.0 otherwise."""

    name = "compilation"

    def score(self, state, submission: dict[str, Any]) -> float:
        compile_status = submission.get("compile_status", "pending")
        ok = compile_status == "success"
        self.last_breakdown = {"compile_status": compile_status, "score": 1.0 if ok else 0.0}
        return 1.0 if ok else 0.0


__all__ = ["CorrectnessRubric", "CompilationRubric"]
