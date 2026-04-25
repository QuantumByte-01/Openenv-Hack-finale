"""Adaptive 4-axis difficulty controller (per plan §3 — MAX INNOVATION).

After every 8-rollout batch the controller computes success_rate and adjusts
ONE of four orthogonal axes:

    function_tier:        0..3   (Tier 1..Tier 4 problem complexity)
    hardware_class:       0..2   (easy → hard hardware profiles)
    fuzzer_strictness:    0..2   (n_cases 100→1000, rtol 1e-3→1e-5 + edge cases)
    portability_required: 0..1   (off → must pass on 3+ profiles for any reward)

Logic:
    success ≥ 0.75 → escalate one random axis (the model is too good)
    success ≤ 0.25 → de-escalate the highest axis (the model is stuck)
    0.25 < success < 0.75 → Goldilocks zone, hold (max variance for GRPO)

Why 4-axis adaptation: prior curriculum work (PLR 2021, SPIRAL 2025, Code-A1
2026) escalates a SINGLE difficulty dimension. We escalate four orthogonal
dimensions, giving a much richer adaptation surface and preventing the model
from "specializing" in one axis. This is the central novelty in §2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


MAX_LEVEL = {
    "function_tier": 3,
    "hardware_class": 2,
    "fuzzer_strictness": 2,
    "portability_required": 1,
}

MIN_LEVEL = {axis: 0 for axis in MAX_LEVEL}


@dataclass
class CurriculumSnapshot:
    """A point-in-time view of the axes + recent batch stats — for wandb logging."""
    axes: dict[str, int]
    success_rate: float
    n_batches_seen: int
    last_action: str = ""           # "escalate function_tier", "de-escalate hardware_class", "hold"
    n_escalations: dict[str, int] = field(default_factory=dict)
    n_deescalations: dict[str, int] = field(default_factory=dict)


class AdaptiveCurriculum:
    """Controller that mutates difficulty axes based on batch success rates.

    Use:
        curriculum = AdaptiveCurriculum()
        for batch_idx in range(n_batches):
            # rollout 8 episodes using curriculum.axes
            # ...
            success_rate = compiles_and_passes / 8
            curriculum.observe_batch(success_rate)
            snapshot = curriculum.snapshot()
            wandb.log({"curriculum/axes": snapshot.axes, ...})
    """

    HIGH_THRESHOLD = 0.75
    LOW_THRESHOLD = 0.25

    def __init__(
        self,
        initial_axes: dict[str, int] | None = None,
        seed: int | None = None,
        min_level: dict[str, int] | None = None,
        max_level: dict[str, int] | None = None,
    ):
        self.axes = dict(initial_axes or {axis: 0 for axis in MAX_LEVEL})
        self.min_level = dict(min_level or MIN_LEVEL)
        self.max_level = dict(max_level or MAX_LEVEL)
        self.rng = random.Random(seed)
        self.n_batches_seen = 0
        self.last_action = "init"
        self.n_escalations = {axis: 0 for axis in MAX_LEVEL}
        self.n_deescalations = {axis: 0 for axis in MAX_LEVEL}
        self._recent_success = 0.0  # last observed batch success_rate

    def observe_batch(self, success_rate: float) -> str:
        """Process one batch result. Returns the action taken as a human-readable string."""
        self.n_batches_seen += 1
        self._recent_success = float(success_rate)

        if success_rate >= self.HIGH_THRESHOLD:
            action = self._escalate()
        elif success_rate <= self.LOW_THRESHOLD:
            action = self._deescalate()
        else:
            action = "hold (Goldilocks zone)"

        self.last_action = action
        return action

    def _escalate(self) -> str:
        """Pick a random axis (uniform over those still below max) and increment it."""
        candidates = [a for a, v in self.axes.items() if v < self.max_level[a]]
        if not candidates:
            return "hold (all axes at max)"
        axis = self.rng.choice(candidates)
        self.axes[axis] = min(self.axes[axis] + 1, self.max_level[axis])
        self.n_escalations[axis] += 1
        return f"escalate {axis} → {self.axes[axis]}"

    def _deescalate(self) -> str:
        """De-escalate the axis currently at the highest level (break ties randomly)."""
        candidates = [a for a, v in self.axes.items() if v > self.min_level[a]]
        if not candidates:
            return "hold (all axes at min)"
        max_value = max(self.axes[a] for a in candidates)
        top = [a for a in candidates if self.axes[a] == max_value]
        axis = self.rng.choice(top)
        self.axes[axis] = max(self.axes[axis] - 1, self.min_level[axis])
        self.n_deescalations[axis] += 1
        return f"de-escalate {axis} → {self.axes[axis]}"

    def snapshot(self) -> CurriculumSnapshot:
        return CurriculumSnapshot(
            axes=dict(self.axes),
            success_rate=self._recent_success,
            n_batches_seen=self.n_batches_seen,
            last_action=self.last_action,
            n_escalations=dict(self.n_escalations),
            n_deescalations=dict(self.n_deescalations),
        )

    def to_dict(self) -> dict[str, Any]:
        s = self.snapshot()
        return {
            "axes": s.axes,
            "success_rate": s.success_rate,
            "n_batches_seen": s.n_batches_seen,
            "last_action": s.last_action,
            "n_escalations": s.n_escalations,
            "n_deescalations": s.n_deescalations,
        }


__all__ = [
    "AdaptiveCurriculum",
    "CurriculumSnapshot",
    "MAX_LEVEL",
    "MIN_LEVEL",
]
