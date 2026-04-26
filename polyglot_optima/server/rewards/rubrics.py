"""Base Rubric class + 3 composers (Sequential, Gate, WeightedSum).

These mirror OpenEnv's documented rubric primitives. Only Sequential, Gate, and
WeightedSum are confirmed in the framework — MaxOf/MinOf/Conditional were
*removed* from the plan in §12 D because they are not in upstream OpenEnv.

A Rubric is a callable: rubric.score(state, submission) -> float in [0, 1].
Rubric subclasses also expose .name (str) and may expose per-call breakdown
via the .last_breakdown dict (used by named_rubrics() introspection).
"""

from __future__ import annotations

from typing import Any, Mapping


class GateFailedError(Exception):
    """Raised by Gate when its child rubric is below threshold.

    Sequential catches this and short-circuits to 0.0.
    """


class Rubric:
    """Base class — concrete subclasses must override score()."""

    name: str = "rubric"

    def score(self, state, submission: dict[str, Any]) -> float:
        raise NotImplementedError("subclass must implement .score()")

    # Optional debug — populated by score() for introspection
    last_breakdown: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# -------------------------- Composers --------------------------

class Sequential(Rubric):
    """Run rubrics in order. Returns (product of Gate multipliers) × (last non-Gate child).

    Each `Gate` child yields a multiplier ∈ [0, 1]:
        hard pass         → 1.0
        hard fail         → raises (Sequential returns 0)
        graduated full    → 1.0
        graduated ramp    → fractional in (0, 1)
        graduated dead    → raises (Sequential returns 0)

    Non-Gate children produce the actual reward score. Sequential outputs
    the final score scaled by the product of gate multipliers — giving GRPO
    a continuous gradient even when the agent is below threshold (per plan §3).
    """

    name = "sequential"

    def __init__(self, *children: Rubric):
        if not children:
            raise ValueError("Sequential needs at least one child rubric")
        self.children = children

    def score(self, state, submission: dict[str, Any]) -> float:
        gate_product = 1.0
        final_score: float | None = None
        breakdown: dict[str, Any] = {}
        for child in self.children:
            try:
                s = child.score(state, submission)
                breakdown[child.name] = s
            except GateFailedError as e:
                breakdown[child.name] = 0.0
                breakdown["_gate_failed"] = str(e)
                self.last_breakdown = breakdown
                return 0.0
            if isinstance(child, Gate):
                gate_product *= s
            else:
                final_score = s

        breakdown["_gate_product"] = gate_product
        breakdown["_final_score"] = final_score if final_score is not None else gate_product
        self.last_breakdown = breakdown

        if final_score is None:
            return gate_product
        return float(max(0.0, min(1.0, gate_product * final_score)))


class Gate(Rubric):
    """Continuous gate multiplier for shaping reward without binary cliffs.

    In default mode, this gate never raises and always returns a multiplier in
    [ramp_min, 1.0], where `ramp_min` is small but non-zero. That preserves
    gradient signal even for weak submissions.

    `hard=True` is kept only for backward compatibility.
    """

    def __init__(self, child: Rubric, threshold: float, dead_floor: float = 0.0,
                 ramp_max: float = 1.0, hard: bool = False, ramp_min: float = 0.05,
                 exponent: float = 2.0):
        self.child = child
        self.threshold = threshold
        self.dead_floor = dead_floor
        self.ramp_max = ramp_max
        self.hard = hard
        self.ramp_min = ramp_min
        self.exponent = exponent
        self.name = f"gate({child.name}>={threshold:.2f})"

    def score(self, state, submission: dict[str, Any]) -> float:
        """Returns a MULTIPLIER ∈ [0, 1] for Sequential to multiply the final score by.

        Hard mode:
            score >= threshold → 1.0
            score < threshold  → raises GateFailedError
        Continuous mode:
            score >= threshold → 1.0
            score < threshold  → smooth multiplier in [ramp_min, ramp_max]
        """
        s = self.child.score(state, submission)

        if self.hard:
            self.last_breakdown = {
                "child": s, "threshold": self.threshold,
                "zone": "hard_pass" if s >= self.threshold else "hard_fail",
            }
            if s < self.threshold:
                raise GateFailedError(f"{self.child.name} = {s:.3f} < {self.threshold} (hard)")
            return 1.0

        if s >= self.threshold:
            self.last_breakdown = {"child": s, "threshold": self.threshold, "zone": "full"}
            return 1.0

        # Smooth ramp in [0, threshold) with non-zero floor.
        normalized = max(0.0, s) / max(self.threshold, 1e-9)
        progress = max(0.0, min(1.0, normalized)) ** self.exponent
        multiplier = self.ramp_min + (self.ramp_max - self.ramp_min) * progress

        self.last_breakdown = {
            "child": s, "threshold": self.threshold,
            "zone": "ramp", "progress": progress, "multiplier": multiplier,
        }
        return float(max(0.0, min(1.0, multiplier)))


class WeightedSum(Rubric):
    """Sum of children weighted. weights must be a dict matching children keys.

    children: Mapping[str, Rubric] — name → rubric
    weights:  Mapping[str, float]   — name → weight (need not sum to 1; we DO NOT normalize)
    """

    name = "weighted_sum"

    def __init__(self, children: Mapping[str, Rubric], weights: Mapping[str, float]):
        if set(children.keys()) != set(weights.keys()):
            raise ValueError(
                f"children keys {set(children.keys())} != weights keys {set(weights.keys())}"
            )
        self.children = dict(children)
        self.weights = dict(weights)

    def score(self, state, submission: dict[str, Any]) -> float:
        breakdown: dict[str, Any] = {}
        total = 0.0
        for name, rubric in self.children.items():
            child_score = rubric.score(state, submission)
            breakdown[name] = {"score": child_score, "weight": self.weights[name]}
            total += child_score * self.weights[name]
        self.last_breakdown = breakdown
        # Clamp to [0, 1]; weights nominally sum to 1 but we don't enforce
        return float(max(0.0, min(1.0, total)))


__all__ = [
    "Rubric",
    "Sequential",
    "Gate",
    "WeightedSum",
    "GateFailedError",
]
