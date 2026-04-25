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
    """Three-zone graduated gate. Replaces the hard binary cliff with continuous reward.

    Zones (per plan §3 "continuous reward, not binary"):
        score < dead_floor:           reward = 0  (anti-cheat — random/wrong code earns nothing)
        dead_floor <= score < threshold:
                                      partial credit, linear ramp 0 -> ramp_max * downstream
        score >= threshold:           full pass-through (downstream gets unmodified score)

    The "raise GateFailedError" behavior is preserved ONLY for the dead-floor zone.
    Sequential catches it and returns 0 for the whole DAG. Above the floor, even a
    sub-threshold submission gets partial reward — GRPO has gradient to climb.

    `hard=True` restores the original binary-cliff behavior (used for CompilationRubric:
    you either compiled or you didn't, no partial credit for "almost compiled").
    """

    def __init__(self, child: Rubric, threshold: float, dead_floor: float = 0.3,
                 ramp_max: float = 0.3, hard: bool = False):
        self.child = child
        self.threshold = threshold
        self.dead_floor = dead_floor
        self.ramp_max = ramp_max
        self.hard = hard
        self.name = f"gate({child.name}>={threshold:.2f})"

    def score(self, state, submission: dict[str, Any]) -> float:
        """Returns a MULTIPLIER ∈ [0, 1] for Sequential to multiply the final score by.

        Hard mode:
            score >= threshold → 1.0  (full pass-through multiplier)
            score < threshold  → raises GateFailedError (Sequential→0)
        Graduated mode:
            score < dead_floor → raises (Sequential→0; anti-cheat preserved)
            dead_floor ≤ score < threshold → ramp_max * progress  (continuous, ∈ (0, ramp_max])
            score >= threshold → 1.0  (full multiplier)
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

        # Graduated
        if s < self.dead_floor:
            self.last_breakdown = {"child": s, "threshold": self.threshold, "zone": "dead"}
            raise GateFailedError(
                f"{self.child.name} = {s:.3f} < dead_floor={self.dead_floor}"
            )

        if s >= self.threshold:
            self.last_breakdown = {"child": s, "threshold": self.threshold, "zone": "full"}
            return 1.0

        # Ramp: linear in [dead_floor, threshold) → multiplier in (0, ramp_max]
        span = self.threshold - self.dead_floor
        progress = (s - self.dead_floor) / max(span, 1e-9)
        multiplier = self.ramp_max * progress

        self.last_breakdown = {
            "child": s, "threshold": self.threshold,
            "zone": "ramp", "progress": progress, "multiplier": multiplier,
        }
        return multiplier


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
