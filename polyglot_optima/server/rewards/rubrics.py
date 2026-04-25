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
    """Run rubrics in order. If any raises GateFailedError → return 0.0.

    The last rubric's score is the final output. Earlier rubrics typically
    do gating only.
    """

    name = "sequential"

    def __init__(self, *children: Rubric):
        if not children:
            raise ValueError("Sequential needs at least one child rubric")
        self.children = children

    def score(self, state, submission: dict[str, Any]) -> float:
        last_score = 0.0
        breakdown: dict[str, Any] = {}
        for child in self.children:
            try:
                last_score = child.score(state, submission)
                breakdown[child.name] = last_score
            except GateFailedError as e:
                breakdown[child.name] = 0.0
                breakdown["_gate_failed"] = str(e)
                self.last_breakdown = breakdown
                return 0.0
        self.last_breakdown = breakdown
        return last_score


class Gate(Rubric):
    """Wrap a child rubric. Raise GateFailedError if score < threshold; else pass through."""

    def __init__(self, child: Rubric, threshold: float):
        self.child = child
        self.threshold = threshold
        self.name = f"gate({child.name}≥{threshold:.2f})"

    def score(self, state, submission: dict[str, Any]) -> float:
        s = self.child.score(state, submission)
        self.last_breakdown = {"child": s, "threshold": self.threshold, "passed": s >= self.threshold}
        if s < self.threshold:
            raise GateFailedError(f"{self.child.name} = {s:.3f} < {self.threshold}")
        return s


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
