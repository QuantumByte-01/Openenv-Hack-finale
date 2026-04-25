"""Composable reward rubric system for Polyglot-Optima.

Per plan §12 D, this is a 4-level composition tree using only the OpenEnv
documented primitives (Sequential, Gate, WeightedSum) plus 5 custom Rubric
subclasses (Speedup, Correctness, Compilation, Diagnosis, Portability,
SelfCorrection).

The composition tree (per plan §10):

    round1_reward = Sequential(
        Gate(CorrectnessRubric, threshold=0.6),
        Gate(CompilationRubric, threshold=1.0),
        WeightedSum(
            SpeedupRubric         w=0.40
            CorrectnessRubric     w=0.30
            DiagnosisRubric       w=0.30
        )
    )

    round3_reward = Sequential(
        Gate(CorrectnessRubric, threshold=0.95),
        Gate(CompilationRubric, threshold=1.0),
        WeightedSum(
            SpeedupRubric         w=0.35
            CorrectnessRubric     w=0.25
            DiagnosisRubric       w=0.20
            SelfCorrectionRubric  w=0.10
            PortabilityRubric     w=0.10  (only counts if portability_required axis on)
        )
    )

    episode_reward = 0.3 * round1_reward + 0.7 * round3_reward
"""

from __future__ import annotations

from .rubrics import (
    Rubric,
    Sequential,
    Gate,
    WeightedSum,
    GateFailedError,
)
from .speedup_rubric import SpeedupRubric
from .correctness_rubric import CorrectnessRubric, CompilationRubric
from .diagnosis_rubric import DiagnosisRubric
from .portability_rubric import PortabilityRubric
from .self_correction_rubric import SelfCorrectionRubric


def build_round_reward_dag(round_number: int):
    """Construct the reward DAG appropriate for a given round (1, 2, or 3).

    Round 1: soft gate (60%), 3 components (Speedup, Correctness, Diagnosis)
    Round 2: medium gate (80%), same 3 components (informational)
    Round 3: strict gate (95%), 5 components (adds SelfCorrection + Portability)
    """
    correctness = CorrectnessRubric()
    compilation = CompilationRubric()

    # CompilationRubric uses HARD gate — you either compiled or you didn't.
    # CorrectnessRubric uses GRADUATED gate — partial credit between dead_floor (0.3)
    # and the round's threshold, full credit above. Preserves anti-cheat (<0.3 = 0)
    # while providing GRPO with a continuous gradient signal (per plan §3 "not binary").
    if round_number == 1:
        return Sequential(
            Gate(correctness, threshold=0.6, dead_floor=0.3, ramp_max=0.4),
            Gate(compilation, threshold=1.0, hard=True),
            WeightedSum(
                {"speedup": SpeedupRubric(),
                 "correctness": correctness,
                 "diagnosis": DiagnosisRubric()},
                weights={"speedup": 0.40, "correctness": 0.30, "diagnosis": 0.30},
            ),
        )

    if round_number == 2:
        return Sequential(
            Gate(correctness, threshold=0.80, dead_floor=0.3, ramp_max=0.35),
            Gate(compilation, threshold=1.0, hard=True),
            WeightedSum(
                {"speedup": SpeedupRubric(),
                 "correctness": correctness,
                 "diagnosis": DiagnosisRubric()},
                weights={"speedup": 0.40, "correctness": 0.30, "diagnosis": 0.30},
            ),
        )

    # Round 3 — strict gate (95%), full 5 components
    return Sequential(
        Gate(correctness, threshold=0.95, dead_floor=0.3, ramp_max=0.30),
        Gate(compilation, threshold=1.0, hard=True),
        WeightedSum(
            {"speedup": SpeedupRubric(),
             "correctness": correctness,
             "diagnosis": DiagnosisRubric(),
             "self_correction": SelfCorrectionRubric(),
             "portability": PortabilityRubric()},
            weights={"speedup": 0.35, "correctness": 0.25,
                     "diagnosis": 0.20, "self_correction": 0.10, "portability": 0.10},
        ),
    )


__all__ = [
    "Rubric",
    "Sequential",
    "Gate",
    "WeightedSum",
    "GateFailedError",
    "SpeedupRubric",
    "CorrectnessRubric",
    "CompilationRubric",
    "DiagnosisRubric",
    "PortabilityRubric",
    "SelfCorrectionRubric",
    "build_round_reward_dag",
]
