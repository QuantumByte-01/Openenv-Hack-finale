"""Hour 10-16: Reward rubric tests.

Validates:
- Sequential short-circuits to 0 on Gate failure
- Gate raises GateFailedError below threshold
- WeightedSum composes correctly
- SpeedupRubric is Roofline-normalized (capped at 1.0)
- CorrectnessRubric penalizes adversarial-pool failures
- DiagnosisRubric:
    - rewards correct keywords
    - penalizes distractor stuffing
    - applies length penalty
    - awards coherence bonus when first tool matches diagnosis
- PortabilityRubric only counts when axis is on
- SelfCorrectionRubric requires R1 to compile (anti-gaming floor)
- Full DAG: R1 vs R3 weighting works end-to-end
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from models import OptimizationState
from server.rewards import (
    Sequential, Gate, WeightedSum, GateFailedError,
    SpeedupRubric, CorrectnessRubric, CompilationRubric,
    DiagnosisRubric, PortabilityRubric, SelfCorrectionRubric,
    build_round_reward_dag,
)


def make_state(**overrides):
    s = OptimizationState(
        episode_id="test",
        python_code="def sum_squares(arr):\n    total = 0.0\n    for x in arr:\n        total += x*x\n    return total\n",
        function_signature_cpp='extern "C" double agent_function(const double*, size_t);',
        hardware_profile={
            "id": "desktop_avx2", "cores": 8, "freq_ghz": 3.8, "l1_kb": 32,
            "simd": "AVX2", "bw_gbs": 51,
        },
        bottleneck_ground_truth=["compute-bound", "vectorizable"],
        bottleneck_distractors=["memory-bound", "branch-heavy", "io-bound"],
        round_number=1,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------- Composers ----------

def test_sequential_passes_through_last():
    state = make_state()
    sub = {"compile_status": "success", "correctness_pass_rate": 0.9, "adversarial_pass_rate": 0.95, "speedup": 5.0}
    seq = Sequential(CorrectnessRubric())
    assert seq.score(state, sub) == pytest.approx(0.9, abs=1e-3)


def test_sequential_short_circuits_on_gate():
    state = make_state()
    sub = {"compile_status": "success", "correctness_pass_rate": 0.4, "adversarial_pass_rate": 0.95, "speedup": 5.0}
    seq = Sequential(Gate(CorrectnessRubric(), threshold=0.6), CorrectnessRubric())
    assert seq.score(state, sub) == 0.0


def test_gate_raises_below_threshold():
    state = make_state()
    sub = {"correctness_pass_rate": 0.3, "adversarial_pass_rate": 0.95}
    g = Gate(CorrectnessRubric(), threshold=0.6)
    with pytest.raises(GateFailedError):
        g.score(state, sub)


def test_gate_passes_through_above_threshold():
    state = make_state()
    sub = {"correctness_pass_rate": 0.85, "adversarial_pass_rate": 0.95}
    g = Gate(CorrectnessRubric(), threshold=0.6)
    assert g.score(state, sub) == pytest.approx(0.85)


def test_weighted_sum_composes():
    state = make_state()
    sub = {"speedup": 5.0, "correctness_pass_rate": 1.0, "adversarial_pass_rate": 1.0}
    ws = WeightedSum(
        {"speedup": SpeedupRubric(), "correctness": CorrectnessRubric()},
        weights={"speedup": 0.5, "correctness": 0.5},
    )
    score = ws.score(state, sub)
    assert 0.0 <= score <= 1.0


# ---------- SpeedupRubric (Roofline) ----------

def test_speedup_zero_yields_zero():
    s = SpeedupRubric().score(make_state(), {"speedup": 0.0})
    assert s == 0.0


def test_speedup_at_roofline_yields_max():
    """speedup == roofline_peak should yield ~1.0 reward (LOG_NORM = 1.0)."""
    state = make_state()
    from server.tools.hardware_profiler import roofline_bound
    peak = roofline_bound(state.hardware_profile)
    score = SpeedupRubric().score(state, {"speedup": peak})
    assert 0.99 <= score <= 1.0


def test_speedup_modest_yields_modest_reward():
    """A modest 5x speedup on AVX2 (peak ~25 GFLOPS) → low-but-positive reward."""
    score = SpeedupRubric().score(make_state(), {"speedup": 5.0})
    assert 0.05 < score < 0.5


# ---------- CorrectnessRubric ----------

def test_correctness_returns_pass_rate():
    s = CorrectnessRubric().score(make_state(),
        {"correctness_pass_rate": 0.92, "adversarial_pass_rate": 0.95})
    assert s == pytest.approx(0.92)


def test_correctness_penalizes_adversarial_failures():
    """Adversarial pass rate < 0.9 → halves the score per plan §10b."""
    s = CorrectnessRubric().score(make_state(),
        {"correctness_pass_rate": 0.92, "adversarial_pass_rate": 0.5})
    assert s == pytest.approx(0.46, abs=1e-3)


def test_compilation_rubric_binary():
    assert CompilationRubric().score(make_state(), {"compile_status": "success"}) == 1.0
    assert CompilationRubric().score(make_state(), {"compile_status": "syntax_error"}) == 0.0


# ---------- DiagnosisRubric ----------

def test_diagnosis_rewards_correct_keywords():
    state = make_state()
    state.round_results = [{"round": 1, "tool_calls": ["get_hardware_profile"]}]
    s = DiagnosisRubric().score(state,
        {"reasoning_trace": "<think>this is compute-bound and vectorizable</think>"})
    assert s > 0.5


def test_diagnosis_penalizes_distractor_stuffing():
    state = make_state()
    state.round_results = [{"round": 1, "tool_calls": ["get_hardware_profile"]}]
    s_clean = DiagnosisRubric().score(state,
        {"reasoning_trace": "compute-bound vectorizable"})
    s_stuffed = DiagnosisRubric().score(state,
        {"reasoning_trace": "compute-bound vectorizable memory-bound branch-heavy io-bound"})
    assert s_stuffed < s_clean


def test_diagnosis_length_penalty():
    state = make_state()
    state.round_results = [{"round": 1, "tool_calls": ["get_hardware_profile"]}]
    short = DiagnosisRubric().score(state, {"reasoning_trace": "compute-bound vectorizable"})
    long_text = "compute-bound vectorizable " + ("filler " * 100)
    long_ = DiagnosisRubric().score(state, {"reasoning_trace": long_text})
    assert long_ < short


def test_diagnosis_coherence_bonus():
    """First tool call matching the diagnosis category gives +0.2 bonus."""
    state = make_state(
        bottleneck_ground_truth=["memory-bound"],
        # Distractors must NOT contain memory-bound, else keyword overlap inflates raw score
        bottleneck_distractors=["branch-heavy", "io-bound"],
    )
    state.round_results = [{"round": 1, "tool_calls": ["check_memory_access"]}]
    matched = DiagnosisRubric().score(state, {"reasoning_trace": "memory-bound"})
    state.round_results = [{"round": 1, "tool_calls": ["analyze_complexity"]}]
    no_match = DiagnosisRubric().score(state, {"reasoning_trace": "memory-bound"})
    assert matched > no_match
    # Bonus is 0.2; clamping to 1.0 may compress the delta slightly
    assert (matched - no_match) == pytest.approx(0.2, abs=0.05) or matched == 1.0


# ---------- PortabilityRubric ----------

def test_portability_rubric_off_axis_returns_zero():
    state = make_state()
    state.difficulty_axes["portability_required"] = 0  # off
    s = PortabilityRubric().score(state, {"portability": {"n_profiles_passing": 5}})
    assert s == 0.0


def test_portability_rubric_on_axis_below_threshold_zero():
    state = make_state()
    state.difficulty_axes["portability_required"] = 1
    s = PortabilityRubric().score(state, {"portability": {"n_profiles_passing": 2}})
    assert s == 0.0


def test_portability_rubric_on_axis_above_threshold_positive():
    state = make_state()
    state.difficulty_axes["portability_required"] = 1
    s = PortabilityRubric().score(state, {"portability": {"n_profiles_passing": 5}})
    assert 0 < s <= 1.0


# ---------- SelfCorrectionRubric ----------

def test_self_correction_only_at_round_3():
    state = make_state(round_number=2)
    s = SelfCorrectionRubric().score(state, {"speedup": 10.0})
    assert s == 0.0


def test_self_correction_floor_r1_must_compile():
    """If R1 didn't compile, R3 self-correction returns 0 (defeats deliberate-bad-R1)."""
    state = make_state(round_number=3)
    state.round_results = [
        {"round": 1, "submission": {"compile_status": "syntax_error", "speedup": 0.0}},
        {"round": 2, "submission": {"compile_status": "success", "speedup": 5.0}},
    ]
    s = SelfCorrectionRubric().score(state, {"speedup": 50.0})
    assert s == 0.0


def test_self_correction_rewards_improvement():
    state = make_state(round_number=3)
    state.round_results = [
        {"round": 1, "submission": {"compile_status": "success", "speedup": 2.0}},
        {"round": 2, "submission": {"compile_status": "success", "speedup": 4.0}},
    ]
    s = SelfCorrectionRubric().score(state, {"speedup": 4.0})  # 100% improvement
    assert s == pytest.approx(1.0, abs=0.01)


# ---------- Full DAG ----------

def test_round1_dag_compile_fail_returns_zero():
    state = make_state(round_number=1)
    sub = {"compile_status": "syntax_error", "correctness_pass_rate": 0.0, "speedup": 0.0,
           "adversarial_pass_rate": 0.0}
    dag = build_round_reward_dag(1)
    assert dag.score(state, sub) == 0.0


def test_round1_dag_correct_below_60_returns_zero():
    state = make_state(round_number=1)
    sub = {"compile_status": "success", "correctness_pass_rate": 0.5,
           "adversarial_pass_rate": 0.95, "speedup": 5.0,
           "reasoning_trace": "compute-bound"}
    dag = build_round_reward_dag(1)
    assert dag.score(state, sub) == 0.0


def test_round1_dag_full_pass_yields_positive():
    state = make_state(round_number=1)
    state.round_results = [{"round": 1, "tool_calls": ["get_hardware_profile"]}]
    sub = {"compile_status": "success", "correctness_pass_rate": 0.95,
           "adversarial_pass_rate": 0.95, "speedup": 8.0,
           "reasoning_trace": "compute-bound vectorizable"}
    dag = build_round_reward_dag(1)
    score = dag.score(state, sub)
    assert 0.3 < score < 1.0


def test_round3_strict_gate_rejects_70_percent_correct():
    """Round 3 demands ≥95% correctness — 70% must fail the gate."""
    state = make_state(round_number=3)
    sub = {"compile_status": "success", "correctness_pass_rate": 0.7,
           "adversarial_pass_rate": 0.95, "speedup": 10.0,
           "reasoning_trace": "compute-bound"}
    dag = build_round_reward_dag(3)
    assert dag.score(state, sub) == 0.0


def test_round3_dag_full_pass_yields_positive():
    state = make_state(round_number=3)
    state.round_results = [
        {"round": 1, "submission": {"compile_status": "success", "speedup": 3.0},
         "tool_calls": ["get_hardware_profile"]},
        {"round": 2, "submission": {"compile_status": "success", "speedup": 6.0},
         "tool_calls": []},
    ]
    sub = {"compile_status": "success", "correctness_pass_rate": 0.97,
           "adversarial_pass_rate": 0.95, "speedup": 9.0,
           "reasoning_trace": "compute-bound vectorizable",
           "portability": {"n_profiles_passing": 4}}
    dag = build_round_reward_dag(3)
    score = dag.score(state, sub)
    assert 0.3 < score < 1.0
