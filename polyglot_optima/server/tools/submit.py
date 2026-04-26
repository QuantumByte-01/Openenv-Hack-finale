"""Tool 9/9: submit_optimization — closes the current round.

This is the only round-closing tool. The environment recognizes its name and:
1. Triggers full-strength verification (n_cases=1000)
2. Triggers portability check (cross-profile compile + correctness)
3. Computes the round's reward via the rubric DAG
4. Stores the submission as the round result

The agent must call this exactly once per round. After 3 calls the episode terminates.
"""

from __future__ import annotations

from typing import Any

from server.tools.cpp_compiler import compile_and_benchmark_tool
from server.tools.verifier import verify_equivalence_tool
from server.tools.portability_checker import check_portability_tool


def submit_optimization_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Final submission for this round. Runs full verifier + portability + benchmark.

    Args:
        cpp_code (str)             — required
        reasoning_trace (str)      — agent's overall <think> trace for this round

    Returns:
        compile_status (str)
        speedup (float)
        correctness_pass_rate (float)
        adversarial_pass_rate (float)
        portability (dict)
        n_profiles_passing (int)
        ready_for_reward (bool)    — True iff hard gates pass; informs the rubric
        cpp_code (str)             — echoed for the round_results history
        reasoning_trace (str)      — echoed
    """
    cpp_code = tool_args.get("cpp_code", "")
    reasoning_trace = tool_args.get("reasoning_trace", state.current_round_reasoning)

    if not cpp_code.strip():
        return {
            "compile_status": "syntax_error",
            "error": "empty cpp_code",
            "speedup": 0.0,
            "correctness_pass_rate": 0.0,
            "ready_for_reward": False,
            "cpp_code": "",
            "reasoning_trace": reasoning_trace,
        }

    # Step 1: compile + benchmark
    bench = compile_and_benchmark_tool({"cpp_code": cpp_code}, state)
    if bench["compile_status"] != "success":
        return {
            "compile_status": bench["compile_status"],
            "error": bench.get("error", ""),
            "speedup": 0.0,
            "correctness_pass_rate": 0.0,
            "adversarial_pass_rate": 0.0,
            "portability": {"n_profiles_passing": 0, "portability_bonus_eligible": False},
            "ready_for_reward": False,
            "cpp_code": cpp_code,
            "reasoning_trace": reasoning_trace,
        }

    # Step 2: full 1000-case verifier (or whatever n_cases the curriculum specifies)
    n_cases = 1000 if state.difficulty_axes.get("fuzzer_strictness", 0) >= 2 else 500
    verifier_result = verify_equivalence_tool(
        {"cpp_code": cpp_code, "n_cases": n_cases},
        state,
    )

    # Step 3: portability check (only if axis is on; informational otherwise)
    portability_result = check_portability_tool({"cpp_code": cpp_code, "n_cases_per_profile": 50}, state)

    # Update episode-best speedup tracker
    if bench["speedup"] > state.best_speedup:
        state.best_speedup = bench["speedup"]
        state.best_cpp_code = cpp_code

    # Round-aware readiness score (continuous) + boolean convenience flag
    round_thresholds = {1: 0.6, 2: 0.8, 3: 0.95}
    threshold = round_thresholds.get(state.round_number, 0.6)
    correctness_ratio = verifier_result["pass_rate"] / max(threshold, 1e-9)
    adversarial_ratio = verifier_result.get("adversarial_pass_rate", 0.0) / 0.9
    compile_quality = 1.0 if bench["compile_status"] == "success" else 0.0
    readiness_score = (
        0.55 * min(1.0, correctness_ratio)
        + 0.30 * min(1.0, adversarial_ratio)
        + 0.15 * compile_quality
    )
    ready = readiness_score >= 0.9

    return {
        "compile_status": bench["compile_status"],
        "speedup": bench["speedup"],
        "python_ms": bench.get("python_ms"),
        "cpp_ms": bench.get("cpp_ms"),
        "correctness_pass_rate": verifier_result["pass_rate"],
        "adversarial_pass_rate": verifier_result.get("adversarial_pass_rate", 0.0),
        "first_correctness_failure": verifier_result.get("first_failure"),
        "portability": portability_result,
        "n_profiles_passing": portability_result.get("n_profiles_passing", 0),
        "readiness_score": readiness_score,
        "ready_for_reward": ready,
        "cpp_code": cpp_code,
        "reasoning_trace": reasoning_trace,
        "round_threshold_correctness": threshold,
    }


__all__ = ["submit_optimization_tool"]
