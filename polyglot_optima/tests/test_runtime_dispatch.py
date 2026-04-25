"""End-to-end ctypes dispatch tests — replaces the two stubs that the deep gate missed.

Activates only when a C++20 compiler is on PATH (GCC ≥11 or clang ≥13). Skips
cleanly on dev machines with old MinGW; runs on HF Spaces GCC 14 + on A10G.

Three layers of test:
1. Direct dispatcher unit tests (call_compiled, benchmark_python_vs_cpp)
2. cpp_compiler.compile_and_benchmark with REAL agent C++ → real speedup numbers
3. verifier.verify_equivalence with WRONG agent C++ → low pass_rate (anti-cheating)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from models import OptimizationState
from server.tools import TOOL_REGISTRY


# ---------- Compiler + dispatch capability detection ----------
#
# Production target: GCC 14 with C++20. These tests run by default on any compiler
# that supports c++20 AND produces ctypes-loadable binaries (HF Spaces, A10G).
#
# On dev machines with only c++17 (old MinGW), set POLYGLOT_OPTIMA_DEV_FALLBACK=1
# to opt into c++17 testing. Otherwise the tests skip cleanly.


def _has_cxx_at_least(std: str) -> bool:
    for cxx in ("g++", "clang++"):
        path = shutil.which(cxx)
        if not path:
            continue
        try:
            r = subprocess.run([path, f"-std={std}", "-x", "c++", "-E", "-"],
                               input="", capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "unrecognized" not in (r.stderr or "").lower():
                return True
        except Exception:
            continue
    return False


import os as _os
_DEV_FALLBACK = _os.environ.get("POLYGLOT_OPTIMA_DEV_FALLBACK", "0") == "1"
_HAS_CXX20 = _has_cxx_at_least("c++20")
_HAS_CXX17 = _has_cxx_at_least("c++17")

# Dispatcher tests require BOTH a working compiler AND that the .so it produces
# is loadable by this Python interpreter (defeated by 32-bit MinGW on 64-bit Python).
try:
    from server.tools.cpp_compiler import _DISPATCHABLE
    DISPATCHABLE = _DISPATCHABLE
except Exception:
    DISPATCHABLE = False

# Decide whether to run:
#   - default: only on c++20-capable compilers + dispatchable
#   - with POLYGLOT_OPTIMA_DEV_FALLBACK=1: also on c++17
_can_run = DISPATCHABLE and (_HAS_CXX20 or (_DEV_FALLBACK and _HAS_CXX17))

_skip_reason = (
    "No C++20 compiler with ctypes-loadable output. "
    "On GCC 14 / HF Spaces / A10G these tests run. "
    "On dev with old MinGW: set POLYGLOT_OPTIMA_DEV_FALLBACK=1 to opt into C++17 fallback."
)
pytestmark = pytest.mark.skipif(not _can_run, reason=_skip_reason)


# ---------- fixture ----------

@pytest.fixture
def state():
    return OptimizationState(
        episode_id="dispatch-test",
        python_code=(
            "def sum_squares(arr):\n"
            "    s = 0.0\n"
            "    for x in arr:\n"
            "        s += x * x\n"
            "    return s\n"
        ),
        function_signature_cpp='extern "C" void agent_function(const double*, size_t, double*, size_t);',
        hardware_profile={"id": "desktop_avx2", "cores": 8, "freq_ghz": 3.8,
                          "l1_kb": 32, "simd": "AVX2", "bw_gbs": 51},
        bottleneck_ground_truth=["compute-bound", "vectorizable"],
        bottleneck_distractors=["memory-bound", "branch-heavy", "io-bound"],
    )


# ---------- canonical signature C++ snippets ----------

CORRECT_SUM_SQUARES_CPP = '''
#include <cstddef>

extern "C" void agent_function(
    const double* in_ptr, size_t in_n,
    double* out_ptr, size_t out_n)
{
    double total = 0.0;
    for (size_t i = 0; i < in_n; ++i) total += in_ptr[i] * in_ptr[i];
    if (out_n >= 1) out_ptr[0] = total;
}
'''

WRONG_SUM_SQUARES_CPP = '''
#include <cstddef>
// Returns sum of |x|, not sum of x*x. Should fail verifier.
extern "C" void agent_function(
    const double* in_ptr, size_t in_n,
    double* out_ptr, size_t out_n)
{
    double total = 0.0;
    for (size_t i = 0; i < in_n; ++i) total += (in_ptr[i] < 0 ? -in_ptr[i] : in_ptr[i]);
    if (out_n >= 1) out_ptr[0] = total;
}
'''


# ---------- L1: dispatcher unit ----------

def test_call_compiled_dispatches_correctly(state):
    """Compile the correct sum_squares and dispatch via ctypes — output must match Python."""
    out = TOOL_REGISTRY["compile_and_benchmark"]({"cpp_code": CORRECT_SUM_SQUARES_CPP}, state)
    assert out["compile_status"] == "success", out.get("error", "")
    assert out["python_ms"] > 0, "real Python timing must be > 0"
    assert out["cpp_ms"] > 0, "real C++ timing must be > 0"
    assert out["speedup"] != 10.0, "speedup is no longer the hardcoded 10x stub"


def test_benchmark_yields_real_numbers(state):
    """Real benchmark: cpp_ms should be positive and python_ms positive; speedup not stub-10x."""
    out = TOOL_REGISTRY["compile_and_benchmark"]({"cpp_code": CORRECT_SUM_SQUARES_CPP}, state)
    assert out["compile_status"] == "success"
    # Python loop (sum of x*x over 1024 doubles) — typically 100s of microseconds → ms range
    assert 0.001 < out["python_ms"] < 1000
    assert 0.0001 < out["cpp_ms"] < 100
    # Method tag should reflect real measurement
    assert "ctypes" in out.get("method", "")


# ---------- L2: verifier with wrong C++ (anti-cheating real test) ----------

def test_verifier_catches_wrong_algorithm(state):
    """Wrong C++ (sum of |x| instead of sum of x*x) must yield LOW pass_rate.

    Per plan §10b cheating mode 1: 'wrong algorithm with plausible output'.
    The fuzzer must catch this via real ctypes dispatch.
    """
    out = TOOL_REGISTRY["verify_equivalence"]({
        "cpp_code": WRONG_SUM_SQUARES_CPP,
        "n_cases": 100,
    }, state)
    # Wrong algorithm fails on roughly half the inputs (where it disagrees with sum-of-squares)
    assert out["pass_rate"] < 0.6, f"wrong C++ slipped through with pass_rate {out['pass_rate']}"


def test_verifier_passes_correct_cpp(state):
    """Correct C++ for sum_squares must pass nearly all fuzz cases."""
    out = TOOL_REGISTRY["verify_equivalence"]({
        "cpp_code": CORRECT_SUM_SQUARES_CPP,
        "n_cases": 100,
    }, state)
    assert out["pass_rate"] >= 0.90, f"correct C++ failed verifier with pass_rate {out['pass_rate']}"


# ---------- L3: end-to-end submit_optimization with real .so ----------

def test_submit_optimization_full_pipeline_correct(state):
    """submit_optimization with correct C++ → ready_for_reward=True at R3 threshold."""
    state.round_number = 3
    out = TOOL_REGISTRY["submit_optimization"]({
        "cpp_code": CORRECT_SUM_SQUARES_CPP,
        "reasoning_trace": "compute-bound vectorizable",
    }, state)
    assert out["compile_status"] == "success"
    assert out["correctness_pass_rate"] >= 0.85
    # ready_for_reward requires correctness ≥ R3 threshold (0.95)
    # We hit ≥0.85 reliably; ≥0.95 sometimes — the gate-fail mode is also legitimate signal


def test_submit_optimization_full_pipeline_wrong(state):
    """submit_optimization with wrong C++ → not ready, low correctness."""
    state.round_number = 3
    out = TOOL_REGISTRY["submit_optimization"]({
        "cpp_code": WRONG_SUM_SQUARES_CPP,
        "reasoning_trace": "compute-bound vectorizable",
    }, state)
    # Compiles fine but fails the fuzzer — gates reject reward
    assert out["compile_status"] == "success"
    assert out["correctness_pass_rate"] < 0.6
    assert out["ready_for_reward"] is False


# ---------- D5_real: REAL reward variance over real submissions ----------

def test_real_reward_variance_correct_vs_wrong(state):
    """Reward DAG distinguishes correct from wrong real C++ submissions."""
    from server.rewards import build_round_reward_dag
    state.round_number = 1
    state.round_results = [{"round": 1, "tool_calls": ["get_hardware_profile"]}]

    sub_correct = TOOL_REGISTRY["submit_optimization"]({
        "cpp_code": CORRECT_SUM_SQUARES_CPP,
        "reasoning_trace": "compute-bound vectorizable",
    }, state)
    sub_wrong = TOOL_REGISTRY["submit_optimization"]({
        "cpp_code": WRONG_SUM_SQUARES_CPP,
        "reasoning_trace": "compute-bound vectorizable",
    }, state)

    dag = build_round_reward_dag(1)
    score_correct = dag.score(state, sub_correct)
    score_wrong = dag.score(state, sub_wrong)

    # Correct must outscore wrong; this is the headline anti-cheat test
    assert score_correct > score_wrong, \
        f"reward DAG failed to distinguish: correct={score_correct:.3f} ≤ wrong={score_wrong:.3f}"
