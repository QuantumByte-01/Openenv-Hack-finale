"""Hour 4-10: Tool unit tests.

Each of the 9 MCP tools verified for shape + key invariants. Compiler-dependent
tests (cpp_compiler, verifier, portability) are gated on g++ being installed —
they skip cleanly if the toolchain is unavailable.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from models import OptimizationState
from server.tools import TOOL_REGISTRY


HAS_GPP = shutil.which("g++") is not None or shutil.which("clang++") is not None


def _has_cxx20() -> bool:
    """True only if a C++20-capable compiler is on PATH (GCC ≥ 11 / clang ≥ 13).

    Dev machines (e.g. ancient MinGW on Windows) often have g++ but not C++20,
    so the cpp_compiler test skips cleanly there. The HF Spaces Docker container
    pins GCC 14, so this passes in CI/deploy.
    """
    import subprocess
    for cxx in ("g++", "clang++"):
        path = shutil.which(cxx)
        if not path:
            continue
        try:
            r = subprocess.run([path, "-std=c++20", "-x", "c++", "-E", "-"],
                               input="", capture_output=True, text=True, timeout=5)
            if r.returncode == 0 or "unrecognized" not in (r.stderr or "").lower():
                return True
        except Exception:
            continue
    return False


HAS_CXX20 = _has_cxx20()


# ----------- common fixture -----------

@pytest.fixture
def state():
    """A representative OptimizationState the tools accept."""
    return OptimizationState(
        episode_id="test-ep",
        python_code="def sum_squares(arr):\n    total = 0.0\n    for x in arr:\n        total += x*x\n    return total\n",
        function_signature_cpp='extern "C" double agent_function(const double*, size_t);',
        hardware_profile={
            "id": "desktop_avx2",
            "cores": 8, "freq_ghz": 3.8, "l1_kb": 32,
            "simd": "AVX2", "bw_gbs": 51,
        },
        bottleneck_ground_truth=["compute-bound", "vectorizable"],
        bottleneck_distractors=["memory-bound", "branch-heavy", "io-bound"],
    )


# ----------- Tool 1: hardware_profiler -----------

def test_get_hardware_profile_returns_roofline(state):
    out = TOOL_REGISTRY["get_hardware_profile"]({}, state)
    assert "roofline_bound_gflops" in out
    assert out["roofline_bound_gflops"] > 0
    assert out["simd_width_floats"] == 8  # AVX2 → 8 floats


# ----------- Tools 2-4: python_analyzer suite -----------

def test_profile_python_hotspots(state):
    out = TOOL_REGISTRY["profile_python_hotspots"]({}, state)
    assert "hotspots" in out
    assert isinstance(out["hotspots"], list)
    assert "total_estimated_cost" in out


def test_analyze_complexity_detects_O_n(state):
    out = TOOL_REGISTRY["analyze_complexity"]({}, state)
    assert out["big_o_estimate"] == "O(n)"
    assert out["max_loop_nesting_depth"] == 1


def test_analyze_complexity_detects_O_n_squared(state):
    state.python_code = (
        "def pairwise(X):\n"
        "    n = len(X)\n"
        "    D = [[0.0]*n for _ in range(n)]\n"
        "    for i in range(n):\n"
        "        for j in range(n):\n"
        "            D[i][j] = (X[i] - X[j])**2\n"
        "    return D\n"
    )
    out = TOOL_REGISTRY["analyze_complexity"]({}, state)
    assert out["big_o_estimate"] == "O(n^2)"
    assert out["max_loop_nesting_depth"] == 2


def test_check_memory_access_flags_stride(state):
    state.python_code = (
        "def transpose_loop(a, b, n):\n"
        "    for i in range(n):\n"
        "        for j in range(n):\n"
        "            b[i, j] = a[j, i]\n"     # column-major access in row-major
    )
    out = TOOL_REGISTRY["check_memory_access"]({}, state)
    assert any(i["type"] == "non_unit_stride" for i in out["issues"])


# ----------- Tool 5: cpp_compiler -----------

@pytest.mark.skipif(not HAS_GPP, reason="g++/clang++ not installed")
def test_compile_with_invalid_cpp_returns_syntax_error(state):
    out = TOOL_REGISTRY["compile_and_benchmark"]({"cpp_code": "this is not c++"}, state)
    assert out["compile_status"] == "syntax_error"
    assert out["speedup"] == 0.0


@pytest.mark.skipif(not HAS_GPP, reason="g++/clang++ not installed")
def test_compile_rejects_banned_headers(state):
    code = '#include <mkl.h>\nextern "C" double agent_function() { return 0.0; }\n'
    out = TOOL_REGISTRY["compile_and_benchmark"]({"cpp_code": code}, state)
    assert out["compile_status"] == "syntax_error"
    assert "mkl" in out["error"].lower() or "banned" in out["error"].lower()


def test_compile_rejects_missing_entry_point(state):
    code = "double f(int x) { return x; }\n"  # no extern "C" agent_function
    out = TOOL_REGISTRY["compile_and_benchmark"]({"cpp_code": code}, state)
    assert out["compile_status"] == "syntax_error"
    assert "agent_function" in out["error"]


@pytest.mark.skipif(not HAS_CXX20, reason="C++20 compiler not available (GCC<11 or clang<13)")
def test_compile_valid_cpp_succeeds(state):
    code = (
        '#include <cstddef>\n'
        'extern "C" double agent_function(const double* arr, size_t n) {\n'
        '    double total = 0.0;\n'
        '    for (size_t i = 0; i < n; ++i) total += arr[i] * arr[i];\n'
        '    return total;\n'
        '}\n'
    )
    out = TOOL_REGISTRY["compile_and_benchmark"]({"cpp_code": code}, state)
    assert out["compile_status"] == "success"
    assert out["speedup"] > 0.0


# ----------- Tool 6: verifier -----------

def test_verify_rejects_empty_cpp(state):
    out = TOOL_REGISTRY["verify_equivalence"]({"cpp_code": ""}, state)
    assert out["pass_rate"] == 0.0


@pytest.mark.skipif(not HAS_GPP, reason="g++/clang++ not installed")
def test_verify_rejects_missing_entry(state):
    out = TOOL_REGISTRY["verify_equivalence"]({"cpp_code": "double f() { return 0; }"}, state)
    assert out["pass_rate"] == 0.0


# ----------- Tool 7: portability -----------

def test_portability_with_empty_cpp_returns_zero(state):
    out = TOOL_REGISTRY["check_portability"]({"cpp_code": ""}, state)
    assert out["n_profiles_passing"] == 0
    assert out["portability_bonus_eligible"] is False


# ----------- Tool 8: bottleneck_reporter -----------

def test_bottleneck_reporter_detects_simd_use(state):
    code = (
        '#include <immintrin.h>\n'
        'extern "C" double agent_function(const double* a, size_t n) {\n'
        '    __m256d acc = _mm256_setzero_pd();\n'
        '    for (size_t i = 0; i + 4 <= n; i += 4) {\n'
        '        __m256d v = _mm256_loadu_pd(a + i);\n'
        '        acc = _mm256_fmadd_pd(v, v, acc);\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )
    out = TOOL_REGISTRY["get_bottleneck_report"]({"cpp_code": code}, state)
    assert out["uses_simd"] is True
    assert out["estimated_vectorization_pct"] >= 80.0


def test_bottleneck_reporter_suggests_simd(state):
    code = (
        'extern "C" double agent_function(const double* a, size_t n) {\n'
        '    double t = 0;\n'
        '    for (size_t i = 0; i < n; ++i) t += a[i]*a[i];\n'
        '    return t;\n'
        '}\n'
    )
    out = TOOL_REGISTRY["get_bottleneck_report"]({"cpp_code": code}, state)
    assert out["uses_simd"] is False
    assert any("SIMD" in s for s in out["suggestions"])


# ----------- Tool 9: submit -----------

def test_submit_with_empty_cpp_not_ready(state):
    out = TOOL_REGISTRY["submit_optimization"]({"cpp_code": ""}, state)
    assert out["ready_for_reward"] is False
    assert out["compile_status"] == "syntax_error"
