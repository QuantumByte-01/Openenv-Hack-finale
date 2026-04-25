"""Tool 8/9: get_bottleneck_report.

Returns a `perf stat`-style report for the agent's compiled C++ — instructions
per cycle, cache miss rate, vectorization status. Helps the agent diagnose
*why* its C++ is slow before refining.

Real implementation (Hour 16) reads /proc/perf_event or uses Linux perf_event_open
to collect counters during the benchmark run. For Hour 4-10, this is a heuristic
estimate based on static C++ analysis (looks for SIMD intrinsics, OpenMP, etc.).
"""

from __future__ import annotations

import re
from typing import Any


_SIMD_INTRINSIC_PATTERN = re.compile(
    r"_mm\d+_|_mm_|vld\d+q?_|vst\d+q?_|vmul[a-z]?_|vadd[a-z]?_|"
    r"__m\d+|svfloat|svint"
)
_OPENMP_PATTERN = re.compile(r"#\s*pragma\s+omp")
_RESTRICT_PATTERN = re.compile(r"\b__restrict__\b|\brestrict\b")
_LIKELY_PATTERN = re.compile(r"\[\[\s*(un)?likely\s*\]\]")


def get_bottleneck_report_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Static analysis of agent's C++ → estimate of vectorization, parallelism, etc.

    Args:
        cpp_code (str)

    Returns:
        uses_simd (bool)
        uses_openmp (bool)
        uses_restrict (bool)
        uses_branch_hints (bool)
        estimated_ipc (float)        — heuristic
        estimated_cache_miss_rate (float)
        estimated_vectorization_pct (float)
        suggestions (list[str])      — hints for next round
    """
    cpp_code = tool_args.get("cpp_code", "")
    if not cpp_code.strip():
        return {"error": "empty cpp_code"}

    uses_simd = bool(_SIMD_INTRINSIC_PATTERN.search(cpp_code))
    uses_openmp = bool(_OPENMP_PATTERN.search(cpp_code))
    uses_restrict = bool(_RESTRICT_PATTERN.search(cpp_code))
    uses_hints = bool(_LIKELY_PATTERN.search(cpp_code))

    # Heuristic IPC estimate (1.0 = scalar, 4.0 = AVX2 SIMD, 8.0 = AVX-512)
    simd_w = {"SSE4.2": 4, "AVX2": 8, "AVX-512": 16, "NEON": 4, "none": 1}.get(
        state.hardware_profile.get("simd", "none"), 1
    )
    estimated_ipc = 0.8
    if uses_simd:
        estimated_ipc = min(simd_w * 0.6, 8.0)
    if uses_openmp:
        estimated_ipc *= min(state.hardware_profile.get("cores", 1), 4) * 0.7

    estimated_cache_miss = 0.20
    if uses_restrict:
        estimated_cache_miss *= 0.7

    estimated_vec_pct = 5.0
    if uses_simd:
        estimated_vec_pct = 80.0
    elif uses_openmp:
        estimated_vec_pct = 20.0  # GCC may auto-vectorize OpenMP loops

    suggestions: list[str] = []
    if not uses_simd and simd_w >= 4:
        suggestions.append(
            f"Hardware supports {state.hardware_profile['simd']} (width {simd_w}). "
            f"Consider explicit SIMD intrinsics."
        )
    if not uses_openmp and state.hardware_profile.get("cores", 1) >= 4:
        suggestions.append(
            f"Hardware has {state.hardware_profile['cores']} cores. "
            f"Add `#pragma omp parallel for` to outer loops."
        )
    if not uses_restrict and "ndarray" in state.python_code.lower():
        suggestions.append(
            "Add `__restrict__` to pointer args — tells the compiler arrays don't alias."
        )
    if not suggestions:
        suggestions.append("Looks well-optimized. Refining further may yield marginal gains.")

    return {
        "uses_simd": uses_simd,
        "uses_openmp": uses_openmp,
        "uses_restrict": uses_restrict,
        "uses_branch_hints": uses_hints,
        "estimated_ipc": estimated_ipc,
        "estimated_cache_miss_rate": estimated_cache_miss,
        "estimated_vectorization_pct": estimated_vec_pct,
        "suggestions": suggestions,
        "method": "static_pattern_match",
    }


__all__ = ["get_bottleneck_report_tool"]
