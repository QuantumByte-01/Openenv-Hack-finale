"""Tool 7/9: check_portability.

Compiles the agent's C++ against each of the 8 hardware profile flag-sets
and runs a quick correctness check (subset of the fuzzer) on each. Awards
the portability bonus if 3+ profiles pass.

Per plan §3 axis 4 (`portability_required`), the agent only earns the
PortabilityRubric bonus when this axis is escalated. Otherwise the result
is informational.
"""

from __future__ import annotations

import json
from typing import Any

from server.tools.cpp_compiler import _compile, _sha256


# Per-profile compile flag overrides (in addition to the base `_BASE_COMPILE_FLAGS`).
# `-march=native` is replaced with the appropriate -m* flag matching the profile's SIMD level.
PROFILE_COMPILE_OVERRIDES = {
    "SSE4.2":  ["-msse4.2", "-mno-avx", "-mno-avx2", "-mno-avx512f"],
    "AVX2":    ["-mavx2", "-mfma", "-mno-avx512f"],
    "AVX-512": ["-mavx512f", "-mavx512cd", "-mavx512vl"],
    "NEON":    ["-mfpu=neon"],     # ARM-only — for cross-compile mode
    "none":    ["-mno-sse", "-mno-avx", "-mno-avx2"],
}


def _override_flags(base_flags: list[str], simd: str) -> list[str]:
    """Replace -march=native with the profile-specific SIMD flag set."""
    out = [f for f in base_flags if not f.startswith("-march=")]
    out += PROFILE_COMPILE_OVERRIDES.get(simd, [])
    return out


def check_portability_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Test compile + quick correctness on all 8 hardware profiles.

    Args:
        cpp_code (str)
        n_cases_per_profile (int=50)  — quick smoke check per profile

    Returns:
        per_profile (dict[str, dict])  — id → {compile, correctness}
        n_profiles_passing (int)
        portability_bonus_eligible (bool)  — True if ≥3 profiles compile + pass correctness
    """
    cpp_code = tool_args.get("cpp_code", "")
    if not cpp_code.strip():
        return {"per_profile": {}, "n_profiles_passing": 0, "portability_bonus_eligible": False, "error": "empty cpp_code"}

    # Lazy-import the full profile list — provided by scenarios.hardware_profiles in Hour 16
    try:
        from server.scenarios.hardware_profiles import HARDWARE_PROFILES
    except ImportError:
        # During Hour 4-10 use a stub list with all 8 profiles inlined
        HARDWARE_PROFILES = _STUB_PROFILES

    per_profile: dict[str, dict[str, Any]] = {}
    n_passing = 0

    # Reuse the simple verifier over a small sample
    from server.tools.verifier import verify_equivalence_tool

    for hw in HARDWARE_PROFILES:
        if hw["id"] == state.hardware_profile.get("id"):
            # Skip the home profile — we test it via the main verifier
            continue
        cache_key = _sha256(cpp_code, json.dumps(hw, sort_keys=True), "portability")
        compile_result = _compile(cpp_code, hw, cache_key)
        compile_ok = compile_result["status"] == "success"

        correctness_ok = False
        if compile_ok:
            # Quick fuzz on this profile (50 cases)
            verifier_args = {
                "cpp_code": cpp_code,
                "python_code": state.python_code,
                "n_cases": int(tool_args.get("n_cases_per_profile", 50)),
            }
            # Temporarily swap the state's hw profile so the verifier compiles for this one
            saved_hw = state.hardware_profile
            state.hardware_profile = hw
            try:
                v = verify_equivalence_tool(verifier_args, state)
                correctness_ok = v.get("pass_rate", 0.0) >= 0.95
            finally:
                state.hardware_profile = saved_hw

        per_profile[hw["id"]] = {
            "compile": "success" if compile_ok else "fail",
            "correctness_ok": correctness_ok,
            "compile_error": compile_result.get("error", "")[:300] if not compile_ok else "",
        }
        if compile_ok and correctness_ok:
            n_passing += 1

    eligible = n_passing >= 3

    return {
        "per_profile": per_profile,
        "n_profiles_passing": n_passing,
        "portability_bonus_eligible": eligible,
        "tested_profiles": [p["id"] for p in HARDWARE_PROFILES if p["id"] != state.hardware_profile.get("id")],
    }


# Inline 8-profile stub used during Hour 4-10 before scenarios module is built
_STUB_PROFILES = [
    {"id": "laptop_sse",    "cores": 4,  "freq_ghz": 3.2, "l1_kb": 32, "simd": "SSE4.2",  "bw_gbs": 40},
    {"id": "desktop_avx2",  "cores": 8,  "freq_ghz": 3.8, "l1_kb": 32, "simd": "AVX2",    "bw_gbs": 51},
    {"id": "server_avx512", "cores": 16, "freq_ghz": 3.0, "l1_kb": 48, "simd": "AVX-512", "bw_gbs": 89},
    {"id": "arm_neon_a",    "cores": 6,  "freq_ghz": 2.4, "l1_kb": 64, "simd": "NEON",    "bw_gbs": 68},
    {"id": "embedded",      "cores": 2,  "freq_ghz": 1.8, "l1_kb": 16, "simd": "none",    "bw_gbs": 25},
    {"id": "workstation",   "cores": 12, "freq_ghz": 4.0, "l1_kb": 48, "simd": "AVX2",    "bw_gbs": 76},
    {"id": "arm_neon_b",    "cores": 8,  "freq_ghz": 2.8, "l1_kb": 32, "simd": "NEON",    "bw_gbs": 68},
    {"id": "laptop_sse2",   "cores": 4,  "freq_ghz": 2.6, "l1_kb": 64, "simd": "SSE4.2",  "bw_gbs": 35},
]


__all__ = ["check_portability_tool", "PROFILE_COMPILE_OVERRIDES"]
