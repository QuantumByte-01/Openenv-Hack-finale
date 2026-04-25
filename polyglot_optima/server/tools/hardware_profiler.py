"""Tool 1/9: get_hardware_profile.

Returns the hardware profile for the current episode along with the precomputed
Roofline bound. The profile is sampled at reset() time and frozen for the episode;
this tool just exposes it to the agent.

Roofline math (per plan §10):
    simd_w = {"SSE4.2": 4, "AVX2": 8, "AVX-512": 16, "NEON": 4, "none": 1}
    peak_flops = cores × freq_ghz × simd_w × 2   (FMA = 2 ops/cycle)
    peak_bandwidth_flops = bandwidth_gbs × 0.5  (rough flop-per-byte ceiling)
    roofline_bound = min(peak_flops, peak_bandwidth_flops)
"""

from __future__ import annotations

from typing import Any


SIMD_WIDTH = {
    "SSE4.2": 4,
    "AVX2": 8,
    "AVX-512": 16,
    "NEON": 4,
    "none": 1,
}


def roofline_bound(hw: dict[str, Any]) -> float:
    """Compute the Roofline-model peak GFLOPS for a hardware profile."""
    simd_w = SIMD_WIDTH.get(hw["simd"], 1)
    peak_flops = hw["cores"] * hw["freq_ghz"] * simd_w * 2
    peak_bw = hw["bw_gbs"] * 0.5
    return float(min(peak_flops, peak_bw))


def get_hardware_profile_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Return the episode's hardware profile + Roofline bound.

    No arguments — the profile is fixed at episode start.
    """
    hw = state.hardware_profile
    return {
        "id": hw.get("id", "unknown"),
        "cores": hw["cores"],
        "freq_ghz": hw["freq_ghz"],
        "l1_kb": hw["l1_kb"],
        "simd": hw["simd"],
        "bandwidth_gbs": hw["bw_gbs"],
        "roofline_bound_gflops": roofline_bound(hw),
        # Extra context the agent may use
        "simd_width_floats": SIMD_WIDTH.get(hw["simd"], 1),
        "bytes_per_flop_threshold": 1.0 / max(roofline_bound(hw), 0.001),
    }


__all__ = ["get_hardware_profile_tool", "roofline_bound", "SIMD_WIDTH"]
