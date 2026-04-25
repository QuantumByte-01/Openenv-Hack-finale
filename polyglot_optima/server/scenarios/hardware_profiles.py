"""8 Roofline-calibrated synthetic hardware profiles (per plan §10).

Profile classes (for the `hardware_class` curriculum axis):
    Class 0 (easy):   laptop_sse, desktop_avx2
    Class 1 (medium): workstation, arm_neon_a, laptop_sse2
    Class 2 (hard):   server_avx512, embedded, arm_neon_b (held-out for Gen-2 eval)

`arm_neon_b` is the held-out profile (never sampled during training). Used for
the Gen-2 evaluation split that tests hardware-reasoning generalization.
"""

from __future__ import annotations

from typing import Any


HARDWARE_PROFILES: list[dict[str, Any]] = [
    # Class 0 — easy, common consumer hardware
    {"id": "laptop_sse",    "cores": 4,  "freq_ghz": 3.2, "l1_kb": 32, "simd": "SSE4.2",  "bw_gbs": 40, "class": 0},
    {"id": "desktop_avx2",  "cores": 8,  "freq_ghz": 3.8, "l1_kb": 32, "simd": "AVX2",    "bw_gbs": 51, "class": 0},

    # Class 1 — medium, varied
    {"id": "workstation",   "cores": 12, "freq_ghz": 4.0, "l1_kb": 48, "simd": "AVX2",    "bw_gbs": 76, "class": 1},
    {"id": "arm_neon_a",    "cores": 6,  "freq_ghz": 2.4, "l1_kb": 64, "simd": "NEON",    "bw_gbs": 68, "class": 1},
    {"id": "laptop_sse2",   "cores": 4,  "freq_ghz": 2.6, "l1_kb": 64, "simd": "SSE4.2",  "bw_gbs": 35, "class": 1},

    # Class 2 — hard, demands real hardware reasoning
    {"id": "server_avx512", "cores": 16, "freq_ghz": 3.0, "l1_kb": 48, "simd": "AVX-512", "bw_gbs": 89, "class": 2},
    {"id": "embedded",      "cores": 2,  "freq_ghz": 1.8, "l1_kb": 16, "simd": "none",    "bw_gbs": 25, "class": 2},

    # HELD-OUT for Gen-2 evaluation — never sampled during training
    {"id": "arm_neon_b",    "cores": 8,  "freq_ghz": 2.8, "l1_kb": 32, "simd": "NEON",    "bw_gbs": 68, "class": 2, "held_out": True},
]


HARDWARE_BY_CLASS: dict[int, list[dict[str, Any]]] = {
    0: [p for p in HARDWARE_PROFILES if p.get("class") == 0 and not p.get("held_out")],
    1: [p for p in HARDWARE_PROFILES if p.get("class") == 1 and not p.get("held_out")],
    2: [p for p in HARDWARE_PROFILES if p.get("class") == 2 and not p.get("held_out")],
}


HELD_OUT_PROFILES: list[dict[str, Any]] = [p for p in HARDWARE_PROFILES if p.get("held_out")]


def profile_by_id(profile_id: str) -> dict[str, Any] | None:
    return next((p for p in HARDWARE_PROFILES if p["id"] == profile_id), None)


def sample_profile(rng, axis_level: int = 0) -> dict[str, Any]:
    """Sample a hardware profile appropriate for the given axis level.

    Per plan §3, axis_level escalates the hardware-class pool:
        level 0 → only Class 0 (easy)
        level 1 → Class 0 + 1
        level 2 → all training profiles (Class 0 + 1 + 2 minus held-out)
    """
    pool: list[dict[str, Any]] = []
    for level in range(min(axis_level, 2) + 1):
        pool.extend(HARDWARE_BY_CLASS[level])
    if not pool:
        pool = HARDWARE_BY_CLASS[0]
    return rng.choice(pool)


__all__ = [
    "HARDWARE_PROFILES",
    "HARDWARE_BY_CLASS",
    "HELD_OUT_PROFILES",
    "profile_by_id",
    "sample_profile",
]
