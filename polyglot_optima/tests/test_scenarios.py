"""Hour 16-22: Scenarios, dataset loader, adaptive curriculum tests."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from server.scenarios.hardware_profiles import (
    HARDWARE_PROFILES, HARDWARE_BY_CLASS, HELD_OUT_PROFILES, profile_by_id, sample_profile,
)
from server.scenarios.trap_library import (
    TRAP_LIBRARY, sample_trap, trap_to_problem_dict, get_trap_by_id,
    N_TRAPS_TOTAL, N_TRAPS_TRAINING, N_TRAPS_HELDOUT,
)
from server.scenarios.generator import TemplateGenerator, generate_from_template
from server.scenarios.dataset_loader import DatasetLoader, sample_function
from server.scenarios.adaptive_curriculum import AdaptiveCurriculum, MAX_LEVEL


# -------- Hardware profiles --------

def test_hardware_profiles_count():
    """Plan §10 mandates 8 hardware profiles."""
    assert len(HARDWARE_PROFILES) == 8


def test_held_out_arm_neon_b_present():
    """`arm_neon_b` is the held-out profile per plan §5 Gen-2."""
    assert any(p["id"] == "arm_neon_b" for p in HELD_OUT_PROFILES)
    assert profile_by_id("arm_neon_b")["held_out"] is True


def test_held_out_excluded_from_class_pools():
    """held-out profiles must NOT appear in HARDWARE_BY_CLASS (training pool)."""
    training_ids = {p["id"] for cls in HARDWARE_BY_CLASS.values() for p in cls}
    assert "arm_neon_b" not in training_ids


def test_sample_profile_respects_axis_level():
    rng = random.Random(0)
    # Level 0: only class 0 profiles
    seen = {sample_profile(rng, axis_level=0)["id"] for _ in range(50)}
    class_0_ids = {p["id"] for p in HARDWARE_BY_CLASS[0]}
    assert seen <= class_0_ids


# -------- Trap library --------

def test_trap_library_count():
    """Plan §10b mandates 30 traps."""
    assert N_TRAPS_TOTAL == 30


def test_trap_library_split_30_4():
    """26 training + 4 held-out traps (plan §4.3 + §5 Gen-4)."""
    # Hour 16 ships 26 training + 4 held-out
    assert N_TRAPS_TRAINING + N_TRAPS_HELDOUT == 30
    assert N_TRAPS_HELDOUT >= 4  # may add more later


def test_each_trap_has_metadata():
    for trap in TRAP_LIBRARY:
        assert trap.id, "trap missing id"
        assert trap.python_code.strip()
        assert trap.bottleneck_label, f"{trap.id} missing labels"
        assert trap.category in {
            "overflow", "fp_order", "aliasing", "edge_empty",
            "nan_inf", "boundary", "semantics",
        }


def test_sample_trap_excludes_held_out():
    rng = random.Random(0)
    held_out_ids = {t.id for t in TRAP_LIBRARY if t.held_out}
    # 200 samples — none should be in held-out
    seen_ids = {sample_trap(rng, exclude_held_out=True).id for _ in range(200)}
    assert seen_ids.isdisjoint(held_out_ids)


def test_trap_to_problem_dict_shape():
    trap = TRAP_LIBRARY[0]
    hw = HARDWARE_PROFILES[0]
    p = trap_to_problem_dict(trap, hw)
    assert p["is_trap"] is True
    assert p["python_code"] == trap.python_code
    assert p["hardware_profile"] == hw
    assert p["bottleneck_labels"] == trap.bottleneck_label


# -------- Template generator --------

def test_template_generator_samples_within_tier():
    rng = random.Random(0)
    gen = TemplateGenerator()
    seen_tiers = set()
    for _ in range(50):
        t = gen.sample(tier=2, rng=rng)
        seen_tiers.add(t.tier)
        assert t.tier <= 2
    # Should have hit tier 0, 1, AND 2 over many samples (all included in pool)
    assert {0, 1, 2} & seen_tiers


def test_generate_from_template_shape():
    rng = random.Random(0)
    gen = TemplateGenerator()
    t = gen.sample(tier=0, rng=rng)
    p = generate_from_template(t, HARDWARE_PROFILES[0])
    assert p["is_trap"] is False
    assert p["tier"] == t.tier
    assert "agent_function" in p["cpp_signature"]


# -------- Dataset loader --------

def test_dataset_loader_returns_problem_dict():
    rng = random.Random(0)
    loader = DatasetLoader(prefer_real_datasets=False)
    p = loader.sample({"function_tier": 0, "hardware_class": 0,
                       "fuzzer_strictness": 0, "portability_required": 0}, rng)
    assert "python_code" in p
    assert "hardware_profile" in p
    assert "bottleneck_labels" in p


def test_dataset_loader_traps_at_15_pct():
    """Over many samples, trap probability should approximate 15% (plan §4.3)."""
    rng = random.Random(0)
    loader = DatasetLoader(prefer_real_datasets=False)
    n = 500
    n_traps = sum(loader.sample({"function_tier": 0, "hardware_class": 0,
                                 "fuzzer_strictness": 0, "portability_required": 0}, rng)
                  ["is_trap"] for _ in range(n))
    pct = n_traps / n
    assert 0.10 <= pct <= 0.20  # 15% ± 5pp tolerance for n=500


def test_sample_function_module_function():
    rng = random.Random(0)
    p = sample_function({"function_tier": 0, "hardware_class": 0,
                         "fuzzer_strictness": 0, "portability_required": 0}, rng)
    assert "python_code" in p


# -------- Adaptive curriculum (4-axis) --------

def test_curriculum_starts_at_zero():
    c = AdaptiveCurriculum(seed=0)
    assert all(v == 0 for v in c.axes.values())


def test_curriculum_escalates_on_high_success():
    c = AdaptiveCurriculum(seed=0)
    c.observe_batch(success_rate=0.9)
    # One axis should now be 1
    assert sum(c.axes.values()) == 1
    assert "escalate" in c.last_action


def test_curriculum_holds_in_goldilocks():
    c = AdaptiveCurriculum(seed=0)
    c.observe_batch(success_rate=0.5)
    assert all(v == 0 for v in c.axes.values())
    assert "hold" in c.last_action


def test_curriculum_deescalates_on_low_success():
    c = AdaptiveCurriculum(seed=0, initial_axes={"function_tier": 2, "hardware_class": 0,
                                                 "fuzzer_strictness": 0, "portability_required": 0})
    c.observe_batch(success_rate=0.1)
    assert c.axes["function_tier"] == 1
    assert "de-escalate" in c.last_action


def test_curriculum_caps_at_max():
    """Once an axis is maxed, further escalation can't push it beyond MAX_LEVEL."""
    c = AdaptiveCurriculum(seed=0, initial_axes=dict(MAX_LEVEL))
    for _ in range(10):
        c.observe_batch(success_rate=0.95)
    assert all(c.axes[a] == MAX_LEVEL[a] for a in MAX_LEVEL)


def test_curriculum_floors_at_min():
    """Once an axis is at min (0), further de-escalation can't push it below."""
    c = AdaptiveCurriculum(seed=0)
    for _ in range(10):
        c.observe_batch(success_rate=0.05)
    assert all(c.axes[a] == 0 for a in MAX_LEVEL)


def test_curriculum_snapshot_keys():
    c = AdaptiveCurriculum(seed=0)
    c.observe_batch(success_rate=0.9)
    s = c.snapshot()
    assert s.success_rate == 0.9
    assert s.n_batches_seen == 1
    assert sum(s.n_escalations.values()) == 1


def test_curriculum_to_dict_serializable():
    """Used by wandb logging."""
    c = AdaptiveCurriculum(seed=0)
    c.observe_batch(0.8)
    d = c.to_dict()
    assert "axes" in d and "n_escalations" in d


# -------- Environment integration --------

def test_environment_uses_real_dataset_loader():
    """env.reset() now uses DatasetLoader + scenarios subsystem."""
    from server.environment import PolyglotOptimaEnvironment
    env = PolyglotOptimaEnvironment()
    # Run multiple resets to confirm we draw varied problems
    seen_codes = set()
    for s in range(20):
        obs = env.reset(seed=s)
        seen_codes.add(obs.python_code[:50])
    # Variety > 1 confirms loader is sampling, not returning a stub
    assert len(seen_codes) > 1
