"""Hour 0-4 skeleton smoke tests.

Verifies the bare minimum:
1. Models import and validate
2. Environment imports and exposes reset/step/state/close
3. reset() returns a typed Observation
4. step() with a stub tool name doesn't crash and advances state
5. submit_optimization closes a round
6. After 3 rounds the episode is terminal
7. Reserved tool names are rejected
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make polyglot_optima importable for tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from models import (
    OptimizationAction,
    OptimizationObservation,
    OptimizationState,
)
from server.environment import PolyglotOptimaEnvironment


def test_models_validate():
    """Pydantic models accept valid input and reject extras."""
    action = OptimizationAction(
        tool_name="get_hardware_profile",
        tool_args={},
        reasoning_trace="<think>just exploring</think>",
    )
    assert action.tool_name == "get_hardware_profile"

    obs = OptimizationObservation(done=False, reward=0.0)
    assert obs.round_number == 1

    state = OptimizationState(episode_id="ep1")
    assert state.step_count == 0
    assert state.is_terminal is False
    assert "function_tier" in state.difficulty_axes


def test_models_reject_extras():
    """extra='forbid' on all three models."""
    with pytest.raises(Exception):
        OptimizationAction(tool_name="x", unknown_field=42)


def test_environment_has_gym_api():
    """Environment exposes the explicit Gym-style API per plan §12 A."""
    env = PolyglotOptimaEnvironment()
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "state")
    assert hasattr(env, "close")
    assert env.SUPPORTS_CONCURRENT_SESSIONS is True


def test_reset_returns_typed_observation():
    """reset() returns an OptimizationObservation with the expected shape."""
    env = PolyglotOptimaEnvironment()
    obs = env.reset(seed=42)
    assert isinstance(obs, OptimizationObservation)
    assert obs.done is False
    assert obs.round_number == 1
    assert obs.python_code != ""
    assert "simd" in obs.hardware_profile
    assert obs.metadata["episode_id"]


def test_state_introspection():
    """state() returns the in-memory OptimizationState."""
    env = PolyglotOptimaEnvironment()
    env.reset(seed=42)
    s = env.state()
    assert isinstance(s, OptimizationState)
    assert s.step_count == 0
    assert s.round_number == 1
    assert s.is_terminal is False


def test_step_with_stub_tool_does_not_crash():
    """A non-submit tool call advances step_count, doesn't terminate the episode."""
    env = PolyglotOptimaEnvironment()
    env.reset(seed=42)
    result = env.step(OptimizationAction(
        tool_name="profile_python_hotspots",
        tool_args={"code": "def f(): pass"},
        reasoning_trace="<think>checking hotspots</think>",
    ))
    assert result.done is False
    assert env.state().step_count == 1


def test_reserved_tool_names_rejected():
    """OpenEnv reserved names (reset/step/state/close) must not be used as tool names."""
    env = PolyglotOptimaEnvironment()
    env.reset(seed=42)
    with pytest.raises(Exception):
        env.step(OptimizationAction(tool_name="reset", tool_args={}, reasoning_trace=""))
    with pytest.raises(Exception):
        env.step(OptimizationAction(tool_name="close", tool_args={}, reasoning_trace=""))


def test_submit_advances_round():
    """submit_optimization closes the current round and bumps round_number."""
    env = PolyglotOptimaEnvironment()
    env.reset(seed=42)
    result = env.step(OptimizationAction(
        tool_name="submit_optimization",
        tool_args={"cpp_code": "// stub", "reasoning_trace": "<think>round 1</think>"},
        reasoning_trace="<think>round 1</think>",
    ))
    assert result.done is False  # 2 more rounds remain
    assert env.state().round_number == 2


def test_three_submits_terminate_episode():
    """3 submits → episode terminal, final reward is computed."""
    env = PolyglotOptimaEnvironment()
    env.reset(seed=42)
    for r in range(3):
        result = env.step(OptimizationAction(
            tool_name="submit_optimization",
            tool_args={"cpp_code": "// stub", "reasoning_trace": f"r{r+1}"},
            reasoning_trace=f"<think>round {r+1}</think>",
        ))
    assert result.done is True
    assert env.state().is_terminal is True
    # Final reward in stub mode is 0.0; real values in Hour 10–16
    assert isinstance(result.reward, float)


def test_close_clears_sessions():
    env = PolyglotOptimaEnvironment()
    env.reset(seed=1)
    assert env._sessions
    env.close()
    assert not env._sessions
