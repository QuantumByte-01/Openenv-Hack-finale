"""PolyglotOptimaEnvironment — MCPEnvironment subclass with explicit Gym API.

Implements:
- reset(seed=None) -> Observation        # samples a Python function + hardware profile
- step(action) -> StepResult              # routes tool calls, advances rounds, computes reward
- state() -> State                        # episode_id, step_count, round_number
- close()                                 # releases compiler subprocesses, fuzzer pool

Round structure per episode:
    round 1: agent has up to N tool calls, then submits via submit_optimization → R1 reward
    round 2: same, with R1 result available in observation → R2 reward
    round 3: same, FINAL strict gate (≥95% fuzz pass) → R3 reward
    episode_reward = 0.3 * R1_reward + 0.7 * R3_reward (R2 is informational)

The four difficulty axes are frozen at reset() time for each episode but the
adaptive_curriculum module updates them across batches based on success rates.
"""

from __future__ import annotations

import random
import uuid
import os
from dataclasses import dataclass
from typing import Any

# OpenEnv imports — actual class names per the framework docs.
# We accept that some specific imports may need to be adjusted at integration time;
# all are documented as confirmed in §12 of the plan.
try:
    from openenv.core import MCPEnvironment, StepResult  # type: ignore
    from openenv.core.exceptions import OpenEnvError  # type: ignore
except ImportError:
    # Allow stubs for local development before openenv is installed
    class MCPEnvironment:  # type: ignore
        SUPPORTS_CONCURRENT_SESSIONS = True
        async def reset_async(self, seed=None): raise NotImplementedError
        async def step_async(self, action): raise NotImplementedError

    @dataclass
    class StepResult:  # type: ignore
        observation: Any
        reward: float
        done: bool
        info: dict[str, Any] | None = None

    class OpenEnvError(Exception):  # type: ignore
        pass


from models import (
    OptimizationAction,
    OptimizationObservation,
    OptimizationState,
)


# Reserved names that MUST NOT be used as MCP tool names per OpenEnv spec
_RESERVED_TOOL_NAMES = {"reset", "step", "state", "close"}


class PolyglotOptimaEnvironment(MCPEnvironment):
    """The hardware-aware Python→C++ optimization environment.

    Public API:
        env.reset(seed=...) -> OptimizationObservation
        env.step(action: OptimizationAction) -> StepResult
        env.state() -> OptimizationState
        env.close()
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        max_rounds: int = 3,
        max_calls_per_round: int = 5,
        adaptive_axes: dict[str, int] | None = None,
        enable_adaptive_curriculum: bool = True,
        curriculum_batch_size: int = 8,
    ):
        super().__init__()
        self.max_rounds = max_rounds
        self.max_calls_per_round = max_calls_per_round
        self.enable_adaptive_curriculum = enable_adaptive_curriculum
        self.curriculum_batch_size = max(1, int(curriculum_batch_size))
        # Default axes — overridden by adaptive_curriculum across batches
        self._global_axes = adaptive_axes or {
            "function_tier": 0,
            "hardware_class": 0,
            "fuzzer_strictness": 0,
            "portability_required": 0,
        }
        self._sessions: dict[str, OptimizationState] = {}
        self._active_episode_id: str | None = None

        # Lazy imports — modules built in subsequent hours
        self._tool_registry: dict[str, Any] = {}
        self._dataset_loader = None
        self._hardware_profiles = None
        self._reward_dag = None
        self._curriculum = None
        self._episode_success_buffer: list[float] = []

    # -------------------- Gym-style explicit API --------------------

    def reset(self, seed: int | None = None) -> OptimizationObservation:
        """Initialize a new episode.

        Samples (Python function, hardware profile, difficulty axes) deterministically
        from `seed` if provided. Returns the initial Observation.
        """
        rng = random.Random(seed)
        episode_id = str(uuid.uuid4())

        # Lazy init of subsystems (built in later hours; placeholders for now)
        self._ensure_subsystems_loaded()

        # Sample the problem instance
        problem = self._sample_problem(rng)

        state = OptimizationState(
            episode_id=episode_id,
            step_count=0,
            round_number=1,
            is_terminal=False,
            python_code=problem["python_code"],
            function_signature_cpp=problem["cpp_signature"],
            hardware_profile=problem["hardware_profile"],
            bottleneck_ground_truth=problem["bottleneck_labels"],
            bottleneck_distractors=problem["bottleneck_distractors"],
            rtol_override=problem.get("rtol_override"),
            difficulty_axes=dict(self._global_axes),
            is_trap=problem.get("is_trap", False),
            trap_id=problem.get("trap_id"),
        )
        self._sessions[episode_id] = state
        self._active_episode_id = episode_id

        return OptimizationObservation(
            done=False,
            reward=0.0,
            tool_result={"event": "episode_start", "episode_id": episode_id},
            python_code=state.python_code,
            hardware_profile=state.hardware_profile,
            round_number=1,
            rounds_remaining=self.max_rounds - 1,
            best_speedup_so_far=0.0,
            metadata={
                "episode_id": episode_id,
                "difficulty_axes": state.difficulty_axes,
                # NOTE: bottleneck_ground_truth is NOT exposed to the agent —
                #   only used by the server when scoring DiagnosisRubric
            },
        )

    def step(self, action: OptimizationAction) -> StepResult:
        """Execute one tool call or final submission.

        The action.tool_name routes to a registered MCP tool. If the tool is
        `submit_optimization`, the current round closes — reward is computed,
        round advances, and on round 3 the episode terminates.
        """
        if not self._sessions:
            raise OpenEnvError("No active episode. Call reset() first.")
        if self._active_episode_id and self._active_episode_id in self._sessions:
            state = self._sessions[self._active_episode_id]
        else:
            # Fall back to the most recently created episode.
            latest_episode_id = next(reversed(self._sessions))
            self._active_episode_id = latest_episode_id
            state = self._sessions[latest_episode_id]

        if state.is_terminal:
            raise OpenEnvError("Episode is already terminal. Call reset() to start a new one.")

        forced_submit = False
        effective_tool_name = action.tool_name
        effective_tool_args = dict(action.tool_args or {})
        if (
            action.tool_name != "submit_optimization"
            and len(state.current_round_tool_calls) >= self.max_calls_per_round
        ):
            forced_submit = True
            effective_tool_name = "submit_optimization"
            effective_tool_args = {
                "cpp_code": effective_tool_args.get("cpp_code", "// auto-forced submit: call budget reached"),
                "reasoning_trace": action.reasoning_trace or "auto forced submit after max tool calls",
            }

        if effective_tool_name in _RESERVED_TOOL_NAMES:
            raise OpenEnvError(
                f"Tool name '{effective_tool_name}' is reserved. "
                f"Reserved names: {sorted(_RESERVED_TOOL_NAMES)}"
            )

        # Track tool call + reasoning trace for this round
        state.step_count += 1
        state.current_round_tool_calls.append(effective_tool_name)
        if action.reasoning_trace:
            state.current_round_reasoning += action.reasoning_trace + "\n"

        # Route to the named tool — full implementation in Hour 4–10
        tool_result = self._dispatch_tool(effective_tool_name, effective_tool_args, state)

        # Is this a round-closing submission?
        is_submit = effective_tool_name == "submit_optimization"
        round_reward = 0.0
        terminal = False

        if is_submit:
            # Compute reward for this round (Hour 10–16 implementation)
            round_reward = self._compute_round_reward(state, tool_result)
            if self._dataset_loader is not None and hasattr(self._dataset_loader, "record_submission_outcome"):
                self._dataset_loader.record_submission_outcome(state, tool_result)
            state.round_results.append({
                "round": state.round_number,
                "reward": round_reward,
                "tool_calls": list(state.current_round_tool_calls),
                "reasoning": state.current_round_reasoning,
                "submission": tool_result,
            })
            # Reset per-round buffers
            state.current_round_tool_calls.clear()
            state.current_round_reasoning = ""
            # Advance round
            state.round_number += 1
            if state.round_number > self.max_rounds:
                terminal = True
                state.is_terminal = True

        observation = OptimizationObservation(
            done=terminal,
            reward=round_reward,
            tool_result=tool_result,
            python_code=state.python_code,
            hardware_profile=state.hardware_profile,
            round_number=min(state.round_number, self.max_rounds),
            rounds_remaining=max(0, self.max_rounds - state.round_number),
            best_speedup_so_far=state.best_speedup,
            last_compile_status=tool_result.get("compile_status", "pending"),
            last_correctness_pass_rate=tool_result.get("pass_rate", 0.0),
            metadata={
                "episode_id": state.episode_id,
                "step_count": state.step_count,
                "tool_called": effective_tool_name,
                "forced_submit": forced_submit,
                "target_isa": state.hardware_profile.get("target", "scalar_only"),
                "round_reward_breakdown": tool_result.get("_rubric_breakdown", {}),
                "round_readiness_score": tool_result.get("readiness_score"),
                "round_correctness_pass_rate": tool_result.get("correctness_pass_rate"),
                "round_compile_status": tool_result.get("compile_status"),
            },
        )

        # Final episode reward = 0.3*R1 + 0.7*R3 (per plan §10)
        if terminal:
            r1 = next((r["reward"] for r in state.round_results if r["round"] == 1), 0.0)
            r3 = next((r["reward"] for r in state.round_results if r["round"] == 3), 0.0)
            observation.reward = 0.3 * r1 + 0.7 * r3
            observation.metadata["episode_reward_breakdown"] = {
                "r1": r1,
                "r3": r3,
                "episode_total": observation.reward,
            }
            self._record_episode_outcome(state, observation)

        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=terminal,
            info={"state_snapshot_id": state.episode_id, "step": state.step_count},
        )

    def state(self) -> OptimizationState:
        """Return current episode state (Gym-style state introspection)."""
        if not self._sessions:
            raise OpenEnvError("No active episode.")
        if self._active_episode_id and self._active_episode_id in self._sessions:
            return self._sessions[self._active_episode_id]
        latest_episode_id = next(reversed(self._sessions))
        self._active_episode_id = latest_episode_id
        return self._sessions[latest_episode_id]

    def close(self) -> None:
        """Release all resources (compiler subprocesses, fuzzer pool)."""
        self._sessions.clear()
        self._active_episode_id = None
        # Subsystem-specific cleanup — implemented as tools come online
        if self._tool_registry:
            for tool in self._tool_registry.values():
                if hasattr(tool, "close"):
                    tool.close()

    # -------------------- Async variants for parallel rollouts ----

    async def reset_async(self, seed: int | None = None) -> OptimizationObservation:
        return self.reset(seed)

    async def step_async(self, action: OptimizationAction) -> StepResult:
        return self.step(action)

    async def close_async(self) -> None:
        self.close()

    # -------------------- Internal scaffolding --------------------

    def _ensure_subsystems_loaded(self) -> None:
        """Lazy-load tools/dataset/profiles. Real implementations land at Hour 16."""
        # Tools registry
        if not self._tool_registry:
            try:
                from server.tools import TOOL_REGISTRY
                self._tool_registry = TOOL_REGISTRY
            except ImportError:
                self._tool_registry = {}

        # Dataset loader (real, post-Hour 16)
        if self._dataset_loader is None:
            try:
                from server.scenarios import DatasetLoader
                prefer_real = os.environ.get("POLYGLOT_OPTIMA_PREFER_REAL_DATASETS", "1") == "1"
                self._dataset_loader = DatasetLoader(prefer_real_datasets=prefer_real)
            except ImportError:
                self._dataset_loader = _StubDatasetLoader()

        # Hardware profiles (full 8-profile set, post-Hour 16)
        if self._hardware_profiles is None:
            try:
                from server.scenarios.hardware_profiles import HARDWARE_PROFILES
                # Filter held-out for training; eval scripts override this
                self._hardware_profiles = [p for p in HARDWARE_PROFILES if not p.get("held_out")]
            except ImportError:
                self._hardware_profiles = _STUB_PROFILES

        if self._curriculum is None and self.enable_adaptive_curriculum:
            try:
                from server.scenarios import AdaptiveCurriculum
                self._curriculum = AdaptiveCurriculum(initial_axes=dict(self._global_axes))
            except ImportError:
                self._curriculum = None

    def _sample_problem(self, rng: random.Random) -> dict[str, Any]:
        """Sample (function, hw_profile, ground_truth_labels) for an episode.

        Uses the DatasetLoader to draw a (function, hardware) tuple weighted by
        the current global difficulty axes. Falls back to a built-in stub if
        the loader is the local dev fallback.
        """
        # Real loader path (post-Hour 16)
        if isinstance(self._dataset_loader, _StubDatasetLoader):
            hw = rng.choice(self._hardware_profiles)
            return {
                "python_code": _STUB_PYTHON_FUNCTION,
                "cpp_signature": 'extern "C" double agent_function(const double* arr, size_t n);',
                "hardware_profile": hw,
                "bottleneck_labels": ["compute-bound", "vectorizable"],
                "bottleneck_distractors": ["memory-bound", "branch-heavy", "io-bound"],
                "rtol_override": None,
                "is_trap": False,
            }

        return self._dataset_loader.sample(self._global_axes, rng)

    def _record_episode_outcome(self, state: OptimizationState, observation: OptimizationObservation) -> None:
        """Update adaptive curriculum after fixed-size batches of completed episodes."""
        if not self.enable_adaptive_curriculum or self._curriculum is None:
            return
        final_submission = state.round_results[-1]["submission"] if state.round_results else {}
        pass_rate = float(final_submission.get("correctness_pass_rate", 0.0))
        compile_ok = final_submission.get("compile_status") == "success"
        episode_success = 1.0 if (compile_ok and pass_rate >= 0.8) else 0.0
        self._episode_success_buffer.append(episode_success)
        observation.metadata["curriculum_pending_batch_count"] = len(self._episode_success_buffer)
        if len(self._episode_success_buffer) < self.curriculum_batch_size:
            return
        success_rate = sum(self._episode_success_buffer) / len(self._episode_success_buffer)
        action = self._curriculum.observe_batch(success_rate)
        self._global_axes = dict(self._curriculum.axes)
        self._episode_success_buffer.clear()
        observation.metadata["curriculum"] = {
            "success_rate": success_rate,
            "action": action,
            "axes": dict(self._global_axes),
            "batches_seen": self._curriculum.n_batches_seen,
        }

    def _dispatch_tool(self, tool_name: str, tool_args: dict[str, Any], state: OptimizationState) -> dict[str, Any]:
        """Route a tool call to the registered handler.

        Real implementations land in Hour 4–10. Until then, stub responses keep the
        Gym API live for smoke tests.
        """
        if tool_name not in self._tool_registry:
            return {
                "stub": True,
                "tool": tool_name,
                "message": f"Tool '{tool_name}' not yet implemented (Hour 4-10).",
            }
        return self._tool_registry[tool_name](tool_args, state)

    def _compute_round_reward(self, state: OptimizationState, submission: dict[str, Any]) -> float:
        """Apply the round-appropriate Sequential(Gate, Gate, WeightedSum) rubric.

        Per plan §10:
            R1: soft gate (60% correctness), 3 components
            R2: medium gate (80%), informational
            R3: strict gate (95%), 5 components incl. portability + self-correction

        Returns the rubric DAG's score in [0, 1], or 0.0 if any gate fails.
        """
        try:
            from server.rewards import build_round_reward_dag
        except ImportError:
            return 0.0

        # Append a synthetic round_result entry NOW so DiagnosisRubric / SelfCorrectionRubric
        # can read the just-completed round's tool calls. The caller (step()) appends the
        # *real* round_results entry after this returns; we only need a temp lookup.
        # Note: we already appended state.round_results in step() BEFORE computing reward,
        # so this is fine. Diagnosis and SelfCorrection both read state.round_results.

        dag = build_round_reward_dag(state.round_number)
        score = dag.score(state, submission)

        # Stash breakdown in submission for telemetry / wandb logging
        submission["_rubric_breakdown"] = getattr(dag, "last_breakdown", {})
        return score


# --------------------------- Stubs (Hour 0–4 only) -------------------

class _StubDatasetLoader:
    """Placeholder. Replaced in Hour 16 by server.scenarios.dataset_loader."""

    def sample(self, axes: dict[str, int], rng: random.Random) -> dict[str, Any]:
        return {"python_code": _STUB_PYTHON_FUNCTION}


_STUB_PROFILES = [
    {
        "id": "desktop_avx2",
        "cores": 8,
        "freq_ghz": 3.8,
        "l1_kb": 32,
        "simd": "AVX2",
        "bw_gbs": 51,
        "roofline_bound_gflops": 25.5,
        "target": "x86_AVX2",
    },
]


_STUB_PYTHON_FUNCTION = '''def sum_squares(arr):
    """Compute the sum of squares of an array — placeholder during Hour 0-4."""
    total = 0.0
    for x in arr:
        total += x * x
    return total
'''


__all__ = [
    "PolyglotOptimaEnvironment",
]
