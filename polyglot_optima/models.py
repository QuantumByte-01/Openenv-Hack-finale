"""Pydantic data models for Polyglot-Optima environment.

Three core types:
- OptimizationAction: what the agent sends to the env each turn
- OptimizationObservation: what the env returns each step
- OptimizationState: episode state tracked by the env (episode_id, step_count, round_number, etc.)

These map onto the OpenEnv Action/Observation/State base classes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ----------------------------- Action -----------------------------

class OptimizationAction(BaseModel):
    """One agent turn.

    Either a tool call (most turns) or a final submission (last turn of round 3).
    The agent's reasoning trace is required so the DiagnosisRubric can score it.
    """

    tool_name: str = Field(..., description="Name of the MCP tool to call")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="Arguments to the tool")
    reasoning_trace: str = Field(
        default="",
        description="Agent's <think>...</think> trace before this action. "
                    "Required to be non-empty for DiagnosisRubric scoring.",
        max_length=2048,
    )

    model_config = {"extra": "forbid"}


# --------------------------- Observation ---------------------------

class OptimizationObservation(BaseModel):
    """One env response.

    Returned by env.step() and env.reset(). Contains tool result, episode state,
    and per-step debug telemetry in `metadata` (sub-rubric scores, axis levels,
    fuzz failure samples, etc.).
    """

    # Standard OpenEnv Observation fields
    done: bool = Field(default=False, description="True iff episode is over")
    reward: float = Field(default=0.0, description="Reward for this step (0 unless terminal)")

    # Domain-specific payload
    tool_result: dict[str, Any] = Field(default_factory=dict, description="Output of the tool just called")

    # Environment context exposed to the agent
    python_code: str = Field(default="", description="The Python function the agent is optimizing")
    hardware_profile: dict[str, Any] = Field(
        default_factory=dict,
        description="Synthetic hardware spec for this episode (cores, simd, bandwidth, roofline_bound)",
    )
    round_number: int = Field(default=1, description="Current refinement round (1, 2, or 3)")
    rounds_remaining: int = Field(default=2)

    # Cumulative state visible to the agent
    best_speedup_so_far: float = Field(default=0.0)
    last_compile_status: Literal["pending", "success", "syntax_error", "link_error", "timeout"] = "pending"
    last_correctness_pass_rate: float = Field(default=0.0)

    # Telemetry — used by training infra, not necessarily shown to the model
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


# ----------------------------- State ------------------------------

class OptimizationState(BaseModel):
    """Episode-level state tracked by the environment server.

    Not every field is exposed to the agent in each Observation. Some are
    server-internal (e.g., the ground-truth bottleneck label, the trap function
    metadata, the curriculum axis levels).
    """

    # Identity
    episode_id: str
    step_count: int = 0
    round_number: int = 1
    is_terminal: bool = False

    # Problem instance
    python_code: str = ""
    function_signature_cpp: str = ""  # extern "C" void agent_function(...) — derived from AST
    hardware_profile: dict[str, Any] = Field(default_factory=dict)

    # Ground-truth (server-only — never sent to agent)
    bottleneck_ground_truth: list[str] = Field(default_factory=list)  # e.g., ["compute-bound", "vectorizable"]
    bottleneck_distractors: list[str] = Field(default_factory=list)
    rtol_override: float | None = None  # Some functions need bit-exact (rtol=0); most use 1e-5

    # Per-round history
    round_results: list[dict[str, Any]] = Field(default_factory=list)
    best_speedup: float = 0.0
    best_cpp_code: str = ""

    # Tool-call history within the current round (for action-coherence diagnosis bonus)
    current_round_tool_calls: list[str] = Field(default_factory=list)
    current_round_reasoning: str = ""

    # Adaptive curriculum axis levels at episode start (frozen for the episode)
    difficulty_axes: dict[str, int] = Field(
        default_factory=lambda: {
            "function_tier": 0,         # 0..3
            "hardware_class": 0,        # 0..2
            "fuzzer_strictness": 0,     # 0..2
            "portability_required": 0,  # 0..1
        }
    )

    # Trap flag — is this episode a known anti-gaming trap?
    is_trap: bool = False
    trap_id: str | None = None

    model_config = {"extra": "forbid"}


# ------------------------- Public re-exports ----------------------

__all__ = [
    "OptimizationAction",
    "OptimizationObservation",
    "OptimizationState",
]
