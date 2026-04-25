"""Polyglot-Optima client — typed wrapper around the WebSocket env API.

Two clients are provided:
- PolyglotOptimaClient: async (the canonical OpenEnv pattern)
- PolyglotOptimaSyncClient: synchronous wrapper, used inside the TRL training loop

Both are typed: `reset()` returns OptimizationObservation, `step()` returns
StepResult containing OptimizationObservation. No raw dicts.

Strict client/server boundary: this module imports nothing from `server/`. All
communication is over HTTP/WebSocket via the OpenEnv EnvClient base.
"""

from __future__ import annotations

from typing import Any

try:
    from openenv.core.client import EnvClient, SyncEnvClient  # type: ignore
except ImportError:
    # Local-dev stub; real client imported once openenv is installed
    class EnvClient:  # type: ignore
        def __init__(self, base_url: str, action_cls=None, observation_cls=None):
            self.base_url = base_url
            self.action_cls = action_cls
            self.observation_cls = observation_cls

        async def reset(self, seed: int | None = None):
            raise NotImplementedError("Install openenv to use the real client")

        async def step(self, action):
            raise NotImplementedError("Install openenv to use the real client")

    class SyncEnvClient(EnvClient):  # type: ignore
        def reset(self, seed: int | None = None):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError


from models import OptimizationAction, OptimizationObservation


class PolyglotOptimaClient(EnvClient):
    """Async typed client.

    Usage:
        async with PolyglotOptimaClient("ws://localhost:8000") as client:
            obs = await client.reset(seed=42)
            obs = await client.step(OptimizationAction(
                tool_name="profile_python_hotspots",
                tool_args={"code": obs.python_code},
                reasoning_trace="<think>...</think>",
            ))
    """

    def __init__(self, base_url: str = "ws://localhost:8000"):
        super().__init__(
            base_url=base_url,
            action_cls=OptimizationAction,
            observation_cls=OptimizationObservation,
        )

    # Convenience wrappers — strongly typed
    async def reset(self, seed: int | None = None) -> OptimizationObservation:  # type: ignore[override]
        return await super().reset(seed=seed)

    async def step(self, action: OptimizationAction) -> Any:  # type: ignore[override]
        # Returns StepResult with .observation : OptimizationObservation
        return await super().step(action)

    async def close(self) -> None:
        # OpenEnv-base lifecycle teardown
        if hasattr(super(), "close"):
            await super().close()  # type: ignore


class PolyglotOptimaSyncClient(SyncEnvClient):
    """Synchronous wrapper for use inside synchronous training loops (TRL GRPOTrainer).

    Per plan §12 A: SyncEnvClient is the recommended pattern when the host loop
    is synchronous (TRL's training loop is). Internally calls the async client.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__(
            base_url=base_url,
            action_cls=OptimizationAction,
            observation_cls=OptimizationObservation,
        )

    def reset(self, seed: int | None = None) -> OptimizationObservation:  # type: ignore[override]
        return super().reset(seed=seed)

    def step(self, action: OptimizationAction) -> Any:  # type: ignore[override]
        return super().step(action)


__all__ = [
    "PolyglotOptimaClient",
    "PolyglotOptimaSyncClient",
]
