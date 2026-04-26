"""FastAPI app factory for Polyglot-Optima.

Uses OpenEnv's create_app() to wire the MCPEnvironment to HTTP/WebSocket transport.
Optionally mounts a Gradio /web UI via gradio_builder for the live demo.

Entry point referenced by openenv.yaml:
    server: entry_point: server.app:app
"""

from __future__ import annotations

import os
from typing import Any

# OpenEnv imports — confirmed APIs per plan §12
_openenv_import_error: str | None = None
try:
    from openenv.core import create_app, ConcurrencyConfig, ServerMode  # type: ignore
except ImportError as _e:
    _openenv_import_error = repr(_e)
    # Compatibility HTTP adapter when openenv.core create_app is unavailable.
    def create_app(env, action_cls, observation_cls, env_name, **kwargs):  # type: ignore
        from fastapi import Body, FastAPI, HTTPException
        app = FastAPI(title=env_name)

        def _to_jsonable(value):
            if hasattr(value, "model_dump"):
                return _to_jsonable(value.model_dump())
            if isinstance(value, dict):
                return {k: _to_jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_jsonable(v) for v in value]
            return value

        @app.get("/health")
        def health():
            return {
                "ok": True,
                "env": env_name,
                "stub": False,
                "compat_adapter": True,
                "import_error": _openenv_import_error,
            }

        @app.post("/reset")
        def reset(payload: dict[str, Any] = Body(default_factory=dict)):
            try:
                obs = env.reset(seed=payload.get("seed"))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"reset failed: {exc!r}") from exc
            return _to_jsonable(obs)

        @app.post("/step")
        def step(payload: dict[str, Any] = Body(default_factory=dict)):
            try:
                action = action_cls(
                    tool_name=payload.get("tool_name", ""),
                    tool_args=payload.get("tool_args", {}) or {},
                    reasoning_trace=payload.get("reasoning_trace"),
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid action payload: {exc}") from exc
            try:
                result = env.step(action)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"step failed: {exc!r}") from exc
            return _to_jsonable(result)

        @app.get("/state")
        def state():
            try:
                return _to_jsonable(env.state())
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"state failed: {exc!r}") from exc

        @app.post("/close")
        def close():
            env.close()
            return {"ok": True}

        return app

    class ConcurrencyConfig:  # type: ignore
        def __init__(self, max_concurrent_envs=8, session_timeout=300):
            self.max_concurrent_envs = max_concurrent_envs
            self.session_timeout = session_timeout

    class ServerMode:  # type: ignore
        SIMULATION = "simulation"
        PRODUCTION = "production"


from models import OptimizationAction, OptimizationObservation
from server.environment import PolyglotOptimaEnvironment


def build_gradio_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Custom Gradio /web UI for the live Polyglot-Optima demo.

    Wired into create_app() via the gradio_builder parameter (per plan §12 F).
    Full implementation lives in Hour 42-48; for now this returns a minimal
    Blocks instance so the framework's web-interface mount succeeds.
    """
    try:
        import gradio as gr
    except ImportError:
        return None

    with gr.Blocks(title="Polyglot-Optima — Python → Optimized C++") as demo:
        gr.Markdown(f"# {title}\n\n{quick_start_md or ''}")
        gr.Markdown(
            "**Status**: Skeleton (Hour 0-4). The live demo (paste Python → see C++ + speedup) "
            "ships in Hour 42-48 of the build."
        )
        with gr.Row():
            gr.Code(
                label="Paste Python function",
                language="python",
                value="def sum_squares(arr):\n    total = 0\n    for x in arr:\n        total += x * x\n    return total\n",
            )
            gr.Code(label="Agent's optimized C++", language="cpp", value="// Coming soon")
        gr.Button("Optimize", interactive=False)
        gr.Markdown("_Demo wires up in Hour 42-48 — current build is the skeleton._")

    return demo


def build_app() -> Any:
    """Build and return the FastAPI app (OpenEnv create_app pattern)."""
    enable_adaptive_curriculum = os.environ.get("POLYGLOT_OPTIMA_ENABLE_ADAPTIVE_CURRICULUM", "1") == "1"
    curriculum_batch_size = int(os.environ.get("POLYGLOT_OPTIMA_CURRICULUM_BATCH_SIZE", "8"))
    env = PolyglotOptimaEnvironment(
        max_rounds=3,
        max_calls_per_round=5,
        enable_adaptive_curriculum=enable_adaptive_curriculum,
        curriculum_batch_size=curriculum_batch_size,
    )

    server_mode_str = os.environ.get("OPENENV_SERVER_MODE", "simulation").lower()
    server_mode = ServerMode.PRODUCTION if server_mode_str == "production" else ServerMode.SIMULATION

    enable_web = os.environ.get("ENABLE_WEB_INTERFACE", "1") == "1"

    app = create_app(
        env=env,
        action_cls=OptimizationAction,
        observation_cls=OptimizationObservation,
        env_name="polyglot-optima",
        max_concurrent_envs=8,
        session_timeout=600,
        server_mode=server_mode,
        gradio_builder=build_gradio_ui if enable_web else None,
    )
    return app


# OpenEnv discovers the FastAPI instance via this module-level binding
app = build_app()
