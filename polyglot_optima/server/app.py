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
try:
    from openenv.core import create_app, ConcurrencyConfig, ServerMode  # type: ignore
except ImportError:
    # Fallback factory for local development before openenv is installed
    def create_app(env, action_cls, observation_cls, env_name, **kwargs):  # type: ignore
        from fastapi import FastAPI
        app = FastAPI(title=env_name)

        @app.get("/health")
        def health():
            return {"ok": True, "env": env_name, "stub": True}

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
            python_input = gr.Code(
                label="Paste Python function",
                language="python",
                value="def sum_squares(arr):\n    total = 0\n    for x in arr:\n        total += x * x\n    return total\n",
            )
            cpp_output = gr.Code(label="Agent's optimized C++", language="cpp", value="// Coming soon")
        gr.Button("Optimize", interactive=False)
        gr.Markdown("_Demo wires up in Hour 42-48 — current build is the skeleton._")

    return demo


def build_app() -> Any:
    """Build and return the FastAPI app (OpenEnv create_app pattern)."""
    env = PolyglotOptimaEnvironment(
        max_rounds=3,
        max_calls_per_round=5,
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
