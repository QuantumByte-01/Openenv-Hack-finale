"""MCP tool registry for Polyglot-Optima.

Exposes 9 tools per plan §9. The TOOL_REGISTRY dict is loaded by the environment
at startup and dispatched from PolyglotOptimaEnvironment._dispatch_tool.

Each tool is a plain Python callable (tool_args: dict, state: OptimizationState) -> dict.
The @tool decorator (Hour 22 deployment-time wrapper) adds Pydantic schema
validation, mode tagging, and async dispatch — for now, plain functions.
"""

from __future__ import annotations

from .hardware_profiler import get_hardware_profile_tool
from .python_analyzer import (
    profile_python_hotspots_tool,
    analyze_complexity_tool,
    check_memory_access_tool,
)
from .cpp_compiler import compile_and_benchmark_tool
from .verifier import verify_equivalence_tool
from .portability_checker import check_portability_tool
from .bottleneck_reporter import get_bottleneck_report_tool
from .submit import submit_optimization_tool


TOOL_REGISTRY = {
    "get_hardware_profile":     get_hardware_profile_tool,
    "profile_python_hotspots":  profile_python_hotspots_tool,
    "analyze_complexity":       analyze_complexity_tool,
    "check_memory_access":      check_memory_access_tool,
    "compile_and_benchmark":    compile_and_benchmark_tool,
    "verify_equivalence":       verify_equivalence_tool,
    "check_portability":        check_portability_tool,
    "get_bottleneck_report":    get_bottleneck_report_tool,
    "submit_optimization":      submit_optimization_tool,
}


__all__ = ["TOOL_REGISTRY"]
