"""Tools 2-4/9: profile_python_hotspots, analyze_complexity, check_memory_access.

Three static-analysis tools the agent uses to *understand the input code* before
writing C++. All run on the AST — no Python execution required for these tools
(the verifier and benchmarker do the actual execution, sandboxed).
"""

from __future__ import annotations

import ast
import re
from typing import Any


# ----------------- Tool 2: profile_python_hotspots ----------------

def profile_python_hotspots_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Return the top hot lines of the Python function (static cost estimate).

    For a static-analysis-only tool, we approximate hotness via:
      - loop nesting depth at the line
      - operations inside loops (multiplied by estimated trip count)
      - presence of np.* calls (vectorized but still expensive on large arrays)

    For a more accurate dynamic profile (cProfile run), pass `dynamic=True` —
    that path will be wired to a sandboxed run in Hour 16+.
    """
    code = tool_args.get("code") or state.python_code

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Python parse error: {e}", "hotspots": []}

    hotspots: list[dict[str, Any]] = []
    line_costs: dict[int, int] = {}

    class HotspotVisitor(ast.NodeVisitor):
        def __init__(self):
            self.loop_depth = 0

        def visit_For(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_While(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_BinOp(self, node):
            cost = 1 << self.loop_depth  # 2^depth — exponential weight per nesting
            line_costs[node.lineno] = line_costs.get(node.lineno, 0) + cost
            self.generic_visit(node)

        def visit_Call(self, node):
            # Penalize np.* calls inside loops more
            cost = (1 << self.loop_depth) * 2
            line_costs[node.lineno] = line_costs.get(node.lineno, 0) + cost
            self.generic_visit(node)

    HotspotVisitor().visit(tree)

    code_lines = code.splitlines()
    sorted_lines = sorted(line_costs.items(), key=lambda x: -x[1])
    for lineno, cost in sorted_lines[:5]:
        if 0 < lineno <= len(code_lines):
            hotspots.append({
                "line_number": lineno,
                "estimated_cost": cost,
                "source": code_lines[lineno - 1].strip(),
            })

    total_cost = sum(line_costs.values())
    return {
        "hotspots": hotspots,
        "total_estimated_cost": total_cost,
        "method": "static_ast_analysis",
        "hint": "Lines deep in loops dominate; vectorize or parallelize them first.",
    }


# ----------------- Tool 3: analyze_complexity ----------------

def analyze_complexity_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Return Big-O class + max loop nesting depth via AST.

    A loop nesting depth of k suggests O(n^k) in the typical case. Recursion
    detection is naive (treats every recursive call as +1 to complexity).
    """
    code = tool_args.get("code") or state.python_code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Python parse error: {e}"}

    max_depth = [0]

    class DepthVisitor(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0

        def visit_For(self, node):
            self.depth += 1
            max_depth[0] = max(max_depth[0], self.depth)
            self.generic_visit(node)
            self.depth -= 1

        def visit_While(self, node):
            self.depth += 1
            max_depth[0] = max(max_depth[0], self.depth)
            self.generic_visit(node)
            self.depth -= 1

    DepthVisitor().visit(tree)

    depth = max_depth[0]
    if depth == 0:
        big_o = "O(1)"
    elif depth == 1:
        big_o = "O(n)"
    else:
        big_o = f"O(n^{depth})"

    # Detect simple recursion (function calls itself)
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    has_recursion = any(
        isinstance(c.func, ast.Name) and c.func.id in func_names
        for c in ast.walk(tree) if isinstance(c, ast.Call)
    )

    return {
        "big_o_estimate": big_o,
        "max_loop_nesting_depth": depth,
        "has_recursion": has_recursion,
        "method": "static_ast_loop_depth",
    }


# ----------------- Tool 4: check_memory_access ----------------

# Patterns that suggest cache-unfriendly access
_STRIDE_PATTERN = re.compile(r"\[\s*j\s*,\s*i\s*\]|\[\s*i\s*\]\s*\[\s*j\s*\]")
_TRANSPOSE_PATTERN = re.compile(r"\.T\s*\[")
_NON_CONTIG_PATTERN = re.compile(r"\bnp\.ascontiguousarray\b|\bnp\.asfortranarray\b")


def check_memory_access_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Detect cache-unfriendly stride patterns / aliasing risks via static patterns.

    This is a heuristic — not perfect, but catches the common cases:
      - column-major access in row-major arrays (D[j, i] inside i,j loops)
      - non-contiguous arrays passed in
      - explicit transpose in hot expression
    """
    code = tool_args.get("code") or state.python_code

    issues: list[dict[str, str]] = []

    if _STRIDE_PATTERN.search(code):
        issues.append({
            "type": "non_unit_stride",
            "severity": "high",
            "hint": "Detected D[j,i]-style access — likely column-major in a row-major array. "
                    "Cache misses dominate. Transpose the layout or swap loop order."
        })
    if _TRANSPOSE_PATTERN.search(code):
        issues.append({
            "type": "in_loop_transpose",
            "severity": "med",
            "hint": "`.T` in hot path may force a copy or non-contiguous access."
        })
    if _NON_CONTIG_PATTERN.search(code):
        issues.append({
            "type": "explicit_layout_handling",
            "severity": "info",
            "hint": "Code already handles contiguity — good; preserve in C++ via `restrict`."
        })

    # Inspect AST for "for i in range" + "for j in range" + a 2D index
    try:
        tree = ast.parse(code)
        nested_for = False
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.For) and sub is not node:
                        nested_for = True
                        break
        if nested_for and not issues:
            issues.append({
                "type": "nested_loop_unanalyzed",
                "severity": "low",
                "hint": "Nested loops detected. Verify that inner-loop index varies the contiguous dimension."
            })
    except SyntaxError:
        pass

    aliasing_risk = "low"
    if "np.ndarray" in code or "ndarray" in code:
        aliasing_risk = "med"  # numpy arrays can alias; agent should consider `restrict`

    return {
        "issues": issues,
        "aliasing_risk": aliasing_risk,
        "recommendation": (
            "Use `__restrict__` qualifier on non-aliasing pointers in C++. "
            "Prefer SoA over AoS for SIMD-friendly access."
            if issues else "No obvious memory-access issues; proceed with default layout."
        ),
    }


__all__ = [
    "profile_python_hotspots_tool",
    "analyze_complexity_tool",
    "check_memory_access_tool",
]
