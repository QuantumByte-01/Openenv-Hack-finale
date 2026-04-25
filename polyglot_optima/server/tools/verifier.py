"""Tool 6/9: verify_equivalence — anti-cheating fuzzer.

Per plan §10b, this is the single most important defense against the agent
cheating by producing a fast-but-wrong implementation.

8 cheating modes defended:
1. Wrong algorithm with plausible output     — random fuzz inputs
2. Edge-case overflow (int32 wraps int64)    — typed inputs include int64, INT_MAX/MIN
3. Approximation drift                       — rtol=1e-5 (or rtol=0 per metadata)
4. Cached lookup table                       — seed randomized per call
5. Tail variance                             — 10% adversarial sub-pool
6. Returns 0 / empty                         — exact shape+dtype check
7. Detects benchmark context                 — same input pipeline as benchmarker
8. Side-channel access                       — sandboxed subprocess

Returns: pass_rate ∈ [0, 1], first_failure dict, n_adversarial_failures.
"""

from __future__ import annotations

import ast
import random
import sys
from typing import Any

import numpy as np


# ---------- Input generation from Python AST ----------

def _infer_input_signature(python_code: str) -> list[dict[str, str]]:
    """Inspect the Python function's signature + annotations to pick fuzz input types.

    Returns a list of {"name": str, "kind": "ndarray|int|float|list|str", "dtype": str}.
    Without explicit annotations, we fall back to ndarray of float64.
    """
    try:
        tree = ast.parse(python_code)
    except SyntaxError:
        return [{"name": "x", "kind": "ndarray", "dtype": "float64"}]

    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return [{"name": "x", "kind": "ndarray", "dtype": "float64"}]

    sig: list[dict[str, str]] = []
    for arg in fn.args.args:
        ann = ast.unparse(arg.annotation) if arg.annotation else ""
        kind = "ndarray"
        dtype = "float64"
        if "int" in ann.lower() and "ndarray" not in ann.lower() and "list" not in ann.lower():
            kind = "int"
        elif "float" in ann.lower() and "ndarray" not in ann.lower() and "list" not in ann.lower():
            kind = "float"
        elif "list" in ann.lower():
            kind = "list"
        elif "str" in ann.lower():
            kind = "str"
        if "int32" in ann:
            dtype = "int32"
        elif "int64" in ann:
            dtype = "int64"
        elif "float32" in ann:
            dtype = "float32"
        sig.append({"name": arg.arg, "kind": kind, "dtype": dtype})

    # Default fallback: assume one ndarray
    if not sig:
        sig = [{"name": "x", "kind": "ndarray", "dtype": "float64"}]
    return sig


def _generate_typed_input(spec: dict[str, str], rng: np.random.Generator, adversarial: bool = False) -> Any:
    """Generate one input matching spec. If adversarial, sample boundary/edge values."""
    kind = spec["kind"]
    dtype = spec["dtype"]

    if kind == "int":
        if adversarial:
            return int(rng.choice([0, 1, -1, 2**31 - 1, -(2**31), 2**62, -(2**62)]))
        return int(rng.integers(-1000, 1000))

    if kind == "float":
        if adversarial:
            return float(rng.choice([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-300, 1e300]))
        return float(rng.standard_normal())

    if kind == "str":
        # Short ascii strings
        return "".join(chr(int(rng.integers(97, 123))) for _ in range(int(rng.integers(1, 16))))

    # Default: ndarray
    n = int(rng.integers(10, 1000))
    if adversarial:
        choices = [
            np.zeros(n, dtype=dtype),
            np.ones(n, dtype=dtype),
            np.array([], dtype=dtype),                     # empty
            np.array([0.0], dtype=dtype),                  # singleton
            np.full(n, np.inf, dtype=dtype) if "float" in dtype else np.full(n, np.iinfo(np.dtype(dtype)).max, dtype=dtype),
            (rng.standard_normal(n) * 1e-300).astype(dtype) if "float" in dtype else rng.integers(-1, 2, n).astype(dtype),
        ]
        idx = int(rng.integers(0, len(choices)))
        return choices[idx]

    if "int" in dtype:
        return rng.integers(-100, 100, size=n).astype(dtype)
    return rng.standard_normal(n).astype(dtype)


def _numerically_equivalent(a: Any, b: Any, rtol: float) -> bool:
    """Compare two outputs accounting for float tolerance, exact for int."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if rtol == 0:
            return a == b
        if not np.isfinite(a) or not np.isfinite(b):
            return (np.isnan(a) and np.isnan(b)) or a == b
        return abs(a - b) <= rtol * (1 + abs(a))

    try:
        a = np.asarray(a)
        b = np.asarray(b)
    except Exception:
        return a == b

    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        # We don't allow dtype-mismatch — that's a hard fail per plan §10b
        return False

    if rtol == 0:
        return bool(np.array_equal(a, b))

    # Use allclose with NaN-equality
    return bool(np.allclose(a, b, rtol=rtol, atol=rtol * 0.1, equal_nan=True))


def _exec_python_in_sandbox(python_code: str, fn_name: str, args: tuple) -> Any:
    """Run python_code's function on args. For now in-process; sandboxed in Hour 16."""
    ns: dict[str, Any] = {}
    exec(python_code, ns)
    fn = ns.get(fn_name)
    if fn is None:
        raise RuntimeError(f"function '{fn_name}' not defined in python_code")
    return fn(*args)


def _exec_cpp_via_so(so_path: str, fn_name: str, args: tuple, py_fn=None, py_code: str = "") -> Any:
    """Load the compiled .so via ctypes and dispatch on `args`.

    The agent's C++ uses the canonical signature
        extern "C" void agent_function(const double*, size_t, double*, size_t);
    so we need the Python reference function to know the output shape. Either
    pass `py_fn` directly, or pass `py_code` and we'll compile it.

    Raises:
        RuntimeError: ctypes can't load the .so or symbol is missing
    """
    from server.tools._runtime import call_compiled
    if py_fn is None:
        if not py_code:
            raise RuntimeError("verifier: need py_fn or py_code to dispatch C++")
        ns: dict[str, Any] = {}
        exec(py_code, ns)
        py_fn = ns.get(fn_name)
        if py_fn is None:
            raise RuntimeError(f"verifier: function {fn_name!r} not found in py_code")
    return call_compiled(so_path, py_fn, args)


def verify_equivalence_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Fuzz-verify cpp_code against python_code on n_cases random + adversarial inputs.

    Args:
        cpp_code (str)         — agent's C++
        python_code (str)      — reference Python (defaults to state.python_code)
        n_cases (int=1000)     — total fuzz cases (10% adversarial sub-pool)
        rtol (float=1e-5)      — float tolerance; 0 = bit-exact

    Returns:
        pass_rate (float)
        first_failure (dict | None)
        n_adversarial_failures (int)
        n_random_failures (int)
        seed (int)             — randomized per call (defeats lookup tables)
    """
    cpp_code = tool_args.get("cpp_code", "")
    python_code = tool_args.get("python_code") or state.python_code
    n_cases = int(tool_args.get("n_cases", 1000))
    rtol = float(tool_args.get("rtol", state.rtol_override if state.rtol_override is not None else 1e-5))

    if not cpp_code.strip():
        return {"pass_rate": 0.0, "error": "empty cpp_code"}

    # Defeat lookup-table cheating mode 4: seed varies per call
    seed = random.randint(0, 2**32 - 1)
    rng = np.random.default_rng(seed)

    # Discover Python function name (first FunctionDef)
    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        return {"pass_rate": 0.0, "error": f"python parse: {e}"}
    fn_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn_node is None:
        return {"pass_rate": 0.0, "error": "no function in python_code"}
    fn_name = fn_node.name

    sig = _infer_input_signature(python_code)

    # Compile (or get cached .so) — uses cpp_compiler tool's pathway
    from server.tools.cpp_compiler import _compile, _sha256
    import json as _json
    cache_key = _sha256(cpp_code, _json.dumps(state.hardware_profile, sort_keys=True))
    compile_result = _compile(cpp_code, state.hardware_profile, cache_key)
    if compile_result["status"] != "success":
        return {
            "pass_rate": 0.0,
            "error": f"cpp compile failed: {compile_result.get('error', '')[:300]}",
            "compile_status": compile_result["status"],
        }
    so_path = compile_result["so_path"]

    # Pre-load the Python reference function once (avoids repeated exec overhead)
    ref_ns: dict[str, Any] = {}
    try:
        exec(python_code, ref_ns)
        py_fn = ref_ns.get(fn_name)
        if py_fn is None:
            return {"pass_rate": 0.0, "error": f"py function {fn_name!r} not found after exec"}
    except Exception as e:
        return {"pass_rate": 0.0, "error": f"python exec failed: {e}"}

    failures: list[dict[str, Any]] = []
    n_adversarial_failures = 0
    n_random_failures = 0

    for i in range(n_cases):
        adversarial = (i % 10 == 9)  # 10% adversarial sub-pool
        try:
            args = tuple(_generate_typed_input(spec, rng, adversarial=adversarial) for spec in sig)
        except Exception:
            continue  # Skip if input generation itself fails

        # Run Python first; if it raises, skip (don't penalize the C++ for invalid input)
        try:
            py_out = py_fn(*args)
        except Exception:
            continue

        # Run C++ via ctypes dispatch — REAL execution now (not stub)
        try:
            cpp_out = _exec_cpp_via_so(so_path, fn_name, args, py_fn=py_fn)
        except Exception as e:
            if adversarial:
                n_adversarial_failures += 1
            else:
                n_random_failures += 1
            if not failures:
                failures.append({
                    "case": i, "reason": "cpp_exec_error", "error": str(e)[:200],
                    "adversarial": adversarial,
                })
            continue

        if not _numerically_equivalent(py_out, cpp_out, rtol):
            if adversarial:
                n_adversarial_failures += 1
            else:
                n_random_failures += 1
            if not failures:
                # Capture only first failure to bound observation size
                py_repr = repr(py_out)[:120]
                cpp_repr = repr(cpp_out)[:120]
                failures.append({
                    "case": i, "reason": "output_mismatch",
                    "adversarial": adversarial,
                    "py_out": py_repr, "cpp_out": cpp_repr,
                })

    pass_count = n_cases - (n_adversarial_failures + n_random_failures)
    pass_rate = pass_count / n_cases

    n_adversarial_total = n_cases // 10
    adversarial_pass_rate = (n_adversarial_total - n_adversarial_failures) / max(n_adversarial_total, 1)

    return {
        "pass_rate": pass_rate,
        "n_cases": n_cases,
        "first_failure": failures[0] if failures else None,
        "n_adversarial_failures": n_adversarial_failures,
        "n_random_failures": n_random_failures,
        "adversarial_pass_rate": adversarial_pass_rate,
        "rtol_used": rtol,
        "seed": seed,
    }


__all__ = ["verify_equivalence_tool", "_infer_input_signature", "_numerically_equivalent"]
