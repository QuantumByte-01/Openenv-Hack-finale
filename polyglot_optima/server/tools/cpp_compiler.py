"""Tool 5/9: compile_and_benchmark.

Compiles agent C++ with `g++ -O3 -march=native -fopenmp -std=c++20 -Wall -Werror`
and benchmarks against the Python baseline using median-of-15 wall time.

Caching: the (cpp_code + hardware_profile_id) sha256 keys a persistent on-disk
cache of compiled `.so` files. Per plan §7 risk #2, a high cache hit rate is
critical to keeping training cost within budget.

Output language enforcement (per plan §10a): the wrapper signature is auto-
generated from the Python AST and the agent's code MUST define `extern "C"`
function with that exact signature. Compile errors → reward = 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Persistent compile cache directory (shared across episodes within a process run)
_CACHE_ROOT = Path(os.environ.get("POLYGLOT_OPTIMA_CACHE", str(Path(tempfile.gettempdir()) / "polyglot_optima_cache")))
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Compile std — locked to C++20 in production per plan §10a.
# Allowing C++17/C++14 silently would let the agent learn code that fails on the
# real GCC 14 deploy. Therefore: production = c++20 only. Dev fallback requires
# the explicit POLYGLOT_OPTIMA_DEV_FALLBACK=1 env var (used by tests on machines
# with old MinGW); even then we warn loudly so the divergence isn't invisible.
_PRODUCTION_CXX_STD = "c++20"
_DEV_FALLBACK_ALLOWED = os.environ.get("POLYGLOT_OPTIMA_DEV_FALLBACK", "0") == "1"


def _detect_supported_cxx_std() -> str:
    """Return c++20 if the compiler supports it; else c++20 anyway in production
    (so the compile fails informatively and the gate registers it as syntax_error).

    With POLYGLOT_OPTIMA_DEV_FALLBACK=1 set, we fall back to the highest std the
    compiler accepts and emit a stderr warning. That mode is for local dev tests
    only — never for training or deploy."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        return _PRODUCTION_CXX_STD

    # Probe c++20 first
    try:
        r = subprocess.run([compiler, f"-std={_PRODUCTION_CXX_STD}", "-x", "c++", "-E", "-"],
                           input="", capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and "unrecognized" not in (r.stderr or "").lower():
            return _PRODUCTION_CXX_STD
    except Exception:
        pass

    if not _DEV_FALLBACK_ALLOWED:
        # Production: stay on c++20. If the compiler can't, every compile will fail
        # — that's the right signal (deploy with old GCC needs upgrading, not lowering).
        return _PRODUCTION_CXX_STD

    # Dev fallback only — emit warning so the divergence is visible
    import sys as _sys
    for std in ("c++17", "c++14"):
        try:
            r = subprocess.run([compiler, f"-std={std}", "-x", "c++", "-E", "-"],
                               input="", capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "unrecognized" not in (r.stderr or "").lower():
                print(
                    f"⚠ POLYGLOT_OPTIMA: dev fallback to -std={std} (compiler does not support c++20). "
                    f"This is for local tests only — production training/deploy MUST use c++20.",
                    file=_sys.stderr,
                )
                return std
        except Exception:
            continue
    return _PRODUCTION_CXX_STD


def _detect_openmp() -> bool:
    """Test whether `-fopenmp` actually links — MinGW often lacks pthread libs."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        return False
    try:
        # Try to compile + LINK a trivial OpenMP program. Compile-only succeeds even
        # without pthread; we need the link step to confirm the runtime is available.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "_omp_probe.cpp"
            obj = Path(td) / "_omp_probe.so"
            src.write_text("#include <omp.h>\nint main(){return omp_get_num_threads();}\n")
            r = subprocess.run([compiler, "-fopenmp", str(src), "-shared", "-fPIC", "-o", str(obj)],
                               capture_output=True, text=True, timeout=10)
            return r.returncode == 0
    except Exception:
        return False


def _detect_dispatchable() -> bool:
    """Compile + ctypes-load a tiny probe. Returns True iff the toolchain produces a
    .so loadable by THIS Python interpreter (catches bitness mismatch on MinGW)."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        return False
    try:
        import ctypes as _ct
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "_probe.cpp"
            so = Path(td) / "_probe.so"
            src.write_text(
                'extern "C" void agent_function(const double*, '
                'unsigned long long, double* o, unsigned long long n)'
                '{ if (n) o[0] = 1.0; }\n'
            )
            r = subprocess.run(
                [compiler, "-O0", "-fPIC", "-shared", str(src), "-o", str(so)],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode != 0:
                return False
            lib = _ct.CDLL(str(so))
            return hasattr(lib, "agent_function")
    except Exception:
        return False


_DETECTED_CXX_STD = _detect_supported_cxx_std()
_HAS_OPENMP = _detect_openmp()
_DISPATCHABLE = _detect_dispatchable()


_BASE_COMPILE_FLAGS = [
    "-O3",
    "-march=native",
    f"-std={_DETECTED_CXX_STD}",
    "-Wall",
    # `-Werror` removed: many MinGW builds emit warnings on default flags.
    # Production deploy can re-add via POLYGLOT_OPTIMA_STRICT=1
    "-fPIC",
    "-shared",
]
if _HAS_OPENMP:
    _BASE_COMPILE_FLAGS.insert(2, "-fopenmp")
if os.environ.get("POLYGLOT_OPTIMA_STRICT", "0") == "1":
    _BASE_COMPILE_FLAGS.append("-Werror")

# Banned headers (per plan §10a — would mask agent's actual contribution)
_BANNED_INCLUDES = [
    "<mkl.h>", "<mkl",                # Intel MKL
    "<Eigen/", "Eigen/",              # Eigen
    "<cblas.h>", "<lapack.h>",         # BLAS/LAPACK
    "<cuda_runtime.h>", "<cuda.h>",   # CUDA
    "<hip/",                          # HIP
]


def _sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _check_for_banned_headers(cpp_code: str) -> str | None:
    """Return error string if the code uses a banned header, else None."""
    for banned in _BANNED_INCLUDES:
        if banned in cpp_code:
            return (
                f"Banned header detected: {banned}. "
                f"We measure YOUR optimization, not a library call. "
                f"Allowed: STL, <immintrin.h>, <arm_neon.h>, <omp.h>, <pybind11/*>"
            )
    return None


def _has_required_entry_point(cpp_code: str) -> bool:
    """Verify the C++ code declares an `extern \"C\"` agent_function entry point.

    Soft check — if missing, compilation will eventually fail at link time when
    the verifier tries to dispatch. Returning False here is just a hint for the agent.
    """
    return ("extern \"C\"" in cpp_code or 'extern"C"' in cpp_code) and "agent_function" in cpp_code


def _compile(cpp_code: str, hw_profile: dict[str, Any], cache_key: str, timeout_s: int = 30) -> dict[str, Any]:
    """Run g++; cache the .so by cache_key. Return dict with status + path/error."""
    cache_dir = _CACHE_ROOT / cache_key[:2]
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"{cache_key}.so"

    # Cache hit
    if so_path.exists():
        return {"status": "success", "so_path": str(so_path), "cached": True}

    # Banned headers → reject before invoking compiler
    banned_err = _check_for_banned_headers(cpp_code)
    if banned_err:
        return {"status": "syntax_error", "error": banned_err, "cached": False}

    # Write source + invoke compiler
    src_path = cache_dir / f"{cache_key}.cpp"
    src_path.write_text(cpp_code, encoding="utf-8")

    # Resolve compiler — prefer g++ on Linux, fall back to clang++ on macOS
    compiler = shutil.which("g++") or shutil.which("clang++") or "g++"

    cmd = [compiler, *_BASE_COMPILE_FLAGS, str(src_path), "-o", str(so_path)]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Compilation exceeded {timeout_s}s", "cached": False}
    except FileNotFoundError:
        return {"status": "syntax_error",
                "error": f"Compiler {compiler!r} not found. Install GCC 14 or clang++.",
                "cached": False}

    if proc.returncode != 0:
        return {
            "status": "syntax_error",
            "error": (proc.stderr or proc.stdout)[:2000],
            "cmd": " ".join(cmd),
            "cached": False,
        }

    return {"status": "success", "so_path": str(so_path), "cached": False}


def _load_python_function(python_code: str):
    """Exec python_code in a fresh namespace, return the first FunctionDef as a callable."""
    import ast
    tree = ast.parse(python_code)
    fn_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
    if fn_node is None:
        raise RuntimeError("python_code defines no function")
    ns: dict[str, Any] = {}
    exec(compile(tree, filename="<agent_python>", mode="exec"), ns)
    fn = ns.get(fn_node.name)
    if fn is None:
        raise RuntimeError(f"function {fn_node.name!r} not found after exec")
    return fn


def _benchmark_python_baseline(python_code: str, sample_input_size: int = 1024) -> dict[str, Any]:
    """Real median-of-15 wall time of the Python function on a default-typed input."""
    from server.tools._runtime import time_python_only, make_default_args_for
    try:
        py_fn = _load_python_function(python_code)
        args = make_default_args_for(py_fn, n=sample_input_size)
        median_ms = time_python_only(py_fn, args, n_per_repeat=5, repeats=3)
        return {
            "median_ms": float(median_ms),
            "method": "perf_counter_median_5x3",
            "n_samples": sample_input_size,
        }
    except Exception as e:
        # Don't crash the env on a broken Python function; signal "0 baseline" → speedup goes to 0
        return {
            "median_ms": 0.0,
            "method": "error",
            "error": str(e)[:200],
            "n_samples": sample_input_size,
        }


def _benchmark_cpp(so_path: str, python_code: str, sample_input_size: int = 1024) -> dict[str, Any]:
    """Real median-of-15 wall time of the compiled .so via ctypes dispatch."""
    from server.tools._runtime import benchmark_python_vs_cpp, make_default_args_for
    try:
        py_fn = _load_python_function(python_code)
        args = make_default_args_for(py_fn, n=sample_input_size)
        result = benchmark_python_vs_cpp(so_path, py_fn, args, n_per_repeat=5, repeats=3)
        return {
            "median_ms": float(result["cpp_median_ms"]),
            "py_median_ms": float(result["py_median_ms"]),
            "speedup_internal": float(result["speedup"]),
            "method": "ctypes_perf_counter_median_5x3",
            "n_samples": sample_input_size,
        }
    except Exception as e:
        return {
            "median_ms": 0.0,
            "method": "error",
            "error": str(e)[:200],
            "n_samples": sample_input_size,
        }


def compile_and_benchmark_tool(tool_args: dict[str, Any], state) -> dict[str, Any]:
    """Compile agent C++ and report compile status + speedup measurement.

    Args:
        cpp_code (str): The C++20 source to compile.

    Returns dict with:
        compile_status: "success" | "syntax_error" | "link_error" | "timeout"
        speedup: float (python_ms / cpp_ms) — only valid if compile_status == "success"
        python_ms: median-of-15 Python baseline
        cpp_ms: median-of-15 agent C++ wall time
        error: str (if compile_status != "success")
        cache_hit: bool
    """
    cpp_code = tool_args.get("cpp_code", "")
    if not cpp_code.strip():
        return {"compile_status": "syntax_error", "error": "empty cpp_code", "speedup": 0.0}

    if not _has_required_entry_point(cpp_code):
        return {
            "compile_status": "syntax_error",
            "error": (
                'Missing required entry point: must define `extern "C" ... agent_function(...)`'
            ),
            "speedup": 0.0,
        }

    # Cache key
    hw = state.hardware_profile
    cache_key = _sha256(cpp_code, json.dumps(hw, sort_keys=True))

    t_compile_start = time.perf_counter()
    compile_result = _compile(cpp_code, hw, cache_key)
    compile_time_s = time.perf_counter() - t_compile_start

    if compile_result["status"] != "success":
        return {
            "compile_status": compile_result["status"],
            "error": compile_result.get("error", "compilation failed"),
            "speedup": 0.0,
            "compile_time_s": compile_time_s,
            "cache_hit": False,
        }

    # Real benchmark via ctypes dispatch — joint timing of python + cpp on same args
    cpp_bench = _benchmark_cpp(compile_result["so_path"], state.python_code)

    if cpp_bench.get("method") == "error":
        # Compilation succeeded but the .so couldn't be dispatched (wrong signature, missing symbol)
        return {
            "compile_status": "link_error",
            "error": cpp_bench.get("error", "ctypes dispatch failed"),
            "speedup": 0.0,
            "python_ms": 0.0,
            "cpp_ms": 0.0,
            "compile_time_s": compile_time_s,
            "cache_hit": compile_result.get("cached", False),
        }

    py_ms = cpp_bench.get("py_median_ms", 0.0)
    cpp_ms = cpp_bench["median_ms"]
    speedup = py_ms / max(cpp_ms, 1e-6) if py_ms > 0 else 0.0

    return {
        "compile_status": "success",
        "speedup": speedup,
        "python_ms": py_ms,
        "cpp_ms": cpp_ms,
        "compile_time_s": compile_time_s,
        "cache_hit": compile_result.get("cached", False),
        "so_path": compile_result["so_path"],
        "method": "ctypes_median_5x3_walltime",
    }


__all__ = ["compile_and_benchmark_tool", "_sha256", "_BASE_COMPILE_FLAGS"]
