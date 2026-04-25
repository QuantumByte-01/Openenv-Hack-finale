"""ctypes-based runtime dispatch for compiled agent C++.

Replaces the Hour 4-10 stubs in cpp_compiler._benchmark_cpp and verifier._exec_cpp_via_so
with real measurement.

Canonical agent function signature (system-prompted, enforced by all training data):

    extern "C" void agent_function(
        const double* in_ptr,    // flattened input (all args concatenated to float64)
        size_t in_n,             // total input length
        double* out_ptr,         // preallocated output buffer (caller-allocated, agent fills)
        size_t out_n             // output buffer size
    );

This uniform signature trades some type richness (everything's float64) for:
- Simple ctypes binding (no per-function ABI generation)
- Trivial for the agent to write
- Covers all numeric training functions (sklearn loops, NumPy ops, math kernels)

Inputs/outputs are float64 (8 bytes). For integer functions we cast at the
boundary; for the few bit-exact integer functions in the trap library, the
fuzzer's `rtol=0` semantics still catch divergence (e.g., int overflow modes
that propagate as different float values).
"""

from __future__ import annotations

import ctypes
import time
from typing import Any, Callable

import numpy as np


# ---------------------- Argument marshalling ----------------------

def _flatten_args(args: tuple) -> tuple[np.ndarray, list]:
    """Concatenate all args into one flat float64 array; remember per-arg shapes for the agent.

    Returns:
        flat: a single contiguous float64 array (the in_ptr buffer)
        shapes: list of (kind, shape, dtype) for each arg — informational, not used by the
                ABI itself but useful for debugging
    """
    flats: list[np.ndarray] = []
    shapes: list[tuple] = []
    for a in args:
        if isinstance(a, np.ndarray):
            shapes.append(("ndarray", a.shape, a.dtype))
            flats.append(np.ascontiguousarray(a, dtype=np.float64).ravel())
        elif isinstance(a, (int, float, np.integer, np.floating)):
            shapes.append(("scalar", (), type(a)))
            flats.append(np.array([float(a)], dtype=np.float64))
        elif isinstance(a, (list, tuple)):
            arr = np.array(a, dtype=np.float64)
            shapes.append(("list", arr.shape, np.float64))
            flats.append(arr.ravel())
        else:
            raise TypeError(f"unsupported arg type for agent_function: {type(a).__name__}")
    if not flats:
        return np.array([], dtype=np.float64), shapes
    return np.concatenate(flats).astype(np.float64, copy=False), shapes


def _infer_output_meta(py_fn: Callable, args: tuple) -> dict[str, Any]:
    """Run py_fn once to discover output shape + dtype. Used to size the C++ output buffer."""
    out = py_fn(*args)
    if isinstance(out, (int, np.integer)):
        return {"kind": "int", "size": 1, "shape": (), "dtype": int}
    if isinstance(out, (float, np.floating)):
        return {"kind": "float", "size": 1, "shape": (), "dtype": float}
    if isinstance(out, np.ndarray):
        return {"kind": "ndarray", "size": int(out.size), "shape": tuple(out.shape), "dtype": out.dtype}
    if isinstance(out, (list, tuple)):
        arr = np.array(out, dtype=np.float64)
        return {"kind": "list", "size": int(arr.size), "shape": tuple(arr.shape), "dtype": np.float64}
    raise TypeError(f"unsupported py_fn output type: {type(out).__name__}")


def _reshape_cpp_output(out_arr: np.ndarray, meta: dict[str, Any]) -> Any:
    """Reshape the flat output buffer back to py_fn's original output kind/shape."""
    if meta["kind"] == "int":
        return int(round(float(out_arr[0])))
    if meta["kind"] == "float":
        return float(out_arr[0])
    if meta["kind"] == "ndarray":
        return out_arr[: meta["size"]].reshape(meta["shape"]).astype(meta["dtype"], copy=False)
    if meta["kind"] == "list":
        return out_arr[: meta["size"]].reshape(meta["shape"]).tolist()
    return out_arr


# ---------------------- .so loader (cached) ----------------------

class _SOLoader:
    """Cache loaded ctypes libraries by path. Each .so loaded only once."""
    _cache: dict[str, ctypes.CDLL] = {}

    @classmethod
    def load(cls, so_path: str) -> ctypes.CDLL:
        if so_path in cls._cache:
            return cls._cache[so_path]
        lib = ctypes.CDLL(so_path)
        if not hasattr(lib, "agent_function"):
            raise RuntimeError(f"{so_path} does not export `agent_function`")
        lib.agent_function.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # in_ptr
            ctypes.c_size_t,                  # in_n
            ctypes.POINTER(ctypes.c_double),  # out_ptr
            ctypes.c_size_t,                  # out_n
        ]
        lib.agent_function.restype = None
        cls._cache[so_path] = lib
        return lib

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


# ---------------------- Public dispatch API ----------------------

def call_compiled(so_path: str, py_fn: Callable, args: tuple) -> Any:
    """Call agent_function in the .so on args. Return value matches py_fn's output shape.

    Raises:
        RuntimeError: if .so can't be loaded or `agent_function` symbol is missing
    """
    lib = _SOLoader.load(so_path)

    in_flat, _ = _flatten_args(args)
    in_arr = np.ascontiguousarray(in_flat, dtype=np.float64)
    in_ptr = in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out_meta = _infer_output_meta(py_fn, args)
    out_arr = np.zeros(out_meta["size"], dtype=np.float64)
    out_ptr = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.agent_function(in_ptr, ctypes.c_size_t(in_arr.size),
                       out_ptr, ctypes.c_size_t(out_meta["size"]))

    return _reshape_cpp_output(out_arr, out_meta)


def benchmark_python_vs_cpp(
    so_path: str,
    py_fn: Callable,
    args: tuple,
    n_per_repeat: int = 5,
    repeats: int = 3,
) -> dict[str, float]:
    """Median-of-(repeats×n_per_repeat) wall time for both Python and C++ on the SAME args.

    Returns:
        py_median_ms: float — median ms per Python call
        cpp_median_ms: float — median ms per C++ call (via ctypes)
        speedup: float — py_median_ms / cpp_median_ms
    """
    lib = _SOLoader.load(so_path)

    # Pre-flatten inputs ONCE — re-flattening would pollute timing
    in_flat, _ = _flatten_args(args)
    in_arr = np.ascontiguousarray(in_flat, dtype=np.float64)
    in_ptr = in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    out_meta = _infer_output_meta(py_fn, args)
    out_arr = np.zeros(out_meta["size"], dtype=np.float64)
    out_ptr = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    in_n = ctypes.c_size_t(in_arr.size)
    out_n = ctypes.c_size_t(out_meta["size"])

    # ---- Python timing ----
    py_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_per_repeat):
            py_fn(*args)
        elapsed = time.perf_counter() - t0
        py_times.append((elapsed / n_per_repeat) * 1000)
    py_times.sort()
    py_median = py_times[len(py_times) // 2]

    # ---- C++ timing ----
    cpp_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_per_repeat):
            lib.agent_function(in_ptr, in_n, out_ptr, out_n)
        elapsed = time.perf_counter() - t0
        cpp_times.append((elapsed / n_per_repeat) * 1000)
    cpp_times.sort()
    cpp_median = cpp_times[len(cpp_times) // 2]

    return {
        "py_median_ms": py_median,
        "cpp_median_ms": cpp_median,
        "speedup": py_median / max(cpp_median, 1e-6),
        "n_per_repeat": n_per_repeat,
        "repeats": repeats,
    }


def time_python_only(py_fn: Callable, args: tuple, n_per_repeat: int = 5, repeats: int = 3) -> float:
    """Pure Python baseline timing (no .so needed). Returns median ms per call."""
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_per_repeat):
            py_fn(*args)
        times.append((time.perf_counter() - t0) / n_per_repeat * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------- Sample-input synthesizer ----------------------

def make_default_args_for(py_fn: Callable, n: int = 1024, seed: int = 0) -> tuple:
    """Construct a default (numeric ndarray + scalars) arg tuple for py_fn from its signature.

    Used for the benchmark baseline when no specific input is provided.
    Falls back to a 1024-element float64 array if introspection fails.
    """
    import inspect
    rng = np.random.default_rng(seed)
    try:
        sig = inspect.signature(py_fn)
        params = list(sig.parameters.values())
    except (ValueError, TypeError):
        return (rng.standard_normal(n).astype(np.float64),)

    out = []
    for p in params:
        ann = str(p.annotation).lower() if p.annotation is not inspect.Parameter.empty else ""
        default = p.default if p.default is not inspect.Parameter.empty else None
        if "int" in ann and "ndarray" not in ann and "list" not in ann:
            out.append(default if isinstance(default, int) else int(rng.integers(2, 16)))
        elif "float" in ann and "ndarray" not in ann and "list" not in ann:
            out.append(default if isinstance(default, float) else float(rng.standard_normal()))
        elif "list" in ann or "ndarray" in ann or ann == "":
            out.append(rng.standard_normal(n).astype(np.float64))
        elif "str" in ann:
            out.append("hello world")
        else:
            out.append(rng.standard_normal(n).astype(np.float64))
    return tuple(out)


__all__ = [
    "call_compiled",
    "benchmark_python_vs_cpp",
    "time_python_only",
    "make_default_args_for",
    "_SOLoader",
]
