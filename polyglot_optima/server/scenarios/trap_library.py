"""30 anti-gaming trap functions (per plan §10b).

Each trap is a Python function designed to fail naive C++ translation through
one of these failure modes:
    overflow    — Python int unbounded; C++ int wraps at 2^31
    fp_order    — float accumulation order changes result
    aliasing    — numpy arrays may alias; C++ `restrict` breaks them
    edge_empty  — empty input
    nan_inf     — special float values
    unicode     — string handling
    boundary    — INT_MAX, denormals
    semantics   — Python-specific behavior (None, slicing, generators)

Each trap has metadata:
    - id: stable identifier
    - category: one of the failure modes above
    - python_code: the source
    - bottleneck_label: ground-truth labels for DiagnosisRubric
    - rtol_override: None (default 1e-5) or 0 for bit-exact

15% of every batch comes from this library (per plan §4.3). 10 traps are held
out for the Gen-4 evaluation split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trap:
    id: str
    category: str
    python_code: str
    bottleneck_label: list[str] = field(default_factory=list)
    bottleneck_distractors: list[str] = field(default_factory=list)
    rtol_override: float | None = None
    held_out: bool = False
    description: str = ""


# Default distractor pool — used by all traps unless overridden
_DEFAULT_DISTRACTORS = ["memory-bound", "branch-heavy", "io-bound", "cache-unfriendly"]


TRAP_LIBRARY: list[Trap] = [
    # -------- Category 1: int overflow (4 traps) --------
    Trap(
        id="overflow_factorial",
        category="overflow",
        python_code=(
            "def factorial(n: int) -> int:\n"
            "    r = 1\n"
            "    for i in range(2, n + 1):\n"
            "        r *= i\n"
            "    return r\n"
        ),
        bottleneck_label=["compute-bound"],
        bottleneck_distractors=_DEFAULT_DISTRACTORS,
        rtol_override=0,  # bit-exact integer
        description="Python big-int math; C++ int overflows past 12!",
    ),
    Trap(
        id="overflow_power",
        category="overflow",
        python_code=(
            "def power_sum(base: int, exp: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(1, exp + 1):\n"
            "        total += base ** i\n"
            "    return total\n"
        ),
        bottleneck_label=["compute-bound"],
        rtol_override=0,
    ),
    Trap(
        id="overflow_signed_bitshift",
        category="overflow",
        python_code=(
            "def shift_accumulate(arr: list) -> int:\n"
            "    total = 0\n"
            "    for x in arr:\n"
            "        total += (x << 30)\n"
            "    return total\n"
        ),
        bottleneck_label=["compute-bound"],
        rtol_override=0,
    ),
    Trap(
        id="overflow_int64_sum",
        category="overflow",
        python_code=(
            "def big_sum(arr: list) -> int:\n"
            "    total = 0\n"
            "    for x in arr:\n"
            "        total += x * x * x\n"
            "    return total\n"
        ),
        bottleneck_label=["compute-bound", "vectorizable"],
        rtol_override=0,
    ),

    # -------- Category 2: floating point accumulation order (5 traps) --------
    Trap(
        id="fp_kahan_drift",
        category="fp_order",
        python_code=(
            "def kahan_sum(arr):\n"
            "    s = 0.0\n"
            "    c = 0.0\n"
            "    for x in arr:\n"
            "        y = x - c\n"
            "        t = s + y\n"
            "        c = (t - s) - y\n"
            "        s = t\n"
            "    return s\n"
        ),
        bottleneck_label=["compute-bound"],
        description="Kahan compensated summation — C++ reorder breaks compensation",
    ),
    Trap(
        id="fp_pairwise_var",
        category="fp_order",
        python_code=(
            "def variance(arr):\n"
            "    n = len(arr)\n"
            "    mean = sum(arr) / n\n"
            "    return sum((x - mean) ** 2 for x in arr) / n\n"
        ),
        bottleneck_label=["compute-bound"],
    ),
    Trap(
        id="fp_chained_mul",
        category="fp_order",
        python_code=(
            "def chain_mul(arr):\n"
            "    p = 1.0\n"
            "    for x in arr:\n"
            "        p *= x\n"
            "    return p\n"
        ),
        bottleneck_label=["compute-bound"],
    ),
    Trap(
        id="fp_subnormal_handling",
        category="fp_order",
        python_code=(
            "def near_zero_sum(arr):\n"
            "    return sum(x for x in arr if abs(x) > 1e-300)\n"
        ),
        bottleneck_label=["compute-bound", "branch-heavy"],
    ),
    Trap(
        id="fp_log_sum_exp",
        category="fp_order",
        python_code=(
            "import math\n"
            "def log_sum_exp(arr):\n"
            "    m = max(arr)\n"
            "    return m + math.log(sum(math.exp(x - m) for x in arr))\n"
        ),
        bottleneck_label=["compute-bound"],
    ),

    # -------- Category 3: aliasing (3 traps) --------
    Trap(
        id="aliasing_in_place",
        category="aliasing",
        python_code=(
            "def in_place_smooth(a):\n"
            "    n = len(a)\n"
            "    for i in range(1, n - 1):\n"
            "        a[i] = (a[i-1] + a[i] + a[i+1]) / 3.0\n"
            "    return a\n"
        ),
        bottleneck_label=["memory-bound"],
        bottleneck_distractors=["compute-bound", "branch-heavy", "io-bound"],
        description="Read-after-write across iterations; `restrict` would break correctness",
    ),
    Trap(
        id="aliasing_two_views",
        category="aliasing",
        python_code=(
            "def add_views(a, b):\n"
            "    n = len(a)\n"
            "    for i in range(n):\n"
            "        a[i] += b[i] * 2\n"
            "    return a\n"
        ),
        bottleneck_label=["memory-bound", "vectorizable"],
        description="`a` and `b` may overlap; agent must not blindly add `__restrict__`",
    ),
    Trap(
        id="aliasing_self_copy",
        category="aliasing",
        python_code=(
            "def shift_left(a):\n"
            "    n = len(a)\n"
            "    for i in range(n - 1):\n"
            "        a[i] = a[i + 1]\n"
            "    return a\n"
        ),
        bottleneck_label=["memory-bound"],
    ),

    # -------- Category 4: edge case empty / single (3 traps) --------
    Trap(
        id="edge_empty_max",
        category="edge_empty",
        python_code=(
            "def safe_max(arr):\n"
            "    if len(arr) == 0:\n"
            "        return 0.0\n"
            "    return max(arr)\n"
        ),
        bottleneck_label=["branch-heavy"],
    ),
    Trap(
        id="edge_singleton",
        category="edge_empty",
        python_code=(
            "def doubled_diff(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return 0.0\n"
            "    return sum(arr[i+1] - arr[i] for i in range(len(arr) - 1))\n"
        ),
        bottleneck_label=["compute-bound", "branch-heavy"],
    ),
    Trap(
        id="edge_zero_division",
        category="edge_empty",
        python_code=(
            "def normalize(arr):\n"
            "    s = sum(arr)\n"
            "    if s == 0:\n"
            "        return [0.0 for _ in arr]\n"
            "    return [x / s for x in arr]\n"
        ),
        bottleneck_label=["compute-bound", "branch-heavy"],
    ),

    # -------- Category 5: NaN/Inf (3 traps) --------
    Trap(
        id="nan_propagation",
        category="nan_inf",
        python_code=(
            "import math\n"
            "def filter_finite(arr):\n"
            "    return sum(x for x in arr if math.isfinite(x))\n"
        ),
        bottleneck_label=["branch-heavy"],
    ),
    Trap(
        id="inf_arithmetic",
        category="nan_inf",
        python_code=(
            "import math\n"
            "def soft_clamp(arr):\n"
            "    return [x if math.isfinite(x) else 0.0 for x in arr]\n"
        ),
        bottleneck_label=["branch-heavy"],
    ),
    Trap(
        id="nan_aware_min",
        category="nan_inf",
        python_code=(
            "import math\n"
            "def nan_aware_min(arr):\n"
            "    finite = [x for x in arr if not math.isnan(x)]\n"
            "    return min(finite) if finite else 0.0\n"
        ),
        bottleneck_label=["branch-heavy"],
    ),

    # -------- Category 6: boundary values (3 traps) --------
    Trap(
        id="boundary_signed_compare",
        category="boundary",
        python_code=(
            "def count_negatives(arr: list) -> int:\n"
            "    return sum(1 for x in arr if x < 0)\n"
        ),
        bottleneck_label=["branch-heavy", "vectorizable"],
        rtol_override=0,
    ),
    Trap(
        id="boundary_min_int",
        category="boundary",
        python_code=(
            "def abs_sum(arr: list) -> int:\n"
            "    return sum(abs(x) for x in arr)\n"
        ),
        bottleneck_label=["compute-bound"],
        rtol_override=0,
        description="abs(INT_MIN) overflows in C++; Python handles transparently",
    ),
    Trap(
        id="boundary_denormal_threshold",
        category="boundary",
        python_code=(
            "def threshold_count(arr):\n"
            "    return sum(1 for x in arr if abs(x) > 1e-308)\n"
        ),
        bottleneck_label=["branch-heavy"],
    ),

    # -------- Category 7: semantics (5 traps) --------
    Trap(
        id="semantics_negative_index",
        category="semantics",
        python_code=(
            "def last_diff(arr):\n"
            "    return arr[-1] - arr[0] if len(arr) >= 1 else 0\n"
        ),
        bottleneck_label=["compute-bound"],
        description="Python a[-1] = last element; C++ a[-1] = UB",
    ),
    Trap(
        id="semantics_empty_sum",
        category="semantics",
        python_code=(
            "def opt_avg(arr):\n"
            "    return sum(arr) / len(arr) if arr else 0.0\n"
        ),
        bottleneck_label=["compute-bound", "branch-heavy"],
    ),
    Trap(
        id="semantics_truthy_filter",
        category="semantics",
        python_code=(
            "def count_truthy(arr):\n"
            "    return sum(1 for x in arr if x)\n"
        ),
        bottleneck_label=["branch-heavy"],
        description="Python truthy includes [], 0, '', None; C++ has different semantics",
        rtol_override=0,
    ),
    Trap(
        id="semantics_int_div",
        category="semantics",
        python_code=(
            "def floor_avg(arr: list) -> int:\n"
            "    return sum(arr) // len(arr) if arr else 0\n"
        ),
        bottleneck_label=["compute-bound"],
        rtol_override=0,
        description="// is floor div in Python (correct for negatives); C++ / truncates toward zero",
    ),
    Trap(
        id="semantics_modulo_negative",
        category="semantics",
        python_code=(
            "def positive_mod_sum(arr: list, m: int) -> int:\n"
            "    return sum(x % m for x in arr)\n"
        ),
        bottleneck_label=["compute-bound"],
        rtol_override=0,
        description="Python % always returns non-negative for positive m; C++ may return negative",
    ),

    # -------- Category 8: held-out for Gen-4 (4 traps) --------
    Trap(
        id="holdout_kahan_sum_2",
        category="fp_order",
        python_code=(
            "def stable_total(arr):\n"
            "    s = 0.0\n"
            "    err = 0.0\n"
            "    for x in arr:\n"
            "        y = x + err\n"
            "        new_s = s + y\n"
            "        err = y - (new_s - s)\n"
            "        s = new_s\n"
            "    return s\n"
        ),
        bottleneck_label=["compute-bound"],
        held_out=True,
    ),
    Trap(
        id="holdout_overflow_combinations",
        category="overflow",
        python_code=(
            "def n_choose_k(n: int, k: int) -> int:\n"
            "    if k > n - k:\n"
            "        k = n - k\n"
            "    r = 1\n"
            "    for i in range(k):\n"
            "        r = r * (n - i) // (i + 1)\n"
            "    return r\n"
        ),
        bottleneck_label=["compute-bound"],
        rtol_override=0,
        held_out=True,
    ),
    Trap(
        id="holdout_aliasing_swap",
        category="aliasing",
        python_code=(
            "def reverse_in_place(a):\n"
            "    n = len(a)\n"
            "    for i in range(n // 2):\n"
            "        a[i], a[n - 1 - i] = a[n - 1 - i], a[i]\n"
            "    return a\n"
        ),
        bottleneck_label=["memory-bound"],
        held_out=True,
    ),
    Trap(
        id="holdout_semantics_chained_compare",
        category="semantics",
        python_code=(
            "def in_range_count(arr, lo: float, hi: float) -> int:\n"
            "    return sum(1 for x in arr if lo < x < hi)\n"
        ),
        bottleneck_label=["branch-heavy"],
        rtol_override=0,
        held_out=True,
        description="Python a < x < b is single test; agent may write incorrect (a < x) < b in C++",
    ),
]


def get_trap_by_id(trap_id: str) -> Trap | None:
    return next((t for t in TRAP_LIBRARY if t.id == trap_id), None)


def sample_trap(rng, exclude_held_out: bool = True) -> Trap:
    """Sample a random trap. By default excludes the Gen-4 held-out subset."""
    pool = [t for t in TRAP_LIBRARY if not (exclude_held_out and t.held_out)]
    return rng.choice(pool)


def trap_to_problem_dict(trap: Trap, hw_profile: dict[str, Any]) -> dict[str, Any]:
    """Convert a Trap into the env._sample_problem() return shape."""
    # Default distractor pool excluding the trap's true labels
    distractors = [d for d in (trap.bottleneck_distractors or _DEFAULT_DISTRACTORS)
                   if d not in trap.bottleneck_label]
    return {
        "python_code": trap.python_code,
        "cpp_signature": _infer_cpp_signature(trap.python_code),
        "hardware_profile": hw_profile,
        "bottleneck_labels": trap.bottleneck_label,
        "bottleneck_distractors": distractors,
        "rtol_override": trap.rtol_override,
        "is_trap": True,
        "trap_id": trap.id,
    }


def _infer_cpp_signature(python_code: str) -> str:
    """Best-effort C++ signature derivation from a Python def. Refined in Hour 22 smoke test."""
    import ast
    try:
        tree = ast.parse(python_code)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        return f'extern "C" void agent_function(/* {len(fn.args.args)} args from Python */ );'
    except Exception:
        return 'extern "C" void agent_function(void* in, size_t n, void* out);'


# Public counts for assertions
N_TRAPS_TOTAL = len(TRAP_LIBRARY)
N_TRAPS_TRAINING = sum(1 for t in TRAP_LIBRARY if not t.held_out)
N_TRAPS_HELDOUT = sum(1 for t in TRAP_LIBRARY if t.held_out)


__all__ = [
    "Trap",
    "TRAP_LIBRARY",
    "get_trap_by_id",
    "sample_trap",
    "trap_to_problem_dict",
    "N_TRAPS_TOTAL",
    "N_TRAPS_TRAINING",
    "N_TRAPS_HELDOUT",
]
