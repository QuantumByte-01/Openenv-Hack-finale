"""Template-based + adversarial Python function generator.

Per plan §16 hard cutoff: ship template-only first, add LLM-based adversarial
generation only if Hour 22 budget allows. This module currently implements the
deterministic template generator. The LLM-adversarial path is wired through a
`generate_adversarial(...)` stub that we can switch to in Hour 22 if time permits.

Templates are tier-parameterized (per plan §9 four tiers):
    Tier 0: Algorithmic — simple loops, sum/argmax/count/prefix
    Tier 1: Memory-aware — transpose, sliding window, histogram
    Tier 2: SIMD+parallel — pairwise distance, batch_norm, RLE
    Tier 3: Frontier — fused attention, sparse, conv2d
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Template:
    id: str
    tier: int
    python_code: str
    bottleneck_label: list[str] = field(default_factory=list)
    description: str = ""


# -------- Tier 0: Algorithmic --------

_TIER_0_TEMPLATES: list[Template] = [
    Template(
        id="t0_simple_sum",
        tier=0,
        python_code=(
            "def total(arr):\n"
            "    s = 0.0\n"
            "    for x in arr:\n"
            "        s += x\n"
            "    return s\n"
        ),
        bottleneck_label=["compute-bound", "vectorizable"],
    ),
    Template(
        id="t0_argmax",
        tier=0,
        python_code=(
            "def argmax(arr):\n"
            "    if not arr:\n"
            "        return -1\n"
            "    best_i, best_v = 0, arr[0]\n"
            "    for i in range(1, len(arr)):\n"
            "        if arr[i] > best_v:\n"
            "            best_v, best_i = arr[i], i\n"
            "    return best_i\n"
        ),
        bottleneck_label=["branch-heavy", "compute-bound"],
    ),
    Template(
        id="t0_count_if",
        tier=0,
        python_code=(
            "def count_pos(arr):\n"
            "    n = 0\n"
            "    for x in arr:\n"
            "        if x > 0:\n"
            "            n += 1\n"
            "    return n\n"
        ),
        bottleneck_label=["branch-heavy", "vectorizable"],
    ),
    Template(
        id="t0_prefix_sum",
        tier=0,
        python_code=(
            "def prefix_sum(arr):\n"
            "    out = [0.0] * len(arr)\n"
            "    s = 0.0\n"
            "    for i, x in enumerate(arr):\n"
            "        s += x\n"
            "        out[i] = s\n"
            "    return out\n"
        ),
        bottleneck_label=["compute-bound"],
    ),
    Template(
        id="t0_sum_squares",
        tier=0,
        python_code=(
            "def sum_squares(arr):\n"
            "    s = 0.0\n"
            "    for x in arr:\n"
            "        s += x * x\n"
            "    return s\n"
        ),
        bottleneck_label=["compute-bound", "vectorizable"],
    ),
]


# -------- Tier 1: Memory-aware --------

_TIER_1_TEMPLATES: list[Template] = [
    Template(
        id="t1_matrix_transpose",
        tier=1,
        python_code=(
            "def transpose(a, n: int, m: int):\n"
            "    out = [[0.0]*n for _ in range(m)]\n"
            "    for i in range(n):\n"
            "        for j in range(m):\n"
            "            out[j][i] = a[i][j]\n"
            "    return out\n"
        ),
        bottleneck_label=["memory-bound", "cache-unfriendly"],
    ),
    Template(
        id="t1_sliding_window",
        tier=1,
        python_code=(
            "def moving_avg(arr, k: int):\n"
            "    n = len(arr)\n"
            "    out = [0.0] * (n - k + 1)\n"
            "    for i in range(n - k + 1):\n"
            "        s = 0.0\n"
            "        for j in range(k):\n"
            "            s += arr[i + j]\n"
            "        out[i] = s / k\n"
            "    return out\n"
        ),
        bottleneck_label=["compute-bound", "memory-bound"],
    ),
    Template(
        id="t1_histogram",
        tier=1,
        python_code=(
            "def histogram(arr, n_bins: int):\n"
            "    bins = [0] * n_bins\n"
            "    lo = min(arr)\n"
            "    hi = max(arr)\n"
            "    width = (hi - lo) / n_bins if hi > lo else 1.0\n"
            "    for x in arr:\n"
            "        b = min(int((x - lo) / width), n_bins - 1)\n"
            "        bins[b] += 1\n"
            "    return bins\n"
        ),
        bottleneck_label=["memory-bound", "branch-heavy"],
    ),
    Template(
        id="t1_bitmask_filter",
        tier=1,
        python_code=(
            "def masked_sum(arr, mask):\n"
            "    return sum(arr[i] for i in range(len(arr)) if mask[i])\n"
        ),
        bottleneck_label=["branch-heavy", "vectorizable"],
    ),
]


# -------- Tier 2: SIMD + parallel --------

_TIER_2_TEMPLATES: list[Template] = [
    Template(
        id="t2_pairwise_dist",
        tier=2,
        python_code=(
            "def pairwise_dist_sq(X, n: int, d: int):\n"
            "    out = [[0.0]*n for _ in range(n)]\n"
            "    for i in range(n):\n"
            "        for j in range(n):\n"
            "            s = 0.0\n"
            "            for k in range(d):\n"
            "                diff = X[i][k] - X[j][k]\n"
            "                s += diff * diff\n"
            "            out[i][j] = s\n"
            "    return out\n"
        ),
        bottleneck_label=["compute-bound", "vectorizable"],
    ),
    Template(
        id="t2_batch_norm",
        tier=2,
        python_code=(
            "def batch_norm(X, gamma, beta, eps: float):\n"
            "    n = len(X)\n"
            "    mean = sum(X) / n\n"
            "    var = sum((x - mean) ** 2 for x in X) / n\n"
            "    inv_std = 1.0 / ((var + eps) ** 0.5)\n"
            "    return [gamma * (x - mean) * inv_std + beta for x in X]\n"
        ),
        bottleneck_label=["compute-bound", "vectorizable"],
    ),
    Template(
        id="t2_inner_product_batch",
        tier=2,
        python_code=(
            "def batch_inner(A, B, n: int, d: int):\n"
            "    out = [0.0] * n\n"
            "    for i in range(n):\n"
            "        s = 0.0\n"
            "        for k in range(d):\n"
            "            s += A[i][k] * B[i][k]\n"
            "        out[i] = s\n"
            "    return out\n"
        ),
        bottleneck_label=["compute-bound", "vectorizable"],
    ),
]


# -------- Tier 3: Frontier --------

_TIER_3_TEMPLATES: list[Template] = [
    Template(
        id="t3_attention_score",
        tier=3,
        python_code=(
            "def attention_score(Q, K, n: int, d: int):\n"
            "    out = [[0.0]*n for _ in range(n)]\n"
            "    for i in range(n):\n"
            "        for j in range(n):\n"
            "            s = 0.0\n"
            "            for k in range(d):\n"
            "                s += Q[i][k] * K[j][k]\n"
            "            out[i][j] = s / (d ** 0.5)\n"
            "    return out\n"
        ),
        bottleneck_label=["compute-bound", "vectorizable"],
    ),
    Template(
        id="t3_softmax_log",
        tier=3,
        python_code=(
            "import math\n"
            "def log_softmax(arr):\n"
            "    m = max(arr)\n"
            "    s = sum(math.exp(x - m) for x in arr)\n"
            "    log_s = m + math.log(s)\n"
            "    return [x - log_s for x in arr]\n"
        ),
        bottleneck_label=["compute-bound"],
    ),
    Template(
        id="t3_conv2d_naive",
        tier=3,
        python_code=(
            "def conv2d(img, kernel, h: int, w: int, kh: int, kw: int):\n"
            "    oh, ow = h - kh + 1, w - kw + 1\n"
            "    out = [[0.0]*ow for _ in range(oh)]\n"
            "    for i in range(oh):\n"
            "        for j in range(ow):\n"
            "            s = 0.0\n"
            "            for ki in range(kh):\n"
            "                for kj in range(kw):\n"
            "                    s += img[i+ki][j+kj] * kernel[ki][kj]\n"
            "            out[i][j] = s\n"
            "    return out\n"
        ),
        bottleneck_label=["compute-bound", "memory-bound"],
    ),
]


_TEMPLATES_BY_TIER = {
    0: _TIER_0_TEMPLATES,
    1: _TIER_1_TEMPLATES,
    2: _TIER_2_TEMPLATES,
    3: _TIER_3_TEMPLATES,
}


_DEFAULT_DISTRACTORS = ["memory-bound", "branch-heavy", "io-bound", "cache-unfriendly", "compute-bound"]


class TemplateGenerator:
    """Deterministic template generator (no LLM call). Hour 16-22 deliverable."""

    def sample(self, tier: int, rng: random.Random) -> Template:
        """Sample a template at the given tier (or below — gives easier mix in early training)."""
        pool: list[Template] = []
        for t in range(min(tier, 3) + 1):
            pool.extend(_TEMPLATES_BY_TIER[t])
        if not pool:
            pool = _TEMPLATES_BY_TIER[0]
        return rng.choice(pool)


def generate_from_template(template: Template, hw_profile: dict[str, Any]) -> dict[str, Any]:
    """Convert a Template into the env._sample_problem() return shape."""
    distractors = [d for d in _DEFAULT_DISTRACTORS if d not in template.bottleneck_label]
    from .trap_library import _infer_cpp_signature
    return {
        "python_code": template.python_code,
        "cpp_signature": _infer_cpp_signature(template.python_code),
        "hardware_profile": hw_profile,
        "bottleneck_labels": template.bottleneck_label,
        "bottleneck_distractors": distractors,
        "rtol_override": None,
        "is_trap": False,
        "template_id": template.id,
        "tier": template.tier,
    }


# Public counts
N_TEMPLATES_TIER_0 = len(_TIER_0_TEMPLATES)
N_TEMPLATES_TIER_1 = len(_TIER_1_TEMPLATES)
N_TEMPLATES_TIER_2 = len(_TIER_2_TEMPLATES)
N_TEMPLATES_TIER_3 = len(_TIER_3_TEMPLATES)


__all__ = [
    "Template",
    "TemplateGenerator",
    "generate_from_template",
    "N_TEMPLATES_TIER_0", "N_TEMPLATES_TIER_1", "N_TEMPLATES_TIER_2", "N_TEMPLATES_TIER_3",
]
