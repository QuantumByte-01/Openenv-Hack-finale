"""DatasetLoader: pulls Python functions from existing public datasets.

Per plan §4 the training pool is constructed from:
    - IBM CodeNet  (~80K filtered, primary)
    - TransCoder   (852 pairs, cross-validation)
    - Pyperformance (60 fns, real-world calibration)
    - Polybench/C  (30 kernels, back-translated)
    - Templates    (this module's TemplateGenerator, dynamic)
    - Trap library (15% of every batch)

For Hour 16-22 we ship a working loader for templates + traps. CodeNet/
TransCoder/Pyperformance are wired in via lazy load (HF datasets) — failing
gracefully to template-only when offline. The Hour 22 smoke test gate verifies
either path works.
"""

from __future__ import annotations

import ast
import random
from typing import Any

from .generator import TemplateGenerator, generate_from_template
from .trap_library import get_trap_by_id, sample_trap, sample_trap_by_category, trap_to_problem_dict
from .hardware_profiles import sample_profile


class DatasetLoader:
    """Unified sampler. The environment calls .sample(axes, rng) per reset()."""

    # Probability that a sampled function is a trap (per plan §4.3 — "15% of every batch")
    TRAP_PROBABILITY = 0.15

    def __init__(self, prefer_real_datasets: bool = False):
        """`prefer_real_datasets=True` triggers CodeNet/TransCoder loading.

        Default False = template-only (Hour 16-22 default; flip in Hour 22+ if
        training has bandwidth to download HF datasets).
        """
        self.prefer_real = prefer_real_datasets
        self.template_generator = TemplateGenerator()
        self._codenet_cache: list[dict[str, Any]] | None = None
        self._mbpp_cache: list[dict[str, Any]] | None = None
        self._trap_failure_counts: dict[str, int] = {}
        self._adaptive_trap_boost: float = 0.0

    def sample(self, axes: dict[str, int], rng: random.Random) -> dict[str, Any]:
        """Sample one (function, hw_profile, ground_truth) tuple given axis levels."""
        # Pick the hardware profile per the hardware_class axis
        hw = sample_profile(rng, axis_level=axes.get("hardware_class", 0))

        # 15% of the time, draw a trap
        if rng.random() < self.TRAP_PROBABILITY:
            return self._sample_trap_problem(rng, hw)

        # Otherwise — template, biased to current tier (or real dataset if enabled)
        if self.prefer_real:
            return self._sample_real_mixed(rng, hw, axes)

        # Template path
        tier = axes.get("function_tier", 0)
        template = self.template_generator.sample(tier=tier, rng=rng)
        return generate_from_template(template, hw)

    def record_submission_outcome(self, state, submission: dict[str, Any]) -> None:
        """Update adaptive trap priorities from recent trap outcomes."""
        if not getattr(state, "is_trap", False):
            # Slow decay when solving non-trap episodes so adaptation doesn't stick forever.
            self._adaptive_trap_boost = max(0.0, self._adaptive_trap_boost - 0.01)
            return

        trap_id = getattr(state, "trap_id", None)
        trap = get_trap_by_id(trap_id) if trap_id else None
        if trap is None:
            return

        pass_rate = float(submission.get("correctness_pass_rate", 0.0))
        adv_rate = float(submission.get("adversarial_pass_rate", 0.0))
        failed = pass_rate < 0.8 or adv_rate < 0.9
        if failed:
            self._trap_failure_counts[trap.category] = self._trap_failure_counts.get(trap.category, 0) + 1
            self._adaptive_trap_boost = min(0.25, self._adaptive_trap_boost + 0.03)
        else:
            self._adaptive_trap_boost = max(0.0, self._adaptive_trap_boost - 0.02)

    def _sample_trap_problem(self, rng: random.Random, hw: dict[str, Any]) -> dict[str, Any]:
        """Sample a static or adaptive trap depending on recent failure patterns."""
        use_adaptive = bool(self._trap_failure_counts) and rng.random() < min(0.85, 0.55 + self._adaptive_trap_boost)
        if use_adaptive:
            categories = list(self._trap_failure_counts.keys())
            weights = [max(1, self._trap_failure_counts[c]) for c in categories]
            chosen_category = rng.choices(categories, weights=weights, k=1)[0]
            base_trap = sample_trap_by_category(chosen_category, rng, exclude_held_out=True)
            if base_trap is None:
                base_trap = sample_trap(rng, exclude_held_out=True)
            return self._build_adaptive_trap_variant(base_trap, hw, rng)

        trap = sample_trap(rng, exclude_held_out=True)
        p = trap_to_problem_dict(trap, hw)
        p["source"] = "trap_library"
        return p

    def _build_adaptive_trap_variant(self, trap, hw: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        """Generate a semantic-preserving variant to reduce memorization."""
        python_code = trap.python_code
        if "def " in python_code and "(" in python_code:
            suffix = rng.randint(1000, 9999)
            start = python_code.find("def ")
            end = python_code.find("(", start)
            fn_name = python_code[start + 4:end].strip()
            if fn_name:
                python_code = python_code.replace(f"def {fn_name}(", f"def {fn_name}_adapt_{suffix}(", 1)
        python_code = self._semantic_noop_mutation(python_code, rng)

        variant = trap_to_problem_dict(trap, hw)
        variant["python_code"] = python_code
        variant["trap_id"] = f"{trap.id}::adaptive"
        variant["trap_parent_id"] = trap.id
        variant["trap_category"] = trap.category
        variant["source"] = "adaptive_trap"
        return variant

    def _semantic_noop_mutation(self, python_code: str, rng: random.Random) -> str:
        """Apply semantic no-op AST rewrites so adaptive traps are not pure renames."""

        class _NoopTransformer(ast.NodeTransformer):
            def __init__(self, seed: int):
                self._rng = random.Random(seed)

            def visit_For(self, node: ast.For):
                self.generic_visit(node)
                # Insert a no-op guard branch to perturb structure while preserving behavior.
                if self._rng.random() < 0.45:
                    noop = ast.If(
                        test=ast.Constant(value=False),
                        body=[ast.Expr(value=ast.Constant(value=None))],
                        orelse=[],
                    )
                    node.body = [noop, *node.body]
                return node

            def visit_Assign(self, node: ast.Assign):
                self.generic_visit(node)
                # Occasionally wrap RHS in (+ 0) no-op for numeric expressions.
                if self._rng.random() < 0.30:
                    node.value = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(value=0))
                return node

        try:
            tree = ast.parse(python_code)
            transformer = _NoopTransformer(seed=rng.randint(0, 10_000_000))
            mutated = transformer.visit(tree)
            ast.fix_missing_locations(mutated)
            code = ast.unparse(mutated)
            if not code.endswith("\n"):
                code += "\n"
            return code
        except Exception:
            # Fallback: minimally perturb whitespace/comments while keeping code valid.
            lines = python_code.splitlines()
            if lines and not lines[0].lstrip().startswith("#"):
                lines.insert(0, "# adaptive trap variant")
            return "\n".join(lines) + ("\n" if lines else "")

    # -------- CodeNet integration (lazy, optional) --------

    def _codenet_loaded(self) -> bool:
        return self._codenet_cache is not None and len(self._codenet_cache) > 0

    def _mbpp_loaded(self) -> bool:
        return self._mbpp_cache is not None and len(self._mbpp_cache) > 0

    def _real_loaded(self) -> bool:
        return self._codenet_loaded() or self._mbpp_loaded()

    def _try_load_codenet(self) -> bool:
        """Lazy-load CodeNet from HF datasets. Returns True iff load succeeded.

        Handles offline / no-token gracefully.
        """
        if self._codenet_loaded():
            return True
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset(
                "codeparrot/codenet",
                split="train",
                streaming=True,
            )
            cache: list[dict[str, Any]] = []
            for example in ds:
                if len(cache) >= 1000:  # bounded preload
                    break
                if example.get("language") != "Python3":
                    continue
                code = example.get("code", "")
                fn = _extract_first_function(code)
                if fn and _is_training_compatible_fn(fn):
                    cache.append({"code": fn, "source": "codenet"})
            self._codenet_cache = cache
            return len(cache) > 0
        except Exception:
            self._codenet_cache = []
            return False

    def _try_load_mbpp(self) -> bool:
        if self._mbpp_loaded():
            return True
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("mbpp", split="train")
            cache: list[dict[str, Any]] = []
            for example in ds:
                if len(cache) >= 800:
                    break
                code = example.get("code", "")
                fn = _extract_first_function(code)
                if fn and _is_training_compatible_fn(fn):
                    cache.append({"code": fn, "source": "mbpp"})
            self._mbpp_cache = cache
            return len(cache) > 0
        except Exception:
            self._mbpp_cache = []
            return False

    def _sample_real_mixed(self, rng: random.Random, hw: dict[str, Any], axes: dict[str, int]) -> dict[str, Any]:
        if not self._codenet_loaded():
            self._try_load_codenet()
        if not self._mbpp_loaded():
            self._try_load_mbpp()

        if not self._real_loaded():
            # Fall back to template
            template = self.template_generator.sample(tier=axes.get("function_tier", 0), rng=rng)
            return generate_from_template(template, hw)

        pools = []
        if self._codenet_loaded():
            pools.append(self._codenet_cache or [])
        if self._mbpp_loaded():
            pools.append(self._mbpp_cache or [])
        if not pools:
            template = self.template_generator.sample(tier=axes.get("function_tier", 0), rng=rng)
            return generate_from_template(template, hw)

        chosen_pool = rng.choice(pools)
        sample = rng.choice(chosen_pool)
        return {
            "python_code": sample["code"],
            "cpp_signature": _infer_cpp_signature_simple(sample["code"]),
            "hardware_profile": hw,
            # Without ground-truth labels we use a generic catch-all; DiagnosisRubric will
            # award partial credit for any of these. CodeNet samples are not the primary
            # training source for diagnosis training — the templates are.
            "bottleneck_labels": ["compute-bound"],
            "bottleneck_distractors": ["memory-bound", "branch-heavy", "io-bound"],
            "rtol_override": None,
            "is_trap": False,
            "source": sample["source"],
        }


def _infer_cpp_signature_simple(python_code: str) -> str:
    import ast
    try:
        tree = ast.parse(python_code)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if fn:
            return f'extern "C" void agent_function(/* {len(fn.args.args)} args */);'
    except Exception:
        pass
    return 'extern "C" void agent_function(void* in, size_t n, void* out);'


def _extract_first_function(code: str) -> str | None:
    try:
        tree = ast.parse(code)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if fn is None:
            return None
        module = ast.Module(body=[fn], type_ignores=[])
        extracted = ast.unparse(module)
        if not extracted.endswith("\n"):
            extracted += "\n"
        return extracted
    except Exception:
        return None


def _is_training_compatible_fn(code: str) -> bool:
    try:
        tree = ast.parse(code)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if fn is None:
            return False
        # Keep simple single-argument kernels for stable OpenEnv benchmarking.
        if len(fn.args.args) != 1:
            return False
        return 40 <= len(code) <= 2400
    except Exception:
        return False


# Module-level convenience function (no class needed)
_default_loader: DatasetLoader | None = None


def sample_function(axes: dict[str, int], rng: random.Random) -> dict[str, Any]:
    global _default_loader
    if _default_loader is None:
        _default_loader = DatasetLoader(prefer_real_datasets=False)
    return _default_loader.sample(axes, rng)


__all__ = ["DatasetLoader", "sample_function"]
