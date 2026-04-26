"""DiagnosisRubric — multi-signal anti-gaming hypothesis scoring (per plan §10b).

Pure keyword match is gameable (agent stuffs all bottleneck keywords into <think>).
Defense-in-depth:

    raw = (correct_kw / |ground_truth|) - 0.5 * (distractor_kw / |distractors|)
    raw = max(0, raw)
    length_penalty = 1 - 0.1 * (len(thinking) / 256)        # concise > verbose
    coherence_bonus = 0.2 if first_tool_call matches diagnosis  else 0
    score = raw * length_penalty + coherence_bonus
"""

from __future__ import annotations

from typing import Any

from .rubrics import Rubric


# Map each diagnosis category to the tool that's "coherent" with it
DIAGNOSIS_TO_FIRST_TOOL = {
    "memory-bound": "check_memory_access",
    "compute-bound": "get_hardware_profile",        # check SIMD width before vectorizing
    "vectorizable": "get_hardware_profile",
    "branch-heavy": "profile_python_hotspots",
    "io-bound": "profile_python_hotspots",          # confirm where time goes
    "cache-unfriendly": "check_memory_access",
}


class DiagnosisRubric(Rubric):
    name = "diagnosis"

    def __init__(self, max_thinking_len: int = 256, length_penalty_rate: float = 0.1,
                 distractor_penalty_weight: float = 0.5, coherence_bonus: float = 0.2,
                 min_correctness_for_credit: float = 0.20, target_alignment_bonus: float = 0.10):
        self.max_thinking_len = max_thinking_len
        self.length_penalty_rate = length_penalty_rate
        self.distractor_penalty_weight = distractor_penalty_weight
        self.coherence_bonus = coherence_bonus
        self.min_correctness_for_credit = min_correctness_for_credit
        self.target_alignment_bonus = target_alignment_bonus

    def score(self, state, submission: dict[str, Any]) -> float:
        compile_ok = submission.get("compile_status") == "success"
        correctness = float(submission.get("correctness_pass_rate", 0.0))
        # Prevent diagnosis-only reward hacking: no diagnosis credit before basic validity.
        if (not compile_ok) or correctness < self.min_correctness_for_credit:
            self.last_breakdown = {
                "score": 0.0,
                "gated": True,
                "compile_ok": compile_ok,
                "correctness_pass_rate": correctness,
                "min_correctness_for_credit": self.min_correctness_for_credit,
            }
            return 0.0

        thinking = (submission.get("reasoning_trace", "") or state.current_round_reasoning or "").lower()
        ground_truth = state.bottleneck_ground_truth or []
        distractors = state.bottleneck_distractors or []

        # Keyword counts (use word-boundary-ish substring match)
        correct_kw = sum(1 for kw in ground_truth if kw.lower() in thinking)
        distractor_kw = sum(1 for kw in distractors if kw.lower() in thinking)

        if not ground_truth:
            self.last_breakdown = {"score": 0.0, "reason": "no_ground_truth_labels"}
            return 0.0

        raw = (correct_kw / len(ground_truth))
        if distractors:
            raw -= self.distractor_penalty_weight * (distractor_kw / len(distractors))
        raw = max(0.0, raw)

        length = len(thinking.encode("utf-8"))  # bytes — closer to token cost
        length_penalty = max(0.0, 1.0 - self.length_penalty_rate * (length / self.max_thinking_len))

        # Coherence bonus: was the FIRST tool call in this round consistent with the diagnosis?
        # During reward computation, current round calls are in state.current_round_tool_calls.
        # Fall back to round_results only when current calls are unavailable.
        first_tool = ""
        calls = list(state.current_round_tool_calls or [])
        if not calls:
            round_idx = state.round_number - 1
            if 0 <= round_idx < len(state.round_results):
                calls = list(state.round_results[round_idx].get("tool_calls", []))
        if calls:
            first_tool = calls[0]
            if first_tool == "get_hardware_profile" and len(calls) > 1:
                first_tool = calls[1]

        # Match: any ground_truth label whose preferred tool == first_tool counts as coherent
        coherence = 0.0
        for label in ground_truth:
            preferred = DIAGNOSIS_TO_FIRST_TOOL.get(label.lower())
            if preferred and preferred == first_tool:
                coherence = self.coherence_bonus
                break

        # Reward explicit target-aware reasoning (e.g., x86_AVX512 vs ARM_NEON).
        target = str(state.hardware_profile.get("target", "")).lower()
        target_tokens = {
            "x86_avx512": ("avx512", "avx-512"),
            "x86_avx2": ("avx2",),
            "x86_sse": ("sse", "sse4"),
            "arm_neon": ("neon", "arm"),
            "scalar_only": ("scalar",),
        }
        target_bonus = 0.0
        for token in target_tokens.get(target, ()):
            if token in thinking:
                target_bonus = self.target_alignment_bonus
                break

        score = raw * length_penalty + coherence + target_bonus
        score = max(0.0, min(1.0, score))

        self.last_breakdown = {
            "correct_kw": correct_kw,
            "distractor_kw": distractor_kw,
            "raw": raw,
            "thinking_bytes": length,
            "length_penalty": length_penalty,
            "first_tool": first_tool,
            "coherence_bonus": coherence,
            "target": target,
            "target_alignment_bonus": target_bonus,
            "score": score,
        }
        return score


__all__ = ["DiagnosisRubric", "DIAGNOSIS_TO_FIRST_TOOL"]
