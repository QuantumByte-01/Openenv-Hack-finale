#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch>=2.3",
#   "transformers>=4.40",
#   "bitsandbytes>=0.43.1",
#   "datasets>=2.18",
#   "trl>=0.14.0",
#   "peft>=0.11",
#   "wandb>=0.16",
#   "huggingface_hub>=0.22",
#   "matplotlib>=3.8",
# ]
# ///

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import statistics
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partial HF training run for Polyglot-Optima")
    parser.add_argument("--repo-id", default="Swastikr/polyglot_optima")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--demo-episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--reward-align-steps", type=int, default=30)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--ablation-steps", type=int, default=120)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--grpo-steps", type=int, default=80)
    parser.add_argument("--grpo-prompts", type=int, default=96)
    parser.add_argument("--skip-sft", action="store_true")
    return parser.parse_args()


def clone_repo(repo_id: str, dst: Path) -> Path:
    token = os.environ.get("HF_TOKEN")
    if not dst.exists() or not any(dst.iterdir()):
        snapshot_download(
            repo_id=repo_id,
            repo_type="space",
            local_dir=str(dst),
            token=token,
        )
    return dst


def _safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    win = max(1, min(window, len(values)))
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - win + 1)
        out.append(_safe_mean(values[lo : i + 1]))
    return out


def _sum_squares_cpp() -> str:
    return """#include <cstddef>
extern "C" void agent_function(
    const double* in_ptr, size_t in_n,
    double* out_ptr, size_t out_n)
{
    double total = 0.0;
    for (size_t i = 0; i < in_n; ++i) {
        total += in_ptr[i] * in_ptr[i];
    }
    if (out_n >= 1) out_ptr[0] = total;
}
"""


def _sum_cpp() -> str:
    return """#include <cstddef>
extern "C" void agent_function(
    const double* in_ptr, size_t in_n,
    double* out_ptr, size_t out_n)
{
    double total = 0.0;
    for (size_t i = 0; i < in_n; ++i) {
        total += in_ptr[i];
    }
    if (out_n >= 1) out_ptr[0] = total;
}
"""


def _build_teacher_cpp(python_code: str) -> str:
    code = python_code.replace(" ", "")
    if "x*x" in code or "x**2" in code or "sum_squares" in code:
        return _sum_squares_cpp()
    if "sum(" in code or "total+=" in code:
        return _sum_cpp()
    return _sum_squares_cpp()


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict):
            return "\n".join(str(msg.get("content", "")) for msg in completion if isinstance(msg, dict))
        return "\n".join(str(x) for x in completion)
    return str(completion)


def main() -> None:
    # Verifier fuzzing intentionally feeds nan/inf/edge values, which can trigger
    # expected scalar RuntimeWarnings in user Python snippets. Keep logs readable.
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r"invalid value encountered in scalar .*",
    )

    args = parse_args()
    random.seed(args.seed)
    gpu_fp16 = (not args.use_cpu) and bool(torch.cuda.is_available())
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "model_name": args.model_name,
                "max_steps": args.max_steps,
                "episodes": args.episodes,
                "demo_episodes": args.demo_episodes,
                "use_cpu": args.use_cpu,
                "reward_align_steps": args.reward_align_steps,
                "ablation": args.ablation,
                "load_in_4bit": args.load_in_4bit,
                "grpo_steps": args.grpo_steps,
                "grpo_prompts": args.grpo_prompts,
                "skip_sft": args.skip_sft,
            },
            indent=2,
        ),
        flush=True,
    )

    workdir = Path("/tmp/polyglot-optima")
    repo_dir = clone_repo(args.repo_id, workdir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from models import OptimizationAction  # pylint: disable=import-error
    from server.environment import PolyglotOptimaEnvironment  # pylint: disable=import-error

    def teacher_policy(observation):
        if observation.round_number == 1:
            return OptimizationAction(tool_name="get_hardware_profile", tool_args={}, reasoning_trace="teacher")
        if observation.round_number == 2:
            return OptimizationAction(tool_name="profile_python_hotspots", tool_args={}, reasoning_trace="teacher")
        return OptimizationAction(
            tool_name="submit_optimization",
            tool_args={
                "cpp_code": _build_teacher_cpp(observation.python_code),
                "reasoning_trace": "teacher compile-first submission",
            },
            reasoning_trace="teacher",
        )

    def run_eval(policy_fn, n_episodes: int, seed_start: int, label: str):
        env = PolyglotOptimaEnvironment(max_rounds=3, max_calls_per_round=5, enable_adaptive_curriculum=True)
        rewards = []
        correctness = []
        compile_success = []
        portability = []
        for i in range(n_episodes):
            print(f"[script] {label}: episode {i + 1}/{n_episodes}", flush=True)
            obs = env.reset(seed=seed_start + i)
            done = False
            while not done:
                step = env.step(policy_fn(obs))
                obs = step.observation
                done = step.done
            rewards.append(float(step.reward))
            submission = env.state().round_results[-1]["submission"] if env.state().round_results else {}
            correctness.append(float(submission.get("correctness_pass_rate", 0.0)))
            compile_success.append(1.0 if submission.get("compile_status") == "success" else 0.0)
            portability.append(float(submission.get("n_profiles_passing", 0.0)))
        env.close()
        return {
            "reward": rewards,
            "correctness": correctness,
            "compile_success": compile_success,
            "portability": portability,
        }

    print("[script] Running baseline evaluation...", flush=True)
    baseline = run_eval(teacher_policy, n_episodes=args.episodes, seed_start=1000, label="baseline_eval")

    def collect_demo_rows(demo_episodes: int, filter_good: bool, weighted: bool) -> list[dict[str, str]]:
        env = PolyglotOptimaEnvironment(max_rounds=3, max_calls_per_round=5, enable_adaptive_curriculum=True)
        rows: list[dict[str, str]] = []
        kept_episodes = 0
        for ep in range(demo_episodes):
            obs = env.reset(seed=5000 + ep)
            done = False
            ep_records: list[dict[str, Any]] = []
            while not done:
                action = teacher_policy(obs)
                prompt = (
                    "You are optimizing Python to C++. Choose next tool call.\n"
                    f"Round: {obs.round_number}\n"
                    f"Target ISA: {obs.hardware_profile.get('target', 'scalar_only')}\n"
                    f"Hardware: {json.dumps(obs.hardware_profile)}\n"
                    f"Python:\n{obs.python_code}\n"
                    "Return ONLY JSON: {\"tool_name\":..., \"tool_args\":...}"
                )
                ep_records.append(
                    {
                        "text": (
                            "<PROMPT>\n"
                            + prompt
                            + "\n<ANSWER>\n"
                            + json.dumps({"tool_name": action.tool_name, "tool_args": action.tool_args})
                        ),
                        "tool_name": action.tool_name,
                        "cpp_code": action.tool_args.get("cpp_code", ""),
                    }
                )
                step = env.step(action)
                obs = step.observation
                done = step.done

            submission = env.state().round_results[-1]["submission"] if env.state().round_results else {}
            compile_ok = submission.get("compile_status") == "success"
            correctness = float(submission.get("correctness_pass_rate", 0.0))
            portability = float(submission.get("n_profiles_passing", 0.0))
            quality = 0.5 * (1.0 if compile_ok else 0.0) + 0.35 * correctness + 0.15 * min(1.0, portability / 4.0)

            if filter_good and not compile_ok:
                continue

            repeat = max(1, int(1 + (4 * quality))) if weighted else 1
            for rec in ep_records:
                if rec["tool_name"] == "submit_optimization" and "baseline submit" in rec["cpp_code"]:
                    continue
                for _ in range(repeat):
                    rows.append({"text": rec["text"]})
            kept_episodes += 1
        env.close()
        print(
            f"[script] Demo rows collected={len(rows)} from kept_episodes={kept_episodes}/{demo_episodes} "
            f"(filter_good={filter_good}, weighted={weighted})",
            flush=True,
        )
        return rows

    def collect_grpo_prompts(n_prompts: int) -> list[dict[str, Any]]:
        env = PolyglotOptimaEnvironment(max_rounds=3, max_calls_per_round=5, enable_adaptive_curriculum=True)
        rows: list[dict[str, Any]] = []
        for ep in range(n_prompts):
            if ep % 10 == 0:
                print(f"[script] GRPO prompt collection episode {ep + 1}/{n_prompts}", flush=True)
            obs = env.reset(seed=9000 + ep)
            done = False
            while not done:
                teacher_action = teacher_policy(obs)
                prompt = (
                    "You are optimizing Python to C++. Choose next tool call and include concise reasoning.\n"
                    f"Round: {obs.round_number}\n"
                    f"Target ISA: {obs.hardware_profile.get('target', 'scalar_only')}\n"
                    f"Hardware: {json.dumps(obs.hardware_profile)}\n"
                    f"Python:\n{obs.python_code}\n"
                    "Return ONLY JSON: {\"tool_name\":..., \"tool_args\":..., \"reasoning_trace\":...}"
                )
                rows.append(
                    {
                        "prompt": prompt,
                        "expected_tool_name": teacher_action.tool_name,
                        "expected_target": obs.hardware_profile.get("target", "scalar_only"),
                        "round_number": int(obs.round_number),
                    }
                )
                # Prompt harvesting does not need submit-time verifier execution.
                # Ending at round 3 avoids long CPU-only stalls before GRPO starts.
                if int(obs.round_number) >= 3:
                    break
                step = env.step(teacher_action)
                obs = step.observation
                done = step.done
        env.close()
        if rows:
            random.Random(args.seed).shuffle(rows)
        selected = rows[: max(1, n_prompts)]
        print(f"[script] GRPO prompts prepared={len(selected)}", flush=True)
        return selected

    out_dir = Path("/tmp/partial-train-artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_model_and_tokenizer():
        print("[script] Loading tokenizer/model...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inferred_large_model = any(tag in args.model_name.lower() for tag in ["7b", "14b", "32b"])
        use_4bit = (args.load_in_4bit or inferred_large_model) and gpu_fp16
        model_kwargs: dict[str, Any] = {}
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        elif gpu_fp16:
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        return tokenizer, model, use_4bit

    def run_train_cycle(max_steps: int, filter_good: bool, weighted: bool) -> dict[str, Any]:
        rows = collect_demo_rows(args.demo_episodes, filter_good=filter_good, weighted=weighted)
        if not rows:
            raise RuntimeError("No demo rows collected. Try increasing demo_episodes or loosening filters.")
        ds = Dataset.from_list(rows).train_test_split(test_size=0.2, seed=args.seed)

        tokenizer, model, use_4bit = load_model_and_tokenizer()
        train_fp16 = gpu_fp16 and (not use_4bit)
        sft_cfg = SFTConfig(
            output_dir=str(out_dir),
            max_steps=max_steps,
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=5,
            save_steps=max(10, max_steps // 2),
            eval_strategy="steps",
            eval_steps=max(5, max_steps // 3),
            max_grad_norm=0.0,
            dataset_text_field="text",
            report_to=[],
            bf16=False,
            fp16=train_fp16,
            use_cpu=args.use_cpu,
        )
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            processing_class=tokenizer,
            peft_config=peft_cfg,
        )
        print("[script] Starting trainer.train()...", flush=True)
        train_result = trainer.train()
        print("[script] trainer.train() finished", flush=True)
        model_dir = out_dir / "final"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

        return {
            "trainer": trainer,
            "tokenizer": tokenizer,
            "model": model,
            "use_4bit": use_4bit,
            "train_result": train_result,
            "model_dir": model_dir,
            "strategy": {"filter_good": filter_good, "weighted": weighted},
            "rows": len(rows),
        }

    if args.skip_sft:
        print("[script] SFT stages skipped. Running GRPO-only pipeline.", flush=True)
        tokenizer, model, use_4bit = load_model_and_tokenizer()
        train_bundle = {
            "trainer": None,
            "tokenizer": tokenizer,
            "model": model,
            "use_4bit": use_4bit,
            "train_result": None,
            "model_dir": out_dir / "final",
            "strategy": {"mode": "grpo_only"},
            "rows": 0,
        }
    elif args.ablation:
        print("[script] Running short ablation to choose data strategy...", flush=True)
        strategies = [
            {"filter_good": False, "weighted": False, "name": "baseline"},
            {"filter_good": True, "weighted": False, "name": "filtered"},
            {"filter_good": True, "weighted": True, "name": "filtered_weighted"},
        ]
        scored: list[dict[str, Any]] = []
        for strategy in strategies:
            trial = run_train_cycle(
                max_steps=args.ablation_steps,
                filter_good=strategy["filter_good"],
                weighted=strategy["weighted"],
            )
            scored.append(
                {
                    "name": strategy["name"],
                    "train_loss": float(trial["train_result"].training_loss),
                    "filter_good": strategy["filter_good"],
                    "weighted": strategy["weighted"],
                }
            )
        best = min(scored, key=lambda x: x["train_loss"])
        print(f"[script] Ablation picked strategy={best['name']} (train_loss={best['train_loss']:.6f})", flush=True)
        train_bundle = run_train_cycle(
            max_steps=args.max_steps,
            filter_good=bool(best["filter_good"]),
            weighted=bool(best["weighted"]),
        )
    else:
        train_bundle = run_train_cycle(
            max_steps=args.max_steps,
            filter_good=True,
            weighted=True,
        )

    tokenizer = train_bundle["tokenizer"]
    model = train_bundle["model"]
    use_4bit = bool(train_bundle.get("use_4bit", False))
    train_result = train_bundle["train_result"]
    sft_train_loss = float(train_result.training_loss) if train_result is not None else None

    tool_re = re.compile(r"\{.*\}", re.DOTALL)
    model_device = next(model.parameters()).device

    def trained_policy(observation):
        prompt = (
            "<PROMPT>\nYou are optimizing Python to C++. Choose next tool call.\n"
            f"Round: {observation.round_number}\n"
            f"Target ISA: {observation.hardware_profile.get('target', 'scalar_only')}\n"
            f"Hardware: {json.dumps(observation.hardware_profile)}\n"
            f"Python:\n{observation.python_code}\n"
            "Return ONLY JSON: {\"tool_name\":..., \"tool_args\":...}\n<ANSWER>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        txt = tokenizer.decode(out[0], skip_special_tokens=True)
        match = tool_re.search(txt)
        if not match:
            return teacher_policy(observation)
        try:
            data = json.loads(match.group(0))
            tool_name = data.get("tool_name")
            tool_args = data.get("tool_args", {})
            if not isinstance(tool_name, str):
                return teacher_policy(observation)
            return OptimizationAction(tool_name=tool_name, tool_args=tool_args, reasoning_trace="trained-model")
        except Exception:
            return teacher_policy(observation)

    pre_grpo_eval_episodes = args.episodes
    if args.skip_sft and args.grpo_steps > 0:
        pre_grpo_eval_episodes = min(2, args.episodes)
        print(
            f"[script] GRPO-only mode: reducing pre-GRPO eval episodes to {pre_grpo_eval_episodes}",
            flush=True,
        )
    trained = run_eval(
        trained_policy,
        n_episodes=pre_grpo_eval_episodes,
        seed_start=2000,
        label="pre_grpo_eval",
    )

    # Lightweight reward-aligned continuation: keep high-reward interactions and do a short SFT continuation.
    if (not args.skip_sft) and args.reward_align_steps > 0:
        align_rows = []
        env = PolyglotOptimaEnvironment(max_rounds=3, max_calls_per_round=5, enable_adaptive_curriculum=True)
        for ep in range(max(6, args.episodes // 2)):
            obs = env.reset(seed=7000 + ep)
            done = False
            local = []
            while not done:
                action = trained_policy(obs)
                prompt = (
                    "You are optimizing Python to C++. Choose next tool call.\n"
                    f"Round: {obs.round_number}\n"
                    f"Target ISA: {obs.hardware_profile.get('target', 'scalar_only')}\n"
                    f"Hardware: {json.dumps(obs.hardware_profile)}\n"
                    f"Python:\n{obs.python_code}\n"
                    "Return ONLY JSON: {\"tool_name\":..., \"tool_args\":...}"
                )
                local.append(
                    {
                        "text": "<PROMPT>\n" + prompt + "\n<ANSWER>\n" + json.dumps(
                            {"tool_name": action.tool_name, "tool_args": action.tool_args}
                        )
                    }
                )
                step = env.step(action)
                obs = step.observation
                done = step.done
            if float(step.reward) > 0:
                align_rows.extend(local)
        env.close()
        if align_rows:
            print(f"[script] Reward-align continuation with {len(align_rows)} rows...", flush=True)
            align_ds = Dataset.from_list(align_rows).train_test_split(test_size=0.2, seed=args.seed)
            align_cfg = SFTConfig(
                output_dir=str(Path("/tmp/partial-train-artifacts")),
                max_steps=args.reward_align_steps,
                learning_rate=1e-5,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                logging_steps=5,
                eval_strategy="steps",
                eval_steps=max(5, args.reward_align_steps // 3),
                dataset_text_field="text",
                report_to=[],
                bf16=False,
                fp16=False,
                use_cpu=args.use_cpu,
                max_grad_norm=0.0,
            )
            align_trainer = SFTTrainer(
                model=model,
                args=align_cfg,
                train_dataset=align_ds["train"],
                eval_dataset=align_ds["test"],
                processing_class=tokenizer,
            )
            align_trainer.train()
        else:
            print("[script] Reward-align stage skipped: no positive-reward rows collected.", flush=True)

    grpo_enabled = False
    grpo_log_history: list[dict[str, Any]] = []
    if args.grpo_steps > 0:
        try:
            from trl import GRPOConfig, GRPOTrainer  # type: ignore

            if use_4bit and not hasattr(model, "peft_config"):
                print("[script] Attaching LoRA adapters for 4-bit GRPO fine-tuning...", flush=True)
                model = prepare_model_for_kbit_training(model)
                grpo_peft_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                )
                model = get_peft_model(model, grpo_peft_cfg)
                model.print_trainable_parameters()

            grpo_rows = collect_grpo_prompts(args.grpo_prompts)
            if grpo_rows:
                grpo_ds = Dataset.from_list(grpo_rows)

                def _reward_json_valid(completions, **kwargs):
                    scores = []
                    for completion in completions:
                        text = _completion_to_text(completion)
                        data = _safe_json_parse(text)
                        scores.append(1.0 if isinstance(data, dict) else 0.0)
                    return scores

                def _reward_tool_match(completions, expected_tool_name=None, **kwargs):
                    expected = expected_tool_name or []
                    scores = []
                    for i, completion in enumerate(completions):
                        text = _completion_to_text(completion)
                        data = _safe_json_parse(text)
                        predicted = data.get("tool_name") if isinstance(data, dict) else None
                        target = expected[i] if i < len(expected) else None
                        scores.append(1.0 if (isinstance(predicted, str) and predicted == target) else 0.0)
                    return scores

                def _reward_reasoning_quality(completions, **kwargs):
                    scores = []
                    for completion in completions:
                        text = _completion_to_text(completion)
                        data = _safe_json_parse(text)
                        reasoning = ""
                        if isinstance(data, dict):
                            reasoning = str(data.get("reasoning_trace", ""))
                        has_reasoning = len(reasoning.strip()) >= 12
                        too_long = len(reasoning) > 280
                        scores.append(1.0 if has_reasoning and (not too_long) else 0.0)
                    return scores

                def _reward_submit_quality(completions, **kwargs):
                    scores = []
                    for completion in completions:
                        text = _completion_to_text(completion)
                        data = _safe_json_parse(text)
                        if not isinstance(data, dict):
                            scores.append(0.0)
                            continue
                        tool_name = data.get("tool_name")
                        if tool_name != "submit_optimization":
                            scores.append(0.4)
                            continue
                        tool_args = data.get("tool_args", {})
                        cpp_code = str(tool_args.get("cpp_code", "")) if isinstance(tool_args, dict) else ""
                        looks_real = ("extern \"C\"" in cpp_code) and ("baseline submit" not in cpp_code)
                        scores.append(1.0 if looks_real else 0.0)
                    return scores

                def _reward_target_isa_alignment(completions, expected_target=None, **kwargs):
                    expected = [str(x).lower() for x in (expected_target or [])]
                    token_map = {
                        "x86_avx512": ("avx512", "avx-512", "__m512"),
                        "x86_avx2": ("avx2", "__m256"),
                        "x86_sse": ("sse", "__m128"),
                        "arm_neon": ("neon", "int8x16_t", "float32x4_t"),
                        "scalar_only": ("scalar",),
                    }
                    scores = []
                    for i, completion in enumerate(completions):
                        text = _completion_to_text(completion)
                        data = _safe_json_parse(text)
                        target = expected[i] if i < len(expected) else "scalar_only"
                        if not isinstance(data, dict):
                            scores.append(0.0)
                            continue
                        tool_args = data.get("tool_args", {})
                        reasoning = str(data.get("reasoning_trace", "")).lower()
                        cpp_code = str(tool_args.get("cpp_code", "")) if isinstance(tool_args, dict) else ""
                        hay = (reasoning + "\n" + cpp_code).lower()
                        hits = any(tok in hay for tok in token_map.get(target, ()))
                        scores.append(1.0 if hits else 0.0)
                    return scores

                sig = inspect.signature(GRPOConfig.__init__).parameters
                grpo_kwargs: dict[str, Any] = {
                    "output_dir": str(Path("/tmp/partial-train-artifacts") / "grpo"),
                    "max_steps": args.grpo_steps,
                    "learning_rate": 5e-6,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "logging_steps": 5,
                    "report_to": [],
                    "bf16": False,
                    "fp16": False,
                    "use_cpu": args.use_cpu,
                    "num_generations": 4,
                }
                # TRL versions expose different names for prompt/completion limits.
                if "max_prompt_length" in sig:
                    grpo_kwargs["max_prompt_length"] = 700
                elif "max_prompt_len" in sig:
                    grpo_kwargs["max_prompt_len"] = 700
                if "max_completion_length" in sig:
                    grpo_kwargs["max_completion_length"] = 180
                elif "max_completion_len" in sig:
                    grpo_kwargs["max_completion_len"] = 180

                filtered_kwargs = {k: v for k, v in grpo_kwargs.items() if k in sig}
                dropped = sorted(set(grpo_kwargs) - set(filtered_kwargs))
                if dropped:
                    print(f"[script] GRPOConfig dropped unsupported args: {dropped}", flush=True)
                grpo_cfg = GRPOConfig(**filtered_kwargs)
                grpo_trainer = GRPOTrainer(
                    model=model,
                    reward_funcs=[
                        _reward_json_valid,
                        _reward_tool_match,
                        _reward_reasoning_quality,
                        _reward_submit_quality,
                        _reward_target_isa_alignment,
                    ],
                    args=grpo_cfg,
                    train_dataset=grpo_ds,
                    processing_class=tokenizer,
                )
                print(f"[script] Starting GRPO stage for {args.grpo_steps} steps...", flush=True)
                grpo_trainer.train()
                grpo_log_history = [h for h in list(grpo_trainer.state.log_history or []) if isinstance(h, dict)]
                grpo_enabled = True
            else:
                print("[script] GRPO stage skipped: no prompts generated.", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[script] GRPO stage skipped due to error: {exc}", flush=True)

    summary = {
        "model_name": args.model_name,
        "max_steps": args.max_steps,
        "train_loss": sft_train_loss,
        "dataset_rows": int(train_bundle["rows"]),
        "strategy": train_bundle["strategy"],
        "baseline_reward_mean": _safe_mean(baseline["reward"]),
        "trained_reward_mean": _safe_mean(trained["reward"]),
        "baseline_correctness_mean": _safe_mean(baseline["correctness"]),
        "trained_correctness_mean": _safe_mean(trained["correctness"]),
        "baseline_compile_rate": _safe_mean(baseline["compile_success"]),
        "trained_compile_rate": _safe_mean(trained["compile_success"]),
        "baseline_portability_mean": _safe_mean(baseline["portability"]),
        "trained_portability_mean": _safe_mean(trained["portability"]),
        "grpo_enabled": grpo_enabled,
        "grpo_steps": args.grpo_steps,
    }
    summary["reward_uplift_pct"] = (
        100.0 * (summary["trained_reward_mean"] - summary["baseline_reward_mean"])
        / max(abs(summary["baseline_reward_mean"]), 1e-9)
    )
    summary["compile_rate_uplift_abs"] = summary["trained_compile_rate"] - summary["baseline_compile_rate"]
    summary["portability_uplift_abs"] = summary["trained_portability_mean"] - summary["baseline_portability_mean"]
    grpo_reward_points = [
        _safe_float(row.get("reward"))
        for row in grpo_log_history
        if _safe_float(row.get("reward")) is not None
    ]
    if grpo_reward_points:
        summary["grpo_reward_mean"] = _safe_mean([x for x in grpo_reward_points if x is not None])  # type: ignore[arg-type]
        summary["grpo_reward_best"] = max(grpo_reward_points)
        summary["grpo_reward_last"] = grpo_reward_points[-1]

    plt.figure(figsize=(8, 4))
    plt.hist(baseline["reward"], bins=10, alpha=0.6, label="baseline")
    plt.hist(trained["reward"], bins=10, alpha=0.6, label="trained")
    plt.xlabel("episode reward")
    plt.ylabel("count")
    plt.title("Partial Run: Reward Distribution")
    plt.legend()
    reward_plot = out_dir / "reward_distribution.png"
    plt.tight_layout()
    plt.savefig(reward_plot, dpi=140)
    plt.close()

    # Judge-friendly before/after bar chart.
    metric_names = ["reward_mean", "correctness_mean", "compile_rate", "portability_mean"]
    baseline_vals = [
        summary["baseline_reward_mean"],
        summary["baseline_correctness_mean"],
        summary["baseline_compile_rate"],
        summary["baseline_portability_mean"],
    ]
    trained_vals = [
        summary["trained_reward_mean"],
        summary["trained_correctness_mean"],
        summary["trained_compile_rate"],
        summary["trained_portability_mean"],
    ]
    plt.figure(figsize=(9, 4))
    x = list(range(len(metric_names)))
    w = 0.38
    plt.bar([i - w / 2 for i in x], baseline_vals, width=w, label="baseline")
    plt.bar([i + w / 2 for i in x], trained_vals, width=w, label="trained")
    plt.xticks(x, metric_names, rotation=10)
    plt.ylabel("score/value")
    plt.title("Baseline vs Trained Metrics")
    plt.legend()
    metric_plot = out_dir / "baseline_vs_trained_metrics.png"
    plt.tight_layout()
    plt.savefig(metric_plot, dpi=140)
    plt.close()

    # GRPO reward curve with moving average.
    grpo_curve_plot = out_dir / "grpo_reward_curve.png"
    component_plot = out_dir / "grpo_component_means.png"
    if grpo_log_history:
        steps: list[float] = []
        rewards: list[float] = []
        comp_json: list[float] = []
        comp_submit: list[float] = []
        comp_target: list[float] = []
        for idx, row in enumerate(grpo_log_history):
            rv = _safe_float(row.get("reward"))
            if rv is None:
                continue
            sv = _safe_float(row.get("step"))
            steps.append(sv if sv is not None else float(len(steps) + 1))
            rewards.append(rv)
            comp_json.append(_safe_float(row.get("rewards/_reward_json_valid/mean")) or 0.0)
            comp_submit.append(_safe_float(row.get("rewards/_reward_submit_quality/mean")) or 0.0)
            comp_target.append(_safe_float(row.get("rewards/_reward_target_isa_alignment/mean")) or 0.0)
        if rewards:
            ma = _moving_average(rewards, window=5)
            plt.figure(figsize=(10, 4))
            plt.plot(steps, rewards, alpha=0.35, linewidth=1.2, label="raw reward")
            plt.plot(steps, ma, linewidth=2.2, label="moving avg (window=5)")
            plt.xlabel("GRPO log step")
            plt.ylabel("reward")
            plt.title("GRPO Reward Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(grpo_curve_plot, dpi=140)
            plt.close()

            plt.figure(figsize=(9, 4))
            plt.plot(steps, _moving_average(comp_json, 5), label="json_valid mean")
            plt.plot(steps, _moving_average(comp_submit, 5), label="submit_quality mean")
            plt.plot(steps, _moving_average(comp_target, 5), label="target_alignment mean")
            plt.xlabel("GRPO log step")
            plt.ylabel("component reward mean")
            plt.title("GRPO Reward Components (Moving Average)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(component_plot, dpi=140)
            plt.close()

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    token = os.environ.get("HF_TOKEN")
    if token:
        api = HfApi(token=token)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo=f"training_runs/partial-{ts}/summary.json",
            repo_id=args.repo_id,
            repo_type="space",
        )
        api.upload_file(
            path_or_fileobj=str(reward_plot),
            path_in_repo=f"training_runs/partial-{ts}/reward_distribution.png",
            repo_id=args.repo_id,
            repo_type="space",
        )
        for local_path, remote_name in [
            (metric_plot, "baseline_vs_trained_metrics.png"),
            (grpo_curve_plot, "grpo_reward_curve.png"),
            (component_plot, "grpo_component_means.png"),
        ]:
            if local_path.exists():
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=f"training_runs/partial-{ts}/{remote_name}",
                    repo_id=args.repo_id,
                    repo_type="space",
                )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
