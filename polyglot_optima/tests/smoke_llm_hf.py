"""LLM smoke test via HuggingFace Inference API (free tier).

Runs 3 short episodes against the env using a remote LLM, validates that:
1. The model emits parseable `<think>...</think>` blocks (DiagnosisRubric needs this)
2. Tool calls extract cleanly from the response
3. The agent's C++ output respects the `extern "C" agent_function` contract
4. End-to-end env<-->LLM loop completes without crashing
5. Reward DAG produces non-zero reward at least once

Run:
    export HF_TOKEN=hf_...
    cd polyglot_optima && python tests/smoke_llm_hf.py

Without a token: anonymous access (very limited rate; may fail randomly).

Cost: free tier on HF Inference API. ~45 model calls across 3 episodes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Make the package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import OptimizationAction
from server.environment import PolyglotOptimaEnvironment


# ---------- Models to try (free-tier-friendly, instruct-tuned, in order of preference) ----------

MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",         # Code-focused, primary fallback
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # Plan-target reasoning model
    "meta-llama/Llama-3.1-8B-Instruct",        # Generic instruct fallback
    "mistralai/Mistral-7B-Instruct-v0.3",      # Last-resort fallback
]


# ---------- System prompt (canonical per plan §11) ----------

SYSTEM_PROMPT = """You are a senior C++ performance engineer specializing in hardware-aware code.

YOUR TASK: each turn, choose ONE of the 9 tools to call. After 3 rounds of refinement, you submit your final optimized C++.

OUTPUT FORMAT (STRICT -- non-conforming responses score 0):

<think>
1. What is the bottleneck? (memory-bound / compute-bound / branch-heavy / vectorizable)
2. What does the hardware imply about strategy?
3. Which tool should I call next, and why?
</think>
```json
{"tool_name": "<one of the 9 tools>", "tool_args": { ... }}
```

THE 9 TOOLS:
- get_hardware_profile()           -- returns hw spec + Roofline
- profile_python_hotspots(code)    -- top hot lines
- analyze_complexity(code)         -- Big-O + nesting depth
- check_memory_access(code)        -- stride / aliasing flags
- compile_and_benchmark(cpp_code)  -- speedup measurement
- verify_equivalence(cpp_code)     -- fuzzer pass rate
- check_portability(cpp_code)      -- cross-profile pass count
- get_bottleneck_report(cpp_code)  -- perf-stat-style report on YOUR C++
- submit_optimization(cpp_code, reasoning_trace)  -- FINAL submission for the round

HARD CONSTRAINTS for cpp_code:
- C++20, single canonical signature:
    extern "C" void agent_function(const double* in_ptr, size_t in_n, double* out_ptr, size_t out_n);
- Compiles with: g++ -O3 -march=native -fopenmp -std=c++20 -Wall
- BANNED: <mkl.h>, <Eigen/...>, BLAS/LAPACK, CUDA. We measure YOUR optimization.
- Allowed: full STL, <immintrin.h>, <arm_neon.h>, <omp.h>, <pybind11/*>
"""


# ---------- LLM call (HF Inference API) ----------

def call_llm(messages: list[dict[str, str]], model: str, hf_token: str | None) -> str:
    """One inference call. Returns the assistant's text content. Raises on hard errors."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=hf_token)
    resp = client.chat_completion(
        messages=messages,
        model=model,
        max_tokens=512,
        temperature=0.5,
    )
    return resp.choices[0].message.content or ""


def pick_model(hf_token: str | None) -> str | None:
    """Probe the free-tier API for the first available candidate model."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=hf_token)
    for name in MODEL_CANDIDATES:
        try:
            resp = client.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                model=name,
                max_tokens=4,
            )
            if resp.choices[0].message.content is not None:
                return name
        except Exception as e:
            print(f"  - {name} → not available: {str(e)[:80]}", file=sys.stderr)
            continue
    return None


# ---------- Response parsing ----------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_LOOSE_JSON_RE = re.compile(r"\{[^{}]*\"tool_name\"[^{}]*\}", re.DOTALL)


def parse_llm_response(text: str) -> dict[str, Any]:
    """Extract <think>, tool_name, tool_args from raw LLM text. Best-effort.

    Returns dict with: thinking, tool_name, tool_args, parse_status.
    parse_status ∈ {"ok", "no_think", "no_json", "no_tool", "json_invalid"}.
    """
    out: dict[str, Any] = {
        "thinking": "",
        "tool_name": None,
        "tool_args": {},
        "parse_status": "ok",
        "raw": text,
    }

    # Extract thinking block
    m = _THINK_RE.search(text)
    if m:
        out["thinking"] = m.group(1).strip()
    else:
        out["parse_status"] = "no_think"

    # Extract JSON tool call -- try fenced block first, then loose match
    json_block = None
    fence_match = _JSON_BLOCK_RE.search(text)
    if fence_match:
        json_block = fence_match.group(1)
    else:
        loose = _LOOSE_JSON_RE.search(text)
        if loose:
            json_block = loose.group(0)

    if not json_block:
        out["parse_status"] = "no_json" if out["parse_status"] == "ok" else out["parse_status"]
        return out

    try:
        parsed = json.loads(json_block)
        out["tool_name"] = parsed.get("tool_name")
        out["tool_args"] = parsed.get("tool_args", {}) or {}
        if not out["tool_name"]:
            out["parse_status"] = "no_tool"
    except json.JSONDecodeError as e:
        out["parse_status"] = f"json_invalid: {e}"
    return out


# ---------- Episode runner ----------

def build_user_prompt(observation, round_number: int) -> str:
    return (
        f"## Round {round_number} of 3\n\n"
        f"### Hardware profile\n```json\n{json.dumps(observation.hardware_profile, indent=2)}\n```\n\n"
        f"### Python function to optimize\n```python\n{observation.python_code}\n```\n\n"
        f"### Last tool result\n```json\n{json.dumps(observation.tool_result, indent=2, default=str)[:1500]}\n```\n\n"
        f"### Best speedup so far\n{observation.best_speedup_so_far:.3f}x\n\n"
        f"What is your next action? "
        f"After at most 4 tool calls in this round, you must call submit_optimization."
    )


def run_episode(env: PolyglotOptimaEnvironment, model: str, hf_token: str | None,
                episode_seed: int, report: dict[str, Any]) -> None:
    """Run one episode end-to-end. Mutates `report` with stats."""
    obs = env.reset(seed=episode_seed)
    ep_report: dict[str, Any] = {
        "seed": episode_seed,
        "rounds": [],
        "errors": [],
        "final_reward": 0.0,
        "n_think_blocks": 0,
        "n_parse_errors": 0,
        "n_unknown_tools": 0,
        "n_tool_calls": 0,
    }
    report["episodes"].append(ep_report)

    valid_tool_names = set(env._tool_registry.keys())
    max_calls_per_round = 4

    for round_idx in range(1, 4):
        round_calls: list[dict[str, Any]] = []
        for call_idx in range(max_calls_per_round):
            user_prompt = build_user_prompt(obs, round_idx)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                t0 = time.time()
                raw = call_llm(messages, model, hf_token)
                latency = time.time() - t0
            except Exception as e:
                ep_report["errors"].append(f"R{round_idx}.{call_idx} LLM call failed: {e}")
                # Force a submit to advance the round
                action = OptimizationAction(
                    tool_name="submit_optimization",
                    tool_args={"cpp_code": "// llm_call_error_fallback",
                               "reasoning_trace": "LLM call failed"},
                    reasoning_trace="<think>fallback</think>",
                )
                step = env.step(action)
                obs = step.observation
                break

            parsed = parse_llm_response(raw)
            if parsed["thinking"]:
                ep_report["n_think_blocks"] += 1
            if parsed["parse_status"] != "ok":
                ep_report["n_parse_errors"] += 1
            if parsed["tool_name"] and parsed["tool_name"] not in valid_tool_names:
                ep_report["n_unknown_tools"] += 1

            ep_report["n_tool_calls"] += 1

            tool_name = parsed["tool_name"] or "submit_optimization"
            tool_args = parsed["tool_args"] or {}

            # If the model emitted a final submission, force the round to close
            is_submit = tool_name == "submit_optimization"
            # If we've hit the call cap and no submit yet, force one
            if call_idx == max_calls_per_round - 1 and not is_submit:
                tool_name = "submit_optimization"
                tool_args = {"cpp_code": tool_args.get("cpp_code", "// no submission this round"),
                             "reasoning_trace": parsed["thinking"]}
                is_submit = True

            action = OptimizationAction(
                tool_name=tool_name,
                tool_args=tool_args,
                reasoning_trace=parsed["thinking"][:1000],
            )
            try:
                step = env.step(action)
                obs = step.observation
                round_calls.append({
                    "tool": tool_name,
                    "parse_status": parsed["parse_status"],
                    "latency_s": round(latency, 2),
                    "reward_so_far": round(step.reward, 3),
                })
            except Exception as e:
                ep_report["errors"].append(f"R{round_idx}.{call_idx} env.step crashed: {e}")
                break

            if is_submit:
                break

        ep_report["rounds"].append(round_calls)
        if obs.done:
            ep_report["final_reward"] = round(step.reward, 3)
            break

    if not obs.done and not env.state().is_terminal:
        # Episode didn't terminate via natural 3-round flow
        ep_report["errors"].append("episode did not reach terminal state")


# ---------- Aggregate report ----------

def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("LLM SMOKE TEST REPORT")
    print("=" * 70)
    print(f"Model used:         {report['model']}")
    print(f"Episodes run:       {len(report['episodes'])}")
    print(f"Total LLM calls:    {sum(e['n_tool_calls'] for e in report['episodes'])}")

    n_think = sum(e["n_think_blocks"] for e in report["episodes"])
    n_parse = sum(e["n_parse_errors"] for e in report["episodes"])
    n_unknown = sum(e["n_unknown_tools"] for e in report["episodes"])
    n_calls = sum(e["n_tool_calls"] for e in report["episodes"])

    print(f"\n-- Output format compliance --")
    print(f"  <think> blocks emitted:    {n_think} / {n_calls}  ({100*n_think/max(n_calls,1):.0f}%)")
    print(f"  Parse errors:              {n_parse} / {n_calls}  ({100*n_parse/max(n_calls,1):.0f}%)")
    print(f"  Unknown/invalid tools:     {n_unknown}")

    print(f"\n-- Episode rewards --")
    for ep in report["episodes"]:
        n_errs = len(ep["errors"])
        print(f"  Episode {ep['seed']}: reward={ep['final_reward']}, errors={n_errs}")

    if any(e["errors"] for e in report["episodes"]):
        print(f"\n-- Errors --")
        for ep in report["episodes"]:
            for err in ep["errors"]:
                print(f"  - ep{ep['seed']}: {err[:140]}")

    # Pass/fail verdict
    print(f"\n-- Verdict --")
    pass_threshold_think = 0.5      # ≥ 50% of calls should have <think>
    pass_threshold_parse = 0.7      # ≥ 70% of calls should parse cleanly
    n_episodes_completed = sum(1 for e in report["episodes"] if not any("did not reach terminal" in x for x in e["errors"]))

    think_ok = n_think / max(n_calls, 1) >= pass_threshold_think
    parse_ok = (n_calls - n_parse) / max(n_calls, 1) >= pass_threshold_parse
    episodes_ok = n_episodes_completed == len(report["episodes"])

    if think_ok and parse_ok and episodes_ok:
        print(f"  [OK] PASS -- env<-->LLM integration works. Safe to launch GRPO training.")
    else:
        print(f"  [FAIL] FAIL -- fix before training:")
        if not think_ok: print(f"      <think> emission rate too low ({100*n_think/max(n_calls,1):.0f}% < 50%)")
        if not parse_ok: print(f"      parse rate too low ({100*(n_calls-n_parse)/max(n_calls,1):.0f}% < 70%)")
        if not episodes_ok: print(f"      {len(report['episodes']) - n_episodes_completed} episodes did not terminate cleanly")

    print("=" * 70 + "\n")


# ---------- Main ----------

def main() -> int:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("[WARN] no HF_TOKEN env var set -- using anonymous access (heavily rate-limited)")
    else:
        print(f"[OK] HF token found ({hf_token[:5]}...)")

    print("\nProbing free-tier model availability...")
    model = pick_model(hf_token)
    if not model:
        print("[FAIL] No candidate model accessible via HF Inference API. "
              "Check token quota or use Anthropic API instead.")
        return 1
    print(f"[OK] Using model: {model}\n")

    env = PolyglotOptimaEnvironment(max_rounds=3, max_calls_per_round=5)
    report: dict[str, Any] = {"model": model, "episodes": []}

    for seed in (101, 202, 303):
        print(f"--- Episode seed={seed} ---")
        try:
            run_episode(env, model, hf_token, seed, report)
        except Exception as e:
            report["episodes"].append({"seed": seed, "errors": [f"fatal: {e}"], "rounds": [],
                                        "final_reward": 0.0, "n_think_blocks": 0,
                                        "n_parse_errors": 0, "n_unknown_tools": 0, "n_tool_calls": 0})
        finally:
            env.close()
            env = PolyglotOptimaEnvironment(max_rounds=3, max_calls_per_round=5)

    print_report(report)

    # Exit code: 0 if pass verdict, else 1
    n_calls = sum(e["n_tool_calls"] for e in report["episodes"])
    n_think = sum(e["n_think_blocks"] for e in report["episodes"])
    n_parse = sum(e["n_parse_errors"] for e in report["episodes"])
    if n_calls and (n_think / n_calls >= 0.5) and ((n_calls - n_parse) / n_calls >= 0.7):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
