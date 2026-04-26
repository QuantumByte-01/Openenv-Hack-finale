---
title: Polyglot-Optima OpenEnv
emoji: "⚙️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Polyglot-Optima

Polyglot-Optima is an OpenEnv environment for training an LLM to translate Python functions into hardware-aware C++ that is both fast and correct.

## Problem

LLMs can generate optimized code, but often fail on edge-case correctness, portability, and anti-gaming behavior (fast but wrong outputs). This environment targets that gap with closed-loop tool use and verifiable rewards.

## Environment Design

- **API shape:** Gym-style `reset`, `step`, `state`.
- **3-round episodes:** iterative refinement, final submission at round 3.
- **9 tools:** profiling, complexity analysis, memory analysis, compile+benchmark, equivalence verifier, portability checker, and final submit.
- **Reward DAG:** composable rubrics for speedup, correctness, diagnosis quality, portability, and self-correction.
- **Continuous rewards:** no hard 0/1 optimization cliff in the main learning path.

## Innovation Highlights

1. **Adaptive 4-axis curriculum** updates global difficulty over batches.
2. **Adversarial trap library** with category-focused adaptive resampling from recent failures.
3. **Semantic trap variation** (AST-level no-op rewrites) to reduce memorization.
4. **Roofline-aware speedup scoring** for hardware-grounded performance reward.
5. **Anti-gaming verification** through fuzzing + adversarial pass checks.

## Why This Matters

The target behavior is not just "compile and run", but robust optimization under realistic constraints: correctness under adversarial inputs, reasoning about bottlenecks, and hardware-aware strategy selection.

## Local Usage

```bash
python -m pytest -q
python -m ruff check .
```

Run smoke LLM integration:

```bash
python tests/smoke_llm_hf.py
```

Cursor/OpenAI-compatible provider mode:

```bash
export LLM_PROVIDER=cursor
export CURSOR_API_KEY=...
export CURSOR_MODEL=gpt-4.1-nano
python tests/smoke_llm_hf.py
```

## Notebook Usage and HF Spaces

You can use this environment directly in a local notebook without deploying to HF Spaces.

- **For development/training:** local usage is enough.
- **For hackathon submission:** deploy to HF Spaces and link it in README per requirements.

## Current Validation Snapshot

- Unit/integration tests passing.
- Smoke integration path validates parseability/tool-loop behavior.
- Reward and gate tests verify coherent scoring behavior.

## Results (Judge-facing)

After running `training/openenv_hackathon_training_colab_grpo_only.ipynb`, add:

- Reward distribution plot: `docs/plots/reward_distribution_baseline_vs_trained.png`
- Correctness curve plot: `docs/plots/correctness_baseline_vs_trained.png`
- Baseline vs trained metrics table (reward mean, correctness, compile rate, portability).

## Required Submission Links

Fill these links before final submission:

- **HF Space (environment URL judges will pull):** `TODO_ADD_HF_SPACE_URL`
- **Training notebook (repo link):** `training/openenv_hackathon_training_colab_grpo_only.ipynb`
- **Public Colab notebook (optional but recommended):** `TODO_ADD_COLAB_NOTEBOOK_URL`
- **W&B run (or equivalent training evidence):** `TODO_ADD_WANDB_RUN_URL`
- **Short writeup/video/slides (<2 min video or mini blog):** `TODO_ADD_STORY_URL`
- **Hugging Face blog post markdown path in repo (if used):** `HF_BLOG_WRITEUP.md`

## Submission Checklist (Final)

### Mandatory (from hackathon rules)

- [ ] OpenEnv environment is deployed to HF Space and the URL is in this README
- [x] Valid OpenEnv manifest exists (`openenv.yaml`)
- [x] Training pipeline exists using TRL/Unsloth
- [ ] Real training evidence is included (loss/reward plots from an actual run)
- [ ] README contains all judge-facing links (Space + training evidence + story asset)

### Notebook and Reproducibility

- [ ] `training/openenv_hackathon_training_colab_grpo_only.ipynb` runs end-to-end without errors
- [ ] Notebook execution produces visible output cells for metrics and plots
- [ ] Notebook includes/prints final summary metrics (baseline vs trained)
- [ ] Share at least one runnable training entry:
  - [ ] Repo notebook link (`training/openenv_hackathon_training_colab_grpo_only.ipynb`), or
  - [ ] Public Google Colab link

### Plots and Evidence Quality

- [ ] Key plots are committed and accessible in repo (`docs/plots/*.png` or `training_runs/...`)
- [ ] Plot axes/titles are readable (step/episode vs reward/loss/metrics)
- [ ] Baseline vs trained comparison table is present in README
- [ ] Include one clear claim metric (for example compile-rate uplift or portability uplift)

### Storytelling Assets

- [ ] Add one short story asset and link it in README:
  - [ ] HF mini-blog (markdown article stored in repo and linked), or
  - [ ] <2 minute YouTube demo, or
  - [ ] Short slide deck
- [ ] Do not upload large video binaries into the env repo; link externally instead
