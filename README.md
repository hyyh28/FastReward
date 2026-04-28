# FastReward

FastReward studies **efficient reward evaluation** in reinforcement learning under a fixed compute budget.
The core question is: when multiple reward candidates exist, how should training steps be allocated online so that strong candidates are identified faster and more reliably?

This repository implements a simulation-optimization perspective (OCBA-style allocation) and applies it to:

- a controlled sparse-reward benchmark (`MountainCar-v0`),
- an LLM-driven reward search workflow on wildfire suppression (`firecastrl/Wildfire-env0` + OpenEvolve).

## Motivation

Following the paper direction in this repo (`Fast Reward Evaluation in Reinforcement Learning`), reward evaluation is treated as a **sequential budget-allocation** problem:

- each reward candidate induces a policy-training trajectory,
- true performance is measured on the original task objective,
- budget is allocated adaptively using current mean performance and uncertainty.

Compared with uniform allocation, this setup aims to reduce wasted compute on clearly weak candidates and focus more budget on candidates that are promising but uncertain.

## What Is Implemented

### 1) MountainCar verification pipeline

- One PPO model per reward candidate.
- Round-based training and evaluation in both shaped and true environments.
- Allocation strategy switch: `Uniform` vs `OCBA`.
- Structured logging for per-round analysis.

### 2) Firecastrl + OpenEvolve pipeline

- CNN + action-masking training environment for wildfire suppression.
- OpenEvolve generates reward-function candidates (`initial_reward_function` edits).
- Candidate evaluation uses adaptive budget allocation from `src/strategies`.
- Selection is based on true-task behavior metrics (not shaped reward alone).

## Project Structure

- `src/main.py` - entry for MountainCar experiments.
- `src/config.py` - config definitions (`ExperimentConfig`, `MountainCarConfig`).
- `src/experiment/runner.py` - MountainCar train/eval/allocation loop.
- `src/envs/scenario_factory.py` - env factories (MountainCar + Firecastrl maskable CNN vec env).
- `src/envs/reward_wrapper.py` - MountainCar shaping wrapper.
- `src/envs/firecastrl_reward_init.py` - initial reward seed for firecastrl search.
- `src/strategies/uniform.py` - uniform budget allocation.
- `src/strategies/ocba.py` - OCBA-style budget allocation.
- `src/train_firecastrl.py` - standalone MaskablePPO training for firecastrl.
- `src/openevolve_firecastrl/`:
  - `candidate_allocator_eval.py` - adaptive budget evaluator for one reward candidate.
  - `evaluate_firecastrl_reward.py` - OpenEvolve evaluator entry.
  - `run_search.py` - OpenEvolve search launcher.
  - `firecastrl_default.yaml` - default OpenEvolve config (task-specific prompt included).
- `src/monitoring/performance_monitor.py` - round metrics tracking.
- `src/reporting/run_logger.py` - JSON/JSONL persistence.
- `src/process_results/` - plotting and aggregation scripts.
- `run_10_main.sh` - batch runner for repeated MountainCar runs.

## Environment and Dependencies

Recommended:

- Python 3.10+
- macOS/Linux
- Conda env `fastreward` (used in current scripts)

Main packages used:

- `numpy`
- `gymnasium`
- `stable-baselines3`
- `sb3-contrib`
- `firecastrl_env`
- `openevolve`
- `matplotlib`

## Quick Start

### A) MountainCar baseline experiment

```bash
python -m src.main
```

Logs are written to `logs/run_YYYYMMDD_HHMMSS/`.

### B) Firecastrl standalone training

```bash
conda run -n fastreward python -m src.train_firecastrl --fast-mode
```

### C) OpenEvolve reward search on firecastrl

```bash
export OPENAI_API_KEY=your_key
conda run -n fastreward python -m src.openevolve_firecastrl.run_search \
  --config src/openevolve_firecastrl/firecastrl_default.yaml \
  --iterations 20 \
  --allocation-strategy ocba
```

## Firecastrl Reward-Search Notes

- Search target is constrained to:
  - `initial_reward_function(env, prev_state, curr_state) -> float`
- Candidate evaluation is expensive (multi-hour), so default config uses:
  - long evaluator timeout (`12h`),
  - adaptive allocation with warmup + incremental rounds,
  - stability-aware selection metrics.
- Prompt in `firecastrl_default.yaml` is task-specific and encodes:
  - environment/action semantics,
  - true-task objective (wildfire suppression under original reward),
  - anti-reward-hacking guidance.

## Outputs

### MountainCar run outputs

- `summary.json`
- `uniform_rounds.jsonl`
- `ocba_rounds.jsonl`

Common round fields:

- `round_index`
- `budget_consumed`
- `best_true_return`
- per-candidate mean/variance/allocation traces

### OpenEvolve firecastrl outputs

- search logs under `--output-dir` (default `logs/openevolve_firecastrl`)
- best evolved candidate saved to `artifacts/best_reward_candidate.py`

## Visualization

```bash
python src/process_results/mountaincar_allocation_curve.py --log-root logs
python src/process_results/mountaincar_budget_by_candidate.py --log-root logs
```

## Batch Execution

```bash
bash run_10_main.sh
```

Optional env vars:

- `JOBS` (default `10`)
- `PYTHON_BIN` (default `python`)
- `ENTRY_MODULE` (default `src.main`)

## Notes

- Current setup is research-oriented and compute-heavy for firecastrl evaluation.
- If you plan to publish this repository, add a `LICENSE` file and pin dependency versions for reproducibility.
