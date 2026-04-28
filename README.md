# FastReward

FastReward is a reinforcement learning experiment project for `MountainCar-v0`.
It compares multiple reward-shaping candidates under a fixed training budget and evaluates budget allocation strategies (`Uniform` vs `OCBA`) for faster best-candidate discovery.

## Overview

The project:

- trains one PPO agent per reward-shaping candidate,
- evaluates each candidate in the true environment and shaped environment,
- allocates the next training budget round using an allocation strategy,
- logs per-round metrics to structured files,
- provides post-processing scripts for aggregated plots.

## Project Structure

- `src/main.py` - entry point, runs both `uniform` and `ocba` experiments.
- `src/config.py` - config definitions (`ExperimentConfig` abstract base + `MountainCarConfig`).
- `src/experiment/runner.py` - core experiment loop (train -> evaluate -> allocate -> log).
- `src/envs/reward_wrapper.py` - MountainCar reward shaping wrapper.
- `src/envs/scenario_factory.py` - environment creation with `VecNormalize`.
- `src/strategies/uniform.py` - uniform allocation strategy.
- `src/strategies/ocba.py` - OCBA allocation strategy.
- `src/monitoring/performance_monitor.py` - round-level tracking and console logs.
- `src/reporting/run_logger.py` - persists `summary.json` and `*_rounds.jsonl`.
- `src/process_results/` - plotting/aggregation scripts.
- `run_10_main.sh` - utility script for parallel batch runs.

## Requirements

Recommended:

- Python 3.10+
- macOS or Linux

Core dependencies used in code:

- `numpy`
- `gymnasium`
- `stable-baselines3`
- `matplotlib`

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy gymnasium stable-baselines3 matplotlib
```

## Quick Start

From project root:

```bash
python -m src.main
```

After completion, logs are saved under `logs/run_YYYYMMDD_HHMMSS/`.

## Run in Parallel

Use the provided batch script:

```bash
bash run_10_main.sh
```

Optional environment variables:

- `JOBS` (default `10`) - number of parallel runs.
- `PYTHON_BIN` (default `python`) - python executable.
- `ENTRY_MODULE` (default `src.main`) - module to run.

Example:

```bash
JOBS=6 PYTHON_BIN=python3 bash run_10_main.sh
```

Batch process logs are written to `logs/batch_YYYYMMDD_HHMMSS/`.

## Output Format

Each run folder typically contains:

- `summary.json` - full config and all strategy results.
- `uniform_rounds.jsonl` - per-round records for Uniform.
- `ocba_rounds.jsonl` - per-round records for OCBA.

Each round record includes fields such as:

- `round_index`
- `budget_consumed`
- `best_true_return`
- `per_candidate`:
  - `true_mean_return`
  - `true_return_var`
  - `shaped_mean_return`
  - `allocation_this_round`
  - `allocation_cumulative`

## Visualization

### 1) Learning curve and allocation view

```bash
python src/process_results/mountaincar_allocation_curve.py --log-root logs
```

Common options:

- `--log-root` path containing `run_*` folders.
- `--output` output image path.
- `--smooth` EMA smoothing weight.
- `--max-budget` max budget shown in plot.

### 2) Budget-by-candidate aggregation

```bash
python src/process_results/mountaincar_budget_by_candidate.py --log-root logs
```

## Configuration Design

- `ExperimentConfig` is an abstract base config.
- `MountainCarConfig` is the concrete config used by the current environment.

To support new environments later, add a new config class like `YourEnvConfig(ExperimentConfig)` and corresponding env/runner implementation.

## Typical Workflow

1. Adjust parameters in `MountainCarConfig`.
2. Run `python -m src.main` (or parallel batches).
3. Inspect `logs/run_*/summary.json` and `*_rounds.jsonl`.
4. Run scripts in `src/process_results/` for aggregated analysis and plots.

## Notes

- The repository currently does not declare a license file.
- If you plan to open-source this project, add a `LICENSE`.
