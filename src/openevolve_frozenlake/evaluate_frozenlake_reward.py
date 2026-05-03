import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

from src.openevolve_frozenlake.candidate_allocator_eval import (
    CandidateAllocatorEvaluator,
    CandidateEvalConfig,
)


def _load_candidate_module(program_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("candidate_reward_program", program_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate program from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_reward_fn(module: ModuleType):
    if not hasattr(module, "initial_reward_function"):
        raise AttributeError("Candidate must define initial_reward_function(env, prev_state, curr_state)")
    reward_fn = getattr(module, "initial_reward_function")
    if not callable(reward_fn):
        raise TypeError("initial_reward_function exists but is not callable")
    return reward_fn


def _build_eval_config(
    stage_scale: float,
    strategy_name: str,
    seed_offset: int = 0,
) -> CandidateEvalConfig:
    n_arms = int(os.getenv("FROZENLAKE_EVAL_ARMS", "4"))
    n_envs = int(os.getenv("FROZENLAKE_EVAL_N_ENVS", "4"))
    map_name = os.getenv("FROZENLAKE_MAP_NAME", "4x4")
    is_slippery = os.getenv("FROZENLAKE_IS_SLIPPERY", "true").strip().lower() == "true"
    base_seed = int(os.getenv("FROZENLAKE_EVAL_SEED", "42")) + seed_offset
    base_total_budget = int(os.getenv("FROZENLAKE_TOTAL_BUDGET", "120000"))
    base_warmup = int(os.getenv("FROZENLAKE_WARMUP_BUDGET_PER_ARM", "6000"))
    base_delta = int(os.getenv("FROZENLAKE_DELTA_BUDGET", "3000"))
    n_steps = int(os.getenv("FROZENLAKE_N_STEPS", "256"))
    batch_size = int(os.getenv("FROZENLAKE_BATCH_SIZE", "64"))
    n_epochs = int(os.getenv("FROZENLAKE_N_EPOCHS", "4"))
    n_eval_episodes = int(os.getenv("FROZENLAKE_N_EVAL_EPISODES", "60"))
    early_stop_enable = os.getenv("FROZENLAKE_EARLY_STOP_ENABLE", "true").strip().lower() == "true"
    early_stop_z = float(os.getenv("FROZENLAKE_EARLY_STOP_Z", "1.96"))
    early_stop_margin = float(os.getenv("FROZENLAKE_EARLY_STOP_MARGIN", "0.01"))
    min_rounds_before_stop = int(os.getenv("FROZENLAKE_MIN_ROUNDS_BEFORE_STOP", "2"))
    elimination_enable = os.getenv("FROZENLAKE_ELIMINATION_ENABLE", "true").strip().lower() == "true"
    elimination_z = float(os.getenv("FROZENLAKE_ELIMINATION_Z", "1.0"))
    min_active_arms = int(os.getenv("FROZENLAKE_MIN_ACTIVE_ARMS", "2"))

    total_budget = max(int(base_total_budget * stage_scale), n_envs * n_steps)
    warmup_budget = max(int(base_warmup * stage_scale), n_envs * n_steps)
    delta_budget = max(int(base_delta * stage_scale), n_envs * n_steps)

    return CandidateEvalConfig(
        strategy_name=strategy_name,
        n_arms=n_arms,
        n_envs=n_envs,
        seed=base_seed,
        total_budget=total_budget,
        warmup_budget_per_arm=warmup_budget,
        delta_budget=delta_budget,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_eval_episodes=n_eval_episodes,
        map_name=map_name,
        is_slippery=is_slippery,
        early_stop_enable=early_stop_enable,
        early_stop_z=early_stop_z,
        early_stop_margin=early_stop_margin,
        min_rounds_before_stop=min_rounds_before_stop,
        elimination_enable=elimination_enable,
        elimination_z=elimination_z,
        min_active_arms=min_active_arms,
    )


def _strategy_mode() -> str:
    mode = os.getenv("FROZENLAKE_ALLOC_STRATEGY", "ocba").strip().lower()
    if mode not in {"both", "ocba", "uniform"}:
        return "ocba"
    return mode


def _run_single_strategy(
    reward_fn,
    strategy_name: str,
    stage_scale: float,
    seed_offset: int,
) -> dict[str, float]:
    evaluator = CandidateAllocatorEvaluator(
        _build_eval_config(
            stage_scale=stage_scale,
            strategy_name=strategy_name,
            seed_offset=seed_offset,
        )
    )
    return evaluator.evaluate_reward_function(reward_fn)


def _append_compare_log(payload: dict) -> None:
    compare_log = os.getenv("FROZENLAKE_COMPARE_LOG", "").strip()
    if not compare_log:
        return
    log_path = Path(compare_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _evaluate(program_path: str, stage_scale: float, seed_offset: int = 0) -> dict[str, float]:
    try:
        module = _load_candidate_module(program_path)
        reward_fn = _extract_reward_fn(module)
        mode = _strategy_mode()
        strategy_seed_base = seed_offset + 1_000

        strategy_metrics: dict[str, dict[str, float]] = {}
        if mode in {"both", "ocba"}:
            strategy_metrics["ocba"] = _run_single_strategy(
                reward_fn=reward_fn,
                strategy_name="improved_ocba",
                stage_scale=stage_scale,
                seed_offset=strategy_seed_base,
            )
        if mode in {"both", "uniform"}:
            strategy_metrics["uniform"] = _run_single_strategy(
                reward_fn=reward_fn,
                strategy_name="uniform",
                stage_scale=stage_scale,
                seed_offset=strategy_seed_base,
            )

        if "ocba" in strategy_metrics and "uniform" in strategy_metrics:
            ocba_metrics = strategy_metrics["ocba"]
            uniform_metrics = strategy_metrics["uniform"]
            metrics = {
                "combined_score": float(
                    0.5 * ocba_metrics.get("combined_score", -1e9)
                    + 0.5 * uniform_metrics.get("combined_score", -1e9)
                ),
                "best_true_reward": float(
                    max(
                        ocba_metrics.get("best_true_reward", -1e9),
                        uniform_metrics.get("best_true_reward", -1e9),
                    )
                ),
                "avg_true_reward": float(
                    0.5 * ocba_metrics.get("avg_true_reward", -1e9)
                    + 0.5 * uniform_metrics.get("avg_true_reward", -1e9)
                ),
                "stability": float(
                    0.5 * ocba_metrics.get("stability", 0.0)
                    + 0.5 * uniform_metrics.get("stability", 0.0)
                ),
                "valid": float(
                    1.0
                    if ocba_metrics.get("valid", 0.0) > 0 and uniform_metrics.get("valid", 0.0) > 0
                    else 0.0
                ),
                "ocba_combined_score": float(ocba_metrics.get("combined_score", -1e9)),
                "uniform_combined_score": float(uniform_metrics.get("combined_score", -1e9)),
                "ocba_best_true_reward": float(ocba_metrics.get("best_true_reward", -1e9)),
                "uniform_best_true_reward": float(uniform_metrics.get("best_true_reward", -1e9)),
                "ocba_avg_true_reward": float(ocba_metrics.get("avg_true_reward", -1e9)),
                "uniform_avg_true_reward": float(uniform_metrics.get("avg_true_reward", -1e9)),
                "ocba_stability": float(ocba_metrics.get("stability", 0.0)),
                "uniform_stability": float(uniform_metrics.get("stability", 0.0)),
                "ocba_budget_used": float(ocba_metrics.get("budget_used", 0.0)),
                "uniform_budget_used": float(uniform_metrics.get("budget_used", 0.0)),
                "delta_best_true_reward_ocba_minus_uniform": float(
                    ocba_metrics.get("best_true_reward", -1e9)
                    - uniform_metrics.get("best_true_reward", -1e9)
                ),
                "delta_combined_score_ocba_minus_uniform": float(
                    ocba_metrics.get("combined_score", -1e9)
                    - uniform_metrics.get("combined_score", -1e9)
                ),
            }
        else:
            single_name = "ocba" if "ocba" in strategy_metrics else "uniform"
            single_metrics = strategy_metrics[single_name]
            metrics = dict(single_metrics)
            metrics["selected_strategy"] = 1.0 if single_name == "ocba" else 0.0

        metrics["stage_scale"] = float(stage_scale)
        metrics["comparison_mode"] = {"both": 2.0, "ocba": 1.0, "uniform": 0.0}[mode]

        _append_compare_log(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "program_path": program_path,
                "stage_scale": float(stage_scale),
                "mode": mode,
                "metrics": metrics,
            }
        )
        return metrics
    except Exception:
        return {
            "combined_score": -1e9,
            "best_true_reward": -1e9,
            "avg_true_reward": -1e9,
            "stability": 0.0,
            "valid": 0.0,
            "stage_scale": float(stage_scale),
        }


def evaluate(program_path: str) -> dict[str, float]:
    return _evaluate(program_path, stage_scale=1.0, seed_offset=0)


def evaluate_stage1(program_path: str) -> dict[str, float]:
    metrics = _evaluate(program_path, stage_scale=0.25, seed_offset=1000)
    metrics["stage1_passed"] = 1.0 if metrics.get("valid", 0.0) > 0 else 0.0
    return metrics


def evaluate_stage2(program_path: str) -> dict[str, float]:
    metrics = _evaluate(program_path, stage_scale=0.6, seed_offset=2000)
    metrics["stage2_passed"] = 1.0 if metrics.get("valid", 0.0) > 0 else 0.0
    return metrics


def evaluate_stage3(program_path: str) -> dict[str, float]:
    metrics = _evaluate(program_path, stage_scale=1.0, seed_offset=3000)
    metrics["stage3_passed"] = 1.0 if metrics.get("valid", 0.0) > 0 else 0.0
    return metrics
