import importlib.util
import os
from types import ModuleType

from src.openevolve_firecastrl.candidate_allocator_eval import CandidateAllocatorEvaluator, CandidateEvalConfig


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


def _build_eval_config(stage_scale: float, seed_offset: int = 0) -> CandidateEvalConfig:
    strategy_name = os.getenv("FIRECASTRL_ALLOC_STRATEGY", "ocba")
    n_arms = int(os.getenv("FIRECASTRL_EVAL_ARMS", "2"))
    n_envs = int(os.getenv("FIRECASTRL_EVAL_N_ENVS", "1"))
    spray_radius = int(os.getenv("FIRECASTRL_SPRAY_RADIUS", "5"))
    base_seed = int(os.getenv("FIRECASTRL_EVAL_SEED", "42")) + seed_offset
    base_total_budget = int(os.getenv("FIRECASTRL_TOTAL_BUDGET", "220000"))
    base_warmup = int(os.getenv("FIRECASTRL_WARMUP_BUDGET_PER_ARM", "50000"))
    base_delta = int(os.getenv("FIRECASTRL_DELTA_BUDGET", "15000"))
    n_steps = int(os.getenv("FIRECASTRL_N_STEPS", "512"))
    batch_size = int(os.getenv("FIRECASTRL_BATCH_SIZE", "128"))
    n_epochs = int(os.getenv("FIRECASTRL_N_EPOCHS", "3"))
    n_eval_episodes = int(os.getenv("FIRECASTRL_N_EVAL_EPISODES", "2"))

    total_budget = max(int(base_total_budget * stage_scale), n_envs * n_steps)
    warmup_budget = max(int(base_warmup * stage_scale), n_envs * n_steps)
    delta_budget = max(int(base_delta * stage_scale), n_envs * n_steps)

    return CandidateEvalConfig(
        strategy_name=strategy_name,
        n_arms=n_arms,
        n_envs=n_envs,
        spray_radius=spray_radius,
        seed=base_seed,
        total_budget=total_budget,
        warmup_budget_per_arm=warmup_budget,
        delta_budget=delta_budget,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_eval_episodes=n_eval_episodes,
    )


def _evaluate(program_path: str, stage_scale: float, seed_offset: int = 0) -> dict[str, float]:
    try:
        module = _load_candidate_module(program_path)
        reward_fn = _extract_reward_fn(module)
        evaluator = CandidateAllocatorEvaluator(_build_eval_config(stage_scale, seed_offset=seed_offset))
        metrics = evaluator.evaluate_reward_function(reward_fn)
        metrics["stage_scale"] = float(stage_scale)
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
