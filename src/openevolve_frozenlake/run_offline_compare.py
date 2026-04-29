import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.openevolve_frozenlake.evaluate_frozenlake_reward import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Offline OCBA vs Uniform comparison for FrozenLake candidates.")
    parser.add_argument("--candidates-dir", type=str, required=True)
    parser.add_argument("--glob", type=str, default="*.py")
    parser.add_argument("--output", type=str, default="logs/openevolve_frozenlake/offline_compare.jsonl")
    parser.add_argument("--total-eval-budget", type=int, default=120_000)
    parser.add_argument("--warmup-budget-per-arm", type=int, default=6_000)
    parser.add_argument("--delta-budget", type=int, default=3_000)
    parser.add_argument("--n-arms", type=int, default=4)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-name", type=str, default="4x4", choices=["4x4", "8x8"])
    parser.add_argument("--is-slippery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--n-eval-episodes", type=int, default=60)
    parser.add_argument("--early-stop-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-z", type=float, default=1.96)
    parser.add_argument("--early-stop-margin", type=float, default=0.01)
    parser.add_argument("--min-rounds-before-stop", type=int, default=2)
    parser.add_argument("--elimination-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elimination-z", type=float, default=1.0)
    parser.add_argument("--min-active-arms", type=int, default=2)
    return parser.parse_args()


def _apply_eval_env(args) -> None:
    os.environ["FROZENLAKE_TOTAL_BUDGET"] = str(args.total_eval_budget)
    os.environ["FROZENLAKE_WARMUP_BUDGET_PER_ARM"] = str(args.warmup_budget_per_arm)
    os.environ["FROZENLAKE_DELTA_BUDGET"] = str(args.delta_budget)
    os.environ["FROZENLAKE_EVAL_ARMS"] = str(args.n_arms)
    os.environ["FROZENLAKE_EVAL_N_ENVS"] = str(args.n_envs)
    os.environ["FROZENLAKE_EVAL_SEED"] = str(args.seed)
    os.environ["FROZENLAKE_MAP_NAME"] = args.map_name
    os.environ["FROZENLAKE_IS_SLIPPERY"] = "true" if args.is_slippery else "false"
    os.environ["FROZENLAKE_N_STEPS"] = str(args.n_steps)
    os.environ["FROZENLAKE_BATCH_SIZE"] = str(args.batch_size)
    os.environ["FROZENLAKE_N_EPOCHS"] = str(args.n_epochs)
    os.environ["FROZENLAKE_N_EVAL_EPISODES"] = str(args.n_eval_episodes)
    os.environ["FROZENLAKE_EARLY_STOP_ENABLE"] = "true" if args.early_stop_enable else "false"
    os.environ["FROZENLAKE_EARLY_STOP_Z"] = str(args.early_stop_z)
    os.environ["FROZENLAKE_EARLY_STOP_MARGIN"] = str(args.early_stop_margin)
    os.environ["FROZENLAKE_MIN_ROUNDS_BEFORE_STOP"] = str(args.min_rounds_before_stop)
    os.environ["FROZENLAKE_ELIMINATION_ENABLE"] = "true" if args.elimination_enable else "false"
    os.environ["FROZENLAKE_ELIMINATION_Z"] = str(args.elimination_z)
    os.environ["FROZENLAKE_MIN_ACTIVE_ARMS"] = str(args.min_active_arms)


def _evaluate_with_strategy(program_path: str, strategy: str) -> dict[str, float]:
    os.environ["FROZENLAKE_ALLOC_STRATEGY"] = strategy
    return evaluate(program_path)


def main():
    args = parse_args()
    _apply_eval_env(args)

    candidates_dir = Path(args.candidates_dir)
    if not candidates_dir.exists():
        raise FileNotFoundError(f"Candidates directory not found: {candidates_dir}")

    program_paths = sorted(candidates_dir.glob(args.glob))
    if not program_paths:
        raise RuntimeError(f"No candidate files matched: {candidates_dir}/{args.glob}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, program_path in enumerate(program_paths):
            ocba_metrics = _evaluate_with_strategy(str(program_path), "ocba")
            uniform_metrics = _evaluate_with_strategy(str(program_path), "uniform")
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_index": idx,
                "program_path": str(program_path),
                "ocba": ocba_metrics,
                "uniform": uniform_metrics,
                "delta_combined_score_ocba_minus_uniform": float(
                    ocba_metrics.get("combined_score", -1e9) - uniform_metrics.get("combined_score", -1e9)
                ),
                "delta_best_true_reward_ocba_minus_uniform": float(
                    ocba_metrics.get("best_true_reward", -1e9) - uniform_metrics.get("best_true_reward", -1e9)
                ),
                "delta_budget_used_ocba_minus_uniform": float(
                    ocba_metrics.get("budget_used", 0.0) - uniform_metrics.get("budget_used", 0.0)
                ),
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"compared_candidates={len(program_paths)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
