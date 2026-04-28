import argparse
import os
from pathlib import Path

from openevolve.api import run_evolution


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenEvolve reward search for firecastrl.")
    parser.add_argument("--config", type=str, required=True, help="Path to OpenEvolve YAML config.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="logs/openevolve_firecastrl")
    parser.add_argument("--allocation-strategy", choices=["uniform", "ocba"], default="uniform")
    parser.add_argument("--total-eval-budget", type=int, default=220_000)
    parser.add_argument("--warmup-budget-per-arm", type=int, default=50_000)
    parser.add_argument("--delta-budget", type=int, default=15_000)
    parser.add_argument("--n-arms", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spray-radius", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--n-eval-episodes", type=int, default=2)
    parser.add_argument(
        "--save-best-path",
        type=str,
        default="artifacts/best_reward_candidate.py",
        help="Where to save the best evolved reward program code.",
    )
    return parser.parse_args()


def _set_eval_env(args) -> None:
    os.environ["FIRECASTRL_ALLOC_STRATEGY"] = args.allocation_strategy
    os.environ["FIRECASTRL_TOTAL_BUDGET"] = str(args.total_eval_budget)
    os.environ["FIRECASTRL_WARMUP_BUDGET_PER_ARM"] = str(args.warmup_budget_per_arm)
    os.environ["FIRECASTRL_DELTA_BUDGET"] = str(args.delta_budget)
    os.environ["FIRECASTRL_EVAL_ARMS"] = str(args.n_arms)
    os.environ["FIRECASTRL_EVAL_N_ENVS"] = str(args.n_envs)
    os.environ["FIRECASTRL_EVAL_SEED"] = str(args.seed)
    os.environ["FIRECASTRL_SPRAY_RADIUS"] = str(args.spray_radius)
    os.environ["FIRECASTRL_N_STEPS"] = str(args.n_steps)
    os.environ["FIRECASTRL_BATCH_SIZE"] = str(args.batch_size)
    os.environ["FIRECASTRL_N_EPOCHS"] = str(args.n_epochs)
    os.environ["FIRECASTRL_N_EVAL_EPISODES"] = str(args.n_eval_episodes)


def main():
    args = parse_args()
    _set_eval_env(args)

    project_root = Path(__file__).resolve().parents[2]
    initial_program = project_root / "src" / "envs" / "firecastrl_reward_init.py"
    evaluator_file = project_root / "src" / "openevolve_firecastrl" / "evaluate_firecastrl_reward.py"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_evolution(
        initial_program=str(initial_program),
        evaluator=str(evaluator_file),
        config=str(args.config),
        iterations=args.iterations,
        output_dir=str(output_dir),
        cleanup=False,
    )

    save_best_path = Path(args.save_best_path)
    save_best_path.parent.mkdir(parents=True, exist_ok=True)
    save_best_path.write_text(result.best_code, encoding="utf-8")

    print(f"best_score={result.best_score:.6f}")
    print(f"metrics={result.metrics}")
    print(f"best_program_saved={save_best_path}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
