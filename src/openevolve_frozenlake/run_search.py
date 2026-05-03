import argparse
import json
import os
from pathlib import Path

from openevolve.api import run_evolution
from openevolve.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenEvolve reward search for FrozenLake.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/openevolve_frozenlake/frozenlake_default.yaml",
        help="Path to OpenEvolve YAML config.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="logs/openevolve_frozenlake")
    parser.add_argument("--allocation-strategy", choices=["uniform", "ocba"], default="ocba")
    parser.add_argument("--total-eval-budget", type=int, default=120_000)
    parser.add_argument("--warmup-budget-per-arm", type=int, default=6_000)
    parser.add_argument("--delta-budget", type=int, default=3_000)
    parser.add_argument("--n-arms", type=int, default=4)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-name", type=str, default="4x4", choices=["4x4", "8x8"])
    parser.add_argument("--is-slippery", action="store_true", help="Enable slippery transition dynamics.")
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
    parser.add_argument(
        "--save-best-path",
        type=str,
        default="artifacts/best_frozenlake_reward_candidate.py",
        help="Where to save the best evolved reward program code.",
    )
    parser.add_argument(
        "--export-candidates-manifest",
        type=str,
        default="",
        help="Optional manifest jsonl path for exporting per-iteration candidate programs.",
    )
    return parser.parse_args()


def _set_eval_env(args) -> None:
    os.environ["FROZENLAKE_ALLOC_STRATEGY"] = args.allocation_strategy
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


def _extract_code_from_program_json(payload: dict) -> str:
    candidate_keys = (
        "code",
        "program_code",
        "source_code",
        "program",
        "content",
        "text",
        "response",
        "raw_output",
    )
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, str) and "def initial_reward_function" in value:
            return value
    for key in ("program", "candidate", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested = _extract_code_from_program_json(value)
            if nested:
                return nested
    return ""


def _safe_iteration(value) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def _parse_iteration_from_path(program_json: Path) -> int:
    for ancestor in [program_json.parent, *program_json.parents]:
        name = ancestor.name
        if name.startswith("checkpoint_"):
            try:
                return int(name.split("_", 1)[1])
            except Exception:
                continue
    return -1


def _export_candidates_manifest(output_dir: Path, manifest_path: Path) -> tuple[int, int, int]:
    program_jsons = sorted(output_dir.rglob("programs/*.json"))
    export_dir = manifest_path.parent / "manifest_candidates"
    export_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    decode_failed = 0
    no_code_found = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, program_json in enumerate(program_jsons):
            try:
                payload = json.loads(program_json.read_text(encoding="utf-8"))
            except Exception:
                decode_failed += 1
                continue
            code = _extract_code_from_program_json(payload)
            if not code:
                no_code_found += 1
                continue
            candidate_id = str(
                payload.get("id")
                or payload.get("program_id")
                or payload.get("uuid")
                or program_json.stem
            )
            iteration = _safe_iteration(payload.get("iteration"))
            if iteration < 0:
                iteration = _parse_iteration_from_path(program_json)
            out_py = export_dir / f"iter_{int(iteration):04d}_{candidate_id}.py"
            out_py.write_text(code, encoding="utf-8")
            rec = {
                "index": idx,
                "iteration": int(iteration),
                "candidate_id": candidate_id,
                "program_path": str(out_py),
                "source_json_path": str(program_json),
                "source_run_dir": str(output_dir),
            }
            mf.write(json.dumps(rec, ensure_ascii=True) + "\n")
            exported += 1
    return exported, len(program_jsons), decode_failed + no_code_found


def main():
    args = parse_args()
    _set_eval_env(args)

    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_obj = load_config(str(config_path))
    if not config_obj.llm.models:
        raise ValueError(f"No LLM models configured in config: {config_path}")

    initial_program = project_root / "src" / "envs" / "frozenlake_reward_init.py"
    evaluator_file = project_root / "src" / "openevolve_frozenlake" / "evaluate_frozenlake_reward.py"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_evolution(
        initial_program=str(initial_program),
        evaluator=str(evaluator_file),
        config=config_obj,
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
    manifest_arg = args.export_candidates_manifest.strip()
    if manifest_arg:
        manifest_path = Path(manifest_arg)
    else:
        manifest_path = output_dir / "candidate_manifest.jsonl"
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exported_count, discovered_count, skipped_count = _export_candidates_manifest(
        output_dir=output_dir, manifest_path=manifest_path
    )
    print(f"candidate_manifest={manifest_path}")
    print(f"candidate_manifest_discovered={discovered_count}")
    print(f"candidate_manifest_count={exported_count}")
    print(f"candidate_manifest_skipped={skipped_count}")


if __name__ == "__main__":
    main()
