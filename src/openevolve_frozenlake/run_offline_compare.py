import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.openevolve_frozenlake.evaluate_frozenlake_reward import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Offline OCBA vs Uniform comparison for FrozenLake candidates.")
    parser.add_argument(
        "--manifest-file",
        type=str,
        default="",
        help="Manifest jsonl exported by run_search.py. Highest priority candidate source.",
    )
    parser.add_argument("--candidates-dir", type=str, default="")
    parser.add_argument(
        "--candidate-file",
        type=str,
        default="",
        help="Single candidate file path. If set, candidates-dir/glob are ignored.",
    )
    parser.add_argument("--glob", type=str, default="*.py")
    parser.add_argument("--output", type=str, default="logs/openevolve_frozenlake/offline_compare.jsonl")
    parser.add_argument(
        "--round-trace-output",
        type=str,
        default="logs/openevolve_frozenlake/offline_compare_rounds.jsonl",
        help="Path to per-round trace jsonl. Use empty string to disable.",
    )
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


def _evaluate_with_strategy(program_path: str, strategy: str, trace_tag: str, trace_output: str) -> dict[str, float]:
    os.environ["FROZENLAKE_ALLOC_STRATEGY"] = strategy
    os.environ["FROZENLAKE_TRACE_TAG"] = trace_tag
    if trace_output.strip():
        os.environ["FROZENLAKE_ROUND_TRACE_PATH"] = trace_output
    else:
        os.environ.pop("FROZENLAKE_ROUND_TRACE_PATH", None)
    return evaluate(program_path)


def _load_manifest_programs(manifest_file: Path) -> list[dict]:
    entries = []
    with manifest_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            program_path = Path(str(rec.get("program_path", "")).strip())
            if not program_path.exists():
                continue
            entries.append(
                {
                    "index": int(rec.get("index", line_idx)),
                    "iteration": int(rec.get("iteration", -1)),
                    "candidate_id": str(rec.get("candidate_id", program_path.stem)),
                    "program_path": str(program_path),
                    "source_json_path": str(rec.get("source_json_path", "")),
                }
            )
    if not entries:
        raise RuntimeError(f"No valid manifest entries found: {manifest_file}")
    entries.sort(key=lambda x: (x["iteration"], x["index"]))
    return entries


def main():
    args = parse_args()
    _apply_eval_env(args)

    program_entries = []
    if args.manifest_file.strip():
        manifest_file = Path(args.manifest_file)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        program_entries = _load_manifest_programs(manifest_file)
    elif args.candidate_file.strip():
        candidate_file = Path(args.candidate_file)
        if not candidate_file.exists():
            raise FileNotFoundError(f"Candidate file not found: {candidate_file}")
        program_entries = [
            {
                "index": 0,
                "iteration": -1,
                "candidate_id": candidate_file.stem,
                "program_path": str(candidate_file),
                "source_json_path": "",
            }
        ]
    elif args.candidates_dir.strip():
        candidates_dir = Path(args.candidates_dir)
        if not candidates_dir.exists():
            raise FileNotFoundError(f"Candidates directory not found: {candidates_dir}")
        program_paths = sorted(candidates_dir.glob(args.glob))
        if not program_paths:
            raise RuntimeError(f"No candidate files matched: {candidates_dir}/{args.glob}")
        program_entries = [
            {
                "index": idx,
                "iteration": -1,
                "candidate_id": p.stem,
                "program_path": str(p),
                "source_json_path": "",
            }
            for idx, p in enumerate(program_paths)
        ]
    else:
        default_candidate = Path("src/envs/frozenlake_reward_init.py")
        if not default_candidate.exists():
            raise FileNotFoundError(f"Default candidate file not found: {default_candidate}")
        program_entries = [
            {
                "index": 0,
                "iteration": -1,
                "candidate_id": default_candidate.stem,
                "program_path": str(default_candidate),
                "source_json_path": "",
            }
        ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.round_trace_output.strip():
        round_trace_path = Path(args.round_trace_output)
        round_trace_path.parent.mkdir(parents=True, exist_ok=True)
        round_trace_path.write_text("", encoding="utf-8")

    with output_path.open("w", encoding="utf-8") as f:
        for item in program_entries:
            idx = int(item["index"])
            program_path = str(item["program_path"])
            iteration = int(item["iteration"])
            candidate_id = str(item["candidate_id"])
            ocba_metrics = _evaluate_with_strategy(
                program_path,
                "ocba",
                trace_tag=f"candidate={idx};iteration={iteration};candidate_id={candidate_id};strategy=ocba",
                trace_output=args.round_trace_output,
            )
            uniform_metrics = _evaluate_with_strategy(
                program_path,
                "uniform",
                trace_tag=f"candidate={idx};iteration={iteration};candidate_id={candidate_id};strategy=uniform",
                trace_output=args.round_trace_output,
            )
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_index": idx,
                "iteration": iteration,
                "candidate_id": candidate_id,
                "program_path": program_path,
                "source_json_path": item.get("source_json_path", ""),
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

    print(f"compared_candidates={len(program_entries)}")
    print(f"output={output_path}")
    if args.round_trace_output.strip():
        print(f"round_trace_output={args.round_trace_output}")


if __name__ == "__main__":
    main()
