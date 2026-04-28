import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

CANDIDATES = ["baseline", "assist_pos_vel", "assist_energy_gate", "deceptive_left"]
CANDIDATE_LABELS = {
    "baseline": "baseline",
    "assist_pos_vel": "assist_pos_vel",
    "assist_energy_gate": "assist_energy_gate",
    "deceptive_left": "deceptive_left",
}
CANDIDATE_COLORS = {
    "baseline": "tab:blue",
    "assist_pos_vel": "tab:orange",
    "assist_energy_gate": "tab:green",
    "deceptive_left": "tab:red",
}


def list_run_dirs(log_root: Path):
    run_dirs = [p for p in log_root.glob("run_*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under: {log_root.resolve()} (cwd={Path.cwd()})"
        )
    return sorted(run_dirs)


def read_allocation_curve(jsonl_path: Path):
    # 返回: step -> {candidate: cumulative_allocation}
    step_to_alloc = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = int(rec["budget_consumed"])
            per_candidate = rec.get("per_candidate", {})
            allocs = {}
            for c in CANDIDATES:
                item = per_candidate.get(c)
                allocs[c] = int(item["allocation_cumulative"]) if item is not None else 0
            step_to_alloc[step] = allocs
    return step_to_alloc


def aggregate_allocations(run_dirs, strategy_filename):
    # steps 对齐后跨 run 求均值
    curves = []
    for run_dir in run_dirs:
        path = run_dir / strategy_filename
        if not path.exists():
            continue
        curve = read_allocation_curve(path)
        if curve:
            curves.append(curve)

    if not curves:
        raise FileNotFoundError(f"No valid {strategy_filename} found in run directories.")

    all_steps = sorted({s for curve in curves for s in curve.keys()})
    mean_alloc = {c: [] for c in CANDIDATES}
    n_per_step = []

    for s in all_steps:
        per_candidate_values = {c: [] for c in CANDIDATES}
        for curve in curves:
            if s not in curve:
                continue
            for c in CANDIDATES:
                per_candidate_values[c].append(curve[s][c])

        n = min(len(per_candidate_values[c]) for c in CANDIDATES)
        n_per_step.append(n)
        for c in CANDIDATES:
            vals = per_candidate_values[c]
            mean_alloc[c].append(float(np.mean(vals)) if vals else np.nan)

    steps = np.array(all_steps, dtype=int)
    for c in CANDIDATES:
        arr = np.array(mean_alloc[c], dtype=np.float64)
        # 前向填充，避免部分 step 缺失导致 stackplot 断裂
        last = np.nan
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                arr[i] = last
            else:
                last = arr[i]
        arr = np.where(np.isnan(arr), 0.0, arr)
        mean_alloc[c] = arr

    return steps, mean_alloc, int(len(curves)), np.array(n_per_step, dtype=int)


def plot_budget_allocation(log_root: Path, output_path: Path):
    run_dirs = list_run_dirs(log_root)

    steps_u, alloc_u, runs_u, n_u = aggregate_allocations(run_dirs, "uniform_rounds.jsonl")
    steps_o, alloc_o, runs_o, n_o = aggregate_allocations(run_dirs, "ocba_rounds.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    axes[0].stackplot(
        steps_u,
        [alloc_u[c] for c in CANDIDATES],
        labels=[CANDIDATE_LABELS[c] for c in CANDIDATES],
        colors=[CANDIDATE_COLORS[c] for c in CANDIDATES],
        alpha=0.75,
    )
    axes[0].set_title("Uniform: Mean Cumulative Budget by Candidate")
    axes[0].set_xlabel("Total Training Budget (Steps)")
    axes[0].set_ylabel("Cumulative Allocated Budget")
    axes[0].grid(alpha=0.25)

    axes[1].stackplot(
        steps_o,
        [alloc_o[c] for c in CANDIDATES],
        labels=[CANDIDATE_LABELS[c] for c in CANDIDATES],
        colors=[CANDIDATE_COLORS[c] for c in CANDIDATES],
        alpha=0.75,
    )
    axes[1].set_title("OCBA: Mean Cumulative Budget by Candidate")
    axes[1].set_xlabel("Total Training Budget (Steps)")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("MountainCar: How Budget is Distributed Across Reward Candidates", y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "output": str(output_path),
        "uniform_runs": runs_u,
        "ocba_runs": runs_o,
        "uniform_points": int(len(steps_u)),
        "ocba_points": int(len(steps_o)),
        "uniform_n_min": int(np.min(n_u)) if len(n_u) > 0 else 0,
        "uniform_n_max": int(np.max(n_u)) if len(n_u) > 0 else 0,
        "ocba_n_min": int(np.min(n_o)) if len(n_o) > 0 else 0,
        "ocba_n_max": int(np.max(n_o)) if len(n_o) > 0 else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot how Uniform/OCBA allocate cumulative budget to four reward candidates over total budget."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=SRC_DIR / "logs" / "MountainCar",
        help="Root directory containing run_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <log_root>/budget_by_candidate.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_root = args.log_root if args.log_root.is_absolute() else (PROJECT_ROOT / args.log_root)
    output = args.output if args.output is not None else (log_root / "budget_by_candidate.png")
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    meta = plot_budget_allocation(log_root=log_root, output_path=output)
    print(f"Plot saved to: {meta['output']}")
    print(
        "Aggregation summary: "
        f"uniform_runs={meta['uniform_runs']} (n range {meta['uniform_n_min']}~{meta['uniform_n_max']}), "
        f"ocba_runs={meta['ocba_runs']} (n range {meta['ocba_n_min']}~{meta['ocba_n_max']}), "
        f"uniform_points={meta['uniform_points']}, ocba_points={meta['ocba_points']}"
    )


if __name__ == "__main__":
    main()
