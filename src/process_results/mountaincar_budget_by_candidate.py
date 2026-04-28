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


def ema_smooth(values, weight=0.6):
    if len(values) == 0:
        return values
    out = np.zeros_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = weight * out[i - 1] + (1.0 - weight) * values[i]
    return out


def list_run_dirs(log_root: Path):
    run_dirs = [p for p in log_root.glob("run_*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under: {log_root.resolve()} (cwd={Path.cwd()})"
        )
    return sorted(run_dirs)


def read_round_records(jsonl_path: Path):
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def aggregate_allocations(run_dirs, strategy_filename):
    curves = []
    for run_dir in run_dirs:
        path = run_dir / strategy_filename
        if not path.exists():
            continue
        recs = read_round_records(path)
        if not recs:
            continue

        step_to_alloc = {}
        for rec in recs:
            s = int(rec["budget_consumed"])
            per_candidate = rec.get("per_candidate", {})
            step_to_alloc[s] = {
                c: int(per_candidate.get(c, {}).get("allocation_cumulative", 0))
                for c in CANDIDATES
            }
        curves.append(step_to_alloc)

    if not curves:
        raise FileNotFoundError(f"No valid {strategy_filename} found in run directories.")

    all_steps = sorted({s for curve in curves for s in curve.keys()})
    mean_alloc = {c: [] for c in CANDIDATES}
    n_per_step = []

    for s in all_steps:
        per_c_values = {c: [] for c in CANDIDATES}
        for curve in curves:
            if s not in curve:
                continue
            for c in CANDIDATES:
                per_c_values[c].append(curve[s][c])

        n_per_step.append(min(len(per_c_values[c]) for c in CANDIDATES))
        for c in CANDIDATES:
            vals = per_c_values[c]
            mean_alloc[c].append(float(np.mean(vals)) if vals else np.nan)

    steps = np.array(all_steps, dtype=int)
    for c in CANDIDATES:
        arr = np.array(mean_alloc[c], dtype=np.float64)
        last = np.nan
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                arr[i] = last
            else:
                last = arr[i]
        mean_alloc[c] = np.where(np.isnan(arr), 0.0, arr)

    return steps, mean_alloc, int(len(curves)), np.array(n_per_step, dtype=int)


def compute_final_avg_rank(run_dirs, strategy_filename):
    rank_samples = {c: [] for c in CANDIDATES}
    used_runs = 0

    for run_dir in run_dirs:
        path = run_dir / strategy_filename
        if not path.exists():
            continue
        recs = read_round_records(path)
        if not recs:
            continue

        final_rec = max(recs, key=lambda r: int(r["budget_consumed"]))
        per_candidate = final_rec.get("per_candidate", {})
        vals = np.array(
            [float(per_candidate.get(c, {}).get("true_mean_return", -1e9)) for c in CANDIDATES],
            dtype=np.float64,
        )
        order = np.argsort(-vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(CANDIDATES) + 1)

        for idx, c in enumerate(CANDIDATES):
            rank_samples[c].append(int(ranks[idx]))

        used_runs += 1

    if used_runs == 0:
        raise FileNotFoundError(f"No valid {strategy_filename} for final-rank aggregation.")

    mean_rank = {c: float(np.mean(rank_samples[c])) for c in CANDIDATES}
    return mean_rank, used_runs


def _stack_mid_y(values_by_candidate, candidate_idx, x_idx, sign=1.0):
    vals = np.array([values_by_candidate[c][x_idx] for c in CANDIDATES], dtype=np.float64)
    cum = np.cumsum(vals)
    lower = 0.0 if candidate_idx == 0 else cum[candidate_idx - 1]
    upper = cum[candidate_idx]
    return sign * (lower + upper) / 2.0


def add_rank_text_in_regions(ax, steps, values_by_candidate, avg_rank, method_name, sign=1.0):
    if len(steps) == 0:
        return

    x_idx = max(int(len(steps) * 0.82), 0)
    x_pos = steps[x_idx]

    for i, c in enumerate(CANDIDATES):
        y_pos = _stack_mid_y(values_by_candidate, i, x_idx, sign=sign)
        txt = f"{method_name}: {CANDIDATE_LABELS[c]}\navg final rank={avg_rank[c]:.2f}"
        ax.text(
            x_pos,
            y_pos,
            txt,
            fontsize=8,
            color="black",
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )


def add_rank_panel(ax, avg_rank, method_name, anchor_x, anchor_y):
    ordered = sorted(avg_rank.items(), key=lambda kv: kv[1])
    lines = [f"{method_name} avg final rank"]
    for c, r in ordered:
        lines.append(f"{CANDIDATE_LABELS[c]}: {r:.2f}")
    ax.text(
        anchor_x,
        anchor_y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
    )


def plot_budget_allocation_single_panel(log_root: Path, output_path: Path, smooth=0.6):
    run_dirs = list_run_dirs(log_root)

    steps_u, alloc_u, runs_u, n_u = aggregate_allocations(run_dirs, "uniform_rounds.jsonl")
    steps_o, alloc_o, runs_o, n_o = aggregate_allocations(run_dirs, "ocba_rounds.jsonl")

    rank_u, rank_runs_u = compute_final_avg_rank(run_dirs, "uniform_rounds.jsonl")
    rank_o, rank_runs_o = compute_final_avg_rank(run_dirs, "ocba_rounds.jsonl")

    # 平滑堆叠边界，提升可读性
    for c in CANDIDATES:
        alloc_u[c] = ema_smooth(alloc_u[c], weight=smooth)
        alloc_o[c] = ema_smooth(alloc_o[c], weight=smooth)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(15, 8.5))

    # 上半区：Uniform
    ax.stackplot(
        steps_u,
        [alloc_u[c] for c in CANDIDATES],
        colors=[CANDIDATE_COLORS[c] for c in CANDIDATES],
        alpha=0.72,
    )

    # 下半区：OCBA（镜像）
    ax.stackplot(
        steps_o,
        [-alloc_o[c] for c in CANDIDATES],
        colors=[CANDIDATE_COLORS[c] for c in CANDIDATES],
        alpha=0.72,
    )

    total_u = np.sum([alloc_u[c] for c in CANDIDATES], axis=0)
    total_o = np.sum([alloc_o[c] for c in CANDIDATES], axis=0)
    ymax = float(max(np.max(total_u), np.max(total_o)))

    ax.axhline(0, color="black", linewidth=1.4, alpha=0.9)
    ax.text(steps_u[0], ymax * 0.97, "Uniform", fontsize=12, fontweight="bold", va="top")
    ax.text(steps_o[0], -ymax * 0.97, "OCBA", fontsize=12, fontweight="bold", va="bottom")

    handles = [
        plt.Line2D([0], [0], color=CANDIDATE_COLORS[c], lw=8, label=CANDIDATE_LABELS[c])
        for c in CANDIDATES
    ]
    ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))

    ax.set_xlabel("Total Training Budget (Steps)")
    ax.set_ylabel("Cumulative Allocated Budget (OCBA mirrored)")
    ax.set_title("MountainCar: Budget Allocation Comparison with Avg Final Rank in Regions")
    ax.grid(alpha=0.22, linestyle="--")
    ax.set_ylim(-ymax * 1.08, ymax * 1.08)

    # 右侧信息面板：更整洁地展示平均最终排名
    add_rank_panel(ax, rank_u, method_name="Uniform", anchor_x=1.01, anchor_y=0.98)
    add_rank_panel(ax, rank_o, method_name="OCBA", anchor_x=1.01, anchor_y=0.46)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
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
        "uniform_rank_runs": rank_runs_u,
        "ocba_rank_runs": rank_runs_o,
        "uniform_avg_final_rank": rank_u,
        "ocba_avg_final_rank": rank_o,
        "smooth": float(smooth),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot single-panel budget allocation comparison (Uniform vs mirrored OCBA) "
            "and annotate average final candidate rank inside regions."
        )
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
        help="Output image path. Default: <log_root>/budget_with_final_rank_single.png",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="EMA smoothing weight for budget boundaries (default: 0.6).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_root = args.log_root if args.log_root.is_absolute() else (PROJECT_ROOT / args.log_root)
    output = args.output if args.output is not None else (log_root / "budget_with_final_rank_single.png")
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    meta = plot_budget_allocation_single_panel(log_root=log_root, output_path=output, smooth=args.smooth)
    print(f"Plot saved to: {meta['output']}")
    print(
        "Aggregation summary: "
        f"smooth={meta['smooth']}, "
        f"uniform_runs={meta['uniform_runs']} (alloc n range {meta['uniform_n_min']}~{meta['uniform_n_max']}, "
        f"final-rank runs={meta['uniform_rank_runs']}), "
        f"ocba_runs={meta['ocba_runs']} (alloc n range {meta['ocba_n_min']}~{meta['ocba_n_max']}, "
        f"final-rank runs={meta['ocba_rank_runs']}), "
        f"uniform_points={meta['uniform_points']}, ocba_points={meta['ocba_points']}"
    )
    print(f"Uniform avg final rank: {meta['uniform_avg_final_rank']}")
    print(f"OCBA avg final rank: {meta['ocba_avg_final_rank']}")


if __name__ == "__main__":
    main()
