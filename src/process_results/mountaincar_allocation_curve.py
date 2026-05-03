import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
CANDIDATES = ["baseline", "assist_pos_vel", "assist_energy_gate", "deceptive_left"]
CANDIDATE_COLORS = {
    "baseline": "tab:blue",
    "assist_pos_vel": "tab:orange",
    "assist_energy_gate": "tab:green",
    "deceptive_left": "tab:red",
}
STRATEGIES = [
    {
        "key": "uniform",
        "filename": "uniform_rounds.jsonl",
        "line_label": "Uniform Allocation",
        "ci_label": "Uniform 95% CI",
        "color": "#B08A72",
        "fill": "#D9CBC0",
        "linestyle": "--",
        "linewidth": 1.4,
    },
    {
        "key": "ocba",
        "filename": "ocba_rounds.jsonl",
        "line_label": "OCBA Allocation",
        "ci_label": "OCBA 95% CI",
        "color": "#6E8FA8",
        "fill": "#CAD6DF",
        "linestyle": "-",
        "linewidth": 1.4,
    },
    {
        "key": "adapted_ocba",
        "filename": "adapted_ocba_rounds.jsonl",
        "line_label": "Adaptive OCBA Allocation",
        "ci_label": "Adaptive OCBA 95% CI",
        "color": "#4F7089",
        "fill": "#B9CAD6",
        "linestyle": "-.",
        "linewidth": 1.4,
    },
]


def read_round_curve(jsonl_path: Path):
    steps = []
    best_returns = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            steps.append(int(rec["budget_consumed"]))
            best_returns.append(float(rec["best_true_return"]))
    return steps, best_returns


def list_run_dirs(log_root: Path):
    run_dirs = [p for p in log_root.glob("run_*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under: {log_root.resolve()} "
            f"(cwd={Path.cwd()})"
        )
    return sorted(run_dirs)


def load_strategy_curves(run_dirs, strategy_filename):
    curves = []
    for run_dir in run_dirs:
        path = run_dir / strategy_filename
        if not path.exists():
            continue
        steps, best_returns = read_round_curve(path)
        if steps and best_returns:
            curves.append((steps, best_returns, run_dir.name))
    if not curves:
        raise FileNotFoundError(f"No valid {strategy_filename} found in provided run directories.")
    return curves


def read_round_records(jsonl_path: Path):
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def aggregate_glyph_data(run_dirs, strategy_filename, max_budget=None):
    # step -> list[alloc_vector], list[rank_vector]
    step_allocs = {}
    step_ranks = {}

    for run_dir in run_dirs:
        path = run_dir / strategy_filename
        if not path.exists():
            continue
        records = read_round_records(path)
        for rec in records:
            s = int(rec["budget_consumed"])
            if max_budget is not None and s > int(max_budget):
                continue
            per_candidate = rec.get("per_candidate", {})

            alloc_vec = np.array(
                [float(per_candidate.get(c, {}).get("allocation_cumulative", 0.0)) for c in CANDIDATES],
                dtype=np.float64,
            )
            ret_vec = np.array(
                [float(per_candidate.get(c, {}).get("true_mean_return", -1e9)) for c in CANDIDATES],
                dtype=np.float64,
            )
            # rank: 1 is best
            order = np.argsort(-ret_vec)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(CANDIDATES) + 1)

            step_allocs.setdefault(s, []).append(alloc_vec)
            step_ranks.setdefault(s, []).append(ranks.astype(np.float64))

    steps = sorted(set(step_allocs.keys()) & set(step_ranks.keys()))
    glyph = {}
    for s in steps:
        alloc_arr = np.array(step_allocs[s], dtype=np.float64)  # [n_run, K]
        rank_arr = np.array(step_ranks[s], dtype=np.float64)    # [n_run, K]
        mean_alloc = np.mean(alloc_arr, axis=0)
        total = float(np.sum(mean_alloc))
        if total <= 0:
            alloc_share = np.full(len(CANDIDATES), 1.0 / len(CANDIDATES))
        else:
            alloc_share = mean_alloc / total
        mean_rank = np.mean(rank_arr, axis=0)
        glyph[s] = {
            "alloc_share": alloc_share,
            "mean_rank": mean_rank,
        }
    return glyph


def _trim_extremes(vals: np.ndarray, trim_count: int) -> np.ndarray:
    if vals.size == 0 or trim_count <= 0:
        return vals
    if vals.size <= 2 * trim_count:
        return vals
    vals_sorted = np.sort(vals)
    return vals_sorted[trim_count:-trim_count]


def aggregate_curves(curves, max_budget=None, min_budget=None, trim_count: int = 0, warmup_budget: int = 800_000):
    # 按 budget(steps) 对齐，支持不同 run 的长度不一致
    all_steps = sorted({s for steps, _, _ in curves for s in steps})
    if min_budget is not None:
        all_steps = [s for s in all_steps if s >= int(min_budget)]
    if max_budget is not None:
        all_steps = [s for s in all_steps if s <= int(max_budget)]
    step_to_values = {s: [] for s in all_steps}
    for steps, values, _ in curves:
        for s, v in zip(steps, values):
            if s in step_to_values:
                step_to_values[s].append(float(v))

    x = []
    mean = []
    low = []
    high = []
    n = []
    for s in all_steps:
        vals = np.array(step_to_values[s], dtype=np.float64)
        vals = _trim_extremes(vals, trim_count=trim_count)
        if vals.size == 0:
            continue
        m = float(np.mean(vals))
        if vals.size > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
            ci = 1.96 * sem
        else:
            ci = 0.0
        x.append(int(s) - int(warmup_budget))
        mean.append(m)
        low.append(m - ci)
        high.append(m + ci)
        n.append(int(vals.size))
    return np.array(x), np.array(mean), np.array(low), np.array(high), np.array(n)


def ema_smooth(values, weight=0.6):
    if len(values) == 0:
        return values
    smoothed = np.zeros_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1.0 - weight) * values[i]
    return smoothed


def draw_point_glyph_bars(ax, x, y, glyph_data, glyph_total_height, bar_width):
    for xi, yi in zip(x, y):
        s = int(xi)
        if s not in glyph_data:
            continue
        alloc_share = glyph_data[s]["alloc_share"]
        mean_rank = glyph_data[s]["mean_rank"]

        # better rank appears on top of mini bar
        order = np.argsort(mean_rank)[::-1]  # worst -> best (best drawn last/top)
        bottom = yi - glyph_total_height / 2.0
        for idx in order:
            h = float(alloc_share[idx]) * glyph_total_height
            if h <= 0:
                continue
            c = CANDIDATES[idx]
            ax.bar(
                xi,
                h,
                width=bar_width,
                bottom=bottom,
                color=CANDIDATE_COLORS[c],
                alpha=0.75,
                align="center",
                linewidth=0.0,
                zorder=4,
            )
            bottom += h


def build_aggregate_plot(
    run_dirs,
    output_path: Path,
    smooth_weight=0.6,
    min_budget=1_500_000,
    max_budget=3_000_000,
    trim_extremes_count: int = 0,
    warmup_budget: int = 800_000,
):
    plotted = []
    for strategy in STRATEGIES:
        try:
            curves = load_strategy_curves(run_dirs, strategy["filename"])
        except FileNotFoundError:
            continue
        glyph = aggregate_glyph_data(run_dirs, strategy["filename"], max_budget=max_budget)
        x, mean, low, high, n = aggregate_curves(
            curves,
            max_budget=max_budget,
            min_budget=min_budget,
            trim_count=trim_extremes_count,
            warmup_budget=warmup_budget,
        )
        plotted.append(
            {
                "strategy": strategy,
                "curves": curves,
                "glyph": glyph,
                "x": x,
                "mean_s": ema_smooth(mean, weight=smooth_weight),
                "low_s": ema_smooth(low, weight=smooth_weight),
                "high_s": ema_smooth(high, weight=smooth_weight),
                "n": n,
            }
        )
    if not plotted:
        expected = ", ".join([s["filename"] for s in STRATEGIES])
        raise FileNotFoundError(f"No strategy jsonl files found. Expected one of: {expected}")

    # Nature-like minimalist plotting style.
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )
    plt.figure(figsize=(3.5, 2.6))
    for item in plotted:
        s = item["strategy"]
        plt.plot(
            item["x"],
            item["mean_s"],
            label=s["line_label"],
            linestyle=s["linestyle"],
            linewidth=s["linewidth"],
            color=s["color"],
        )
        plt.fill_between(
            item["x"],
            item["low_s"],
            item["high_s"],
            color=s["fill"],
            alpha=0.65,
            label="_nolegend_",
        )

    plt.xlabel("Added Budget (Steps)", fontweight="bold")
    plt.ylabel("Best Episodic Return", fontweight="bold")
    ax = plt.gca()
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=-160.0)
    plt.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    strategy_meta = {}
    for item in plotted:
        key = item["strategy"]["key"]
        n_arr = item["n"]
        strategy_meta[key] = {
            "runs": int(len(item["curves"])),
            "points": int(len(item["x"])),
            "min_n": int(np.min(n_arr)) if len(n_arr) > 0 else 0,
            "max_n": int(np.max(n_arr)) if len(n_arr) > 0 else 0,
        }
    return {
        "max_budget": int(max_budget),
        "smooth_weight": float(smooth_weight),
        "strategy_meta": strategy_meta,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MountainCar allocation best-return curves from logged jsonl files."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=SRC_DIR / "logs" / "MountainCar_Adaptive",
        help="Root directory containing run_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <log_root>/allocation_best_return_ci.pdf",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="EMA smoothing weight, TensorBoard-style (default: 0.6).",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=800_000,
        help="Minimum total budget(step) shown on learning curve (default: 1,500,000).",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=3_000_000,
        help="Maximum budget(step) shown on learning curve (default: 3,000,000).",
    )
    parser.add_argument(
        "--trim-extremes-count",
        type=int,
        default=0,
        help="Remove lowest N and highest N returns per step before aggregation (default: 0).",
    )
    parser.add_argument(
        "--warmup-budget",
        type=int,
        default=800_000,
        help="Warmup budget to subtract from total budget for x-axis (default: 800,000).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_root = args.log_root if args.log_root.is_absolute() else (PROJECT_ROOT / args.log_root)
    run_dirs = list_run_dirs(log_root)
    output = args.output if args.output is not None else log_root / "allocation_best_return_ci.pdf"
    if output is not None and not output.is_absolute():
        output = PROJECT_ROOT / output

    meta = build_aggregate_plot(
        run_dirs=run_dirs,
        output_path=output,
        smooth_weight=args.smooth,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        trim_extremes_count=args.trim_extremes_count,
        warmup_budget=args.warmup_budget,
    )
    print(f"Plot saved to: {output}")
    strategy_parts = []
    for strategy in STRATEGIES:
        key = strategy["key"]
        if key not in meta["strategy_meta"]:
            continue
        s_meta = meta["strategy_meta"][key]
        strategy_parts.append(
            f"{key}: runs={s_meta['runs']}, points={s_meta['points']} (n range {s_meta['min_n']}~{s_meta['max_n']})"
        )
    print(
        "Aggregation summary: "
        f"smooth={meta['smooth_weight']}, min_budget={args.min_budget}, max_budget={meta['max_budget']}, "
        f"trim_extremes_count={args.trim_extremes_count}, warmup_budget={args.warmup_budget}; "
        + "; ".join(strategy_parts)
    )


if __name__ == "__main__":
    main()
