"""
Success@delta contour plots (Added Budget × regret threshold) for FrozenLake logs.

Uses the same tricontour styling as ``mountaincar_regret_success_curve.plot_success_contour``:
one PDF/PNG per allocation strategy (uniform, OCBA, Adaptive OCBA).

Default logs: ``src/logs/FrozenLake/run_*``. Default oracle is the uniform-reference best
candidate mean from ``frozenlake_reward_reference.csv`` (safe_distance ~0.5606).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = SRC_DIR.parent

# Align with ``frozenlake_budget_by_factual_rank`` warmup total.
_FL_WARMUP_TOTAL = 1_048_576
_FL_TOTAL_BUDGET = 3_932_160

STRATEGIES = [
    {
        "key": "uniform",
        "filename": "uniform_rounds.jsonl",
        "label": "Uniform Allocation",
        "color": "#B08A72",
        "fill": "#D9CBC0",
        "linestyle": "--",
        "linewidth": 1.4,
    },
    {
        "key": "ocba",
        "filename": "ocba_rounds.jsonl",
        "label": "OCBA Allocation",
        "color": "#6E8FA8",
        "fill": "#CAD6DF",
        "linestyle": "-",
        "linewidth": 1.4,
    },
    {
        "key": "adapted_ocba",
        "filename": "adapted_ocba_rounds.jsonl",
        "label": "Adaptive OCBA Allocation",
        "color": "#4F7089",
        "fill": "#B9CAD6",
        "linestyle": "-.",
        "linewidth": 1.4,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FrozenLake success contour plots from *_rounds.jsonl.")
    p.add_argument(
        "--log-root",
        type=Path,
        default=SRC_DIR / "logs" / "FrozenLake",
        help="Root directory containing run_* folders.",
    )
    p.add_argument(
        "--oracle-mean",
        type=float,
        default=0.5605769230769231,
        help="Oracle best true return V* (default: uniform-ref top candidate mean).",
    )
    p.add_argument(
        "--oracle-std",
        type=float,
        default=0.0349278101672138,
        help="Reference std for token 'std' in delta lists (default: top candidate std).",
    )
    p.add_argument(
        "--contour-output",
        type=Path,
        default=None,
        help="Base output path; writes stem_<strategy>.suffix (default: <log-root>/frozenlake_success_contour.pdf).",
    )
    p.add_argument(
        "--contour-deltas",
        type=str,
        default="0.18,0.15,0.12,0.10,0.08,0.06",
        help="Comma-separated regret thresholds for the contour vertical axis.",
    )
    p.add_argument(
        "--min-budget",
        type=int,
        default=_FL_WARMUP_TOTAL,
        help="Minimum budget_consumed included.",
    )
    p.add_argument(
        "--max-budget",
        type=int,
        default=_FL_TOTAL_BUDGET,
        help="Maximum budget_consumed included.",
    )
    p.add_argument("--smooth", type=float, default=0.6, help="EMA smoothing on success curves.")
    p.add_argument(
        "--trim-extremes-count",
        type=int,
        default=0,
        help="Trim lowest/highest regrets per step before aggregation.",
    )
    p.add_argument(
        "--warmup-budget",
        type=int,
        default=_FL_WARMUP_TOTAL,
        help="Warmup subtracted for x-axis (added budget).",
    )
    return p.parse_args()


def list_run_dirs(log_root: Path) -> list[Path]:
    run_dirs = sorted([p for p in log_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {log_root}")
    return run_dirs


def read_round_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _safe_float(value, default: float = -1e9) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _best_estimated_reward(per_candidate: dict) -> float:
    if not per_candidate:
        return -1e9
    best = -1e18
    for _, info in per_candidate.items():
        val = _safe_float((info or {}).get("true_mean_return"), -1e9)
        if val > best:
            best = val
    return float(best)


def _parse_deltas(delta_text: str, oracle_std: float) -> list[float]:
    parts = [x.strip().lower() for x in delta_text.split(",") if x.strip()]
    out = []
    for p in parts:
        if p == "std":
            out.append(float(oracle_std))
        else:
            out.append(float(p))
    unique = sorted(set([float(x) for x in out]))
    if not unique:
        raise ValueError("No valid delta values.")
    return unique


def _mean_ci95(vals: np.ndarray) -> tuple[float, float, float]:
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(vals))
    if vals.size == 1:
        return m, m, m
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    half = 1.96 * sem
    return m, m - half, m + half


def _trim_extremes(vals: np.ndarray, trim_count: int) -> np.ndarray:
    if vals.size == 0 or trim_count <= 0:
        return vals
    if vals.size <= 2 * trim_count:
        return vals
    sorted_vals = np.sort(vals)
    return sorted_vals[trim_count:-trim_count]


def ema_smooth(values: np.ndarray, weight: float = 0.6) -> np.ndarray:
    if len(values) == 0:
        return values
    smoothed = np.zeros_like(values, dtype=np.float64)
    smoothed[0] = float(values[0])
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1.0 - weight) * float(values[i])
    return smoothed


def collect_step_regrets(
    run_dirs: list[Path],
    oracle_best: float,
    min_budget: int | None,
    max_budget: int | None,
) -> tuple[dict[str, dict[int, list[float]]], dict[str, int]]:
    strategy_step_regrets = {s["key"]: {} for s in STRATEGIES}
    available_runs = {s["key"]: 0 for s in STRATEGIES}

    for run_dir in run_dirs:
        for s in STRATEGIES:
            key = s["key"]
            path = run_dir / s["filename"]
            if not path.exists():
                continue
            records = read_round_records(path)
            if not records:
                continue
            available_runs[key] += 1
            for rec in records:
                step = int(rec.get("budget_consumed", -1))
                if step < 0:
                    continue
                if min_budget is not None and step < int(min_budget):
                    continue
                if max_budget is not None and step > int(max_budget):
                    continue
                best_est = _best_estimated_reward(rec.get("per_candidate", {}) or {})
                regret = float(oracle_best - best_est)
                strategy_step_regrets[key].setdefault(step, []).append(regret)
    return strategy_step_regrets, available_runs


def aggregate_success_curve(
    step_regrets: dict[int, list[float]],
    delta: float,
    trim_count: int = 0,
    warmup_budget: int = _FL_WARMUP_TOTAL,
):
    x = []
    mean = []
    low = []
    high = []
    n = []
    for step in sorted(step_regrets.keys()):
        added_budget = int(step) - int(warmup_budget)
        if added_budget < 0:
            continue
        regrets = np.array(step_regrets[step], dtype=np.float64)
        regrets = _trim_extremes(regrets, trim_count=trim_count)
        hits = (regrets <= float(delta)).astype(np.float64)
        m, lo, hi = _mean_ci95(hits)
        x.append(int(added_budget))
        mean.append(float(m))
        low.append(max(0.0, float(lo)))
        high.append(min(1.0, float(hi)))
        n.append(int(hits.size))
    return np.array(x), np.array(mean), np.array(low), np.array(high), np.array(n)


def _setup_matplotlib():
    import matplotlib.pyplot as plt

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
    return plt


def plot_success_contour(contour_items: list[dict], output_path: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    plt = _setup_matplotlib()
    morandi_cmap = LinearSegmentedColormap.from_list(
        "morandi_success",
        ["#6F6259", "#A79686", "#CBBFAF", "#EAE3D8", "#FCFBF8"],
        N=256,
    )
    all_x = np.array([float(r["added_budget"]) for r in contour_items], dtype=np.float64)
    all_y = np.array([float(r["delta"]) for r in contour_items], dtype=np.float64)
    finite_xy = np.isfinite(all_x) & np.isfinite(all_y)
    if np.any(finite_xy):
        x_min, x_max = float(np.min(all_x[finite_xy])), float(np.max(all_x[finite_xy]))
        y_min, y_max = float(np.min(all_y[finite_xy])), float(np.max(all_y[finite_xy]))
    else:
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0
    levels = np.linspace(0.0, 1.0, 11)
    iso_levels = [0.2, 0.4, 0.6, 0.8, 0.9]
    for s in STRATEGIES:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        rows = [r for r in contour_items if r["strategy"] == s["key"]]
        if not rows:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            x = np.array([float(r["added_budget"]) for r in rows], dtype=np.float64)
            y = np.array([float(r["delta"]) for r in rows], dtype=np.float64)
            z = np.array([float(r["success_prob"]) for r in rows], dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[finite], y[finite], z[finite]
            if x.size < 3:
                ax.text(0.5, 0.5, "Insufficient points", transform=ax.transAxes, ha="center", va="center")
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            else:
                ax.tricontourf(x, y, z, levels=levels, cmap=morandi_cmap, vmin=0.0, vmax=1.0)
                line_ct = ax.tricontour(x, y, z, levels=iso_levels, colors="black", linewidths=0.6, alpha=0.72)
                ax.clabel(line_ct, inline=True, fmt="%0.1f", fontsize=6)
                for txt in line_ct.labelTexts:
                    txt.set_fontweight("bold")
                    txt.set_fontsize(7)
                    txt.set_color("black")
                ax.text(
                    0.98,
                    0.98,
                    "Levels: 0.2/0.4/0.6/0.8/0.9/0.95",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.5,
                    fontweight="bold",
                    color="black",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "#F2EFE8", "ec": "none", "alpha": 0.55},
                )
        ax.set_xlabel("Added Budget", fontweight="bold")
        ax.set_ylabel("Regret (V* - best return)", fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.12, linestyle="-", linewidth=0.4)
        ax.tick_params(direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        single_path = output_path.with_name(f"{output_path.stem}_{s['key']}{output_path.suffix}")
        plt.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    log_root = args.log_root if args.log_root.is_absolute() else (PROJECT_ROOT / args.log_root)
    contour_output = (
        args.contour_output
        if args.contour_output is not None
        else log_root / "frozenlake_success_contour.pdf"
    )
    if not contour_output.is_absolute():
        contour_output = PROJECT_ROOT / contour_output

    contour_deltas = _parse_deltas(args.contour_deltas, oracle_std=args.oracle_std)
    run_dirs = list_run_dirs(log_root)
    step_regrets, available_runs = collect_step_regrets(
        run_dirs=run_dirs,
        oracle_best=float(args.oracle_mean),
        min_budget=args.min_budget,
        max_budget=args.max_budget,
    )

    contour_items = []
    for s in STRATEGIES:
        key = s["key"]
        for d in contour_deltas:
            x2, m2, _lo2, _hi2, n2 = aggregate_success_curve(
                step_regrets[key],
                delta=d,
                trim_count=int(args.trim_extremes_count),
                warmup_budget=int(args.warmup_budget),
            )
            if len(x2) == 0:
                continue
            m2_s = ema_smooth(m2, weight=float(args.smooth))
            n_runs = int(np.max(n2)) if len(n2) > 0 else 0
            for xb, pb in zip(x2, m2_s):
                contour_items.append(
                    {
                        "strategy": key,
                        "delta": float(d),
                        "added_budget": int(xb),
                        "success_prob": float(pb),
                        "n_runs": n_runs,
                    }
                )

    if not contour_items:
        raise RuntimeError("No contour points produced (missing jsonl or empty regret grids).")

    plot_success_contour(contour_items, contour_output)

    summary = []
    for s in STRATEGIES:
        key = s["key"]
        if available_runs.get(key, 0) > 0:
            summary.append(f"{key}: runs={available_runs[key]}")

    print(
        "Contour plots saved:",
        contour_output.with_name(f"{contour_output.stem}_uniform{contour_output.suffix}"),
        contour_output.with_name(f"{contour_output.stem}_ocba{contour_output.suffix}"),
        contour_output.with_name(f"{contour_output.stem}_adapted_ocba{contour_output.suffix}"),
    )
    print("Run coverage:", "; ".join(summary))


if __name__ == "__main__":
    main()
