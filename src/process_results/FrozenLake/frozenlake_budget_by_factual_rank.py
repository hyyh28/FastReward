"""
FrozenLake allocation analysis from ``*_rounds.jsonl`` logs.

1) **Reward-shaping figure** (default PDF): four candidates ordered by uniform-run mean
   final true return; top = budget fraction for **OCBA + Adaptive OCBA only**; bottom =
   compact **horizontal** bars: uniform final true return ± σ (≈1/3 figure height).

2) **Best-true learning curve** (default PDF): same aggregation + EMA + line/band styling as
   ``mountaincar_allocation_curve.py`` (Added Budget vs best episodic return).

3) **Rank-at-budget figure** (default PDF): at fixed added budgets, mean snapshot rank of
   each reward (by current true_mean_return); legend marks **ground-truth rank** from uniform
   reference ordering.

4) **Rank-bucket figures** (default PDFs): budget fraction by factual rank; Δ vs uniform;
   Top-1 hit budget.

Outputs default under ``src/logs/FrozenLake/``. Run with no arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent

DEFAULT_LOG_ROOT = SRC_DIR / "logs" / "FrozenLake"
DEFAULT_OUTPUT_DIR = SRC_DIR / "logs" / "FrozenLake"

_FL_K = 4
_FL_WARMUP_PER_CANDIDATE = 262_144
_FL_WARMUP_TOTAL = _FL_K * _FL_WARMUP_PER_CANDIDATE  # 1_048_576
_FL_TOTAL_BUDGET = 3_932_160

STRATEGIES = [
    {
        "key": "uniform",
        "filename": "uniform_rounds.jsonl",
        "label": "Uniform Allocation",
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
        "label": "OCBA Allocation",
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
        "label": "Adaptive OCBA Allocation",
        "line_label": "Adaptive OCBA Allocation",
        "ci_label": "Adaptive OCBA 95% CI",
        "color": "#4F7089",
        "fill": "#B9CAD6",
        "linestyle": "-.",
        "linewidth": 1.4,
    },
]

# Only OCBA variants on the top panel of the four-reward allocation PDF.
STRATEGIES_OCBA_TOP = [s for s in STRATEGIES if s["key"] in ("ocba", "adapted_ocba")]

# Distinct colors for the four shaped rewards (aligned with ``candidates_ordered`` indexing).
_REWARD_BAR_COLORS = ["#6E8FA8", "#C4876E", "#8F85A3", "#6E9B78"]


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


def read_round_curve(jsonl_path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    best_returns: list[float] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            steps.append(int(rec["budget_consumed"]))
            best_returns.append(float(rec["best_true_return"]))
    return steps, best_returns


def load_strategy_curves(run_dirs: list[Path], strategy_filename: str) -> list[tuple[list[int], list[float], str]]:
    curves: list[tuple[list[int], list[float], str]] = []
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


def _trim_extremes_mc(vals: np.ndarray, trim_count: int) -> np.ndarray:
    """Same trimming rule as ``mountaincar_allocation_curve.aggregate_curves``."""
    if vals.size == 0 or trim_count <= 0:
        return vals
    if vals.size <= 2 * trim_count:
        return vals
    vals_sorted = np.sort(vals)
    return vals_sorted[trim_count:-trim_count]


def aggregate_curves_mc(
    curves: list[tuple[list[int], list[float], str]],
    max_budget: int | None = None,
    min_budget: int | None = None,
    trim_count: int = 0,
    warmup_budget: int = 800_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_steps = sorted({s for steps, _, _ in curves for s in steps})
    if min_budget is not None:
        all_steps = [s for s in all_steps if s >= int(min_budget)]
    if max_budget is not None:
        all_steps = [s for s in all_steps if s <= int(max_budget)]
    step_to_values: dict[int, list[float]] = {s: [] for s in all_steps}
    for steps, values, _ in curves:
        for s, v in zip(steps, values):
            if s in step_to_values:
                step_to_values[s].append(float(v))

    x: list[int] = []
    mean: list[float] = []
    low: list[float] = []
    high: list[float] = []
    n: list[int] = []
    for s in all_steps:
        vals = np.array(step_to_values[s], dtype=np.float64)
        vals = _trim_extremes_mc(vals, trim_count=trim_count)
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


def ema_smooth(values: np.ndarray, weight: float = 0.6) -> np.ndarray:
    if len(values) == 0:
        return values
    smoothed = np.zeros_like(values, dtype=np.float64)
    smoothed[0] = float(values[0])
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1.0 - weight) * float(values[i])
    return smoothed


def _safe_float(value, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pretty_name(candidate: str) -> str:
    return candidate.replace("_", " ")


def factual_rank_by_candidate(per_candidate: dict) -> dict[str, int]:
    """Rank 1 = best true_mean_return on this trajectory; ties broken by candidate name."""
    if not per_candidate:
        return {}
    scored = []
    for c, info in per_candidate.items():
        m = _safe_float((info or {}).get("true_mean_return"), -1e9)
        scored.append((c, m))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return {c: r + 1 for r, (c, _) in enumerate(scored)}


def snapshot_ranks(per_candidate: dict) -> dict[str, int]:
    """Same ordering rule as ``factual_rank_by_candidate`` (snapshot true_mean_return)."""
    return factual_rank_by_candidate(per_candidate)


def record_at_added_budget(records: list[dict], warmup_budget: int, added_target: int) -> dict | None:
    want = int(warmup_budget) + int(added_target)
    for rec in records:
        if int(rec.get("budget_consumed", -1)) == want:
            return rec
    return None


def _parse_added_budget_checkpoints(text: str) -> list[int]:
    vals = []
    for part in [x.strip() for x in str(text).split(",") if x.strip()]:
        vals.append(int(float(part)))
    uniq = sorted(set(vals))
    if not uniq:
        raise ValueError("No valid added-budget checkpoints.")
    return uniq


def _candidate_color_map(candidates_ordered: list[str]) -> dict[str, str]:
    return {
        c: _REWARD_BAR_COLORS[i % len(_REWARD_BAR_COLORS)] for i, c in enumerate(candidates_ordered)
    }


def fraction_by_factual_rank(final_rec: dict) -> tuple[dict[int, float], float]:
    """Per-rank budget fraction using each candidate's final-rank on this trajectory."""
    per = final_rec.get("per_candidate") or {}
    ranks = factual_rank_by_candidate(per)
    total_b = float(final_rec.get("budget_consumed", 0))
    by_rank: dict[int, float] = {}
    for c, info in per.items():
        alloc = float((info or {}).get("allocation_cumulative", 0))
        r = ranks.get(c, -1)
        if r < 1:
            continue
        by_rank[r] = by_rank.get(r, 0.0) + alloc
    frac = {r: (by_rank[r] / total_b if total_b > 0 else np.nan) for r in by_rank}
    return frac, total_b


def _fix_pick_leader(per_candidate: dict) -> str | None:
    if not per_candidate:
        return None
    scored = [(c, _safe_float((info or {}).get("true_mean_return"), -1e9)) for c, info in per_candidate.items()]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[0][0]


def first_hit_top1_budget(records: list[dict], final_leader: str | None) -> float | None:
    if final_leader is None:
        return None
    for rec in records:
        per = rec.get("per_candidate") or {}
        if _fix_pick_leader(per) == final_leader:
            return float(rec["budget_consumed"])
    return None


def _mean_ci95(vals: np.ndarray) -> tuple[float, float, float]:
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(vals))
    if vals.size == 1:
        return m, m, m
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    half = 1.96 * sem
    return m, m - half, m + half


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Allocation by reward shaping vs uniform reference stats (FrozenLake jsonl)."
    )
    p.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PDF + CSV outputs.")
    p.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Combined PDF path (default: <output-dir>/frozenlake_reward_allocation.pdf).",
    )
    p.add_argument("--reference-csv", type=Path, default=None)
    p.add_argument("--allocation-csv", type=Path, default=None)
    p.add_argument("--detail-csv", type=Path, default=None)
    p.add_argument("--no-detail-csv", action="store_true")
    p.add_argument("--no-figure", action="store_true", help="Skip the reward-shaping combined PDF only.")
    p.add_argument(
        "--rank-share-pdf",
        type=Path,
        default=None,
        help="Default: <output-dir>/frozenlake_factual_rank_share.pdf",
    )
    p.add_argument(
        "--rank-delta-pdf",
        type=Path,
        default=None,
        help="Default: <output-dir>/frozenlake_factual_rank_delta.pdf",
    )
    p.add_argument(
        "--rank-identify-pdf",
        type=Path,
        default=None,
        help="Default: <output-dir>/frozenlake_factual_rank_identify.pdf",
    )
    p.add_argument("--rank-share-csv", type=Path, default=None)
    p.add_argument("--rank-delta-csv", type=Path, default=None)
    p.add_argument("--rank-identify-csv", type=Path, default=None)
    p.add_argument("--warmup-budget", type=int, default=_FL_WARMUP_TOTAL)
    p.add_argument("--no-rank-figures", action="store_true", help="Skip rank-bucket + delta + identify PDFs/CSVs.")
    p.add_argument("--no-identify", action="store_true", help="Skip Top-1 identification outputs only.")
    p.add_argument(
        "--learning-curve-pdf",
        type=Path,
        default=None,
        help="Default: <output-dir>/frozenlake_best_true_learning_curve.pdf",
    )
    p.add_argument("--no-learning-curve", action="store_true", help="Skip best_true_return vs added-budget PDF.")
    p.add_argument(
        "--rank-budget-pdf",
        type=Path,
        default=None,
        help="Default: <output-dir>/frozenlake_rank_at_added_budget.pdf",
    )
    p.add_argument("--no-rank-budget-figure", action="store_true", help="Skip rank-at-checkpoints PDF.")
    p.add_argument(
        "--rank-checkpoints",
        type=str,
        default="262144,786432,1310720,1835008,2359296",
        help="Comma-separated post-warmup added budgets for rank snapshot figure.",
    )
    p.add_argument(
        "--learning-smooth",
        type=float,
        default=0.6,
        help="EMA smoothing for learning curve (same as mountaincar_allocation_curve).",
    )
    p.add_argument(
        "--learning-min-budget",
        type=int,
        default=_FL_WARMUP_TOTAL,
        help="Minimum total budget_consumed included on learning curve (default: warmup total).",
    )
    p.add_argument(
        "--learning-max-budget",
        type=int,
        default=_FL_TOTAL_BUDGET,
        help="Maximum total budget_consumed on learning curve.",
    )
    p.add_argument(
        "--learning-trim-extremes",
        type=int,
        default=0,
        help="Trim lowest/highest returns per step before CI (mountaincar_allocation_curve).",
    )
    return p.parse_args()


def plot_combined_figure(
    candidates_ordered: list[str],
    mean_true: dict[str, float],
    std_true: dict[str, float],
    var_true: dict[str, float],
    mean_frac: dict[str, list[float]],
    err_lo: dict[str, list[float]],
    err_hi: dict[str, list[float]],
    output_path: Path,
    top_strategies: list[dict] | None = None,
) -> None:
    """Two-row PDF: (top) budget fraction for OCBA variants only; (bottom) horizontal uniform-reference bars."""
    plt = _setup_matplotlib()
    import matplotlib.gridspec as gridspec

    top_strategies = STRATEGIES_OCBA_TOP if top_strategies is None else top_strategies

    fig = plt.figure(figsize=(5.4, 4.35))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.0, 1.0], hspace=0.48)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    x = np.arange(len(candidates_ordered), dtype=np.float64)
    n_s = len(top_strategies)
    # Bars must stay inside each category band [i - 0.5, i + 0.5] so xticks stay centered.
    # With two series: bar centers at i ± half_sep, need half_sep + bar_w/2 <= 0.5 (minus tiny margin).
    _edge_slack = 0.02
    if n_s == 2:
        bar_w = 0.28
        half_sep = min(0.19, 0.5 - bar_w / 2.0 - _edge_slack)
        offsets = np.array([-half_sep, half_sep], dtype=np.float64)
    elif n_s == 1:
        bar_w = 0.42
        offsets = np.array([0.0], dtype=np.float64)
    else:
        cluster_half_w = 0.42
        span = 2.0 * cluster_half_w
        bar_w = min(0.34, (1.0 - _edge_slack * 2) / max(float(n_s) + 0.5, 1.0))
        offsets = np.linspace(-cluster_half_w, cluster_half_w, num=n_s) if n_s else np.array([])

    for s_idx, s in enumerate(top_strategies):
        key = s["key"]
        ys = np.array([mean_frac[key][i] for i in range(len(candidates_ordered))], dtype=np.float64)
        lo = np.array([err_lo[key][i] for i in range(len(candidates_ordered))], dtype=np.float64)
        hi = np.array([err_hi[key][i] for i in range(len(candidates_ordered))], dtype=np.float64)
        yerr = np.vstack([np.maximum(0, ys - lo), np.maximum(0, hi - ys)])
        ax0.bar(
            x + offsets[s_idx],
            ys,
            width=bar_w * 0.96,
            color=s["color"],
            alpha=0.88,
            edgecolor=s["color"],
            linewidth=0.55,
            label=s["label"],
            yerr=yerr,
            capsize=2.0,
            error_kw={"linewidth": 0.55, "color": "#333333"},
        )

    ax0.set_ylabel("Fraction of total budget", fontweight="bold")
    ax0.set_ylim(0.0, 0.5)
    n_cat = len(candidates_ordered)
    ax0.set_xlim(-0.5, n_cat - 0.5)
    ax0.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax0.tick_params(direction="out", labelbottom=True, bottom=True)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(loc="upper right", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    ax0.set_xticks(np.arange(n_cat, dtype=np.float64))
    ax0.set_xticklabels([_pretty_name(c) for c in candidates_ordered])
    for tick in ax0.get_xticklabels():
        tick.set_fontsize(7)
        tick.set_rotation(22)
        tick.set_ha("right")
    ax0.set_xlabel("Reward shaping candidate", fontweight="bold")

    means = np.array([mean_true[c] for c in candidates_ordered], dtype=np.float64)
    stds = np.array([std_true[c] for c in candidates_ordered], dtype=np.float64)
    bar_gray = "#8B8680"
    y_pos = np.arange(len(candidates_ordered), dtype=np.float64)
    bar_h = 0.26
    ax1.barh(
        y_pos,
        means,
        height=bar_h,
        color="#E8E4DF",
        edgecolor=bar_gray,
        linewidth=0.55,
        xerr=stds,
        capsize=2.0,
        error_kw={"linewidth": 0.55, "color": "#333333"},
    )
    ax1.set_xlabel("Final true return (uniform runs)", fontweight="bold")
    ax1.set_yticks(y_pos, [_pretty_name(c) for c in candidates_ordered])
    ax1.set_ylim(y_pos[0] - 0.5, y_pos[-1] + 0.5)
    for lbl in ax1.get_yticklabels():
        lbl.set_fontweight("bold")
    ax1.tick_params(direction="out", labelsize=7)
    ax1.grid(axis="x", alpha=0.18, linestyle="-", linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    if means.size:
        span = float(np.nanmax(means + np.nan_to_num(stds)))
    else:
        span = 1.0
    ax1.set_xlim(left=0.0, right=max(span * 1.2, 0.05))
    for yi, c in enumerate(candidates_ordered):
        mu = mean_true[c]
        v = var_true[c]
        xi = float(means[yi] + (stds[yi] if np.isfinite(stds[yi]) else 0.0))
        ax1.text(
            xi + max(span * 0.02, 0.01),
            yi,
            f"μ={mu:.3f}, Var={v:.5f}",
            va="center",
            ha="left",
            fontsize=6,
            clip_on=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_best_true_learning_curve(
    run_dirs: list[Path],
    strategies: list[dict],
    output_path: Path,
    smooth_weight: float = 0.6,
    min_budget: int | None = None,
    max_budget: int | None = None,
    trim_extremes_count: int = 0,
    warmup_budget: int = _FL_WARMUP_TOTAL,
) -> None:
    """Match ``mountaincar_allocation_curve.build_aggregate_plot`` line + band styling."""
    plotted: list[dict] = []
    for strategy in strategies:
        try:
            curves = load_strategy_curves(run_dirs, strategy["filename"])
        except FileNotFoundError:
            continue
        x, mean, low, high, _n = aggregate_curves_mc(
            curves,
            max_budget=max_budget,
            min_budget=min_budget,
            trim_count=trim_extremes_count,
            warmup_budget=warmup_budget,
        )
        if x.size == 0:
            continue
        plotted.append(
            {
                "strategy": strategy,
                "x": x,
                "mean_s": ema_smooth(mean, weight=smooth_weight),
                "low_s": ema_smooth(low, weight=smooth_weight),
                "high_s": ema_smooth(high, weight=smooth_weight),
            }
        )
    if not plotted:
        return

    plt = _setup_matplotlib()
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
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_rank_at_added_budgets(
    strategies: list[dict],
    candidates_ordered: list[str],
    truth_ranks: dict[str, int],
    rank_samples: dict[str, dict[int, dict[str, list[float]]]],
    checkpoints: list[int],
    output_path: Path,
) -> None:
    """Grouped bars: mean snapshot rank (1 = best at that budget) per reward × checkpoint; truth rank in legend."""
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, len(strategies), figsize=(4.2 * len(strategies), 2.85), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    cmap = _candidate_color_map(candidates_ordered)
    n_c = len(candidates_ordered)
    usable = []
    for b in checkpoints:
        if b < 0:
            continue
        has = any(
            len((rank_samples.get(s["key"]) or {}).get(b, {}).get(c, [])) > 0
            for s in strategies
            for c in candidates_ordered
        )
        if has:
            usable.append(b)
    if not usable:
        return

    x_grp = np.arange(len(usable), dtype=np.float64)
    inner_w = 0.78 / max(n_c, 1)

    for ax_idx, (ax, s) in enumerate(zip(axes, strategies)):
        ax.set_facecolor("white")
        key = s["key"]
        legend_last = ax_idx == len(strategies) - 1
        for ci, c in enumerate(candidates_ordered):
            offs = (ci - (n_c - 1) / 2.0) * inner_w
            heights = []
            for b in usable:
                vals = np.array((rank_samples.get(key) or {}).get(b, {}).get(c, []), dtype=np.float64)
                heights.append(float(np.mean(vals)) if vals.size else float("nan"))
            heights = np.array(heights, dtype=np.float64)
            truth_r = truth_ranks.get(c, -1)
            label = f"{_pretty_name(c)} (truth rank {truth_r})" if legend_last else "_nolegend_"
            ax.bar(
                x_grp + offs,
                heights,
                width=inner_w * 0.92,
                color=cmap[c],
                alpha=0.88,
                edgecolor=cmap[c],
                linewidth=0.5,
                label=label,
            )
        ax.axhline(1.0, color="#BBBBBB", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_xticks(x_grp, [str(int(b)) for b in usable])
        ax.set_xlabel("Added budget", fontweight="bold")
        ax.set_title(s["label"], fontweight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
        ax.tick_params(direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(4.6, 0.0)

    axes[0].set_ylabel("Mean snapshot rank\n(1 = best at budget)", fontweight="bold")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        prop={"size": 7},
        borderaxespad=0,
    )
    fig.supxlabel("Post warm-up steps at which ranks are evaluated", fontsize=8, y=0.02)
    fig.tight_layout(rect=[0.0, 0.04, 0.82, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_rank_share_bars(
    ranks: list[int],
    strategies: list[dict],
    mean_frac: dict[str, list[float]],
    err_lo: dict[str, list[float]],
    err_hi: dict[str, list[float]],
    output_path: Path,
) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(ranks), dtype=np.float64)
    n_s = len(strategies)
    width = 0.24
    offsets = np.linspace(-width, width, num=n_s)

    for s_idx, s in enumerate(strategies):
        key = s["key"]
        ys = np.array([mean_frac[key][i] for i in range(len(ranks))], dtype=np.float64)
        lo = np.array([err_lo[key][i] for i in range(len(ranks))], dtype=np.float64)
        hi = np.array([err_hi[key][i] for i in range(len(ranks))], dtype=np.float64)
        yerr = np.vstack([np.maximum(0, ys - lo), np.maximum(0, hi - ys)])
        ax.bar(
            x + offsets[s_idx],
            ys,
            width=width * 0.92,
            color=s["color"],
            alpha=0.82,
            edgecolor=s["color"],
            linewidth=0.6,
            label=s["label"],
            yerr=yerr,
            capsize=1.8,
            error_kw={"linewidth": 0.6, "color": "#333333"},
        )

    ax.set_xticks(x, [str(r) for r in ranks])
    ax.set_xlabel("Factual rank (per-run final true return)", fontweight="bold")
    ax.set_ylabel("Fraction of total budget", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_rank_delta_bars(
    ranks: list[int],
    strategies: list[dict],
    mean_delta: dict[str, list[float]],
    err_lo: dict[str, list[float]],
    err_hi: dict[str, list[float]],
    output_path: Path,
) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(ranks), dtype=np.float64)
    n_s = len(strategies)
    # Two OCBA variants: nearly touching bars (tiny gap); wider default spacing if ever >2 series.
    if n_s == 2:
        bar_face_gap = 0.032
        bar_w = 0.38
        half_sep = bar_w / 2.0 + bar_face_gap / 2.0
        offsets = np.array([-half_sep, half_sep], dtype=np.float64)
        bar_width_draw = bar_w * 0.97
    else:
        width = 0.24
        offsets = np.linspace(-width, width, num=n_s)
        bar_width_draw = width * 0.92

    for s_idx, s in enumerate(strategies):
        key = s["key"]
        ys = np.array([mean_delta[key][i] for i in range(len(ranks))], dtype=np.float64)
        lo = np.array([err_lo[key][i] for i in range(len(ranks))], dtype=np.float64)
        hi = np.array([err_hi[key][i] for i in range(len(ranks))], dtype=np.float64)
        yerr = np.vstack([np.maximum(0, ys - lo), np.maximum(0, hi - ys)])
        ax.bar(
            x + offsets[s_idx],
            ys,
            width=bar_width_draw,
            color=s["color"],
            alpha=0.82,
            edgecolor=s["color"],
            linewidth=0.6,
            label=s["label"],
            yerr=yerr,
            capsize=1.8,
            error_kw={"linewidth": 0.6, "color": "#333333"},
        )

    ax.axhline(0.0, color="#555555", linewidth=0.7, linestyle="-")
    ax.set_xticks(x, [str(r) for r in ranks])
    ax.set_xlabel("Factual rank (per-run final true return)", fontweight="bold")
    ax.set_ylabel("Delta fraction vs uniform", fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_identify_bars(
    strategies: list[dict],
    mean_added: list[float],
    low_added: list[float],
    high_added: list[float],
    output_path: Path,
) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(strategies), dtype=np.float64)
    width = 0.62
    colors = [s["color"] for s in strategies]
    ys = np.nan_to_num(np.array(mean_added, dtype=np.float64), nan=0.0)
    lo = np.nan_to_num(np.array(low_added, dtype=np.float64), nan=0.0)
    hi = np.nan_to_num(np.array(high_added, dtype=np.float64), nan=0.0)
    yerr = np.vstack([np.maximum(0, ys - lo), np.maximum(0, hi - ys)])
    ax.bar(
        x,
        ys,
        width=width,
        color=colors,
        alpha=0.82,
        edgecolor=colors,
        linewidth=0.6,
        yerr=yerr,
        capsize=2.0,
        error_kw={"linewidth": 0.6, "color": "#333333"},
    )
    ax.set_xticks(x, [s["label"] for s in strategies])
    ax.set_ylabel("Added budget at Top-1 hit", fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    figure_path = args.figure if args.figure is not None else out / "frozenlake_reward_allocation.pdf"
    reference_csv = args.reference_csv if args.reference_csv is not None else out / "frozenlake_reward_reference.csv"
    allocation_csv = args.allocation_csv if args.allocation_csv is not None else out / "frozenlake_allocation_by_reward.csv"
    detail_csv_path = args.detail_csv if args.detail_csv is not None else out / "frozenlake_allocation_detail.csv"
    rank_share_pdf = args.rank_share_pdf if args.rank_share_pdf is not None else out / "frozenlake_factual_rank_share.pdf"
    rank_delta_pdf = args.rank_delta_pdf if args.rank_delta_pdf is not None else out / "frozenlake_factual_rank_delta.pdf"
    rank_identify_pdf = args.rank_identify_pdf if args.rank_identify_pdf is not None else out / "frozenlake_factual_rank_identify.pdf"
    rank_share_csv = args.rank_share_csv if args.rank_share_csv is not None else out / "frozenlake_factual_rank_summary.csv"
    rank_delta_csv = args.rank_delta_csv if args.rank_delta_csv is not None else out / "frozenlake_factual_rank_delta.csv"
    rank_identify_csv = args.rank_identify_csv if args.rank_identify_csv is not None else out / "frozenlake_factual_rank_identify.csv"
    learning_pdf = args.learning_curve_pdf if args.learning_curve_pdf is not None else out / "frozenlake_best_true_learning_curve.pdf"
    rank_budget_pdf = args.rank_budget_pdf if args.rank_budget_pdf is not None else out / "frozenlake_rank_at_added_budget.pdf"

    run_dirs = list_run_dirs(args.log_root)
    rank_checkpoints = _parse_added_budget_checkpoints(args.rank_checkpoints)

    uniform_true_samples: dict[str, list[float]] = defaultdict(list)
    fraction_samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    detail_rows: list[dict] = []
    rank_fraction_samples: dict[str, dict[int, list[float]]] = {s["key"]: {} for s in STRATEGIES}
    delta_samples: dict[str, dict[int, list[float]]] = {"ocba": {}, "adapted_ocba": {}}
    identify_added_samples: dict[str, list[float]] = {s["key"]: [] for s in STRATEGIES}
    rank_checkpoint_samples: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run_dir in run_dirs:
        records_by_key: dict[str, list[dict]] = {}
        finals: dict[str, dict] = {}
        ok = True
        for s in STRATEGIES:
            path = run_dir / s["filename"]
            if not path.exists():
                ok = False
                break
            recs = read_round_records(path)
            if not recs:
                ok = False
                break
            records_by_key[s["key"]] = recs
            finals[s["key"]] = max(recs, key=lambda r: int(r.get("budget_consumed", -1)))
        if not ok or len(finals) != len(STRATEGIES):
            continue

        for s in STRATEGIES:
            key = s["key"]
            recs = records_by_key[key]
            for B in rank_checkpoints:
                rec = record_at_added_budget(recs, args.warmup_budget, B)
                if rec is None:
                    continue
                per = rec.get("per_candidate") or {}
                sr = snapshot_ranks(per)
                for c, rnk in sr.items():
                    rank_checkpoint_samples[key][B][c].append(float(rnk))

        uni_per = finals["uniform"].get("per_candidate") or {}
        for c, info in uni_per.items():
            uniform_true_samples[c].append(_safe_float((info or {}).get("true_mean_return")))

        run_name = run_dir.name
        for key in [s["key"] for s in STRATEGIES]:
            final_rec = finals[key]
            per = final_rec.get("per_candidate") or {}
            total_b = float(final_rec.get("budget_consumed", 0))
            for c, info in per.items():
                alloc = float((info or {}).get("allocation_cumulative", 0))
                frac = float(alloc / total_b) if total_b > 0 else np.nan
                fraction_samples[(key, c)].append(frac)
                tm = _safe_float((info or {}).get("true_mean_return"))
                if not args.no_detail_csv:
                    detail_rows.append(
                        {
                            "run_id": run_name,
                            "strategy": key,
                            "candidate": c,
                            "true_mean_final": tm,
                            "allocation_cumulative": alloc,
                            "fraction_of_total": frac,
                            "total_budget": total_b,
                        }
                    )

        if not args.no_rank_figures:
            for key in [s["key"] for s in STRATEGIES]:
                fr, _ = fraction_by_factual_rank(finals[key])
                for r, fv in fr.items():
                    rank_fraction_samples[key].setdefault(r, []).append(float(fv))

            u_fr, _ = fraction_by_factual_rank(finals["uniform"])
            for key in ("ocba", "adapted_ocba"):
                s_fr, _ = fraction_by_factual_rank(finals[key])
                for r in set(u_fr) | set(s_fr):
                    uf = u_fr.get(r, np.nan)
                    sf = s_fr.get(r, np.nan)
                    if np.isfinite(uf) and np.isfinite(sf):
                        delta_samples[key].setdefault(r, []).append(float(sf - uf))

            if not args.no_identify:
                for s in STRATEGIES:
                    key = s["key"]
                    per = finals[key].get("per_candidate") or {}
                    leader = _fix_pick_leader(per)
                    hit = first_hit_top1_budget(records_by_key[key], leader)
                    if hit is not None:
                        added = float(hit) - float(args.warmup_budget)
                        identify_added_samples[key].append(max(0.0, added))

    candidates = sorted(uniform_true_samples.keys())
    if not candidates:
        raise RuntimeError("No complete runs (all three strategies present with data).")

    ref_rows = []
    mean_true: dict[str, float] = {}
    var_true: dict[str, float] = {}
    std_true: dict[str, float] = {}
    for c in candidates:
        arr = np.array(uniform_true_samples[c], dtype=np.float64)
        n = int(arr.size)
        mu = float(np.mean(arr)) if n else np.nan
        var = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        mean_true[c] = mu
        var_true[c] = var
        std_true[c] = std

    candidates_ordered = sorted(candidates, key=lambda cc: (-mean_true[cc], cc))
    truth_ranks = {c: i + 1 for i, c in enumerate(candidates_ordered)}

    for rank, c in enumerate(candidates_ordered, start=1):
        ref_rows.append(
            {
                "candidate": c,
                "rank_by_mean_final_true_uniform": rank,
                "mean_final_true_uniform": mean_true[c],
                "var_final_true_uniform": var_true[c],
                "std_final_true_uniform": std_true[c],
                "n_uniform_runs": len(uniform_true_samples[c]),
            }
        )

    reference_csv.parent.mkdir(parents=True, exist_ok=True)
    with reference_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "rank_by_mean_final_true_uniform",
                "mean_final_true_uniform",
                "var_final_true_uniform",
                "std_final_true_uniform",
                "n_uniform_runs",
            ],
        )
        w.writeheader()
        w.writerows(ref_rows)

    alloc_rows = []
    mean_frac = {s["key"]: [] for s in STRATEGIES}
    err_lo = {s["key"]: [] for s in STRATEGIES}
    err_hi = {s["key"]: [] for s in STRATEGIES}

    for c in candidates_ordered:
        for s in STRATEGIES:
            key = s["key"]
            vals = np.array(fraction_samples.get((key, c), []), dtype=np.float64)
            m, lo, hi = _mean_ci95(vals)
            alloc_rows.append(
                {
                    "strategy": key,
                    "candidate": c,
                    "n_runs": int(vals.size),
                    "fraction_mean": m,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
            mean_frac[key].append(m)
            err_lo[key].append(lo)
            err_hi[key].append(hi)

    with allocation_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["strategy", "candidate", "n_runs", "fraction_mean", "ci95_low", "ci95_high"],
        )
        w.writeheader()
        w.writerows(alloc_rows)

    if not args.no_detail_csv and detail_rows:
        with detail_csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "strategy",
                    "candidate",
                    "true_mean_final",
                    "allocation_cumulative",
                    "fraction_of_total",
                    "total_budget",
                ],
            )
            w.writeheader()
            w.writerows(detail_rows)

    if not args.no_figure:
        plot_combined_figure(
            candidates_ordered,
            mean_true,
            std_true,
            var_true,
            mean_frac,
            err_lo,
            err_hi,
            figure_path,
            top_strategies=STRATEGIES_OCBA_TOP,
        )

    if not args.no_learning_curve:
        plot_best_true_learning_curve(
            run_dirs,
            STRATEGIES,
            learning_pdf,
            smooth_weight=args.learning_smooth,
            min_budget=args.learning_min_budget,
            max_budget=args.learning_max_budget,
            trim_extremes_count=args.learning_trim_extremes,
            warmup_budget=args.warmup_budget,
        )

    if not args.no_rank_budget_figure:
        plot_rank_at_added_budgets(
            STRATEGIES,
            candidates_ordered,
            truth_ranks,
            rank_checkpoint_samples,
            rank_checkpoints,
            rank_budget_pdf,
        )

    if not args.no_rank_figures:
        rank_axis = sorted({r for bucket in rank_fraction_samples.values() for r in bucket})
        if not rank_axis:
            raise RuntimeError("Rank-bucket aggregation empty (no complete runs).")

        rank_summary_rows = []
        r_mean_frac = {s["key"]: [] for s in STRATEGIES}
        r_err_lo = {s["key"]: [] for s in STRATEGIES}
        r_err_hi = {s["key"]: [] for s in STRATEGIES}
        for r in rank_axis:
            for s in STRATEGIES:
                key = s["key"]
                vals = np.array(rank_fraction_samples[key].get(r, []), dtype=np.float64)
                m, lo, hi = _mean_ci95(vals)
                rank_summary_rows.append(
                    {
                        "factual_rank": r,
                        "strategy": key,
                        "n_runs": int(vals.size),
                        "fraction_mean": m,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
                r_mean_frac[key].append(m)
                r_err_lo[key].append(lo)
                r_err_hi[key].append(hi)

        with rank_share_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["strategy", "factual_rank", "n_runs", "fraction_mean", "ci95_low", "ci95_high"],
            )
            w.writeheader()
            w.writerows(rank_summary_rows)

        plot_rank_share_bars(rank_axis, STRATEGIES, r_mean_frac, r_err_lo, r_err_hi, rank_share_pdf)

        delta_strategies = [s for s in STRATEGIES if s["key"] in delta_samples]
        delta_summary_rows = []
        d_mean = {s["key"]: [] for s in delta_strategies}
        d_lo = {s["key"]: [] for s in delta_strategies}
        d_hi = {s["key"]: [] for s in delta_strategies}
        for r in rank_axis:
            for s in delta_strategies:
                key = s["key"]
                vals = np.array(delta_samples[key].get(r, []), dtype=np.float64)
                m, lo, hi = _mean_ci95(vals)
                delta_summary_rows.append(
                    {
                        "factual_rank": r,
                        "strategy": key,
                        "n_runs": int(vals.size),
                        "delta_fraction_mean": m,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
                d_mean[key].append(m)
                d_lo[key].append(lo)
                d_hi[key].append(hi)

        with rank_delta_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["strategy", "factual_rank", "n_runs", "delta_fraction_mean", "ci95_low", "ci95_high"],
            )
            w.writeheader()
            w.writerows(delta_summary_rows)

        plot_rank_delta_bars(rank_axis, delta_strategies, d_mean, d_lo, d_hi, rank_delta_pdf)

        if not args.no_identify:
            identify_rows = []
            mean_i: list[float] = []
            lo_i: list[float] = []
            hi_i: list[float] = []
            for s in STRATEGIES:
                key = s["key"]
                vals = np.array(identify_added_samples[key], dtype=np.float64)
                m, lo, hi = _mean_ci95(vals)
                identify_rows.append(
                    {
                        "strategy": key,
                        "n_runs": int(vals.size),
                        "added_budget_at_hit_mean": m,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
                mean_i.append(m if np.isfinite(m) else float("nan"))
                lo_i.append(lo if np.isfinite(lo) else float("nan"))
                hi_i.append(hi if np.isfinite(hi) else float("nan"))

            with rank_identify_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["strategy", "n_runs", "added_budget_at_hit_mean", "ci95_low", "ci95_high"],
                )
                w.writeheader()
                w.writerows(identify_rows)

            plot_identify_bars(STRATEGIES, mean_i, lo_i, hi_i, rank_identify_pdf)


if __name__ == "__main__":
    main()
