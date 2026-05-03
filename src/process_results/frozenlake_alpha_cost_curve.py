import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairKey:
    group_id: int
    candidate_id: str
    iteration: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build alpha cost curves from FrozenLake offline compare logs. "
            "X-axis: alpha, Y-axis: rounds to reach alpha * target_best_reward."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run directory like logs/openevolve_frozenlake/offline_compare_5groups_20260430_xxxxxx",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,1.00",
        help="Comma separated alpha values in (0, 1].",
    )
    parser.add_argument(
        "--curve-csv-output",
        type=Path,
        default=None,
        help="Output CSV path for alpha-level curve metrics.",
    )
    parser.add_argument(
        "--pair-csv-output",
        type=Path,
        default=None,
        help="Output CSV path for per-pair hit details.",
    )
    parser.add_argument(
        "--summary-json-output",
        type=Path,
        default=None,
        help="Output JSON path for overall summary metrics.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Output PNG path for alpha-vs-rounds curve.",
    )
    parser.add_argument(
        "--steps-plot-output",
        type=Path,
        default=None,
        help="Output PNG path for alpha-vs-steps curve.",
    )
    parser.add_argument(
        "--tradeoff-plot-output",
        type=Path,
        default=None,
        help="Output PNG path for delta-round/delta-steps tradeoff curve.",
    )
    return parser.parse_args()


def _parse_alphas(alpha_text: str) -> list[float]:
    raw = [x.strip() for x in alpha_text.split(",") if x.strip()]
    alphas: list[float] = []
    for item in raw:
        value = float(item)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Invalid alpha {value}. Must be in (0, 1].")
        alphas.append(value)
    if not alphas:
        raise ValueError("No valid alpha values provided.")
    return sorted(set(alphas))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _group_id_from_path(file_path: Path) -> int:
    # Expected parent path: group_1, group_2, ...
    name = file_path.parent.name
    if not name.startswith("group_"):
        return -1
    return _safe_int(name.split("_", 1)[1], -1)


def load_final_records(run_root: Path) -> tuple[dict[PairKey, dict], dict[str, float]]:
    final_records: dict[PairKey, dict] = {}
    total_ocba_steps = 0.0
    total_uniform_steps = 0.0
    n_rows = 0

    for file_path in sorted(run_root.glob("group_*/offline_compare.jsonl")):
        group_id = _group_id_from_path(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = PairKey(
                    group_id=group_id,
                    candidate_id=str(row.get("candidate_id", "")),
                    iteration=_safe_int(row.get("iteration", -1)),
                )
                ocba = row.get("ocba", {}) or {}
                uniform = row.get("uniform", {}) or {}
                ocba_best = _safe_float(ocba.get("best_true_reward", -1e9), -1e9)
                uniform_best = _safe_float(uniform.get("best_true_reward", -1e9), -1e9)
                final_records[key] = {
                    "target_best": max(ocba_best, uniform_best),
                    "ocba_best_final": ocba_best,
                    "uniform_best_final": uniform_best,
                }
                total_ocba_steps += _safe_float(ocba.get("budget_used", 0.0), 0.0)
                total_uniform_steps += _safe_float(uniform.get("budget_used", 0.0), 0.0)
                n_rows += 1

    totals = {
        "n_records": float(n_rows),
        "total_ocba_steps": float(total_ocba_steps),
        "total_uniform_steps": float(total_uniform_steps),
        "delta_steps_ocba_minus_uniform": float(total_ocba_steps - total_uniform_steps),
    }
    return final_records, totals


def load_round_trajectories(run_root: Path) -> dict[tuple[PairKey, str], list[tuple[int, float, float]]]:
    """
    Return mapping: (pair_key, strategy) -> list[(round_idx, consumed_before, round_best_mean)] sorted by round.
    strategy in {"ocba", "uniform"}.
    """
    trajectories: dict[tuple[PairKey, str], list[tuple[int, float, float]]] = {}

    pat_candidate = re.compile(r"candidate_id=([^;]+)")
    pat_iteration = re.compile(r"iteration=([^;]+)")
    pat_strategy = re.compile(r"strategy=([^;]+)")

    for file_path in sorted(run_root.glob("group_*/offline_compare_rounds.jsonl")):
        group_id = _group_id_from_path(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tag = str(row.get("trace_tag", ""))

                m_candidate = pat_candidate.search(tag)
                m_iteration = pat_iteration.search(tag)
                m_strategy = pat_strategy.search(tag)
                if not (m_candidate and m_iteration and m_strategy):
                    continue

                strategy = m_strategy.group(1).strip().lower()
                if strategy not in {"ocba", "uniform"}:
                    continue

                key = PairKey(
                    group_id=group_id,
                    candidate_id=m_candidate.group(1).strip(),
                    iteration=_safe_int(m_iteration.group(1), -1),
                )
                round_idx = _safe_int(row.get("round_idx", -1), -1)
                consumed_before = _safe_float(row.get("consumed_budget_before_round", 0.0), 0.0)
                means = row.get("means", []) or []
                round_best = max(float(x) for x in means) if means else -1e9

                trajectories.setdefault((key, strategy), []).append((round_idx, consumed_before, round_best))

    for map_key, records in list(trajectories.items()):
        trajectories[map_key] = sorted(records, key=lambda x: x[0])
    return trajectories


def first_hit(records: list[tuple[int, float, float]], threshold: float) -> tuple[int, float] | None:
    for round_idx, consumed_before, round_best in records:
        if round_best >= threshold:
            return round_idx, consumed_before
    return None


def _mean_median(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.median(arr))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(
    curve_rows: list[dict],
    pair_rows: list[dict],
    rounds_path: Path | None,
    steps_path: Path | None,
    tradeoff_path: Path | None,
) -> tuple[bool, bool, bool]:
    if rounds_path is None and steps_path is None and tradeoff_path is None:
        return False, False, False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False, False, False

    alphas = [float(r["alpha"]) for r in curve_rows]
    ocba_rounds = [float(r["ocba_round_mean"]) for r in curve_rows]
    uniform_rounds = [float(r["uniform_round_mean"]) for r in curve_rows]
    ocba_steps = [float(r["ocba_steps_mean_to_hit"]) for r in curve_rows]
    uniform_steps = [float(r["uniform_steps_mean_to_hit"]) for r in curve_rows]

    rounds_written = False
    if rounds_path is not None:
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
        plot_alphas = list(alphas)
        x_total = np.arange(len(plot_alphas), dtype=np.float64)

        ocba_total_by_alpha: dict[float, list[float]] = {a: [] for a in plot_alphas}
        uniform_total_by_alpha: dict[float, list[float]] = {a: [] for a in plot_alphas}
        for row in pair_rows:
            a = float(row["alpha"])
            if a not in ocba_total_by_alpha:
                continue
            ocba_val = row.get("ocba_round_to_hit")
            uniform_val = row.get("uniform_round_to_hit")
            if isinstance(ocba_val, (int, float)) and np.isfinite(float(ocba_val)):
                ocba_total_by_alpha[a].append(float(ocba_val))
            if isinstance(uniform_val, (int, float)) and np.isfinite(float(uniform_val)):
                uniform_total_by_alpha[a].append(float(uniform_val))

        def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
            if not values:
                return np.nan, np.nan, np.nan
            arr = np.array(values, dtype=np.float64)
            mean = float(np.mean(arr))
            if arr.size <= 1:
                return mean, mean, mean
            sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            half = 1.96 * sem
            return mean, mean - half, mean + half

        ocba_stats = [_mean_ci95(ocba_total_by_alpha[a]) for a in plot_alphas]
        uniform_stats = [_mean_ci95(uniform_total_by_alpha[a]) for a in plot_alphas]
        ocba_mean = np.array([s[0] for s in ocba_stats], dtype=np.float64)
        uniform_mean = np.array([s[0] for s in uniform_stats], dtype=np.float64)
        ocba_ci_low = np.array([s[1] for s in ocba_stats], dtype=np.float64)
        ocba_ci_high = np.array([s[2] for s in ocba_stats], dtype=np.float64)
        uniform_ci_low = np.array([s[1] for s in uniform_stats], dtype=np.float64)
        uniform_ci_high = np.array([s[2] for s in uniform_stats], dtype=np.float64)

        # Typical single-column Nature figure width is around 85-90 mm.
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        # Low-saturation palette.
        color_ocba = "#6E8FA8"
        color_uniform = "#B08A72"
        fill_ocba = "#CAD6DF"
        fill_uniform = "#D9CBC0"
        ax.plot(x_total, ocba_mean, color=color_ocba, linewidth=1.4, marker="o", markersize=3.2, label="OCBA")
        ax.plot(
            x_total, uniform_mean, color=color_uniform, linewidth=1.4, marker="o", markersize=3.2, label="Uniform"
        )
        # Formal K-box style CI glyphs: rectangle body + whiskers.
        box_width = 0.10
        ocba_x = x_total - 0.08
        uniform_x = x_total + 0.08
        for x, lo, hi in zip(ocba_x, ocba_ci_low, ocba_ci_high):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.vlines(x, lo, hi, color=color_ocba, linewidth=0.55, alpha=0.9, zorder=3)
                ax.hlines([lo, hi], x - box_width * 0.45, x + box_width * 0.45, color=color_ocba, linewidth=0.55, zorder=3)
                ax.bar(
                    x,
                    max(hi - lo, 0.001),
                    width=box_width,
                    bottom=lo,
                    color=fill_ocba,
                    edgecolor=color_ocba,
                    linewidth=0.55,
                    alpha=0.75,
                    zorder=4,
                )
        for x, lo, hi in zip(uniform_x, uniform_ci_low, uniform_ci_high):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.vlines(x, lo, hi, color=color_uniform, linewidth=0.55, alpha=0.9, zorder=3)
                ax.hlines(
                    [lo, hi], x - box_width * 0.45, x + box_width * 0.45, color=color_uniform, linewidth=0.55, zorder=3
                )
                ax.bar(
                    x,
                    max(hi - lo, 0.001),
                    width=box_width,
                    bottom=lo,
                    color=fill_uniform,
                    edgecolor=color_uniform,
                    linewidth=0.55,
                    alpha=0.75,
                    zorder=4,
                )
        ax.set_xlabel("alpha", fontweight="bold")
        ax.set_ylabel("Rounds to threshold", fontweight="bold")
        ax.set_xticks(x_total, [f"{a:.2f}" for a in plot_alphas])
        ax.set_xlim(-0.35, len(plot_alphas) - 0.65)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
        ax.tick_params(direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
        rounds_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        plt.savefig(rounds_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        rounds_written = True

    steps_written = False
    if steps_path is not None:
        # Build distribution by alpha for boxplots.
        ocba_by_alpha: dict[float, list[float]] = {a: [] for a in alphas}
        uniform_by_alpha: dict[float, list[float]] = {a: [] for a in alphas}
        for row in pair_rows:
            a = float(row["alpha"])
            if a not in ocba_by_alpha:
                continue
            ocba_val = row.get("ocba_steps_to_hit")
            uniform_val = row.get("uniform_steps_to_hit")
            if isinstance(ocba_val, (int, float)) and np.isfinite(float(ocba_val)):
                ocba_by_alpha[a].append(float(ocba_val))
            if isinstance(uniform_val, (int, float)) and np.isfinite(float(uniform_val)):
                uniform_by_alpha[a].append(float(uniform_val))

        plt.figure(figsize=(8, 5))
        offset = 0.008
        width = 0.012
        ocba_positions = [a - offset for a in alphas]
        uniform_positions = [a + offset for a in alphas]
        ocba_data = [ocba_by_alpha[a] if ocba_by_alpha[a] else [np.nan] for a in alphas]
        uniform_data = [uniform_by_alpha[a] if uniform_by_alpha[a] else [np.nan] for a in alphas]

        plt.boxplot(
            ocba_data,
            positions=ocba_positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "tab:blue", "alpha": 0.20, "edgecolor": "tab:blue"},
            medianprops={"color": "tab:blue"},
            whiskerprops={"color": "tab:blue"},
            capprops={"color": "tab:blue"},
        )
        plt.boxplot(
            uniform_data,
            positions=uniform_positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "tab:orange", "alpha": 0.20, "edgecolor": "tab:orange"},
            medianprops={"color": "tab:orange"},
            whiskerprops={"color": "tab:orange"},
            capprops={"color": "tab:orange"},
        )

        plt.plot(alphas, ocba_steps, marker="o", color="tab:blue", label="OCBA mean")
        plt.plot(alphas, uniform_steps, marker="o", color="tab:orange", label="Uniform mean")
        plt.xlabel("alpha")
        plt.ylabel("Mean steps to hit alpha * target")
        plt.title("Alpha Cost Curve (Steps): Mean + Boxplot")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.xticks(alphas, [f"{a:.2f}" for a in alphas])
        steps_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(steps_path, dpi=160)
        plt.close()
        steps_written = True

    tradeoff_written = False
    if tradeoff_path is not None:
        delta_round = [float(r["delta_round_mean_ocba_minus_uniform"]) for r in curve_rows]
        delta_steps = [float(r["delta_steps_mean_to_hit_ocba_minus_uniform"]) for r in curve_rows]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(alphas, delta_round, color="tab:blue", marker="o", label="Delta rounds (OCBA-Uniform)")
        ax1.set_xlabel("alpha")
        ax1.set_ylabel("Delta rounds", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.axhline(0.0, color="tab:blue", linestyle="--", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(alphas, delta_steps, color="tab:red", marker="s", label="Delta steps (OCBA-Uniform)")
        ax2.set_ylabel("Delta steps", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.axhline(0.0, color="tab:red", linestyle="--", alpha=0.3)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        plt.title("Alpha Speed-Cost Tradeoff")
        fig.tight_layout()
        tradeoff_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(tradeoff_path, dpi=160)
        plt.close(fig)
        tradeoff_written = True

    return rounds_written, steps_written, tradeoff_written


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    alphas = _parse_alphas(args.alphas)
    final_records, totals = load_final_records(run_root)
    trajectories = load_round_trajectories(run_root)

    pair_rows: list[dict] = []
    curve_rows: list[dict] = []

    all_keys = sorted(final_records.keys(), key=lambda x: (x.group_id, x.iteration, x.candidate_id))

    for alpha in alphas:
        ocba_round_hits: list[float] = []
        uniform_round_hits: list[float] = []
        ocba_step_hits: list[float] = []
        uniform_step_hits: list[float] = []

        ocba_hit_count = 0
        uniform_hit_count = 0
        pair_count = 0

        for key in all_keys:
            target_best = _safe_float(final_records[key]["target_best"], -1e9)
            threshold = float(alpha * target_best)
            ocba_traj = trajectories.get((key, "ocba"), [])
            uniform_traj = trajectories.get((key, "uniform"), [])
            if not ocba_traj or not uniform_traj:
                continue
            pair_count += 1

            ocba_hit = first_hit(ocba_traj, threshold)
            uniform_hit = first_hit(uniform_traj, threshold)
            ocba_round = float(ocba_hit[0]) if ocba_hit is not None else float("nan")
            ocba_step = float(ocba_hit[1]) if ocba_hit is not None else float("nan")
            uniform_round = float(uniform_hit[0]) if uniform_hit is not None else float("nan")
            uniform_step = float(uniform_hit[1]) if uniform_hit is not None else float("nan")

            if ocba_hit is not None:
                ocba_hit_count += 1
                ocba_round_hits.append(ocba_round)
                ocba_step_hits.append(ocba_step)
            if uniform_hit is not None:
                uniform_hit_count += 1
                uniform_round_hits.append(uniform_round)
                uniform_step_hits.append(uniform_step)

            pair_rows.append(
                {
                    "alpha": alpha,
                    "group_id": key.group_id,
                    "iteration": key.iteration,
                    "candidate_id": key.candidate_id,
                    "target_best_reward": target_best,
                    "threshold_reward": threshold,
                    "ocba_hit": int(ocba_hit is not None),
                    "uniform_hit": int(uniform_hit is not None),
                    "ocba_round_to_hit": ocba_round,
                    "uniform_round_to_hit": uniform_round,
                    "ocba_steps_to_hit": ocba_step,
                    "uniform_steps_to_hit": uniform_step,
                    "delta_round_ocba_minus_uniform": (
                        ocba_round - uniform_round
                        if (ocba_hit is not None and uniform_hit is not None)
                        else float("nan")
                    ),
                    "delta_steps_ocba_minus_uniform": (
                        ocba_step - uniform_step
                        if (ocba_hit is not None and uniform_hit is not None)
                        else float("nan")
                    ),
                }
            )

        ocba_round_mean, ocba_round_median = _mean_median(ocba_round_hits)
        uniform_round_mean, uniform_round_median = _mean_median(uniform_round_hits)
        ocba_step_mean, ocba_step_median = _mean_median(ocba_step_hits)
        uniform_step_mean, uniform_step_median = _mean_median(uniform_step_hits)

        curve_rows.append(
            {
                "alpha": alpha,
                "n_pairs": pair_count,
                "ocba_hit_rate": (ocba_hit_count / pair_count) if pair_count > 0 else float("nan"),
                "uniform_hit_rate": (uniform_hit_count / pair_count) if pair_count > 0 else float("nan"),
                "ocba_round_mean": ocba_round_mean,
                "uniform_round_mean": uniform_round_mean,
                "delta_round_mean_ocba_minus_uniform": ocba_round_mean - uniform_round_mean,
                "ocba_round_median": ocba_round_median,
                "uniform_round_median": uniform_round_median,
                "delta_round_median_ocba_minus_uniform": ocba_round_median - uniform_round_median,
                "ocba_steps_mean_to_hit": ocba_step_mean,
                "uniform_steps_mean_to_hit": uniform_step_mean,
                "delta_steps_mean_to_hit_ocba_minus_uniform": ocba_step_mean - uniform_step_mean,
                "ocba_steps_median_to_hit": ocba_step_median,
                "uniform_steps_median_to_hit": uniform_step_median,
                "delta_steps_median_to_hit_ocba_minus_uniform": ocba_step_median - uniform_step_median,
            }
        )

    curve_csv = args.curve_csv_output or (run_root / "alpha_cost_curve.csv")
    pair_csv = args.pair_csv_output or (run_root / "alpha_pair_hits.csv")
    summary_json = args.summary_json_output or (run_root / "alpha_cost_summary.json")
    rounds_plot_path = args.plot_output or (run_root / "alpha_rounds_curve.pdf")
    steps_plot_path = args.steps_plot_output or (run_root / "alpha_steps_curve.png")
    tradeoff_plot_path = args.tradeoff_plot_output or (run_root / "alpha_tradeoff_curve.png")

    write_csv(
        curve_csv,
        curve_rows,
        [
            "alpha",
            "n_pairs",
            "ocba_hit_rate",
            "uniform_hit_rate",
            "ocba_round_mean",
            "uniform_round_mean",
            "delta_round_mean_ocba_minus_uniform",
            "ocba_round_median",
            "uniform_round_median",
            "delta_round_median_ocba_minus_uniform",
            "ocba_steps_mean_to_hit",
            "uniform_steps_mean_to_hit",
            "delta_steps_mean_to_hit_ocba_minus_uniform",
            "ocba_steps_median_to_hit",
            "uniform_steps_median_to_hit",
            "delta_steps_median_to_hit_ocba_minus_uniform",
        ],
    )
    write_csv(
        pair_csv,
        pair_rows,
        [
            "alpha",
            "group_id",
            "iteration",
            "candidate_id",
            "target_best_reward",
            "threshold_reward",
            "ocba_hit",
            "uniform_hit",
            "ocba_round_to_hit",
            "uniform_round_to_hit",
            "ocba_steps_to_hit",
            "uniform_steps_to_hit",
            "delta_round_ocba_minus_uniform",
            "delta_steps_ocba_minus_uniform",
        ],
    )

    rounds_plot_written, steps_plot_written, tradeoff_plot_written = maybe_plot(
        curve_rows, pair_rows, rounds_plot_path, steps_plot_path, tradeoff_plot_path
    )

    payload = {
        "run_root": str(run_root),
        "alphas": alphas,
        "overall_steps": totals,
        "curve_csv": str(curve_csv),
        "pair_csv": str(pair_csv),
        "rounds_plot_output": str(rounds_plot_path) if rounds_plot_written else "",
        "rounds_plot_written": bool(rounds_plot_written),
        "steps_plot_output": str(steps_plot_path) if steps_plot_written else "",
        "steps_plot_written": bool(steps_plot_written),
        "tradeoff_plot_output": str(tradeoff_plot_path) if tradeoff_plot_written else "",
        "tradeoff_plot_written": bool(tradeoff_plot_written),
        "n_curve_rows": len(curve_rows),
        "n_pair_rows": len(pair_rows),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
