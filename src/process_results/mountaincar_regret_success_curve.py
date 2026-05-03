import argparse
import csv
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

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
    parser = argparse.ArgumentParser(
        description=(
            "Plot MountainCar simple regret and success@delta curves "
            "using a fixed oracle best reward."
        )
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=SRC_DIR / "logs" / "MountainCar_Adaptive",
        help="Root directory containing run_* folders.",
    )
    parser.add_argument(
        "--oracle-mean",
        type=float,
        default=-108.2,
        help="Fixed oracle best reward mean (default: -108.2).",
    )
    parser.add_argument(
        "--oracle-std",
        type=float,
        default=8.158431221748456,
        help="Reference std of oracle reward (default: baseline std).",
    )
    parser.add_argument(
        "--deltas",
        type=str,
        default="std",
        help="Comma separated deltas. Supports numeric values and token 'std'.",
    )
    parser.add_argument(
        "--simple-output",
        type=Path,
        default=None,
        help="Output PNG path for simple regret curve.",
    )
    parser.add_argument(
        "--simple-csv-output",
        type=Path,
        default=None,
        help="Output CSV path for simple regret curve data.",
    )
    parser.add_argument(
        "--success-output",
        type=Path,
        default=None,
        help="Output PNG path for success@delta curves.",
    )
    parser.add_argument(
        "--success-csv-output",
        type=Path,
        default=None,
        help="Output CSV path for success@delta curve data.",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=1_500_000,
        help="Minimum budget(step) shown on curves.",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=3_000_000,
        help="Maximum budget(step) shown on curves.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="EMA smoothing weight (default: 0.6).",
    )
    parser.add_argument(
        "--trim-extremes-count",
        type=int,
        default=0,
        help="Remove lowest N and highest N regrets per step before aggregation (default: 0).",
    )
    parser.add_argument(
        "--warmup-budget",
        type=int,
        default=800_000,
        help="Warmup budget to subtract from total budget for x-axis (default: 800000).",
    )
    parser.add_argument(
        "--hitting-output",
        type=Path,
        default=None,
        help="(Deprecated) Output PNG path for target-hitting budget bar chart.",
    )
    parser.add_argument(
        "--hitting-csv-output",
        type=Path,
        default=None,
        help="(Deprecated) Output CSV path for target-hitting budget summary.",
    )
    parser.add_argument(
        "--hitting-regret",
        type=str,
        default="10",
        help="Target regret (delta) used by success@delta bar chart.",
    )
    parser.add_argument(
        "--pass-levels",
        type=str,
        default="0.2,0.4,0.6,0.8",
        help="Pass probability thresholds for cost bars, e.g. 0.2,0.4,0.6,0.8.",
    )
    parser.add_argument(
        "--iso-output",
        type=Path,
        default=None,
        help="Output PNG path for iso-budget success bar chart.",
    )
    parser.add_argument(
        "--iso-csv-output",
        type=Path,
        default=None,
        help="Output CSV path for iso-budget success summary.",
    )
    parser.add_argument(
        "--iso-budgets",
        type=str,
        default="400000,800000,1200000,1600000,2000000",
        help="Added budget checkpoints for iso-budget chart.",
    )
    parser.add_argument(
        "--auc-output",
        type=Path,
        default=None,
        help="(Deprecated) Output PNG path for success AUC bar chart.",
    )
    parser.add_argument(
        "--auc-csv-output",
        type=Path,
        default=None,
        help="(Deprecated) Output CSV path for success AUC summary.",
    )
    parser.add_argument(
        "--contour-output",
        type=Path,
        default=None,
        help="Output PNG path for success contour figure.",
    )
    parser.add_argument(
        "--contour-deltas",
        type=str,
        default="30,25,20,15,10",
        help="Delta values used to build contour map.",
    )
    parser.add_argument(
        "--hit-consecutive-k",
        type=int,
        default=1,
        help="Consecutive points required for hit criterion (default: 1).",
    )
    return parser.parse_args()


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


def _parse_single_delta(value_text: str, oracle_std: float) -> float:
    text = str(value_text).strip().lower()
    if text == "std":
        return float(oracle_std)
    return float(text)


def _parse_pass_levels(text: str) -> list[float]:
    vals = []
    for part in [x.strip() for x in str(text).split(",") if x.strip()]:
        v = float(part)
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Invalid pass level {v}, must be in (0, 1].")
        vals.append(v)
    uniq = sorted(set(vals))
    if not uniq:
        raise ValueError("No valid pass levels.")
    return uniq


def _parse_iso_budgets(text: str) -> list[int]:
    vals = []
    for part in [x.strip() for x in str(text).split(",") if x.strip()]:
        v = int(float(part))
        if v < 0:
            raise ValueError(f"Invalid iso budget {v}, must be >= 0.")
        vals.append(v)
    uniq = sorted(set(vals))
    if not uniq:
        raise ValueError("No valid iso budgets.")
    return uniq


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
    run_dirs: list[Path], oracle_best: float, min_budget: int | None, max_budget: int | None
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


def collect_run_regret_trajectories(
    run_dirs: list[Path], oracle_best: float, min_budget: int | None, max_budget: int | None, warmup_budget: int
) -> dict[str, list[list[tuple[int, float]]]]:
    out = {s["key"]: [] for s in STRATEGIES}
    for run_dir in run_dirs:
        for s in STRATEGIES:
            key = s["key"]
            path = run_dir / s["filename"]
            if not path.exists():
                continue
            records = read_round_records(path)
            if not records:
                continue
            seq = []
            for rec in records:
                step = int(rec.get("budget_consumed", -1))
                if step < 0:
                    continue
                if min_budget is not None and step < int(min_budget):
                    continue
                if max_budget is not None and step > int(max_budget):
                    continue
                added_budget = int(step) - int(warmup_budget)
                if added_budget < 0:
                    continue
                best_est = _best_estimated_reward(rec.get("per_candidate", {}) or {})
                regret = float(oracle_best - best_est)
                seq.append((added_budget, regret))
            seq = sorted(seq, key=lambda x: x[0])
            if seq:
                out[key].append(seq)
    return out


def aggregate_regret_curve(step_regrets: dict[int, list[float]], trim_count: int = 0, warmup_budget: int = 800_000):
    x = []
    mean = []
    low = []
    high = []
    n = []
    for step in sorted(step_regrets.keys()):
        added_budget = int(step) - int(warmup_budget)
        if added_budget < 0:
            continue
        vals = np.array(step_regrets[step], dtype=np.float64)
        vals = _trim_extremes(vals, trim_count=trim_count)
        m, lo, hi = _mean_ci95(vals)
        x.append(int(added_budget))
        mean.append(float(m))
        low.append(float(lo))
        high.append(float(hi))
        n.append(int(vals.size))
    return np.array(x), np.array(mean), np.array(low), np.array(high), np.array(n)


def aggregate_success_curve(
    step_regrets: dict[int, list[float]],
    delta: float,
    trim_count: int = 0,
    warmup_budget: int = 800_000,
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


def plot_simple_regret(curves: list[dict], output_path: Path) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for item in curves:
        s = item["strategy"]
        ax.plot(
            item["x"],
            item["mean"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=s["linewidth"],
            label=s["label"],
        )
        ax.fill_between(
            item["x"],
            item["low"],
            item["high"],
            color=s["fill"],
            alpha=0.65,
            label="_nolegend_",
        )
    ax.set_xlabel("Added Budget (Steps)", fontweight="bold")
    ax.set_ylabel("Simple Regret", fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_success_delta(success_items: list[dict], output_path: Path) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for item in success_items:
        s = item["strategy"]
        label = s["label"]
        ax.plot(
            item["x"],
            item["mean"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=s["linewidth"],
            label=label,
        )
        ax.fill_between(
            item["x"],
            item["low"],
            item["high"],
            color=s["fill"],
            alpha=0.30,
            label="_nolegend_",
        )
    ax.set_xlabel("Added Budget (Steps)", fontweight="bold")
    ax.set_ylabel("Success@delta Probability", fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_simple_csv(curves: list[dict], csv_path: Path) -> None:
    rows = []
    for item in curves:
        key = item["strategy"]["key"]
        for step, m, lo, hi, n in zip(item["x"], item["mean"], item["low"], item["high"], item["n"]):
            rows.append(
                {
                    "strategy": key,
                    "added_budget": int(step),
                    "simple_regret_mean": float(m),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "n_runs": int(n),
                }
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "added_budget", "simple_regret_mean", "ci95_low", "ci95_high", "n_runs"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_success_csv(items: list[dict], csv_path: Path) -> None:
    rows = []
    for item in items:
        key = item["strategy"]["key"]
        delta = float(item["delta"])
        for step, m, lo, hi, n in zip(item["x"], item["mean"], item["low"], item["high"], item["n"]):
            rows.append(
                {
                    "strategy": key,
                    "delta": delta,
                    "added_budget": int(step),
                    "success_prob_mean": float(m),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "n_runs": int(n),
                }
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "delta", "added_budget", "success_prob_mean", "ci95_low", "ci95_high", "n_runs"],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_hitting_budget(
    success_items: list[dict], pass_levels: list[float]
) -> list[dict]:
    rows = []
    for item in success_items:
        key = item["strategy"]["key"]
        x = np.array(item["x"], dtype=np.float64)
        p = np.array(item["mean"], dtype=np.float64)
        n_arr = np.array(item["n"], dtype=np.float64)
        n_runs = int(np.max(n_arr)) if n_arr.size > 0 else 0
        for level in pass_levels:
            idx = np.where(p >= float(level))[0]
            budget = float(x[idx[0]]) if idx.size > 0 else np.nan
            rows.append(
                {
                    "strategy": key,
                    "pass_level": float(level),
                    "n_runs": n_runs,
                    "budget_to_pass": budget,
                }
            )
    return rows


def plot_hitting_budget(summary_rows: list[dict], output_path: Path) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    pass_levels = sorted(set([float(r["pass_level"]) for r in summary_rows]))
    x = np.arange(len(pass_levels), dtype=np.float64)
    width = 0.24
    offsets = np.linspace(-width, width, num=len(STRATEGIES))
    for s_idx, s in enumerate(STRATEGIES):
        bar_x = x + offsets[s_idx]
        meds = []
        for lv in pass_levels:
            row = next((r for r in summary_rows if float(r["pass_level"]) == lv and r["strategy"] == s["key"]), None)
            meds.append(float(row["budget_to_pass"]) if row is not None else np.nan)
        ax.bar(
            bar_x,
            meds,
            color=s["color"],
            alpha=0.8,
            width=width * 0.92,
            edgecolor=s["color"],
            linewidth=0.6,
            label=s["label"].replace(" Allocation", ""),
        )
        for xi, yi in zip(bar_x, meds):
            if not np.isfinite(yi):
                ax.text(xi, 0.02, "NA", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, [f"{int(lv*100)}%" for lv in pass_levels])
    ax.set_ylabel("Added Budget Cost", fontweight="bold")
    ax.set_xlabel("Pass Probability Threshold", fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_hitting_csv(summary_rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "pass_level", "n_runs", "budget_to_pass"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def summarize_iso_budget(success_items: list[dict], iso_budgets: list[int]) -> list[dict]:
    rows = []
    for item in success_items:
        key = item["strategy"]["key"]
        x = np.array(item["x"], dtype=np.float64)
        p = np.array(item["mean"], dtype=np.float64)
        n_arr = np.array(item["n"], dtype=np.float64)
        n_runs = int(np.max(n_arr)) if n_arr.size > 0 else 0
        if x.size == 0:
            for b in iso_budgets:
                rows.append({"strategy": key, "added_budget": int(b), "success_prob": np.nan, "n_runs": n_runs})
            continue
        for b in iso_budgets:
            idx = int(np.argmin(np.abs(x - float(b))))
            rows.append(
                {
                    "strategy": key,
                    "added_budget": int(b),
                    "success_prob": float(p[idx]),
                    "n_runs": n_runs,
                }
            )
    return rows


def plot_iso_budget(summary_rows: list[dict], output_path: Path) -> None:
    import matplotlib.ticker as mticker

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    budgets = sorted(set(int(r["added_budget"]) for r in summary_rows))
    x = np.arange(len(budgets), dtype=np.float64)
    width = 0.24
    offsets = np.linspace(-width, width, num=len(STRATEGIES))
    for s_idx, s in enumerate(STRATEGIES):
        vals = []
        for b in budgets:
            row = next((r for r in summary_rows if r["strategy"] == s["key"] and int(r["added_budget"]) == b), None)
            vals.append(float(row["success_prob"]) if row is not None else np.nan)
        ax.bar(
            x + offsets[s_idx],
            vals,
            width=width * 0.92,
            color=s["color"],
            alpha=0.82,
            edgecolor=s["color"],
            linewidth=0.6,
            label=s["label"].replace(" Allocation", ""),
        )
    ax.set_xticks(x, [f"{b:.1e}" for b in budgets])
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Added Budget Checkpoints", fontweight="bold")
    ax.set_ylabel("Success@delta Probability", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_iso_csv(summary_rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "added_budget", "success_prob", "n_runs"])
        writer.writeheader()
        writer.writerows(summary_rows)


def summarize_auc(success_items: list[dict]) -> list[dict]:
    rows = []
    for item in success_items:
        x = np.array(item["x"], dtype=np.float64)
        y = np.array(item["mean"], dtype=np.float64)
        if x.size < 2:
            auc = np.nan
            x_min, x_max = np.nan, np.nan
        else:
            order = np.argsort(x)
            x_ord = x[order]
            y_ord = y[order]
            auc_raw = float(np.trapz(y_ord, x_ord))
            span = float(x_ord[-1] - x_ord[0])
            auc = float(auc_raw / span) if span > 0 else np.nan
            x_min, x_max = float(x_ord[0]), float(x_ord[-1])
        rows.append(
            {
                "strategy": item["strategy"]["key"],
                "auc_success": auc,
                "x_min": x_min,
                "x_max": x_max,
                "delta": float(item["delta"]),
            }
        )
    return rows


def plot_auc(summary_rows: list[dict], output_path: Path) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    labels, vals, colors = [], [], []
    for s in STRATEGIES:
        row = next((r for r in summary_rows if r["strategy"] == s["key"]), None)
        if row is None:
            continue
        labels.append(s["label"].replace(" Allocation", ""))
        vals.append(float(row["auc_success"]))
        colors.append(s["color"])
    x = np.arange(len(labels), dtype=np.float64)
    ax.bar(x, vals, color=colors, alpha=0.82, edgecolor=colors, linewidth=0.6, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized AUC of Success@delta", fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
                # On-plot reference avoids separate colorbar.
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


def write_auc_csv(summary_rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "delta", "auc_success", "x_min", "x_max"])
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    log_root = args.log_root if args.log_root.is_absolute() else (PROJECT_ROOT / args.log_root)
    simple_output = (
        args.simple_output
        if args.simple_output is not None
        else log_root / "mountaincar_simple_regret_curve.pdf"
    )
    simple_csv_output = (
        args.simple_csv_output
        if args.simple_csv_output is not None
        else log_root / "mountaincar_simple_regret_curve.csv"
    )
    success_output = (
        args.success_output
        if args.success_output is not None
        else log_root / "mountaincar_success_at_delta_curve.pdf"
    )
    success_csv_output = (
        args.success_csv_output
        if args.success_csv_output is not None
        else log_root / "mountaincar_success_at_delta_curve.csv"
    )
    hitting_output = (
        args.hitting_output
        if args.hitting_output is not None
        else log_root / "mountaincar_target_hitting_budget_bar.pdf"
    )
    hitting_csv_output = (
        args.hitting_csv_output
        if args.hitting_csv_output is not None
        else log_root / "mountaincar_target_hitting_budget_bar.csv"
    )
    iso_output = (
        args.iso_output
        if args.iso_output is not None
        else log_root / "mountaincar_iso_budget_success_bar.pdf"
    )
    iso_csv_output = (
        args.iso_csv_output
        if args.iso_csv_output is not None
        else log_root / "mountaincar_iso_budget_success_bar.csv"
    )
    auc_output = (
        args.auc_output
        if args.auc_output is not None
        else log_root / "mountaincar_success_auc_bar.pdf"
    )
    auc_csv_output = (
        args.auc_csv_output
        if args.auc_csv_output is not None
        else log_root / "mountaincar_success_auc_bar.csv"
    )
    contour_output = (
        args.contour_output
        if args.contour_output is not None
        else log_root / "mountaincar_success_contour.pdf"
    )
    if not simple_output.is_absolute():
        simple_output = PROJECT_ROOT / simple_output
    if not simple_csv_output.is_absolute():
        simple_csv_output = PROJECT_ROOT / simple_csv_output
    if not success_output.is_absolute():
        success_output = PROJECT_ROOT / success_output
    if not success_csv_output.is_absolute():
        success_csv_output = PROJECT_ROOT / success_csv_output
    if not hitting_output.is_absolute():
        hitting_output = PROJECT_ROOT / hitting_output
    if not hitting_csv_output.is_absolute():
        hitting_csv_output = PROJECT_ROOT / hitting_csv_output
    if not iso_output.is_absolute():
        iso_output = PROJECT_ROOT / iso_output
    if not iso_csv_output.is_absolute():
        iso_csv_output = PROJECT_ROOT / iso_csv_output
    if not auc_output.is_absolute():
        auc_output = PROJECT_ROOT / auc_output
    if not auc_csv_output.is_absolute():
        auc_csv_output = PROJECT_ROOT / auc_csv_output
    if not contour_output.is_absolute():
        contour_output = PROJECT_ROOT / contour_output

    deltas = _parse_deltas(args.deltas, oracle_std=args.oracle_std)
    hitting_regret = _parse_single_delta(args.hitting_regret, oracle_std=args.oracle_std)
    pass_levels = _parse_pass_levels(args.pass_levels)
    iso_budgets = _parse_iso_budgets(args.iso_budgets)
    contour_deltas = _parse_deltas(args.contour_deltas, oracle_std=args.oracle_std)
    run_dirs = list_run_dirs(log_root)
    step_regrets, available_runs = collect_step_regrets(
        run_dirs=run_dirs,
        oracle_best=float(args.oracle_mean),
        min_budget=args.min_budget,
        max_budget=args.max_budget,
    )
    simple_curves = []
    success_items = []
    for s in STRATEGIES:
        key = s["key"]
        x, m, lo, hi, n = aggregate_regret_curve(
            step_regrets[key],
            trim_count=int(args.trim_extremes_count),
            warmup_budget=int(args.warmup_budget),
        )
        if len(x) == 0:
            continue
        m_s = ema_smooth(m, weight=float(args.smooth))
        lo_s = ema_smooth(lo, weight=float(args.smooth))
        hi_s = ema_smooth(hi, weight=float(args.smooth))
        simple_curves.append(
            {
                "strategy": s,
                "x": x,
                "mean": m_s,
                "low": lo_s,
                "high": hi_s,
                "n": n,
            }
        )
        for d in deltas:
            x2, m2, lo2, hi2, n2 = aggregate_success_curve(
                step_regrets[key],
                delta=d,
                trim_count=int(args.trim_extremes_count),
                warmup_budget=int(args.warmup_budget),
            )
            m2_s = ema_smooth(m2, weight=float(args.smooth))
            lo2_s = ema_smooth(lo2, weight=float(args.smooth))
            hi2_s = ema_smooth(hi2, weight=float(args.smooth))
            success_items.append(
                {
                    "strategy": s,
                    "delta": float(d),
                    "x": x2,
                    "mean": m2_s,
                    "low": lo2_s,
                    "high": hi2_s,
                    "n": n2,
                }
            )

    if not simple_curves:
        raise FileNotFoundError("No valid strategy records found to build simple regret curve.")

    write_simple_csv(simple_curves, simple_csv_output)
    plot_simple_regret(simple_curves, simple_output)
    write_success_csv(success_items, success_csv_output)
    plot_success_delta(success_items, success_output)
    success_items_hitting = []
    for s in STRATEGIES:
        key = s["key"]
        xh, mh, loh, hih, nh = aggregate_success_curve(
            step_regrets[key],
            delta=hitting_regret,
            trim_count=int(args.trim_extremes_count),
            warmup_budget=int(args.warmup_budget),
        )
        if len(xh) == 0:
            continue
        success_items_hitting.append(
            {
                "strategy": s,
                "delta": float(hitting_regret),
                "x": xh,
                "mean": ema_smooth(mh, weight=float(args.smooth)),
                "low": ema_smooth(loh, weight=float(args.smooth)),
                "high": ema_smooth(hih, weight=float(args.smooth)),
                "n": nh,
            }
        )
    iso_summary = summarize_iso_budget(success_items_hitting, iso_budgets=iso_budgets)
    write_iso_csv(iso_summary, iso_csv_output)
    plot_iso_budget(iso_summary, iso_output)
    contour_items = []
    for s in STRATEGIES:
        key = s["key"]
        for d in contour_deltas:
            x2, m2, lo2, hi2, n2 = aggregate_success_curve(
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
    if contour_items:
        plot_success_contour(contour_items, contour_output)

    summary = []
    for s in STRATEGIES:
        key = s["key"]
        if available_runs.get(key, 0) > 0:
            summary.append(f"{key}: runs={available_runs[key]}")
    print(f"Simple regret plot saved to: {simple_output}")
    print(f"Simple regret csv saved to: {simple_csv_output}")
    print(f"Success@delta plot saved to: {success_output}")
    print(f"Success@delta csv saved to: {success_csv_output}")
    print(f"Iso-budget plot saved to: {iso_output}")
    print(f"Iso-budget csv saved to: {iso_csv_output}")
    if contour_items:
        print(
            "Success contour plots saved to: "
            f"{contour_output.with_name(f'{contour_output.stem}_uniform{contour_output.suffix}')}, "
            f"{contour_output.with_name(f'{contour_output.stem}_ocba{contour_output.suffix}')}, "
            f"{contour_output.with_name(f'{contour_output.stem}_adapted_ocba{contour_output.suffix}')}"
        )
    print(
        "Oracle config: "
        f"mean={args.oracle_mean}, std={args.oracle_std}, "
        f"deltas={','.join([f'{d:g}' for d in deltas])}, smooth={args.smooth}, "
        f"trim_extremes_count={args.trim_extremes_count}, warmup_budget={args.warmup_budget}, "
        f"hitting_regret={hitting_regret:g}, pass_levels={','.join([f'{x:g}' for x in pass_levels])}, "
        f"iso_budgets={','.join([str(x) for x in iso_budgets])}, "
        f"contour_deltas={','.join([f'{d:g}' for d in contour_deltas])}"
    )
    print("Run coverage: " + "; ".join(summary))


if __name__ == "__main__":
    main()
