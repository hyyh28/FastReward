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
            "Plot probability of selecting the oracle-best reward candidate "
            "as total training budget increases."
        )
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
        help="Output png path. Default: <log_root>/best_candidate_hit_probability.png",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Output csv path. Default: <log_root>/best_candidate_hit_probability.csv",
    )
    parser.add_argument(
        "--top2-output",
        type=Path,
        default=None,
        help="Output png path for top-2 hit probability. Default: <log_root>/best_candidate_top2_hit_probability.png",
    )
    parser.add_argument(
        "--top2-csv-output",
        type=Path,
        default=None,
        help="Output csv path for top-2 hit probability. Default: <log_root>/best_candidate_top2_hit_probability.csv",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=3_000_000,
        help="Maximum budget(step) shown on curve.",
    )
    return parser.parse_args()


def _safe_float(value, default: float = -1e9) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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


def _argmax_candidate(per_candidate: dict) -> str:
    best_name = ""
    best_value = -1e18
    for name in sorted(per_candidate.keys()):
        v = _safe_float((per_candidate.get(name) or {}).get("true_mean_return"), -1e9)
        if v > best_value:
            best_value = v
            best_name = str(name)
    return best_name


def _topk_candidates(per_candidate: dict, k: int) -> list[str]:
    scored = []
    for name, info in per_candidate.items():
        scored.append((str(name), _safe_float((info or {}).get("true_mean_return"), -1e9)))
    scored_sorted = sorted(scored, key=lambda x: (-x[1], x[0]))
    return [name for name, _ in scored_sorted[: max(1, int(k))]]


def _oracle_best_candidate(strategy_records: dict[str, list[dict]]) -> str:
    candidate_to_best_final = {}
    for records in strategy_records.values():
        if not records:
            continue
        final = records[-1].get("per_candidate", {}) or {}
        for c_name, c_info in final.items():
            cur = _safe_float((c_info or {}).get("true_mean_return"), -1e9)
            prev = candidate_to_best_final.get(c_name, -1e18)
            if cur > prev:
                candidate_to_best_final[c_name] = cur
    if not candidate_to_best_final:
        return ""
    return sorted(candidate_to_best_final.items(), key=lambda x: (-x[1], x[0]))[0][0]


def collect_hits_by_step(
    run_dirs: list[Path], max_budget: int | None = None
) -> tuple[dict[str, dict[int, list[int]]], dict[str, dict[int, list[int]]], dict[str, int]]:
    strategy_step_hits: dict[str, dict[int, list[int]]] = {s["key"]: {} for s in STRATEGIES}
    strategy_step_hits_top2: dict[str, dict[int, list[int]]] = {s["key"]: {} for s in STRATEGIES}
    available_runs: dict[str, int] = {s["key"]: 0 for s in STRATEGIES}

    for run_dir in run_dirs:
        strategy_records: dict[str, list[dict]] = {}
        for s in STRATEGIES:
            p = run_dir / s["filename"]
            if p.exists():
                strategy_records[s["key"]] = read_round_records(p)
        if not strategy_records:
            continue

        oracle = _oracle_best_candidate(strategy_records)
        if not oracle:
            continue

        for s in STRATEGIES:
            key = s["key"]
            records = strategy_records.get(key, [])
            if not records:
                continue
            available_runs[key] += 1
            for rec in records:
                step = int(rec.get("budget_consumed", -1))
                if step < 0:
                    continue
                if max_budget is not None and step > int(max_budget):
                    continue
                chosen = _argmax_candidate(rec.get("per_candidate", {}) or {})
                chosen_top2 = _topk_candidates(rec.get("per_candidate", {}) or {}, k=2)
                hit = int(chosen == oracle)
                hit_top2 = int(oracle in chosen_top2)
                strategy_step_hits[key].setdefault(step, []).append(hit)
                strategy_step_hits_top2[key].setdefault(step, []).append(hit_top2)

    return strategy_step_hits, strategy_step_hits_top2, available_runs


def aggregate_hit_curve(step_hits: dict[int, list[int]]):
    steps = sorted(step_hits.keys())
    x = []
    mean = []
    low = []
    high = []
    n = []
    for step in steps:
        vals = np.array(step_hits[step], dtype=np.float64)
        if vals.size == 0:
            continue
        m = float(np.mean(vals))
        if vals.size > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
            ci = 1.96 * sem
        else:
            ci = 0.0
        x.append(step)
        mean.append(m)
        low.append(max(0.0, m - ci))
        high.append(min(1.0, m + ci))
        n.append(int(vals.size))
    return np.array(x), np.array(mean), np.array(low), np.array(high), np.array(n)


def write_curve_csv(curves: list[dict], csv_path: Path) -> None:
    rows = []
    for item in curves:
        key = item["strategy"]["key"]
        for step, prob, lo, hi, n in zip(item["x"], item["mean"], item["low"], item["high"], item["n"]):
            rows.append(
                {
                    "strategy": key,
                    "budget_consumed": int(step),
                    "hit_probability": float(prob),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "n_runs": int(n),
                }
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "budget_consumed", "hit_probability", "ci95_low", "ci95_high", "n_runs"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(curves: list[dict], output_path: Path, ylabel: str) -> None:
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

    ax.set_xlabel("Total Training Budget (Steps)", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
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


def main() -> None:
    args = parse_args()
    log_root = args.log_root if args.log_root.is_absolute() else (PROJECT_ROOT / args.log_root)
    output = args.output if args.output is not None else log_root / "best_candidate_hit_probability.png"
    csv_output = (
        args.csv_output
        if args.csv_output is not None
        else log_root / "best_candidate_hit_probability.csv"
    )
    top2_output = (
        args.top2_output
        if args.top2_output is not None
        else log_root / "best_candidate_top2_hit_probability.png"
    )
    top2_csv_output = (
        args.top2_csv_output
        if args.top2_csv_output is not None
        else log_root / "best_candidate_top2_hit_probability.csv"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if not csv_output.is_absolute():
        csv_output = PROJECT_ROOT / csv_output
    if not top2_output.is_absolute():
        top2_output = PROJECT_ROOT / top2_output
    if not top2_csv_output.is_absolute():
        top2_csv_output = PROJECT_ROOT / top2_csv_output

    run_dirs = list_run_dirs(log_root)
    step_hits, step_hits_top2, available_runs = collect_hits_by_step(run_dirs, max_budget=args.max_budget)

    curves = []
    curves_top2 = []
    for s in STRATEGIES:
        key = s["key"]
        x, mean, low, high, n = aggregate_hit_curve(step_hits[key])
        if len(x) == 0:
            pass
        else:
            curves.append({"strategy": s, "x": x, "mean": mean, "low": low, "high": high, "n": n})
        x2, mean2, low2, high2, n2 = aggregate_hit_curve(step_hits_top2[key])
        if len(x2) == 0:
            continue
        curves_top2.append({"strategy": s, "x": x2, "mean": mean2, "low": low2, "high": high2, "n": n2})

    if not curves:
        raise FileNotFoundError("No valid strategy records found to plot.")
    if not curves_top2:
        raise FileNotFoundError("No valid strategy records found to plot top-2 hit probability.")

    write_curve_csv(curves, csv_output)
    plot_curves(curves, output, ylabel="P(select oracle-best candidate)")
    write_curve_csv(curves_top2, top2_csv_output)
    plot_curves(curves_top2, top2_output, ylabel="P(oracle-best in selected top-2)")

    parts = []
    for s in STRATEGIES:
        key = s["key"]
        if key not in available_runs or available_runs[key] <= 0:
            continue
        last_prob = np.nan
        for item in curves:
            if item["strategy"]["key"] == key and len(item["mean"]) > 0:
                last_prob = float(item["mean"][-1])
        parts.append(f"{key}: runs={available_runs[key]}, last_prob={last_prob:.3f}")

    print(f"Plot saved to: {output}")
    print(f"CSV saved to: {csv_output}")
    print(f"Top-2 plot saved to: {top2_output}")
    print(f"Top-2 CSV saved to: {top2_csv_output}")
    print("Summary: " + "; ".join(parts))


if __name__ == "__main__":
    main()
