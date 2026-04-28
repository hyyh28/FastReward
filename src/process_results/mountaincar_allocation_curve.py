import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


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


def aggregate_curves(curves):
    # 按 budget(steps) 对齐，支持不同 run 的长度不一致
    all_steps = sorted({s for steps, _, _ in curves for s in steps})
    step_to_values = {s: [] for s in all_steps}
    for steps, values, _ in curves:
        for s, v in zip(steps, values):
            step_to_values[s].append(float(v))

    x = []
    mean = []
    low = []
    high = []
    n = []
    for s in all_steps:
        vals = np.array(step_to_values[s], dtype=np.float64)
        if vals.size == 0:
            continue
        m = float(np.mean(vals))
        if vals.size > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
            ci = 1.96 * sem
        else:
            ci = 0.0
        x.append(int(s))
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


def build_aggregate_plot(run_dirs, output_path: Path, smooth_weight=0.6):
    uniform_curves = load_strategy_curves(run_dirs, "uniform_rounds.jsonl")
    ocba_curves = load_strategy_curves(run_dirs, "ocba_rounds.jsonl")

    x_u, m_u, l_u, h_u, n_u = aggregate_curves(uniform_curves)
    x_o, m_o, l_o, h_o, n_o = aggregate_curves(ocba_curves)
    m_u_s = ema_smooth(m_u, weight=smooth_weight)
    l_u_s = ema_smooth(l_u, weight=smooth_weight)
    h_u_s = ema_smooth(h_u, weight=smooth_weight)
    m_o_s = ema_smooth(m_o, weight=smooth_weight)
    l_o_s = ema_smooth(l_o, weight=smooth_weight)
    h_o_s = ema_smooth(h_o, weight=smooth_weight)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_u,
        m_u_s,
        label="Uniform Allocation (mean)",
        linestyle="--",
        linewidth=2,
        color="tab:blue",
    )
    plt.fill_between(x_u, l_u_s, h_u_s, color="tab:blue", alpha=0.18, label="Uniform 95% CI")
    plt.plot(
        x_o,
        m_o_s,
        label="OCBA Allocation (mean)",
        linewidth=2.5,
        color="tab:red",
    )
    plt.fill_between(x_o, l_o_s, h_o_s, color="tab:red", alpha=0.18, label="OCBA 95% CI")
    plt.xlabel("Total Training Budget (Steps)")
    plt.ylabel("Best Episodic Return")
    plt.title("MountainCar: Best Return vs Total Steps (Mean ± 95% CI)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return {
        "uniform_runs": len(uniform_curves),
        "ocba_runs": len(ocba_curves),
        "uniform_points": int(len(x_u)),
        "ocba_points": int(len(x_o)),
        "smooth_weight": float(smooth_weight),
        "uniform_min_n": int(np.min(n_u)) if len(n_u) > 0 else 0,
        "uniform_max_n": int(np.max(n_u)) if len(n_u) > 0 else 0,
        "ocba_min_n": int(np.min(n_o)) if len(n_o) > 0 else 0,
        "ocba_max_n": int(np.max(n_o)) if len(n_o) > 0 else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MountainCar OCBA/Uniform best return vs steps curve from logged jsonl files."
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
        help="Output image path. Default: <log_root>/ocba_vs_uniform_best_return_ci.png",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="EMA smoothing weight, TensorBoard-style (default: 0.6).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_root = args.log_root if args.log_root.is_absolute() else (PROJECT_ROOT / args.log_root)
    run_dirs = list_run_dirs(log_root)
    output = args.output if args.output is not None else log_root / "ocba_vs_uniform_best_return_ci.png"
    if output is not None and not output.is_absolute():
        output = PROJECT_ROOT / output

    meta = build_aggregate_plot(run_dirs=run_dirs, output_path=output, smooth_weight=args.smooth)
    print(f"Plot saved to: {output}")
    print(
        "Aggregation summary: "
        f"smooth={meta['smooth_weight']}, "
        f"uniform_runs={meta['uniform_runs']}, ocba_runs={meta['ocba_runs']}, "
        f"uniform_points={meta['uniform_points']} (n range {meta['uniform_min_n']}~{meta['uniform_max_n']}), "
        f"ocba_points={meta['ocba_points']} (n range {meta['ocba_min_n']}~{meta['ocba_max_n']})"
    )


if __name__ == "__main__":
    main()
