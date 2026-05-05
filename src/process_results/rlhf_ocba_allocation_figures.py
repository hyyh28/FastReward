"""
Plot RLHF reward-model screening logs (single JSON per strategy).

Expects ``src/logs/RLHF/*.json`` with fields:
``warmup_per_candidate``, ``delta_per_round``, ``means`` (T×K), ``allocations`` (T×K),
optional ``reward_models`` for candidate labels.

Example:

    python -m src.process_results.rlhf_ocba_allocation_figures \\
        --output-dir figures/rlhf

Optional cumulative comparisons (you choose candidate indices, 0-based):

    python -m src.process_results.rlhf_ocba_allocation_figures \\
        --output-dir figures/rlhf \\
        --output-d figures/rlhf/cumulative_counts.png \\
        --output-e figures/rlhf/cumulative_fraction.png \\
        --cumulative-candidate-indices 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Strategy styling aligned with mountaincar_regret_success_curve.py
STRATEGIES = [
    {
        "key": "uniform",
        "json_filename": "uniform.json",
        "label": "Uniform Allocation",
        "color": "#B08A72",
        "fill": "#D9CBC0",
        "linestyle": "--",
        "linewidth": 1.4,
    },
    {
        "key": "ocba",
        "json_filename": "ocba.json",
        "label": "OCBA Allocation",
        "color": "#6E8FA8",
        "fill": "#CAD6DF",
        "linestyle": "-",
        "linewidth": 1.4,
    },
    {
        "key": "adapted_ocba",
        "json_filename": "reward_adapted_ocba.json",
        "label": "Adaptive OCBA Allocation",
        "color": "#4F7089",
        "fill": "#B9CAD6",
        "linestyle": "-.",
        "linewidth": 1.4,
    },
]

# Candidate colors for stackplot / heatmap rows (distinct from strategy palette)
CANDIDATE_COLORS = ["#8C9B9F", "#C4A574", "#7A8F7E", "#9B8CB8", "#B5838D"]


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLHF OCBA figures: best mean vs budget, stacks, heatmaps.")
    p.add_argument(
        "--rlhf-dir",
        type=Path,
        default=SRC_DIR / "logs" / "RLHF",
        help="Directory containing uniform.json, ocba.json, reward_adapted_ocba.json",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Write three PNGs with default names.")
    p.add_argument("--output-a", type=Path, default=None, help="Fig A: best mean vs cumulative evaluations.")
    p.add_argument("--output-b", type=Path, default=None, help="Fig B: cumulative allocation stackplots.")
    p.add_argument("--output-c", type=Path, default=None, help="Fig C: allocation heatmaps.")
    p.add_argument(
        "--output-d",
        type=Path,
        default=None,
        help=(
            "Fig D (optional): cumulative samples N_k(t) vs cumulative evaluations, "
            "one subplot per index from --cumulative-candidate-indices; three strategy lines each."
        ),
    )
    p.add_argument(
        "--output-e",
        type=Path,
        default=None,
        help=(
            "Fig E (optional): cumulative share N_k(t)/B(t) vs cumulative evaluations, "
            "same layout as D; B(t) is total cumulative evaluations after round t."
        ),
    )
    p.add_argument(
        "--cumulative-candidate-indices",
        type=str,
        default=None,
        help=(
            "Comma-separated 0-based candidate indices (e.g. 0 or 0,2). "
            "Required if --output-d or --output-e is set. No default; script does not pick a 'best' arm."
        ),
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--best-so-far",
        action="store_true",
        help="Use max-so-far of per-round best mean instead of instantaneous max.",
    )
    p.add_argument(
        "--candidate-labels",
        type=str,
        default=None,
        help="Comma-separated candidate labels (length K). Overrides reward_models basename.",
    )
    return p.parse_args()


def _labels_from_reward_models(paths: list[str] | None, k: int, override: str | None) -> list[str]:
    if override:
        parts = [x.strip() for x in override.split(",") if x.strip()]
        if len(parts) != k:
            raise ValueError(f"--candidate-labels expects {k} labels, got {len(parts)}")
        return parts
    if paths and len(paths) == k:
        out = []
        for p in paths:
            stem = Path(p.replace("\\", "/")).name
            if len(stem) > 32:
                stem = stem[:29] + "..."
            out.append(stem)
        return out
    return [f"Candidate {i}" for i in range(k)]


def load_rlhf_run(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing RLHF log: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    means = np.asarray(raw["means"], dtype=np.float64)
    allocs = np.asarray(raw["allocations"], dtype=np.float64)
    if means.shape != allocs.shape:
        raise ValueError(f"{path}: means shape {means.shape} != allocations {allocs.shape}")
    if means.ndim != 2:
        raise ValueError(f"{path}: expected 2D means/allocations")
    warmup = int(raw["warmup_per_candidate"])
    delta = int(raw["delta_per_round"])
    row_sums = allocs.sum(axis=1)
    if not np.allclose(row_sums, float(delta)):
        raise ValueError(f"{path}: allocations row sums {row_sums} not all equal to delta_per_round={delta}")
    return {
        "path": path,
        "raw": raw,
        "means": means,
        "allocations": allocs,
        "warmup_per_candidate": warmup,
        "delta_per_round": delta,
        "seed": raw.get("seed"),
        "reward_models": raw.get("reward_models"),
    }


def validate_aligned(runs: list[dict]) -> None:
    k0 = runs[0]["means"].shape[1]
    t0 = runs[0]["means"].shape[0]
    w0 = runs[0]["warmup_per_candidate"]
    d0 = runs[0]["delta_per_round"]
    for r in runs[1:]:
        if r["means"].shape != (t0, k0):
            raise ValueError(
                f"Shape mismatch: {r['path'].name} has {r['means'].shape}, "
                f"expected {(t0, k0)} like {runs[0]['path'].name}"
            )
        if r["warmup_per_candidate"] != w0 or r["delta_per_round"] != d0:
            raise ValueError(f"warmup/delta mismatch vs {runs[0]['path'].name}")


def cumulative_budget_axis(allocations: np.ndarray, k: int, warmup_per_candidate: int) -> np.ndarray:
    """Budget after each adaptive round (includes warmup)."""
    warm_total = k * warmup_per_candidate
    per_round = allocations.sum(axis=1)
    return warm_total + np.cumsum(per_round)


def cumulative_samples_per_candidate(allocations: np.ndarray, warmup_per_candidate: int) -> np.ndarray:
    """Shape (T, K): N_j[t] after round t."""
    return warmup_per_candidate + np.cumsum(allocations, axis=0)


def best_mean_curve(means: np.ndarray, best_so_far: bool) -> np.ndarray:
    j = np.max(means, axis=1)
    if best_so_far:
        return np.maximum.accumulate(j)
    return j


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.output_dir is not None:
        base = args.output_dir
        return (
            base / "rlhf_best_mean_vs_budget.png",
            base / "rlhf_cumulative_allocation_stack.png",
            base / "rlhf_allocation_heatmap.png",
        )
    if args.output_a and args.output_b and args.output_c:
        return args.output_a, args.output_b, args.output_c
    raise SystemExit("Provide --output-dir or all of --output-a, --output-b, --output-c")


def parse_candidate_indices(text: str | None, k: int) -> list[int]:
    if text is None or not str(text).strip():
        raise ValueError("Candidate indices string is empty.")
    out: list[int] = []
    seen: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= k:
            raise ValueError(f"Candidate index {idx} out of range [0, {k - 1}].")
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    if not out:
        raise ValueError("No valid candidate indices parsed.")
    return out


def plot_figure_a(
    runs: list[dict],
    strategies: list[dict],
    budget: np.ndarray,
    best_so_far: bool,
    output_path: Path,
    dpi: int,
) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for run, s in zip(runs, strategies):
        y = best_mean_curve(run["means"], best_so_far)
        ax.plot(
            budget,
            y,
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=s["linewidth"],
            label=s["label"],
        )
    ax.set_xlabel("Cumulative evaluations", fontweight="bold")
    ylab = "Best-so-far performance (max mean)" if best_so_far else "Best estimated performance (max mean)"
    ax.set_ylabel(ylab, fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_figure_b(
    runs: list[dict],
    strategies: list[dict],
    budget: np.ndarray,
    cand_labels: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    plt = _setup_matplotlib()
    k = len(cand_labels)
    colors = [CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)] for i in range(k)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8), sharey=True)
    for ax, run, s in zip(axes, runs, strategies):
        cum = cumulative_samples_per_candidate(run["allocations"], run["warmup_per_candidate"])
        ax.stackplot(budget, [cum[:, j] for j in range(k)], labels=cand_labels, colors=colors, alpha=0.92)
        ax.set_title(s["label"], fontweight="bold", fontsize=8)
        ax.set_xlabel("Cumulative evaluations", fontweight="bold")
        ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
        ax.tick_params(direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        seed = run.get("seed")
        if seed is not None:
            ax.text(
                0.02,
                0.98,
                f"seed={seed}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                color="#444444",
            )
    axes[0].set_ylabel("Cumulative samples per candidate", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(k, 4),
            frameon=False,
            prop={"weight": "bold", "size": 7},
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_figure_c(
    runs: list[dict],
    strategies: list[dict],
    cand_labels: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    plt = _setup_matplotlib()
    vmax = float(max(np.max(r["allocations"]) for r in runs))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), constrained_layout=True)
    ims = []
    for ax, run, s in zip(axes, runs, strategies):
        mat = run["allocations"].T  # K x T
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=vmax,
            cmap="Blues",
            interpolation="nearest",
        )
        ims.append(im)
        ax.set_title(s["label"], fontweight="bold", fontsize=8)
        ax.set_xlabel("Round index", fontweight="bold")
        ax.set_yticks(range(len(cand_labels)))
        ax.set_yticklabels(cand_labels, fontsize=6)
        ax.tick_params(direction="out")
        seed = run.get("seed")
        if seed is not None:
            ax.text(
                0.98,
                0.02,
                f"seed={seed}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="#444444",
            )
    axes[0].set_ylabel("Candidate", fontweight="bold")
    cbar = fig.colorbar(ims[0], ax=list(axes), shrink=0.78, pad=0.02)
    cbar.set_label("Allocated samples\nthis round", fontweight="bold", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_figure_d_or_e(
    runs: list[dict],
    strategies: list[dict],
    budget: np.ndarray,
    cand_labels: list[str],
    indices: list[int],
    output_path: Path,
    dpi: int,
    *,
    fraction: bool,
) -> None:
    """Per candidate index: three strategy lines vs cumulative budget (counts or share)."""
    plt = _setup_matplotlib()
    n_p = len(indices)
    fig_h = 2.5 * n_p
    fig, axes = plt.subplots(n_p, 1, figsize=(3.5, min(fig_h, 8.0)), sharex=True)
    if n_p == 1:
        axes = [axes]
    for row, k_idx in enumerate(indices):
        ax = axes[row]
        for run, s in zip(runs, strategies):
            cum = cumulative_samples_per_candidate(run["allocations"], run["warmup_per_candidate"])
            y = cum[:, k_idx] / budget if fraction else cum[:, k_idx]
            ax.plot(
                budget,
                y,
                color=s["color"],
                linestyle=s["linestyle"],
                linewidth=s["linewidth"],
                label=s["label"],
            )
        ax.set_ylabel("Cumulative share" if fraction else "Cumulative samples", fontweight="bold")
        ax.set_title(cand_labels[k_idx], fontweight="bold", fontsize=8)
        ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.5)
        ax.tick_params(direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if fraction:
            ax.set_ylim(0.0, 1.0)
    axes[-1].set_xlabel("Cumulative evaluations", fontweight="bold")
    axes[0].legend(loc="best", frameon=False, handlelength=2.0, prop={"weight": "bold", "size": 7})
    fig.suptitle(
        "Cumulative share of total evaluations" if fraction else "Cumulative samples on candidate",
        fontweight="bold",
        fontsize=8,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_a, out_b, out_c = resolve_outputs(args)
    need_de = args.output_d is not None or args.output_e is not None
    if need_de and args.cumulative_candidate_indices is None:
        raise SystemExit("--cumulative-candidate-indices is required when using --output-d or --output-e")

    strategies = STRATEGIES
    runs = []
    for s in strategies:
        path = args.rlhf_dir / s["json_filename"]
        runs.append(load_rlhf_run(path))
    validate_aligned(runs)

    k = runs[0]["means"].shape[1]
    cand_labels = _labels_from_reward_models(runs[0]["reward_models"], k, args.candidate_labels)

    budget = cumulative_budget_axis(runs[0]["allocations"], k, runs[0]["warmup_per_candidate"])

    plot_figure_a(runs, strategies, budget, args.best_so_far, out_a, args.dpi)
    plot_figure_b(runs, strategies, budget, cand_labels, out_b, args.dpi)
    plot_figure_c(runs, strategies, cand_labels, out_c, args.dpi)

    written = [out_a, out_b, out_c]
    if need_de:
        indices = parse_candidate_indices(args.cumulative_candidate_indices, k)
        if args.output_d is not None:
            plot_figure_d_or_e(
                runs, strategies, budget, cand_labels, indices, args.output_d, args.dpi, fraction=False
            )
            written.append(args.output_d)
        if args.output_e is not None:
            plot_figure_d_or_e(
                runs, strategies, budget, cand_labels, indices, args.output_e, args.dpi, fraction=True
            )
            written.append(args.output_e)

    print("Wrote:\n  " + "\n  ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
