import numpy as np


class PerformanceMonitor:
    def __init__(self, candidates):
        self.candidates = candidates
        self.budgets_history = []
        self.best_reward_history = []
        self.true_reward_history = {c: [] for c in candidates}
        self.shaped_reward_history = {c: [] for c in candidates}
        self.round_alloc_history = {c: [] for c in candidates}
        self.cumulative_allocation = np.zeros(len(candidates), dtype=int)
        self.round_records = []

    def log_round(self, consumed_budget, stats, shaped_means, allocations):
        self.budgets_history.append(consumed_budget)
        self.best_reward_history.append(max(s["mean"] for s in stats))

        per_candidate = {}
        for k, c in enumerate(self.candidates):
            self.true_reward_history[c].append(stats[k]["mean"])
            self.shaped_reward_history[c].append(shaped_means[k])
            self.round_alloc_history[c].append(int(allocations[k]))
            self.cumulative_allocation[k] += int(allocations[k])
            per_candidate[c] = {
                "true_mean_return": float(stats[k]["mean"]),
                "true_return_var": float(stats[k]["var"]),
                "shaped_mean_return": float(shaped_means[k]),
                "allocation_this_round": int(allocations[k]),
                "allocation_cumulative": int(self.cumulative_allocation[k]),
            }

        self.round_records.append(
            {
                "round_index": len(self.budgets_history) - 1,
                "budget_consumed": int(consumed_budget),
                "best_true_return": float(max(s["mean"] for s in stats)),
                "per_candidate": per_candidate,
            }
        )

    def print_round(self, tag, consumed_budget, stats, allocations, shaped_means):
        print(
            f"[{tag:7s}] Budget:{consumed_budget:7d} | "
            f"Scores:{[round(s['mean'], 1) for s in stats]} | "
            f"Round Allocation:{dict(zip(self.candidates, allocations.tolist()))} | "
            f"Cumulative Allocation:{dict(zip(self.candidates, self.cumulative_allocation.tolist()))} | "
            f"Mean Shaped Return per Episode:{dict(zip(self.candidates, [round(x, 1) for x in shaped_means]))}"
        )

    def export(self):
        return {
            "candidates": self.candidates,
            "budgets_history": self.budgets_history,
            "best_reward_history": self.best_reward_history,
            "per_candidate_reward_history": self.true_reward_history,
            "shaped_return_history": self.shaped_reward_history,
            "per_round_allocation_history": self.round_alloc_history,
            "round_records": self.round_records,
        }
