import matplotlib.pyplot as plt


class ResultReporter:
    @staticmethod
    def plot_strategy_comparison(results_uniform, results_ocba):
        plt.figure(figsize=(10, 6))
        plt.plot(
            results_uniform["budgets_history"],
            results_uniform["best_reward_history"],
            label="Uniform Allocation",
            linestyle="--",
            linewidth=2,
        )
        plt.plot(
            results_ocba["budgets_history"],
            results_ocba["best_reward_history"],
            label="OCBA Allocation",
            linewidth=2.5,
        )
        plt.xlabel("Total Training Budget (Steps)")
        plt.ylabel("Best Identified Policy Return (True Env)")
        plt.title("MountainCar: Uniform vs OCBA")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("mountaincar_budget_allocation_comparison.png", dpi=300)

    @staticmethod
    def plot_candidate_true_rewards(results_ocba):
        plt.figure(figsize=(10, 6))
        for c in results_ocba["candidates"]:
            plt.plot(
                results_ocba["budgets_history"],
                results_ocba["per_candidate_reward_history"][c],
                label=c,
                linewidth=2,
            )
        plt.xlabel("Total Training Budget (Steps)")
        plt.ylabel("Policy Return on True Env")
        plt.title("MountainCar: Reward Candidate Performance (OCBA)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("mountaincar_reward_candidates_ocba.png", dpi=300)

    @staticmethod
    def plot_candidate_shaped_rewards(results_ocba):
        plt.figure(figsize=(10, 6))
        for c in results_ocba["candidates"]:
            plt.plot(
                results_ocba["budgets_history"],
                results_ocba["shaped_return_history"][c],
                label=c,
                linewidth=2,
            )
        plt.xlabel("Total Training Budget (Steps)")
        plt.ylabel("Mean Episodic Shaped Return")
        plt.title("MountainCar: Mean Episodic Shaped Return by Reward Candidate (OCBA)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("mountaincar_cumulative_reward_ocba.png", dpi=300)
