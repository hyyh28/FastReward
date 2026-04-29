import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from src.config import FrozenLakeConfig
from src.envs.scenario_factory import FrozenLakeScenarioFactory
from src.monitoring.performance_monitor import PerformanceMonitor


class FrozenLakeExperimentRunner:
    def __init__(self, config: FrozenLakeConfig, allocation_strategy):
        self.cfg = config
        self.strategy = allocation_strategy
        self.factory = FrozenLakeScenarioFactory(gamma=config.gamma)
        self.monitor = PerformanceMonitor(config.candidates)

        self.envs = [
            self.factory.make_vec_env(
                shaping_type=c,
                n_envs=config.n_envs,
                map_name=config.map_name,
                is_slippery=config.is_slippery,
            )
            for c in config.candidates
        ]
        self.models = [PPO("MlpPolicy", env, **self._ppo_kwargs()) for env in self.envs]
        self.stats = [{"mean": 0.0, "var": 1.0} for _ in config.candidates]
        self.consumed_budget = 0

    def _ppo_kwargs(self):
        return dict(
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            learning_rate=self.cfg.learning_rate,
            ent_coef=self.cfg.ent_coef,
            clip_range=0.2,
            n_epochs=self.cfg.n_epochs,
            vf_coef=0.7,
            max_grad_norm=0.5,
            verbose=0,
        )

    def _eval_true(self, idx):
        eval_env = self.factory.make_vec_env(
            shaping_type="true_env",
            n_envs=1,
            map_name=self.cfg.map_name,
            is_slippery=self.cfg.is_slippery,
        )
        mean_r, std_r = evaluate_policy(
            self.models[idx], eval_env, n_eval_episodes=self.cfg.n_eval_episodes_true, warn=False
        )
        eval_env.close()
        return float(mean_r), float(std_r)

    def _eval_shaped(self, idx):
        eval_env = self.factory.make_vec_env(
            shaping_type=self.cfg.candidates[idx],
            n_envs=1,
            map_name=self.cfg.map_name,
            is_slippery=self.cfg.is_slippery,
        )
        mean_r, _ = evaluate_policy(
            self.models[idx], eval_env, n_eval_episodes=self.cfg.n_eval_episodes_shaped, warn=False
        )
        eval_env.close()
        return float(mean_r)

    def _evaluate_all(self):
        shaped_means = np.zeros(len(self.cfg.candidates), dtype=np.float64)
        for k in range(len(self.cfg.candidates)):
            mean_r, std_r = self._eval_true(k)
            self.stats[k] = {"mean": mean_r, "var": max(std_r**2, 1.0)}
            shaped_means[k] = self._eval_shaped(k)
        return shaped_means

    def _train_with_allocations(self, allocations):
        for k in range(len(self.cfg.candidates)):
            if allocations[k] > 0:
                self.models[k].learn(total_timesteps=int(allocations[k]), reset_num_timesteps=False)

    def run(self, strategy_name: str):
        print(f"\n{'=' * 70}\n启动 [{strategy_name.upper()}] FrozenLake 预算分配实验\n{'=' * 70}")
        k = len(self.cfg.candidates)

        warm_alloc = np.full(k, self.cfg.warmup_budget_per_candidate, dtype=int)
        self._train_with_allocations(warm_alloc)
        self.consumed_budget += int(np.sum(warm_alloc))
        shaped_means = self._evaluate_all()
        self.monitor.log_round(self.consumed_budget, self.stats, shaped_means, warm_alloc)
        self.monitor.print_round("WARMUP", self.consumed_budget, self.stats, warm_alloc, shaped_means)

        while self.consumed_budget < self.cfg.total_budget:
            means = np.array([s["mean"] for s in self.stats], dtype=np.float64)
            variances = np.array([s["var"] for s in self.stats], dtype=np.float64)
            best_idx = int(np.argmax(means))
            round_idx = len(self.monitor.budgets_history) - 1
            allocations = self.strategy.allocate(
                means=means,
                variances=variances,
                best_idx=best_idx,
                delta_budget=self.cfg.delta_budget_per_round,
                update_unit=self.cfg.update_unit,
                round_idx=round_idx,
            )
            self._train_with_allocations(allocations)
            self.consumed_budget += int(self.cfg.delta_budget_per_round)
            shaped_means = self._evaluate_all()
            self.monitor.log_round(self.consumed_budget, self.stats, shaped_means, allocations)
            self.monitor.print_round(
                strategy_name.upper(), self.consumed_budget, self.stats, allocations, shaped_means
            )

        return self.monitor.export()
