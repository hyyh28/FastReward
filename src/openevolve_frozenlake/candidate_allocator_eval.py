import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.frozenlake_wrappers import make_frozenlake_vec_env
from src.strategies.ocba import OCBAllocationStrategy, ImprovedOCBAllocationStrategy
from src.strategies.uniform import UniformAllocationStrategy


RewardFn = Callable[[gym.Env, dict, dict], float]


@dataclass
class CandidateEvalConfig:
    strategy_name: str = "improved_ocba"
    n_arms: int = 4
    n_envs: int = 4
    seed: int = 42
    total_budget: int = 120_000
    warmup_budget_per_arm: int = 6_000
    delta_budget: int = 3_000
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    n_eval_episodes: int = 60
    map_name: str = "4x4"
    is_slippery: bool = True
    early_stop_enable: bool = True
    early_stop_z: float = 1.96
    early_stop_margin: float = 0.01
    min_rounds_before_stop: int = 2
    elimination_enable: bool = True
    elimination_z: float = 1.0
    min_active_arms: int = 2

    @property
    def update_unit(self) -> int:
        return self.n_envs * self.n_steps


def _build_strategy(strategy_name: str):
    normalized = strategy_name.lower()
    if normalized == "uniform":
        return UniformAllocationStrategy()
    if normalized == "ocba":
        return OCBAllocationStrategy()
    if normalized == "improved_ocba":
        return ImprovedOCBAllocationStrategy()
    raise ValueError(f"Unsupported allocation strategy: {strategy_name}")


class CandidateAllocatorEvaluator:
    def __init__(self, config: CandidateEvalConfig):
        self.cfg = config
        self.strategy = _build_strategy(config.strategy_name)

    def _new_model(self, env: DummyVecEnv, seed: int) -> PPO:
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            seed=seed,
            verbose=0,
        )

    def _eval_true_reward(self, model: PPO, seed: int) -> tuple[float, float]:
        eval_env = make_frozenlake_vec_env(
            n_envs=1,
            seed=seed,
            reward_fn=None,
            map_name=self.cfg.map_name,
            is_slippery=self.cfg.is_slippery,
        )
        try:
            mean_r, std_r = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=self.cfg.n_eval_episodes,
                deterministic=True,
                warn=False,
            )
            return float(mean_r), float(std_r)
        finally:
            eval_env.close()

    def evaluate_reward_function(self, reward_fn: RewardFn, seed: int | None = None) -> dict[str, float]:
        run_seed = self.cfg.seed if seed is None else seed
        train_envs: list[DummyVecEnv] = []
        models: list[PPO] = []
        stats = [{"mean": -1e6, "var": 1.0} for _ in range(self.cfg.n_arms)]
        active_mask = np.ones(self.cfg.n_arms, dtype=bool)
        consumed_budget = 0
        early_stop_triggered = False
        round_records: list[dict] = []

        try:
            for arm_idx in range(self.cfg.n_arms):
                arm_seed = run_seed + 1_000 * (arm_idx + 1)
                env = make_frozenlake_vec_env(
                    n_envs=self.cfg.n_envs,
                    seed=arm_seed,
                    reward_fn=reward_fn,
                    map_name=self.cfg.map_name,
                    is_slippery=self.cfg.is_slippery,
                )
                model = self._new_model(env=env, seed=arm_seed)
                train_envs.append(env)
                models.append(model)

            warmup = np.full(self.cfg.n_arms, self.cfg.warmup_budget_per_arm, dtype=int)
            for idx, budget in enumerate(warmup):
                if budget > 0:
                    models[idx].learn(
                        total_timesteps=int(budget),
                        reset_num_timesteps=False,
                        log_interval=1,
                        progress_bar=True,
                    )
            consumed_budget += int(np.sum(warmup))

            round_idx = 0
            while consumed_budget < self.cfg.total_budget:
                means = np.zeros(self.cfg.n_arms, dtype=np.float64)
                variances = np.ones(self.cfg.n_arms, dtype=np.float64)
                for arm_idx in range(self.cfg.n_arms):
                    if not active_mask[arm_idx]:
                        means[arm_idx] = -1e9
                        variances[arm_idx] = 1e6
                        continue
                    mean_r, std_r = self._eval_true_reward(
                        models[arm_idx], seed=run_seed + 10_000 + round_idx * 100 + arm_idx
                    )
                    stats[arm_idx] = {"mean": mean_r, "var": max(std_r * std_r, 1e-6)}
                    means[arm_idx] = stats[arm_idx]["mean"]
                    variances[arm_idx] = stats[arm_idx]["var"]

                best_idx = int(np.argmax(means))
                remaining = self.cfg.total_budget - consumed_budget
                if self.cfg.early_stop_enable and round_idx >= self.cfg.min_rounds_before_stop:
                    active_indices = np.where(active_mask)[0]
                    if active_indices.size <= 1:
                        early_stop_triggered = True
                        break
                    active_means = means[active_indices]
                    order = active_indices[np.argsort(active_means)[::-1]]
                    best = int(order[0])
                    second_best = int(order[1])
                    eval_n = max(self.cfg.n_eval_episodes, 1)
                    best_se = np.sqrt(max(variances[best], 1e-8) / eval_n)
                    second_se = np.sqrt(max(variances[second_best], 1e-8) / eval_n)
                    best_lcb = means[best] - self.cfg.early_stop_z * best_se
                    second_ucb = means[second_best] + self.cfg.early_stop_z * second_se
                    if best_lcb > second_ucb + self.cfg.early_stop_margin:
                        early_stop_triggered = True
                        round_records.append(
                            {
                                "round_idx": int(round_idx),
                                "consumed_budget_before_round": int(consumed_budget),
                                "strategy_name": self.cfg.strategy_name,
                                "means": [float(x) for x in means.tolist()],
                                "variances": [float(x) for x in variances.tolist()],
                                "active_mask": [bool(x) for x in active_mask.tolist()],
                                "allocations": [0 for _ in range(self.cfg.n_arms)],
                                "best_idx": int(best),
                                "second_best_idx": int(second_best),
                                "best_lcb": float(best_lcb),
                                "second_ucb": float(second_ucb),
                                "early_stop": True,
                                "eliminated_arms": [],
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        break

                eliminated_arms: list[int] = []
                if self.cfg.elimination_enable:
                    active_indices = np.where(active_mask)[0]
                    if active_indices.size > self.cfg.min_active_arms:
                        eval_n = max(self.cfg.n_eval_episodes, 1)
                        best_mean = means[best_idx]
                        best_se = np.sqrt(max(variances[best_idx], 1e-8) / eval_n)
                        best_lcb = best_mean - self.cfg.elimination_z * best_se
                        eliminated = 0
                        for arm_idx in active_indices:
                            if arm_idx == best_idx:
                                continue
                            arm_se = np.sqrt(max(variances[arm_idx], 1e-8) / eval_n)
                            arm_ucb = means[arm_idx] + self.cfg.elimination_z * arm_se
                            if arm_ucb < best_lcb and np.count_nonzero(active_mask) - eliminated > self.cfg.min_active_arms:
                                active_mask[arm_idx] = False
                                eliminated += 1
                                eliminated_arms.append(int(arm_idx))

                max_alloc = (remaining // self.cfg.update_unit) * self.cfg.update_unit
                if max_alloc <= 0:
                    break
                delta_budget = int(min(self.cfg.delta_budget, max_alloc))
                allocations = self.strategy.allocate(
                    means=means,
                    variances=variances,
                    best_idx=best_idx,
                    delta_budget=delta_budget,
                    update_unit=self.cfg.update_unit,
                    round_idx=round_idx,
                )
                allocations = np.asarray(allocations, dtype=int)
                allocations[~active_mask] = 0
                alloc_sum = int(np.sum(allocations))
                if alloc_sum > max_alloc and alloc_sum > 0:
                    scale = max_alloc / alloc_sum
                    allocations = (allocations.astype(np.float64) * scale).astype(int)
                    allocations = (allocations // self.cfg.update_unit) * self.cfg.update_unit
                    remainder = max_alloc - int(np.sum(allocations))
                    if remainder > 0:
                        allocations[best_idx] += remainder
                for arm_idx, budget in enumerate(allocations):
                    if budget > 0:
                        models[arm_idx].learn(
                            total_timesteps=int(budget),
                            reset_num_timesteps=False,
                            log_interval=1,
                            progress_bar=True,
                        )

                round_records.append(
                    {
                        "round_idx": int(round_idx),
                        "consumed_budget_before_round": int(consumed_budget),
                        "consumed_budget_after_round": int(consumed_budget + np.sum(allocations)),
                        "strategy_name": self.cfg.strategy_name,
                        "means": [float(x) for x in means.tolist()],
                        "variances": [float(x) for x in variances.tolist()],
                        "active_mask": [bool(x) for x in active_mask.tolist()],
                        "allocations": [int(x) for x in allocations.tolist()],
                        "best_idx": int(best_idx),
                        "early_stop": False,
                        "eliminated_arms": eliminated_arms,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                consumed_budget += int(np.sum(allocations))
                round_idx += 1

            final_means = [item["mean"] for item in stats]
            final_vars = [item["var"] for item in stats]
            best_mean = float(np.max(final_means))
            avg_mean = float(np.mean(final_means))
            stability = float(1.0 / (1.0 + np.mean(final_vars)))
            budget_efficiency = float(self.cfg.total_budget / max(consumed_budget, 1))
            combined_score = float(best_mean + 0.25 * avg_mean + 10.0 * stability)

            if not math.isfinite(combined_score):
                return {
                    "combined_score": -1e9,
                    "best_true_reward": -1e9,
                    "avg_true_reward": -1e9,
                    "stability": 0.0,
                    "budget_used": float(consumed_budget),
                    "round_count": float(round_idx),
                    "active_arms_final": float(np.count_nonzero(active_mask)),
                    "early_stop_triggered": 1.0 if early_stop_triggered else 0.0,
                    "valid": 0.0,
                }

            return {
                "combined_score": combined_score,
                "best_true_reward": best_mean,
                "avg_true_reward": avg_mean,
                "stability": stability,
                "budget_efficiency": budget_efficiency,
                "budget_used": float(consumed_budget),
                "round_count": float(round_idx),
                "active_arms_final": float(np.count_nonzero(active_mask)),
                "early_stop_triggered": 1.0 if early_stop_triggered else 0.0,
                "valid": 1.0,
            }
        finally:
            trace_path = os.getenv("FROZENLAKE_ROUND_TRACE_PATH", "").strip()
            trace_tag = os.getenv("FROZENLAKE_TRACE_TAG", "").strip()
            if trace_path and round_records:
                out_path = Path(trace_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("a", encoding="utf-8") as f:
                    for rec in round_records:
                        payload = dict(rec)
                        payload["trace_tag"] = trace_tag
                        payload["seed"] = int(run_seed)
                        payload["n_eval_episodes"] = int(self.cfg.n_eval_episodes)
                        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
            for env in train_envs:
                env.close()
