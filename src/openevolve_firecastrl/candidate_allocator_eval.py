import math
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from firecastrl_env.wrappers import CellObservationWrapper, CustomRewardWrapper
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.scenario_factory import DetailedCellsObsWrapper, FireActionMaskWrapper, MaskableMonitor
from src.strategies.ocba import OCBAllocationStrategy
from src.strategies.uniform import UniformAllocationStrategy


RewardFn = Callable[[gym.Env, dict, dict], float]


@dataclass
class CandidateEvalConfig:
    strategy_name: str = "ocba"
    n_arms: int = 2
    n_envs: int = 1
    spray_radius: int = 5
    seed: int = 42
    total_budget: int = 220_000
    warmup_budget_per_arm: int = 50_000
    delta_budget: int = 15_000
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 3
    n_eval_episodes: int = 2

    @property
    def update_unit(self) -> int:
        return self.n_envs * self.n_steps


def _build_strategy(strategy_name: str):
    if strategy_name.lower() == "uniform":
        return UniformAllocationStrategy()
    if strategy_name.lower() == "ocba":
        return OCBAllocationStrategy()
    raise ValueError(f"Unsupported allocation strategy: {strategy_name}")


def _make_maskable_cnn_env(
    n_envs: int,
    seed: int,
    spray_radius: int,
    reward_fn: RewardFn | None,
) -> DummyVecEnv:
    def make_env(rank: int):
        def _init():
            env = gym.make("firecastrl/Wildfire-env0", render_mode=None)
            env = CellObservationWrapper(env)
            if reward_fn is not None:
                env = CustomRewardWrapper(env, reward_fn=reward_fn)
            env = FireActionMaskWrapper(env, r=spray_radius)
            env = DetailedCellsObsWrapper(env)
            env = MaskableMonitor(env)
            env.reset(seed=seed + rank)
            return env

        return _init

    return DummyVecEnv([make_env(rank) for rank in range(n_envs)])


class CandidateAllocatorEvaluator:
    """
    Evaluate one reward candidate with adaptive training resource allocation.

    The "arms" are independent training replicas with different random seeds.
    AllocationStrategy distributes extra timesteps to replicas each round.
    Candidate score is measured on true environment reward (without custom wrapper).
    """

    def __init__(self, config: CandidateEvalConfig):
        self.cfg = config
        self.strategy = _build_strategy(config.strategy_name)

    def _new_model(self, env: DummyVecEnv, seed: int) -> MaskablePPO:
        return MaskablePPO(
            policy=MaskableActorCriticCnnPolicy,
            env=env,
            learning_rate=1e-4,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            gamma=0.997,
            gae_lambda=0.98,
            ent_coef=0.01,
            policy_kwargs={"normalize_images": False},
            seed=seed,
            verbose=0,
        )

    def _eval_true_reward(self, model: MaskablePPO, seed: int) -> tuple[float, float]:
        eval_env = _make_maskable_cnn_env(
            n_envs=1,
            seed=seed,
            spray_radius=self.cfg.spray_radius,
            reward_fn=None,  # true env reward
        )
        try:
            mean_r, std_r = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=self.cfg.n_eval_episodes,
                deterministic=True,
                warn=False,
                use_masking=True,
            )
            return float(mean_r), float(std_r)
        finally:
            eval_env.close()

    def evaluate_reward_function(self, reward_fn: RewardFn, seed: int | None = None) -> dict[str, float]:
        run_seed = self.cfg.seed if seed is None else seed
        train_envs: list[DummyVecEnv] = []
        models: list[MaskablePPO] = []
        stats = [{"mean": -1e6, "var": 1.0} for _ in range(self.cfg.n_arms)]
        consumed_budget = 0

        try:
            for arm_idx in range(self.cfg.n_arms):
                arm_seed = run_seed + 1_000 * (arm_idx + 1)
                env = _make_maskable_cnn_env(
                    n_envs=self.cfg.n_envs,
                    seed=arm_seed,
                    spray_radius=self.cfg.spray_radius,
                    reward_fn=reward_fn,
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
                    mean_r, std_r = self._eval_true_reward(
                        models[arm_idx], seed=run_seed + 10_000 + round_idx * 100 + arm_idx
                    )
                    stats[arm_idx] = {"mean": mean_r, "var": max(std_r * std_r, 1.0)}
                    means[arm_idx] = stats[arm_idx]["mean"]
                    variances[arm_idx] = stats[arm_idx]["var"]

                best_idx = int(np.argmax(means))
                remaining = self.cfg.total_budget - consumed_budget
                delta_budget = int(min(self.cfg.delta_budget, remaining))
                if delta_budget < self.cfg.update_unit:
                    delta_budget = self.cfg.update_unit
                allocations = self.strategy.allocate(
                    means=means,
                    variances=variances,
                    best_idx=best_idx,
                    delta_budget=delta_budget,
                    update_unit=self.cfg.update_unit,
                    round_idx=round_idx,
                )
                allocations = np.asarray(allocations, dtype=int)
                for arm_idx, budget in enumerate(allocations):
                    if budget > 0:
                        models[arm_idx].learn(
                            total_timesteps=int(budget),
                            reset_num_timesteps=False,
                            log_interval=1,
                            progress_bar=True,
                        )

                consumed_budget += int(np.sum(allocations))
                round_idx += 1

            final_means = [item["mean"] for item in stats]
            final_vars = [item["var"] for item in stats]
            best_final_idx = int(np.argmax(final_means))
            best_mean = float(final_means[best_final_idx])
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
                    "valid": 0.0,
                }

            return {
                "combined_score": combined_score,
                "best_true_reward": best_mean,
                "avg_true_reward": avg_mean,
                "stability": stability,
                "budget_efficiency": budget_efficiency,
                "budget_used": float(consumed_budget),
                "valid": 1.0,
            }
        finally:
            for env in train_envs:
                env.close()
