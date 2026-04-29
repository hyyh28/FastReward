from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


RewardFn = Callable[[gym.Env, dict, dict], float]


class FrozenLakePrevCurrWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        desc = self.unwrapped.desc
        self.nrows, self.ncols = desc.shape
        goal_idx = np.argwhere(desc == b"G")
        if goal_idx.size == 0:
            raise ValueError("FrozenLake map has no goal tile 'G'")
        self.goal_row, self.goal_col = int(goal_idx[0][0]), int(goal_idx[0][1])
        self._prev_state: dict | None = None

    def _decode_state(self, obs: int, terminated: bool, truncated: bool) -> dict:
        state = int(obs)
        row, col = divmod(state, self.ncols)
        tile = self.unwrapped.desc[row, col]
        manhattan = abs(row - self.goal_row) + abs(col - self.goal_col)
        on_goal = float(tile == b"G")
        fell_in_hole = float(tile == b"H" and terminated)
        return {
            "state": float(state),
            "row": float(row),
            "col": float(col),
            "nrows": float(self.nrows),
            "ncols": float(self.ncols),
            "goal_row": float(self.goal_row),
            "goal_col": float(self.goal_col),
            "manhattan_to_goal": float(manhattan),
            "on_goal": on_goal,
            "fell_in_hole": fell_in_hole,
            "terminated": float(terminated),
            "truncated": float(truncated),
            "delta_manhattan": 0.0,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self._decode_state(obs=int(obs), terminated=False, truncated=False)
        self._prev_state = dict(state)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        curr_state = self._decode_state(obs=int(obs), terminated=terminated, truncated=truncated)
        prev_state = self._prev_state if self._prev_state is not None else dict(curr_state)
        curr_state["delta_manhattan"] = float(
            prev_state["manhattan_to_goal"] - curr_state["manhattan_to_goal"]
        )
        shaped_reward = self.reward_fn(self.env, prev_state, curr_state)
        self._prev_state = dict(curr_state)
        return obs, float(shaped_reward), terminated, truncated, info


def make_frozenlake_vec_env(
    n_envs: int,
    seed: int,
    reward_fn: RewardFn | None,
    map_name: str = "4x4",
    is_slippery: bool = True,
) -> DummyVecEnv:
    def make_env(rank: int):
        def _init():
            env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
            if reward_fn is not None:
                env = FrozenLakePrevCurrWrapper(env, reward_fn=reward_fn)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        return _init

    return DummyVecEnv([make_env(rank) for rank in range(n_envs)])
