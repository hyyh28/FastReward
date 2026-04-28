import gymnasium as gym
import numpy as np


class MountainCarRewardWrapper(gym.Wrapper):
    def __init__(self, env, shaping_type: str, gamma: float):
        super().__init__(env)
        self.shaping_type = shaping_type
        self.gamma = gamma
        self.prev_obs = None
        self.beta = {
            "baseline": 0.00,
            "assist_pos_vel": 0.14,
            "assist_energy_gate": 0.18,
            "deceptive_left": 0.20,
        }[shaping_type]

    @staticmethod
    def _phi_base(obs):
        pos, vel = obs
        pos_term = (pos + 1.2) / 1.8
        vel_energy = (vel * vel) / (0.07 * 0.07)
        vel_abs = abs(vel) / 0.07
        return pos_term, vel_abs, vel_energy

    def _phi(self, obs):
        if self.shaping_type == "baseline":
            return 0.0
        pos_term, vel_abs, vel_energy = self._phi_base(obs)
        if self.shaping_type == "assist_pos_vel":
            return pos_term + 0.45 * vel_abs
        if self.shaping_type == "assist_energy_gate":
            gate = np.tanh(7.5 * (obs[0] - 0.15))
            return pos_term + 0.55 * vel_energy + 0.35 * gate
        if self.shaping_type == "deceptive_left":
            left_bias = np.tanh(8.0 * (-obs[0] - 0.25))
            return 0.85 * left_bias - 0.20 * pos_term
        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = np.array(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_arr = np.array(obs, dtype=np.float32)
        if self.prev_obs is None:
            self.prev_obs = obs_arr.copy()

        shaping = self.beta * (self.gamma * self._phi(obs_arr) - self._phi(self.prev_obs))
        if self.shaping_type == "deceptive_left" and int(action) == 2:
            shaping -= 0.03

        self.prev_obs = obs_arr
        return obs, reward + shaping, terminated, truncated, info
