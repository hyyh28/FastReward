import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from firecastrl_env import

from src.envs.reward_wrapper import MountainCarRewardWrapper


class MountainCarScenarioFactory:
    def __init__(self, gamma: float):
        self.gamma = gamma

    def make_vec_env(self, shaping_type: str, n_envs: int, training: bool, obs_rms=None):
        def make_env():
            base = gym.make("MountainCar-v0")
            if shaping_type == "true_env":
                return Monitor(base)
            return Monitor(MountainCarRewardWrapper(base, shaping_type=shaping_type, gamma=self.gamma))

        vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=training)
        if obs_rms is not None:
            vec_env.obs_rms = obs_rms.copy()
        return vec_env
