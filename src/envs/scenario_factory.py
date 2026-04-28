import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import firecastrl_env
from firecastrl_env.wrappers import CustomRewardWrapper, CellObservationWrapper
from src.envs.reward_wrapper import MountainCarRewardWrapper
from src.envs.firecastrl_reward_init import initial_reward_function



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


class FirecastrlFactory:
    def __init__(self, gamma: float):
        self.gamma = gamma

    def make_vec_env(
        self,
        shaping_type: str,
        n_envs: int,
        training: bool,
        obs_rms=None,
        reward_fn=None,
    ):
        def make_env():
            base = gym.make("firecastrl/Wildfire-env0", render_mode=None)
            base = CellObservationWrapper(base, properties=['ignition_time', 'fire_state', 'elevation', 'position'], remove_basic_cells=True)
            if shaping_type == "true_env":
                return Monitor(base)
            fn = reward_fn or initial_reward_function
            return Monitor(CustomRewardWrapper(base, reward_fn=fn))

        vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=training)
        if obs_rms is not None:
            vec_env.obs_rms = obs_rms.copy()
        return vec_env