import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import firecastrl_env
from firecastrl_env.wrappers import CustomRewardWrapper, CellObservationWrapper
from firecastrl_env.envs.environment.enums import FireState
from src.envs.reward_wrapper import MountainCarRewardWrapper
from src.envs.firecastrl_reward_init import initial_reward_function


class FireActionMaskWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, r: int = 5):
        super().__init__(env)
        self.r = r

    def action_masks(self) -> np.ndarray:
        mask = np.ones(5, dtype=bool)
        x, y = self.unwrapped.state["helicopter_coord"]
        fire = self.unwrapped.cell_state.fire_state
        h, w = fire.shape

        mask[0] = y < h - 1  # down
        mask[1] = y > 0      # up
        mask[2] = x > 0      # left
        mask[3] = x < w - 1  # right

        x0, x1 = max(0, x - self.r), min(w, x + self.r + 1)
        y0, y1 = max(0, y - self.r), min(h, y + self.r + 1)
        mask[4] = bool(np.any(fire[y0:y1, x0:x1] == FireState.Burning))
        return mask


class DetailedCellsObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        detailed_cells_space = env.observation_space["detailed_cells"]
        self.observation_space = gym.spaces.Box(
            low=detailed_cells_space.low,
            high=detailed_cells_space.high,
            shape=detailed_cells_space.shape,
            dtype=detailed_cells_space.dtype,
        )

    def observation(self, observation):
        return observation["detailed_cells"]


class MaskableMonitor(Monitor):
    def action_masks(self) -> np.ndarray:
        action_masks_fn = self.env.get_wrapper_attr("action_masks")
        return action_masks_fn()


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
            base = CellObservationWrapper(gym.make("firecastrl/Wildfire-env0", render_mode=None))
            if shaping_type == "true_env":
                return Monitor(base)
            reward_function = reward_fn or initial_reward_function
            return Monitor(CustomRewardWrapper(base, reward_fn=reward_function))

        vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=training)
        if obs_rms is not None:
            vec_env.obs_rms = obs_rms.copy()
        return vec_env

    def make_maskable_cnn_vec_env(
        self,
        n_envs: int = 8,
        seed: int = 42,
        spray_radius: int = 5,
        reward_fn=None,
        vec_env_type: str = "subproc",
    ):
        reward_function = reward_fn or initial_reward_function

        def make_env(rank: int):
            def _init():
                env = gym.make("firecastrl/Wildfire-env0", render_mode=None)
                env = CellObservationWrapper(env)
                env = CustomRewardWrapper(env, reward_fn=reward_function)
                env = FireActionMaskWrapper(env, r=spray_radius)
                env = DetailedCellsObsWrapper(env)
                env = MaskableMonitor(env)
                env.reset(seed=seed + rank)
                return env

            return _init

        env_fns = [make_env(rank) for rank in range(n_envs)]
        if vec_env_type.lower() == "subproc":
            return SubprocVecEnv(env_fns)
        if vec_env_type.lower() == "dummy":
            return DummyVecEnv(env_fns)
        raise ValueError(f"Unsupported vec_env_type: {vec_env_type}. Use 'subproc' or 'dummy'.")