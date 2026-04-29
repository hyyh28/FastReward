import gymnasium as gym


class FrozenLakeRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, shaping_type: str, gamma: float):
        super().__init__(env)
        self.shaping_type = shaping_type
        self.gamma = gamma
        self.ncols = int(self.unwrapped.ncol)
        self.nrows = int(self.unwrapped.nrow)
        desc = self.unwrapped.desc
        goal_pos = None
        for r in range(self.nrows):
            for c in range(self.ncols):
                if desc[r, c] == b"G":
                    goal_pos = (r, c)
                    break
            if goal_pos is not None:
                break
        if goal_pos is None:
            raise ValueError("FrozenLake map has no goal tile 'G'")
        self.goal_row, self.goal_col = goal_pos
        self.prev_obs = None
        self.beta = {
            "baseline": 0.0,
            "assist_distance": 0.12,
            "assist_progress": 0.20,
            "deceptive_hole_bias": 0.18,
        }[shaping_type]

    def _decode(self, obs: int) -> tuple[int, int]:
        row, col = divmod(int(obs), self.ncols)
        return row, col

    def _phi(self, obs: int) -> float:
        if self.shaping_type == "baseline":
            return 0.0
        row, col = self._decode(obs)
        manhattan = abs(row - self.goal_row) + abs(col - self.goal_col)
        norm_dist = 1.0 - (manhattan / max(self.nrows + self.ncols - 2, 1))

        if self.shaping_type == "assist_distance":
            return float(norm_dist)
        if self.shaping_type == "assist_progress":
            edge_bonus = 1.0 if (row in (0, self.nrows - 1) or col in (0, self.ncols - 1)) else 0.0
            return float(norm_dist + 0.25 * edge_bonus)
        if self.shaping_type == "deceptive_hole_bias":
            tile = self.unwrapped.desc[row, col]
            return float(0.8 if tile == b"H" else -0.2 * norm_dist)
        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = int(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_i = int(obs)
        if self.prev_obs is None:
            self.prev_obs = obs_i
        shaping = self.beta * (self.gamma * self._phi(obs_i) - self._phi(self.prev_obs))
        self.prev_obs = obs_i
        return obs, float(reward + shaping), terminated, truncated, info
