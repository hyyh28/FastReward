import gymnasium as gym
import numpy as np
from collections import deque


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

        # ----------- 距离图 -----------
        self.safe_dist_map = self._bfs_distance(avoid_hole=True)
        self.hole_dist_map = self._compute_hole_distance()

        self.max_safe_dist = np.max(self.safe_dist_map[self.safe_dist_map < np.inf])
        self.max_hole_dist = np.max(self.hole_dist_map[self.hole_dist_map < np.inf])
        self.max_dist = self.nrows + self.ncols - 2

        self.prev_obs = None

        self.beta = {
            "safe_distance": 0.5,
            "risk_aware": 0.8,
            "bad": 0.5,
        }.get(shaping_type, 0.0)

        self.deceptive_scale = {
            "deceptive": 0.1,
        }.get(shaping_type, 0.0)

    # =========================================================
    # 工具函数
    # =========================================================

    def _decode(self, obs: int):
        return divmod(int(obs), self.ncols)

    def _neighbors(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.nrows and 0 <= nc < self.ncols:
                yield nr, nc

    def _bfs_distance(self, avoid_hole=True):
        desc = self.unwrapped.desc
        dist = np.full((self.nrows, self.ncols), np.inf)
        q = deque()

        dist[self.goal_row, self.goal_col] = 0
        q.append((self.goal_row, self.goal_col))

        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors(r, c):
                if avoid_hole and desc[nr, nc] == b"H":
                    continue
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

        return dist

    def _compute_hole_distance(self):
        desc = self.unwrapped.desc
        dist = np.full((self.nrows, self.ncols), np.inf)
        q = deque()

        for r in range(self.nrows):
            for c in range(self.ncols):
                if desc[r, c] == b"H":
                    dist[r, c] = 0
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors(r, c):
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

        return dist

    def _phi(self, obs: int):
        row, col = self._decode(obs)
        tile = self.unwrapped.desc[row, col]

        if self.shaping_type == "safe_distance":
            d = self.safe_dist_map[row, col]
            if np.isinf(d):
                return -1.0
            return 1.0 - d / max(self.max_safe_dist, 1)

        elif self.shaping_type == "risk_aware":
            dist_goal = abs(row - self.goal_row) + abs(col - self.goal_col)
            dist_hole = self.hole_dist_map[row, col]

            return (
                1.0 - dist_goal / max(self.max_dist, 1)
                + 0.4 * (dist_hole / max(self.max_hole_dist, 1))
            )

        elif self.shaping_type == "deceptive":
            dist_goal = abs(row - self.goal_row) + abs(col - self.goal_col)
            norm_goal = 1.0 - dist_goal / max(self.max_dist, 1)

            dist_hole = self.hole_dist_map[row, col]

            if dist_goal <= 2:
                hole_bonus = 0.6 * (1.0 - dist_hole / max(self.max_hole_dist, 1))
            else:
                hole_bonus = 0.0
            if tile == b"H":
                hole_bonus += 0.2
            return norm_goal + hole_bonus

        elif self.shaping_type == "bad":
            dist_goal = abs(row - self.goal_row) + abs(col - self.goal_col)
            return -dist_goal / max(self.max_dist, 1)

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

        # 🔥 Step 1: 原始 reward 放大
        reward = 100.0 * reward

        # 🔵 PBRS
        if self.shaping_type in ["safe_distance", "risk_aware", "bad"]:
            shaping = self.beta * (
                self.gamma * self._phi(obs_i) - self._phi(self.prev_obs)
            )
            reward = reward + shaping

        # 🔴 NON-PBRS（deceptive）
        elif "deceptive" in self.shaping_type:
            phi_val = self._phi(obs_i)
            scale = self.deceptive_scale

            reward = 0.01 * reward + scale * phi_val

        # bad 额外惩罚（同步放大）
        if self.shaping_type == "bad":
            reward -= 2.0

        self.prev_obs = obs_i

        return obs, float(reward), terminated, truncated, info