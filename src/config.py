from abc import ABC
from dataclasses import dataclass
from typing import List


class ExperimentConfig(ABC):
    """Abstract base config. Concrete env configs define all fields."""


@dataclass
class MountainCarConfig(ExperimentConfig):
    candidates: List[str]
    total_budget: int = 3_600_000
    warmup_budget_per_candidate: int = 200_000
    delta_budget_per_round: int = 32_768
    # 4 个候选时: 4 * 200_000 = 800_000 作为 warmup
    n_envs: int = 16
    n_steps: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.98
    n_epochs: int = 4
    ent_coef: float = 0.0
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_eval_episodes_true: int = 30
    n_eval_episodes_shaped: int = 20

    @property
    def update_unit(self) -> int:
        return self.n_envs * self.n_steps


@dataclass
class FrozenLakeConfig(ExperimentConfig):
    candidates: List[str]
    total_budget: int = 800_000
    warmup_budget_per_candidate: int = 80_000
    delta_budget_per_round: int = 20_480
    n_envs: int = 8
    n_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 4
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_eval_episodes_true: int = 50
    n_eval_episodes_shaped: int = 30
    map_name: str = "8x8"
    is_slippery: bool = True

    @property
    def update_unit(self) -> int:
        return self.n_envs * self.n_steps
