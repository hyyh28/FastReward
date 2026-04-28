from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentConfig:
    candidates: List[str]
    total_budget: int = 3_600_000
    # 4 个候选时: 4 * 300_000 = 1_200_000 = total_budget / 3
    warmup_budget_per_candidate: int = 300_000
    delta_budget_per_round: int = 32_768
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
