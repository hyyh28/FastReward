import numpy as np

from src.strategies.base import AllocationStrategy


class UniformAllocationStrategy(AllocationStrategy):
    def allocate(self, means, variances, best_idx, delta_budget, update_unit, round_idx):
        k = len(means)
        total_chunks = int(delta_budget // update_unit)
        base = total_chunks // k
        extra = total_chunks % k
        chunks = np.full(k, base, dtype=int)
        for j in range(extra):
            chunks[(round_idx + j) % k] += 1
        return (chunks * update_unit).astype(int)
