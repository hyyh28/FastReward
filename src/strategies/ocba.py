import numpy as np

from src.strategies.base import AllocationStrategy


class OCBAllocationStrategy(AllocationStrategy):
    @staticmethod
    def _discretize(continuous_allocs, delta_budget, update_unit, fallback_idx):
        chunks = continuous_allocs // update_unit
        remainder = continuous_allocs - chunks * update_unit
        missing_chunks = int((delta_budget - np.sum(chunks * update_unit)) // update_unit)
        if missing_chunks > 0:
            for idx in np.argsort(remainder)[::-1][:missing_chunks]:
                chunks[idx] += 1

        allocations = chunks * update_unit
        alloc_remainder = int(delta_budget - np.sum(allocations))
        if alloc_remainder > 0:
            allocations[fallback_idx] += alloc_remainder
        return allocations.astype(int)

    def allocate(self, means, variances, best_idx, delta_budget, update_unit, round_idx):
        k = len(means)
        w = np.zeros(k, dtype=np.float64)
        safe_vars = np.maximum(variances, 1.0)

        for i in range(k):
            if i == best_idx:
                continue
            delta_ib = max(means[best_idx] - means[i], 1.0)
            w[i] = safe_vars[i] / (delta_ib ** 2)

        sum_sq_ratio = np.sum([(w[i] ** 2) / safe_vars[i] for i in range(k) if i != best_idx])
        w[best_idx] = np.sqrt(safe_vars[best_idx]) * np.sqrt(max(sum_sq_ratio, 1e-8))

        if np.sum(w) <= 0:
            continuous = np.full(k, delta_budget / k)
        else:
            continuous = (w / np.sum(w)) * delta_budget

        return self._discretize(continuous, delta_budget, update_unit, best_idx)
