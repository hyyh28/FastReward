import numpy as np

from src.strategies.base import AllocationStrategy

# Floor σ² at this fraction of max σ² in the current batch (≈1% relative error scale).
_VARIANCE_RELATIVE_FLOOR_FRAC = 0.01


def _gap_epsilon(means: np.ndarray) -> float:
    """Minimum OCBA mean-gap so divisions stay stable without collapsing distinct arms.

    A fixed floor like ``max(gap, 1.0)`` erases real gaps when returns live on ~[0, 1];
    poor arms then get weights similar to near-best arms. Use a tiny scale-aware eps instead.
    """
    m = np.asarray(means, dtype=np.float64)
    spread = float(np.ptp(m))
    mag = float(np.max(np.abs(m))) if m.size else 0.0
    scale_ref = max(spread, mag, 1e-12)
    return max(scale_ref * 1e-10, 1e-12)


def _relative_variance_floor(variances, floor_frac: float = _VARIANCE_RELATIVE_FLOOR_FRAC) -> np.ndarray:
    """Clip nonnegative variances so none fall below ``floor_frac * max(variances)``.

    Using a fixed absolute floor (e.g. 1.0) ignores the reward/evaluation scale; a relative
    floor keeps OCBA sensitivity to observed dispersion while bounding numerical blow-ups.
    """
    v = np.maximum(np.asarray(variances, dtype=np.float64), 0.0)
    vmax = float(np.max(v)) if v.size else 0.0
    floor = vmax * floor_frac if vmax > 0 else 1e-12
    return np.maximum(v, floor)


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
        safe_vars = _relative_variance_floor(variances)
        eps_gap = _gap_epsilon(means)

        for i in range(k):
            if i == best_idx:
                continue
            delta_ib = max(float(means[best_idx] - means[i]), eps_gap)
            w[i] = safe_vars[i] / (delta_ib ** 2)

        sum_sq_ratio = np.sum([(w[i] ** 2) / safe_vars[i] for i in range(k) if i != best_idx])
        w[best_idx] = np.sqrt(safe_vars[best_idx]) * np.sqrt(max(sum_sq_ratio, 1e-8))

        if np.sum(w) <= 0:
            continuous = np.full(k, delta_budget / k)
        else:
            continuous = (w / np.sum(w)) * delta_budget

        return self._discretize(continuous, delta_budget, update_unit, best_idx)


class ImprovedOCBAllocationStrategy(OCBAllocationStrategy):
    def __init__(self, eps_var: float = 1e-6, eps_gap: float = 1e-6):
        self.eps_var = float(max(eps_var, 1e-12))
        self.eps_gap = float(max(eps_gap, 1e-12))

    def allocate(self, means, variances, best_idx, delta_budget, update_unit, round_idx):
        k = len(means)
        w = np.zeros(k, dtype=np.float64)
        safe_vars = np.maximum(_relative_variance_floor(variances), self.eps_var)

        for i in range(k):
            if i == best_idx:
                continue
            delta_ib = max(float(means[best_idx] - means[i]), self.eps_gap)
            w[i] = safe_vars[i] / (delta_ib ** 2)

        sum_sq_ratio = np.sum([(w[i] ** 2) / safe_vars[i] for i in range(k) if i != best_idx])
        w[best_idx] = np.sqrt(safe_vars[best_idx]) * np.sqrt(max(sum_sq_ratio, self.eps_var))

        if np.sum(w) <= 0:
            continuous = np.full(k, delta_budget / max(k, 1))
        else:
            continuous = (w / np.sum(w)) * delta_budget

        return self._discretize(continuous, delta_budget, update_unit, best_idx)


class RewardAdaptedOCBAAllocationStrategy(AllocationStrategy):
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

        # ----------- Step 1: 安全处理 -----------
        safe_vars = _relative_variance_floor(variances)

        # ----------- Step 2: 找 second best -----------
        sorted_idx = np.argsort(means)[::-1]
        b = best_idx
        second = sorted_idx[1] if sorted_idx[0] == b else sorted_idx[0]

        # ----------- Step 3: 构造 boundary c -----------
        s_b = np.sqrt(safe_vars[b])
        s_2 = np.sqrt(safe_vars[second])

        denom = s_b + s_2
        if denom <= 1e-12:
            c = (means[b] + means[second]) / 2.0
        else:
            c = (s_2 * means[b] + s_b * means[second]) / denom

        # ----------- Step 4: 计算 d_k -----------
        d = np.zeros(k, dtype=np.float64)
        eps = 1e-8

        for i in range(k):
            if i == b:
                d[i] = max(means[b] - c, eps)
            else:
                d[i] = max(c - means[i], eps)

        # ----------- Step 5: OCBA-style 权重 -----------
        w = safe_vars / (d ** 2)

        # ----------- Step 6: fallback（全零情况）-----------
        if np.sum(w) <= 1e-12:
            continuous = np.full(k, delta_budget / k)
        else:
            continuous = (w / np.sum(w)) * delta_budget

        # ----------- Step 7: 离散化 -----------
        return self._discretize(continuous, delta_budget, update_unit, best_idx)