class AllocationStrategy:
    def allocate(self, means, variances, best_idx, delta_budget, update_unit, round_idx):
        raise NotImplementedError
