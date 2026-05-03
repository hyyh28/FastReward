def initial_reward_function(env, prev_state, curr_state) -> float:
    """
    Initial custom reward seed for future reward search (e.g. OpenEvolve).

    Expected state keys:
    - cells_burning
    - cells_burnt
    - helicopter_coord
    - quenched_cells
    """
    delta_burning = curr_state["cells_burning"] - prev_state["cells_burning"]
    reward = -2.0 * delta_burning
    reward += 10.0 * curr_state["quenched_cells"]
    return float(reward)
