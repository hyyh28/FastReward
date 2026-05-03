def initial_reward_function(env, prev_state, curr_state) -> float:
    """
    Initial reward seed for FrozenLake OpenEvolve search.

    Expected state keys:
    - manhattan_to_goal
    - delta_manhattan
    - on_goal
    - fell_in_hole
    - terminated
    Additional available keys: row, col, nrows, ncols, goal_row, goal_col
    """
    reward = 0.0
    # Encourage progress toward goal (positive delta = closer)
    reward += 0.20 * float(curr_state["delta_manhattan"])
    # Penalize being far from goal
    reward -= 0.01 * float(curr_state["manhattan_to_goal"])

    # Heavier outcome penalties/bonuses
    if curr_state["on_goal"] > 0.5:
        reward += 2.0
    if curr_state["fell_in_hole"] > 0.5:
        reward -= 2.0

    # Small incentive to move (exploration) – penalize staying put
    row_changed = float(curr_state["row"] != prev_state["row"])
    col_changed = float(curr_state["col"] != prev_state["col"])
    if row_changed or col_changed:
        reward += 0.05
    else:
        reward -= 0.05

    # Per-step cost to discourage loitering
    if curr_state["terminated"] < 0.5:
        reward -= 0.02

    return float(reward)
