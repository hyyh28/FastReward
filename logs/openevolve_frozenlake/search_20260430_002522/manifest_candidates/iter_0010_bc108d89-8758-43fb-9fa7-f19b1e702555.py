def initial_reward_function(env, prev_state, curr_state) -> float:
    """
    Reward shaping: encourage progress toward goal, penalize distance,
    heavily reward reaching goal, heavily penalize falling into holes,
    small step cost, and incentivize movement to avoid staying still.
    """
    reward = 0.0
    # Progress toward goal (positive delta = closer)
    reward += 0.30 * float(curr_state["delta_manhattan"])
    # Penalize being far from goal
    reward -= 0.01 * float(curr_state["manhattan_to_goal"])

    # Strong outcome bonuses/penalties
    if curr_state["on_goal"] > 0.5:
        reward += 2.0
    if curr_state["fell_in_hole"] > 0.5:
        reward -= 2.0

    # Movement incentive: reward changing row or col, penalize staying still
    # Use row/col from state if available (they are part of the environment)
    if "row" in curr_state and "row" in prev_state and "col" in prev_state:
        row_changed = float(curr_state["row"] != prev_state["row"])
        col_changed = float(curr_state["col"] != prev_state["col"])
        if row_changed or col_changed:
            reward += 0.05
        else:
            reward -= 0.05

    # Per-step cost to discourage loafing
    if curr_state["terminated"] < 0.5:
        reward -= 0.005

    return float(reward)
