def initial_reward_function(env, prev_state, curr_state) -> float:
    """
    Reward shaping: encourage progress toward goal, penalize distance,
    moderate outcome bonuses, and small step cost to encourage shorter episodes.
    """
    reward = 0.0
    # Progress toward goal (positive delta = closer)
    reward += 0.30 * float(curr_state["delta_manhattan"])
    # Penalize being far from goal
    reward -= 0.01 * float(curr_state["manhattan_to_goal"])

    # Moderate outcome bonuses (same scale as sparse true reward)
    if curr_state["on_goal"] > 0.5:
        reward += 1.0
    if curr_state["fell_in_hole"] > 0.5:
        reward -= 1.0

    # Per-step cost to discourage loitering (only during episode)
    if curr_state["terminated"] < 0.5:
        reward -= 0.01

    return float(reward)
