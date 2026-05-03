def initial_reward_function(env, prev_state, curr_state) -> float:
    """
    Initial reward seed for FrozenLake OpenEvolve search.

    Expected state keys:
    - manhattan_to_goal
    - delta_manhattan
    - on_goal
    - fell_in_hole
    - terminated
    """
    reward = 0.0
    reward += 0.20 * float(curr_state["delta_manhattan"])
    reward -= 0.005 * float(curr_state["manhattan_to_goal"])

    if curr_state["on_goal"] > 0.5:
        reward += 1.0
    if curr_state["fell_in_hole"] > 0.5:
        reward -= 1.0
    if curr_state["terminated"] < 0.5:
        reward -= 0.01
    return float(reward)
