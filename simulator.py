def run_episode(env, policy, horizon):
    """
    Run one episode of interaction between a policy and an environment.

    What happens at each time step:
    1. Ask the policy to choose an action
    2. Pass that action into the environment
    3. Get back a reward and the current regime
    4. Let the policy observe the result via update()
    5. Record everything

    Parameters
    ----------
    env : RegimeEnvironment
        Environment that produces rewards.
    policy : object
        Policy with select_action() and update(action, reward).
    horizon : int
        Number of steps to simulate.

    Returns
    -------
    dict
        Dictionary with:
        - actions
        - rewards
        - regimes
    """

    # Start the environment from time step 0.
    env.reset()

    # Store episode history in lists.
    actions = []
    rewards = []
    regimes = []

    # Main simulation loop.
    for _ in range(horizon):
        
        # Step 1: ask the policy which action it wants to take.
        action = policy.select_action()
        
        # Step 2: send that action to the environment.
        # The environment returns:
        # - reward: the realized reward after noise
        # - regime: the regime that was active at this time
        reward, regime = env.step(action)

        # Step 3: allow the policy to react to the reward.
        # For fixed policies this may do nothing.
        # For adaptive policies later, this is where learning happens.
        policy.update(action, reward)

        # Step 4: record what happened this step.
        actions.append(action)
        rewards.append(reward)
        regimes.append(regime)

    return {
        "actions": actions,
        "rewards": rewards,
        "regimes": regimes,
    }
