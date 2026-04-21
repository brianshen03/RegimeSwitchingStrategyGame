from environment import RegimeEnvironment
from policies import StaticPolicy, RandomPolicy, RoundRobinPolicy
from simulator import run_episode
from metrics import total_reward, average_reward


def run_baseline_experiment():
    """
    Run a Week 5 baseline experiment under fixed conditions.

    What this function does:
    - defines one fixed environment
    - creates several baseline policies
    - runs one simulation episode for each policy
    - computes simple performance metrics
    - stores everything in a results dictionary

    """

    # Define the true reward structure for each regime.
    #
    # Rows = regimes
    # Columns = actions
    #
    # For Week 5, we are using fixed conditions, so we will hold the regime constant during each run by setting fixed_regime=0 later.
    means = [
        [1.0, 0.5, 0.2],  # Regime 0: action 0 is best
        [0.2, 1.0, 0.5],  # Regime 1: action 1 is best
        [0.5, 0.2, 1.0],  # Regime 2: action 2 is best
    ]

    # Number of decision steps in one simulation run.
    horizon = 300

    # Reward noise level.
    sigma = 0.1

    # For Week 5, hold the environment fixed in one regime for the whole run.
    fixed_regime = 0

    # This dictionary defines which baseline policies to test.
    #
    # Each value is a FUNCTION (lambda) that creates a NEW policy instance.
    # We do this instead of storing policy objects directly because:
    # - it guarantees each run starts with a fresh policy
    # - it is safer once policies begin to have internal state
    policy_factories = {
        "Static_A": lambda: StaticPolicy(action=0),
        "Static_B": lambda: StaticPolicy(action=1),
        "Static_C": lambda: StaticPolicy(action=2),
        "Random": lambda: RandomPolicy(num_actions=3),
        "RoundRobin": lambda: RoundRobinPolicy(num_actions=3),
    }

    # Store all final results here.
    results = {}

    # Run one episode for each policy.
    for policy_name, make_policy in policy_factories.items():
        # Create a fresh environment for this policy's run.
        env = RegimeEnvironment(
            means=means,
            sigma=sigma,
            regime_length=100,   # not used when fixed_regime is set
            fixed_regime=fixed_regime
        )

        # Create a fresh policy instance for this run.
        policy = make_policy()

        # Run one episode and collect raw simulation data.
        episode = run_episode(env, policy, horizon)

        # Compute simple summary metrics from the reward history.
        total = total_reward(episode["rewards"])
        avg = average_reward(episode["rewards"])

        # Store both the summary metrics and the raw episode data.
        results[policy_name] = {
            "total_reward": total,
            "average_reward": avg,
            "episode": episode,
        }

    return results


def print_results(results):
    # Print a simple summary of experiment results.
    print("Week 5 Baseline Experiment Results")
    print("-" * 40)

    for policy_name, data in results.items():
        print(
            f"{policy_name:12s} | "
            f"Total Reward: {data['total_reward']:.2f} | "
            f"Average Reward: {data['average_reward']:.4f}"
        )


if __name__ == "__main__":
    results = run_baseline_experiment()
    print_results(results)