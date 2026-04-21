import numpy as np

class RegimeEnvironment:
    """
    Environment for a repeated decision-making game with hidden regimes.

    Main idea:
    - There are multiple actions the agent can choose from.
    - The environment is in one regime at a time.
    - The current regime determines the expected reward of each action.
    - The agent does not directly observe the regime; it only sees the reward from the action it chose.

    Example:
        means = [
            [1.0, 0.5, 0.2],   # Regime 0: action 0 is best
            [0.2, 1.0, 0.5],   # Regime 1: action 1 is best
            [0.5, 0.2, 1.0],   # Regime 2: action 2 is best
        ]

    In that example:
    - each row corresponds to a regime
    - each column corresponds to an action
    - means[r][a] is the expected reward of taking action a in regime r
    """

    def __init__(self, means, sigma=0.1, regime_length=100, fixed_regime=None):
        """
        Initialize the environment.

        Parameters
        ----------
        means : list[list[float]] or np.ndarray
            2D structure with shape (num_regimes, num_actions).

            means[r][a] = expected reward for action a when the environment
            is in regime r.

        sigma : float
            Standard deviation of reward noise.

            Rewards are not deterministic. Instead, when the agent picks an
            action, the actual reward is sampled from a normal distribution:

                reward ~ Normal(mean, sigma)

            So sigma controls how noisy the environment is:
            - small sigma -> rewards stay close to expected value
            - large sigma -> rewards are more random

        regime_length : int
            Number of time steps spent in a regime before switching to the next
            one, if switching mode is being used.

            Example:
            - if regime_length = 100
            - then steps 0-99 use regime 0
            - steps 100-199 use regime 1
            - etc.

        fixed_regime : int or None
            If provided, the environment stays in this one regime forever.

            Example: fixed_regime=0 means the entire run uses regime 0 and never switches.
        """

        # Convert means to a NumPy array so indexing is easier and more reliable.
        self.means = np.array(means, dtype=float)
        self.sigma = sigma
        self.regime_length = regime_length
        # If this is not None, we ignore switching and always use this regime.
        self.fixed_regime = fixed_regime
        # Current time step in the simulation, starts at 0 and increases by 1 every time step() is called.
        self.t = 0
        # Store useful dimensions for convenience.
        self.num_regimes = self.means.shape[0]
        self.num_actions = self.means.shape[1]

    def reset(self):
        # Reset the environment to the beginning of a new episode.
        self.t = 0

    def get_regime(self):
        # Return the current regime index.

        # If a fixed regime was specified, do not switch at all.
        if self.fixed_regime is not None:
            return self.fixed_regime

        # Otherwise, determine regime based on the current time step.
        # t // regime_length tells us which regime block we are in
        # % num_regimes wraps around once we reach the last regime
        return (self.t // self.regime_length) % self.num_regimes

    def step(self, action):
        """
        Advance the environment by one time step after the agent chooses an action.

        This is the main method the simulator will call every round.

        What happens here:
        1. Figure out the current regime
        2. Look up the expected reward for the chosen action
        3. Sample a noisy reward from a normal distribution
        4. Advance time by one step
        5. Return the observed reward and the regime used
        """
        # Determine which regime we are currently in.
        regime = self.get_regime()

        # Look up the expected reward for this regime-action pair.
        mean = self.means[regime, action]

        # Sample the actual reward from a normal distribution centered at `mean`.
        # This introduces randomness/noise into the environment.
        reward = np.random.normal(loc=mean, scale=self.sigma)

        # Move time forward by one step so the next call to step() happens at t+1.
        self.t += 1

        # Return both the observed reward and the underlying regime.
        return reward, regime