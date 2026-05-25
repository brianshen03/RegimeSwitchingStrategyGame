import numpy as np
from .price_generator import generate_regime_price_series

# Maps action index to position
POSITION_MAP = {
    0:  1,   # Long
    1:  0,   # Neutral
    2: -1,   # Short
}

ACTION_NAMES = {
    0: "Long",
    1: "Neutral",
    2: "Short",
}


class PriceEnvironment:
    """
    Environment that generates a price series across regimes and computes
    rewards based on the position taken by a strategy.

    Instead of sampling rewards from a means matrix, this environment:
    1. Generates a full price series upfront with distinct regime behavior
    2. At each step, takes a position (long/neutral/short) from the strategy
    3. Computes reward as position * return

    The strategy never directly observes the regime — it only sees price history
    and its own reward, just like a real trader.

    Parameters
    ----------
    regime_lengths : list[int]
        Number of steps for each regime in order.
        Example: [200, 200, 200]
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, regime_lengths, regime_sequence=None, seed=None):
        self.regime_lengths = regime_lengths
        self.seed = seed

        # generate full price series and regime labels upfront
        self.prices, self.regime_labels = generate_regime_price_series(
            regime_lengths=regime_lengths,
            regime_sequence=regime_sequence,
            seed=seed
        )

        # current time step
        self.t = 0

    def reset(self):
        """Reset environment to beginning."""
        self.t = 0

    def get_price_history(self):
        """
        Return all prices observed so far.
        Strategies use this to compute signals.
        """
        return self.prices[:self.t + 1]

    def step(self, action):
        """
        Advance one step after the strategy chooses an action.

        Parameters
        ----------
        action : int
            0 = Long, 1 = Neutral, 2 = Short

        Returns
        -------
        reward : float
            position * return for this step
        regime : int
            the regime that was active this step
        """
        # need at least two prices to compute a return
        if self.t == 0:
            self.t += 1
            return 0.0, self.regime_labels[self.t]

        current_price = self.prices[self.t]
        next_price = self.prices[self.t + 1]

        # compute percentage return
        return_t = (next_price - current_price) / current_price

        # convert action to position
        position = POSITION_MAP[action]

        # reward is how well the position matched the price move
        reward = position * return_t * 100

        # record current regime before advancing
        regime = self.regime_labels[self.t]

        self.t += 1

        return reward, regime

    @property
    def num_actions(self):
        return 3

    @property
    def total_steps(self):
        return len(self.prices) - 1
