import numpy as np

#SYSTEMATIC STRATEGIES 

class AlwaysLong:
    """
    Systematic strategy that always takes a long position.
    Equivalent to passive buy-and-hold.
    Profits in trending up regimes, loses in downtrends and high volatility.
    """
    def select_action(self, price_history):
        return 0  # Long

    def update(self, action, reward):
        # Fixed strategies ignore feedback. update() is called by the simulator
        # uniformly so adaptive strategies can learn here in the future.
        pass


class AlwaysShort:
    """
    Systematic strategy that always takes a short position.
    Profits when price falls, loses when price rises.
    """
    def select_action(self, price_history):
        return 2  # Short

    def update(self, action, reward):
        pass


class AlwaysNeutral:
    """
    Systematic strategy that always holds cash.
    Zero reward regardless of market conditions.
    Useful as a baseline — any strategy worse than this
    is actively destroying value.
    """
    def select_action(self, price_history):
        return 1  # Neutral

    def update(self, action, reward):
        pass

#DISCRETIONARY STRATEGIES 


class MovingAverageCrossover:
    """
    Systematic strategy based on moving average crossover.

    Goes long when short-term average is above long-term average
    — price is trending up.
    Goes short when short-term average is below long-term average
    — price is trending down.
    Holds neutral until enough price history is available.

    Parameters
    ----------
    short_window : int
        Lookback period for short-term moving average.
    long_window : int
        Lookback period for long-term moving average.
    """

    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window

    def select_action(self, price_history):
        # not enough history yet to compute both averages
        if len(price_history) < self.long_window:
            return 1  # Neutral

        short_ma = np.mean(price_history[-self.short_window:])
        long_ma = np.mean(price_history[-self.long_window:])

        if short_ma > long_ma:
            return 0  # Long — short term trend is up
        else:
            return 2  # Short — short term trend is down

    def update(self, action, reward):
        pass


class MomentumStrategy:
    """
    Adaptive strategy that follows recent price momentum.

    Computes average return over a lookback window.
    Goes long if momentum is positive, short if negative, neutral if weak.

    Performs well in trending regimes where price moves persist.
    Performs poorly in mean-reverting regimes — keeps chasing moves that reverse.

    Parameters
    ----------
    lookback : int
        Number of recent steps to compute momentum from.
    threshold : float
        Minimum momentum magnitude to take a position.
        Below this, holds neutral.
    """

    def __init__(self, lookback=20, threshold=0.0005):
        self.lookback = lookback
        self.threshold = threshold

    def select_action(self, price_history):
        if len(price_history) < self.lookback + 1:
            return 1

        price_now = price_history[-1]
        price_before = price_history[-self.lookback]
        momentum = (price_now - price_before) / price_before

        if momentum > self.threshold:
            return 0
        elif momentum < -self.threshold:
            return 2
        else:
            return 1

    def update(self, action, reward):
        pass


class MeanReversionStrategy:
    """
    Adaptive strategy that bets on price reverting to its recent mean.

    Computes how far current price has deviated from its rolling average,
    normalized by rolling standard deviation (z-score).

    If price is well above mean: go short, expect reversion down.
    If price is well below mean: go long, expect reversion up.
    If price is near mean: hold neutral.

    Performs well in mean-reverting regimes.
    Performs poorly in trending regimes — keeps fading moves that continue.

    Parameters
    ----------
    lookback : int
        Window for computing rolling mean and standard deviation.
    threshold : float
        Z-score magnitude required to take a position.
    """

    def __init__(self, lookback=20, threshold=1.0):
        self.lookback = lookback
        self.threshold = threshold

    def select_action(self, price_history):
        if len(price_history) < self.lookback + 1:
            return 1  # Neutral

        recent_prices = price_history[-(self.lookback + 1):]
        rolling_mean = np.mean(recent_prices)
        rolling_std = np.std(recent_prices)

        # avoid division by zero in flat price periods
        if rolling_std == 0:
            return 1

        current_price = price_history[-1]

        # z-score: how many standard deviations from mean
        z_score = (current_price - rolling_mean) / rolling_std

        # print("step: ", len(price_history), "z_score" , z_score)

        if z_score > self.threshold:
            return 2  # Short — price too high relative to recent mean
        elif z_score < -self.threshold:
            return 0  # Long — price too low relative to recent mean
        else:
            return 1  # Neutral — price close enough to mean

    def update(self, action, reward):
        pass


class VolatilityScaledStrategy:
    """
    Adaptive strategy that scales position based on detected volatility.

    In calm conditions: follows momentum signal.
    In high volatility: steps back to neutral — capital preservation mode.

    This models a discretionary trader who reduces risk exposure
    when market conditions become unpredictable.

    Designed to outperform pure momentum in high volatility regimes
    by avoiding large losses during chaotic price moves.

    Parameters
    ----------
    lookback : int
        Window for computing momentum and volatility.
    vol_threshold : float
        Rolling volatility level above which strategy goes neutral.
    momentum_threshold : float
        Minimum momentum magnitude to take a position in calm conditions.
    """

    def __init__(self, lookback=20, vol_threshold=0.008, momentum_threshold=0.0005):
        self.lookback = lookback
        self.vol_threshold = vol_threshold
        self.momentum_threshold = momentum_threshold

    def select_action(self, price_history):
        if len(price_history) < self.lookback + 1:
            return 1

        recent_prices = price_history[-(self.lookback + 1):]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        recent_vol = np.std(returns)
        momentum = np.mean(returns)

        # temporarily print to understand what vol levels look like
        # print(f"vol: {recent_vol:.6f}  threshold: {self.vol_threshold:.6f}  triggered: {recent_vol > self.vol_threshold}")

        if recent_vol > self.vol_threshold:
            return 1
        if momentum > self.momentum_threshold:
            return 0
        elif momentum < -self.momentum_threshold:
            return 2
        else:
            return 1

    def update(self, action, reward):
        pass