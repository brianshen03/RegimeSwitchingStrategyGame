import numpy as np


def generate_trending(num_steps, start_price=100.0, drift=0.001, volatility=0.01, seed=None):
    """
    Generate a trending price series using Geometric Brownian Motion.

    Price has a persistent directional drift plus random noise.
    A positive drift produces an uptrend, negative produces a downtrend.

    Parameters
    ----------
    num_steps : int
        Number of price steps to generate.
    start_price : float
        Starting price.
    drift : float
        Expected return per step. Positive = uptrend, negative = downtrend.
    volatility : float
        Standard deviation of random shocks per step.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of prices of length num_steps + 1 (includes starting price).
    """
    if seed is not None:
        np.random.seed(seed)


    prices = np.zeros(num_steps + 1)
    prices[0] = start_price

    for t in range(1, num_steps + 1):
        shock = np.random.normal(0, 1)
        # new price is previous price * (trend nudge + random noise) 
        prices[t] = prices[t-1] * np.exp(drift + volatility * shock)

    return prices


def generate_mean_reverting(num_steps, start_price=100.0, mean_level=100.0, speed=0.05, volatility=0.8, seed=None):
    """
    Generate a mean-reverting price series using Ornstein-Uhlenbeck process.

    Price gets pulled back toward mean_level whenever it strays too far.
    The further price drifts from mean_level, the stronger the pull back.

    Parameters
    ----------
    num_steps : int
        Number of price steps to generate.
    start_price : float
        Starting price.
    mean_level : float
        The stable level prices revert toward.
    speed : float
        How aggressively price snaps back to mean_level.
        Higher speed = faster reversion.
    volatility : float
        Standard deviation of random shocks per step.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of prices of length num_steps + 1.
    """
    if seed is not None:
        np.random.seed(seed)

    prices = np.zeros(num_steps + 1)
    prices[0] = start_price

    for t in range(1, num_steps + 1):
        shock = np.random.normal(0, 1)
        # pull toward mean_level proportional to how far price has strayed
        change = speed * (mean_level - prices[t-1]) + volatility * shock
        prices[t] = prices[t-1] + change

    return prices


def generate_high_volatility(num_steps, start_price=100.0, 
                              volatility=3.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    prices = np.zeros(num_steps + 1)
    prices[0] = start_price

    # generate all shocks at once then demean them
    # this guarantees zero net drift regardless of seed
    shocks = np.random.normal(0, 1, num_steps)
    shocks = shocks - shocks.mean()  # force mean to exactly zero

    for t in range(1, num_steps + 1):
        prices[t] = prices[t-1] + volatility * shocks[t-1]

    return prices


def generate_regime_price_series(regime_lengths, volatilities=None, seed=None):
    """
    Generate a full price series that switches between regimes.

    Stitches together trending, mean-reverting, and high-volatility
    segments into one continuous price series.

    Parameters
    ----------
    regime_lengths : list[int]
        Number of steps for each regime in order.
        Example: [200, 200, 200] means 200 steps each of trending,
        mean-reverting, and high volatility.

    Returns
    -------
    prices : np.ndarray
        Full price series across all regimes.
    regimes : np.ndarray
        Regime label at each step (0, 1, or 2).
    """
    if seed is not None:
        np.random.seed(seed)

    all_prices = []
    all_regimes = []

    # start price carries over between regimes for continuity
    current_price = 100.0

    generators = [
        lambda n, p: generate_trending(n, start_price=p, drift=0.003, volatility=0.01),
        lambda n, p: generate_mean_reverting(n, start_price=p, mean_level=p, speed=0.05, volatility=0.8),
        lambda n, p: generate_high_volatility(n, start_price=p, volatility=3.0),
        ]

    for regime_idx, length in enumerate(regime_lengths):
        prices = generators[regime_idx](length, current_price)
        # exclude first price to avoid duplicating the join point
        # except for the very first segment
        if len(all_prices) == 0:
            all_prices.extend(prices)
            all_regimes.extend([regime_idx] * len(prices))
        else:
            all_prices.extend(prices[1:])
            all_regimes.extend([regime_idx] * (len(prices) - 1))

        current_price = prices[-1]

    return np.array(all_prices), np.array(all_regimes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    prices, regimes = generate_regime_price_series(
        regime_lengths=[200, 200, 200],
        seed=42
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = {0: "lightyellow", 1: "lightblue", 2: "lightcoral"}
    names = {0: "Trending", 1: "Mean-Reverting", 2: "High Volatility"}

    # shade regimes
    start = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start]:
            ax.axvspan(start, i, alpha=0.3, color=colors[regimes[start]], 
                      label=names[regimes[start]])
            start = i

    ax.plot(prices, color="black", linewidth=0.8)
    ax.set_title("Price Series Across Regimes")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.show()