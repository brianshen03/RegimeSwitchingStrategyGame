import numpy as np

def total_reward(rewards):
    # Compute the total reward earned over an episode.
    return float(np.sum(rewards))

def average_reward(rewards):
    # Compute the mean reward per step.
    return float(np.mean(rewards))

def cumulative_reward(rewards):
    # Compute the cumulative (running total) reward over time.
    return np.cumsum(rewards)

def num_steps(rewards):
    # Return the number of time steps in the episode.
    return len(rewards)

def reward_by_regime(rewards, regimes):
    rewards = np.array(rewards)
    regimes = np.array(regimes)
    return {
        int(r): float(np.sum(rewards[regimes == r]))
        for r in np.unique(regimes)
    }

def compute_reward_distribution(reward_arrays):
    """
    For each strategy, compute reward distribution stats across all seeds.

    Parameters
    ----------
    reward_arrays : dict
        {strategy_name: [np.array of rewards per seed]}

    Returns
    -------
    dict: {strategy_name: {"pct_positive": float, "mean_gain": float, "mean_loss": float}}
    """
    stats = {}
    for name, arrays in reward_arrays.items():
        all_rewards = np.concatenate(arrays)
        positive = all_rewards[all_rewards > 0]
        negative = all_rewards[all_rewards < 0]

        stats[name] = {
            "pct_positive": len(positive) / len(all_rewards) * 100,
            "mean_gain":    float(np.mean(positive)) if len(positive) > 0 else 0.0,
            "mean_loss":    float(np.mean(negative)) if len(negative) > 0 else 0.0,
        }
    return stats


def compute_regime_switch_lags(rewards, regimes, window=20):
    """
    For each regime entry (including the first), measure how many steps until
    the strategy's rolling average reward first turns positive in that regime.

    Parameters
    ----------
    rewards : array-like
    regimes : array-like
    window : int
        Rolling average window size.

    Returns
    -------
    dict: {regime_idx: [lag_in_steps, ...]}
        One lag value per regime entry. None means the strategy never turned
        profitable during that regime segment.
    """
    rewards = np.array(rewards)
    regimes = np.array(regimes)
    n = len(regimes)

    # collect the start index of each regime segment
    segment_starts = [0]
    for i in range(1, n):
        if regimes[i] != regimes[i - 1]:
            segment_starts.append(i)
    segment_starts.append(n)  # sentinel

    lags = {}

    for i in range(len(segment_starts) - 1):
        t_start = segment_starts[i]
        t_end = segment_starts[i + 1]
        regime = int(regimes[t_start])
        segment = rewards[t_start:t_end]

        if len(segment) < window:
            continue

        rolling = np.convolve(segment, np.ones(window) / window, mode="valid")
        positive = np.where(rolling > 0)[0]
        lag = int(positive[0]) if len(positive) > 0 else None

        lags.setdefault(regime, []).append(lag)

    return lags