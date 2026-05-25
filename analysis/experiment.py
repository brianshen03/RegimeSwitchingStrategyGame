import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.price_environment import PriceEnvironment
from strategies.strategies import (
    AlwaysLong, AlwaysShort, AlwaysNeutral, MovingAverageCrossover,
    MomentumStrategy, MeanReversionStrategy, VolatilityScaledStrategy
)
from core.simulator import run_episode
from core.metrics import total_reward, average_reward, reward_by_regime, compute_regime_switch_lags, compute_reward_distribution
from core.price_generator import generate_regime_price_series
from analysis.visualize import plot_price_series, plot_cumulative_rewards, plot_action_distribution, plot_lag_analysis, plot_heatmap, plot_reward_distribution
import csv

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout
    def write(self, data):
        self._file.write(data)
        self._stdout.write(data)
    def flush(self):
        self._file.flush()
        self._stdout.flush()

# Maps each adaptive strategy to the regimes where its logic is designed to profit.
# Lag analysis — steps until rolling reward turns positive after a regime entry —
# is only interpretable for these pairings. Fixed strategies (AlwaysLong, etc.)
# are excluded because they never adapt, so "lag" has no meaning for them.
STRATEGY_HOME_REGIMES = {
    "MA_Crossover":     [0, 1],
    "Momentum":         [0, 1],
    "MeanReversion":    [2],
    "VolatilityScaled": [3],
}

def run_full_experiment(regime_lengths=[500, 500, 500, 500], seed=42, regime_sequence=None):
    """
    Run all strategies on one price series and return per-strategy results.

    Parameters
    ----------
    regime_lengths : list[int]
        Number of steps for each regime segment.
    seed : int
        Random seed controlling the price path.
    regime_sequence : list[int] or None
        Order of regime types. Defaults to [0, 1, 2, 3] if None.

    Returns
    -------
    dict : {strategy_name: {"total_reward", "average_reward", "episode"}}
    """
    strategy_factories = {
        "AlwaysLong":       lambda: AlwaysLong(),
        "AlwaysShort":      lambda: AlwaysShort(),
        "AlwaysNeutral":    lambda: AlwaysNeutral(),
        "MA_Crossover":     lambda: MovingAverageCrossover(short_window=10, long_window=30),
        "Momentum":         lambda: MomentumStrategy(lookback=20, threshold=0.001),
        "MeanReversion":    lambda: MeanReversionStrategy(lookback=20, threshold=1.0),
        "VolatilityScaled": lambda: VolatilityScaledStrategy(lookback=20, vol_threshold=0.011),
    }

    horizon = sum(regime_lengths)
    results = {}

    for name, make_strategy in strategy_factories.items():
        env = PriceEnvironment(regime_lengths=regime_lengths, regime_sequence=regime_sequence, seed=seed)
        strategy = make_strategy()
        episode = run_episode(env, strategy, horizon)

        results[name] = {
            "total_reward":   total_reward(episode["rewards"]),
            "average_reward": average_reward(episode["rewards"]),
            "episode":        episode,
        }

    return results

def print_results(results, title="Experiment Results"):
    print(title)
    print("-" * 40)
    for name, data in results.items():
        print(
            f"{name:15s} | "
            f"Total Reward: {data['total_reward']:.4f} | "
            f"Average Reward: {data['average_reward']:.6f}"
        )


def run_multi_seed_experiment(regime_lengths=[500, 500, 500, 500], seeds=None):
    """
    Run all strategies across multiple seeds and collect per-step reward arrays.

    Each seed produces a different price series with the same regime structure,
    so results can be aggregated to see which findings hold consistently.

    Returns
    -------
    reward_arrays : dict
        {strategy_name: [np.array of rewards per seed]}
    regimes : list
        Regime labels from one episode (identical structure across seeds).
    """
    if seeds is None:
        seeds = list(range(30))

    first_run = run_full_experiment(regime_lengths=regime_lengths, seed=seeds[0])
    strategy_names = list(first_run.keys())
    reward_arrays = {name: [np.array(first_run[name]["episode"]["rewards"])] for name in strategy_names}
    regimes = first_run[strategy_names[0]]["episode"]["regimes"]

    for seed in seeds[1:]:
        single_run = run_full_experiment(regime_lengths=regime_lengths, seed=seed)
        for name in strategy_names:
            reward_arrays[name].append(np.array(single_run[name]["episode"]["rewards"]))

    return reward_arrays, regimes


def print_multi_seed_results(reward_arrays, title="Multi-Seed Results"):
    print(title)
    print("-" * 50)
    for name, arrays in reward_arrays.items():
        totals = [float(np.sum(a)) for a in arrays]
        mean = np.mean(totals)
        std = np.std(totals)
        print(f"{name:15s} | Mean Total Reward: {mean:8.2f} | Std: {std:.2f}")


def sample_regime_lengths(total_steps, num_regimes, rng, min_len=100):
    """
    Split total_steps randomly across num_regimes segments.
    Each segment is guaranteed at least min_len steps.
    """
    remaining = total_steps - min_len * num_regimes
    extra = rng.multinomial(remaining, [1 / num_regimes] * num_regimes)
    return (np.ones(num_regimes, dtype=int) * min_len + extra).tolist()


def run_random_regime_experiment(regime_lengths=[500, 500, 500, 500], seeds=None,
                                  random_lengths=False, total_steps=2000):
    """
    Run all strategies across multiple seeds, each with a randomly shuffled
    regime order. Optionally also randomizes regime lengths each seed.

    Parameters
    ----------
    random_lengths : bool
        If True, each seed gets a random split of total_steps across regimes
        (minimum 100 steps per regime). regime_lengths is ignored.
    total_steps : int
        Total episode length when random_lengths=True.

    Returns
    -------
    reward_arrays : dict
        {strategy_name: [np.array of rewards per seed]}
    regime_rewards : dict
        {strategy_name: {regime_idx: [total_reward_per_seed]}}
    """
    if seeds is None:
        seeds = list(range(30))

    num_regimes = len(regime_lengths)

    def _lengths_and_sequence(seed):
        rng = np.random.default_rng(seed)
        sequence = rng.permutation(num_regimes).tolist()
        lengths = sample_regime_lengths(total_steps, num_regimes, rng) if random_lengths else regime_lengths
        return lengths, sequence

    first_lengths, first_sequence = _lengths_and_sequence(seeds[0])
    first_run = run_full_experiment(regime_lengths=first_lengths, seed=seeds[0], regime_sequence=first_sequence)

    strategy_names = list(first_run.keys())
    reward_arrays = {name: [np.array(first_run[name]["episode"]["rewards"])] for name in strategy_names}
    regime_rewards = {name: {r: [] for r in range(num_regimes)} for name in strategy_names}

    for name in strategy_names:
        ep = first_run[name]["episode"]
        for r, val in reward_by_regime(ep["rewards"], ep["regimes"]).items():
            regime_rewards[name][r].append(val)

    for seed in seeds[1:]:
        lengths, sequence = _lengths_and_sequence(seed)
        single_run = run_full_experiment(regime_lengths=lengths, seed=seed, regime_sequence=sequence)
        for name in strategy_names:
            reward_arrays[name].append(np.array(single_run[name]["episode"]["rewards"]))
            ep = single_run[name]["episode"]
            for r, val in reward_by_regime(ep["rewards"], ep["regimes"]).items():
                regime_rewards[name][r].append(val)

    return reward_arrays, regime_rewards


def run_lag_analysis(regime_lengths=[500, 500, 500, 500], seeds=None, window=20,
                     random_order=False, random_lengths=False, total_steps=2000):
    """
    Across multiple seeds, compute how many steps each strategy takes to turn
    profitable after entering each regime.

    Parameters
    ----------
    random_order : bool
        If True, each seed gets a randomly shuffled regime sequence.
    random_lengths : bool
        If True, each seed gets randomly split regime lengths summing to
        total_steps. Uses the same RNG as run_random_regime_experiment so
        sequences and lengths are consistent between the two functions.
    """
    if seeds is None:
        seeds = list(range(30))

    num_regimes = len(regime_lengths)
    lag_data = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        sequence = rng.permutation(num_regimes).tolist() if random_order else None
        lengths = sample_regime_lengths(total_steps, num_regimes, rng) if random_lengths else regime_lengths
        single_run = run_full_experiment(regime_lengths=lengths, seed=seed, regime_sequence=sequence)
        for name, data in single_run.items():
            ep = data["episode"]
            lags = compute_regime_switch_lags(ep["rewards"], ep["regimes"], window=window)
            if name not in lag_data:
                lag_data[name] = {}
            for regime_idx, regime_lags in lags.items():
                lag_data[name].setdefault(regime_idx, []).extend(regime_lags)

    return lag_data


def print_lag_results(lag_data, title="Lag Analysis (steps until positive rolling reward)", home_regimes=None):
    from analysis.visualize import REGIME_NAMES
    print(title)
    print("-" * 60)
    for name, regimes in lag_data.items():
        if home_regimes and name not in home_regimes:
            continue
        print(f"{name}")
        for regime_idx in sorted(regimes):
            if home_regimes and regime_idx not in home_regimes.get(name, []):
                continue
            vals = [v for v in regimes[regime_idx] if v is not None]
            never = sum(1 for v in regimes[regime_idx] if v is None)
            mean = f"{np.mean(vals):.1f}" if vals else "N/A"
            print(f"  {REGIME_NAMES[regime_idx]:18s} | mean lag: {mean:>6} steps | never recovered: {never}/{len(regimes[regime_idx])}")


def print_random_regime_results(reward_arrays, title="Random Regime Results"):
    print(title)
    print("-" * 50)
    for name, arrays in reward_arrays.items():
        totals = [float(np.sum(a)) for a in arrays]
        mean = np.mean(totals)
        std = np.std(totals)
        print(f"{name:15s} | Mean Total Reward: {mean:8.2f} | Std: {std:.2f}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    _results_file = open("data/results.txt", "w")
    sys.stdout = _Tee(_results_file)

    regime_lengths = [500, 500, 500, 500]
    seed = 42

    prices, regimes = generate_regime_price_series(
        regime_lengths=regime_lengths,
        seed=seed
    )
    for folder in ["graphs/seed42", "graphs/random_order_fixed_length", "graphs/random_order_random_length"]:
        os.makedirs(folder, exist_ok=True)

    plot_price_series(prices, regimes, filename="graphs/seed42/price_series.png")

    results = run_full_experiment(regime_lengths=regime_lengths, seed=seed)
    print_results(results, title="Single-Seed Results (seed=42)")

    plot_cumulative_rewards(results, filename="graphs/seed42/cumulative_rewards.png")
    plot_action_distribution(results, filename="graphs/seed42/action_distribution.png")

    print()
    reward_arrays, multi_regimes = run_multi_seed_experiment(
        regime_lengths=regime_lengths,
        seeds=list(range(30))
    )
    print_multi_seed_results(reward_arrays, title="Multi-Seed Results (30 seeds, fixed order)")

    print()
    rr_arrays, regime_rewards = run_random_regime_experiment(
        regime_lengths=regime_lengths,
        seeds=list(range(30))
    )
    print_random_regime_results(rr_arrays, title="Random Regime Order Results (30 seeds, fixed lengths)")
    plot_heatmap(regime_rewards,
                 title="Mean Reward by Strategy and Regime (30 Seeds, Random Order)",
                 filename="graphs/random_order_fixed_length/heatmap.png")
    dist_stats = compute_reward_distribution(rr_arrays)
    plot_reward_distribution(dist_stats,
                             filename="graphs/random_order_fixed_length/reward_distribution.png")

    print()
    rl_arrays, rl_regime_rewards = run_random_regime_experiment(
        seeds=list(range(30)),
        random_lengths=True,
        total_steps=2000
    )
    print_random_regime_results(rl_arrays, title="Random Regime Order + Length Results (30 seeds)")
    plot_heatmap(rl_regime_rewards,
                 title="Mean Reward by Strategy and Regime (30 Seeds, Random Order + Length)",
                 filename="graphs/random_order_random_length/heatmap.png")
    rl_dist_stats = compute_reward_distribution(rl_arrays)
    plot_reward_distribution(rl_dist_stats,
                             title="Reward Distribution by Strategy (30 Seeds, Random Order + Length)",
                             filename="graphs/random_order_random_length/reward_distribution.png")

    print()
    lag_data_random = run_lag_analysis(regime_lengths=regime_lengths, seeds=list(range(30)), random_order=True)
    print_lag_results(lag_data_random, title="Lag Analysis — Random Order (30 seeds)", home_regimes=STRATEGY_HOME_REGIMES)
    plot_lag_analysis(lag_data_random, title="Lag Analysis — Random Regime Order",
                      filename="graphs/random_order_fixed_length/lag_analysis.png",
                      home_regimes=STRATEGY_HOME_REGIMES)

    print()
    lag_data_random_lengths = run_lag_analysis(seeds=list(range(30)), random_order=True,
                                               random_lengths=True, total_steps=2000)
    print_lag_results(lag_data_random_lengths, title="Lag Analysis — Random Order + Length (30 seeds)", home_regimes=STRATEGY_HOME_REGIMES)
    plot_lag_analysis(lag_data_random_lengths, title="Lag Analysis — Random Regime Order + Length",
                      filename="graphs/random_order_random_length/lag_analysis.png",
                      home_regimes=STRATEGY_HOME_REGIMES)

    sys.stdout = sys.__stdout__
    _results_file.close()