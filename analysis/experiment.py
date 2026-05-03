import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.price_environment import PriceEnvironment
from strategies.strategies import (
    AlwaysLong, AlwaysShort, AlwaysNeutral, MovingAverageCrossover,
    MomentumStrategy, MeanReversionStrategy, VolatilityScaledStrategy
)
from core.simulator import run_episode
from core.metrics import total_reward, average_reward, reward_by_regime
from core.price_generator import generate_regime_price_series
from analysis.visualize import plot_price_series, plot_cumulative_rewards, plot_action_distribution, plot_rolling_reward, plot_step_rewards
import csv

def run_full_experiment(regime_lengths=[200, 200, 200], seed=42):
    strategy_factories = {
        "AlwaysLong":       lambda: AlwaysLong(),
        "AlwaysShort":      lambda: AlwaysShort(),
        "AlwaysNeutral":    lambda: AlwaysNeutral(),
        "MA_Crossover":     lambda: MovingAverageCrossover(short_window=10, long_window=30),
        "Momentum":         lambda: MomentumStrategy(lookback=30, threshold=0.001),
        "MeanReversion":    lambda: MeanReversionStrategy(lookback=20, threshold=1.0),
        "VolatilityScaled": lambda: VolatilityScaledStrategy(lookback=20, vol_threshold=0.011),
    }

    horizon = sum(regime_lengths)
    results = {}

    for name, make_strategy in strategy_factories.items():
        env = PriceEnvironment(regime_lengths=regime_lengths, seed=seed)
        strategy = make_strategy()
        episode = run_episode(env, strategy, horizon)

        results[name] = {
            "total_reward":   total_reward(episode["rewards"]),
            "average_reward": average_reward(episode["rewards"]),
            "regime_rewards": reward_by_regime(episode["rewards"], episode["regimes"]),
            "episode":        episode,
        }

    return results

def print_results(results, title="Experiment Results"):
    print(title)
    print("-" * 40)
    for name, data in results.items():
        regime_str = "  ".join(
            f"R{r}: {v:+.4f}" for r, v in sorted(data["regime_rewards"].items())
        )
        print(
            f"{name:15s} | "
            f"Total: {data['total_reward']:+.4f} | "
            f"Avg: {data['average_reward']:+.6f} | "
            f"{regime_str}"
        )


if __name__ == "__main__":
    regime_lengths = [200, 200, 200]
    seed = 42

    # save price series first
    prices, regimes = generate_regime_price_series(regime_lengths=regime_lengths, seed=seed)
    plot_price_series(prices, regimes, filename="graphs/price_series.png")

    # run experiment
    results = run_full_experiment(regime_lengths=regime_lengths, seed=seed)
    print_results(results, title="All Strategy Experiment Results")

    plot_cumulative_rewards(results)
    plot_rolling_reward(results, window=20)
    plot_action_distribution(results)
    plot_step_rewards(results)

    with open("data/strategy_debug.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "step", "price", "action", "reward", "cumulative_reward", "regime"])

        for strategy_name, data in results.items():
            cumulative = 0
            episode = data["episode"]
            for step, (action, reward, regime) in enumerate(zip(episode["actions"], episode["rewards"], episode["regimes"])):
                cumulative += reward
                price = prices[step]
                writer.writerow([strategy_name, step, f"{price:.4f}", action, f"{reward:.4f}", f"{cumulative:.4f}", regime])
