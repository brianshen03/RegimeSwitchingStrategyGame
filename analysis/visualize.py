import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REGIME_COLORS = {
    0: "lightyellow",
    1: "lightblue", 
    2: "lightcoral"
}

REGIME_NAMES = {
    0: "Trending",
    1: "Mean-Reverting ",
    2: "High Volatility "
}

ACTION_NAMES = {
    0: "Long",
    1: "Neutral",
    2: "Short"
}


def shade_regimes(ax, regimes):
    """
    Shade the background of a plot to show which regime was active at each step.
    """
    n = len(regimes)
    start = 0
    for i in range(1, n + 1):
        if i == n or regimes[i] != regimes[start]:
            ax.axvspan(start, i, alpha=0.3, color=REGIME_COLORS[regimes[start]], label="_nolegend_")
            start = i


def plot_cumulative_rewards(results, title="Cumulative Reward Over Time"):
    """
    Plot cumulative reward over time for all strategies.
    Regime switches are visible as background shading.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # use any episode's regime history for background shading
    sample_episode = next(iter(results.values()))["episode"]
    shade_regimes(ax, sample_episode["regimes"])

    for policy_name, data in results.items():
        rewards = np.array(data["episode"]["rewards"])
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, label=policy_name)

    # regime legend patches
    regime_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.3, label=REGIME_NAMES[r])
        for r in REGIME_COLORS
    ]

    strategy_handles, strategy_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=strategy_handles + regime_patches,
        loc="upper left",
        fontsize=8
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("graphs/cumulative_rewards.png", dpi=150)
    plt.close()


def plot_action_distribution(results, regime_length=100, title="Action Distribution Per Regime"):
    """
    For each strategy, show what percentage of the time it chose each action
    broken down by regime.
    """
    policy_names = list(results.keys())
    num_policies = len(policy_names)
    num_regimes = 3
    num_actions = 3

    fig, axes = plt.subplots(1, num_regimes, figsize=(16, 6), sharey=True)
    fig.suptitle(title)

    bar_width = 0.25
    x = np.arange(num_policies)

    for regime_idx, ax in enumerate(axes):
        ax.set_title(f"Regime {regime_idx}: {REGIME_NAMES[regime_idx]}", fontsize=9)
        ax.set_facecolor(REGIME_COLORS[regime_idx])

        for action_idx in range(num_actions):
            percentages = []
            for name in policy_names:
                episode = results[name]["episode"]
                actions = np.array(episode["actions"])
                regimes = np.array(episode["regimes"])

                regime_mask = regimes == regime_idx
                regime_actions = actions[regime_mask]

                if len(regime_actions) == 0:
                    percentages.append(0)
                else:
                    pct = np.mean(regime_actions == action_idx) * 100
                    percentages.append(pct)

            offset = (action_idx - 1) * bar_width
            bars = ax.bar(x + offset, percentages, bar_width, label=ACTION_NAMES[action_idx])

        ax.set_xticks(x)
        ax.set_xticklabels(policy_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% of steps" if regime_idx == 0 else "")
        ax.set_ylim(0, 110)

    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig("graphs/action_distribution.png", dpi=150)
    plt.close()


def plot_rolling_reward(results, window=20, title="Rolling Average Reward Over Time"):
    """
    Plot rolling average reward per strategy.
    Smoother than raw rewards, makes regime switch recovery visible.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    sample_episode = next(iter(results.values()))["episode"]
    shade_regimes(ax, sample_episode["regimes"])

    for policy_name, data in results.items():
        rewards = np.array(data["episode"]["rewards"])
        # compute rolling average
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(rolling, label=policy_name)

    regime_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.3, label=REGIME_NAMES[r])
        for r in REGIME_COLORS
    ]

    strategy_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=strategy_handles + regime_patches, loc="upper left", fontsize=8)

    ax.set_xlabel("Step")
    ax.set_ylabel(f"Rolling Average Reward (window={window})")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("graphs/rolling_reward.png", dpi=150)
    plt.close()

def plot_step_rewards(results, filename="graphs/step_rewards.png"):
    import matplotlib.pyplot as plt

    colors = {0: "lightyellow", 1: "lightblue", 2: "lightcoral"}
    names = {0: "Trending", 1: "Mean-Reverting", 2: "High Volatility"}

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)

    if len(results) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        rewards = data["episode"]["rewards"]
        regimes = data["episode"]["regimes"]

        # shade regimes
        start = 0
        labeled = set()
        for i in range(1, len(regimes) + 1):
            if i == len(regimes) or regimes[i] != regimes[start]:
                label = names[regimes[start]] if regimes[start] not in labeled else None
                ax.axvspan(start, i, alpha=0.3, color=colors[regimes[start]], label=label)
                labeled.add(regimes[start])
                start = i

        ax.plot(rewards, linewidth=0.7, color="steelblue")
        ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
        ax.set_title(name)
        ax.set_ylabel("Reward")
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Step")
    plt.suptitle("Per-Step Rewards", fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_price_series(prices, regimes, filename="graphs/price_series.png"):
    """
    Plot and save the price series with regime shading.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # shade regimes
    start = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start]:
            ax.axvspan(start, i, alpha=0.3, 
                      color=REGIME_COLORS[regimes[start]],
                      label=REGIME_NAMES[regimes[start]])
            start = i

    ax.plot(prices, color="black", linewidth=0.8)
    ax.set_title("Price Series Across Regimes")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

