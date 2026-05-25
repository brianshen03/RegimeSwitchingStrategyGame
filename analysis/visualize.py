import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REGIME_COLORS = {
    0: "lightyellow",
    1: "lightgreen",
    2: "lightblue",
    3: "lightcoral"
}

REGIME_NAMES = {
    0: "Trending Up",
    1: "Trending Down",
    2: "Mean-Reverting",
    3: "High Volatility"
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


def plot_cumulative_rewards(results, title="Cumulative Reward Over Time", filename="graphs/seed42/cumulative_rewards.png"):
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
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_action_distribution(results, regime_length=100, title="Action Distribution Per Regime", filename="graphs/seed42/action_distribution.png"):
    """
    For each strategy, show what percentage of the time it chose each action
    broken down by regime.
    """
    policy_names = list(results.keys())
    num_policies = len(policy_names)
    num_regimes = len(REGIME_NAMES)
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
    plt.savefig(filename, dpi=150)
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
                label = REGIME_NAMES[regimes[start]] if regimes[start] not in labeled else None
                ax.axvspan(start, i, alpha=0.3, color=REGIME_COLORS[regimes[start]], label=label)
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

def plot_reward_distribution(dist_stats, title="Reward Distribution by Strategy (30 Seeds, Random Order)", filename="graphs/reward_distribution_30seeds_random.png"):
    """
    Two subplots answering: do strategies win by big gains or by avoiding losses?
      Left:  % of steps with positive reward per strategy
      Right: mean gain (positive steps) vs mean loss magnitude (negative steps)
    """
    strategy_names = list(dist_stats.keys())
    x = np.arange(len(strategy_names))
    bar_width = 0.5

    fig, (ax_pct, ax_mag) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)

    # --- left: % positive steps ---
    pct_vals = [dist_stats[n]["pct_positive"] for n in strategy_names]
    ax_pct.bar(x, pct_vals, bar_width, color="steelblue")
    ax_pct.axhline(50, color="black", linewidth=0.8, linestyle="--", label="50% line")
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(strategy_names, rotation=45, ha="right", fontsize=9)
    ax_pct.set_ylabel("% of Steps")
    ax_pct.set_title("% of Steps with Positive Reward")
    ax_pct.set_ylim(0, 100)
    ax_pct.legend(fontsize=8)

    # --- right: mean gain vs mean loss ---
    gains = [dist_stats[n]["mean_gain"] for n in strategy_names]
    losses = [abs(dist_stats[n]["mean_loss"]) for n in strategy_names]

    ax_mag.bar(x - bar_width / 4, gains,  bar_width / 2, label="Mean Gain",         color="seagreen")
    ax_mag.bar(x + bar_width / 4, losses, bar_width / 2, label="Mean Loss (abs)", color="tomato")
    ax_mag.set_xticks(x)
    ax_mag.set_xticklabels(strategy_names, rotation=45, ha="right", fontsize=9)
    ax_mag.set_ylabel("Reward per Step")
    ax_mag.set_title("Mean Gain vs Mean Loss Magnitude")
    ax_mag.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_lag_analysis(lag_data, title="Regime Adaptation Lag (Steps Until Positive Rolling Reward)", filename="graphs/lag_analysis.png", home_regimes=None):
    """
    For each regime, show mean steps until each strategy first turns profitable.
    If home_regimes is provided, only plots strategy-regime pairs where the
    strategy is designed to profit in that regime.
    """
    num_regimes = len(REGIME_NAMES)
    fig, axes = plt.subplots(1, num_regimes, figsize=(18, 6))
    fig.suptitle(title)

    bar_width = 0.6

    for regime_idx, ax in enumerate(axes):
        ax.set_title(REGIME_NAMES[regime_idx], fontsize=9)
        ax.set_facecolor(REGIME_COLORS[regime_idx])

        relevant = [
            name for name in lag_data
            if home_regimes is None or regime_idx in home_regimes.get(name, [])
        ]

        if not relevant:
            ax.set_xticks([])
            continue

        x = np.arange(len(relevant))
        means, stds, never_counts = [], [], []
        for name in relevant:
            vals = lag_data[name].get(regime_idx, [])
            recovered = [v for v in vals if v is not None]
            means.append(np.mean(recovered) if recovered else 0)
            stds.append(np.std(recovered) if recovered else 0)
            never_counts.append(sum(1 for v in vals if v is None))

        ax.bar(x, means, bar_width, yerr=stds, capsize=3,
               color=REGIME_COLORS[regime_idx], edgecolor="steelblue", linewidth=0.8)

        for i, (mean, never) in enumerate(zip(means, never_counts)):
            if never > 0:
                ax.text(i, mean + 2, f"n/a:{never}", ha="center", fontsize=6, color="red")

        ax.set_xticks(x)
        ax.set_xticklabels(relevant, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Lag (steps)" if regime_idx == 0 else "")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_random_regime_cumulative(reward_arrays, title="Random Regime Order: Cumulative Reward (Mean ± 1 Std)"):
    """
    Mean ± 1 std cumulative reward across seeds with random regime ordering.
    No regime shading since each seed has a different regime sequence.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for strategy_name, arrays in reward_arrays.items():
        stacked = np.stack(arrays)
        cum = np.cumsum(stacked, axis=1)
        mean_cum = np.mean(cum, axis=0)
        std_cum = np.std(cum, axis=0)

        line, = ax.plot(mean_cum, label=strategy_name)
        ax.fill_between(
            range(len(mean_cum)),
            mean_cum - std_cum,
            mean_cum + std_cum,
            alpha=0.15,
            color=line.get_color()
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("graphs/random_regime_cumulative.png", dpi=150)
    plt.close()


def plot_reward_by_regime(regime_rewards, title="Mean Reward by Regime Type (Random Ordering)"):
    """
    For each regime type, show mean reward earned per strategy across all seeds.
    Error bars show ±1 std. Answers: does each strategy consistently do well
    in the regime it was designed for, regardless of when that regime appears?
    """
    strategy_names = list(regime_rewards.keys())
    num_strategies = len(strategy_names)
    num_regimes = len(REGIME_NAMES)

    fig, axes = plt.subplots(1, num_regimes, figsize=(18, 6), sharey=True)
    fig.suptitle(title)

    x = np.arange(num_strategies)
    bar_width = 0.6

    for regime_idx, ax in enumerate(axes):
        ax.set_title(REGIME_NAMES[regime_idx], fontsize=9)
        ax.set_facecolor(REGIME_COLORS[regime_idx])
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

        means = []
        stds = []
        for name in strategy_names:
            vals = regime_rewards[name][regime_idx]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        bars = ax.bar(x, means, bar_width, yerr=stds, capsize=3,
                      color=[REGIME_COLORS[regime_idx]] * num_strategies,
                      edgecolor="steelblue", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Total Reward" if regime_idx == 0 else "")

    plt.tight_layout()
    plt.savefig("graphs/reward_by_regime_30seeds_random.png", dpi=150)
    plt.close()


def plot_multi_seed_cumulative(reward_arrays, regimes, title="Multi-Seed Cumulative Reward (Mean ± 1 Std)"):
    """
    Plot mean cumulative reward with a ±1 std shaded band per strategy.
    Regime shading uses the consistent regime structure shared across all seeds.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    shade_regimes(ax, regimes)

    for strategy_name, arrays in reward_arrays.items():
        stacked = np.stack(arrays)                        # (num_seeds, num_steps)
        cum = np.cumsum(stacked, axis=1)                  # cumulative reward per seed
        mean_cum = np.mean(cum, axis=0)
        std_cum = np.std(cum, axis=0)

        line, = ax.plot(mean_cum, label=strategy_name)
        ax.fill_between(
            range(len(mean_cum)),
            mean_cum - std_cum,
            mean_cum + std_cum,
            alpha=0.15,
            color=line.get_color()
        )

    regime_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.3, label=REGIME_NAMES[r])
        for r in REGIME_COLORS
    ]
    strategy_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=strategy_handles + regime_patches, loc="upper left", fontsize=8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("graphs/multi_seed_cumulative.png", dpi=150)
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


def plot_heatmap(regime_rewards, title="Mean Reward by Strategy and Regime", filename="graphs/heatmap_30seeds_random.png"):
    """
    Plot a heatmap of mean reward per strategy per regime.
    Rows = strategies, columns = regimes.
    """
    strategy_names = list(regime_rewards.keys())
    num_regimes = len(REGIME_NAMES)

    # build matrix: rows = strategies, columns = regimes
    matrix = np.zeros((len(strategy_names), num_regimes))
    for i, name in enumerate(strategy_names):
        for r in range(num_regimes):
            vals = regime_rewards[name][r]
            matrix[i, r] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # diverging colormap — red for negative, white for zero, green for positive
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", 
                   vmin=-vmax, vmax=vmax)
    
    # labels
    ax.set_xticks(range(num_regimes))
    ax.set_xticklabels([REGIME_NAMES[r] for r in range(num_regimes)], fontsize=11)
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names, fontsize=11)
    
    # annotate each cell with the value
    for i in range(len(strategy_names)):
        for j in range(num_regimes):
            val = matrix[i, j]
            color = "black" if abs(val) < vmax * 0.7 else "white"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                   fontsize=10, color=color, fontweight="bold")
    
    plt.colorbar(im, ax=ax, label="Mean Total Reward")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()