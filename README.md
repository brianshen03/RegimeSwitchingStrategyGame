# RegimeSwitchingStrategyGame

A financial market simulation that tests fixed trading strategies across four distinct market regimes. The core question: which strategies profit in which conditions, and how quickly do they adapt when the regime changes?

---

## How to Run

```bash
# From the project root
python analysis/experiment.py
```

Runs all experiments sequentially. Outputs:
- Graphs saved to `graphs/` subdirectories
- Full printed results saved to `data/results.txt`
- Runtime: ~1–2 minutes (30 seeds × 4 experiments)

---

## File Structure

```
RegimeSwitchingStrategyGame/
│
├── core/
│   ├── price_generator.py     # GBM / OU price series generation per regime
│   ├── price_environment.py   # Gym-like env: step(), reward = position × return × 100
│   ├── simulator.py           # run_episode(env, strategy, horizon) loop
│   └── metrics.py             # total_reward, reward_by_regime, lag analysis, reward distribution
│
├── strategies/
│   └── strategies.py          # All 7 strategy classes
│
├── analysis/
│   ├── experiment.py          # Entry point — runs all experiments, prints results
│   └── visualize.py           # All plotting functions (price series, heatmap, lag, etc.)
│
├── graphs/
│   ├── seed42/                # Single-seed (seed=42) baseline plots
│   ├── random_order_fixed_length/   # 30 seeds, random regime order, fixed 500-step lengths
│   └── random_order_random_length/  # 30 seeds, random regime order, random regime lengths
│
├── data/
│   └── results.txt            # Full text output from last experiment run
│
├── week7_analysis.md          # Analysis write-up through Week 7 checkpoint
├── week8_analysis.md          # Analysis write-up for Week 8 additions
└── TODO.txt                   # Remaining work and ideas
```

---

## Market Regimes

Four regimes, each with a distinct price-generating process:

| # | Name | Generator | Behavior |
|---|------|-----------|----------|
| 0 | Trending Up | GBM, drift = +0.001 | Persistent upward drift |
| 1 | Trending Down | GBM, drift = −0.001 | Persistent downward drift |
| 2 | Mean-Reverting | Ornstein-Uhlenbeck | Prices pulled toward a mean |
| 3 | High Volatility | GBM, drift = 0, high σ | Large random swings, no trend |

Default regime length: **500 steps each** (2000 steps total per episode).

---

## Strategies

| Name | Logic |
|------|-------|
| AlwaysLong | Always hold long position |
| AlwaysShort | Always hold short position |
| AlwaysNeutral | Never take a position (zero reward baseline) |
| MA_Crossover | Long when short MA > long MA, short otherwise |
| Momentum | Long/short based on recent return exceeding threshold |
| MeanReversion | Fade the trend — short after up moves, long after down moves |
| VolatilityScaled | Go neutral when volatility is high, long otherwise |

**Key parameters (set in `experiment.py`):**
- `MA_Crossover`: `short_window=10`, `long_window=30`
- `Momentum`: `lookback=20`, `threshold=0.001`
- `MeanReversion`: `lookback=20`, `threshold=1.0`
- `VolatilityScaled`: `lookback=20`, `vol_threshold=0.011`

---

## Experiments

### 1. Single-Seed Baseline (`seed=42`, fixed order)
- Fixed regime sequence: Trending Up → Trending Down → Mean-Reverting → High Volatility
- Generates: price series plot, cumulative reward curves, action distribution by regime
- Output: `graphs/seed42/`
- Purpose: establish a readable baseline before aggregating across seeds

### 2. Multi-Seed Fixed Order (30 seeds, fixed regime order)
- Same regime sequence each seed; only the price path changes
- Aggregates total reward across seeds to get mean ± std per strategy
- No graph output (variance bands on cumulative rewards are unreadable at high volatility)
- Purpose: distinguish genuine strategy edge from single-seed luck

### 3. Random Regime Order, Fixed Length (30 seeds)
- Each seed: regime order is shuffled randomly (same 4 regimes, randomized sequence)
- Regime lengths remain fixed at 500 steps each
- Eliminates ordering artifacts — strategies encounter each regime from any position
- Generates: heatmap (mean reward by strategy × regime), reward distribution, lag analysis
- Output: `graphs/random_order_fixed_length/`

### 4. Random Regime Order + Random Length (30 seeds)
- Each seed: regime order shuffled AND regime lengths split randomly
- Total episode length fixed at 2000 steps; each regime gets at least 100 steps
- Length sampling: `rng.multinomial(remaining, [1/4]*4)` with `min_len=100`
- Generates: same plots as experiment 3
- Output: `graphs/random_order_random_length/`
- Purpose: test whether findings hold when regime durations are also unpredictable

---

## Lag Analysis

Measures how many steps until a strategy's rolling average reward first turns positive after entering a regime.

- Rolling window: **20 steps**
- `None` means the strategy never turned profitable during that regime segment
- Only reported for strategies in their "home" regimes (where their logic is designed to work):
  - `MA_Crossover`, `Momentum` → Trending Up, Trending Down
  - `MeanReversion` → Mean-Reverting
  - `VolatilityScaled` → High Volatility
- Run across 30 seeds with random regime ordering for more segment observations

---

## Key Findings

- **AlwaysLong dominates overall** due to GBM asymmetry: uptrend gains exceed downtrend losses because `exp(+drift + σ²/2) > |exp(−drift + σ²/2)|`
- **Random ordering reduced AlwaysLong variance** from ±353 to ±35 — most single-seed variance was ordering artifacts
- **VolatilityScaled goes neutral in High Volatility** by design, so its lag reads as "never recovered" — this is correct behavior, not failure
- **MeanReversion has a ~2-step lag** even in its home regime — signal computation introduces a small delay
- **Reward distribution**: strategies with low % positive steps (MeanReversion ~27%, VolatilityScaled ~24%) are frequently neutral, not frequently losing

---

## Reward Formula

```
reward = position × return × 100
```

Where:
- `position` ∈ {+1 (Long), 0 (Neutral), −1 (Short)}
- `return` = (next_price − current_price) / current_price
- ×100 scaling keeps rewards in a readable range (~±1 per step)

Strategies observe only price history — regime labels are never revealed.
