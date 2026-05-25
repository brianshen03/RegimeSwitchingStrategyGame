WHAT WAS ADDED SINCE WEEK 7

Trending Down regime (regime 1)
    - added as a 4th regime using GBM with negative drift (-0.001)
    - price series now cycles through: trending up, trending down, mean-reverting, high volatility
    - regime order and lengths are configurable

Multi-seed analysis
    - previously all results came from a single price series (seed=42)
    - can't distinguish consistent findings from lucky/unlucky outcomes on one seed
    - now running 30 seeds per experiment

Three experiment types
    - fixed order, fixed length: same regime sequence every seed, different price paths
    - random order, fixed length: regime sequence shuffled each seed, lengths stay at 500 each
    - random order, random length: both sequence and duration randomized each seed, total always 2000 steps (min 100 per regime)


WHAT MULTI-SEED REVEALED

seed=42 was misleading
    - MA_Crossover looked like best strategy on seed=42 (+59.7)
    - across 30 seeds: mean +8.59, std ±429 → wildly inconsistent, not reliable
    - seed=42 just happened to be a lucky run for it

ordering effects inflated variance
    - fixed order: AlwaysLong mean +150, std ±353
    - random order: AlwaysLong mean +32, std ±35
    - most of the variance in fixed order was just from always seeing the uptrend first
    - random order gives a much more honest picture of true strategy consistency

VolatilityScaled is the most consistent strategy
    - fixed order std: ±29 (tightest of all strategies by far)
    - random order std: ±31
    - doesn't make the most money but doesn't blow up either


REGIME × STRATEGY PERFORMANCE (heatmap, 30 seeds random order)

AlwaysLong
    - best overall performer despite being the simplest strategy
    - wins in trending up, loses in trending down, roughly neutral in mean-reverting and volatile
    - why it wins overall → GBM asymmetry (see below)

AlwaysShort
    - exact mirror of AlwaysLong

MA_Crossover
    - profitable in both trending regimes (detects direction correctly)
    - struggles in mean-reverting: keeps flipping direction as short-term average crosses long-term
    - struggles in high volatility: random crossovers generate noise trades

Momentum
    - profits in trending regimes (signal aligns with direction)
    - loses in mean-reverting: chases moves that immediately reverse
    - goes neutral in weak signals, so less exposure in volatile

MeanReversion
    - best performer in mean-reverting regime (designed for it)
    - consistently loses in trending regimes: keeps shorting a rising trend or going long a falling one
    - positive overall mean in random order experiments (vs negative in fixed order) → was being punished by always seeing the uptrend first

VolatilityScaled
    - similar to momentum in trending regimes
    - steps back to neutral in high volatility (the filter works as intended)
    - slightly underperforms pure momentum in trending because it sometimes misidentifies trending as volatile


WHY ALWAYSLONG WINS (GBM asymmetry)

trending up and trending down have equal magnitude drifts (+0.001 and -0.001)
    - but GBM is multiplicative, so expected arithmetic return is NOT symmetric
    - trending up: E[return per step] ≈ +0.00105
    - trending down: E[return per step] ≈ -0.00095
    - uptrend gains slightly more than downtrend loses (Jensen's inequality / Ito correction)

over 500 steps × 2 trending regimes → roughly +5 total reward just from the asymmetry
    - mean-reverting and high volatility both have zero expected return by construction
    - so AlwaysLong quietly collects the asymmetry without giving it away by going short or neutral
    - every smarter strategy partially sacrifices this by switching to short or neutral at some point

mirrors the equity premium in real markets
    - passive buy-and-hold often beats active strategies for exactly this reason


LAG ANALYSIS (adaptive strategies only, home regimes only, 30 seeds random order)

what it measures
    - after entering a regime a strategy is designed for, how many steps until rolling reward turns positive
    - measures detection/adaptation speed, not just outcome
    - only measured for meaningful pairings (e.g., Momentum in trending, not Momentum in mean-reverting)

MA_Crossover
    - trending up: mean lag ~9 steps
    - trending down: mean lag ~6 steps
    - slightly faster detecting downtrends → short signal is sharper since moving averages fall more crisply than they rise

Momentum
    - trending up: ~10 steps
    - trending down: ~5 steps
    - same pattern as MA_Crossover, detects downtrends faster

MeanReversion
    - mean-reverting: ~2 steps → fastest adaptation of any strategy in any regime
    - z-score signal fires almost immediately once prices start oscillating

VolatilityScaled
    - high volatility: mean lag ~80 steps, never recovers in 25/30 seeds
    - NOT a failure → strategy correctly goes neutral, earns 0 instead of losing
    - 0 reward never crosses "positive", so lag metric reads it as not recovering
    - demonstrates the limit of reward-based lag: can't distinguish "didn't adapt" from "adapted by doing nothing"

random length vs fixed length lag
    - shorter regime segments reduce how many steps are available to recover
    - VolatilityScaled high volatility lag jumps from ~80 to ~130 with random lengths → some seeds get very short volatile segments where it never has time to turn profitable even if it tried


REWARD DISTRIBUTION (does best strategy win by big gains or loss avoidance?)

answer: neither → everyone wins or loses by frequency, not magnitude
    - mean gain ≈ mean loss for all strategies (~1.2 per step)
    - no strategy is making outsized wins to compensate for frequent losses
    - AlwaysLong, AlwaysShort, MA_Crossover, Momentum: all ~50% positive steps

MeanReversion and VolatilityScaled look different
    - only 27% and 24% positive steps respectively
    - NOT because they lose more often → because they frequently go neutral (earning exactly 0)
    - zero reward steps count as neither positive nor negative, pulling % positive down
    - gain and loss magnitudes when they do act are similar to other strategies


OPEN QUESTIONS

directional lag
    - does coming FROM volatile make it harder to adapt TO trending, vs coming from trending to trending?
    - would need more seeds (~100+) to get reliable estimates per (from, to) transition pair

learning strategy
    - would an epsilon-greedy or Q-learning strategy do better than rule-based adaptive strategies?
    - most interesting question: does it learn to detect regime switches faster than Momentum or MeanReversion?
