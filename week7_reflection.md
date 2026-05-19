## What you have achieved so far

Building on the week 5 framework, I extended the simulation to have it grounded in a specific domain space. 

First, I transitioned from an abstract bandit-style environment to a finance-grounded price simulation. The new environment generates realistic price series across three distinct regimes — trending, mean-reverting, and high volatility. Rewards are now computed as position * return rather than sampled from a means matrix.

Second, I implemented a set of systematic and adaptive strategies. Systematic strategies include AlwaysLong, AlwaysShort, AlwaysNeutral, and MovingAverageCrossover. Adaptive strategies include MomentumStrategy, MeanReversionStrategy, and VolatilityScaledStrategy, each computing signals directly from price history rather than reward history.
I also extended the visualization pipeline to produce cumulative reward plots, rolling average reward plots, action distribution breakdowns per regime, and a price series plot, all using the same seed for direct comparability.

---

## What you are happy with, from your project work so far

I am most happy with the domain space and the transition from an abstract multi-armed bandit game to a price based environment. Additionally, it's cool to learn the math and 
relationships between regimes and strategies, so I'm able to generate actual conclusions and trends behind why a strategy is performing well or badly. 

The debugging process has also been insightful. For example, at a certain iteration, VolatilityScaledStrategy was going neutral during the trending regime which should not be happening - and by printing out volatility values and comparing them to the treshhold, I saw that the treshhold value was too low to trigger the strategy to make any meaningful moves - so it kept going neutral. 

And for the momentum strategy, the same thing happened in the trending regime - where it was unable to detect any momentum since the drift (price increase) was too low relative to the noise, so the momentum signal never exceeded the threshold. So by increasing the drift, we were able to trigger the momentum strategy. 

I am also happy with the continued modularity of the project, where I'm able to separate each file into different folders, and the graph generation from the experiment which is 
able to paint a nice visual of how each strategy does through each regime. 


---

## What you are struggling with, or what challenges you are facing next

Right now, the issue is that a strong uptrend for 200 steps makes AlwaysLong nearly unbeatable, which drowns out the interesting strategy differences. 

For next steps, I want to start messing around with the environment more - different regime lengths, more regimes, adding a trending downward regime, running multiple seeds, etc..

So for the most part, the logic is finished implemented, now the main challenge going forward is moving from single run results to statistically meaningful conclusions. Currently every result comes from a single price series (seed=42), which means I cannot distinguish consistent findings from lucky or unlucky outcomes.

Finally, the lag analysis - measuring how many steps each adaptive strategy takes to detect a regime switch and adjust behavior — is still unimplemented. 

---

## How to Run

Run the experiment with:

```bash
python analysis/experiment.py