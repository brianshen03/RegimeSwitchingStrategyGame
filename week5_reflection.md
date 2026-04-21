## What you have achieved so far

So far, I have built the MVP core simulation framework for my project. This includes implementing a regime-based environment that generates rewards, a set of baseline strategies, and a simulator that runs repeated interactions between a policy and the environment. I also created a metrics module to evaluate performance and an experiments pipeline to run and compare different strategies under fixed conditions.

At this stage, I can run a complete experiment where multiple baseline strategies (such as static, random, and round-robin policies) are evaluated in a controlled environment. The system collects reward data and produces summary metrics like total and average reward.

---

## What you are happy with, from your project work so far

I am most happy with the modular structure of the project. Each component has a clear responsibility:

- the environment handles reward generation and regimes  
- policies define decision-making behavior  
- the simulator manages the interaction loop  
- metrics handle evaluation  
- experiments coordinate everything  

This separation makes the system easy to reason about and extend. For example, I can add new strategies or change the environment without modifying the rest of the code.

I am also satisfied that the results match expected behavior under fixed conditions. For example, when the regime is fixed, the strategy that always selects the best action performs the best, while others perform worse. This confirms that the framework is working correctly and gives me confidence to build on it.

---

## What you are struggling with, or what challenges you are facing next

One of the main challenges is moving from fixed strategies to adaptive strategies. Implementing policies that respond to changing rewards requires more careful design, especially in terms of how they store and use past information. I want to make sure these strategies are implemented in a way that is both correct and comparable.

Another challenge is extending the environment to include regime switching and analyzing how quickly different strategies adapt. This introduces more complexity, especially when trying to measure things like adaptation lag and consistency.

---

## How to Run

Run the baseline experiment with:

```bash
python3 experiments.py