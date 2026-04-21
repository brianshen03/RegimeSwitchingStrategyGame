# RegimeSwitchingStrategyGame

## Title  
**Strategy Performance in Regime-Switching Environments**

---

## Prompt  
I will build a simulation that compares different decision-making strategies in an environment where conditions change over time (“regimes”). In each round, strategies produce outcomes based on the current environment, and their performance depends on how well they match the current conditions.

I will compare fixed strategies, which follow the same rule consistently, to adaptive strategies that adjust their choices based on past performance. The goal is to understand how different approaches to decision-making perform when conditions are not stable.

---

## Key Research Questions
- How do fixed strategies perform when conditions change over time?  
- Do adaptive strategies outperform fixed ones, and under what circumstances?  
- How quickly must a strategy adjust to remain effective?  
- What tradeoffs exist between consistency and adaptability?  

---

## High-Level Approach
- Simulate repeated rounds where strategies generate outcomes based on the current environment  
- Define multiple regimes where different strategies are favored  
- Introduce changes in regimes over time  
- Evaluate strategies using:
  - cumulative reward  
  - consistency of performance  
  - responsiveness to changing conditions  

---

## Execution Plan

### Week 5 — Core Framework + Baseline Strategies
- Build simulation framework for repeated decision-making  
- Implement a set of fixed strategies  
- Define basic regimes with different reward patterns  
- Track and store performance data across runs  

**Goal:** establish a system for comparing how different strategies perform under fixed conditions  

---

### Week 7 — Adaptive Strategies + Changing Conditions
- Implement adaptive strategies that adjust based on past outcomes  
- Introduce regime switching over time  
- Run structured experiments comparing fixed and adaptive approaches  
- Measure responsiveness and performance changes  

**Goal:** analyze how adaptability affects performance when conditions change  

---

### Final — Full Analysis + Insights
- Run experiments across multiple regime patterns and frequencies  
- Aggregate results across runs to identify consistent trends  
- Create visualizations:
  - performance over time  
  - comparison across strategies  
  - response to regime changes  
- Analyze tradeoffs between consistency and adaptability  

---