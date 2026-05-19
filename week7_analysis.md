REGIMES 

Trending upward 
 - prices incrementally move upward

 Mean reversion
 - price keeps osciliating around mean, it getse pulled back toward mean whenever it strays too far.

 high volatility
 - no trend, just randomness and chaos -> large price spikes and down turns 

STRATEGIES 

Always long
    - making money in trending upward
    - goes pretty much even in mean reversion and volatility
    - with mean reversion, it goes even
    - with high volatility, it does too , but this is due to noise , and I doubt this would happen over multiple runs

Always short
    - the exact opposite reward of always long 
        - losing a lot of money in trending upward

MA_Crossover (goes long if short term avg is higher than long term avg, goes short if short term avg is lower than long term avg )
    - going to make money in upward trends since short term average will keep being above long term average
    - in a mean reversion regime , it will sway if short term trends are above and below long term trends, it will eventually bounce back to mean
    - in high volatility, it lost money due to random nature of the price spikes 

Momentum (discretionary - calculates "momentum" - average return over a window, long if positive momentum , short if negative, neutral if weak)
    - going to perform well in upward trends - momentum keeps going upward
    - loses money in mean reversion - since it chases moves that keep reversing in mean reversion regime 
    - lost money in volatility - results would change? 


Mean reversion (discretionary - bets that a trend will reverse)
    - suffering from trending regime
        - trending keeps going up , mean reversion keeps shorting thinking it will go down
    - is making money in mean reverting regime
    - is pretty much going even in volatile 
    - overall a loss 


Volatility scaled (similar to momentum, but calculates volatilty and will not short/long if high volatility)
    - same to momentum, performs well in upward trends 
    - lost money in mean reversion - noise? not sure if it would losses would be reproducible among multiple runs
    - made money in volatility - filter working as intended as its only making money when price steps are somewhat stable 



Explain how strategies adapt based on regime , why they behave the way they do 

Also document well how reward structures change with regimes, how different strategies adapt etc

how do each of them adapt, what are the trade-offs? Also how does action selection change? You should also be able to describe when/when not a strategy is aligned with the regime or when there is a lag.