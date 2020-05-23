___

<a href='https://github.com/eliasmelul/'> <img src='https://s3.us-east-2.amazonaws.com/wordontheamazon.com/NoMargin_NewLogo.png' style='width: 15em;' align='right' /></a>
# Finance with Python

___
<h4 align="right">by Elias Melul, Data Scientist </h4> 

___

# Table of Contents

In this repository, I will include:

1. Introduction to CAPM, Beta and Sharpe Ratio 
2. Introduction to Monte-Carlo Simulations
3. Predicting Stock Prices: Monte Carlo Simulations Automated (coded to an easy-to-use function)
4. Predicting Stock Prices: Monte Carlo Simulations with Cholesky Decomposition (as a means to correlate returns)
5. Predicting Stock Prices: Monte Carlo Simulations with Cholesky Automated (coded to an easy-to-use function)
6. Predicting Option Prices: Introduction to Black-Scholes-Merton
7. Predicting Option Prices: Monte Carlo Simulations with Euler Discretization
8. Optimal Porfolio

## Capital Asset Pricing Model
The CAPM model describes the relationship between expected returns and volatility (systematic risk). Why does this matter? Because investors expect to be compensated for risk and time value of money. So the CAPM is used as a theoretical model that adjusts for risk when evaluating the value of a stock.

This model assumes the existance of a market portfolio - all possible investments in the world combined - hence the existance of a risk-free asset. However, this is not true. It also assumes that all investors are rational, and therefore hold the optimal portfolio. This is in consequence of the mutual fund theorem: _all investors hold the same portfolio of risky assets, the tangency portfolio_. Therefore, the CAPM assumes that the tangency portfolio is the market portfolio. Again... not true.

In this context, the tangency portfolio is the portfolio with the largest Sharpe Ratio. But what is the _Sharpe Ratio?_


**Sharpe Ratio**: measures the performance of a security compared to a risk-free asset, after adjusting for its risk. This is the excess return per unit of risk of an investment.
$$
Sharpe = \frac{\overline{r_{i}} - r_f}{\sigma_{i}}
$$
        When Sharpe > 1, GOOD risk-adjusted returns
    
        When Sharpe > 2, VERY GOOD risk-adjusted returns
    
        When Sharpe > 3, EXCELLENT risk-adjusted returns


_How do we measure risk?_ There are many ways to measure risk, although variance (standard deviation) is one of the most common. However, when it comes to the risk that cannot be avoided through diversification, the Beta is king!

**Beta**: measures the market risk that cannot be avoided through diversification. This is the relationship between the stock and the market portfolio. In other words, it is a measure of how much risk the investment will add to a portfolio that looks like the market.
$$ 
\beta_{i} = \frac{\sigma_{i,m}}{\sigma_{m}^2}
$$

        When beta = 0, it means that there's no relationship.
    
        When beta < 1, it means that the stock is defensive (less prone to high highs and low lows)
    
        When beta > 1, it means that the stock is aggresive (more prone to high highs and low lows)
        
Amazing! We're only one small step away. The risk-adjusted returns. 

**Expected Return CAPM**: calculates the expected return of a security adjusted to the risk taken. This equates to the return expected from taking the extra risk of purchasing this security.
$$
\overline{r_{i}} = r_f + \beta_{i}(\overline{r_{m}} - r_f) 
$$

Awesome! There are a couple more things we will discuss later, but for now, now that we understand the underlying theory of the CAPM model, let's get coding!

