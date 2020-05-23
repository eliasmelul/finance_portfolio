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

<img src="https://i.ibb.co/n62cwmm/sharpe.png" alt="sharpe" border="0" width="150" height="56">

        When Sharpe > 1, GOOD risk-adjusted returns
    
        When Sharpe > 2, VERY GOOD risk-adjusted returns
    
        When Sharpe > 3, EXCELLENT risk-adjusted returns


_How do we measure risk?_ There are many ways to measure risk, although variance (standard deviation) is one of the most common. However, when it comes to the risk that cannot be avoided through diversification, the Beta is king!

**Beta**: measures the market risk that cannot be avoided through diversification. This is the relationship between the stock and the market portfolio. In other words, it is a measure of how much risk the investment will add to a portfolio that looks like the market.

<img src="https://i.ibb.co/Z18dwRn/beta.png" alt="beta" border="0" width="100" height="56">

        When beta = 0, it means that there's no relationship.
    
        When beta < 1, it means that the stock is defensive (less prone to high highs and low lows)
    
        When beta > 1, it means that the stock is aggresive (more prone to high highs and low lows)
        
Amazing! We're only one small step away. The risk-adjusted returns. 

**Expected Return CAPM**: calculates the expected return of a security adjusted to the risk taken. This equates to the return expected from taking the extra risk of purchasing this security.

<img src="https://i.ibb.co/FnCSkDs/return.png" alt="return" border="0" width="200" height="40">

Awesome! There are a couple more things we will discuss later, but for now, now that we understand the underlying theory of the CAPM model, let's get coding!

```
# Import libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as wb

# Load stock and market data
tickers = ['AMZN','^GSPC']
data = pd.DataFrame()
for t in tickers:
    data[t] = wb.DataReader(t, data_source='yahoo', start='2010-1-1')['Adj Close']
    
#Calculate logarithmic daily returns
sec_returns = np.log(data / data.shift(1))

# To calculate the beta, we need the covariance between the specific stock and the market...
cov = sec_returns.cov() *252 #Annualize by multiplying by 252 (trading days in a year)
cov_with_market = cov.iloc[0,1]
# ...we also need the variance of the daily returns of the market
market_var = sec_returns['^GSPC'].var()*252

# Calculate Beta
amazon_beta = cov_with_market / market_var
```

Before calculating the expected risk-adjusted return, we must clarify a couple assumptions:
1. A 10 year US government bond is a good proxy for a risk-free asset, with a yield of 2.5%
2. The common risk premium is between 4.5% and 5.5%, so we will use 5%. Risk premium is the expected return of the market minus the risk-free return.

```
riskfree = 0.025
riskpremium = 0.05
amazon_capm_return = riskfree + amazon_beta*riskpremium
```
This yields an annualized risk-adjusted return of 7.52% (as per May 23rd 2020). Let's try the same procedure with the arithmetic mean of the returns of the market, instead of assuming a risk premium of 5%.

```
riskfree = 0.025
riskpremium = (sec_returns['^GSPC'].mean()*252) - riskfree
amazon_capm_return = riskfree + amazon_beta*riskpremium
```
This yields a 9.26% return - a considerable significant change.

Last but not least, the Sharpe Ratio!
```
log_returns = np.log(data / data.shift(1))
sharpe_amazon = (amazon_capm_return-riskfree)/(log_returns['AMZN'].std()*250**0.5)
```

Great! Now that we have demonstrated how to compute the metrics derived from the CAPM, let's make it into a convenient way of using it.
