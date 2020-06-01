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
3. Portfolio Optimization
4. Predicting Stock Prices: Monte Carlo Simulations Automated (coded to an easy-to-use function)
5. Predicting Stock Prices: Monte Carlo Simulations with Cholesky Decomposition (as a means to correlate returns)
6. Predicting Stock Prices: Monte Carlo Simulations with Cholesky Automated (coded to an easy-to-use function)
7. Introduction to Time Series: ETS, EWMA, ARIMA, ACF, PACF
7. Momentum Metrics: Relative Strength Index and MACD
8. Predicting Option Prices: Introduction to Black-Scholes-Merton
9. Predicting Option Prices: Monte Carlo Simulations with Euler Discretization
10. Introduction to Quantopian

## Capital Asset Pricing Model
The CAPM model describes the relationship between expected returns and volatility (systematic risk). Why does this matter? Because investors expect to be compensated for risk and time value of money. So the CAPM is used as a theoretical model that adjusts for risk when evaluating the value of a stock.

This model assumes the existance of a market portfolio - all possible investments in the world combined - hence the existance of a risk-free asset. However, this is not true. It also assumes that all investors are rational, and therefore hold the optimal portfolio. This is in consequence of the mutual fund theorem: _all investors hold the same portfolio of risky assets, the tangency portfolio_. Therefore, the CAPM assumes that the tangency portfolio is the market portfolio. Again... not necessarily true. However, the model is great for understanding and conceptualizing the intricacies of risk in investing and the concept of diversification, so let's continue.

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

```
from datetime import datetime
#Import the data of any stock of set of stocks
def import_stock_data(tickers, start = '2010-1-1', end = datetime.today().strftime('%Y-%m-%d')):
    data = pd.DataFrame()
    if len([tickers]) ==1:
        data[tickers] = wb.DataReader(tickers, data_source='yahoo', start = start)['Adj Close']
        data = pd.DataFrame(data)
    else:
        for t in tickers:
            data[t] = wb.DataReader(t, data_source='yahoo', start = start)['Adj Close']
    return(data)

# Compute beta function   
def compute_beta(data, stock, market):
    log_returns = np.log(data / data.shift(1))
    cov = log_returns.cov()*250
    cov_w_market = cov.loc[stock,market]
    market_var = log_returns[market].var()*250
    return cov_w_market/market_var

#Compute risk adjusted return function
def compute_capm(data, stock, market, riskfree = 0.025, riskpremium = 'market'):
    log_returns = np.log(data / data.shift(1))
    if riskpremium == 'market':
        riskpremium = (log_returns[market].mean()*252) - riskfree
    beta = compute_beta(data, stock, market)
    return (riskfree + (beta*riskpremium))
   
#Compute Sharpe Ratio
def compute_sharpe(data, stock, market, riskfree = 0.025, riskpremium='market'):
    log_returns = np.log(data / data.shift(1))
    ret = compute_capm(data, stock, market, riskfree, riskpremium)
    return ((ret-riskfree)/(log_returns[stock].std()*250**0.5))
    
# All in one function
def stock_CAPM(stock_ticker, market_ticker, start_date = '2010-1-1', riskfree = 0.025, riskpremium = 'set'):
    data = import_stock_data([stock_ticker,market_ticker], start = start_date)
    beta = compute_beta(data, stock_ticker, market_ticker)
    capm = compute_capm(data, stock_ticker, market_ticker)
    sharpe = compute_sharpe(data, stock_ticker, market_ticker)
    #listcapm = [beta,capm,sharpe]
    capmdata = pd.DataFrame([beta,capm,sharpe], columns=[stock_ticker], index=['Beta','Return','Sharpe'])
    return capmdata.T
    
stock_CAPM("AAPL","^GSPC")
```
<img src="https://i.ibb.co/1XzZtDj/aapl.png" alt="aapl" border="0" width="300" height="70" style="margin: auto">
    
## Monte Carlo Simulations
Monte Carlo Simulations are an incredibly powerful tool in numerous contexts, including operations research, game theory, physics, business and finance, among others. It is a technique used to understand the impact of risk and uncertainty when making a decision. Simply put, a Monte Carlo simulation runs an enourmous amount of trials with different random numbers generated from an underlying distribution for the uncertain variables.

Here, we will dive into how to predict stock prices using a Monte Carlo simulation!

**What do we need to understand before we start?**

<img src="https://i.ibb.co/0cQnBkn/Basic-Pricing.png" alt="aapl" border="0" width="300" style="margin: auto">


* We know yesterday's price. 

* We want to predict today's price. 

* What we do not know is the rate of return, r, of the share price between yesterday and today. 

This is where the Monte Carlo simulation comes in! But first, how do we compute the return?

### Brownian Motion

Brownian motion will be the main driver for estimating the return. It is a stochastic process used for modeling random behavior over time. For simplicity, we will use regular brownian motion, instead of the Geometric Brownian Motion, which is more common and less questionable in stock pricing applications.

**Brownian Motion** has two main main components:
1. Drift - the direction that rates of returns have had in the past. That is, the expected return of the stock.

<img src="https://i.ibb.co/TqHmzJy/drift.png" alt="drift" border="0" width = "200">
    Why do we multiply the variance by 0.5? Because historical values are eroded in the future.
    

2. Volatility -  random variable. This is the historical volatility multiplied by a random, standard normally distributed variable.

<img src="https://i.ibb.co/0QjBvtx/Volatility.png" alt="Volatility" border="0" width = "350">

Therefore, our asset pricing equation ends up looking like this:

<img src="https://i.ibb.co/1M7MPc6/Pricing-Eq.png" alt="Pricing-Eq" border="0" width = "400">


This technique will be used for every day into the future you want to predict, and for however many trials the monte carlo simulation will run!

---

First, import required libraries.
```
#Import libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm, gmean, cauchy
import seaborn as sns
from datetime import datetime

%matplotlib inline
```
Import data for one or multiple stocks from a specified date until the last available data. Data source: yahoo finance.

For this, it's better if we define a function that imports stock(s) daily data for any publicly traded company as defined by the user starting at a user-defined date until today. We will use the Adjusted Close price. We will continue using Amazon as a running example.

```
#Import stock data function
ticker = 'AMZN'
data = pd.DataFrame()
data[ticker] = wb.DataReader(ticker, data_source='yahoo', start='2010-1-1')['Adj Close']
data.head()
 
#Compute log or simple returns
def log_returns(data):
    return (np.log(1+data.pct_change()))
    
def simple_returns(data):
    return ((data/data.shift(1))-1)

log_return = log_returns(data)

#Plot returns histogram for AMZN
sns.distplot(log_returns.iloc[1:])
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
```
<img src="https://i.ibb.co/bgGt1zQ/AMZNREts.png" alt="AMZNREts" border="0" width = "500">

```
data.plot(figsize=(15,6))
```
<img src="https://i.ibb.co/cJ4ttFG/AMZNStock.png" alt="AMZNStock" border="0" width = "700">

Great! Next, we have to calculate the Brownian Motion with randomly generated returns. 
```
#Calculate the Drift
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)
```
Before we generate the variable portion of the generated returns, I'll show you how to generate uncorrelated random daily returns. Note that correlated returns are very valuable when discussing derivates that are based on an underlying basket of assets. We will discuss the Cholesky Decomposition method in a later. 
```
#Returns random variables between 0 and 1
x = np.random.rand(10,2)

#Percent Point Function - the inverse of a CDF
norm.ppf(x)
```
Using these, we can generate random returns. For example, we can run 1,000 iterations of random walks consisting of 50 steps (days). Now we can generate the variable part of the Brownian Motion.
```
#Calculate standard deviation of returns
stddev = log_returns.std()
#Calculate expected daily returns for all of the iterations
daily_returns = np.exp(drift.values + stddev.values * norm.ppf(np.random.rand(50,1000)))
```
So close! Now that we have randomly generated 50 random returns for every one of the ten thousand trials, all we need is to calculate the price path for each of the trials! 
```
# Create matrix with same size as daily returns matrix
price_list = np.zeros_like(daily_returns)

# Introduce the last known price for the stock in the first item of every iteration - ie Day 0  for every trial in the simulation
price_list[0] = data.iloc[-1]

# Run a loop to calculate the price today for every simulation based on the daily returns generated
for t in range(1,50):
    price_list[t] = price_list[t-1]*daily_returns[t]
```
Voila! We have officially finished the Monte Carlo simulation to predict stock prices. Let's see how it looks!

The first 30 simulations:

<img src="https://i.ibb.co/C62LZ76/amazonpred.png" alt="amazonpred" border="0" width = "700">

The histogram of the final prices for each simulation:

<img src="https://i.ibb.co/rkVz4kK/Amznpredhist.png" alt="Amznpredhist" border="0" width = "500">

With these predictions, we can now calcualte Value at Risk, or simply the probability of a certain event occuring and the expected annualized return. We will do this once we create automated versions of this process that can handle multiple stocks and reports certain metrics, included the aforementioned and other CAPM metrics! 

