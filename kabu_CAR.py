import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm

# Define a function to calculate cumulative abnormal returns (CAR)
def calculate_car(stock_symbol, event_dates, window_size):
    stock_data = yf.download(stock_symbol, start="2000-01-01")
    stock_data['Return'] = stock_data['Adj Close'].pct_change()
    stock_data = stock_data.dropna()

    market_data = yf.download('^GSPC', start="2000-01-01")
    market_data['Return'] = market_data['Adj Close'].pct_change()
    market_data = market_data.dropna()

    car_results = {}

    for event_date in event_dates:
        event_date = pd.to_datetime(event_date)
        start_date = event_date - pd.Timedelta(days=window_size)
        end_date = event_date + pd.Timedelta(days=window_size)
        
        stock_window_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
        market_window_data = market_data[(market_data.index >= start_date) & (market_data.index <= end_date)]

        # Check if the lengths match
        if len(stock_window_data) != len(market_window_data):
            continue

        X = market_window_data['Return'].values
        Y = stock_window_data['Return'].values
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()
        abnormal_returns = Y - model.predict(X)
        car = np.cumsum(abnormal_returns)

        car_results[event_date.strftime('%Y-%m')] = car

    return car_results

# Event dates (example)
event_dates = [
    "2012-06-01",
    "2012-12-01",
    "2013-06-01",
    "2013-12-01",
    "2014-06-01",
    "2014-12-01",
    "2015-06-01",
    "2015-12-01",
    "2016-06-01",
    "2016-12-01",
    "2017-06-01",
    "2017-12-01"
]

# Parameters
window_size = 60  # Window size (days)

# Calculate CAR for the stock symbol (example: AAPL)
car_results = calculate_car('AAPL', event_dates, window_size)

# Plot the results
plt.figure(figsize=(12, 8))
for event_date, car in car_results.items():
    # Adjust x axis values based on the length of CAR data
    x_values = range(-len(car) + 1, len(car))
    plt.plot(x_values, car, label=event_date)

plt.title('Cumulative Abnormal Returns (CAR)')
plt.xlabel('Days Relative to Event')
plt.ylabel('CAR')
plt.legend()
plt.grid(True)
plt.show()
