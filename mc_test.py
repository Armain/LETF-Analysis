import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import time

def get_data(stocks, start, end):
    all_data = []

    for ticker in stocks:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start, end=end, auto_adjust=False)['Close']
            if not data.empty:
                all_data.append(data.rename(ticker))
            time.sleep(1.5)
        except:
            continue

    if not all_data:
        raise ValueError("Failed to download any stock data. Yahoo Finance may be blocking requests.")

    stock_data = pd.concat(all_data, axis=1)
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

stocks = ['AAPL', 'AMZN', 'TSLA', 'META', 'GOOG', 'PLTR']
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

mean_returns, cov_matrix = get_data(stocks, start_date, end_date)

#%%
n = len(mean_returns)
weights = np.random.random(n)
weights /= np.sum(weights)

init = 1000
mc_sims = 100
T = 1000

mean_M = np.full(shape=(T, n), fill_value=mean_returns)
mean_M = mean_M.T

portoflio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

L = np.linalg.cholesky(cov_matrix)

for m in range(mc_sims):
    Z = np.random.normal(size=(T, n))
    daily_returns = mean_M + np.inner(L, Z)
    weighted_returns = np.inner(weights, daily_returns.T)
    portoflio_sims[:, m] = np.cumprod(weighted_returns + 1) * init

plt.figure(figsize=(10, 6))
plt.plot(portoflio_sims, alpha=0.4, linewidth=0.5, color='gray')

p50 = np.percentile(portoflio_sims, 50, axis=1)
p5 = np.percentile(portoflio_sims, 5, axis=1)
p95 = np.percentile(portoflio_sims, 95, axis=1)

plt.plot(p50, 'r-', linewidth=2, label='Median')
plt.plot(p5, 'b--', linewidth=1.5, label='5th percentile')
plt.plot(p95, 'g--', linewidth=1.5, label='95th percentile')

plt.xlabel('Days')
plt.ylabel('Portfolio Value ($)')
plt.title(f'Monte Carlo Simulation ({mc_sims} paths, {T} days)')
plt.legend()
plt.ylim(0, p95[-1] * 1.2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
