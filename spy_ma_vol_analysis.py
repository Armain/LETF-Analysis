# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:52:09 2025

@author: Armain
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start_date = "1928-08-14"

window_sizes = np.arange(10, 500, 10)

sp500_file_path = 'portfolio_data/spy.xlsx'
sp500 = pd.read_excel(sp500_file_path, index_col='Date', parse_dates=True)
sp500.columns =['SPY']

rotation_signals = pd.DataFrame()
trade_count = pd.DataFrame(columns = ['Trades per Year'], index=window_sizes)

for w in window_sizes:
    sp500[f'{w} SPY_MA'] = sp500['SPY'].rolling(window=w).mean()
sp500 = sp500.loc[start_date:]
sp500.index = pd.to_datetime(sp500.index)

n_years = (sp500.index[-1] - sp500.index[0]).days / 365.25

for w in window_sizes:
    raw_signal = (sp500['SPY'] > sp500[f'{w} SPY_MA']).astype(int)

    rotation_signals[w] = pd.concat([raw_signal.iloc[:1], raw_signal.iloc[:-1]], ignore_index=True)
    total_trades = abs(rotation_signals[w].diff()).sum()
    trade_count['Trades per Year'].loc[w] = total_trades / n_years

rotation_signals.index = sp500.index

rotation_signals = rotation_signals.loc[start_date:]

trade_count.plot(title='Average Trades per Year vs MA Window Size',
                 xlabel='MA Window Size (days)',
                 ylabel='Trades per Year',
                 grid=True)
plt.show()




