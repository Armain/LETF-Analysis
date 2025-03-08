# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:52:09 2025

@author: Armain
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime

import numpy as np

start_date = "1928-08-14"

window_sizes = np.arange(10, 500, 10)

sp500_file_path = 'D:/Schoolwork - McGill/Other/LETF Analysis/sp500.csv'
sp500 = pd.read_csv(sp500_file_path, index_col='Date', parse_dates=True)
sp500.columns =['SPY']

rotation_signals = pd.DataFrame()
trade_count = pd.DataFrame(columns = ['Trade Count'], index=window_sizes)

for w in window_sizes:
    sp500[f'{w} SPY_MA'] = sp500['SPY'].rolling(window=w).mean()
sp500 = sp500.loc[start_date:]

for w in window_sizes:
    raw_signal = (sp500['SPY'] > sp500[f'{w} SPY_MA']).astype(int)
    
    rotation_signals[w] = pd.concat([raw_signal.iloc[:1], raw_signal.iloc[:-1]], ignore_index=True)
    trade_count['Trade Count'].loc[w] = abs(rotation_signals[w].diff()).sum()

rotation_signals.index = sp500.index


rotation_signals = rotation_signals.loc[start_date:]

trade_count.plot()




