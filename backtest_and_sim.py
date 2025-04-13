
import sys
import math
import random

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

cwd = Path(__file__).parent
sys.path.append(str(cwd))

'''These are the variables you can change to adjust the simulation.'''

voo_er = 0.0003
upro_er = 0.0091
sso_er = 0.0089
 
voo_lr = 1
sso_lr = 2
upro_lr = 3

borrowing_spread = 0.007
trading_days = 252 
start_price = 50000
starting_value = 10000
rf_rate = 1.03 ** (1 / trading_days) - 1

monthly_addition = 10000
days_simulated = (trading_days * 15) 
number_of_simulations = 15000
    
#%%
'''SETUP SECTION'''

window_size = 200

tickers = ['spy', 'sso', 'upro', 'ffr', 'goldx', 'zroz', 'hfea']

# Load all files in one-liner dictionary comprehension
prices_dfs = {ticker: pd.read_excel(cwd / 'portfolio_data' / f'{ticker}.xlsx', 
                           index_col='Date', parse_dates=True) for ticker in tickers}

# Keep price series and create returns
sp500_price = prices_dfs['spy'].copy() 
pcnt_dfs = {ticker: df.pct_change().dropna() for ticker, df in prices_dfs.items()}

# Calculate MA and assign variables
sp500_price['SPY_MA'] = sp500_price['SPY'].rolling(window=window_size).mean()
spy, sso, upro, ffr, gld, zroz, hfea = [pcnt_dfs[ticker] for ticker in tickers]

for df in [sp500_price, spy, sso, upro, ffr, gld, zroz]:
    df.index = pd.to_datetime(df.index, format='mixed')

#%% See SPY moving average    

sp500_price['SPY_MA'] = sp500_price['SPY'].rolling(window=window_size, min_periods=1).mean()
sp500_price.plot(logy=True)

#%%
# Define ETF Expense and Leverage Ratios
etf_params = {
    'VOO': {'lr': 1, 'er': 0.0003},
    'SSO': {'lr': 2, 'er': 0.0089},
    'UPRO': {'lr': 3, 'er': 0.0091}
}

daily_pcnt = pd.concat([spy, ffr, gld, zroz], axis=1)
 
'''Below we calculate the expected returns of the SSO and UPRO considering 
the borrowing costs and the expense ratios. We then calculate the isolated 
tracking error that is caused from algorithms and optimisation.'''

for etf, params in etf_params.items():
    daily_pcnt[etf] = ((daily_pcnt['SPY'] * params['lr']) 
                      - ((params['lr'] - 1) * (daily_pcnt['FFR'] + borrowing_spread / trading_days))
                      - ((1 + params['er']) ** (1/trading_days) - 1))

#%%
''''Compare calclated expected returns with actual returns to find tracking errors'''
 
sso['SSO Simulated'] = daily_pcnt['SSO']
upro['UPRO Simulated'] = daily_pcnt['UPRO']

sso['T.E.'] = sso['SSO Simulated'] - sso['SSO']
upro['T.E.'] = upro['UPRO Simulated'] - upro['UPRO']

sso_price = starting_value * (1 + sso).cumprod()
upro_price = starting_value * (1 + upro).cumprod()
    
#%% Plot Simulated and Actual LETFs
sso_price[['SSO', 'SSO Simulated']].plot(logy=True)
upro_price[['UPRO', 'UPRO Simulated']].plot(logy=True)

#%% Plot tracking errors
sso['T.E.'].plot()
upro['T.E.'].plot()

# %%

'''Prepare the dataframe for the simulation, ensuring the dates are accurate 
for all sources.'''

combined_df_3 = pd.concat([spy, ffr], axis=1).dropna()

# daily_pcnt['60/40_SSO_ZROZ'] = 0.6 * daily_pcnt['SSO'] + 0.4 * daily_pcnt['ZROZ']

# daily_pcnt['50/25/25_SSO_ZROZ_GLD'] = 0.5 * daily_pcnt['SSO'] + 0.25 * daily_pcnt['ZROZ'] + 0.25 * daily_pcnt['GLD']

portfolios = starting_value * (1 + daily_pcnt).cumprod()
pf_tracker = portfolios.copy()
pf_tracker['SPY_MA'] = pf_tracker['SPY'] * sp500_price['SPY_MA'] / sp500_price['SPY']

portfolios = portfolios.drop(['SPY', 'FFR', 'GLD', 'ZROZ'], axis=1)
portfolios.plot(logy=True, grid=True)

# %%

def calculate_metrics(period_data, df=False, reindex=False, start_date=None, end_date=None,
                      pcnt_df=daily_pcnt, ffr_col='FFR'):
    if reindex:
        period_data = period_data.dropna(axis=0)
        
    if not start_date:
        start_date = period_data.index[0]
        
    if not end_date:
        end_date = period_data.index[-1]
    
    # Get price data for the specified period
    period_data = period_data.loc[start_date:end_date]
    ffr_period = pcnt_df[ffr_col].loc[start_date:end_date]
    
    # Calculate daily returns
    returns = period_data.pct_change().dropna()
    
    portfolios = returns.columns
    # Calculate metrics for each portfolio
    if df:
        metrics = pd.DataFrame(index=portfolios, columns=['CAGR (%)', 'Volatility (%)', 
                                               'Sharpe', 'Sortino', 'Maximum Drawdown (%)'])
    else: 
        metrics = {}
    for portfolio in portfolios:
        if portfolio == 'FFR':
            continue

        # CAGR (%)
        n_years = (period_data.index[-1] - period_data.index[0]).days / 365.25
        cagr = 100 * ((period_data[portfolio].iloc[-1] / period_data[portfolio].iloc[0]) ** (1/n_years) - 1)
        
        # Volatility (%)
        annual_vol = 100 * returns[portfolio].std() * np.sqrt(trading_days)
        
        # Sharpe Ratio
        aligned_ffr = ffr_period.reindex(returns.index)
        excess_returns = returns[portfolio] - aligned_ffr
        sharpe = np.sqrt(trading_days) * excess_returns.mean() / returns[portfolio].std()
        
        # Sortino Ratio
        downside_returns = returns[portfolio][returns[portfolio] < 0]
        downside_std = downside_returns.std() * np.sqrt(trading_days)
        sortino = np.sqrt(trading_days) * excess_returns.mean() / downside_std
        
        # Maximum Drawdown (%)
        cumulative_returns = (1 + returns[portfolio]).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns/rolling_max - 1
        max_drawdown = 100 * drawdowns.min()
        
        if df:
            metrics.loc[portfolio, 'CAGR (%)'] = cagr
            metrics.loc[portfolio, 'Volatility (%)'] = annual_vol
            metrics.loc[portfolio, 'Sharpe'] = sharpe
            metrics.loc[portfolio, 'Sortino'] = sortino
            metrics.loc[portfolio, 'Maximum Drawdown (%)'] = max_drawdown
        else:    
            # Store metrics
            metrics[f'{portfolio} CAGR (%)'] = cagr
            metrics[f'{portfolio} Volatility (%)'] = annual_vol
            metrics[f'{portfolio} Sharpe'] = sharpe
            metrics[f'{portfolio} Sortino'] = sortino
            metrics[f'{portfolio} Maximum Drawdown (%)'] = max_drawdown
    
    if df:
        return pd.DataFrame(metrics)
    else: 
        return metrics

def analyze_rolling_periods(portfolio, period_length, consider_tax=False, daily_pcnt_df=None, reindex=False):
    '''
    Analyze portfolio metrics over rolling periods with option for tax calculations.
    
    Args:
        portfolio: DataFrame containing portfolio values
        period_length: Length of rolling period in months
        consider_tax: Boolean indicating whether to calculate after-tax metrics
        daily_pcnt_df: DataFrame of daily returns (required if consider_tax=True)
    '''
    if reindex:
        portfolio = portfolio.dropna(axis=0)
    if consider_tax and daily_pcnt_df is None:
        raise ValueError('daily_pcnt_df must be provided when consider_tax is True')
    
    portfolio.index = pd.to_datetime(portfolio.index)

    dates = pd.DataFrame(index=portfolio.index)
    dates['year_month'] = dates.index.to_period('M')
    unique_months = dates['year_month'].unique()
    start_months = unique_months[1:-period_length]
    
    results = []
    for start_month in start_months:
        # Get period dates
        start_date = dates[dates['year_month'] == start_month].index[0]
        
        end_month = start_month + period_length - 1
        end_dates = dates[dates['year_month'] == end_month]
        
        if len(end_dates) == 0:
            continue
        end_date = end_dates.index[-1]
        
        # Calculate metrics for untaxed portfolios
        period_metrics = calculate_metrics(portfolio, start_date=start_date, end_date=end_date)
        
        if consider_tax:
            # Get period data including BF_Signal for tax calculations
            period_data = pf_tracker.loc[start_date:end_date].copy()  # Use pf_tracker instead of portfolio
            period_returns = daily_pcnt_df.loc[start_date:end_date]
            
            # Apply trade taxes and final tax for the period
            period_data_taxed, cost_basis = apply_trade_taxes(period_data, period_returns, starting_value)
            period_data_final = apply_final_tax(period_data_taxed, cost_basis)
            
            # Calculate metrics for taxed portfolios
            taxed_metrics = calculate_metrics(period_data_final, start_date=start_date, end_date=end_date)
            
            # Rename taxed metrics columns to include _net suffix
            net_metrics = {}
            for col in taxed_metrics:
                if '_net' in col:  # Only keep _net columns from taxed metrics
                    net_metrics[col] = taxed_metrics[col]
            
            # Combine untaxed and taxed metrics
            period_metrics.update(net_metrics)
        
        period_metrics['Start Date'] = start_date
        period_metrics['End Date'] = end_date
        results.append(period_metrics)
    
    results_df = pd.DataFrame(results)
    cols = ['Start Date', 'End Date'] + [col for col in results_df.columns if col not in ['Start Date', 'End Date']]
    results_df = results_df[cols]
    
    # Calculate win_rate statistics
    win_rate_stats = {}
    metric_types = ['CAGR (%)', 'Sharpe', 'Sortino', 'Volatility (%)', 'Maximum Drawdown (%)']
    
    for metric in metric_types:
        benchmark_col = f'VOO {metric}'
        if consider_tax:
            benchmark_col = f'VOO_net {metric}'
            
        # Get all portfolio columns for this metric (excluding SPY/SPY_net)
        portfolio_cols = [col for col in results_df.columns if metric in col 
                        and col != benchmark_col]
        
        for portfolio_col in portfolio_cols:
            df = results_df[[portfolio_col, benchmark_col]].dropna()
            win_rate = (df[portfolio_col] > df[benchmark_col]).mean() * 100
            if metric == 'Volatility (%)':
                win_rate = 100 - win_rate
        
            portfolio_name = portfolio_col.split()[0]  # Get portfolio name without metric
            if portfolio_name not in win_rate_stats:
                win_rate_stats[portfolio_name] = {}
            win_rate_stats[portfolio_name][metric] = win_rate
    
    win_rate_df = pd.DataFrame(win_rate_stats).T
    
    return results_df, win_rate_df

def plot_metric_comparison(results_df, metric_name, show_tax=None):
    plt.figure(figsize=(12, 6))
    
    # Get the unique portfolio names (without the metric suffix)
    portfolios = [col.split()[0] for col in results_df.columns if metric_name in col]
    
    # Filter columns based on show_tax parameter
    if show_tax is not None:
        if show_tax:
            # Show only taxed portfolios (_net)
            portfolios = [col for col in portfolios if '_net' in col]
        else:
            # Show only untaxed portfolios (no _net)
            portfolios = [col for col in portfolios if '_net' not in col]
    
    colors = {
        # Base portfolios
        'VOO': '#41ab5d',
        'SSO': '#88419d',
        'UPRO': '#810f7c',
        'GLD': 'gold',
        'HFEA': 'gray',
        'ZROZ': 'green',
        '60/40_SSO_ZROZ': 'crimson',
        '50/25/25_SSO_ZROZ_GLD': 'magenta',
        # LRS portfolios
        'VOO_LRS': '#8c96c6',
        'SSO_LRS': '#88419d',
        'UPRO_LRS': '#810f7c',
        '70/15/15_SSO_LRS_ZROZ_GLD': 'pink',
        '70/15/15_SSO_LRS_ZROZ_GLD2': 'fuchsia',
        # Net portfolios (after tax)
        'VOO_net': '#41ab5d',
        'SSO_net': '#88419d',
        'UPRO_net': '#810f7c',
        'GLD_net': 'gold',
        'HFEA_net': 'gray',
        'ZROZ_net': 'g',
        '60/40_SSO_ZROZ_net': 'crimson',
        '50/25/25_SSO_ZROZ_GLD_net': 'magenta',
        '70/15/15_SSO_LRS_ZROZ_GLD_net': 'pink',
        '70/15/15_SSO_LRS_ZROZ_GLD2_net': 'fuchsia',
        # LRS net portfolios (after tax)
        'VOO_LRS_net': '#8c96c6',
        'SSO_LRS_net': '#88419d',
        'UPRO_LRS_net': '#810f7c'
        
    }  
    
    for portfolio in portfolios:
        df = results_df[['End Date', f'{portfolio} {metric_name}']].dropna()
        plt.plot(df['End Date'], df[f'{portfolio} {metric_name}'], label=portfolio, color=colors[portfolio], linewidth=1.0)
    
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))  # Major ticks every 10 years
    plt.gca().xaxis.set_minor_locator(mdates.YearLocator(1))   # Minor ticks every 2 years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Rotate and align the tick labels so they look better
    plt.setp(plt.gca().get_xticklabels(), rotation=90)
    
    # Add gridlines for both major and minor ticks
    plt.grid(True, which='major', linestyle='-', alpha=1)
    plt.grid(True, which='minor', linestyle=':', alpha=0.7)
    
    plt.title(f'{metric_name} Over {period_length}-Month Rolling Periods')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# %%

metrics = calculate_metrics(portfolios, reindex=True, df=True)
print(metrics)

period_length = 240
rolling_analysis, win_rate_df = analyze_rolling_periods(portfolios, period_length, reindex=True)

print('\nWin Rate vs SPY (% of periods):')
print(win_rate_df)

# Example usage
plot_metric_comparison(rolling_analysis, 'CAGR (%)')
plot_metric_comparison(rolling_analysis, 'Volatility (%)')
plot_metric_comparison(rolling_analysis, 'Sharpe')
plot_metric_comparison(rolling_analysis, 'Sortino')
plot_metric_comparison(rolling_analysis, 'Maximum Drawdown (%)')

# %%
'''LEVERAGE ROTATION STRATEGY'''

# Create a signal column (1 when price > MA on previous day, 0 when price < MA on previous day)

raw_signal = (pf_tracker['SPY'] > pf_tracker['SPY_MA']).astype(int)
pf_tracker['LRS_Signal_raw'] = raw_signal

pf_tracker['LRS_Signal'] = raw_signal.iloc[0]
for i in range(1, len(pf_tracker)):
    pf_tracker['LRS_Signal'].iloc[i] = raw_signal.iloc[i-1]

buffer_days = 0

if buffer_days == 0:
    pf_tracker['BF_Signal'] = pf_tracker['LRS_Signal']
else:  
    # Create buffered signal
    pf_tracker['BF_Signal'] = raw_signal.iloc[0]  # Initialize with first day's signal

    # Loop through starting from the buffer_days point
    current_signal = raw_signal.iloc[0]


    for i in range(buffer_days, len(raw_signal)):
        # Look back buffer_days to check if signal has been consistent
        lookback = pf_tracker['LRS_Signal'].iloc[i - buffer_days: i + 1]
    
        if all(lookback == 1) and current_signal == 0:
            # If we see buffer_days consecutive 1s and we're currently in cash
            current_signal = 1
        elif all(lookback == 0) and current_signal == 1:
            # If we see buffer_days consecutive 0s and we're currently in leverage
            current_signal = 0
        
        pf_tracker.iloc[i, pf_tracker.columns.get_loc('BF_Signal')] = current_signal

trade_count = abs(pf_tracker['LRS_Signal'].diff()).sum()
bf_trade_count = abs(pf_tracker['BF_Signal'].diff()).sum()
print(f'Number of trades on raw signal: {trade_count}')
print(f'Number of trades on buffered signal: {bf_trade_count}')

# Add LRS return columns to daily_pcnt DataFrame
daily_pcnt['VOO_LRS'] = daily_pcnt['VOO'] * pf_tracker['BF_Signal'] + \
                        daily_pcnt['FFR'] * (1 - pf_tracker['BF_Signal'])
                        
daily_pcnt['SSO_LRS'] = daily_pcnt['SSO'] * pf_tracker['BF_Signal'] + \
                        daily_pcnt['FFR'] * (1 - pf_tracker['BF_Signal'])
                        
daily_pcnt['UPRO_LRS'] = daily_pcnt['UPRO'] * pf_tracker['BF_Signal'] + \
                         daily_pcnt['FFR'] * (1 - pf_tracker['BF_Signal'])
                         
daily_pcnt['70/15/15_SSO_LRS_ZROZ_GLD'] = 0.7 * daily_pcnt['SSO_LRS'] + 0.15 * daily_pcnt['ZROZ'] + 0.15 * daily_pcnt['GLD']

daily_pcnt['SSO_LRS2'] = daily_pcnt['SSO'] * pf_tracker['BF_Signal'] + \
                        0.5 * daily_pcnt['ZROZ'] * (1 - pf_tracker['BF_Signal']) + \
                        0.5 * daily_pcnt['GLD'] * (1 - pf_tracker['BF_Signal'])

daily_pcnt['70/15/15_SSO_LRS_ZROZ_GLD2'] = 0.7 * daily_pcnt['SSO_LRS2'] + 0.15 * daily_pcnt['ZROZ'] + 0.15 * daily_pcnt['GLD']

lrs_cols = ['VOO_LRS', 'SSO_LRS', 'UPRO_LRS', '70/15/15_SSO_LRS_ZROZ_GLD']
# Calculate LRS pf_tracker
pf_tracker[lrs_cols] = starting_value * (1 + daily_pcnt[lrs_cols]).cumprod()

lrs_portfolios = pf_tracker[['VOO', 'HFEA', '60/40_SSO_ZROZ', '50/25/25_SSO_ZROZ_GLD'] + lrs_cols]
# lrs_portfolios.plot(logy=True)

lrs_portfolios = starting_value * (1 + daily_pcnt).cumprod()

lrs_portfolios = lrs_portfolios.drop(['SPY', 'FFR', 'GLD', 'ZROZ', 'SSO', 'UPRO', '60/40_SSO_ZROZ', '50/25/25_SSO_ZROZ_GLD', 'SSO_LRS2'], axis=1)
lrs_portfolios.plot(logy=True)

# %%
lrs_metrics = calculate_metrics(lrs_portfolios, reindex=True, df=True)
print(lrs_metrics)

period_length = 240
lrs_rolling_analysis, lrs_win_rate_df = analyze_rolling_periods(lrs_portfolios, period_length, reindex=True)

print('\nWin Rate vs SPY (% of periods):')
print(lrs_win_rate_df)

plot_metric_comparison(lrs_rolling_analysis, 'CAGR (%)')
plot_metric_comparison(lrs_rolling_analysis, 'Volatility (%)')
plot_metric_comparison(lrs_rolling_analysis, 'Sharpe')
plot_metric_comparison(lrs_rolling_analysis, 'Sortino')
plot_metric_comparison(lrs_rolling_analysis, 'Maximum Drawdown (%)')

win_rates = pd.concat([win_rate_df, lrs_win_rate_df], axis =0)
win_rates = win_rates.drop_duplicates()

tot_metrics = pd.concat([metrics, lrs_metrics], axis =0)
tot_metrics = tot_metrics.drop_duplicates()


# %%

def analyze_multi_period_win_rate(portfolio, max_months=360, step=12, consider_tax=False, daily_pcnt_df=None):
    '''
    Analyze Win rate across multiple rolling periods
    '''
    periods = range(step, max_months + step, step)
    win_rate_results = {}
    
    for period_length in periods:
        _, win_rate_df = analyze_rolling_periods(portfolio, period_length, 
                                                     consider_tax, daily_pcnt_df)
        win_rate_results[period_length] = win_rate_df
    
    # Restructure data for plotting
    metrics = ['CAGR (%)', 'Sharpe', 'Sortino']
    plot_data = {metric: pd.DataFrame() for metric in metrics}
    
    for period_length, df in win_rate_results.items():
        years = period_length / 12
        for metric in metrics:
            plot_data[metric][years] = df[metric]
    
    return plot_data

def plot_win_rate(plot_data):
    '''
    Plot Win Rate across different time periods for each metric
    '''
    colors = {
        'VOO_LRS': '#8c96c6',
        'SSO_LRS': '#88419d',
        'UPRO_LRS': '#810f7c',
        'VOO_LRS_net': '#8c96c6',
        'SSO_LRS_net': '#88419d',
        'UPRO_LRS_net': '#810f7c'
    }
    
    for metric, data in plot_data.items():
        plt.figure(figsize=(8, 6))
        
        for portfolio in data.index:
            plt.plot(data.columns, data.loc[portfolio], 
                    label=portfolio, color=colors[portfolio], linewidth=2.0)
        
        plt.title(f'{metric} Win Rate vs SPY Over Different Rolling Periods')
        plt.xlabel('Rolling Period Length (Years)')
        plt.ylabel('% of Rolling Periods')
        
        plt.grid(True)        
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Usage:
plot_data = analyze_multi_period_win_rate(lrs_portfolios, max_months=360, step=12)
plot_win_rate(plot_data)

# %%

'''TAX ON CAPITAL GAINS'''

def apply_trade_taxes(df, daily_pcnt_df, starting_value):
    '''Apply capital gains tax at each trade for LRS strategies'''
    # Get trade dates from signal changes
    pf_tracker_df = df.copy()
    signal_changes = pf_tracker_df['BF_Signal'].diff().fillna(0)
    trade_dates = signal_changes[signal_changes != 0].index
    
    # Initialize net columns for LRS portfolios
    for col in ['VOO_LRS', 'SSO_LRS', 'UPRO_LRS']:
        net_col = f'{col}_net'
        pf_tracker_df[net_col] = 0
        pf_tracker_df.loc[pf_tracker_df.index[0], net_col] = starting_value * (1 + daily_pcnt_df[col].iloc[0])
    
    # Initialize cost basis
    cost_basis = {col+'_net': pf_tracker_df[col+'_net'].iloc[0] 
                 for col in ['VOO_LRS', 'SSO_LRS', 'UPRO_LRS']}
    
    # Calculate net values with taxes for LRS portfolios
    for i in range(1, len(daily_pcnt_df)):
        current_date = daily_pcnt_df.index[i]
        
        for col in ['VOO_LRS', 'SSO_LRS', 'UPRO_LRS']:
            net_col = f'{col}_net'
            current_value = pf_tracker_df[net_col].iloc[i-1] * (1 + daily_pcnt_df[col].iloc[i])
            
            if current_date in trade_dates:
                capital_gain = max(0, float(current_value - cost_basis[net_col]))
                tax = capital_gain * 0.3 * 0.5
                pf_tracker_df.loc[current_date, net_col] = current_value - tax
                cost_basis[net_col] = pf_tracker_df.loc[current_date, net_col]
            else:
                pf_tracker_df.loc[current_date, net_col] = current_value
    
    return pf_tracker_df, cost_basis

def apply_final_tax(pf_tracker_df, cost_basis):
    '''Apply capital gains tax on the last day for net portfolio values'''
    df_final = pf_tracker_df.copy()
    
    # Create SPY_net column and initialize it with SPY values
    df_final['SPY_net'] = df_final['SPY']
    cost_basis['SPY_net'] = starting_value  # SPY's cost basis is always the initial value
    
    # Apply final tax to all net columns
    for col in ['SPY_net', 'VOO_LRS_net', 'SSO_LRS_net', 'UPRO_LRS_net']:
        final_value = df_final[col].iloc[-1]
        capital_gain = max(0, float(final_value - cost_basis[col]))
        tax = capital_gain * 0.3 * 0.5
        df_final.loc[df_final.index[-1], col] -= tax
    
    return df_final

# Usage:
# First apply trade taxes
pf_tracker2, cost_basis = apply_trade_taxes(pf_tracker, daily_pcnt, starting_value)

# Then apply final tax
pf_tracker_net = apply_final_tax(pf_tracker2, cost_basis)


# Plot comparison including net values for LRS strategies
net_cols = ['SPY_net', 'VOO_LRS_net', 'SSO_LRS_net', 'UPRO_LRS_net']
lrs_taxed_pf = lrs_portfolios.copy()
lrs_taxed_pf[net_cols] = pf_tracker_net[net_cols]

lrs_taxed_pf.plot(logy=True)
plt.title('Portfolio Values (Before and After Trade Taxes)')
plt.grid(True)
plt.show()

# %%
period_length = 240

taxed_lrs_rolling_analysis = analyze_rolling_periods(lrs_portfolios, period_length, 
                                               consider_tax=True, 
                                               daily_pcnt_df=daily_pcnt)

# rolling_analysis_taxed360.to_xlsx('D:/Schoolwork - McGill/Other/LETF Analysis/360m_spy_letf_rolling_analysis_taxed.xlsx', index=False)

plot_metric_comparison(taxed_lrs_rolling_analysis, 'CAGR (%)', show_tax=False)
plot_metric_comparison(taxed_lrs_rolling_analysis, 'CAGR (%)', show_tax=True)

# For taxed analysis:(way too long)
# plot_data_taxed = analyze_multi_period_win_rate(lrs_portfolios, max_months=360, step=12, consider_tax=True, daily_pcnt_df=daily_pcnt)
# plot_win_rate(plot_data_taxed)

# %%

'''SIMULATION SECTION'''

 
rows = []
percentiles = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
 
for i in range(number_of_simulations):
 
    print(f'#{i + 1}')
 
    '''Below we simulate the S&P500. This is the most critical step because
    the simulated SSO and UPRO is simply 2x/3x this simulation, plus the 
    borrowing costs and expense ratio, and some randomized tracking 
    error. We use the block bootstrap method to simulate
    this instead of the bootstrap method, so we capture any auto-correlation.
 
    simulated_spy: a simulated path of the S&P500's daily returns for a
    specified amount of days, stored in a pandas dataframe as decimals. The 
    federal funds rate (FRR) is also stored in this dataframe.'''
 
    block_size = 7  # days
    if block_size > len(combined_df_3):
        sys.exit('ERROR: Block size is greater than the sample size')
    n_blocks = len(combined_df_3) - block_size + 1
    blocks = []
    for i in range(n_blocks):
        block = combined_df_3.iloc[i:i + block_size]
        blocks.append(block)
 
    required_n_blocks = math.ceil(days_simulated / block_size)
    simulated_blocks = []
    for n in range(required_n_blocks):
        random_i = random.randint(0, len(blocks) - 1)
        random_block = blocks[random_i]
        simulated_blocks.append(random_block)
 
    simulated_spy = pd.concat(simulated_blocks).reset_index(drop=True)
    days_to_subtract = required_n_blocks * block_size - days_simulated
    simulated_spy = simulated_spy.iloc[
                      :len(simulated_spy) - days_to_subtract]
 
    '''Below we simulate the SSO and UPRO beyond their inception dates.
    To do this, we take the simulated S&P500 above, and multiply each daily
    return by 2x for the SSO and 3x for the UPRO. We subtract the expected 
    borrowing costs using the interest rates at that time, subtract the 
    expense ratios, and then, for each daily return, we RANDOMLY add some 
    tracking error to keep it realistic.
 
    simulated_sso: a simulated path of the SSO's daily returns for a
    specified amount of days, stored in a pandas dataframe as decimals.
    simulated_upro: same as simulated_sso but for the UPRO ETF.'''
 
    simulated_sso = pd.Series(np.random.choice(sso['T.E.'].values, size=days_simulated) +
                              simulated_spy['SPY'] * 2
                              - ((etf_params['SSO']['lr'] - 1) * (simulated_spy['FFR'] + borrowing_spread / 360))
                              - (etf_params['SSO']['er'] / 365)).reset_index(drop=True)
    
    simulated_upro = pd.Series(np.random.choice(upro['T.E.'].values, size=days_simulated) +
                               simulated_spy['SPY'] * 3
                               - ((etf_params['UPRO']['lr'] - 1) * (simulated_spy['FFR'] + borrowing_spread / 360))
                               - (etf_params['UPRO']['er'] / 365)).reset_index(drop=True)

    simulated_spy = pd.Series(simulated_spy['SPY'])
 
    '''Below we simulate the portfolio value growth over time using the
    simulated ETF path performed above, using the initial deposit and the
    monthly contributions which are added approximately every 21 trading 
    days.'''
 
    simulated_spy_pf_tracker_with_additions = np.zeros(len(simulated_spy) + 1)
    simulated_spy_pf_tracker_with_additions[0] = start_price
    simulated_sso_pf_tracker_with_additions = np.zeros(len(simulated_sso) + 1)
    simulated_sso_pf_tracker_with_additions[0] = start_price
    simulated_upro_pf_tracker_with_additions = np.zeros(len(simulated_upro) + 1)
    simulated_upro_pf_tracker_with_additions[0] = start_price
 
    for t in range(1, len(simulated_spy) + 1):
        simulated_spy_pf_tracker_with_additions[t] = simulated_spy_pf_tracker_with_additions[t - 1] * (1 + simulated_spy[t - 1])
        simulated_sso_pf_tracker_with_additions[t] = simulated_sso_pf_tracker_with_additions[t - 1] * (1 + simulated_sso[t - 1])
        simulated_upro_pf_tracker_with_additions[t] = simulated_upro_pf_tracker_with_additions[t - 1] * (1 + simulated_upro[t - 1])
        if t % 21 == 0:
            simulated_spy_pf_tracker_with_additions[t] += monthly_addition
            simulated_sso_pf_tracker_with_additions[t] += monthly_addition
            simulated_upro_pf_tracker_with_additions[t] += monthly_addition
 
    rows.append({
        'spy_Final': simulated_spy_pf_tracker_with_additions[-1],
        'SSO_Final': simulated_sso_pf_tracker_with_additions[-1],
        'UPRO_Final': simulated_upro_pf_tracker_with_additions[-1]
    })
 
#%%
 
'''Below is where all of the results are summarized and visualized.'''
 
potential_final_pf_tracker = pd.DataFrame(rows)
percentile_results = potential_final_pf_tracker.quantile(percentiles)

def format_currency(value):
    return f'${value:,.0f}'
 
 
formatted_percentile_results = percentile_results.apply(lambda x: x.map(format_currency))
print('\n\nPercentiles of Final Outcomes:')
print(formatted_percentile_results)
 
print('\n')
print(f'{days_simulated / trading_days} years in the market')
print(f'${start_price + monthly_addition * (days_simulated / 21)} total invested')
print(f'${start_price} initial investment, ${monthly_addition} monthly additions')
print(f'{spy.index[0].strftime('%Y-%m-%d')} historical data start date')
print(f'{spy.index[-1].strftime('%Y-%m-%d')} historical data end date')
print(f'{number_of_simulations} simulations')