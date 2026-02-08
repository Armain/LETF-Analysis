import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import warnings
warnings.filterwarnings('ignore')

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    tickers = tables[0]['Symbol'].tolist()
    return [ticker.replace('.', '-') for ticker in tickers if isinstance(ticker, str)]

def get_stock_data(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            'period1': 0,
            'period2': int(pd.Timestamp('2024-12-31').timestamp()),
            'interval': '1d'
        }
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if 'chart' not in data or not data['chart']['result']:
            return None
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        prices = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Close': prices['close']
        }).dropna().set_index('Date')
        
        if len(df) < 100:
            return None
        
        daily_returns = df['Close'].pct_change().dropna()
        years = (df.index[-1] - df.index[0]).days / 365.25
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        annual_return = ((1 + total_return) ** (1/years) - 1) * 100
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        try:
            info_url = f"https://query1.finance.yahoo.com/v1/finance/quoteType/{ticker}"
            info_response = requests.get(info_url, headers=headers)
            company = info_response.json()['quoteType']['result'][0].get('longName', ticker)
        except:
            company = ticker
        
        return {
            'Ticker': ticker,
            'Company': company,
            'Annual_Return_%': annual_return,
            'Volatility_%': volatility
        }
    except:
        return None

def process_stocks(tickers):
    results = []
    for i, ticker in enumerate(tickers):
        print(f"Processing {ticker} ({i+1}/{len(tickers)})")
        data = get_stock_data(ticker)
        if data:
            results.append(data)
        time.sleep(2)
    return pd.DataFrame(results)

def create_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Volatility_%'],
        y=df['Annual_Return_%'],
        mode='markers',
        marker=dict(size=8, opacity=0.7),
        text=df['Ticker'],
        customdata=df['Company'],
        hovertemplate='<b>%{text}</b><br>%{customdata}<br>Return: %{y:.1f}%<br>Volatility: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='S&P 500 Risk-Return Analysis',
        xaxis_title='Volatility (%)',
        yaxis_title='Annual Return (%)',
        width=1200,
        height=800
    )
    return fig

tickers = get_sp500_tickers()
print(f"Found {len(tickers)} tickers")

df = process_stocks(tickers)
if not df.empty:
    fig = create_plot(df)
    df.to_csv('sp500_data.csv', index=False)
    fig.write_html("sp500_plot.html")
    fig.show()