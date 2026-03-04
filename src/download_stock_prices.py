import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Load your trades
df = pd.read_csv('data/capitoltrades_data.csv')

# Get unique tickers (remove empty ones)
tickers = df[df['ticker'] != '']['ticker'].unique()
print(f"Need to download prices for {len(tickers)} tickers")

# Download stock prices
stock_data = {}
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        # Get 2 years of history to cover all trades
        hist = stock.history(period="2y")
        stock_data[ticker] = hist
        print(f"Downloaded {ticker}")
    except:
        print(f"Failed: {ticker}")

# Save to pickle for easy loading later
import pickle
with open('data/stock_prices.pkl', 'wb') as f:
    pickle.dump(stock_data, f)