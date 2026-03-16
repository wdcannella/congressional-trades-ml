"""
Stock Price Downloader
======================

Downloads historical stock prices for all tickers in the trades dataset.

Usage:
    python src/download_stock_prices.py
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import time
from tqdm import tqdm


def load_trades():
    """Load processed trades data."""
    print("Loading trades data...")
    df = pd.read_csv('data/trades_with_features.csv')
    print(f"Loaded {len(df)} trades")
    return df


def get_unique_tickers(df):
    """Get unique tickers, removing empty/invalid ones."""
    # Remove empty tickers
    tickers = df[df['ticker'].notna() & (df['ticker'] != '')]['ticker'].unique()

    print(f"\nFound {len(tickers)} unique tickers")
    print(f"Sample tickers: {list(tickers[:10])}")

    return tickers


def download_stock_data(tickers, start_date='2020-01-01'):
    """
    Download historical stock prices for all tickers with retries.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for historical data

    Returns:
        dict: Dictionary mapping ticker -> price DataFrame
    """
    print(f"\nDownloading stock data from {start_date} to present...")

    stock_data = {}
    failed_tickers = {}

    for ticker in tqdm(tickers, desc="Downloading"):
        # Clean ticker
        ticker_clean = str(ticker).strip().upper()
        if not ticker_clean:
            continue

        success = False
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # Download data
                stock = yf.Ticker(ticker_clean)
                # Using repair=True if available in this yfinance version
                hist = stock.history(start=start_date, end=datetime.now())

                if len(hist) > 0:
                    stock_data[ticker_clean] = hist
                    success = True
                    break
                else:
                    # Some tickers return empty DF without error
                    error_msg = "No data returned (possibly delisted)"
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        failed_tickers[ticker_clean] = error_msg

            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    failed_tickers[ticker_clean] = error_msg

        if not success and ticker_clean not in failed_tickers:
             failed_tickers[ticker_clean] = "Unknown error"

        time.sleep(0.2)  # Base delay between tickers

    print(f"\nSuccessfully downloaded: {len(stock_data)}/{len(tickers)}")
    print(f"Failed: {len(failed_tickers)}")

    if len(failed_tickers) > 0:
        print("\nTop 20 Failed Tickers:")
        for t, err in list(failed_tickers.items())[:20]:
            print(f"  {t}: {err}")

    return stock_data, list(failed_tickers.keys())


def download_sp500_data(start_date='2020-01-01'):
    """Download S&P 500 data for benchmark comparison."""
    print("\nDownloading S&P 500 benchmark data...")

    try:
        sp500 = yf.Ticker('SPY')  # S&P 500 ETF
        hist = sp500.history(start=start_date, end=datetime.now())
        print(f"Downloaded {len(hist)} days of S&P 500 data")
        return hist
    except Exception as e:
        print(f"Failed to download S&P 500: {str(e)}")
        return None


def save_stock_data(stock_data, sp500_data):
    """Save stock data to files."""
    print("\nSaving data...")

    # Save as pickle for easy loading
    with open('data/stock_prices.pkl', 'wb') as f:
        pickle.dump(stock_data, f)
    print("Saved stock_prices.pkl")

    # Save S&P 500
    if sp500_data is not None:
        sp500_data.to_csv('data/sp500_prices.csv')
        print("Saved sp500_prices.csv")

    # Also save summary CSV for quick reference
    summary_data = []
    for ticker, df in stock_data.items():
        summary_data.append({
            'ticker': ticker,
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'num_days': len(df),
            'latest_price': df['Close'].iloc[-1] if len(df) > 0 else None
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('data/stock_data_summary.csv', index=False)
    print("Saved stock_data_summary.csv")


def main():
    """Main function."""
    print("=" * 60)
    print("STOCK PRICE DOWNLOADER")
    print("=" * 60)

    # Load trades
    trades_df = load_trades()

    # Get tickers
    tickers = get_unique_tickers(trades_df)

    # Determine date range
    min_trade_date = pd.to_datetime(trades_df['traded_date']).min()
    # Start downloading from 1 year before earliest trade
    start_date = (min_trade_date - timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"\nDate range: {start_date} to present")

    # Download stock data
    stock_data, failed = download_stock_data(tickers, start_date)

    # Download S&P 500
    sp500_data = download_sp500_data(start_date)

    # Save
    save_stock_data(stock_data, sp500_data)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"Stock data: data/stock_prices.pkl")
    print(f"S&P 500 data: data/sp500_prices.csv")
    print(f"Summary: data/stock_data_summary.csv")
    print("\nNext step: Calculate returns for each trade")


if __name__ == "__main__":
    main()