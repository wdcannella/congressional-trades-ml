"""
Stock Price Downloader

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
    """Get unique tickers, remove empty ones."""
    tickers = df[df['ticker'].notna() & (df['ticker'] != '')]['ticker'].unique()
    print(f"\nFound {len(tickers)} unique tickers")
    print(f"Sample tickers: {list(tickers[:10])}")

    return tickers


#tickers stored under the old name so calculate_returns.py can find them without changes
TICKER_ALIASES = {
    'FLT':  'CPAY',   # FleetCor renamed to Corpay (2024)
    'PEAK': 'DOC',    # Healthpeak renamed (2024)
    'NCR':  'VYX',    # NCR split — Voyix kept the legacy business
    'FB':   'META',   # Facebook renamed to Meta (2021)
    'TWTR': 'X',      # Twitter renamed to X (2023)
}

NON_STOCK_TICKERS = {'XSP', 'SPX', 'VIX', 'NDX', 'RUT', 'DJX'}



def _fetch_with_fallback(ticker_clean, start_date):
    """
    Try downloading ticker history, fall back to period='max' for delisted
    stocks
    """
    stock = yf.Ticker(ticker_clean)
    hist = stock.history(start=start_date, end=datetime.now())
    if len(hist) > 0:
        return hist, None
    hist = stock.history(period='max')

    if len(hist) > 0:
        return hist, None

    return None, "No data returned (delisted or invalid ticker)"


def download_stock_data(tickers, start_date='2020-01-01'):
    """
    Download historical stock prices for all tickers with retries.

    """
    print(f"\nDownloading stock data from {start_date} to present...")

    tickers_to_fetch =  []
    skipped = []
    for t in tickers:
        t_clean = str(t).strip().upper()
        if t_clean in  NON_STOCK_TICKERS:
            skipped.append(t_clean)
        else:
            tickers_to_fetch.append(t_clean)

    if skipped:
        print(f"Skipping {len(skipped)} non-stock tickers: {skipped}")

    stock_data ={}
    failed_tickers = {}

    for ticker in tqdm(tickers_to_fetch, desc="Downloading"):
        if not ticker:
            continue

        fetch_ticker = TICKER_ALIASES.get(ticker, ticker)
        success = False

        for attempt in range(3):
            try:
                hist, err = _fetch_with_fallback(fetch_ticker, start_date)
                if hist is not None:
                    stock_data[ticker] = hist
                    if fetch_ticker  != ticker:
                        print(f"  {ticker} → fetched as {fetch_ticker} ({len(hist)} rows)")
                    success = True

                    break
                else:
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
                    else:
                        failed_tickers[ticker] = err
            except Exception as e:
                if attempt< 2:
                    time.sleep(2*(attempt + 1))
                else:
                    failed_tickers[ticker] = str(e)

        if not success and ticker not in failed_tickers:
            failed_tickers[ticker] = "Unknown error"

        time.sleep(0.2)

    print(f"\nSuccessfully downloaded: {len(stock_data)}/{len(tickers_to_fetch)}")
    print(f"Skipped (non-stock):     {len(skipped)}")
    print(f"Failed:                  {len(failed_tickers)}")

    if failed_tickers:
        print("\nFailed tickers (likely delisted with no recoverable history):")
        for t, err in list(failed_tickers.items())[:30]:
            print(f"  {t}: {err}")

    return stock_data, list(failed_tickers.keys())


def download_sp500_data(start_date='2020-01-01'):
    print("\nDownloading S&P 500 benchmark data...")
    try:
        hist = yf.Ticker('SPY').history(start=start_date, end=datetime.now())
        print(f"Downloaded {len(hist)} days of S&P 500 data")
        return hist
    except Exception as e:
        print(f"Failed to download S&P 500: {str(e)}")
        return None


SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Healthcare': 'XLV',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
}


def download_sector_etf_prices(start_date='2020-01-01'):
    print(f"\nDownloading sector ETF prices ({len(SECTOR_ETF_MAP)} ETFs)...")
    etf_prices = {}
    for sector, etf in SECTOR_ETF_MAP.items():
        try:
            hist = yf.Ticker(etf).history(start=start_date)
            if len(hist) > 0:
                etf_prices[sector] = hist
                print(f"  {etf} ({sector}): {len(hist)} days")
            else:
                print(f"  {etf} ({sector}): no data")
        except Exception as e:
            print(f"  {etf} ({sector}): failed — {e}")

        time.sleep(0.2)

    with open('data/sector_etf_prices.pkl', 'wb') as f:
        pickle.dump(etf_prices,  f)
    print(f"Saved sector_etf_prices.pkl ({len(etf_prices)} sectors)")
    return etf_prices


def download_sector_data(tickers):
    """
    Fetch sector classification for each ticker from yfinance
    """
    SECTOR_NORMALIZE={
        'Technology': 'Technology',
        'Financial Services': 'Financials',
        'Healthcare': 'Healthcare',
        'Consumer Cyclical': 'Consumer Discretionary',
        'Consumer Defensive': 'Consumer Staples',
        'Industrials': 'Industrials',
        'Energy': 'Energy',
        'Utilities': 'Utilities',
        'Basic Materials': 'Materials',
        'Real Estate': 'Real Estate',
        'Communication Services': 'Communication Services',
    }

    print(f"\nFetching sector data for {len(tickers)} tickers...")
    sector_map ={}

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Fetching sectors {i+1}/{len(tickers)}...")
        try:
            info = yf.Ticker(ticker).info
            raw_sector = info.get('sector', None)
            sector_map[ticker] = SECTOR_NORMALIZE.get(raw_sector, raw_sector or 'Unknown')

        except Exception:
            sector_map[ticker] = 'Unknown'

        time.sleep(0.1)

    known = sum(1 for v in sector_map.values() if v != 'Unknown')
    print(f"Sector data: {known}/{len(tickers)} tickers classified")
    return sector_map


def save_stock_data(stock_data, sp500_data, sector_map=None):
    print("\nSaving data...")

    with open('data/stock_prices.pkl', 'wb') as f:
        pickle.dump(stock_data, f)
    print("Saved stock_prices.pkl")

    if sp500_data is not None:
        sp500_data.to_csv('data/sp500_prices.csv')
        print("Saved sp500_prices.csv")

    if sector_map is not None:
        sector_df= pd.DataFrame(list(sector_map.items()), columns=['ticker', 'sector'])
        sector_df.to_csv('data/ticker_sectors.csv', index=False)
        print("Saved ticker_sectors.csv")

    summary_data = []
    for ticker, df in stock_data.items():
        summary_data.append({
            'ticker': ticker,
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'num_days': len(df),
            'latest_price': df['Close'].iloc[-1] if len(df) > 0 else None
        })

    pd.DataFrame(summary_data).to_csv('data/stock_data_summary.csv', index=False)
    print("Saved stock_data_summary.csv")


def main():
    print("=" * 60)
    print("STOCK PRICE DOWNLOADER")
    print("="  * 60)

    trades_df = load_trades()
    tickers=get_unique_tickers(trades_df)

    min_trade_date = pd.to_datetime(trades_df['traded_date']).min()
    start_date = (min_trade_date - timedelta(days=365)).strftime('%Y-%m-%d')
    print(f"\nDate range: {start_date} to present")

    stock_data, failed = download_stock_data(tickers, start_date)
    sp500_data = download_sp500_data(start_date)
    sector_map = download_sector_data(list(stock_data.keys()))
    download_sector_etf_prices(start_date)
    save_stock_data(stock_data, sp500_data, sector_map)


    print("\n\nDownload Complete\n")


if __name__ == "__main__":
    main()