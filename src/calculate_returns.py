"""
Returns Calculator

Calculates stock returns at various time horizons after congressional trades.
Creates target variable for ML: did the trade outperform the market?

Usage:
    python src/calculate_returns.py
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta


def load_data():
    print("Loading data... ")

    trades = pd.read_csv('data/trades_with_features.csv')
    trades['traded_date'] = pd.to_datetime(trades['traded_date'])
    print(f"Loaded {len(trades)} trades")

    with open('data/stock_prices.pkl', 'rb') as f:
        stock_prices = pickle.load(f)
    print(f"Loaded prices for {len(stock_prices)} stocks")

    sp500 =  pd.read_csv('data/sp500_prices.csv', index_col=0, parse_dates=True)
    print(f"Loaded S&P 500 benchmark")

    try:
        with open('data/sector_etf_prices.pkl', 'rb') as f:
            sector_etf_prices =pickle.load(f)
        print(f"Loaded sector ETF prices for {len(sector_etf_prices)} sectors")
    except FileNotFoundError:
        print("WARNING: sector_etf_prices.pkl not found — sector_etf_momentum_30d will be NaN")
        sector_etf_prices= {}

    return trades, stock_prices, sp500, sector_etf_prices


def get_price_on_date(stock_df, target_date, lookback_days=5):
    """
    Get stock price on a specific date, check if market was closed.

    """
    if not hasattr(get_price_on_date, "_index_cache"):
        get_price_on_date._index_cache = {}

    df_id = id(stock_df)
    if df_id not in get_price_on_date._index_cache:
        if not isinstance(stock_df.index, pd.DatetimeIndex):
            dt_index = pd.to_datetime(stock_df.index, utc=True)
        else:
            dt_index  = stock_df.index
        get_price_on_date._index_cache[df_id] = dt_index.strftime('%Y-%m-%d')


    date_strings = get_price_on_date._index_cache[df_id]
    target_str = target_date.strftime('%Y-%m-%d')


    mask = date_strings == target_str
    if mask.any():
        return stock_df.loc[mask, 'Close'].iloc[0]

    for i in range(1, lookback_days + 1):
        check_str = (target_date - timedelta(days=i)).strftime('%Y-%m-%d')
        mask = date_strings == check_str
        if mask.any():
            return stock_df.loc[mask, 'Close'].iloc[0]

    return None


def calculate_return(price_start, price_end):
    """Calculate percentage return."""
    if price_start is None or price_end is None or price_start == 0:
        return None
    return ((price_end - price_start) / price_start) * 100


def calculate_returns_for_trade(trade, stock_prices, sp500, sector_etf_prices, horizons=[30, 60, 90]):
    """
    Calculate returns at multiple time horizons for a single trade.

    """
    ticker = trade['ticker']
    trade_date= trade['traded_date']

    results = {'ticker': ticker, 'traded_date': trade_date}

    if ticker not in stock_prices:
        for horizon in horizons:
            results[f'return_{horizon}d'] = None
            results[f'sp500_return_{horizon}d'] = None
            results[f'excess_return_{horizon}d'] = None

            results[f'outperformed_{horizon}d'] = None
        return results

    stock_df = stock_prices[ticker]

    #get price at trade date
    price_at_trade = get_price_on_date(stock_df, trade_date)

    if price_at_trade is None:
        for horizon in horizons:
            results[f'return_{horizon}d']= None
            results[f'sp500_return_{horizon}d']= None
            results[f'excess_return_{horizon}d']= None
            results[f'outperformed_{horizon}d']= None

        return results

    #get S&P 500 price at trade date
    sp500_price_at_trade = get_price_on_date(sp500, trade_date)

    for lookback in [30, 90]:
        prior_date = trade_date - timedelta(days=lookback)
        spy_prior = get_price_on_date(sp500, prior_date)
        results[f'spy_momentum_{lookback}d'] = calculate_return(spy_prior, sp500_price_at_trade)

    for lookback in [30, 60]:
        prior_date = trade_date - timedelta(days=lookback)
        price_prior= get_price_on_date(stock_df, prior_date)
        results[f'stock_momentum_{lookback}d'] = calculate_return(price_prior, price_at_trade)

    #sector ETF momentum
    sector = trade.get('sector', 'Unknown') if hasattr(trade, 'get') else trade['sector']
    if sector and sector != 'Unknown' and sector in sector_etf_prices:
        etf_df = sector_etf_prices[sector]
        etf_prior = get_price_on_date(etf_df, trade_date - timedelta(days=30))
        etf_now = get_price_on_date(etf_df, trade_date)
        results['sector_etf_momentum_30d'] = calculate_return(etf_prior, etf_now)

    else:
        results['sector_etf_momentum_30d'] = None

    for horizon in horizons:
        future_date = trade_date + timedelta(days=horizon)

        price_future= get_price_on_date(stock_df, future_date)
        stock_return =calculate_return(price_at_trade, price_future)

        results[f'return_{horizon}d'] = stock_return

        sp500_price_future = get_price_on_date(sp500, future_date)
        sp500_return = calculate_return(sp500_price_at_trade, sp500_price_future)
        results[f'sp500_return_{horizon}d'] = sp500_return

        if stock_return is  not None and sp500_return is not None:
            excess_return = stock_return - sp500_return
            results[f'excess_return_{horizon}d'] = excess_return
            results[f'outperformed_{horizon}d'] = 1 if excess_return > 5.0 else 0
        else:

            results[f'excess_return_{horizon}d'] = None
            results[f'outperformed_{horizon}d'] = None

    return results


def calculate_all_returns(trades, stock_prices, sp500, sector_etf_prices):
    """Calculate returns for all trades."""

    print("\nCalculating returns for all trades...")
    returns_list = []

    for idx, trade in trades.iterrows():
        if idx % 1000 == 0:
            print(f"Processing trade {idx + 1}/{len(trades)}")
        returns_list.append(calculate_returns_for_trade(trade, stock_prices, sp500, sector_etf_prices))

    return pd.DataFrame(returns_list)


def merge_returns_with_trades(trades, returns_df):
    """Merge calculated returns back with trade data."""
    print("\nMerging returns with trade data...")
    trades['traded_date'] = pd.to_datetime(trades['traded_date'])
    returns_df['traded_date'] = pd.to_datetime(returns_df['traded_date'])
    merged  = trades.merge(returns_df, on=['ticker', 'traded_date'], how='left')
    print(f"Merged dataset has {len(merged)} rows")
    return merged


def print_summary_statistics(df):
    """Print summary statistics about returns."""
    print("\n" + "="*60)
    print("RETURNS SUMMARY STATISTICS")
    print("="*60)

    for horizon in [30, 60, 90]:
        return_col = f'return_{horizon}d'
        sp500_col= f'sp500_return_{horizon}d'
        excess_col = f'excess_return_{horizon}d'
        outperform_col = f'outperformed_{horizon}d'

        print(f"\n{horizon}-Day Returns:")
        print(f"  Trades with data: {df[return_col].notna().sum()}/{len(df)}")

        if df[return_col].notna().sum() > 0:
            print(f"  Mean stock return: {df[return_col].mean():.2f}%")
            print(f"  Mean S&P 500 return: {df[sp500_col].mean():.2f}%")
            print(f"  Mean excess return: {df[excess_col].mean():.2f}%")
            print(f"  % that outperformed: {df[outperform_col].mean()*100:.1f}%")
            print(f"  Median excess return: {df[excess_col].median():.2f}%")



def save_final_dataset(df, sp500):
    """Saves final dataset"""
    sp500_last_date = pd.to_datetime(sp500.index, utc=True).tz_convert(None).max()
    window_cutoff = sp500_last_date - pd.Timedelta(days=85)

    before = len(df)
    df = df[df['traded_date'] <= window_cutoff].copy()
    dropped = before - len(df)
    print(f"\nDropped {dropped} trades with incomplete 90-day windows "
          f"(traded after {window_cutoff.date()}, last market data: {sp500_last_date.date()})")

    output_path = 'data/trades_with_returns.csv'
    df.to_csv(output_path, index=False)

    print(f"\n" + "="*60)
    print("SAVED FINAL DATASET")
    print("="*60)
    print(f"File:  {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")


def main():
    print("="*60)
    print("CALCULATING STOCK RETURNS")
    print("="*60)

    trades, stock_prices, sp500, sector_etf_prices = load_data()
    returns_df = calculate_all_returns(trades, stock_prices, sp500, sector_etf_prices)
    final_df = merge_returns_with_trades(trades, returns_df)
    print_summary_statistics(final_df)
    save_final_dataset(final_df, sp500)

    print("\n\nDone\n")


if __name__ == "__main__":
    main()