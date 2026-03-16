"""
Returns Calculator
==================

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
    """Load trades and stock price data."""
    print("Loading data...")

    # Load trades (check both possible locations)
    try:
        trades = pd.read_csv('data/trades_with_features.csv')
    except FileNotFoundError:
        trades = pd.read_csv('data/trades_with_features.csv')

    trades['traded_date'] = pd.to_datetime(trades['traded_date'])
    print(f"Loaded {len(trades)} trades")

    # Load stock prices
    with open('data/stock_prices.pkl', 'rb') as f:
        stock_prices = pickle.load(f)
    print(f"Loaded prices for {len(stock_prices)} stocks")

    # Load S&P 500
    sp500 = pd.read_csv('data/sp500_prices.csv', index_col=0, parse_dates=True)
    print(f"Loaded S&P 500 benchmark")

    return trades, stock_prices, sp500


def get_price_on_date(stock_df, target_date, lookback_days=5):
    """
    Get stock price on a specific date, with lookback if market was closed.

    Args:
        stock_df: DataFrame with stock prices (index is date)
        target_date: Date to get price for
        lookback_days: How many days to look back if exact date not found

    Returns:
        float: Close price, or None if not found
    """
    # Ensure index is datetime and normalized (no timezone for comparison)
    # Using a cached version of the index as strings for speed
    if not hasattr(get_price_on_date, "_index_cache"):
        get_price_on_date._index_cache = {}
    
    df_id = id(stock_df)
    if df_id not in get_price_on_date._index_cache:
        # Convert index to DatetimeIndex if it isn't already, then to YYYY-MM-DD strings
        if not isinstance(stock_df.index, pd.DatetimeIndex):
            dt_index = pd.to_datetime(stock_df.index, utc=True)
        else:
            dt_index = stock_df.index
        
        get_price_on_date._index_cache[df_id] = dt_index.strftime('%Y-%m-%d')

    date_strings = get_price_on_date._index_cache[df_id]
    
    # Create normalized date string (YYYY-MM-DD) for lookup
    target_str = target_date.strftime('%Y-%m-%d')
    
    # Check if exact date exists
    mask = date_strings == target_str
    if mask.any():
        return stock_df.loc[mask, 'Close'].iloc[0]

    # Look back up to lookback_days
    for i in range(1, lookback_days + 1):
        check_date = target_date - timedelta(days=i)
        check_str = check_date.strftime('%Y-%m-%d')
        mask = date_strings == check_str
        if mask.any():
            return stock_df.loc[mask, 'Close'].iloc[0]

    return None


def calculate_return(price_start, price_end):
    """Calculate percentage return."""
    if price_start is None or price_end is None or price_start == 0:
        return None
    return ((price_end - price_start) / price_start) * 100


def calculate_returns_for_trade(trade, stock_prices, sp500, horizons=[30, 60, 90]):
    """
    Calculate returns at multiple time horizons for a single trade.

    Args:
        trade: Single row from trades DataFrame
        stock_prices: Dict of ticker -> price DataFrame
        sp500: S&P 500 price DataFrame
        horizons: List of days to calculate returns for

    Returns:
        dict: Returns at each horizon + comparison to S&P 500
    """
    ticker = trade['ticker']
    trade_date = trade['traded_date']

    results = {
        'ticker': ticker,
        'traded_date': trade_date,
    }

    # Check if we have data for this ticker
    if ticker not in stock_prices:
        for horizon in horizons:
            results[f'return_{horizon}d'] = None
            results[f'sp500_return_{horizon}d'] = None
            results[f'excess_return_{horizon}d'] = None
            results[f'outperformed_{horizon}d'] = None
        return results

    stock_df = stock_prices[ticker]

    # Get price at trade date
    price_at_trade = get_price_on_date(stock_df, trade_date)

    if price_at_trade is None:
        for horizon in horizons:
            results[f'return_{horizon}d'] = None
            results[f'sp500_return_{horizon}d'] = None
            results[f'excess_return_{horizon}d'] = None
            results[f'outperformed_{horizon}d'] = None
        return results

    # Get S&P 500 price at trade date
    sp500_price_at_trade = get_price_on_date(sp500, trade_date)

    # Calculate returns at each horizon
    for horizon in horizons:
        future_date = trade_date + timedelta(days=horizon)

        # Stock return
        price_future = get_price_on_date(stock_df, future_date)
        stock_return = calculate_return(price_at_trade, price_future)
        results[f'return_{horizon}d'] = stock_return

        # S&P 500 return
        sp500_price_future = get_price_on_date(sp500, future_date)
        sp500_return = calculate_return(sp500_price_at_trade, sp500_price_future)
        results[f'sp500_return_{horizon}d'] = sp500_return

        # Excess return (stock - market)
        if stock_return is not None and sp500_return is not None:
            excess_return = stock_return - sp500_return
            results[f'excess_return_{horizon}d'] = excess_return

            # Binary target: outperformed by more than 5%?
            results[f'outperformed_{horizon}d'] = 1 if excess_return > 5.0 else 0
        else:
            results[f'excess_return_{horizon}d'] = None
            results[f'outperformed_{horizon}d'] = None

    return results


def calculate_all_returns(trades, stock_prices, sp500):
    """Calculate returns for all trades."""
    print("\nCalculating returns for all trades...")

    returns_list = []

    for idx, trade in trades.iterrows():
        if idx % 10 == 0:
            print(f"Processing trade {idx + 1}/{len(trades)}")

        trade_returns = calculate_returns_for_trade(trade, stock_prices, sp500)
        returns_list.append(trade_returns)

    # Convert to DataFrame
    returns_df = pd.DataFrame(returns_list)

    return returns_df


def merge_returns_with_trades(trades, returns_df):
    """Merge calculated returns back with trade data."""
    print("\nMerging returns with trade data...")

    # Merge on ticker and traded_date
    trades['traded_date'] = pd.to_datetime(trades['traded_date'])
    returns_df['traded_date'] = pd.to_datetime(returns_df['traded_date'])

    merged = trades.merge(
        returns_df,
        on=['ticker', 'traded_date'],
        how='left'
    )

    print(f"Merged dataset has {len(merged)} rows")

    return merged


def print_summary_statistics(df):
    """Print summary statistics about returns."""
    print("\n" + "="*60)
    print("RETURNS SUMMARY STATISTICS")
    print("="*60)

    horizons = [30, 60, 90]

    for horizon in horizons:
        return_col = f'return_{horizon}d'
        sp500_col = f'sp500_return_{horizon}d'
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


def save_final_dataset(df):
    """Save final dataset with returns."""
    output_path = 'data/trades_with_returns.csv'
    df.to_csv(output_path, index=False)

    print(f"\n" + "="*60)
    print("SAVED FINAL DATASET")
    print("="*60)
    print(f"File: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Print column list
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")


def main():
    """Main function."""
    print("="*60)
    print("CALCULATING STOCK RETURNS")
    print("="*60)

    # Load data
    trades, stock_prices, sp500 = load_data()

    # Calculate returns
    returns_df = calculate_all_returns(trades, stock_prices, sp500)

    # Merge with original data
    final_df = merge_returns_with_trades(trades, returns_df)

    # Print statistics
    print_summary_statistics(final_df)

    # Save
    save_final_dataset(final_df)

    print("\n" + "="*60)
    print("COMPLETE! Ready for ML modeling")
    print("="*60)
    print("\nNext steps:")
    print("1. Open Jupyter notebook for exploratory data analysis")
    print("2. Visualize returns distributions")
    print("3. Check class balance for target variable")
    print("4. Build baseline ML models")


if __name__ == "__main__":
    main()