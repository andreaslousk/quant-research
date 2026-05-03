# autopep8: off
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path.home() / 'quant_research'
sys.path.insert(0, str(ROOT_DIR))

# isort: split
from portfolio_management.data.config import *

# autopep8: on


def adjust_dividends_for_splits(dividends_df, splits_df):
    # Step 1: normalise dividend amounts to account for subsequent splits
    '''Adjust historical dividend amounts to account for subsequent stock splits.

    Parameters
    ----------
    dividends_df : pd.DataFrame
        Columns: ['ticker', 'ex_dividend_date', 'dividend_amount']
    splits_df : pd.DataFrame
        Columns: ['ticker', 'split_date', 'split_ratio']
        split_ratio is new_shares/old_shares (e.g. 2.0 for a 2-for-1 split).

    Returns
    -------
    pd.DataFrame
        Copy of dividends_df with an 'adjusted_dividend' column added.
    '''
    dividends = dividends_df.copy()
    splits = splits_df.copy()

    dividends['ex_dividend_date'] = pd.to_datetime(
        dividends['ex_dividend_date'])
    splits['split_date'] = pd.to_datetime(splits['split_date'])

    result_list = []

    for ticker in dividends['ticker'].unique():
        ticker_dividends = dividends[dividends['ticker'] == ticker].copy()
        ticker_splits = splits[splits['ticker'] == ticker].copy()

        if len(ticker_splits) == 0:
            ticker_dividends['adjusted_dividend'] = ticker_dividends['dividend_amount']
            result_list.append(ticker_dividends[[
                               'ticker', 'ex_dividend_date', 'dividend_amount', 'adjusted_dividend']])
            continue

        ticker_splits = ticker_splits.sort_values(
            'split_date', ascending=False)
        ticker_splits['cumulative_split_factor'] = ticker_splits['split_ratio'].cumprod()

        merged = pd.merge_asof(
            ticker_dividends.sort_values('ex_dividend_date'),
            ticker_splits[['split_date', 'cumulative_split_factor']
                          ].sort_values('split_date'),
            left_on='ex_dividend_date',
            right_on='split_date',
            direction='forward'
        )

        merged['cumulative_split_factor'] = merged['cumulative_split_factor'].fillna(
            1.0)
        merged['adjusted_dividend'] = merged['dividend_amount'] / \
            merged['cumulative_split_factor']

        result_list.append(
            merged[['ticker', 'ex_dividend_date', 'dividend_amount', 'adjusted_dividend']])

    return pd.concat(result_list, ignore_index=True)


def adjust_prices_for_dividends(prices_df, adjusted_dividends_df):
    # Step 2: apply the normalised dividends to OHLC prices
    '''Adjust OHLC prices for dividends across multiple tickers.

    Assumes prices are already adjusted for splits.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Columns: ['ticker', 'date', 'open', 'high', 'low', 'close']
    adjusted_dividends_df : pd.DataFrame
        Columns: ['ticker', 'ex_dividend_date', 'adjusted_dividend']

    Returns
    -------
    pd.DataFrame
        Copy of prices_df with OHLC columns scaled by the cumulative
        dividend adjustment factor.
    '''
    prices = prices_df.copy()
    dividends = adjusted_dividends_df.copy()

    prices['date'] = pd.to_datetime(prices['date'])
    dividends['ex_dividend_date'] = pd.to_datetime(
        dividends['ex_dividend_date'])

    result_list = []

    for ticker in prices['ticker'].unique():
        ticker_prices = prices[prices['ticker']
                               == ticker].copy().sort_values('date')
        ticker_dividends = dividends[dividends['ticker'] == ticker].copy(
        ).sort_values('ex_dividend_date')

        if len(ticker_dividends) == 0:
            result_list.append(ticker_prices)
            continue

        dividends_with_prev_close = pd.merge_asof(
            ticker_dividends,
            ticker_prices[['date', 'close']],
            left_on='ex_dividend_date',
            right_on='date',
            direction='backward'
        )
        dividends_with_prev_close = dividends_with_prev_close.rename(
            columns={'close': 'prev_close'})

        dividends_with_prev_close['adj_factor'] = 1 - (
            dividends_with_prev_close['adjusted_dividend'] /
            dividends_with_prev_close['prev_close']
        )

        dividends_with_prev_close = dividends_with_prev_close.sort_values(
            'ex_dividend_date', ascending=True)
        dividends_with_prev_close['cumulative_adj_factor'] = dividends_with_prev_close['adj_factor'].cumprod(
        )

        merged = pd.merge_asof(
            ticker_prices,
            dividends_with_prev_close[[
                'ex_dividend_date', 'cumulative_adj_factor']],
            left_on='date',
            right_on='ex_dividend_date',
            direction='forward'
        )

        merged['cumulative_adj_factor'] = merged['cumulative_adj_factor'].fillna(
            1.0)

        merged['open'] = merged['open'] * merged['cumulative_adj_factor']
        merged['high'] = merged['high'] * merged['cumulative_adj_factor']
        merged['low'] = merged['low'] * merged['cumulative_adj_factor']
        merged['close'] = merged['close'] * merged['cumulative_adj_factor']

        result_list.append(merged.drop(
            columns=['ex_dividend_date', 'cumulative_adj_factor']))

    return pd.concat(result_list, ignore_index=True)


def apply_dividend_adjustment(df, dividends_df):
    # Orchestrator: calls adjust_prices_for_dividends per ticker
    '''Apply dividend price adjustments to all tickers in a combined DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with a 'ticker' column.
    dividends_df : pd.DataFrame
        Dividend data with columns ['ticker', 'ex_dividend_date', 'adjusted_dividend'].

    Returns
    -------
    pd.DataFrame
        Price data with dividend-adjusted OHLC values.
    '''
    print('Applying dividend adjustments (subtracting cash amounts)...')
    df = df.groupby('ticker', group_keys=False).apply(
        lambda x: adjust_prices_for_dividends(x, dividends_df))
    print('✓ Dividend adjustments complete')
    return df


def add_metadata(df, stocks_dict):
    '''Add sector and region metadata to a ticker DataFrame using a lookup dict.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'ticker' column.
    stocks_dict : dict
        Mapping of ticker -> {'sector': ..., 'region': ...}.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'sector' and 'region' columns added.
    '''
    print('Adding sector and region metadata...')
    df['region'] = df['ticker'].map(
        lambda x: stocks_dict.get(x, {}).get('region', 'Unknown'))
    df['sector'] = df['ticker'].map(
        lambda x: stocks_dict.get(x, {}).get('sector', 'Unknown'))
    print('✓ Metadata added')
    return df


def calculate_returns(group, returns_dict=RETURNS_PERIODS):
    '''Calculate percentage returns for multiple periods for a single ticker group.

    Adds columns named 'return_{period_label}' for each entry in returns_dict.

    Parameters
    ----------
    group : pd.DataFrame
        Single-ticker price DataFrame with a 'close' column, sorted by date.
    returns_dict : dict, optional
        Mapping of {period_days: period_label}.

    Returns
    -------
    pd.DataFrame
        Input group with return columns appended.
    '''
    for period, period_label in returns_dict.items():
        group[f'return_{period_label}'] = group['close'].pct_change(
            periods=period)
    return group


def calculate_volatility(group, vols_dict=VOLS_PERIODS):
    '''Calculate annualised rolling volatility for multiple windows for a single ticker group.

    Adds columns named 'vol_{period_label}' for each entry in vols_dict.

    Parameters
    ----------
    group : pd.DataFrame
        Single-ticker DataFrame with a 'return_1d' column, sorted by date.
    vols_dict : dict, optional
        Mapping of {rolling_window_days: period_label}.

    Returns
    -------
    pd.DataFrame
        Input group with volatility columns appended.
    '''
    for period, period_label in vols_dict.items():
        group[f'vol_{period_label}'] = group['return_1d'].rolling(
            period).std() * np.sqrt(252)
    return group


def calculate_volume_change(group):
    '''Calculate the 1-month (21-day) percentage change in volume for a single ticker group.

    Adds a 'volume_change_1m' column to the group.

    Parameters
    ----------
    group : pd.DataFrame
        Single-ticker DataFrame with a 'volume' column, sorted by date.

    Returns
    -------
    pd.DataFrame
        Input group with 'volume_change_1m' column appended.
    '''
    group['volume_change_1m'] = group['volume'].pct_change(periods=21)
    return group


def process_all_tickers(df):
    '''Compute returns, volatility, and volume change for all tickers and drop NaN rows.

    Parameters
    ----------
    df : pd.DataFrame
        Combined price DataFrame with 'ticker', 'date', 'close', and 'volume' columns.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with return, volatility, and volume change columns added,
        and NaN rows removed.
    '''
    print('Processing all tickers (returns, volatility, volume)...')
    df = df.sort_values(['ticker', 'date'])
    df = df.groupby('ticker', group_keys=False).apply(calculate_returns)
    df = df.groupby('ticker', group_keys=False).apply(calculate_volatility)
    df = df.groupby('ticker', group_keys=False).apply(calculate_volume_change)
    df = df.dropna()
    print(f'✓ Processing complete. Rows after dropna: {len(df):,}')
    return df


def one_hot_encode_categoricals(df):
    '''One-hot encode the sector and region columns and append the dummies to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'sector' and 'region' columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with integer dummy columns for each sector and region appended.
    '''
    print('One-hot encoding sectors and regions...')
    sector_dummies = pd.get_dummies(df['sector']).astype(int)
    region_dummies = pd.get_dummies(df['region']).astype(int)
    df = pd.concat([df, sector_dummies, region_dummies], axis=1)
    print(
        f'✓ Added {len(sector_dummies.columns)} sector columns and {len(region_dummies.columns)} region columns')
    return df
