'''
data_loader.py
--------------
Data loading and transformation functions for the futures pipeline.
All config imported from config.py — nothing hardcoded here.
Functions are pure transforms with no side effects.
verbose=False by default — callers control logging.
'''

import glob
import os
import pandas as pd
from data.config import (
    PRICE_SCALE, TZ, SESSION_OPEN, SESSION_CLOSE,
    STAT_TYPE_SETTLEMENT, INSTRUMENTS,
    ES_OHLCV_DIR, ES_STATS_DIR,
    OTHER_OHLCV_DIR, OTHER_STATS_DIR,
    FX_OHLCV_DIRS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_by_instruments(df: pd.DataFrame, instruments: list[str]) -> pd.DataFrame:
    '''Filter rows to symbols starting with one of the given root symbols.'''
    pattern = '|'.join(f'^{r}' for r in instruments)
    return df[df['symbol'].str.match(pattern)].reset_index(drop=True)


def _get_dirs(instruments: list[str]) -> tuple[list[str], list[str]]:
    '''
    Resolve (ohlcv_dirs, stats_dirs) for the given instruments.
    ES has its own directory; NQ/RTY/YM share one; FX instruments each have their own.
    '''
    ohlcv_dirs, stats_dirs = [], []

    if 'ES' in instruments:
        ohlcv_dirs.append(ES_OHLCV_DIR)
        stats_dirs.append(ES_STATS_DIR)

    equity_others = [i for i in instruments if i not in ('ES', *FX_OHLCV_DIRS)]
    if equity_others:
        ohlcv_dirs.append(OTHER_OHLCV_DIR)
        stats_dirs.append(OTHER_STATS_DIR)

    for fx in instruments:
        if fx in FX_OHLCV_DIRS:
            ohlcv_dirs.append(FX_OHLCV_DIRS[fx])

    return ohlcv_dirs, stats_dirs


# ── Loading ───────────────────────────────────────────────────────────────────

def load_ohlcv(
    start_date: str,
    end_date: str,
    instruments: list[str] = INSTRUMENTS,
    verbose: bool = False,
) -> pd.DataFrame:
    '''
    Load, clean and filter OHLCV-1m CSVs for the given instruments and date range.
    Returns columns: ts_event, dt_utc, dt_ny, date, time, day, symbol, open, high, low, close, volume
    '''
    ohlcv_dirs, _ = _get_dirs(instruments)
    dfs = []

    for directory in ohlcv_dirs:
        files = sorted(glob.glob(os.path.join(directory, '*.csv')))
        if not files:
            raise FileNotFoundError(f'No CSVs found in {directory}')
        for f in files:
            dfs.append(pd.read_csv(
                f,
                usecols=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol'],
                dtype={
                    'ts_event': 'int64', 'open': 'int64', 'high': 'int64',
                    'low': 'int64', 'close': 'int64', 'volume': 'int64', 'symbol': 'str',
                },
            ))

    df = pd.concat(dfs, ignore_index=True)
    df = _clean_ohlcv(df)
    df = _filter_by_instruments(df, instruments)

    start = pd.Timestamp(start_date).date()
    end   = pd.Timestamp(end_date).date()
    df    = df[(df['date'] >= start) & (df['date'] <= end)].reset_index(drop=True)

    if verbose:
        print(f'OHLCV loaded: {len(df):,} rows | {start} → {end} | instruments: {instruments}')

    return df


def load_stats(
    start_date: str,
    end_date: str,
    instruments: list[str] = INSTRUMENTS,
    verbose: bool = False,
) -> pd.DataFrame:
    '''
    Load, clean and filter Statistics CSVs for the given instruments and date range.
    Only settlement prices (stat_type=3) are kept.
    Returns columns: date, symbol, settlement, dt_ny
    '''
    _, stats_dirs = _get_dirs(instruments)
    dfs = []

    for directory in stats_dirs:
        files = sorted(glob.glob(os.path.join(directory, '*.csv')))
        if not files:
            raise FileNotFoundError(f'No CSVs found in {directory}')
        for f in files:
            dfs.append(pd.read_csv(
                f,
                usecols=['ts_event', 'price', 'stat_type', 'symbol'],
                dtype={
                    'ts_event': 'int64', 'price': 'int64',
                    'stat_type': 'int32', 'symbol': 'str',
                },
            ))

    df = pd.concat(dfs, ignore_index=True)
    df = _clean_stats(df)
    df = _filter_by_instruments(df, instruments)

    start = pd.Timestamp(start_date).date()
    end   = pd.Timestamp(end_date).date()
    df    = df[(df['date'] >= start) & (df['date'] <= end)].reset_index(drop=True)

    if verbose:
        print(f'Stats loaded:  {len(df):,} rows | {start} → {end} | instruments: {instruments}')

    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    '''Remove spreads and bad prices, add datetime columns, scale prices.'''
    df = df[~df['symbol'].str.contains('-', na=False)]
    df = df[df['close'] > 0].copy()

    df['dt_utc'] = pd.to_datetime(df['ts_event'], unit='ns').dt.tz_localize('UTC')
    df['dt_ny']  = df['dt_utc'].dt.tz_convert(TZ)

    df['date']   = df['dt_ny'].dt.date
    df['time']   = df['dt_ny'].dt.time
    df['day']    = df['dt_ny'].dt.day_name()

    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * PRICE_SCALE

    return df.sort_values('ts_event').reset_index(drop=True)


def _clean_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''Filter to settlements for outright contracts, add datetime columns, scale prices.'''
    df = df[df['stat_type'] == STAT_TYPE_SETTLEMENT]
    df = df[~df['symbol'].str.contains('-', na=False)]
    df = df[df['price'] > 0].copy()

    df['dt_utc'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)
    df['dt_ny']  = df['dt_utc'].dt.tz_convert(TZ)
    df['date']   = df['dt_ny'].dt.date

    df['settlement'] = df['price'] * PRICE_SCALE

    return df[['date', 'symbol', 'settlement', 'dt_ny']].reset_index(drop=True)


# ── NYSE trading days ─────────────────────────────────────────────────────────

def get_nyse_trading_days(start_date: str, end_date: str) -> set:
    '''Return set of NYSE trading days (as datetime.date) between start and end.'''
    import pandas_market_calendars as mcal
    cal      = mcal.get_calendar('NYSE')
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    return set(schedule.index.date)


# ── Session splitting ─────────────────────────────────────────────────────────

def get_all_sessions(ohlcv: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    trading_days   = get_nyse_trading_days(str(ohlcv['date'].min()), str(ohlcv['date'].max()))
    mask = (ohlcv['time'] >= SESSION_OPEN) & (ohlcv['time'] < SESSION_CLOSE) & (ohlcv['date'].isin(trading_days))
    intraday_flat  = ohlcv[mask].copy()
    overnight_flat = ohlcv[~mask].copy()

    first_intraday = (
        intraday_flat.groupby('date')['dt_ny']
        .min()
        .reset_index()
        .rename(columns={'date': 'session_date', 'dt_ny': 'next_open'})
        .sort_values('next_open')
    )

    overnight_flat = pd.merge_asof(
        overnight_flat.sort_values('dt_ny'),
        first_intraday,
        left_on='dt_ny',
        right_on='next_open',
        direction='forward'
    ).dropna(subset=['session_date'])

    overnight_flat['day_of_week'] = pd.to_datetime(overnight_flat['session_date']).dt.day_name()
    intraday_flat['session_date'] = intraday_flat['date']
    intraday_flat['day_of_week']  = pd.to_datetime(intraday_flat['session_date']).dt.day_name()

    if verbose:
        print(f'Intraday bars:  {len(intraday_flat)}')
        print(f'Overnight bars: {len(overnight_flat)}')

    return intraday_flat, overnight_flat


def split_sessions(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split cleaned OHLCV data into intraday and overnight bars across all dates.
    Returns (intraday_df, overnight_df) — both retain all original columns including day.

    Intraday  : 09:30–16:00 ET
    Overnight : everything outside regular session hours
    '''
    time = ohlcv['time']

    intraday  = ohlcv[(time >= SESSION_OPEN) & (time <= SESSION_CLOSE)].reset_index(drop=True)
    overnight = ohlcv[(time < SESSION_OPEN)  | (time > SESSION_CLOSE)].reset_index(drop=True)

    return intraday, overnight


# ── Front month ───────────────────────────────────────────────────────────────

def get_front_month_daily(ohlcv: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    '''
    Returns daily front month mapping (date, front_month) by selecting
    the highest volume symbol during the regular session each day.
    '''
    trading_days = get_nyse_trading_days(str(ohlcv['date'].min()), str(ohlcv['date'].max()))
    session = ohlcv[
        (ohlcv['time'] >= SESSION_OPEN) &
        (ohlcv['time'] < SESSION_CLOSE) &
        (ohlcv['date'].isin(trading_days))
    ]

    front_months = (
        session
        .groupby(['date', 'symbol'])['volume']
        .sum()
        .reset_index()
        .sort_values('volume', ascending=False)
        .groupby('date')
        .first()
        .reset_index()
        .rename(columns={'symbol': 'front_month'})
        [['date', 'front_month']]
    )

    if verbose:
        print(f'Front months computed: {len(front_months)} days')

    return front_months
