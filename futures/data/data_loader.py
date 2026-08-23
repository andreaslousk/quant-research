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
import re
import pandas as pd
from data.config import (
    PRICE_SCALE, TZ, SESSION_OPEN, SESSION_CLOSE,
    STAT_TYPE_SETTLEMENT, STAT_TYPE_OPEN_INTEREST, EQUITIES, DATA_DIR, INSTRUMENT_CALENDARS,
)
from data.readers import read_records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_by_instruments(df: pd.DataFrame, instruments: list[str]) -> pd.DataFrame:
    '''Filter rows to symbols starting with one of the given root symbols.'''
    pattern = '|'.join(f'^{r}' for r in instruments)
    return df[df['symbol'].str.match(pattern)].reset_index(drop=True)


_DATE_RANGE_RE = re.compile(r'-(\d{8})-(\d{8})\.')

# Outright contract symbols are root + month code + 1-2 digit year (e.g. NGF9, NGU27, 6EZ8).
# Anything else is a spread or strip — calendar spreads "NGF9-NGG9", strips "NG:SA 07M J9" —
# and is excluded. Allowlisting the outright shape is robust to every spread delimiter.
_OUTRIGHT_RE = r'[A-Z0-9]+[FGHJKMNQUVXZ][0-9]{1,2}'


def _file_in_range(path: str, start_date, end_date) -> bool:
    '''
    Keep a file if its filename date range overlaps [start_date, end_date].
    Filenames look like glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.<sym>.csv.
    Dates are the raw UTC range; we pad by 1 day each side so the NY-converted
    `date` boundary rows are never dropped. Files with no parseable range are kept.
    '''
    m = _DATE_RANGE_RE.search(os.path.basename(path))
    if not m:
        return True
    f_start = pd.Timestamp(m.group(1)).date()
    f_end   = pd.Timestamp(m.group(2)).date()
    start   = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).date()
    end     = (pd.Timestamp(end_date)   + pd.Timedelta(days=1)).date()
    return f_start <= end and f_end >= start


def _get_dirs(instruments: list[str]) -> tuple[list[str], list[str]]:
    '''Resolve (ohlcv_dirs, stats_dirs) for the given instruments.'''
    ohlcv_dirs = [os.path.join(DATA_DIR, i, 'futures_ohlcv') for i in instruments]
    stats_dirs = [os.path.join(DATA_DIR, i, 'futures_statistics') for i in instruments]
    return ohlcv_dirs, stats_dirs


def _data_files(directory: str) -> list[str]:
    '''Sorted Databento record files in `directory` (csv / dbn / dbn.zst). The
    `glbx-mdp3-*` prefix skips the condition/metadata/manifest json sidecars.'''
    files = sorted(glob.glob(os.path.join(directory, 'glbx-mdp3-*')))
    if not files:
        raise FileNotFoundError(f'No data files found in {directory}')
    return files


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    '''Downsample raw OHLCV bars to `freq` per symbol (open=first, high=max,
    low=min, close=last, volume=sum). Operates on the raw int columns (ts_event in
    ns) before cleaning, and is applied per file so memory stays bounded — the 1s
    source is coarsened to e.g. 1min the moment it is read, never held whole.'''
    bucket = pd.Timedelta(freq).value                       # bucket width in ns
    df = df.sort_values('ts_event')
    df['ts_event'] = (df['ts_event'] // bucket) * bucket    # floor to the bucket (UTC ns)
    return (df.groupby(['symbol', 'ts_event'], sort=False)
              .agg(open=('open', 'first'), high=('high', 'max'),
                   low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'))
              .reset_index())


# ── Loading ───────────────────────────────────────────────────────────────────

def load_ohlcv(
    start_date: str,
    end_date: str,
    instruments: list[str] = EQUITIES,
    freq: str | None = '1min',
    verbose: bool = False,
) -> pd.DataFrame:
    '''
    Load, clean and filter OHLCV bars (CSV or DBN) for the given instruments and
    date range. Source bars are downsampled per file to `freq` (default '1min',
    replicating the historical 1m pipeline); pass freq=None to keep the native
    resolution (e.g. 1s) for a focused window.
    Returns columns: ts_event, dt_utc, dt_ny, date, time, day, symbol, open, high, low, close, volume
    '''
    ohlcv_dirs, _ = _get_dirs(instruments)
    dfs = []

    for directory in ohlcv_dirs:
        for f in _data_files(directory):
            if not _file_in_range(f, start_date, end_date):
                continue
            part = read_records(
                f,
                usecols=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol'],
                dtype={
                    'ts_event': 'int64', 'open': 'int64', 'high': 'int64',
                    'low': 'int64', 'close': 'int64', 'volume': 'int64', 'symbol': 'str',
                },
            )
            if freq is not None:
                part = _resample_ohlcv(part, freq)
            dfs.append(part)

    if not dfs:
        raise FileNotFoundError(f'No OHLCV files overlap {start_date} → {end_date} for {instruments}')
    df = pd.concat(dfs, ignore_index=True)
    df = _clean_ohlcv(df)
    df = _filter_by_instruments(df, instruments)

    start = pd.Timestamp(start_date).date()
    end   = pd.Timestamp(end_date).date()
    df    = df[(df['date'] >= start) & (df['date'] <= end)].reset_index(drop=True)

    if verbose:
        print(f'OHLCV loaded: {len(df):,} rows | {start} → {end} | instruments: {instruments}')

    return df


def load_statistics(
    start_date: str,
    end_date: str,
    instruments: list[str] = EQUITIES,
    verbose: bool = False,
) -> pd.DataFrame:
    '''
    Load daily settlement + open interest for OUTRIGHT contracts from the Statistics
    CSVs (spreads/strips excluded). Settlement is stat_type=3 (value in `price`);
    open interest is stat_type=9 (value in `quantity`). Rows are dated by `ts_ref`
    (the settlement date), not `ts_event` (publication time).
    Returns one row per (date, symbol): date, symbol, settlement, open_interest.
    '''
    _, stats_dirs = _get_dirs(instruments)
    dfs = []

    for directory in stats_dirs:
        for f in _data_files(directory):
            if not _file_in_range(f, start_date, end_date):
                continue
            dfs.append(read_records(
                f,
                usecols=['ts_ref', 'price', 'quantity', 'stat_type', 'symbol'],
                dtype={
                    'ts_ref': 'int64', 'price': 'int64', 'quantity': 'int64',
                    'stat_type': 'int32', 'symbol': 'str',
                },
            ))

    if not dfs:
        raise FileNotFoundError(f'No stats files overlap {start_date} → {end_date} for {instruments}')
    df = pd.concat(dfs, ignore_index=True)
    df = _clean_stats(df)
    df = _filter_by_instruments(df, instruments)

    start = pd.Timestamp(start_date).date()
    end   = pd.Timestamp(end_date).date()
    df    = df[(df['date'] >= start) & (df['date'] <= end)].reset_index(drop=True)

    if verbose:
        print(f'Statistics loaded:  {len(df):,} (date, symbol) rows | {start} → {end} | instruments: {instruments}')

    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    '''Remove spreads and bad prices, add datetime columns, scale prices.'''
    df = df[df['symbol'].str.fullmatch(_OUTRIGHT_RE, na=False)]
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
    '''Outright settlement (stat_type=3, from `price`) + open interest (stat_type=9,
    from `quantity`), dated by `ts_ref` (the settlement date, not `ts_event`
    publication). Spreads/strips dropped. One row per (date, symbol); the last value
    wins if a stat is revised intraday.'''
    df = df[df['symbol'].str.fullmatch(_OUTRIGHT_RE, na=False)]
    df = df[df['ts_ref'] < 8_000_000_000_000_000_000].copy()      # drop the int64-max 'unset' ts_ref
    # ts_ref is 00:00 UTC of the settlement DATE — take the UTC date (NY-converting rolls it back a day)
    df['date'] = pd.to_datetime(df['ts_ref'], unit='ns', utc=True).dt.date

    settle = df[(df['stat_type'] == STAT_TYPE_SETTLEMENT) & (df['price'] > 0)].copy()
    settle['settlement'] = settle['price'] * PRICE_SCALE
    settle = settle.drop_duplicates(['date', 'symbol'], keep='last')[['date', 'symbol', 'settlement']]

    oi = df[df['stat_type'] == STAT_TYPE_OPEN_INTEREST].copy()
    oi['open_interest'] = oi['quantity']
    oi = oi.drop_duplicates(['date', 'symbol'], keep='last')[['date', 'symbol', 'open_interest']]

    return settle.merge(oi, on=['date', 'symbol'], how='left').reset_index(drop=True)


# ── Trading days ──────────────────────────────────────────────────────────────

def get_trading_days(start_date: str, end_date: str, calendar: str = 'NYSE') -> set:
    '''Return set of trading days (as datetime.date) between start and end for the given calendar.'''
    import pandas_market_calendars as mcal
    cal      = mcal.get_calendar(calendar)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    return set(schedule.index.date)


# ── Session splitting ─────────────────────────────────────────────────────────

def get_all_sessions(
    ohlcv: pd.DataFrame,
    calendar: str = 'NYSE',
    session_open: object = SESSION_OPEN,
    session_close: object = SESSION_CLOSE,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trading_days   = get_trading_days(str(ohlcv['date'].min()), str(ohlcv['date'].max()), calendar=calendar)
    mask = (ohlcv['time'] >= session_open) & (ohlcv['time'] < session_close) & (ohlcv['date'].isin(trading_days))
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

def get_front_month_daily(ohlcv: pd.DataFrame, calendar: str = 'NYSE', verbose: bool = False) -> pd.DataFrame:
    '''
    Returns daily front month mapping (date, front_month) by selecting
    the highest volume symbol on each trading day.
    Uses full-day volume so the result is correct for 24-hour markets.
    '''
    trading_days = get_trading_days(str(ohlcv['date'].min()), str(ohlcv['date'].max()), calendar=calendar)
    session = ohlcv[ohlcv['date'].isin(trading_days)]

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
