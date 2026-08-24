'''
ng_storage_study.py
-------------------
Event study: NG intraday behaviour around the weekly EIA natural-gas storage
release (Lower-48 working gas, released ~10:30 NY on Thursdays).

Splits each release day into pre- and post-release windows and compares
volume, return, and volatility.

Data is cached to disk (research/.cache) so repeated runs don't re-read the
raw 1-minute CSVs or re-hit the EIA API. Pass --refresh to rebuild the cache.
'''

import os
import sys
import hashlib
import pickle
from pathlib import Path

ROOT_DIR = Path.home() / 'quant_research' / 'futures'
sys.path.insert(0, str(ROOT_DIR))

import argparse
import requests
import pandas as pd

from data.config import START_DATE, END_DATE, INSTRUMENT_CALENDARS, COMMODITY_SESSION_CLOSE
from data.data_loader import load_ohlcv, get_front_month_daily, get_all_sessions
from data.data_processor import (
    build_continuous_series,
    get_session_returns,
    infer_session_open,
)

CACHE_DIR  = ROOT_DIR / 'research' / '.cache'
EIA_SERIES = 'NG.NW2_EPG0_SWO_R48_BCF.W'   # Lower-48 working gas in storage, weekly, Bcf
INSTRUMENT = 'NG'


# ── Cache ───────────────────────────────────────────────────────────────────────

def cached(key: str, builder, refresh: bool = False):
    '''
    Return builder() but persist its result to research/.cache/<key>.pkl.
    `key` should encode every input that changes the output, so a parameter
    change produces a different file and stale results are never served.
    '''
    CACHE_DIR.mkdir(exist_ok=True)
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    path   = CACHE_DIR / f'{digest}.pkl'

    if path.exists() and not refresh:
        print(f'cache hit  : {key}')
        with open(path, 'rb') as f:
            return pickle.load(f)

    print(f'cache build: {key}')
    obj = builder()
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return obj


# ── EIA storage ──────────────────────────────────────────────────────────────────

def load_storage_events() -> pd.DataFrame:
    '''Pull weekly EIA storage and build release events timed at 10:30 NY.'''
    api_key = os.getenv('EIA_API_KEY')
    if not api_key:
        raise RuntimeError('Set EIA_API_KEY in your environment before running.')

    res = requests.get(
        f'https://api.eia.gov/v2/seriesid/{EIA_SERIES}',
        params={'api_key': api_key}, timeout=30,
    )
    res.raise_for_status()

    storage = (
        pd.DataFrame(res.json()['response']['data'])
        .rename(columns={'period': 'week_ending', 'value': 'storage_bcf'})
        [['week_ending', 'storage_bcf']]
    )
    storage['week_ending'] = pd.to_datetime(storage['week_ending'])
    storage['storage_bcf'] = pd.to_numeric(storage['storage_bcf'])
    storage = storage.sort_values('week_ending').reset_index(drop=True)

    storage['storage_change_bcf'] = storage['storage_bcf'].diff()
    storage['storage_change_pct'] = storage['storage_bcf'].pct_change() * 100
    storage['release_date'] = storage['week_ending'] + pd.Timedelta(days=6)
    storage['release_ts_ny'] = (
        pd.to_datetime(storage['release_date'].dt.strftime('%Y-%m-%d') + ' 10:30')
        .dt.tz_localize('America/New_York')
    )

    events = storage[[
        'week_ending', 'release_ts_ny', 'storage_bcf',
        'storage_change_bcf', 'storage_change_pct',
    ]].copy()
    events['release_date'] = events['release_ts_ny'].dt.date
    events['release_time'] = events['release_ts_ny'].dt.time
    return events


# ── NG continuous + sessions ───────────────────────────────────────────────────

def load_ng_sessions():
    '''Load NG, build the back-adjusted continuous series, return intraday sessions.'''
    calendar      = INSTRUMENT_CALENDARS[INSTRUMENT]
    ohlcv         = load_ohlcv(START_DATE, END_DATE, [INSTRUMENT])
    session_open  = infer_session_open(ohlcv, freq='30min')
    front_months  = get_front_month_daily(ohlcv, calendar=calendar)
    continuous    = build_continuous_series(ohlcv, front_months)
    intraday, _   = get_all_sessions(
        continuous, calendar=calendar,
        session_open=session_open, session_close=COMMODITY_SESSION_CLOSE,
    )
    return intraday


# ── Analysis ─────────────────────────────────────────────────────────────────────

def build_release_bars(intraday: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    '''Merge storage releases onto intraday bars and flag pre/post release.'''
    cols = ['open', 'high', 'low', 'close', 'volume', 'time', 'date',
            'storage_bcf', 'storage_change_bcf', 'storage_change_pct', 'release_time']
    bars = intraday.merge(
        events, left_on='session_date', right_on='release_date', how='inner',
    )[cols]

    release_time = bars['release_time'].iloc[0]
    if bars['release_time'].nunique() != 1:
        print(f'⚠ multiple release times present: {sorted(bars["release_time"].unique())}')

    # Keep the release bar itself — it holds the reaction — and count it as
    # post-release (the print lands at its open).
    bars['after_release'] = bars['time'] >= release_time
    return bars


def compare_release_days(intraday: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    '''
    Whole-day comparison: do days WITH a storage release behave differently
    from days without one? Uses daily intraday session returns (open→close)
    and session volume, flagged by whether the date is a release day.
    '''
    intraday_ret, _ = get_session_returns(intraday)
    intraday_ret = intraday_ret.reset_index()

    release_dates = set(events['release_date'].unique())
    intraday_ret['event_day'] = intraday_ret['date'].isin(release_dates)

    return intraday_ret.groupby('event_day').agg(
        n_days     = ('return', 'count'),
        mean_ret   = ('return', 'mean'),
        vol_ret    = ('return', 'std'),
        avg_volume = ('volume', 'mean'),
        std_volume = ('volume', 'std'),
    ).rename(index={False: 'no_release', True: 'release_day'})


def event_zscores(intraday: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    '''
    For release days only: z-score the storage change and z-score the day's
    intraday return, so the surprise and the reaction are on a comparable scale.

    Note: storage_change_bcf is strongly seasonal (summer injections vs winter
    withdrawals), so a plain z-score mixes regimes — this is the basic version;
    a seasonally-adjusted surprise would be the next refinement.
    '''
    intraday_ret, _ = get_session_returns(intraday)
    intraday_ret = intraday_ret.reset_index()

    df = intraday_ret.merge(
        events[['release_date', 'storage_change_pct']],
        left_on='date', right_on='release_date', how='inner',
    )

    df['storage_z'] = (df['storage_change_pct'] - df['storage_change_pct'].mean()) / df['storage_change_pct'].std()
    df['return_z']  = (df['return'] - df['return'].mean()) / df['return'].std()

    return df[['date', 'storage_change_pct', 'storage_z', 'return', 'return_z']].set_index('date')


def summarise(bars: pd.DataFrame) -> pd.DataFrame:
    '''Pre vs post release: volume, return, and per-bar volatility, pooled.'''
    bars = bars.copy()
    bars['bar_ret'] = bars['close'] / bars['open'] - 1
    return bars.groupby('after_release').agg(
        n_bars     = ('bar_ret', 'count'),
        total_vol  = ('volume',  'sum'),
        avg_vol    = ('volume',  'mean'),
        mean_ret   = ('bar_ret', 'mean'),
        volatility = ('bar_ret', 'std'),
    ).rename(index={False: 'pre_release', True: 'post_release'})


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Rebuild the data cache.')
    args = parser.parse_args()

    key = f'{INSTRUMENT}|{START_DATE}|{END_DATE}'
    events   = cached(f'eia|{EIA_SERIES}', load_storage_events, refresh=args.refresh)
    intraday = cached(f'sessions|{key}', load_ng_sessions, refresh=args.refresh)

    day_compare = compare_release_days(intraday, events)
    print('\n── release days vs non-release days (whole-day intraday session) ──')
    print(day_compare.to_string())

    z = event_zscores(intraday, events)
    print('\n── event-day z-scores (storage change vs intraday return) ──')
    print(z.head(10).round(3).to_string())
    print(f'\ncorr(storage_z, return_z): {z["storage_z"].corr(z["return_z"]):+.3f}  (n={len(z)})')

    bars    = build_release_bars(intraday, events)
    summary = summarise(bars)

    print(f'\nrelease days: {bars["date"].nunique()}')
    print('\n── pre vs post release (intraday bars on release days) ──')
    print(summary.to_string())


if __name__ == '__main__':
    main()
