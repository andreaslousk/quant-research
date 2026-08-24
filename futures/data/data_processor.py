'''
data_processor.py
-----------------
Transforms cleaned OHLCV data into derived time series.
All functions take DataFrames and return DataFrames — no side effects.
'''

import pandas as pd
from data.data_loader import load_ohlcv, get_front_month_daily, get_all_sessions
from data.config import (
    INSTRUMENT_CALENDARS, COMMODITIES,
    SESSION_OPEN, SESSION_CLOSE, COMMODITY_SESSION_CLOSE,
)


def build_continuous_series(ohlcv: pd.DataFrame, front_months: pd.DataFrame) -> pd.DataFrame:
    '''
    Build a back-adjusted continuous front-month series.
    Filters to the front month, detects rolls, and applies a multiplicative
    back-adjustment anchored to the most recent contract (recent prices stay
    unadjusted; historical prices are scaled) so the series is seamless across
    rolls. Replaces open/high/low/close in place with adjusted values —
    downstream code always works with `close`, never `adj_close`.

    Cumulative returns of the result equal those of a held-and-rolled futures
    position: in persistent contango (e.g. NG) the series bleeds — that carry is
    real, not an artifact.
    '''
    df = ohlcv.copy()
    df['date'] = df['dt_ny'].dt.date
    df = df.merge(front_months, on='date', how='left')
    df = df[df['symbol'] == df['front_month']].sort_values('dt_ny').reset_index(drop=True)

    df['prev_symbol'] = df['symbol'].shift(1)
    df['prev_date']   = df['date'].shift(1)
    df['roll']        = df['symbol'].notna() & df['prev_symbol'].notna() & (df['symbol'] != df['prev_symbol'])

    # Per-roll splice ratio new/old at the LAST minute on prev_date where BOTH
    # contracts trade — same-instant prices, so only the contract spread is
    # removed and no overnight move of either leg leaks into the series.
    c     = ohlcv[['dt_ny', 'date', 'symbol', 'close']]
    rolls = df.loc[df['roll'], ['prev_date', 'prev_symbol', 'symbol']].rename(
        columns={'prev_symbol': 'old_sym', 'symbol': 'new_sym'})
    old = (rolls.merge(c, left_on=['prev_date', 'old_sym'], right_on=['date', 'symbol'])
                [['prev_date', 'dt_ny', 'close']].rename(columns={'close': 'close_o'}))
    new = (rolls.merge(c, left_on=['prev_date', 'new_sym'], right_on=['date', 'symbol'])
                [['prev_date', 'dt_ny', 'close']].rename(columns={'close': 'close_n'}))
    shared = old.merge(new, on=['prev_date', 'dt_ny']).groupby('prev_date').last()
    factor = shared['close_n'] / shared['close_o']

    df['adj_factor'] = df['prev_date'].map(factor).where(df['roll']).fillna(1.0)
    # Reverse cumprod anchored to the present, EXCLUDING each bar's own factor so
    # the roll factor scales the old contract (bars before the roll), not the
    # new contract's first bar.
    df['cum_adj'] = df['adj_factor'][::-1].cumprod()[::-1] / df['adj_factor']

    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * df['cum_adj']

    return df.drop(columns=['prev_symbol', 'prev_date', 'roll', 'adj_factor', 'cum_adj', 'front_month'], errors='ignore')


def get_ohlcv_resampled(continuous: pd.DataFrame, freq: str = '1h') -> pd.DataFrame:
    '''
    Resample a continuous front-month series to the given frequency.
    Filters out the 17:00 settlement hour before and after resampling.
    prev_close is a plain shift(1) on the already-adjusted, already-filtered series
    so bar returns chain correctly across rolls.
    Returns DataFrame indexed by dt_ny with columns: close, prev_close.
    '''
    bars = (
        continuous[continuous['dt_ny'].dt.hour != 17]
        .set_index('dt_ny')['close']
        .resample(freq)
        .last()
        .reset_index()
    )
    bars = bars[bars['dt_ny'].dt.hour != 17]
    bars = bars.dropna(subset=['close']).sort_values('dt_ny')
    bars['prev_close'] = bars['close'].shift(1)
    return bars.set_index('dt_ny')


def filter_front_month(bars: pd.DataFrame, front_months: pd.DataFrame) -> pd.DataFrame:
    '''
    Filter resampled bars to front month contract only.
    Merges on date and keeps rows where symbol == front_month.
    Returns DataFrame indexed by dt_ny with symbol as column.
    Note: not used in the main pipeline — build_continuous_series handles
    front-month filtering with back-adjustment.
    '''
    bars = bars.reset_index()
    bars['date'] = bars['dt_ny'].dt.date
    front_months = front_months.copy()
    front_months['date'] = pd.to_datetime(front_months['date']).dt.date
    bars = bars.merge(front_months, on='date', how='left')
    bars = bars[bars['symbol'] == bars['front_month']]
    return bars.set_index('dt_ny').drop(columns=['date', 'front_month'])


def compute_bar_returns(bars: pd.DataFrame) -> pd.DataFrame:
    '''
    Add bar_return column: close / prev_close - 1.
    '''
    bars = bars.copy()
    bars['bar_return'] = bars['close'] / bars['prev_close'] - 1
    return bars


def get_profile_volume(continuous: pd.DataFrame, freq: str = '1h') -> pd.DataFrame:
    '''
    Resample minute volume from the continuous series to the given frequency.
    Returns DataFrame with columns: dt_ny, volume.
    '''
    vol = (
        continuous[continuous['dt_ny'].dt.hour != 17]
        .set_index('dt_ny')['volume']
        .resample(freq)
        .sum()
        .reset_index()
    )
    return vol[vol['dt_ny'].dt.hour != 17].set_index('dt_ny')


def get_volume_acceleration_profile(continuous: pd.DataFrame, freq: str = '1h') -> pd.Series:
    '''
    Compute average volume acceleration (bar-over-bar % change) by time of day.
    Acceleration is computed within each day then averaged across days per time slot.
    Returns Series indexed by time (HH:MM) named avg_vol_accel.
    '''
    vol = (
        continuous[continuous['dt_ny'].dt.hour != 17]
        .set_index('dt_ny')['volume']
        .resample(freq)
        .sum()
        .reset_index()
    )
    vol = vol[vol['dt_ny'].dt.hour != 17].reset_index(drop=True)
    vol['date']      = vol['dt_ny'].dt.date
    vol['time']      = vol['dt_ny'].dt.strftime('%H:%M')
    vol['vol_accel'] = vol.groupby('date')['volume'].pct_change()
    return vol.groupby('time')['vol_accel'].mean().rename('avg_vol_accel')


def get_session_returns(intraday: pd.DataFrame, has_volume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Compute intraday and overnight returns from intraday session bars.
    Input must already be filtered to the front-month continuous series.

      R_intraday  = C_t / O_t - 1        (open-to-close within session)
      R_overnight = O_t / C_{t-1} - 1   (prev session close to next open)

    O_t = first bar open on date t, C_t = last bar close on date t.

    Returns (intraday_returns, overnight_returns), each indexed by date
    with columns: open, close, return, day_of_week, session.
    intraday_returns also carries `volume` (session total) when `has_volume`
    (true for OHLCV bars; set False for statistics-sourced bars with no per-bar
    volume) — overnight never has volume, since this function only sees intraday
    bars, so the overnight window's volume is not available here regardless.
    '''
    agg = dict(open=('open', 'first'), close=('close', 'last'))
    if has_volume:
        agg['volume'] = ('volume', 'sum')
    daily = intraday.sort_values('dt_ny').groupby('date').agg(**agg).reset_index()
    daily['day_of_week']    = pd.to_datetime(daily['date']).dt.day_name()
    daily                   = daily.sort_values('date')
    daily['intraday_return']  = daily['close'] / daily['open'] - 1
    daily['overnight_return'] = daily['open'] / daily['close'].shift(1) - 1

    intraday_cols = ['date', 'open', 'close'] + (['volume'] if has_volume else []) + ['intraday_return', 'day_of_week']
    intraday_ret = daily[intraday_cols].copy()
    intraday_ret = intraday_ret.rename(columns={'intraday_return': 'return'})
    intraday_ret['session'] = 'intraday'

    overnight_ret = daily[['date', 'open', 'close', 'overnight_return', 'day_of_week']].dropna(subset=['overnight_return']).copy()
    overnight_ret = overnight_ret.rename(columns={'overnight_return': 'return'})
    overnight_ret['session'] = 'overnight'

    return (
        intraday_ret.set_index('date'),
        overnight_ret.set_index('date'),
    )


def get_session_summary(intraday_sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Compare intraday vs overnight session returns.
    Returns (summary, by_day):
      summary: indexed by session with columns avg_return, cum_return, sharpe
      by_day:  avg return by session and day of week
    '''
    intraday, overnight = get_session_returns(intraday_sessions)
    combined = pd.concat([intraday, overnight])

    def sharpe(x: pd.Series) -> float:
        return x.mean() / x.std() * (252 ** 0.5) if x.std() > 0 else float('nan')

    summary = combined.groupby('session')['return'].agg(
        avg_return='mean',
        cum_return=lambda x: (1 + x).prod() - 1,
        sharpe=sharpe,
    )

    total_cum = (1 + summary['cum_return']).prod() - 1
    summary.loc['total'] = {
        'avg_return': float('nan'),
        'cum_return': total_cum,
        'sharpe'    : float('nan'),
    }

    by_day    = combined.groupby(['session', 'day_of_week'])['return'].mean().unstack('day_of_week')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    by_day    = by_day[[d for d in day_order if d in by_day.columns]]

    return summary, by_day


def get_profile_summary(bars: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    '''
    Aggregate returns and volume by time of day.
    Works for any resampling frequency.
    Returns DataFrame indexed by time (HH:MM) with columns:
    avg_return, cum_return, avg_volume, cum_volume.
    '''
    dt          = bars.index.get_level_values('dt_ny') if isinstance(bars.index, pd.MultiIndex) else bars.index
    time_of_day = dt.strftime('%H:%M')

    returns_summary = bars.groupby(time_of_day)['bar_return'].agg(
        avg_return='mean',
        cum_return=lambda x: (1 + x).prod() - 1,
    )
    returns_summary.index.name = 'time'

    vol_dt          = vol.index.get_level_values('dt_ny')
    vol_time_of_day = vol_dt.strftime('%H:%M')

    vol_summary = vol.groupby(vol_time_of_day)['volume'].agg(
        avg_volume='mean',
        cum_volume='sum',
    )
    vol_summary.index.name = 'time'

    return returns_summary.join(vol_summary)


def infer_session_open(ohlcv: pd.DataFrame, freq: str = '30min', threshold: float = 0.10) -> object:
    '''
    Infer the effective session open from the cumulative volume profile.
    Computes average volume per time slot across all days, then returns the first
    slot whose cumulative share of total daily volume reaches `threshold`.
    Falls back to SESSION_OPEN if the threshold is never reached.
    '''
    vol = (
        ohlcv[ohlcv['dt_ny'].dt.hour != 17]
        .set_index('dt_ny')['volume']
        .resample(freq)
        .sum()
        .reset_index()
    )
    avg     = vol.groupby(vol['dt_ny'].dt.strftime('%H:%M'))['volume'].mean().sort_index()
    cum_pct = avg.cumsum() / avg.sum()
    above   = cum_pct[cum_pct >= threshold]
    return pd.Timestamp(above.index[0]).time() if not above.empty else SESSION_OPEN


def build_instrument_data(
    instruments: list[str],
    start_date: str,
    end_date: str,
    freq: str = '30min',
    verbose: bool = False,
) -> dict:
    '''
    Full pipeline for a list of instruments.
    Commodity instruments use an inferred session open and a fixed 16:30 close.
    Returns {instrument: {'summary', 'vol_accel', 'intraday_ret', 'overnight_ret'}}.
    '''
    results = {}
    for instrument in instruments:
        calendar  = INSTRUMENT_CALENDARS.get(instrument, 'NYSE')
        ohlcv     = load_ohlcv(start_date, end_date, [instrument], verbose=verbose)

        if instrument in COMMODITIES:
            session_open  = infer_session_open(ohlcv, freq=freq)
            session_close = COMMODITY_SESSION_CLOSE
        else:
            session_open  = SESSION_OPEN
            session_close = SESSION_CLOSE

        front_months = get_front_month_daily(ohlcv, calendar=calendar, verbose=verbose)
        continuous   = build_continuous_series(ohlcv, front_months)
        bars         = get_ohlcv_resampled(continuous, freq=freq)
        bars         = compute_bar_returns(bars)
        vol          = get_profile_volume(continuous, freq=freq)

        intraday_sessions, _ = get_all_sessions(
            continuous, calendar=calendar,
            session_open=session_open, session_close=session_close,
        )
        intraday_ret, overnight_ret = get_session_returns(intraday_sessions)

        results[instrument] = {
            'summary'       : get_profile_summary(bars, vol),
            'vol_accel'     : get_volume_acceleration_profile(continuous, freq=freq),
            'intraday_ret'  : intraday_ret,
            'overnight_ret' : overnight_ret,
            'session_open'  : session_open,
            'session_close' : session_close,
        }
    return results
