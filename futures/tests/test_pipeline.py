'''
test_pipeline.py
----------------
Pytest tests for data_loader.py and data_processor.py.
Uses a synthetic OHLCV fixture to avoid dependency on real data files.
'''

import datetime
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path.home() / 'quant_research' / 'futures'
sys.path.insert(0, str(ROOT_DIR))

# isort: split
import numpy as np

from data.data_loader import get_all_sessions, get_front_month_daily, load_ohlcv
from data.data_processor import (
    build_continuous_series,
    compute_bar_returns,
    get_ohlcv_resampled,
    get_session_returns,
    get_session_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_bars(symbol: str, start: str, end: str, freq: str = 'min', price: float = 100.0) -> pd.DataFrame:
    tz     = 'America/New_York'
    index  = pd.date_range(start, end, freq=freq, tz=tz)
    n      = len(index)
    prices = [price + i * 0.01 for i in range(n)]
    return pd.DataFrame({
        'dt_ny' : index,
        'dt_utc': index.tz_convert('UTC'),
        'symbol': symbol,
        'open'  : prices,
        'high'  : prices,
        'low'   : prices,
        'close' : prices,
        'volume': [100] * n,
        'date'  : [t.date() for t in index],
        'time'  : [t.time() for t in index],
        'day'   : [t.day_name() for t in index],
    })


@pytest.fixture
def ohlcv():
    '''Three trading days of minute bars for ESH3, session 09:30-16:00.'''
    day1_session   = _make_bars('ESH3', '2013-01-02 09:30', '2013-01-02 16:00')
    day1_overnight = _make_bars('ESH3', '2013-01-02 16:01', '2013-01-02 23:59')
    day2_morning   = _make_bars('ESH3', '2013-01-03 00:00', '2013-01-03 09:29')
    day2_session   = _make_bars('ESH3', '2013-01-03 09:30', '2013-01-03 16:00', price=101.0)
    day2_overnight = _make_bars('ESH3', '2013-01-03 16:01', '2013-01-03 23:59', price=102.0)
    day3_morning   = _make_bars('ESH3', '2013-01-04 00:00', '2013-01-04 09:29', price=102.0)
    day3_session   = _make_bars('ESH3', '2013-01-04 09:30', '2013-01-04 16:00', price=103.0)
    return pd.concat([
        day1_session, day1_overnight,
        day2_morning, day2_session, day2_overnight,
        day3_morning, day3_session,
    ], ignore_index=True)


@pytest.fixture
def front_months():
    return pd.DataFrame({
        'date'       : [datetime.date(2013, 1, 2), datetime.date(2013, 1, 3), datetime.date(2013, 1, 4)],
        'front_month': ['ESH3', 'ESH3', 'ESH3'],
    })


@pytest.fixture
def sessions(ohlcv, front_months):
    continuous = build_continuous_series(ohlcv, front_months)
    return get_all_sessions(continuous)


# ── _clean_ohlcv (via load_ohlcv) ────────────────────────────────────────────

def test_timezone(ohlcv):
    assert str(ohlcv['dt_ny'].dt.tz) == 'America/New_York'


def test_no_17xx_in_resampled(ohlcv, front_months):
    continuous = build_continuous_series(ohlcv, front_months)
    bars = get_ohlcv_resampled(continuous)
    assert (bars.index.hour == 17).sum() == 0


# ── get_all_sessions ──────────────────────────────────────────────────────────

def test_overnight_includes_morning_bars(sessions):
    intraday, overnight = sessions
    day2_morning_bars = overnight[
        (overnight['session_date'] == datetime.date(2013, 1, 3)) &
        (overnight['dt_ny'].dt.date == datetime.date(2013, 1, 3))
    ]
    assert len(day2_morning_bars) > 0, 'Morning bars missing from overnight session'


def test_overnight_session_date_is_next_trading_day(sessions):
    _, overnight = sessions
    evening_bars = overnight[overnight['dt_ny'].dt.date == datetime.date(2013, 1, 2)]
    assert (evening_bars['session_date'] == datetime.date(2013, 1, 3)).all()


def test_overnight_day_of_week(sessions):
    _, overnight = sessions
    session = overnight[overnight['session_date'] == datetime.date(2013, 1, 3)]
    assert (session['day_of_week'] == 'Thursday').all()


def test_intraday_session_date_equals_date(sessions):
    intraday, _ = sessions
    assert (intraday['session_date'] == intraday['date']).all()


def test_no_overlap_between_sessions(sessions):
    intraday, overnight = sessions
    overlap = set(intraday['dt_ny']) & set(overnight['dt_ny'])
    assert len(overlap) == 0


# ── get_ohlcv_resampled ────────────────────────────────────────────────

def test_prev_close_is_previous_hour(ohlcv, front_months):
    continuous = build_continuous_series(ohlcv, front_months)
    hourly     = get_ohlcv_resampled(continuous).reset_index().sort_values('dt_ny')
    for i in range(1, len(hourly)):
        assert hourly.iloc[i]['prev_close'] == hourly.iloc[i - 1]['close']


# ── get_session_returns ───────────────────────────────────────────────────────

def test_intraday_return_known_value():
    '''open=100, close=105 → intraday return = C_t/O_t - 1 = 0.05 exactly.'''
    tz   = 'America/New_York'
    date = datetime.date(2013, 1, 2)
    bars = pd.DataFrame({
        'dt_ny' : pd.to_datetime(['2013-01-02 09:30', '2013-01-02 16:00']).tz_localize(tz),
        'symbol': 'ESH3',
        'open'  : [100.0, 104.0],
        'close' : [100.0, 105.0],
        'date'  : date,
    })
    intraday_ret, _ = get_session_returns(bars)
    assert abs(intraday_ret['return'].iloc[0] - 0.05) < 1e-10


def test_overnight_return_known_value():
    '''prev close=100, next open=103 → overnight return = O_t/C_{t-1} - 1 = 0.03 exactly.'''
    tz    = 'America/New_York'
    date1 = datetime.date(2013, 1, 2)
    date2 = datetime.date(2013, 1, 3)
    bars  = pd.DataFrame({
        'dt_ny' : pd.to_datetime([
            '2013-01-02 09:30', '2013-01-02 16:00',
            '2013-01-03 09:30', '2013-01-03 16:00',
        ]).tz_localize(tz),
        'symbol': 'ESH3',
        'open'  : [100.0, 100.0, 103.0, 103.0],
        'close' : [100.0, 100.0, 103.0, 103.0],
        'date'  : [date1, date1, date2, date2],
    })
    _, overnight_ret = get_session_returns(bars)
    assert abs(overnight_ret['return'].iloc[0] - 0.03) < 1e-10


# ── get_session_summary ───────────────────────────────────────────────────────

def test_summary_has_both_sessions(sessions):
    intraday, _ = sessions
    summary, _  = get_session_summary(intraday)
    assert set(summary.index) == {'intraday', 'overnight', 'total'}


def test_summary_sharpe_is_finite(sessions):
    intraday, _ = sessions
    summary, _  = get_session_summary(intraday)
    assert summary.loc[['intraday', 'overnight'], 'sharpe'].notna().all()


def test_by_day_columns_are_ordered(sessions):
    intraday, _ = sessions
    _, by_day   = get_session_summary(intraday)
    day_order   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    assert list(by_day.columns) == [d for d in day_order if d in by_day.columns]


def test_total_cum_return_equals_first_to_last_close():
    '''
    Chain overnight and intraday returns sequentially in chronological order.
    For each date: overnight first (C_{t-1} → O_t), then intraday (O_t → C_t).
    Sequential product telescopes to C_N / O_1 — first intraday open to last close.
    '''
    ohlcv                       = load_ohlcv('2013-01-01', '2013-02-01')
    front_months                = get_front_month_daily(ohlcv)
    continuous                  = build_continuous_series(ohlcv, front_months)
    intraday, _                 = get_all_sessions(continuous)
    summary, _                  = get_session_summary(intraday)
    intraday_ret, overnight_ret = get_session_returns(intraday)

    # Build a flat series of returns sorted chronologically:
    # overnight (order=0) before intraday (order=1) for each date.
    all_rets = pd.concat([
        overnight_ret.reset_index()[['date', 'return']].assign(order=0),
        intraday_ret.reset_index()[['date', 'return']].assign(order=1),
    ]).sort_values(['date', 'order'])
    sequential_total = (1 + all_rets['return']).prod() - 1

    # 1. Sequential chain must match get_session_summary's total
    assert abs(summary.loc['total', 'cum_return'] - sequential_total) < 1e-10

    # 2. Sequential chain telescopes to last_close / first_open - 1
    first_open = intraday_ret.sort_index()['open'].iloc[0]
    last_close = intraday_ret.sort_index()['close'].iloc[-1]
    assert abs(sequential_total - (last_close / first_open - 1)) < 1e-10


# ── compute_bar_returns ────────────────────────────────────────────────────

def test_bar_return_known_value():
    '''close=101, prev_close=100 → return = 0.01 exactly.'''
    df     = pd.DataFrame({'close': [101.0], 'prev_close': [100.0]})
    result = compute_bar_returns(df)
    assert abs(result['bar_return'].iloc[0] - 0.01) < 1e-10
