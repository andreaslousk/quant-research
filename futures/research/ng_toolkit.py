'''
ng_toolkit.py
-------------
Reusable building blocks extracted from sandbox.ipynb so the same NG / commodity
research can be run from any notebook.

Everything that touched a notebook global (commodity_data, ng_ohlcv, panel,
in_window, release_dates, ...) is refactored to take explicit arguments.

Typical use
-----------
    import ng_toolkit as ng
    from data.config import START_DATE, END_DATE

    START, END = '2018-01-01', '2020-12-31'        # working window

    storage = ng.load_storage_events(START, END)   # cached; z-score over full history
    data    = ng.load_commodities(START, END)      # {inst: {ohlcv, continuous, ...}}, cached per inst

    intraday  = ng.all_intraday(data, START, END)
    panel_all = ng.build_panel(data, storage_release_dates(storage), START, END)
    panel     = panel_all.xs('NG', level='commodity')

    ng.plot_forward_curve(data['NG']['ohlcv'], '2018-11-14')

Caching: results persist to ng_toolkit's `.cache/` (the same dir sandbox.ipynb
uses), so existing pickles are reused. Pass refresh=True to rebuild.
'''

import os
import sys
import time
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Make the `data` package importable regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.config import (
    START_DATE, END_DATE, INSTRUMENT_CALENDARS, COMMODITY_SESSION_CLOSE,
    COMMODITIES, DATA_DIR,
)
from data.data_loader import load_ohlcv, load_statistics, get_all_sessions, get_front_month_daily
from data.data_processor import (
    build_continuous_series, get_session_returns, infer_session_open,
)
from data.symbols import expiry

# Defaults (override per call if needed).
WIN_1W, WIN_1M = 5, 21
TRADING_DAYS = 252


# ── Caching ───────────────────────────────────────────────────────────────────
# The cache now lives in the generic `cache` module; re-exported here so existing
# callers (ng.cached, ng.set_cache_dir) and internal uses keep working unchanged.
from cache import CACHE_DIR, set_cache_dir, cached   # noqa: E402,F401


# ── Window helper ─────────────────────────────────────────────────────────────

def window_mask(dates, start=None, end=None):
    '''Boolean mask for `dates` within [start, end] inclusive. None = open-ended.
    Works for date / Timestamp Series, Index, or arrays.'''
    idx = pd.to_datetime(pd.Index(dates))
    m = np.ones(len(idx), dtype=bool)
    if start is not None:
        m &= np.asarray(idx >= pd.Timestamp(start))
    if end is not None:
        m &= np.asarray(idx <= pd.Timestamp(end))
    return m


# ── EIA storage ───────────────────────────────────────────────────────────────

def fetch_storage_events(api_key=None):
    '''Pull weekly Lower-48 working gas in storage from the EIA v2 API.
    Returns columns: week_ending, release_ts_ny, release_date, release_time,
    storage_bcf, storage_change_bcf, storage_change_pct.'''
    api_key = api_key or os.getenv('EIA_API_KEY')
    if not api_key:
        raise RuntimeError('Set EIA_API_KEY in your environment (or use a cached result).')

    SERIES_ID    = 'NG.NW2_EPG0_SWO_R48_BCF.W'
    RELEASE_TIME = '10:30:00'   # ET — EIA publishes Thursday mornings

    res = requests.get(f'https://api.eia.gov/v2/seriesid/{SERIES_ID}',
                       params={'api_key': api_key}, timeout=30)
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

    # Report covers the week ending Friday; EIA releases it the following Thursday.
    storage['release_ts_ny'] = (
        (storage['week_ending'] + pd.Timedelta(days=6) + pd.Timedelta(RELEASE_TIME))
        .dt.tz_localize('America/New_York')
    )
    storage['release_date'] = storage['release_ts_ny'].dt.date
    storage['release_time'] = storage['release_ts_ny'].dt.time

    return storage[[
        'week_ending', 'release_ts_ny', 'release_date', 'release_time',
        'storage_bcf', 'storage_change_bcf', 'storage_change_pct',
    ]].copy()


def add_storage_zscore(storage):
    '''Add storage_change_z (z-score of the weekly change) computed over the input.'''
    storage = storage.copy()
    chg = storage['storage_change_bcf']
    storage['storage_change_z'] = (chg - chg.mean()) / chg.std()
    return storage


def load_storage_events(start=None, end=None, refresh=False, api_key=None):
    '''Cached storage events with the z-score computed over FULL history, then
    sliced to [start, end].'''
    full = cached('storage_events', lambda: fetch_storage_events(api_key), refresh=refresh)
    full = add_storage_zscore(full)
    if start is None and end is None:
        return full
    return full[window_mask(full['release_date'], start, end)].reset_index(drop=True)


def storage_release_dates(storage_events):
    '''Set of release dates, for event flags.'''
    return set(storage_events['release_date'])


# ── Weather: HDD / CDD (Open-Meteo, no API key) ───────────────────────────────

DD_BASE = 65.0   # degree-day base, °F

# Population-weighted gas-demand proxy cities (lat, lon, weight).
DEMAND_CITIES = {
    'New York':    (40.71, -74.01, 0.20), 'Chicago':      (41.85, -87.65, 0.16),
    'Boston':      (42.36, -71.06, 0.08), 'Philadelphia': (39.95, -75.17, 0.08),
    'Washington':  (38.90, -77.04, 0.07), 'Detroit':      (42.33, -83.05, 0.09),
    'Minneapolis': (44.98, -93.27, 0.08), 'Kansas City':  (39.10, -94.58, 0.06),
    'Atlanta':     (33.75, -84.39, 0.06), 'Dallas':       (32.78, -96.80, 0.06),
}


def _openmeteo_temp(lat, lon, forecast_days=None, start=None, end=None, retries=5):
    '''Daily mean temp (°F) for one point. Historical archive when forecast_days
    is None, else the next `forecast_days` of forecast. Retries with backoff to ride
    out Open-Meteo rate limits on wide / multi-city fetches.'''
    archive = forecast_days is None
    url = ('https://archive-api.open-meteo.com/v1/archive' if archive
           else 'https://api.open-meteo.com/v1/forecast')
    p = dict(latitude=lat, longitude=lon, daily='temperature_2m_max,temperature_2m_min',
             temperature_unit='fahrenheit', timezone='America/New_York')
    p.update(dict(start_date=str(start), end_date=str(end)) if archive
             else dict(forecast_days=forecast_days))
    for attempt in range(retries):
        r = requests.get(url, params=p, timeout=60)
        d = r.json().get('daily') if r.ok else None
        if d:
            tavg = (np.array(d['temperature_2m_max'], float) + np.array(d['temperature_2m_min'], float)) / 2
            return pd.Series(tavg, index=pd.to_datetime(d['time']))
        time.sleep(5 * (attempt + 1))                      # back off (rate limit / transient)
    raise RuntimeError(f'Open-Meteo fetch failed after {retries} tries (lat={lat}, lon={lon})')


def degree_days(start=None, end=None, forecast_days=None, cities=DEMAND_CITIES, base=DD_BASE):
    '''Population-weighted daily HDD & CDD (°F-days). Returns a DataFrame indexed
    by date with columns hdd, cdd. Degree days are computed per city then weighted.

        History:  degree_days('2018-01-01', '2020-12-31')
        Forecast: degree_days(forecast_days=16)
    '''
    hdd = cdd = 0.0
    for lat, lon, w in cities.values():
        t = _openmeteo_temp(lat, lon, forecast_days=forecast_days, start=start, end=end)
        hdd = hdd + np.clip(base - t, 0, None) * w
        cdd = cdd + np.clip(t - base, 0, None) * w
        time.sleep(0.5)                                    # be gentle between cities (rate limit)
    return pd.DataFrame({'hdd': hdd, 'cdd': cdd})


def _openmeteo_prev_runs_temp(lat, lon, start, end, leads=7, model='gfs_seamless', retries=5):
    '''Daily mean temp (°F) per forecast lead, for one point, from Open-Meteo's
    Previous Runs archive (GFS 2 m temp available from 2021-03-23). Returns a
    DataFrame indexed by VALID date with columns lead_1..lead_<leads>: lead_k is
    the temp that was forecast k days ahead of that valid date (i.e. issued k days
    before it). Hourly is fetched (daily aggregates don't accept the _previous_dayN
    suffix) and reduced to (max+min)/2 to match the historical convention above.'''
    url = 'https://previous-runs-api.open-meteo.com/v1/forecast'
    hourly = [f'temperature_2m_previous_day{k}' for k in range(1, leads + 1)]
    p = dict(latitude=lat, longitude=lon, hourly=','.join(hourly),
             temperature_unit='fahrenheit', timezone='America/New_York',
             models=model, start_date=str(start), end_date=str(end))
    for attempt in range(retries):
        r = requests.get(url, params=p, timeout=120)
        h = r.json().get('hourly') if r.ok else None
        if h:
            t = pd.to_datetime(h['time'])
            out = {}
            for k in range(1, leads + 1):
                s = pd.Series(np.array(h[f'temperature_2m_previous_day{k}'], float), index=t).resample('D')
                out[f'lead_{k}'] = (s.max() + s.min()) / 2
            return pd.DataFrame(out)
        time.sleep(5 * (attempt + 1))                      # back off (rate limit / transient)
    raise RuntimeError(f'Open-Meteo previous-runs fetch failed after {retries} tries (lat={lat}, lon={lon})')


def forecast_degree_days(start=None, end=None, leads=7, cities=DEMAND_CITIES,
                         base=DD_BASE, model='gfs_seamless'):
    '''Population-weighted forecast HDD & CDD SUMMED over the next `leads` days,
    keyed by ISSUE date (no look-ahead). For issue date T the value is the forecast
    degree-day total for T+1..T+leads that was knowable on T, assembled along the
    lead-time diagonal of GFS Previous Runs: forecast for valid date T+k, known at
    T, is lead_k shifted up by k. Returns a daily DataFrame indexed by issue date
    with columns hdd_fc, cdd_fc; resample to weekly to align with storage. GFS 2 m
    temp starts 2021-03-23, so request no earlier. Mirrors `degree_days` — the
    caller wraps it in `cached(f'forecast_dd_{start}_{end}', ...)`.

        forecast_degree_days('2021-03-23', '2026-06-01')
    '''
    hdd_fc = cdd_fc = 0.0
    for lat, lon, w in cities.values():
        t = _openmeteo_prev_runs_temp(lat, lon, start, end, leads=leads, model=model)
        issued = pd.DataFrame({k: t[f'lead_{k}'].shift(-k) for k in range(1, leads + 1)})  # diagonal -> issue date
        # min_count=leads -> an archive gap in any forecast day yields NaN (dropped below),
        # never a misleading 0; the caller's as-of grid then carries the last good value.
        hdd_fc = hdd_fc + (base - issued).clip(lower=0).sum(axis=1, min_count=leads) * w
        cdd_fc = cdd_fc + (issued - base).clip(lower=0).sum(axis=1, min_count=leads) * w
        time.sleep(0.5)                                    # be gentle between cities (rate limit)
    out = pd.DataFrame({'hdd_fc': hdd_fc, 'cdd_fc': cdd_fc}).dropna()  # tail lacks a full forward window
    out.index.name = 'date'
    return out


# ── CFTC positioning & EIA-930 power burn ─────────────────────────────────────

# Henry Hub natural gas, NYMEX (CFTC contract code).
COT_NG_CODE = '023651'


def fetch_cot(contract_code=COT_NG_CODE, refresh=False):
    '''CFTC Commitments of Traders — managed-money positioning for Henry Hub NG
    (disaggregated futures-only; weekly, Tuesday-dated). Cached. Returns a
    DataFrame indexed by report date with mm_long / mm_short / mm_spread, mm_net
    (= long - short), open_interest, and mm_net_pct_oi.'''
    def compute():
        cols = ('report_date_as_yyyy_mm_dd,open_interest_all,'
                'm_money_positions_long_all,m_money_positions_short_all,m_money_positions_spread')
        rows = requests.get(
            'https://publicreporting.cftc.gov/resource/72hh-3qpy.json',
            params={'cftc_contract_market_code': contract_code, '$select': cols,
                    '$order': 'report_date_as_yyyy_mm_dd ASC', '$limit': 50000},
            timeout=60).json()
        rename = {'open_interest_all': 'open_interest',
                  'm_money_positions_long_all': 'mm_long',
                  'm_money_positions_short_all': 'mm_short',
                  'm_money_positions_spread': 'mm_spread'}
        df = pd.DataFrame(rows).rename(columns=rename)
        for c in rename.values():
            df[c] = pd.to_numeric(df[c])
        df.index = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
        df.index.name = 'date'
        df['mm_net'] = df['mm_long'] - df['mm_short']
        df['mm_net_pct_oi'] = df['mm_net'] / df['open_interest'] * 100
        return df[['mm_long', 'mm_short', 'mm_spread', 'mm_net', 'open_interest', 'mm_net_pct_oi']]
    return cached('cot_ng', compute, refresh=refresh)


def fetch_power_burn(api_key=None, refresh=False):
    '''EIA-930 daily natural-gas-fired generation for the Lower-48 (US48) — a
    power-burn proxy (MWh). Cached. Returns a daily Series indexed by date;
    resample to weekly to align with storage.'''
    def compute():
        key = api_key or os.getenv('EIA_API_KEY')
        if not key:
            raise RuntimeError('Set EIA_API_KEY for the EIA-930 pull.')
        rows = requests.get(
            'https://api.eia.gov/v2/electricity/rto/daily-fuel-type-data/data/',
            params={'api_key': key, 'frequency': 'daily', 'data[0]': 'value',
                    'facets[respondent][]': 'US48', 'facets[fueltype][]': 'NG',
                    'facets[timezone][]': 'Eastern',
                    'sort[0][column]': 'period', 'sort[0][direction]': 'asc', 'length': 5000},
            timeout=60).json()['response']['data']
        s = pd.Series({pd.Timestamp(d['period']): float(d['value']) for d in rows},
                      name='power_burn_mwh').sort_index()
        s.index.name = 'date'
        return s
    return cached('power_burn', compute, refresh=refresh)


def fetch_crude_production(api_key=None, refresh=False):
    '''Weekly U.S. field production of crude oil from the EIA petroleum API
    (thousand barrels/day, series PET.WCRFPUS2.W). Cached. Returns a weekly Series
    indexed by date; aligns with storage / COT on the weekly grid. The weekly figure
    is a model-based estimate — use the monthly series (PET.MCRFPUS2.M) for accuracy.'''
    def compute():
        key = api_key or os.getenv('EIA_API_KEY')
        if not key:
            raise RuntimeError('Set EIA_API_KEY for the crude production pull.')
        rows = requests.get(
            'https://api.eia.gov/v2/seriesid/PET.WCRFPUS2.W',
            params={'api_key': key}, timeout=60).json()['response']['data']
        s = pd.Series({pd.Timestamp(d['period']): float(d['value']) for d in rows},
                      name='crude_prod_kbd').sort_index()
        s.index.name = 'date'
        return s
    return cached('crude_production', compute, refresh=refresh)


def fetch_gas_production(api_key=None, refresh=False):
    '''Monthly U.S. dry natural gas production from the EIA NG API, converted to
    Bcf/day (series NG.N9070US2.M, native units MMcf/month). Cached. Returns a
    monthly Series indexed by month-start date. Note: monthly (not weekly) and
    lagged ~2 months — the slow structural supply variable for the gas balance.'''
    def compute():
        key = api_key or os.getenv('EIA_API_KEY')
        if not key:
            raise RuntimeError('Set EIA_API_KEY for the gas production pull.')
        rows = requests.get(
            'https://api.eia.gov/v2/seriesid/NG.N9070US2.M',
            params={'api_key': key}, timeout=60).json()['response']['data']
        mmcf = pd.Series({pd.Timestamp(d['period']): float(d['value'])
                          for d in rows if d['value'] is not None}).sort_index()
        s = (mmcf / 1000 / mmcf.index.days_in_month).rename('gas_prod_bcfd')   # MMcf/mo -> Bcf/day
        s.index.name = 'date'
        return s
    return cached('gas_production', compute, refresh=refresh)


# EIA monthly natural-gas consumption by end-use sector (MMcf/month).
NG_CONSUMPTION_SERIES = {
    'residential':    'NG.N3010US2.M',
    'commercial':     'NG.N3020US2.M',
    'industrial':     'NG.N3035US2.M',
    'electric_power': 'NG.N3045US2.M',
    'total':          'NG.N9140US2.M',
}


def fetch_consumption_by_sector(api_key=None, refresh=False):
    '''Monthly U.S. natural gas consumption by end-use sector from the EIA NG API,
    converted to Bcf/day (native units MMcf/month). Cached. Returns a monthly
    DataFrame indexed by month-start date with columns residential, commercial,
    industrial, electric_power, total — the demand side of the gas balance. Monthly
    and lagged ~2 months, so it is a structural-demand variable, not a weekly signal.'''
    def compute():
        key = api_key or os.getenv('EIA_API_KEY')
        if not key:
            raise RuntimeError('Set EIA_API_KEY for the gas consumption pull.')
        out = {}
        for name, sid in NG_CONSUMPTION_SERIES.items():
            rows = requests.get(f'https://api.eia.gov/v2/seriesid/{sid}',
                                params={'api_key': key}, timeout=60).json()['response']['data']
            mmcf = pd.Series({pd.Timestamp(d['period']): float(d['value'])
                              for d in rows if d['value'] is not None}).sort_index()
            out[name] = mmcf / 1000 / mmcf.index.days_in_month          # MMcf/mo -> Bcf/day
            time.sleep(0.3)                                             # be gentle on the EIA API
        df = pd.DataFrame(out)
        df.index.name = 'date'
        return df
    return cached('gas_consumption', compute, refresh=refresh)


# ── Commodity build / load ────────────────────────────────────────────────────

def _daily_spot(ohlcv, front_months):
    '''Daily-aggregated OHLCV of the unadjusted front-month contract (the spot
    proxy). The daily close is anchored to the 14:30 ET settlement: bars after
    COMMODITY_SESSION_CLOSE are dropped, so `close` is the settlement mark (this
    also excludes the overnight session, not just the 17:00 maintenance hour).'''
    fm = ohlcv.merge(front_months, on='date', how='left')
    fm = fm[(fm['symbol'] == fm['front_month']) & (fm['dt_ny'].dt.time <= COMMODITY_SESSION_CLOSE)]
    return (fm.sort_values('dt_ny').groupby('date')
              .agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                   close=('close', 'last'), volume=('volume', 'sum')))


def build_commodity(instrument, start=START_DATE, end=END_DATE):
    '''Full per-instrument build: raw bars, back-adjusted continuous series,
    inferred session open, and `spot` — daily OHLCV of the unadjusted front month.'''
    ohlcv        = load_ohlcv(start, end, [instrument], verbose=False)
    session_open = infer_session_open(ohlcv, freq='30min')
    front_months = get_front_month_daily(ohlcv, calendar=INSTRUMENT_CALENDARS[instrument])
    continuous   = build_continuous_series(ohlcv, front_months)
    spot         = _daily_spot(ohlcv, front_months)

    return {'ohlcv': ohlcv, 'continuous': continuous,
            'session_open': session_open, 'spot': spot}


def available_commodities():
    '''Commodities from config that actually have OHLCV data on disk.'''
    return [c for c in COMMODITIES if os.path.isdir(os.path.join(DATA_DIR, c, 'futures_ohlcv'))]


def load_commodities(start=START_DATE, end=END_DATE, instruments=None, refresh=False):
    '''Build (and cache) each commodity under '<inst>_continuous'. Returns
    {inst: build dict}. Defaults to every commodity with data on disk.

    Caches that predate `spot` (the older 'front_close' schema) are migrated in
    memory: `spot` is derived from the cached raw bars without re-parsing CSVs.
    Pass refresh=True to rebuild from source and persist.'''
    instruments = instruments or available_commodities()
    out = {}
    for inst in instruments:
        d = cached(f'{inst.lower()}_continuous',
                   lambda inst=inst: build_commodity(inst, start, end), refresh=refresh)
        if 'spot' not in d:                                   # migrate legacy cache
            fms = get_front_month_daily(d['ohlcv'], calendar=INSTRUMENT_CALENDARS[inst])
            d = {k: v for k, v in d.items() if k != 'front_close'}
            d['spot'] = _daily_spot(d['ohlcv'], fms)
        out[inst] = d
    return out


# ── Intraday sessions ─────────────────────────────────────────────────────────

def commodity_intraday(comm, calendar, start=None, end=None, session_close=COMMODITY_SESSION_CLOSE):
    '''Intraday session bars for one commodity build dict, using its own inferred
    session open and the given calendar, optionally windowed.'''
    cont = comm['continuous']
    if start is not None or end is not None:
        cont = cont[window_mask(cont['date'], start, end)]
    sessions, _ = get_all_sessions(
        cont, calendar=calendar,
        session_open=comm['session_open'], session_close=session_close,
    )
    return sessions


def all_intraday(commodity_data, start=None, end=None):
    '''{inst: intraday sessions} for every commodity in commodity_data.'''
    return {inst: commodity_intraday(commodity_data[inst], INSTRUMENT_CALENDARS[inst], start, end)
            for inst in commodity_data}


# ── Release analysis (volume on / around the print) ───────────────────────────

def session_returns(intraday, release_dates):
    '''Long per-commodity session frame: date, return, volume, commodity, event_date.
    `intraday` is the {inst: sessions} dict.'''
    out = []
    for inst, sess in intraday.items():
        r, _ = get_session_returns(sess)
        r = r.reset_index()[['date', 'return', 'volume']]
        r['commodity']  = inst
        r['event_date'] = r['date'].isin(release_dates)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def release_volume(session_ret):
    '''Mean session volume on release vs non-release days, per commodity, + uplift_%.'''
    rv = (session_ret.groupby(['commodity', 'event_date'])['volume'].mean()
          .unstack('event_date').rename(columns={False: 'non_release', True: 'release'}))
    rv['uplift_%'] = (rv['release'] / rv['non_release'] - 1) * 100
    return rv


def before_after(intraday, storage_events):
    '''Per release-day, per commodity: volume / open / close / ret split before vs
    after the 10:30 print (the release bar dropped).'''
    release_time = storage_events['release_time'].iloc[0]
    out = []
    for inst, sess in intraday.items():
        s = sess.merge(storage_events[['release_date']],
                       left_on='session_date', right_on='release_date', how='inner')
        s = s[s['time'] != release_time]
        s['after_release'] = s['time'] > release_time
        g = (s.groupby(['session_date', 'after_release'])
               .agg(volume=('volume', 'sum'), open=('open', 'first'), close=('close', 'last'))
               .assign(ret=lambda d: d['close'] / d['open'] - 1)
               .reset_index())
        g['commodity'] = inst
        out.append(g)
    return pd.concat(out, ignore_index=True)


def before_after_volume(release_split):
    '''Mean volume before vs after the print, per commodity, + after_vs_before_%.'''
    bav = (release_split.groupby(['commodity', 'after_release'])['volume'].mean()
           .unstack('after_release').rename(columns={False: 'before', True: 'after'}))
    bav['after_vs_before_%'] = (bav['after'] / bav['before'] - 1) * 100
    return bav


# ── Daily panel & realized vol ────────────────────────────────────────────────

def daily_panel(continuous, release_dates=None, start=None, end=None,
                win_1w=WIN_1W, win_1m=WIN_1M, trading_days=TRADING_DAYS):
    '''Daily panel from a continuous series: close, volume, returns, rolling
    weekly/monthly vol (annualized %), their % changes, and an event flag.'''
    cont = continuous
    if start is not None or end is not None:
        cont = cont[window_mask(cont['date'], start, end)]
    ann = np.sqrt(trading_days) * 100
    d = (cont[cont['dt_ny'].dt.hour != 17]
         .groupby('date').agg(close=('close', 'last'), volume=('volume', 'sum')))
    d.index = pd.to_datetime(d.index)
    d['ret_d']          = d['close'].pct_change()
    d['ret_w']          = d['close'].pct_change(win_1w)
    d['vol_w']          = d['ret_d'].rolling(win_1w).std() * ann
    d['vol_m']          = d['ret_d'].rolling(win_1m).std() * ann
    d['vol_w_chg']      = d['vol_w'].pct_change()
    d['vol_m_chg']      = d['vol_m'].pct_change()
    d['prev_vol_m_chg'] = d['vol_m_chg'].shift(1)
    d['volume_chg']     = d['volume'].pct_change()
    if release_dates is not None:
        d['event_date'] = d.index.to_series().dt.date.isin(release_dates)
    return d


def build_panel(commodity_data, release_dates=None, start=None, end=None, dropna=True, **kw):
    '''Daily panel for every commodity, long, indexed by (commodity, date).'''
    pa = pd.concat(
        {inst: daily_panel(commodity_data[inst]['continuous'], release_dates, start, end, **kw)
         for inst in commodity_data},
        names=['commodity', 'date'],
    )
    return pa.dropna() if dropna else pa


def merge_panel_storage(panel_all, storage_events):
    '''Inner-join the daily panel with storage on release dates (one row per
    commodity per release day).'''
    return (panel_all.reset_index()
            .merge(storage_events.assign(date=pd.to_datetime(storage_events['release_date'])),
                   on='date', how='inner')
            .drop(columns='release_date')
            .set_index(['commodity', 'date'])
            .sort_index())


def realized_vol(continuous, start=None, end=None,
                 win_1w=WIN_1W, win_1m=WIN_1M, trading_days=TRADING_DAYS):
    '''Rolling realized vol from daily LOG returns (annualized %): vol_1w, vol_1m
    and their % changes. (Log returns, matching the §6 vol plot.)'''
    cont = continuous
    if start is not None or end is not None:
        cont = cont[window_mask(cont['date'], start, end)]
    dc = cont[cont['dt_ny'].dt.hour != 17].groupby('date')['close'].last()
    dc.index = pd.to_datetime(dc.index)
    r = np.log(dc).diff()
    ann = np.sqrt(trading_days) * 100
    vol = pd.DataFrame({'vol_1w': r.rolling(win_1w).std() * ann,
                        'vol_1m': r.rolling(win_1m).std() * ann})
    vol['vol_1w_chg'] = vol['vol_1w'].pct_change()
    vol['vol_1m_chg'] = vol['vol_1m'].pct_change()
    return vol


# ── Forward curve ─────────────────────────────────────────────────────────────



def forward_snapshot(ohlcv, date, max_expiries=None):
    '''All outright contracts on `date` (from a raw ohlcv frame): start (first bar
    open) vs end (last bar close) price, by expiry. volume > 0 only.'''
    date = pd.Timestamp(date).date()
    day = ohlcv[ohlcv['date'] == date]
    if day.empty:
        raise ValueError(f'No data for {date}')
    snap = (day.sort_values('dt_ny').groupby('symbol')
              .agg(open_px=('open', 'first'), close_px=('close', 'last'), volume=('volume', 'sum'))
              .reset_index())
    snap['expiry'] = snap['symbol'].map(lambda s: expiry(s, date.year))
    snap = snap.dropna(subset=['expiry']).query('volume > 0').sort_values('expiry')
    return snap.head(max_expiries) if max_expiries else snap


def curve_change(snap):
    '''Within-day (start->end) curve move reduced to scalars (percent):
    level (parallel shift), slope (twist), magnitude (RMS), front-back.'''
    mo  = snap['expiry'].dt.year * 12 + snap['expiry'].dt.month
    mo  = (mo - mo.min()).to_numpy()
    chg = (snap['close_px'] / snap['open_px'] - 1) * 100
    return pd.Series({
        'level_%'           : chg.mean(),
        'slope_%_per_mo'    : np.polyfit(mo, chg, 1)[0],
        'magnitude_%'       : np.sqrt((chg**2).mean()),
        'front_minus_back_%': chg.iloc[0] - chg.iloc[-1],
    })


def curve_move(prev_snap, snap):
    '''Day-over-day curve move (prev close -> today close, matched per contract)
    reduced to the same four scalars.'''
    m = (prev_snap[['symbol', 'close_px', 'expiry']]
         .merge(snap[['symbol', 'close_px']], on='symbol', suffixes=('_prev', '_now'))
         .sort_values('expiry'))
    mo  = m['expiry'].dt.year * 12 + m['expiry'].dt.month
    mo  = (mo - mo.min()).to_numpy()
    chg = (m['close_px_now'] / m['close_px_prev'] - 1) * 100
    return pd.Series({
        'level_%'           : chg.mean(),
        'slope_%_per_mo'    : np.polyfit(mo, chg, 1)[0],
        'magnitude_%'       : np.sqrt((chg**2).mean()),
        'front_minus_back_%': chg.iloc[0] - chg.iloc[-1],
    })


def curve_change_series(ohlcv, dates):
    '''Run curve_change on each date in `dates`; rows that error/lack >=2 contracts
    are dropped. Returns a DataFrame indexed by date.'''
    rows = {}
    for d in dates:
        try:
            snap = forward_snapshot(ohlcv, d)
            if len(snap) >= 2:
                rows[d] = curve_change(snap)
        except ValueError:
            continue
    return pd.DataFrame(rows).T.dropna(how='all')


# ── Plots ─────────────────────────────────────────────────────────────────────

def _annotate_metrics(ax, cc):
    metrics = [('level', f"{cc['level_%']:+.2f}%"),
               ('slope', f"{cc['slope_%_per_mo']:+.3f}%/mo"),
               ('magnitude', f"{cc['magnitude_%']:.2f}%"),
               ('front-back', f"{cc['front_minus_back_%']:+.2f}%")]
    ax.text(0.98, 0.97, '\n'.join(f'{l:<10} {v:>9}' for l, v in metrics),
            transform=ax.transAxes, ha='right', va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


def plot_forward_curve(ohlcv, date, max_expiries=18, ax=None):
    '''Superimpose a forward curve start vs end of `date`, annotated with curve_change.'''
    date = pd.Timestamp(date).date()
    snap = forward_snapshot(ohlcv, date, max_expiries)
    cc = curve_change(snap)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6), dpi=120)
    ax.plot(snap['expiry'], snap['open_px'],  marker='o', lw=1.5, label=f'{date} — start of day')
    ax.plot(snap['expiry'], snap['close_px'], marker='o', lw=1.5, label=f'{date} — end of day')
    ax.set_title(f'Forward curve — {date} (start vs end of day)')
    ax.set_xlabel('Contract expiry'); ax.set_ylabel('Price')
    _annotate_metrics(ax, cc)
    ax.legend(loc='upper left'); ax.grid(alpha=0.3)
    plt.tight_layout()
    return snap


def plot_day_over_day(panel, ohlcv, date, ax=None, max_expiries=12):
    '''Prev-day close curve vs signal-day close curve, using panel.index for the
    prior trading day.'''
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6), dpi=120)
    days = panel.index
    i = days.get_loc(pd.Timestamp(date))
    if i == 0:
        raise ValueError('no prior day')
    prev, today = days[i - 1], days[i]
    ps = forward_snapshot(ohlcv, prev, max_expiries)
    ts = forward_snapshot(ohlcv, today, max_expiries)
    ax.plot(ps['expiry'], ps['close_px'], color='tab:blue', alpha=0.8, lw=1.3, marker='o', ms=3,
            label=f'{prev.date()} — prev day')
    ax.plot(ts['expiry'], ts['close_px'], color='tab:red', lw=2.0, marker='o', ms=4,
            label=f'{today.date()} — signal day')
    ax.set_xlabel('Contract expiry'); ax.set_ylabel('Price')
    ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.3)
    return ps, ts


def plot_signal_grid(panel, ohlcv, signal='vol_m_chg', top_k=10, lag=1, max_expiries=12):
    '''Grid of prev-day -> signal-day curve moves for the top_k days by a LEADING
    signal (panel[signal] lagged by `lag`).'''
    sig = panel[signal].shift(lag)
    top_days = sig.nlargest(top_k).index
    ncols = 2
    nrows = int(np.ceil(top_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5.2 * nrows), dpi=100)
    axes = np.atleast_1d(axes).ravel()
    for ax, d in zip(axes, top_days):
        try:
            plot_day_over_day(panel, ohlcv, d, ax=ax, max_expiries=max_expiries)
            ax.set_title(f'{d.date()}  —  {signal}(t-{lag}) {sig.loc[d]:.2f}  |  '
                         f'ret {panel.loc[d, "ret_d"] * 100:+.1f}%')
        except (ValueError, KeyError):
            ax.set_visible(False)
    for ax in axes[len(top_days):]:
        ax.set_visible(False)
    fig.suptitle(f'Curve move (prev -> signal day) — top {top_k} by {signal} (t-{lag})',
                 y=1.002, fontsize=14)
    plt.tight_layout()
    return top_days


def plot_realized_vol(vol, storage_events=None, win_1w=WIN_1W, win_1m=WIN_1M, ax=None):
    '''1-week vs 1-month realized vol, with optional storage-surprise vlines.'''
    if ax is None:
        _, ax = plt.subplots(figsize=(22, 8), dpi=120)
    ax.plot(vol.index, vol['vol_1w'], lw=0.8, alpha=0.7, color='tab:orange',
            label=f'{win_1w}d (1-week) realized vol')
    ax.plot(vol.index, vol['vol_1m'], lw=1.6, color='tab:blue',
            label=f'{win_1m}d (1-month) realized vol')
    handles, _ = ax.get_legend_handles_labels()
    if storage_events is not None:
        absz = storage_events['storage_change_z'].abs()
        for d in pd.to_datetime(storage_events.loc[(absz > 1) & (absz <= 2), 'release_date']):
            ax.axvline(d, color='red', lw=0.6, alpha=0.20)
        for d in pd.to_datetime(storage_events.loc[absz > 2, 'release_date']):
            ax.axvline(d, color='red', lw=1.3, alpha=0.75)
        handles += [Line2D([0], [0], color='red', lw=0.6, alpha=0.5, label='1 < |z| <= 2 surprise'),
                    Line2D([0], [0], color='red', lw=1.3, alpha=0.75, label='|z| > 2 surprise')]
    ax.set_title('Realized volatility — 1-week vs 1-month rolling (annualized)')
    ax.set_xlabel('Date'); ax.set_ylabel('Annualized vol (%)')
    ax.legend(handles=handles); ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax


def plot_vol_all(panel_all, col='vol_m', ax=None):
    '''Overlay one rolling-vol line per commodity (col unstacked from panel_all).'''
    wide = panel_all[col].unstack('commodity')
    if ax is None:
        _, ax = plt.subplots(figsize=(22, 8), dpi=120)
    for inst in wide.columns:
        ax.plot(wide.index, wide[inst], lw=1.2, label=inst)
    ax.set_title(f'{col} — all commodities')
    ax.set_xlabel('Date'); ax.set_ylabel('Annualized vol (%)')
    ax.legend(title='Commodity', ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax


def plot_price_vs_storage(spot, storage_events, start=None, end=None, ax=None):
    '''Unadjusted front-month close (left) vs working gas in storage (right).
    `spot` is the daily OHLCV frame from the build (its 'close' is used); a bare
    close Series is also accepted.'''
    close = spot['close'] if hasattr(spot, 'columns') else spot
    fc = close[window_mask(close.index, start, end)].copy()
    fc.index = pd.to_datetime(fc.index)
    if ax is None:
        _, ax1 = plt.subplots(figsize=(22, 8), dpi=120)
    else:
        ax1 = ax
    ax1.plot(fc.index, fc.values, color='tab:blue', lw=1.2, label='Front-month price (unadjusted)')
    ax1.set_xlabel('Date'); ax1.set_ylabel('Front-month price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(storage_events['week_ending'], storage_events['storage_bcf'],
             color='tab:green', lw=1.4, label='Working gas in storage (Bcf)')
    ax2.set_ylabel('Working gas in storage (Bcf)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax1.set_title('Front-month price (unadjusted) vs working gas in storage')
    ax1.grid(alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')
    plt.tight_layout()
    return ax1


def plot_lines(data, title=None, ylabel=None, normalize=False, figsize=(20, 7), ax=None):
    '''Plot every column of `data` (Series or DataFrame) on ONE shared axis. Each line
    is labelled by its column / Series name (rename to change the legend). No second
    axis. normalize=True z-scores each column so different magnitudes overlay.'''
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=120)
    df = data.to_frame() if isinstance(data, pd.Series) else data
    for label, s in df.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        if normalize and s.std():
            s = (s - s.mean()) / s.std()
        ax.plot(s.index, s.values, lw=1.3, label=label)
    ax.set_xlabel('Date')
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return ax


def plot_dual(left, right, title=None, ylabel=None, ylabel_right=None, figsize=(20, 7), ax=None):
    '''Exactly one Series on the left axis and one on the right (dashed) axis — two
    different scales. Axis and legend labels default to each Series' name.'''
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=120)
    ax2 = ax.twinx()
    for s, target, color, ls in [(left, ax, 'C0', '-'), (right, ax2, 'C1', '--')]:
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        target.plot(s.index, s.values, lw=1.3, color=color, linestyle=ls, label=s.name)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel or left.name)
    ax2.set_ylabel(ylabel_right or right.name)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='best')
    plt.tight_layout()
    return ax


# ── Post-release reaction signal (storage print -> settlement) ─────────────────

def post_release_return(continuous, release_dates, start='10:30', end='14:30'):
    '''Front-month % return over [start, end] on each storage-release date — the
    market's reaction to the print (EIA storage prints 10:30 ET; window ends at the
    14:30 settlement). Uses the continuous series (intraday roll-adjustment is
    constant within a day, so this is the true front-month reaction). Returns a
    Series indexed by release date.'''
    rel = set(pd.to_datetime(pd.Index(sorted(set(release_dates)))).date)
    c = continuous[['dt_ny', 'open', 'close']].copy()
    c['d'] = c['dt_ny'].dt.date
    c = c[c['d'].isin(rel)]
    tt = c['dt_ny'].dt.time
    s_t, e_t = pd.Timestamp(start).time(), pd.Timestamp(end).time()
    win = c[(tt >= s_t) & (tt <= e_t)].sort_values('dt_ny')
    o  = win.groupby('d')['open'].first()      # price at the 10:30 release
    cl = win.groupby('d')['close'].last()      # price at the 14:30 settlement
    r = (cl / o - 1.0).rename('post_release_ret')
    r.index = pd.to_datetime(r.index)
    return r.sort_index()


def _reaction_z(r, z_window=104):
    '''Trailing z-score of a per-Thursday reaction series (rolling `z_window` weeks,
    shifted 1 so the z uses only PRIOR reactions -> no look-ahead).'''
    mu = r.rolling(z_window, min_periods=20).mean().shift(1)
    sd = r.rolling(z_window, min_periods=20).std().shift(1)
    return (r - mu) / sd


def post_release_signal(continuous, release_dates, start='10:30', end='14:30', z_window=104):
    '''Front-month post-release reaction return + its trailing z-score. Returns a
    DataFrame indexed by release date: post_release_ret, post_release_z.'''
    r = post_release_return(continuous, release_dates, start, end)
    return pd.DataFrame({'post_release_ret': r, 'post_release_z': _reaction_z(r, z_window)})


def spread_post_release_change(ohlcv, release_dates, near_month=3, far_month=4,
                               start='10:30', end='14:30'):
    '''Dollar change in the PROMPT calendar spread (near_month − far_month) over
    [start, end] on each release date — the spread's own reaction to the print. A $
    change (not %), since a calendar spread can cross zero. The prompt pairing matches
    the widowmaker: nearest near-month not-yet-expired, paired with the same-year far
    leg. Returns a Series indexed by release date.'''
    rel = set(pd.to_datetime(pd.Index(sorted(set(release_dates)))).date)
    o = ohlcv[['dt_ny', 'symbol', 'open', 'close']].copy()
    o['d'] = o['dt_ny'].dt.date
    o = o[o['d'].isin(rel)]
    tt = o['dt_ny'].dt.time
    s_t, e_t = pd.Timestamp(start).time(), pd.Timestamp(end).time()
    win = o[(tt >= s_t) & (tt <= e_t)].sort_values('dt_ny')
    g = win.groupby(['d', 'symbol'])
    px = pd.DataFrame({'p_start': g['open'].first(), 'p_end': g['close'].last()}).reset_index()
    px['d'] = pd.to_datetime(px['d'])
    px['expiry'] = [expiry(s, dd.year) for s, dd in zip(px['symbol'], px['d'])]
    px = px.dropna(subset=['expiry'])
    px['m'], px['yr'] = px['expiry'].dt.month, px['expiry'].dt.year
    near = (px[px['m'] == near_month][['d', 'yr', 'p_start', 'p_end']]
            .rename(columns={'p_start': 'n_s', 'p_end': 'n_e'}))
    far  = (px[px['m'] == far_month][['d', 'yr', 'p_start', 'p_end']]
            .rename(columns={'p_start': 'f_s', 'p_end': 'f_e'}))
    sp = near.merge(far, on=['d', 'yr'])
    sp['exp'] = pd.to_datetime(dict(year=sp['yr'], month=near_month, day=1))
    sp = sp[sp['exp'] >= sp['d']].sort_values(['d', 'exp']).groupby('d').first()   # prompt pair
    r = ((sp['n_e'] - sp['f_e']) - (sp['n_s'] - sp['f_s'])).rename('spread_post_release_chg')
    r.index = pd.to_datetime(r.index)
    return r.sort_index()


def spread_post_release_signal(ohlcv, release_dates, near_month=3, far_month=4,
                               start='10:30', end='14:30', z_window=104):
    '''Prompt calendar-spread post-release $ change + its trailing z-score. The z
    column is named `post_release_z` so it drops into the same weekly grid / regression
    as the front-month version. Returns: spread_post_release_chg, post_release_z.'''
    r = spread_post_release_change(ohlcv, release_dates, near_month, far_month, start, end)
    return pd.DataFrame({'spread_post_release_chg': r, 'post_release_z': _reaction_z(r, z_window)})
