'''
return_analysis.py
------------------
Cross-instrument return and volume analysis.

Sections:
  1. Data loading — one pass per instrument
  2. Cumulative return, avg return, std by instrument
  3. Return and volume correlation across instruments
  4. Overnight → next intraday return correlation by instrument
  5. Extreme move dates via Z-score (MAD), saved to CSV
'''

import sys
from pathlib import Path

ROOT_DIR = Path.home() / 'quant_research' / 'futures'
sys.path.insert(0, str(ROOT_DIR))

import argparse
import numpy as np
import pandas as pd

from data.config import START_DATE, END_DATE, INSTRUMENTS, FX_INSTRUMENTS
from data.data_loader import load_ohlcv, get_front_month_daily, get_all_sessions
from data.data_processor import (
    build_continuous_series,
    get_ohlcv_resampled,
    compute_bar_returns,
    get_profile_volume,
    get_session_returns,
)

parser = argparse.ArgumentParser(description='Cross-instrument return and volume analysis.')
parser.add_argument(
    '--instruments', nargs='+', default=None,
    metavar='INSTRUMENT',
    help='Instruments to analyse (e.g. ES NQ). Defaults to all.',
)
parser.add_argument(
    '--save', action='store_true', default=False,
    help='Save extreme moves to CSV.',
)
args = parser.parse_args()

ALL_INSTRUMENTS = args.instruments if args.instruments else INSTRUMENTS + FX_INSTRUMENTS
SAVE_RESULTS    = args.save

# ── 1. Load data ───────────────────────────────────────────────────────────────

returns      = {}   # instrument -> bar_return Series (dt_ny index)
volumes      = {}   # instrument -> volume Series (dt_ny index)
session_rets = {}   # instrument -> (intraday_ret, overnight_ret)
session_volumes = {}   # instrument -> (intraday_vol Series, overnight_vol Series) indexed by date

for instrument in ALL_INSTRUMENTS:
    print(f'Loading {instrument}...')
    ohlcv        = load_ohlcv(START_DATE, END_DATE, [instrument])
    front_months = get_front_month_daily(ohlcv)
    continuous   = build_continuous_series(ohlcv, front_months)

    bars = get_ohlcv_resampled(continuous, freq='30min')
    bars = compute_bar_returns(bars)
    vol  = get_profile_volume(continuous, freq='30min').set_index('dt_ny')

    returns[instrument] = bars['bar_return']
    volumes[instrument] = vol['volume']

    intraday, overnight = get_all_sessions(continuous)
    session_rets[instrument] = get_session_returns(intraday)
    session_volumes[instrument] = (
        intraday.groupby('date')['volume'].sum(),
        overnight.groupby('session_date')['volume'].sum(),
    )

returns_df = pd.concat(returns, axis=1).sort_index()
volume_df  = pd.concat(volumes, axis=1).sort_index()

# ── 2. Intraday and overnight return summary ───────────────────────────────────

print('\n=== Intraday Session Return Summary ===')
intraday_rows = []
for instrument in ALL_INSTRUMENTS:
    intraday_ret, _ = session_rets[instrument]
    r = intraday_ret['return'].dropna()
    intraday_rows.append({
        'instrument' : instrument,
        'avg_return' : r.mean(),
        'std_return' : r.std(),
        'cum_return' : (1 + r).prod() - 1,
        'max_return' : r.max(),
        'min_return' : r.min(),
        'n_days'     : len(r),
    })

intraday_summary = pd.DataFrame(intraday_rows).set_index('instrument')
print(intraday_summary.to_string(float_format='{:.4%}'.format))

print('\n=== Overnight Session Return Summary ===')
overnight_rows = []
for instrument in ALL_INSTRUMENTS:
    _, overnight_ret = session_rets[instrument]
    r = overnight_ret['return'].dropna()
    overnight_rows.append({
        'instrument' : instrument,
        'avg_return' : r.mean(),
        'std_return' : r.std(),
        'cum_return' : (1 + r).prod() - 1,
        'max_return' : r.max(),
        'min_return' : r.min(),
        'n_days'     : len(r),
    })

overnight_summary = pd.DataFrame(overnight_rows).set_index('instrument')
print(overnight_summary.to_string(float_format='{:.4%}'.format))

# ── 3. Cross-instrument correlations ──────────────────────────────────────────

print('\n=== Return Correlation Across Instruments ===')
return_corr = returns_df.corr()
print(return_corr.to_string(float_format='{:.3f}'.format))

print('\n=== Volume Correlation Across Instruments ===')
volume_corr = volume_df.corr()
print(volume_corr.to_string(float_format='{:.3f}'.format))

# ── 4. Overnight → next intraday return correlation ────────────────────────────

print('\n=== Overnight → Next Intraday Correlation (per instrument) ===')
on_vs_id_rows = []
for instrument in ALL_INSTRUMENTS:
    intraday_ret, overnight_ret = session_rets[instrument]
    merged = overnight_ret[['return']].rename(columns={'return': 'overnight'}).join(
        intraday_ret[['return']].rename(columns={'return': 'intraday'}),
        how='inner',
    )
    corr = merged['overnight'].corr(merged['intraday'])
    on_vs_id_rows.append({'instrument': instrument, 'corr_overnight_to_next_intraday': corr})

on_vs_id_df = pd.DataFrame(on_vs_id_rows).set_index('instrument')
print(on_vs_id_df.to_string(float_format='{:.4f}'.format))

# ── 5. Extreme move dates via MAD Z-score ─────────────────────────────────────

Z_THRESHOLD = 3.0
OUTPUT_CSV  = ROOT_DIR / 'research' / 'extreme_moves.csv'


def mad_zscore(s: pd.Series) -> pd.Series:
    median = s.median()
    mad    = (s - median).abs().median()
    return (s - median) / (mad * 1.4826) if mad > 0 else pd.Series(np.nan, index=s.index)


all_extremes = []

for instrument in ALL_INSTRUMENTS:
    intraday_ret, overnight_ret       = session_rets[instrument]
    intraday_volume, overnight_volume = session_volumes[instrument]

    for session, ret_df, volume_s in [
        ('intraday',  intraday_ret,  intraday_volume),
        ('overnight', overnight_ret, overnight_volume),
    ]:
        r              = ret_df['return'].dropna()
        r_z            = mad_zscore(r)
        v              = volume_s.reindex(r.index).dropna()
        v_z            = mad_zscore(v)
        extreme        = r_z[r_z.abs() >= Z_THRESHOLD]
        for date, z in extreme.items():
            all_extremes.append({
                'instrument'    : instrument,
                'session'       : session,
                'date'          : date,
                'return'        : r[date],
                'z_score'       : z,
                'volume_z_score': v_z.get(date, float('nan')),
            })

extremes_df = pd.DataFrame(all_extremes).sort_values(['instrument', 'date', 'session'])
if SAVE_RESULTS:
    extremes_df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nExtreme moves saved to {OUTPUT_CSV} ({len(extremes_df)} rows)')

print(f'\n=== Top 10 Extreme Dates by Session (|Z| >= {Z_THRESHOLD}) ===')
for instrument in ALL_INSTRUMENTS:
    inst_df         = extremes_df[extremes_df['instrument'] == instrument]
    intraday_dates  = set(inst_df[inst_df['session'] == 'intraday']['date'])
    overnight_dates = set(inst_df[inst_df['session'] == 'overnight']['date'])

    only_intraday  = intraday_dates - overnight_dates
    only_overnight = overnight_dates - intraday_dates
    overlap_dates  = intraday_dates & overnight_dates

    print(f'\n── {instrument} ──')

    def fmt(rows, cols):
        r = rows[cols].copy()
        r['return'] = (r['return'] * 100).map('{:.2f}%'.format)
        r['z_score'] = r['z_score'].map('{:.4f}'.format)
        r['volume_z_score'] = r['volume_z_score'].map('{:.4f}'.format)
        return r.sort_values('date').to_string(index=False)

    print('  Non-overlapping intraday (top 10 by |z_score|, sorted chronologically):')
    rows = inst_df[(inst_df['session'] == 'intraday') & (inst_df['date'].isin(only_intraday))]
    print(fmt(rows.nlargest(10, 'z_score'), ['date', 'return', 'z_score', 'volume_z_score']))

    print('  Non-overlapping overnight (top 10 by |z_score|, sorted chronologically):')
    rows = inst_df[(inst_df['session'] == 'overnight') & (inst_df['date'].isin(only_overnight))]
    print(fmt(rows.nlargest(10, 'z_score'), ['date', 'return', 'z_score', 'volume_z_score']))

    print(f'  Overlapping extreme dates: {len(overlap_dates)}')
    if overlap_dates:
        id_extreme   = inst_df[(inst_df['session'] == 'intraday')  & (inst_df['date'].isin(overlap_dates))].set_index('date')['return']
        on_extreme   = inst_df[(inst_df['session'] == 'overnight') & (inst_df['date'].isin(overlap_dates))].set_index('date')['return']
        overlap_corr = on_extreme.corr(id_extreme)
        print(f'  Correlation (overnight vs intraday) on overlapping dates: {overlap_corr:.4f}')

        print('  Overlapping dates (top 10 by combined |z_score|, sorted chronologically):')
        rows      = inst_df[inst_df['date'].isin(overlap_dates)].copy()
        top_dates = rows.groupby('date')['z_score'].apply(lambda x: x.abs().sum()).nlargest(10).index
        rows      = rows[rows['date'].isin(top_dates)].sort_values(['date', 'session'])
        print(fmt(rows, ['date', 'session', 'return', 'z_score', 'volume_z_score']))

# ── 6. Returns with extreme dates filtered out ────────────────────────────────

print(f'\n=== Session Returns Excluding Extreme Dates (|Z| >= {Z_THRESHOLD}) ===')

def session_stats(r: pd.Series) -> dict:
    return {
        'avg_return': r.mean(),
        'std_return': r.std(),
        'cum_return': (1 + r).prod() - 1,
        'n_days'    : len(r),
    }

def fmt_stats(df: pd.DataFrame) -> str:
    d = df.copy()
    for col in ['avg_return', 'std_return', 'cum_return']:
        d[col] = (d[col] * 100).map('{:.4f}%'.format)
    return d.to_string()

for instrument in ALL_INSTRUMENTS:
    intraday_ret, overnight_ret = session_rets[instrument]
    inst_extreme_dates = set(extremes_df[extremes_df['instrument'] == instrument]['date'])

    filtered_rows = []
    for session, ret_df in [('intraday', intraday_ret), ('overnight', overnight_ret)]:
        r_all      = ret_df['return'].dropna()
        r_filtered = r_all[~r_all.index.isin(inst_extreme_dates)]
        row_all    = {'session': session, 'universe': 'all',      **session_stats(r_all)}
        row_filt   = {'session': session, 'universe': 'filtered', **session_stats(r_filtered)}
        filtered_rows.extend([row_all, row_filt])

    print(f'\n── {instrument} ── ({len(inst_extreme_dates)} extreme dates removed)')
    print(fmt_stats(pd.DataFrame(filtered_rows).set_index(['session', 'universe'])))
