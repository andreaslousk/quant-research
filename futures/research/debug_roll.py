'''
Debug script for futures roll / back-adjustment logic on NG.
Run incrementally; we build up checks step by step.
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.data_loader import load_ohlcv, get_front_month_daily
from data.config import INSTRUMENT_CALENDARS

START_DATE = '2021-05-28'
END_DATE   = '2026-01-01'
INSTRUMENT = 'NG'
CALENDAR   = INSTRUMENT_CALENDARS.get(INSTRUMENT, 'NYSE')

print('=' * 70)
print(f'ROLL DEBUG — {INSTRUMENT}  {START_DATE} → {END_DATE}  (calendar={CALENDAR})')
print('=' * 70)

ohlcv = load_ohlcv(START_DATE, END_DATE, [INSTRUMENT], verbose=True)
front = get_front_month_daily(ohlcv, calendar=CALENDAR, verbose=True)

print('\n── raw data ──')
print(f'bars            : {len(ohlcv):,}')
print(f'distinct symbols: {ohlcv["symbol"].nunique()}')
print(f'date span       : {ohlcv["dt_ny"].min()} → {ohlcv["dt_ny"].max()}')
print(f'front-month days: {len(front)}  ({front["front_month"].nunique()} contracts)')


# ── Front-month series + roll detection ────────────────────────────────────────
df = ohlcv.copy()
df['date'] = df['dt_ny'].dt.date
df = df.merge(front, on='date', how='left')
n_before = len(df)
df = df[df['symbol'] == df['front_month']].sort_values('dt_ny').reset_index(drop=True)

print('\n── front-month filter ──')
print(f'bars before : {n_before:,}')
print(f'bars after  : {len(df):,}  ({len(df) / n_before:.1%} kept)')

df['prev_sym']  = df['symbol'].shift(1)
df['prev_date'] = df['date'].shift(1)
df['roll']      = df['symbol'].notna() & df['prev_sym'].notna() & (df['symbol'] != df['prev_sym'])

print(f'rolls detected: {df["roll"].sum()}')


# ── Roll factors: new/old at the last shared minute on prev_date ───────────────
c     = ohlcv[['dt_ny', 'date', 'symbol', 'close']]
rolls = df.loc[df['roll'], ['prev_date', 'prev_sym', 'symbol']].rename(
    columns={'prev_sym': 'old_sym', 'symbol': 'new_sym'})

old = (rolls.merge(c, left_on=['prev_date', 'old_sym'], right_on=['date', 'symbol'])
            [['prev_date', 'dt_ny', 'old_sym', 'close']]
            .rename(columns={'close': 'close_o'}))
new = (rolls.merge(c, left_on=['prev_date', 'new_sym'], right_on=['date', 'symbol'])
            [['prev_date', 'dt_ny', 'new_sym', 'close']]
            .rename(columns={'close': 'close_n'}))
m   = old.merge(new, on=['prev_date', 'dt_ny'])              # same minute only
g   = m.groupby('prev_date').last()
factor = g.eval('close_n / close_o')                         # last shared minute

print('\n── per-roll factors ──')
summary = g[['old_sym', 'new_sym', 'close_o', 'close_n']].assign(factor=factor)
print(summary.to_string())

print('\n── factor sanity ──')
print(f'count   : {factor.count()}')
print(f'NaN     : {factor.isna().sum()}')
print(f'min/max : {factor.min():.4f} / {factor.max():.4f}')

missing = set(rolls['prev_date']) - set(factor.index)
print(f'rolls with no shared minute: {len(missing)}  {sorted(missing)}')


# ── Apply adjustment + check returns ───────────────────────────────────────────
df['adj_factor'] = df['prev_date'].map(factor).where(df['roll']).fillna(1.0)
# reverse cumprod, but EXCLUDE each bar's own factor so the roll factor scales
# the OLD contract (bars before the roll), not the new contract's first bar.
df['cum_adj']    = df['adj_factor'][::-1].cumprod()[::-1] / df['adj_factor']

raw_close = df['close'].copy()
for col in ['open', 'high', 'low', 'close']:
    df[col] *= df['cum_adj']

print('\n── adjusted series ──')
print(f'adj close min/max : {df["close"].min():.4f} / {df["close"].max():.4f}')
print(f'any negative      : {(df["close"] <= 0).any()}')

# within-contract returns must be unchanged by adjustment (factor cancels)
non_roll = ~df['roll']
raw_ret  = (raw_close / raw_close.shift(1) - 1)[non_roll]
adj_ret  = (df['close'] / df['close'].shift(1) - 1)[non_roll]
print(f'max |Δreturn| on non-roll bars: {(raw_ret - adj_ret).abs().max():.2e}')


# ── Cumulative return of the adjusted series ───────────────────────────────────
# A ratio-back-adjusted series chains the within-contract returns, so its
# cumulative return = the realized return of holding & rolling the futures.
# For NG in persistent contango that is deeply negative (≈ UNG) — that bleed is
# REAL carry, not a bug. The naive spliced 'raw' series is the wrong one: it has
# spurious upward jumps at each roll and so looks artificially flat.
ret      = df['close'] / df['close'].shift(1) - 1
cum_adj_ret = (1 + ret.fillna(0)).prod() - 1
cum_raw_ret = (1 + (raw_close / raw_close.shift(1) - 1).fillna(0)).prod() - 1
print('\n── cumulative return ──')
print(f'adjusted : {cum_adj_ret:+.1%}')
print(f'raw      : {cum_raw_ret:+.1%}   (front-month spliced w/o adjustment)')


# ── Roll-boundary return spikes (leak detector) ────────────────────────────────
# If the splice is correct, returns on roll bars should look like normal bars.
# A fat tail here means the contract spread is leaking into returns.
roll_ret = ret[df['roll']].dropna()
print('\n── roll-bar returns ──')
print(f'count          : {roll_ret.count()}')
print(f'mean           : {roll_ret.mean():+.4%}')
print(f'min/max        : {roll_ret.min():+.2%} / {roll_ret.max():+.2%}')
print(f'|ret| > 5%     : {(roll_ret.abs() > 0.05).sum()} rolls')
print(roll_ret.to_string())


# ── Verdict ────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
# The splice is CLEAN when every roll is adjusted and roll bars carry no spread
# spike. A deeply negative adjusted cum return is EXPECTED for NG (contango carry)
# and is not itself evidence of a bug.
n_big = int((roll_ret.abs() > 0.05).sum())
if len(missing):
    print(f'⚠ {len(missing)} roll(s) had no shared minute → unadjusted, spread leaks in.')
if n_big:
    print(f'⚠ {n_big} roll bar(s) > 5% → contract spread leaking into returns.')
if not len(missing) and not n_big:
    print('✓ splice is clean: all rolls adjusted, no spread spikes on roll bars.')
    print(f'  adjusted cum return {cum_adj_ret:+.0%} is real contango carry — '
          'sanity-check it against UNG total return over the same span.')
