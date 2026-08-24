'''
options.py
----------
Generic CME option-chain tooling, instrument-agnostic so it works for any chain
(LNE = Henry Hub NG, ES, NQ, ...). Parameterized by the option `root`, its data
dir (<DATA_DIR>/<root>/options_ohlcv), and the strike scaling.

Pipeline (LNE = Henry Hub NG shown; pass the chain's own knobs)
--------------------------------------------------------------
    load_options('LNE')                                       # 1-min OHLCV -> daily close per option
    option_iv_panel(options, contracts, energy_option_expiry) # Black-76 IV vs each option's own-month future
    option_iv_vwap('LNE', fut_ohlcv, energy_option_expiry)    # IV-space VWAP daily mark (per-minute IV, vol-weighted)
    daily_atm_iv(vwap_panel)                                  # daily ATM-forward IV per contract
    constant_maturity_iv(atm_iv, energy_option_expiry, 30)    # roll-free 30d ATM IV (variance-interpolated)
    iv_diff(atm_iv, grid, near, far)                          # prompt calendar IV-spread on a weekly grid

Greeks (chain-wide, vectorized: one call handles a full day's array of strikes/expiries)
    black76_delta(F, K, T, sigma, is_call, r)                 # closed-form, in this module already
    black76_greeks(F, K, T, sigma, is_call, r)                # gamma/vega, European, via py_vollib_vectorized
    american_iv(calc_date, expiry, F, K, r, premium, is_call) # American IV, solved vs. real premium (QuantLib BAW)
    american_greeks(calc_date, expiry, F, K, r, sigma, is_call) # American gamma/vega, bump-and-reprice (BAW)

Per-chain knobs (nothing about a product is assumed implicitly)
    root          option root + data dir <DATA_DIR>/<root>/options_ohlcv (e.g. 'LNE', 'ES')
    strike_scale  symbol strike encoding -> $ (NG: 3550 -> 3.55, scale 1000)
    expiry_fn     REQUIRED expiry rule: energy_option_expiry (NG/CL) | third_friday (ES/NQ) | your own
    r             discount rate for Black-76 (default 0.0)

Assumptions: Databento GLBX (`glbx-mdp3-*` CSV or DBN, int64 prices x PRICE_SCALE),
monthly serial option symbols `<root><month><year> <C|P><strike>`, US session date
(NY tz), and Black-76 with EUROPEAN exercise (exact for LNE/cash-settled, only
approximate for American chains like standard ES) -- `american_iv`/`american_greeks`
solve directly under American exercise instead, for chains where that matters.

`expiry` keys a contract by the 1st of its expiry month (matching the `expiry()`
symbol helper) — the join key between an option and its own-month future. These are
pure transforms (no internal caching); the caller wraps the expensive ones
(`load_options`, `option_iv_vwap`) in its own cache, e.g. ng_toolkit.cached.
'''

from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.stats import norm

# Vectorizes py_vollib's analytical BSM greeks (numba-accelerated, array-aware).
# Only the closed-form gamma/vega path is used (not get_all_greeks / implied_volatility,
# whose numba paths don't compile against current numba/py_lets_be_rational).
import py_vollib_vectorized  # noqa: F401  (side effect: patches py_vollib in place)
from py_vollib.black_scholes_merton.greeks.analytical import gamma as _bsm_gamma
from py_vollib.black_scholes_merton.greeks.analytical import vega as _bsm_vega

from data.config import DATA_DIR, PRICE_SCALE, STAT_TYPE_SETTLEMENT, STAT_TYPE_OPEN_INTEREST
from data.symbols import expiry, MONTH_CODE, MONTH_ABBR, resolve_year, prompt_year
from data.readers import read_records


def _option_dir(root):
    return Path(DATA_DIR) / root / 'options_ohlcv'


def _option_symbol_re(root):
    '''Regex for a CME option symbol: <root><monthcode><year> <C|P><strike>,
    e.g. "LNEH4 C3550" -> (H, 4, C, 3550).'''
    return rf'^{root}([FGHJKMNQUVXZ])(\d{{1,2}})\s+([CP])(\d+)$'


# ── Per-chain option-expiry rules (pass one as `expiry_fn`) ───────────────────
# The expiry rule is product-specific, so it is never assumed implicitly — the
# caller picks the rule for its chain. Two common ones are provided; add others as
# needed. `exp_month` is the 1st of the contract's expiry month.

def energy_option_expiry(exp_month, n=4):
    '''Energy convention: option expiry ≈ `n` business days before the 1st of the
    expiry month (NG/CL: future last-trade ~3 bdays prior, option 1 bday before that
    -> n=4). Holiday-agnostic. Use functools.partial(energy_option_expiry, n=...)
    for a different offset.'''
    return pd.Timestamp(np.busday_offset(np.datetime64(pd.Timestamp(exp_month).date()),
                                         -n, roll='backward'))


def third_friday(exp_month):
    '''Equity-index convention: the 3rd Friday of the expiry month (ES/NQ quarterly
    options). Holiday-agnostic.'''
    first = pd.Timestamp(pd.Timestamp(exp_month).year, pd.Timestamp(exp_month).month, 1)
    first_friday = first + pd.Timedelta(days=(4 - first.dayofweek) % 7)   # Friday = weekday 4
    return first_friday + pd.Timedelta(weeks=2)


def load_options(root, strike_scale=1000.0):
    '''Load every 1-min OHLCV CSV for option `root`, reduce to the DAILY close per
    option instrument, and parse the symbol into contract fields. Returns a
    DataFrame: date, expiry (1st of the expiry month — the join key to the underlying
    future), cp ('C'/'P'), strike, premium ($, the daily close), volume. These are
    TRADED bars, so an instrument only appears on days it printed.'''
    rx = _option_symbol_re(root)
    frames = []
    for f in sorted(_option_dir(root).glob('glbx-mdp3-*')):
        df = read_records(f, usecols=['ts_event', 'close', 'volume', 'symbol'])
        parts = df['symbol'].str.extract(rx)                    # code, digits, cp, strike
        df = df[parts[0].notna()].copy()
        if df.empty:
            continue
        df[['code', 'digits', 'cp', 'strike']] = parts.loc[df.index]
        df['date'] = (pd.to_datetime(df['ts_event'], utc=True)
                      .dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None))
        frames.append(df[['date', 'code', 'digits', 'cp', 'strike', 'close', 'volume']])
    raw = pd.concat(frames, ignore_index=True).sort_values('date')
    key = ['date', 'code', 'digits', 'cp', 'strike']            # daily close per instrument
    daily = raw.groupby(key, as_index=False).agg(close=('close', 'last'), volume=('volume', 'sum'))
    month = daily['code'].map(MONTH_CODE)
    year  = resolve_year(daily['digits'], daily['date'].dt.year.to_numpy())
    daily['expiry']  = pd.to_datetime(dict(year=year, month=month, day=1))
    daily['strike']  = daily['strike'].astype(float) / strike_scale
    daily['premium'] = daily['close'].astype(float) * PRICE_SCALE
    return daily[['date', 'expiry', 'cp', 'strike', 'premium', 'volume']]


def load_option_statistics(root, strike_scale=1000.0):
    '''Daily option SETTLEMENT (stat_type=3, from `price`) + open interest
    (stat_type=9, from `quantity`) per option instrument, from the full-chain
    <root>/options_statistics CSVs — every listed strike, not just traded ones.
    Dated by `ts_ref` (the settlement date). Returns one row per (date, expiry, cp,
    strike): date, expiry, cp, strike, premium ($, the settlement), open_interest.
    Drop-in for `option_iv_panel` (its `premium` is the settlement price).'''
    rx = _option_symbol_re(root)
    sdir = Path(DATA_DIR) / root / 'options_statistics'
    keep = (STAT_TYPE_SETTLEMENT, STAT_TYPE_OPEN_INTEREST)
    frames = []
    for f in sorted(sdir.glob('glbx-mdp3-*')):
        df = read_records(f, usecols=['ts_ref', 'price', 'quantity', 'stat_type', 'symbol'])
        df = df[df['stat_type'].isin(keep) & (df['ts_ref'] < 8_000_000_000_000_000_000)]
        parts = df['symbol'].str.extract(rx)                    # options only (drops spreads/combos)
        df = df[parts[0].notna()].copy()
        if df.empty:
            continue
        df[['code', 'digits', 'cp', 'strike']] = parts.loc[df.index]
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    ts = pd.to_datetime(raw['ts_ref'], unit='ns', utc=True)   # ts_ref is 00:00 UTC of the settlement DATE — keep UTC
    raw['date']   = ts.dt.date                                # (NY-converting would roll it back a day)
    year          = resolve_year(raw['digits'], ts.dt.year.to_numpy())
    raw['expiry'] = pd.to_datetime(dict(year=year, month=raw['code'].map(MONTH_CODE), day=1))
    raw['strike'] = raw['strike'].astype(float) / strike_scale
    key = ['date', 'expiry', 'cp', 'strike']

    settle = raw[(raw['stat_type'] == STAT_TYPE_SETTLEMENT) & (raw['price'] > 0)].copy()
    settle['premium'] = settle['price'] * PRICE_SCALE
    settle = settle.drop_duplicates(key, keep='last')[key + ['premium']]

    oi = raw[raw['stat_type'] == STAT_TYPE_OPEN_INTEREST].copy()
    oi['open_interest'] = oi['quantity']
    oi = oi.drop_duplicates(key, keep='last')[key + ['open_interest']]

    return settle.merge(oi, on=key, how='left').reset_index(drop=True)


def load_option_quotes(root, snap, strike_scale=1000.0):
    '''Bid (stat_type=8) and ask (stat_type=7) per option on a single snapshot day,
    from <root>/options_statistics — the last quote of the day per instrument. A
    diagnostic companion to load_option_statistics for plotting the settlement smile
    against the market. Bid/ask are dated by ts_event (a real intraday quote time, so
    NY-converted, unlike the midnight-UTC settlement ts_ref). Returns date, expiry,
    cp, strike, bid, ask.'''
    snap = pd.Timestamp(snap)
    rx = _option_symbol_re(root)
    sdir = Path(DATA_DIR) / root / 'options_statistics'
    target = snap.strftime('%Y%m%d')
    files = [f for f in sorted(sdir.glob('glbx-mdp3-*'))
             if f.name.split('-')[2] <= target <= f.name.split('-')[3][:8]]
    frames = []
    for f in files:
        df = read_records(f, usecols=['ts_event', 'price', 'stat_type', 'symbol'])
        df = df[df['stat_type'].isin((7, 8)) & (df['price'] > 0) & (df['price'] < 8_000_000_000_000_000_000)]
        parts = df['symbol'].str.extract(rx)
        df = df[parts[0].notna()].copy()
        if df.empty:
            continue
        df[['code', 'digits', 'cp', 'strike']] = parts.loc[df.index]
        df['dt'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True).dt.tz_convert('America/New_York')
        frames.append(df[df['dt'].dt.date == snap.date()])
    raw = pd.concat(frames, ignore_index=True).sort_values('ts_event')
    year = resolve_year(raw['digits'], raw['dt'].dt.year.to_numpy())
    raw['expiry'] = pd.to_datetime(dict(year=year, month=raw['code'].map(MONTH_CODE), day=1))
    raw['strike'] = raw['strike'].astype(float) / strike_scale
    raw['px']     = raw['price'] * PRICE_SCALE
    key = ['expiry', 'cp', 'strike']
    bid = raw[raw['stat_type'] == 8].drop_duplicates(key, keep='last')[key + ['px']].rename(columns={'px': 'bid'})
    ask = raw[raw['stat_type'] == 7].drop_duplicates(key, keep='last')[key + ['px']].rename(columns={'px': 'ask'})
    out = bid.merge(ask, on=key, how='inner')
    out['date'] = snap.date()
    return out[['date', 'expiry', 'cp', 'strike', 'bid', 'ask']]


def _read_option_minute(root, strike_scale=1000.0):
    '''Parse every 1-min option bar for `root` (NOT reduced to daily). Returns
    minute-level rows: minute (NY, tz-naive), date, expiry, cp, strike, price, volume.'''
    rx = _option_symbol_re(root)
    frames = []
    for f in sorted(_option_dir(root).glob('glbx-mdp3-*')):
        df = read_records(f, usecols=['ts_event', 'close', 'volume', 'symbol'])
        parts = df['symbol'].str.extract(rx)
        df = df[parts[0].notna()].copy()
        if df.empty:
            continue
        df[['code', 'digits', 'cp', 'strike']] = parts.loc[df.index]
        df['minute'] = (pd.to_datetime(df['ts_event'], utc=True)
                        .dt.tz_convert('America/New_York').dt.tz_localize(None).dt.floor('min'))
        frames.append(df[['minute', 'code', 'digits', 'cp', 'strike', 'close', 'volume']])
    raw = pd.concat(frames, ignore_index=True)
    month = raw['code'].map(MONTH_CODE)
    year  = resolve_year(raw['digits'], raw['minute'].dt.year.to_numpy())
    raw['expiry'] = pd.to_datetime(dict(year=year, month=month, day=1))
    raw['date']   = raw['minute'].dt.normalize()
    raw['strike'] = raw['strike'].astype(float) / strike_scale
    raw['price']  = raw['close'].astype(float) * PRICE_SCALE
    return raw[['minute', 'date', 'expiry', 'cp', 'strike', 'price', 'volume']]


def black76_iv(price, F, K, T, r, is_call, n_iter=60):
    '''Vectorized Black-76 implied vol via clipped Newton (exact for European
    exercise). NaN where the price is outside no-arb bounds or doesn't converge.'''
    price, F, K, T = (np.asarray(x, float) for x in (price, F, K, T))
    is_call = np.asarray(is_call, bool)
    disc, sqrtT = np.exp(-r * T), np.sqrt(T)
    intrinsic = disc * np.where(is_call, np.clip(F - K, 0, None), np.clip(K - F, 0, None))
    upper     = disc * np.where(is_call, F, K)                       # loose no-arb cap
    valid = (np.isfinite(price) & (T > 0) & (F > 0) & (K > 0)
             & (price > intrinsic + 1e-9) & (price < upper))
    sigma = np.full(F.shape, 0.5)
    for _ in range(n_iter):
        with np.errstate(all='ignore'):
            d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
            d2 = d1 - sigma * sqrtT
            model = disc * np.where(is_call, F * norm.cdf(d1) - K * norm.cdf(d2),
                                             K * norm.cdf(-d2) - F * norm.cdf(-d1))
            vega = disc * F * norm.pdf(d1) * sqrtT
            step = np.where(vega > 1e-12, (model - price) / vega, 0.0)
            sigma = np.clip(sigma - np.clip(step, -1.0, 1.0), 1e-4, 5.0)
    return np.where(valid, sigma, np.nan)


def black76_delta(F, K, T, sigma, is_call, r=0.0):
    '''Black-76 delta wrt the forward (futures option): call = e^{-rT} N(d1),
    put = -e^{-rT} N(-d1). With r=0 (futures-style) it's just N(d1) / -N(-d1).
    Vectorized; NaN where sigma/T are not positive.'''
    F, K, T, sigma = (np.asarray(x, float) for x in (F, K, T, sigma))
    is_call = np.asarray(is_call, bool)
    with np.errstate(all='ignore'):
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        delta = np.exp(-r * T) * np.where(is_call, norm.cdf(d1), -norm.cdf(-d1))
    return np.where((sigma > 0) & (T > 0) & np.isfinite(delta), delta, np.nan)


def black76_greeks(F, K, T, sigma, is_call, r=0.0):
    '''Vectorized Black-76 gamma and vega, chain-wide (one call per array of strikes/
    expiries/dates). Computed via py_vollib_vectorized's closed-form BSM gamma/vega
    with S=F, q=r: setting the dividend yield equal to the discount rate cancels the
    process's drift (r-q=0), so its forward stays exactly F for ANY r -- this is
    Black-76 gamma/vega exactly, not just at r=0. (A q=0 substitution, as used ad hoc
    in earlier notebook work, only coincides with this when r=0.)
    Returns (gamma, vega); vega is per vol-point (py_vollib convention: dPrice/dsigma
    / 100), matching black76_delta's scale.'''
    F, K, T, sigma = (np.asarray(x, float) for x in (F, K, T, sigma))
    is_call = np.asarray(is_call, bool)
    r = np.broadcast_to(np.asarray(r, float), F.shape)
    flags = np.where(is_call, 'c', 'p')
    gamma = np.asarray(_bsm_gamma(flags, F, K, T, r, sigma, r), dtype=float)
    vega  = np.asarray(_bsm_vega(flags, F, K, T, r, sigma, r), dtype=float)
    valid = (sigma > 0) & (T > 0) & np.isfinite(gamma) & np.isfinite(vega)
    return np.where(valid, gamma, np.nan), np.where(valid, vega, np.nan)


def _american_price(calc_date, maturity, F, K, r, sigma, is_call):
    '''One American (BAW) price via QuantLib, S=F/q=r (see black76_greeks) so the
    process's forward stays F -- makes a standard equity-style American engine
    consistent with Black-76 on a future. Not vectorized (QuantLib has no numpy
    interface); american_iv/american_greeks loop this per row.'''
    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()
    spot = ql.QuoteHandle(ql.SimpleQuote(float(F)))
    rf   = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, float(r), dc))
    div  = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, float(r), dc))
    vol  = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calc_date, cal, float(sigma), dc))
    process = ql.BlackScholesMertonProcess(spot, div, rf, vol)
    payoff  = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, float(K))
    option  = ql.VanillaOption(payoff, ql.AmericanExercise(calc_date, maturity))
    option.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(process))
    return option.NPV()


def _to_ql_date(d):
    d = pd.Timestamp(d)
    return ql.Date(d.day, d.month, d.year)


def american_iv(calc_date, expiry, F, K, r, premium, is_call, lo=1e-4, hi=5.0):
    '''Vectorized (chain-wide) American implied vol via QuantLib's Barone-Adesi-Whaley
    engine, solved directly against each option's real market `premium` -- NOT by
    reusing a European-solved IV, which is biased whenever r != 0 (see black76_greeks).
    `calc_date` is a single evaluation date (scalar): QuantLib's evaluation date is
    process-global, so one call = one date's chain. `expiry`, `F`, `K`, `r`, `premium`,
    `is_call` are arrays, one row per option -- `expiry` may vary per row, so a single
    call can cover every expiry in that date's chain at once. r may vary per row too
    (e.g. a term-matched curve interpolated per option's own T).
    Returns an array of solved sigma, NaN where price <= intrinsic or the root isn't
    bracketed. Not internally vectorized (QuantLib is row-at-a-time); benchmarked at
    roughly 2,000 rows/sec on this chain -- reuse across (date, expiry) groups if a
    much larger batch needs to be faster.'''
    calc_date = _to_ql_date(calc_date)
    ql.Settings.instance().evaluationDate = calc_date
    expiry, F, K, r, premium = (np.asarray(x) for x in (expiry, F, K, r, premium))
    is_call = np.asarray(is_call, bool)
    r = np.broadcast_to(np.asarray(r, float), F.shape)

    sigma = np.full(F.shape, np.nan)
    for i in range(len(F)):
        maturity = _to_ql_date(expiry[i])
        intrinsic = max(F[i] - K[i], 0.0) if is_call[i] else max(K[i] - F[i], 0.0)
        if not (premium[i] > intrinsic + 1e-10):
            continue
        f = lambda s: _american_price(calc_date, maturity, F[i], K[i], r[i], s, is_call[i]) - premium[i]
        if f(lo) * f(hi) > 0:
            continue
        sigma[i] = ql.Brent().solve(f, 1e-8, 0.2, lo, hi)
    return sigma


def american_greeks(calc_date, expiry, F, K, r, sigma, is_call, h_F=0.01, h_sigma=1e-4):
    '''Vectorized (chain-wide) American gamma/vega via bump-and-reprice on the BAW
    price (no closed form exists for American greeks). Same calling convention as
    american_iv (one evaluation date per call, arrays for the rest, expiry may vary
    per row). Bump sizes were checked for stability across several orders of
    magnitude; vega is per vol-point (dPrice/dsigma / 100), matching black76_greeks.
    NaN rows in `sigma` (unsolved IV) pass through as NaN gamma/vega.'''
    calc_date = _to_ql_date(calc_date)
    ql.Settings.instance().evaluationDate = calc_date
    expiry = np.asarray(expiry)
    F, K, sigma = (np.asarray(x, float) for x in (F, K, sigma))
    is_call = np.asarray(is_call, bool)
    r = np.broadcast_to(np.asarray(r, float), F.shape)

    gamma = np.full(F.shape, np.nan)
    vega  = np.full(F.shape, np.nan)
    for i in range(len(F)):
        if not np.isfinite(sigma[i]):
            continue
        maturity = _to_ql_date(expiry[i])
        p_up   = _american_price(calc_date, maturity, F[i] + h_F, K[i], r[i], sigma[i], is_call[i])
        p_mid  = _american_price(calc_date, maturity, F[i],       K[i], r[i], sigma[i], is_call[i])
        p_down = _american_price(calc_date, maturity, F[i] - h_F, K[i], r[i], sigma[i], is_call[i])
        gamma[i] = (p_up - 2 * p_mid + p_down) / h_F ** 2
        v_up   = _american_price(calc_date, maturity, F[i], K[i], r[i], sigma[i] + h_sigma, is_call[i])
        v_down = _american_price(calc_date, maturity, F[i], K[i], r[i], sigma[i] - h_sigma, is_call[i])
        vega[i] = (v_up - v_down) / (2 * h_sigma) / 100.0
    return gamma, vega


def imply_forward(options, date_col='date', expiry_col='expiry'):
    '''Back out the forward F (and discount factor DF) per (date, expiry) from the
    option chain via put-call parity: C(K) - P(K) = DF*(F - K). Regressing C - P on
    strike K over the chain gives slope = -DF and intercept = DF*F, so DF = -slope
    and F = -intercept/slope. Uses every matched call/put strike (needs >=2 strikes
    with strike variance). DF ~ 1 for futures-style options. Returns date, expiry,
    F, DF. This makes the call and put IVs coincide (parity-exact) with no rate or
    underlying-future input.'''
    c = options[options['cp'] == 'C'][[date_col, expiry_col, 'strike', 'premium']].rename(columns={'premium': 'C'})
    p = options[options['cp'] == 'P'][[date_col, expiry_col, 'strike', 'premium']].rename(columns={'premium': 'P'})
    m = c.merge(p, on=[date_col, expiry_col, 'strike'])
    m['cp_diff'] = m['C'] - m['P']
    m['k_cp']    = m['strike'] * m['cp_diff']
    m['kk']      = m['strike'] ** 2
    g = m.groupby([date_col, expiry_col], as_index=False).agg(   # closed-form OLS of cp_diff on strike
        n=('strike', 'size'), sk=('strike', 'sum'), sd=('cp_diff', 'sum'),
        skd=('k_cp', 'sum'), skk=('kk', 'sum'))
    denom = g['n'] * g['skk'] - g['sk'] ** 2
    slope = (g['n'] * g['skd'] - g['sk'] * g['sd']) / denom
    intercept = (g['sd'] - slope * g['sk']) / g['n']
    g['DF'] = -slope
    g['F']  = -intercept / slope
    g = g[(denom != 0) & np.isfinite(g['F'])]
    return g.rename(columns={date_col: 'date', expiry_col: 'expiry'})[['date', 'expiry', 'F', 'DF']]


def option_iv_panel(options, contracts, expiry_fn, expiry_col='expiry',
                    price_col='close', date_col='date', r=0.0, start=None, end=None):
    '''Black-76 implied vol for every daily option close.

    The forward F per (date, expiry) comes from `contracts` — a daily per-contract
    future frame with an expiry-month key (`expiry_col`), a close (`price_col`) and a
    `date_col` — OR, if `contracts is None`, is implied from the chain itself
    (`imply_forward`, parity-exact so call/put IVs coincide). T is from `expiry_fn`
    (the chain's expiry rule, e.g. `energy_option_expiry`). `options` is
    `load_options(...)`/`load_option_statistics(...)`. Returns tidy rows: date,
    expiry, cp, strike, F, moneyness (=log(F/K)), T, premium, iv, delta
    (Black-76, signed: +N(d1) calls, -N(-d1) puts), + volume/OI.'''
    opt = options.copy()
    opt['date'] = pd.to_datetime(opt['date'])
    if contracts is None:                                   # back F out of the chain (put-call parity)
        fut = imply_forward(opt)
    else:
        fut = (contracts[[date_col, expiry_col, price_col]]
               .rename(columns={date_col: 'date', expiry_col: 'expiry', price_col: 'F'}))
        fut['date'], fut['expiry'] = pd.to_datetime(fut['date']), pd.to_datetime(fut['expiry'])
        fut = fut.dropna(subset=['expiry']).drop_duplicates(['date', 'expiry'])

    m = opt.merge(fut[['date', 'expiry', 'F']], on=['date', 'expiry'], how='inner')
    exp_map = {e: expiry_fn(e) for e in m['expiry'].unique()}
    m['T'] = (m['expiry'].map(exp_map) - m['date']).dt.days / 365.0
    m = m[m['T'] > 0].copy()

    m['iv'] = black76_iv(m['premium'].to_numpy(), m['F'].to_numpy(), m['strike'].to_numpy(),
                         m['T'].to_numpy(), r, (m['cp'] == 'C').to_numpy())
    m['delta'] = black76_delta(m['F'].to_numpy(), m['strike'].to_numpy(), m['T'].to_numpy(),
                               m['iv'].to_numpy(), (m['cp'] == 'C').to_numpy(), r)
    m['moneyness'] = np.log(m['F'] / m['strike'])
    if start is not None or end is not None:
        idx = pd.to_datetime(m['date'])
        keep = np.ones(len(m), dtype=bool)
        if start is not None:
            keep &= np.asarray(idx >= pd.Timestamp(start))
        if end is not None:
            keep &= np.asarray(idx <= pd.Timestamp(end))
        m = m[keep]
    liq = [c for c in ('volume', 'open_interest') if c in m.columns]   # trade data -> volume; settlement -> OI
    return (m[['date', 'expiry', 'cp', 'strike', 'F', 'moneyness', 'T', 'premium', 'iv', 'delta'] + liq]
            .sort_values(['expiry', 'strike', 'date']))


def iv_snapshot(iv_panel, snap, lookback=5):
    '''As-of IV surface on `snap`: the most recent IV for each (expiry, cp, strike)
    that printed within the prior `lookback` business days. The lookback fills the
    trade-only sparsity so a single-day surface isn't full of holes.'''
    snap = pd.Timestamp(snap)
    lo = snap - pd.tseries.offsets.BDay(lookback)
    w = iv_panel[(iv_panel['date'] <= snap) & (iv_panel['date'] >= lo) & iv_panel['iv'].notna()]
    return w.sort_values('date').groupby(['expiry', 'cp', 'strike'], as_index=False).last()


def atm_term_structure(snapshot):
    '''ATM-forward IV per expiry: at the strike nearest the forward, average the
    call and put IV. Returns a frame indexed by expiry with atm_iv, F, n_legs.'''
    s = snapshot.assign(absm=snapshot['moneyness'].abs())
    near = s.sort_values('absm').groupby(['expiry', 'cp'], as_index=False).first()   # nearest-ATF per side
    ts = near.groupby('expiry').agg(atm_iv=('iv', 'mean'), F=('F', 'first'),
                                    strike=('strike', 'mean'), n_legs=('iv', 'size'))
    return ts.sort_index()


def iv_smile(snapshot, contract_expiry):
    '''IV vs strike for one expiry on the snapshot (the smile/skew). Returns a frame
    indexed by strike with columns C / P (whichever traded), plus the forward F for
    that expiry as a plain `F` column.'''
    sm = snapshot[snapshot['expiry'] == pd.Timestamp(contract_expiry)]
    out = sm.pivot_table(index='strike', columns='cp', values='iv').sort_index()
    out['F'] = sm['F'].iloc[0] if len(sm) else np.nan
    return out


# ── IV-space VWAP (synthetic daily mark while waiting for settlements) ─────────

def _future_minute_price(fut_ohlcv):
    '''Per-minute future close keyed by (minute, expiry month), 17:00 dropped.
    expiry = 1st of the contract's expiry month (the option join key).'''
    x = fut_ohlcv[fut_ohlcv['dt_ny'].dt.hour != 17][['dt_ny', 'symbol', 'close', 'volume']].copy()
    x['minute'] = x['dt_ny'].dt.floor('min')
    if x['minute'].dt.tz is not None:
        x['minute'] = x['minute'].dt.tz_localize(None)
    x['year'] = x['minute'].dt.year
    key = x[['symbol', 'year']].drop_duplicates()
    key['expiry'] = [expiry(s, y) for s, y in zip(key['symbol'], key['year'])]
    x = x.merge(key, on=['symbol', 'year']).dropna(subset=['expiry'])
    return x.groupby(['minute', 'expiry'], as_index=False).agg(F=('close', 'last'),
                                                               fvol=('volume', 'sum'))


def option_iv_vwap(root, fut_ohlcv, expiry_fn, strike_scale=1000.0, r=0.0):
    '''Volume-weighted daily implied vol per option, inverted in IV space: each
    1-min option bar is priced against the SAME-minute future (so price and F are
    aligned — no intraday-F bias), Black-76-inverted, then the per-minute IVs are
    volume-weighted to a daily number per (date, expiry, cp, strike). Also attaches
    F_day (daily VWAP of the future) and moneyness=log(F_day/K). `expiry_fn` is the
    chain's expiry rule (e.g. energy_option_expiry, third_friday).'''
    opt = _read_option_minute(root, strike_scale)
    fut = _future_minute_price(fut_ohlcv)
    m = opt.merge(fut[['minute', 'expiry', 'F']], on=['minute', 'expiry'], how='inner')
    exp_map = {e: expiry_fn(e) for e in m['expiry'].unique()}
    m['T'] = (m['expiry'].map(exp_map) - m['date']).dt.days / 365.0
    m = m[m['T'] > 0]
    m['iv'] = black76_iv(m['price'].to_numpy(), m['F'].to_numpy(), m['strike'].to_numpy(),
                         m['T'].to_numpy(), r, (m['cp'] == 'C').to_numpy())
    m = m.dropna(subset=['iv'])
    m['ivw'] = m['iv'] * m['volume']
    agg = (m.groupby(['date', 'expiry', 'cp', 'strike'], as_index=False)
             .agg(ivw=('ivw', 'sum'), volume=('volume', 'sum')))
    agg['vwap_iv'] = agg['ivw'] / agg['volume']

    fd = fut.assign(date=fut['minute'].dt.normalize(), Fw=fut['F'] * fut['fvol'])
    fd = fd.groupby(['date', 'expiry'], as_index=False).agg(Fw=('Fw', 'sum'), fvol=('fvol', 'sum'))
    fd['F_day'] = fd['Fw'] / fd['fvol']
    agg = agg.merge(fd[['date', 'expiry', 'F_day']], on=['date', 'expiry'], how='left')
    agg['moneyness'] = np.log(agg['F_day'] / agg['strike'])
    return agg[['date', 'expiry', 'cp', 'strike', 'vwap_iv', 'volume', 'F_day', 'moneyness']]


def daily_atm_iv(panel, iv_col='vwap_iv', f_col='F_day'):
    '''Daily ATM-forward IV per contract: at the strike nearest the forward, average
    the call and put IV. Works on the VWAP panel (defaults) or an option_iv_panel
    (pass iv_col='iv', f_col='F'). Returns long rows: date, expiry, atm_iv, F, n_legs.'''
    s = panel.dropna(subset=[iv_col, 'moneyness']).assign(absm=lambda d: d['moneyness'].abs())
    near = s.sort_values('absm').groupby(['date', 'expiry', 'cp'], as_index=False).first()  # nearest-ATF per side
    atm = (near.groupby(['date', 'expiry'])
               .agg(atm_iv=(iv_col, 'mean'), F=(f_col, 'first'),
                    strike=('strike', 'mean'), n_legs=(iv_col, 'size'))
               .reset_index())
    return atm.sort_values(['expiry', 'date']).reset_index(drop=True)


def constant_maturity_iv(atm_iv, expiry_fn, tenor_days=30, iv_col='atm_iv'):
    '''Constant-maturity ATM implied vol: ONE number per date at a fixed horizon,
    so the series isn't sawtoothed by the contract roll (a raw front-month ATM IV
    jumps every expiry as T resets). For each date the two expiries bracketing the
    tenor are interpolated in TOTAL VARIANCE (iv^2 * T), not in vol — variance is
    what's additive in time. `atm_iv` is `daily_atm_iv(...)`; `expiry_fn` is the
    chain's expiry rule (the same one passed to `option_iv_panel`). Never
    extrapolates: NaN on dates where the tenor isn't bracketed on both sides.
    Returns a Series indexed by date, named iv_<tenor_days>d.'''
    target = tenor_days / 365.0
    a = atm_iv.dropna(subset=[iv_col]).copy()
    a['date'] = pd.to_datetime(a['date'])
    exp_map = {e: expiry_fn(e) for e in a['expiry'].unique()}
    a['T'] = (a['expiry'].map(exp_map) - a['date']).dt.days / 365.0
    a = a[a['T'] > 0].copy()
    a['Tm'] = a['T']                                    # survives the merge_asof on 'T'
    a['w']  = a[iv_col] ** 2 * a['T']                   # total variance
    a = a.sort_values('T')

    g = pd.DataFrame({'date': np.sort(a['date'].unique())}).assign(T=target)
    cols = ['date', 'T', 'Tm', 'w']
    lo = pd.merge_asof(g, a[cols], on='T', by='date', direction='backward')   # nearest expiry at/below
    hi = pd.merge_asof(g, a[cols], on='T', by='date', direction='forward')    # nearest expiry at/above

    T0, w0 = lo['Tm'].to_numpy(), lo['w'].to_numpy()
    T1, w1 = hi['Tm'].to_numpy(), hi['w'].to_numpy()
    span = T1 - T0
    frac = np.divide(target - T0, span, out=np.zeros_like(span), where=span > 0)  # exact hit -> 0
    with np.errstate(all='ignore'):
        iv = np.sqrt((w0 + frac * (w1 - w0)) / target)
    return pd.Series(iv, index=pd.DatetimeIndex(g['date']), name=f'iv_{tenor_days}d').dropna()


# ── ATM-IV term structure & calendar IV-diffs on a weekly grid ─────────────────

def rolling_atm_iv(atm_iv, grid, month, name=None):
    '''As-of ATM-forward IV of the prompt contract of calendar `month`, sampled on
    each grid date (roll-safe, no look-ahead) — one term-structure point. `atm_iv`
    is `daily_atm_iv(...)`; `grid` is the weekly Thursday index. Returns a Series.'''
    leg = (atm_iv[atm_iv['expiry'].dt.month == month]
           .assign(yr=lambda x: x['expiry'].dt.year)[['date', 'yr', 'atm_iv']]
           .dropna().sort_values('date'))
    leg['date'] = leg['date'].astype('datetime64[ns]')
    g = pd.DataFrame({'date': pd.DatetimeIndex(grid).as_unit('ns')}).sort_values('date')
    g['yr'] = prompt_year(g['date'], month)
    out = pd.merge_asof(g, leg, on='date', by='yr', direction='backward')
    return out.set_index('date')['atm_iv'].rename(name or f'iv_{MONTH_ABBR[month]}')


def iv_term_structure(atm_iv, grid, months, months_filter=None):
    '''As-of prompt-contract ATM IV for several `months` on the weekly grid — the IV
    term structure. months=[3, 4] for the H/J pair, [1,2,3,4,...] for a curve.
    months_filter (e.g. [11,12,1,2]) keeps only those calendar months of the grid.'''
    df = pd.concat([rolling_atm_iv(atm_iv, grid, m) for m in months], axis=1)
    return df if months_filter is None else df[df.index.month.isin(months_filter)]


def iv_diff(atm_iv, grid, near_month, far_month, months_filter=None):
    '''Prompt-contract ATM-IV calendar spread (near_month − far_month) on the weekly
    grid, plus its roll-safe week-over-week change (iv_diff_chg resets when the near
    leg rolls). months_filter restricts the grid to e.g. winter [11,12,1,2].

        iv_diff(atm_iv, grid, 3, 4, months_filter=[11, 12, 1, 2])   # March − April, winter
    '''
    near = rolling_atm_iv(atm_iv, grid, near_month)
    far  = rolling_atm_iv(atm_iv, grid, far_month)
    df = pd.concat([near, far], axis=1)
    df['iv_diff'] = df.iloc[:, 0] - df.iloc[:, 1]
    df['contract_yr'] = prompt_year(df.index, near_month)            # roll-safe grouping = near leg's contract year
    if months_filter is not None:
        df = df[df.index.month.isin(months_filter)]
    df['iv_diff_chg'] = df.groupby('contract_yr')['iv_diff'].diff()
    return df.drop(columns='contract_yr')
