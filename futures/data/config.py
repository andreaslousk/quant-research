"""
config.py
---------
Central configuration for the futures data pipeline.
All constants, paths, and parameters are defined here.
Every other module imports from this file — nothing is hardcoded elsewhere.
"""

import os
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2013-01-01"
END_DATE   = "2026-01-01"

# ── Instruments ───────────────────────────────────────────────────────────────
# Root symbols to process. Used to filter contracts from mixed-instrument files.
# ES lives in its own directory. NQ, RTY, YM share a directory.
EQUITIES   = ["ES", "NQ", "YM", "RTY"]
FX         = ["6E"]
COMMODITIES = ["CL", "GC", "HG", "NG", "ZC", "ZS", "ZW"]

# ── Data paths ────────────────────────────────────────────────────────────────
# Each instrument lives at: DATA_DIR/<symbol>/futures_ohlcv (and futures_statistics)

# ── Price scaling ─────────────────────────────────────────────────────────────
# Databento stores prices as fixed-point int64 with 1e-9 scale
PRICE_SCALE = 1e-9

# ── Timezone ──────────────────────────────────────────────────────────────────
# All session boundaries are defined in New York time
TZ = "America/New_York"

# ── Session boundaries ────────────────────────────────────────────────────────
# CME regular session for equity index futures
SESSION_OPEN  = pd.Timestamp("09:30").time()
SESSION_CLOSE = pd.Timestamp("16:00").time()

# Commodity daily close anchored to the 14:30 ET energy settlement (NG/CL/HO),
# not the end of the Globex session. Daily-close sampling keeps bars up to this
# time (see _daily_spot / the notebook's _daily_contracts).
COMMODITY_SESSION_CLOSE = pd.Timestamp("14:30").time()

# ── Market calendars ──────────────────────────────────────────────────────────
# Maps each instrument to its pandas_market_calendars calendar name.
# Used to correctly classify trading vs non-trading days per instrument.
INSTRUMENT_CALENDARS = {
    'ES':  'NYSE',
    'NQ':  'NYSE',
    'YM':  'NYSE',
    'RTY': 'NYSE',
    '6E':  'CME_FX',
    'CL':  'CL',
    'GC':  'GC',
    'HG':  'HG',
    'NG':  'NG',
    'ZC':  'CME_Agriculture',
    'ZS':  'CME_Agriculture',
    'ZW':  'CME_Agriculture',
}

# ── Statistics ────────────────────────────────────────────────────────────────
# stat_type=3 is the official CME settlement price (value in `price`);
# stat_type=9 is open interest (value in `quantity`).
# See: https://databento.com/docs/schemas-and-data-formats/statistics
STAT_TYPE_SETTLEMENT    = 3
STAT_TYPE_OPEN_INTEREST = 9
