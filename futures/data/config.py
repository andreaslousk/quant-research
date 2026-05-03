"""
config.py
---------
Central configuration for the futures data pipeline.
All constants, paths, and parameters are defined here.
Every other module imports from this file — nothing is hardcoded elsewhere.
"""

import os
import pandas as pd

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2013-01-01"
END_DATE   = "2026-01-01"

# ── Instruments ───────────────────────────────────────────────────────────────
# Root symbols to process. Used to filter contracts from mixed-instrument files.
# ES lives in its own directory. NQ, RTY, YM share a directory.
INSTRUMENTS    = ["ES", "NQ", "YM", "RTY"]
FX_INSTRUMENTS = ["6E"]

# ── Data paths ────────────────────────────────────────────────────────────────
# ES has its own folder. All other equity index futures share one folder.
ES_OHLCV_DIR    = os.path.join(_DATA_DIR, "ES", "futures_ohlcv")
ES_STATS_DIR    = os.path.join(_DATA_DIR, "ES", "futures_statistics")
OTHER_OHLCV_DIR = os.path.join(_DATA_DIR, "NQ_RTY_YM", "futures_ohlcv")
OTHER_STATS_DIR = os.path.join(_DATA_DIR, "NQ_RTY_YM", "futures_statistics")
FX_OHLCV_DIRS   = {
    '6E': os.path.join(_DATA_DIR, "6E", "futures_ohlcv"),
}

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

# ── Market calendar ───────────────────────────────────────────────────────────
CALENDAR         = 'CME Globex Equity'   # used for session open/close times
HOLIDAY_CALENDAR = 'CME_Equity'        # used for full holiday schedule

# ── Statistics ────────────────────────────────────────────────────────────────
# stat_type=3 is the official CME settlement price
# See: https://databento.com/docs/schemas-and-data-formats/statistics
STAT_TYPE_SETTLEMENT = 3