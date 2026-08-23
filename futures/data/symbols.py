'''
symbols.py
----------
CME contract-symbol parsing and calendar helpers shared by the futures-curve and
options tooling: month codes, expiry-month resolution, and prompt-contract logic.
'''

import re

import numpy as np
import pandas as pd

MONTH_CODE = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
              'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
MONTH_ABBR = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

_SYMBOL_RE = re.compile(r'^[A-Z0-9]+?([FGHJKMNQUVXZ])(\d{1,2})$')


def expiry(symbol, trade_year):
    '''Contract symbol (e.g. NGZ8 / NGU27) -> expiry month as a Timestamp (1st of
    the delivery month). 1-digit years resolve to the soonest year >= trade_year
    ending in that digit.'''
    m = _SYMBOL_RE.match(symbol)
    if not m:
        return pd.NaT
    month = MONTH_CODE[m.group(1)]
    digits = m.group(2)
    year = 2000 + int(digits) if len(digits) == 2 else trade_year + ((int(digits) - trade_year) % 10)
    return pd.Timestamp(year=year, month=month, day=1)


def resolve_year(digits, trade_year):
    '''Vectorized 1- or 2-digit contract year -> full year (digits is a string
    Series, trade_year an int array). 1-digit resolves to the soonest year >=
    trade_year ending in that digit (matches `expiry`).'''
    d = digits.astype(int)
    return np.where(digits.str.len() == 2, 2000 + d,
                    trade_year + ((d - trade_year) % 10))


def prompt_year(dates, month):
    '''Delivery year of the PROMPT (nearest-not-expired) contract of `month` for
    each date: this year if month-1st is still ahead, else next year.'''
    d = pd.DatetimeIndex(dates)
    mstart = pd.to_datetime(dict(year=d.year, month=month, day=1))
    return np.where(np.asarray(mstart) >= np.asarray(d), d.year, d.year + 1)
