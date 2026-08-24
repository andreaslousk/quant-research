'''
readers.py
----------
Format-agnostic record reader for the data pipeline. Maps a single Databento file
to a raw DataFrame with the SAME columns whether the source is CSV or DBN
(.dbn / .dbn.zst). Everything downstream (cleaning, symbol parsing) consumes the
raw Databento schema, so callers never branch on the on-disk format.

DBN is read with price_type='fixed' and pretty_ts=False so prices stay int64
fixed-point and timestamps stay int64 ns — identical to the CSV columns. That
means no double-scaling and no downstream changes when a directory switches from
CSV to DBN. `databento` is imported lazily so CSV-only use needs no dependency.
'''

from pathlib import Path

import pandas as pd


def read_records(path, usecols=None, dtype=None) -> pd.DataFrame:
    '''Read one Databento file (.csv, .dbn, .dbn.zst) into a raw DataFrame whose
    columns match the Databento record schema. `usecols` selects/orders columns;
    `dtype` is honoured for CSV only (DBN is already typed at the source).'''
    path = Path(path)
    if path.suffix == '.csv':
        return pd.read_csv(path, usecols=usecols, dtype=dtype)

    import databento as db                                  # lazy: only needed for DBN
    df = (db.DBNStore.from_file(path)
          .to_df(price_type='fixed', pretty_ts=False, map_symbols=True)
          .reset_index())                                   # ts_event/ts_recv index -> column
    return df[usecols] if usecols else df
