'''
cache.py
--------
Generic on-disk result cache for research notebooks. One pickle per key under
`.cache/` (sibling to this file), so expensive loads/builds are computed once and
reused across notebooks and sessions. Instrument-agnostic — nothing here knows
about NG, ES, options, or any particular product.

    from cache import cached
    es_settle = cached('es_option_statistics', lambda: opt.load_option_statistics('ES'))

Pass refresh=True to recompute and overwrite. set_cache_dir(path) points the
cache elsewhere.
'''

import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / '.cache'


def set_cache_dir(path):
    '''Point the cache at a different directory.'''
    global CACHE_DIR
    CACHE_DIR = Path(path)


def cached(key, compute, refresh=False, verbose=True):
    '''Return compute(), persisting the result to <CACHE_DIR>/<key>.pkl.

    Keys are plain names (never date-stamped) — one pickle per series. Pass
    refresh=True to recompute and overwrite.
    '''
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f'{key}.pkl'
    if path.exists() and not refresh:
        if verbose:
            print(f'[cache] hit  {path.name}')
        with open(path, 'rb') as f:
            return pickle.load(f)
    if verbose:
        print(f'[cache] miss {path.name} — computing')
    obj = compute()
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return obj
