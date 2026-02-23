# autopep8: off
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT_DIR = Path.home() / 'quant_research'
sys.path.insert(0, str(ROOT_DIR))

# isort: split
from portfolio_management.strategy import (ExecutionTiming, LongShortStrategy,
                                           SignalTiming, StrategyFrequency)

# autopep8: on


class MomentumStrategy(LongShortStrategy):
    def __init__(self, tickers: List[str], lookback_days: int = 7):
        '''Initialise a weekly momentum strategy.

        Signals are generated at market close and executed at the next close.

        Parameters
        ----------
        tickers : list of str
            Tickers to include in the strategy universe.
        lookback_days : int, optional
            Minimum calendar days between signal generation events.
        '''
        super().__init__(name='Momentum', tickers=tickers, rebalance_frequency=StrategyFrequency.CUSTOM,
                         signal_timing=SignalTiming.MARKET_CLOSE, execution_timing=ExecutionTiming.NEXT_CLOSE,
                         lookback_days=lookback_days)

    def _generate_signal(self, ticker: str, data: pd.DataFrame) -> float:
        '''Return +1.0 if the 1-week return is positive, -1.0 otherwise.

        Parameters
        ----------
        ticker : str
            Ticker symbol (unused; present for interface compatibility).
        data : pd.DataFrame
            Feature row for the ticker on the current date; must contain 'return_1w'.

        Returns
        -------
        float
            +1.0 (long signal) or -1.0 (short signal).
        '''
        if data['return_1w'] > 0:
            return 1.0
        else:
            return -1.0
