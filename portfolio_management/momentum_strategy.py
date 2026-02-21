# autopep8: off
import pandas as pd
import sys

from typing import List
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# isort: split
from portfolio_management.strategy import StrategyFrequency, SignalTiming, ExecutionTiming, LongShortStrategy

# autopep8: on


class MomentumStrategy(LongShortStrategy):
    def __init__(self, tickers: List[str], lookback_days: int = 7):
        super().__init__(name='Momentum', tickers=tickers, rebalance_frequency=StrategyFrequency.WEEKLY,
                         signal_timing=SignalTiming.MARKET_CLOSE, execution_timing=ExecutionTiming.NEXT_CLOSE)
        self.lookback_days = lookback_days

    def _generate_signal(self, ticker: str, data: pd.DataFrame) -> float:
        if data['return_1w'] > 0:
            return 1.0
        else:
            return -1.0
