import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from typing import Dict, List
from enum import Enum


class StrategyFrequency(Enum):
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    CUSTOM = 'custom'


class SignalTiming(Enum):
    MARKET_CLOSE = 'market_close'
    MARKET_OPEN = 'market_open'


class ExecutionTiming(Enum):
    NEXT_OPEN = 'next_open'
    NEXT_CLOSE = 'next_close'
    # do this only if the signal is generated in market open
    SAME_DAY_CLOSE = 'same_day_close'


class LongShortStrategy:
    def __init__(self, name: str, tickers: List[str], rebalance_frequency: StrategyFrequency,
                 signal_timing: SignalTiming = SignalTiming.MARKET_CLOSE,
                 execution_timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN,
                 custom_days: int = None):
        self.name = name
        self.tickers = tickers
        self.rebalance_frequency = rebalance_frequency
        self.signal_timing = signal_timing
        self.execution_timing = execution_timing

        if self.signal_timing == SignalTiming.MARKET_CLOSE and self.execution_timing == ExecutionTiming.SAME_DAY_CLOSE:
            raise ValueError(
                'Not valid strategy inputs because this will lead to lookahead bias.')

        self.custom_days = custom_days

        self.last_signal_date = None
        self.last_execution_date = None

        self.execution_needed = False

        self.current_weights = {}
        self.pending_weights = {}      

        self.signal_history = []

    def signal_needed(self, current_date: datetime) -> bool:
        if self.last_signal_date is None:
            return True

        days_since = (current_date - self.last_signal_date).days

        if self.rebalance_frequency == StrategyFrequency.DAILY:
            return days_since >= 1
        elif self.rebalance_frequency == StrategyFrequency.WEEKLY:
            return days_since >= 7
        elif self.rebalance_frequency == StrategyFrequency.MONTHLY:
            return current_date.month != self.last_rebalance_date.month
        elif self.rebalance_frequency == StrategyFrequency.CUSTOM:
            return days_since >= self.custom_days
        
        return False

    def rebalance_needed(self, current_date: datetime) -> bool:
        if self.execution_needed:
            if self.execution_timing == ExecutionTiming.SAME_DAY_CLOSE:
                return (current_date - self.last_signal_date).days == 0
            else:
                return (current_date - self.last_signal_date).days > 0
        return False

    def generate_signals(self, current_date: datetime, data: pd.DataFrame) -> Dict[str, float]:
        signals = {}
        for ticker in self.tickers:
            signals[ticker] = self._generate_signal(ticker, data.loc[ticker])
        self.last_signal_date = current_date
        return signals

    def _generate_signal(self, ticker: str, data: pd.DataFrame) -> float:
        raise NotImplementedError

    def generate_weights(self, current_date:datetime, data: pd.DataFrame):
        signals = self.generate_signals(current_date, data)
        
        normal_factor = np.sum(np.abs(list(signals.values())))
        for ticker, signal in signals.items():
            self.pending_weights[ticker] = signal / normal_factor

        self.signal_history.append({
        'date': current_date,
        'signals': signals.copy(),
        'weights': self.pending_weights.copy()
        })

        self.execution_needed = True
        
        # Calculate when this will execute
        if self.execution_timing == ExecutionTiming.SAME_DAY_CLOSE:
            self.last_execution_date = self.last_signal_date
        else:  # NEXT_OPEN or NEXT_CLOSE
            self.last_execution_date = self.last_signal_date + timedelta(days=1)
        return self.pending_weights
    
    def rebalance(self):
        self.current_weights = self.pending_weights
        self.pending_weights = {}
        self.execution_needed = False
        return self.current_weights
    
    def get_execution_timing(self):
        return self.execution_timing
    
    def get_signal_timing(self):
        return self.signal_timing
