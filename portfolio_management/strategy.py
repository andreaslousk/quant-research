# autopep8: off
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List

import numpy as np
import pandas as pd

# autopep8: on


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
                 lookback_days: int = None):
        '''Initialise a long/short strategy.

        Parameters
        ----------
        name : str
            Unique strategy identifier used for allocation lookups.
        tickers : list of str
            Universe of tickers the strategy trades.
        rebalance_frequency : StrategyFrequency
            How often to regenerate signals.
        signal_timing : SignalTiming, optional
            Whether signals are generated at market open or close.
        execution_timing : ExecutionTiming, optional
            When the generated weights are executed (next open, next close,
            or same-day close).
        custom_days : int, optional
            Required when rebalance_frequency is CUSTOM; interval in calendar days.

        Raises
        ------
        ValueError
            If signal_timing is MARKET_CLOSE and execution_timing is
            SAME_DAY_CLOSE, which would introduce lookahead bias.
        '''
        self.name = name
        self.tickers = tickers
        self.rebalance_frequency = rebalance_frequency
        self.signal_timing = signal_timing
        self.execution_timing = execution_timing

        if self.signal_timing == SignalTiming.MARKET_CLOSE and self.execution_timing == ExecutionTiming.SAME_DAY_CLOSE:
            raise ValueError(
                'Not valid strategy inputs because this will lead to lookahead bias.')

        self.lookback_days = lookback_days

        self.last_signal_date = None
        self.last_execution_date = None

        self.execution_needed = False

        self.current_weights = {}
        self.pending_weights = {}

        self.signal_history = []

    def signal_needed(self, current_date: datetime) -> bool:
        '''Return True if enough time has passed to generate a new signal.

        Parameters
        ----------
        current_date : datetime
            The date being evaluated.

        Returns
        -------
        bool
        '''
        if self.last_signal_date is None:
            return True

        days_since = (current_date - self.last_signal_date).days

        if self.rebalance_frequency == StrategyFrequency.DAILY:
            return days_since >= 1
        elif self.rebalance_frequency == StrategyFrequency.WEEKLY:
            return days_since >= 7
        elif self.rebalance_frequency == StrategyFrequency.MONTHLY:
            return (current_date.year, current_date.month) != (self.last_signal_date.year, self.last_signal_date.month)
        elif self.rebalance_frequency == StrategyFrequency.CUSTOM:
            return days_since >= self.lookback_days

        return False

    def rebalance_needed(self, current_date: datetime) -> bool:
        '''Return True if a pending signal is ready to be executed today.

        Parameters
        ----------
        current_date : datetime
            The date being evaluated.

        Returns
        -------
        bool
        '''
        if self.execution_needed:
            if self.execution_timing == ExecutionTiming.SAME_DAY_CLOSE:
                return (current_date - self.last_signal_date).days == 0
            else:
                return (current_date - self.last_signal_date).days > 0
        return False

    def generate_signals(self, current_date: datetime,
                         data: pd.DataFrame) -> Dict[str, float]:
        '''Generate a raw signal for every ticker and record the signal date.

        Parameters
        ----------
        current_date : datetime
            The date on which signals are being generated.
        data : pd.DataFrame
            DataFrame indexed by ticker with feature columns for the current date.

        Returns
        -------
        dict of str -> float
            Mapping of ticker -> raw signal value.
        '''
        signals = {}
        for ticker in self.tickers:
            signals[ticker] = self._generate_signal(ticker, data.loc[ticker])
        self.last_signal_date = current_date
        return signals

    def _generate_signal(self, ticker: str, data: pd.DataFrame) -> float:
        '''Generate a signal for a single ticker. Must be overridden by subclasses.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        data : pd.DataFrame
            Feature row for the ticker on the current date.

        Returns
        -------
        float
            Signal value (positive = bullish, negative = bearish).

        Raises
        ------
        NotImplementedError
        '''
        raise NotImplementedError

    def generate_weights(self, current_date: datetime, data: pd.DataFrame):
        '''Generate normalised portfolio weights and queue them for execution.

        Calls _generate_signal for each ticker, normalises by total absolute signal
        so weights sum to 1 in absolute value, and appends a record to signal_history.

        Parameters
        ----------
        current_date : datetime
            The date on which weights are generated.
        data : pd.DataFrame
            DataFrame indexed by ticker with feature columns for the current date.

        Returns
        -------
        dict of str -> float
            Pending normalised weights.
        '''
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
            self.last_execution_date = self.last_signal_date + \
                timedelta(days=1)
        return self.pending_weights

    def rebalance(self):
        '''Promote pending weights to current weights and clear the execution flag.

        Returns
        -------
        dict of str -> float
            The newly activated weights.
        '''
        self.current_weights = self.pending_weights
        self.pending_weights = {}
        self.execution_needed = False
        return self.current_weights

    def get_execution_timing(self):
        '''Return the strategy's execution timing setting.'''
        return self.execution_timing

    def get_signal_timing(self):
        '''Return the strategy's signal timing setting.'''
        return self.signal_timing
