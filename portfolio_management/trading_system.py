# autopep8: off
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path.home() / 'quant_research'
sys.path.insert(0, str(ROOT_DIR))

# isort: split
from portfolio_management.portfolio import Portfolio
from portfolio_management.strategy import (ExecutionTiming, LongShortStrategy,
                                           SignalTiming)

# autopep8: on


class TradingSystem:
    def __init__(self,
                 portfolio: Portfolio,
                 strategies: List[LongShortStrategy],
                 strategy_allocations: Dict[str, float],
                 initial_date: datetime,
                 end_date: datetime):
        '''Initialise the trading system.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio instance that trades are executed against.
        strategies : list of LongShortStrategy
            Strategies to run during the backtest.
        strategy_allocations : dict of str -> float
            Mapping of strategy name -> fraction of GMV. Must sum to 1.0.
        initial_date : datetime
            Backtest start date.
        end_date : datetime
            Backtest end date (inclusive).

        Raises
        ------
        ValueError
            If strategy allocations do not sum to 1.0.
        '''
        self.portfolio = portfolio
        self.strategies = strategies
        self.strategy_allocations = strategy_allocations
        self.initial_date = initial_date
        self.end_date = end_date

        total_allocation = sum(strategy_allocations.values())
        if not np.isclose(total_allocation, 1.0):
            raise ValueError(
                f'Strategy allocations must sum to 1.0, got {total_allocation}')

        self.rebalance_dates = []

        self.execute_today_open = []
        self.execute_today_close = []

        self.execute_next_close = []
        self.execute_next_open = []

    def _get_strategy_allocation(self, strategy_name: str):
        '''Return the GMV allocation fraction for a given strategy name.'''
        return self.strategy_allocations[strategy_name]

    def rebalance_open(self, current_date: datetime, prices: Dict[str, float]):
        '''Execute pending rebalances at the market open.

        Applies weights from strategies queued in execute_today_open, combines them
        with all other strategy weights, and updates portfolio positions if
        the target weights have changed.

        Parameters
        ----------
        current_date : datetime
            The current trading date.
        prices : dict of str -> float
            Open prices used for execution.
        '''
        rebalanced_strategy_names = []

        # Get OLD weights before rebalancing
        old_weights = {}
        for strategy in self.strategies:
            for ticker, weight in strategy.current_weights.items():
                old_weights[ticker] = old_weights.get(
                    ticker, 0) + weight * self.strategy_allocations[strategy.name]

        # Step 1: Rebalance only strategies that need it (in execute_today_open)
        for strategy in self.execute_today_open:
            strategy.rebalance()
            rebalanced_strategy_names.append(strategy.name)

        # Step 2: Combine weights from ALL strategies (not just those
        # rebalancing)
        target_weights = {}
        for strategy in self.strategies:
            strategy_allocation = self.strategy_allocations[strategy.name]

            for ticker, weight in strategy.current_weights.items():
                target_weights[ticker] = target_weights.get(
                    ticker, 0) + weight * strategy_allocation

        # Step 3: Check if any weights changed
        weights_changed = (target_weights != old_weights)

        # Step 4: Update portfolio with ALL target weights if anything changed
        if weights_changed and rebalanced_strategy_names:
            self.rebalance_dates.append({
                'date': current_date,
                'timing': 'open',
                'strategies': rebalanced_strategy_names
            })
            self.portfolio.update_positions(
                target_weights, prices, current_date)

        # Update execution queue
        self.execute_today_open = self.execute_next_open.copy()
        self.execute_next_open = []

    def rebalance_close(self, current_date: datetime,
                        prices: Dict[str, float]):
        '''Execute pending rebalances at the market close.

        Applies weights from strategies queued in execute_today_close, combines them
        with all other strategy weights, and updates portfolio positions if
        the target weights have changed.

        Parameters
        ----------
        current_date : datetime
            The current trading date.
        prices : dict of str -> float
            Close prices used for execution.
        '''
        rebalanced_strategy_names = []

        # Get OLD weights before rebalancing
        old_weights = {}
        for strategy in self.strategies:
            for ticker, weight in strategy.current_weights.items():
                old_weights[ticker] = old_weights.get(
                    ticker, 0) + weight * self.strategy_allocations[strategy.name]

        # Step 1: Rebalance only strategies that need it (in execute_today_close)
        for strategy in self.execute_today_close:
            strategy.rebalance()
            rebalanced_strategy_names.append(strategy.name)

        # Step 2: Combine weights from ALL strategies (not just those
        # rebalancing)
        target_weights = {}
        for strategy in self.strategies:
            strategy_allocation = self.strategy_allocations[strategy.name]

            for ticker, weight in strategy.current_weights.items():
                target_weights[ticker] = target_weights.get(
                    ticker, 0) + weight * strategy_allocation

        # Step 3: Check if any weights changed
        weights_changed = (target_weights != old_weights)

        # Step 4: Update portfolio with ALL target weights if anything changed
        if weights_changed and rebalanced_strategy_names:
            self.rebalance_dates.append({
                'date': current_date,
                'timing': 'close',
                'strategies': rebalanced_strategy_names
            })
            self.portfolio.update_positions(
                target_weights, prices, current_date)

        # Update execution queue
        self.execute_today_close = self.execute_next_close.copy()
        self.execute_next_close = []

    def get_signal_open(self, current_date: datetime, data: pd.DataFrame):
        '''Check each strategy for market-open signal generation and queue executions.

        Parameters
        ----------
        current_date : datetime
            The current trading date.
        data : pd.DataFrame
            Market data indexed by ticker for the current date.
        '''
        for strategy in self.strategies:
            if strategy.signal_needed(
                    current_date) and strategy.signal_timing == SignalTiming.MARKET_OPEN:
                strategy.generate_weights(current_date, data)

                execution_timing = strategy.execution_timing

                if execution_timing == ExecutionTiming.SAME_DAY_CLOSE:
                    self.execute_today_close.append(strategy)
                elif execution_timing == ExecutionTiming.NEXT_OPEN:
                    self.execute_next_open.append(strategy)
                else:
                    self.execute_next_close.append(strategy)

    def get_signal_close(self, current_date: datetime, data: pd.DataFrame):
        '''Check each strategy for market-close signal generation and queue executions.

        Parameters
        ----------
        current_date : datetime
            The current trading date.
        data : pd.DataFrame
            Market data indexed by ticker for the current date.
        '''
        for strategy in self.strategies:
            if strategy.signal_needed(
                    current_date) and strategy.signal_timing == SignalTiming.MARKET_CLOSE:
                strategy.generate_weights(current_date, data)

                execution_timing = strategy.execution_timing

                if execution_timing == ExecutionTiming.NEXT_OPEN:
                    self.execute_next_open.append(strategy)
                else:
                    self.execute_next_close.append(strategy)

    def run_process_open(self, current_date: datetime,
                         data: pd.DataFrame, prices: Dict[str, float]):
        '''Run the full open-of-day process: generate open signals then execute open rebalances.

        Parameters
        ----------
        current_date : datetime
            The current trading date.
        data : pd.DataFrame
            Market data for the current date.
        prices : dict of str -> float
            Open prices.
        '''
        self.get_signal_open(current_date, data)
        self.rebalance_open(current_date, prices)

    def run_process_close(self, current_date: datetime,
                          data: pd.DataFrame, prices: Dict[str, float]):
        '''Run the full close-of-day process: generate close signals then execute close rebalances.

        Parameters
        ----------
        current_date : datetime
            The current trading date.
        data : pd.DataFrame
            Market data for the current date.
        prices : dict of str -> float
            Close prices.
        '''
        self.get_signal_close(current_date, data)
        self.rebalance_close(current_date, prices)

    def run_backtest(self, price_data: pd.DataFrame):
        '''Run the backtest over the configured date range.

        Iterates day by day, processing open and close events for each trading
        day present in price_data, recording daily snapshots, and returning a
        performance summary.

        Parameters
        ----------
        price_data : pd.DataFrame
            Multi-index DataFrame (date, ticker) with at least 'open' and
            'close' columns.

        Returns
        -------
        dict
            Portfolio performance summary from _compile_results.
        '''
        close_prices = {}
        current_date = self.initial_date

        while current_date <= self.end_date:
            if current_date in price_data.index.get_level_values(0):
                data = price_data.loc[current_date]

                open_prices = price_data.loc[current_date][['open']].to_dict()[
                    'open']
                self.run_process_open(current_date, data, open_prices)

                close_prices = price_data.loc[current_date][['close']].to_dict()[
                    'close']
                self.run_process_close(current_date, data, close_prices)

                self._record_daily_snapshot(current_date, close_prices)

            current_date += timedelta(days=1)

        return self._compile_results(close_prices)

    def _record_daily_snapshot(self, date: datetime, prices: Dict[str, float]):
        '''Record end-of-day portfolio state: cash, positions, and NAV.

        Parameters
        ----------
        date : datetime
            The current trading date.
        prices : dict of str -> float
            Close prices used for valuation.
        '''
        all_tickers = list(
            set(ticker for strategy in self.strategies for ticker in strategy.tickers))

        self.portfolio.record_cash(date)
        self.portfolio.record_position(date, all_tickers, prices)
        self.portfolio.record_nav(date, prices)

    def _compile_results(self, current_prices: Dict[str, float]) -> Dict:
        '''Assemble the final backtest results dict.

        Parameters
        ----------
        current_prices : dict of str -> float
            Final prices used to mark any remaining open positions to market.

        Returns
        -------
        dict
            Portfolio summary enriched with 'start_date' and 'end_date'.
        '''
        portfolio_summary = self.portfolio.get_portfolio_summary(
            current_prices)
        portfolio_summary['start_date'] = self.initial_date
        portfolio_summary['end_date'] = self.end_date

        return portfolio_summary
