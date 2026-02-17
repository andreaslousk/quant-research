from data import data_processor as feature_engineering
from pathlib import Path
from strategy import LongShortStrategy, SignalTiming, ExecutionTiming
from portfolio import Portfolio
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class TradingSystem:
    def __init__(self,
                 portfolio: Portfolio,
                 strategies: List[LongShortStrategy],
                 strategy_allocations: Dict[str, float],
                 initial_date: datetime,
                 end_date: datetime):
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

        self.execute_open = []
        self.execute_close = []

        self.execute_next_close = []
        self.execute_next_open = []

    def _get_strategy_allocation(self, strategy_name: str):
        return self.strategy_allocations[strategy_name]

    def rebalance_open(self, current_date: datetime, prices: Dict[str, float]):
        rebalance_open = []

        # Get OLD weights before rebalancing
        old_weights = {}
        for strategy in self.strategies:
            for ticker, weight in strategy.current_weights.items():
                old_weights[ticker] = old_weights.get(
                    ticker, 0) + weight * self.strategy_allocations[strategy.name]

        # Step 1: Rebalance only strategies that need it (in execute_open)
        for strategy in self.execute_open:
            strategy.rebalance()
            rebalance_open.append(strategy.name)

        # Step 2: Combine weights from ALL strategies (not just those rebalancing)
        target_weights = {}
        for strategy in self.strategies:
            strategy_allocation = self.strategy_allocations[strategy.name]

            for ticker, weight in strategy.current_weights.items():
                target_weights[ticker] = target_weights.get(
                    ticker, 0) + weight * strategy_allocation

        # Step 3: Check if any weights changed
        weights_changed = (target_weights != old_weights)

        # Step 4: Update portfolio with ALL target weights if anything changed
        if weights_changed and rebalance_open:
            self.rebalance_dates.append({
                'date': current_date,
                'timing': 'open',
                'strategies': rebalance_open
            })
            self.portfolio.update_positions(
                target_weights, prices, current_date)

        # Update execution queue
        self.execute_open = self.execute_next_open.copy()
        self.execute_next_open = []

    def rebalance_close(self, current_date: datetime, prices: Dict[str, float]):
        rebalance_close = []

        # Get OLD weights before rebalancing
        old_weights = {}
        for strategy in self.strategies:
            for ticker, weight in strategy.current_weights.items():
                old_weights[ticker] = old_weights.get(
                    ticker, 0) + weight * self.strategy_allocations[strategy.name]

        # Step 1: Rebalance only strategies that need it (in execute_close)
        for strategy in self.execute_close:
            strategy.rebalance()
            rebalance_close.append(strategy.name)

        # Step 2: Combine weights from ALL strategies (not just those rebalancing)
        target_weights = {}
        for strategy in self.strategies:
            strategy_allocation = self.strategy_allocations[strategy.name]

            for ticker, weight in strategy.current_weights.items():
                target_weights[ticker] = target_weights.get(
                    ticker, 0) + weight * strategy_allocation

        # Step 3: Check if any weights changed
        weights_changed = (target_weights != old_weights)

        # Step 4: Update portfolio with ALL target weights if anything changed
        if weights_changed and rebalance_close:
            self.rebalance_dates.append({
                'date': current_date,
                'timing': 'close',
                'strategies': rebalance_close
            })
            self.portfolio.update_positions(
                target_weights, prices, current_date)

        # Update execution queue
        self.execute_close = self.execute_next_close.copy()
        self.execute_next_close = []

    def get_signal_open(self, current_date: datetime, data: pd.DataFrame):
        for strategy in self.strategies:
            if strategy.signal_needed(current_date) and strategy.signal_timing == SignalTiming.MARKET_OPEN:
                strategy.generate_weights(current_date, data)

                execution_timing = strategy.execution_timing

                if execution_timing == ExecutionTiming.SAME_DAY_CLOSE:
                    self.execute_close.append(strategy)
                elif execution_timing == ExecutionTiming.NEXT_OPEN:
                    self.execute_next_open.append(strategy)
                else:
                    self.execute_next_close.append(strategy)

    def get_signal_close(self, current_date: datetime, data: pd.DataFrame):
        for strategy in self.strategies:
            if strategy.signal_needed(current_date) and strategy.signal_timing == SignalTiming.MARKET_CLOSE:
                strategy.generate_weights(current_date, data)

                execution_timing = strategy.execution_timing

                if execution_timing == ExecutionTiming.NEXT_OPEN:
                    self.execute_next_open.append(strategy)
                else:
                    self.execute_next_close.append(strategy)

    def run_process_open(self, current_date: datetime, data: pd.DataFrame, prices: Dict[str, float]):
        self.get_signal_open(current_date, data)
        self.rebalance_open(current_date, prices)

    def run_process_close(self, current_date: datetime, data: pd.DataFrame, prices: Dict[str, float]):
        self.get_signal_close(current_date, data)
        self.rebalance_close(current_date, prices)

    def run_backtest(self, price_data: pd.DataFrame):
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

        return self._compile_results()

    def _record_daily_snapshot(self, date: datetime, prices: Dict[str, float]):
        all_tickers = list(set(ticker for strategy in self.strategies for ticker in strategy.tickers))
    
        self.portfolio.record_cash(date)
        self.portfolio.record_position(date, all_tickers, prices)
        self.portfolio.record_nav(prices)

    def _compile_results(self, current_prices: Dict[str, float]) -> Dict:
        portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)

        nav = list(self.portfolio.nav_history.values())

    def _calculate_max_drawdown(self, nav_series: np.ndarray) -> float:
        '''
        Calculate maximum drawdown

        Drawdown = peak-to-trough decline during a specific period
        '''
        cummax = np.maximum.accumulate(nav_series)
        drawdowns = (cummax - nav_series) / cummax
        return np.max(drawdowns) if len(drawdowns) > 0 else 0
