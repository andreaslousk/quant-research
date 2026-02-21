# autopep8: off
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))  # root directory

# isort: split
from data import data_processor

from portfolio_management.momentum_strategy import MomentumStrategy
from portfolio_management.portfolio import Portfolio
from portfolio_management.trading_system import TradingSystem

# autopep8: on


@pytest.fixture
def aapl_data_path():
    return Path(__file__).parent.parent / \
        'tests' / 'test_data' / 'aapl_return_positive_returns.json'


@pytest.fixture
def msft_data_path():
    return Path(__file__).parent.parent / \
        'tests' / 'test_data' / 'msft_return_positive_returns.json'


@pytest.fixture
def positive_return_data_path():
    return Path(__file__).parent.parent / \
        'tests' / 'test_data' / 'test_trading_system_positive_returns.csv'


@pytest.fixture
def market_return_data_path():
    return Path(__file__).parent.parent / \
        'tests' / 'test_data' / 'test_trading_system_market_returns.csv'


@pytest.fixture
def tickers():
    return ['AAPL', 'MSFT']


@pytest.fixture
def positive_return_data(positive_return_data_path):
    raw_data = pd.read_csv(positive_return_data_path)
    raw_data['date'] = pd.to_datetime(raw_data['date'], format='%m/%d/%y')

    positive_return_data = raw_data.groupby('ticker', group_keys=True).apply(
        data_processor.calculate_returns, include_groups=False).reset_index('ticker').set_index(['date', 'ticker']).dropna()

    return positive_return_data


@pytest.fixture
def market_return_data(market_return_data_path):
    raw_data = pd.read_csv(market_return_data_path)
    raw_data['date'] = pd.to_datetime(raw_data['date'], format='%m/%d/%y')

    market_return_data = raw_data.groupby('ticker', group_keys=True).apply(
        data_processor.calculate_returns, include_groups=False).reset_index('ticker').set_index(['date', 'ticker']).dropna()

    return market_return_data


@pytest.fixture
def portfolio():
    return Portfolio(initial_cash=1000000, target_gmv=750000)


@pytest.fixture
def strategies(tickers):
    return [MomentumStrategy(tickers=tickers, lookback_days=7)]


@pytest.fixture
def strategy_allocations(strategies):
    strategy_allocations = {}
    num_strategies = len(strategies)
    for strategy in strategies:
        strategy_allocations[strategy.name] = 1 / num_strategies
    return strategy_allocations


@pytest.fixture
def trading_system_positive_returns(
        portfolio, strategies, strategy_allocations, positive_return_data):
    dates = positive_return_data.index.get_level_values(
        'date').unique().to_list()
    initial_date = dates[0]
    end_date = dates[-1]

    return TradingSystem(portfolio, strategies,
                         strategy_allocations, initial_date, end_date)


@pytest.fixture
def trading_system_market_returns(
        portfolio, strategies, strategy_allocations, market_return_data):
    dates = market_return_data.index.get_level_values(
        'date').unique().to_list()
    initial_date = dates[0]
    end_date = dates[-1]

    return TradingSystem(portfolio, strategies,
                         strategy_allocations, initial_date, end_date)


def test_backtesting_positive_returns(
        trading_system_positive_returns, positive_return_data, aapl_data_path, msft_data_path, tolerance=1e-4):
    backtest_res = trading_system_positive_returns.run_backtest(
        positive_return_data)

    with open(aapl_data_path, 'r') as f:
        aapl_data = json.load(f)
        aapl_return = aapl_data['return']

    with open(msft_data_path, 'r') as f:
        msft_data = json.load(f)
        msft_return = msft_data['return']

    aapl_expected_weight = 0.5
    msft_expected_weight = 0.5

    # Get initial prices to calculate actual shares bought

    dates = positive_return_data.index.get_level_values(
        'date').unique().to_list()
    initial_date = dates[0]

    initial_aapl_price = positive_return_data.loc[(
        initial_date, 'AAPL')]['close']
    initial_msft_price = positive_return_data.loc[(
        initial_date, 'MSFT')]['close']

    target_gmv = trading_system_positive_returns.portfolio.target_gmv
    initial_cash = trading_system_positive_returns.portfolio.initial_cash
    effective_gmv = min(target_gmv, initial_cash)

    # Calculate actual shares using int() like your code does
    aapl_notional = aapl_expected_weight * effective_gmv
    msft_notional = msft_expected_weight * effective_gmv

    aapl_shares = int(aapl_notional / initial_aapl_price)
    msft_shares = int(msft_notional / initial_msft_price)

    # Calculate actual return based on floored shares
    actual_aapl_notional = aapl_shares * initial_aapl_price
    actual_msft_notional = msft_shares * initial_msft_price
    total_deployed = actual_aapl_notional + actual_msft_notional

    actual_aapl_weight = actual_aapl_notional / total_deployed
    actual_msft_weight = actual_msft_notional / total_deployed

    expected_position_return = aapl_return * \
        actual_aapl_weight + msft_return * actual_msft_weight
    expected_return = expected_position_return * \
        (total_deployed / initial_cash)

    backtest_return = backtest_res['return']

    assert abs(expected_return - backtest_return) <= tolerance, \
        f'Expected return {expected_return} does not match backtest return {backtest_return}!'


def test_nav_reconciliation(
        trading_system_market_returns, market_return_data, tolerance=1e-4):
    backtest_res = trading_system_market_returns.run_backtest(
        market_return_data)

    final_nav = backtest_res['final_nav']

    last_date = trading_system_market_returns.end_date
    last_prices = market_return_data.xs(
        last_date, level='date')['close'].to_dict()

    final_cash = trading_system_market_returns.portfolio.cash

    positions = trading_system_market_returns.portfolio.get_position_details(
        last_prices)

    position_val = sum(position['notional'] for position in positions)

    assert abs(final_nav - final_cash -
               position_val) <= tolerance, 'Final NAV does not match sum of individual components!'


def test_pnl_equals_nav_change(
        trading_system_market_returns, market_return_data, tolerance=1e-4):
    backtest_res = trading_system_market_returns.run_backtest(
        market_return_data)

    nav_history = trading_system_market_returns.portfolio.nav_history

    initial_nav = nav_history[0]['nav']
    final_nav = nav_history[-1]['nav']

    assert abs(backtest_res['pnl'] - final_nav +
               initial_nav) <= tolerance, 'Change in NAV does not equal PnL!'
