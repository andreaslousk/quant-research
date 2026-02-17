import sys
import pytest

import pandas as pd
import numpy as np

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))  # root directory

from data import data_processor
from portfolio_management.momentum_strategy import MomentumStrategy
from portfolio_management.portfolio import Portfolio

TEST_DATA_PATH = TEST_DATA_PATH = Path(__file__).parent.parent / \
        'tests' / 'test_data' / 'AAPL_MSFT.csv'
TICKERS = ['AAPL', 'MSFT']

@pytest.fixture
def portfolio():
    return Portfolio(initial_cash=1000000, target_gmv=750000)

@pytest.fixture
def return_data():
    raw_data = pd.read_csv(TEST_DATA_PATH)
    raw_data['date'] = pd.to_datetime(raw_data['date'])

    return_data = raw_data.groupby('ticker', group_keys=True).apply(
        data_processor.calculate_returns, include_groups=False).reset_index('ticker').set_index(['date', 'ticker']).dropna()
    
    return return_data

def test_portfolio_implementation(portfolio, return_data):
    np.random.seed(42)
    dates = return_data.index.get_level_values('date').unique().to_list()
    tickers = return_data.index.get_level_values('ticker').unique().to_list()

    cash_records = []
    position_records = []
    portfolio_value_records = []

    for date in dates:
        prices = return_data.loc[date]['close'].to_dict()
        
        aapl_weight = np.random.randint(low=0, high=11) / 10

        weights = {
            'AAPL': -0.5, #aapl_weight,
            'MSFT': 0.5
        }
        
        portfolio.update_positions(weights, prices, date)
        
        portfolio.record_position(date, tickers, prices)
        portfolio.record_cash(date)
        portfolio.record_nav(date, prices)

        latest_cash = portfolio.get_latest_cash()
        latest_position = portfolio.get_latest_positions(tickers)
        latest_portfolio_value = portfolio.get_portfolio_value(prices)
        
        for ticker, shares in latest_position.items():
            position_records.append({
                'date': date,
                'ticker': ticker,
                'shares': shares,
                'close_price': prices[ticker]
            })

        cash_records.append({'date' : date, 'cash' : latest_cash})
        portfolio_value_records.append({'date' : date, 'portfolio_value' : latest_portfolio_value})

    positions = pd.DataFrame(position_records).set_index(['date', 'ticker'])
    positions['nmv'] = positions['shares'] * positions['close_price']
    nmv = positions.groupby('date')['nmv'].sum()

    cash = pd.DataFrame(cash_records).set_index(['date'])
    portfolio_value = pd.DataFrame(portfolio_value_records).set_index(['date'])

    nav = cash.join(nmv)
    nav = nav.join(portfolio_value)
    nav['nav'] = nav['cash'] + nav['nmv']

    assert portfolio.initial_cash == portfolio.nav_history[0]['nav'], 'Initial cash and initial NAV must match!'
    assert nav['cash'].min() >= 0, 'Cash can never be negative!'
    assert (nav['nav'] == nav['portfolio_value']).all(), 'Portfolio value calculations are incorrect!'
