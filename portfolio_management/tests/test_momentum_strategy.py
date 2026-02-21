# autopep8: off
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))  # root directory

# isort: split
from data import data_processor
from portfolio_management.momentum_strategy import MomentumStrategy

# autopep8: on
TEST_DATA_PATH = TEST_DATA_PATH = Path(__file__).parent.parent / \
    'tests' / 'test_data' / 'AAPL_MSFT.csv'
TICKERS = ['AAPL', 'MSFT']


@pytest.fixture
def strategy():
    return MomentumStrategy(tickers=TICKERS, lookback_days=7)


@pytest.fixture
def return_data():
    raw_data = pd.read_csv(TEST_DATA_PATH)
    raw_data['date'] = pd.to_datetime(raw_data['date'])

    return_data = raw_data.groupby('ticker', group_keys=True).apply(
        data_processor.calculate_returns, include_groups=False).reset_index('ticker').set_index(['date', 'ticker']).dropna()

    return return_data


def test_signal_generation(strategy, return_data):
    dates = return_data.index.get_level_values(0).unique().to_list()

    assert strategy.signal_needed(
        dates[0]), 'On the first day we require a signal!'

    records = []
    for date in dates:
        if strategy.signal_needed(date):
            date_data = return_data.loc[date]
            weights = strategy.generate_weights(date, date_data)
            for ticker, weight in weights.items():
                records.append({
                    'date': date,
                    'ticker': ticker,
                    'signal_needed': True,
                    'weight': weight
                })
        else:
            for ticker, weight in strategy.current_weights.items():
                records.append({
                    'date': date,
                    'ticker': ticker,
                    'signal_needed': False,
                    'weight': weight  # Carry forward last weights
                })

    signals = pd.DataFrame(records).set_index(['date', 'ticker'])
    weights_total = signals['weight'].abs().groupby('date').sum()

    assert (weights_total == 1).all(), 'Position weights must all total to 1!'

    signals = signals.merge(return_data['return_1w'], on=['date', 'ticker'])
    signals['weight_sign'] = np.sign(signals['weight']).replace(0, -1)
    signals['return_1w_sign'] = np.sign(signals['return_1w']).replace(0, -1)

    assert (signals['weight_sign'] == signals['return_1w_sign']).all(
    ), 'Signal generation does not line up with momentum strategy logic!'

    dates = signals.index.get_level_values('date').unique()
    date_gaps = pd.Series(
        dates, name='days_since_last_signal').diff().dt.days.dropna()

    assert (date_gaps >= strategy.lookback_days).all(
    ), 'Signals generated too early!'
