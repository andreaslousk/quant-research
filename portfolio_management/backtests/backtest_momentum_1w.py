# autopep8: off
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path.home() / 'quant_research'
sys.path.insert(0, str(ROOT_DIR))

# isort: split
from portfolio_management.backtests import utils
from portfolio_management.momentum_strategy import MomentumStrategy

# autopep8: on
TICKERS = ['AAPL', 'MSFT']
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
INITIAL_CASH = 1_000_000
TARGET_GMV = 750_000
LOOKBACK_DAYS = 7

DATA_PATH = Path(__file__).parent.parent / 'tests' / \
    'test_data' / 'AAPL_MSFT_SPY.csv'

OUTPUT_DIR = Path(__file__).parent / 'results' / 'momentum-1w'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    '''Load data, run the weekly momentum backtest, and print the performance summary.'''
    # Load data
    data = utils.load_data(DATA_PATH)

    # Create strategy
    strategy = MomentumStrategy(tickers=TICKERS, lookback_days=LOOKBACK_DAYS)

    # Configuration
    config = {
        'start_date': START_DATE,
        'end_date': END_DATE,
        'initial_cash': INITIAL_CASH,
        'target_gmv': TARGET_GMV
    }

    # Run backtest
    print('\nRunning backtest...')
    results = utils.run_standard_backtest(
        strategy=strategy,
        data=data,
        config=config,
        output_dir=OUTPUT_DIR
    )

    # Print summary
    print('\n' + '=' * 60)
    print('BACKTEST SUMMARY')
    print('=' * 60)
    print(f'PnL: ${results['pnl']:,.2f}')
    print(f'Final NAV: ${results['final_nav']:,.2f}')
    print(f'Total Return: {results['return']:,.2%}')
    print(f'Annual Return: {results['annual_return']:,.2%}')
    print(f'Max Drawdown: {results['max_dd']:,.2%}')
    print(f'Max Drawdown Start Date: {results['max_dd_start'].date()}')
    print(f'Max Drawdown End Date: {results['max_dd_end'].date()}')
    print(f'Max Drawdown Duration: {results['max_dd_duration']}')
    print(f'Volume Traded: ${results['volume_trade']:,.2f}')
    print(f'Return Per Trade: {results['return_per_trade']:,.2%}')
    print(f'Average Daily Return: {results['mean_return']:,.2%}')
    print(f'Std Daily Return: {results['std_return']:,.2%}')
    print(f'Sharpe ratio: {results['sharpe']:,.2f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
