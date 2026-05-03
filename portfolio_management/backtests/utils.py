# autopep8: off
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path.home() / 'quant_research'
sys.path.insert(0, str(ROOT_DIR))

# isort: split
from portfolio_management.data import data_processor
from portfolio_management.portfolio import Portfolio
from portfolio_management.trading_system import TradingSystem

# autopep8: on


def load_data(data_path):
    '''Load price data from a CSV file and compute 1-day returns for each ticker.

    Parameters
    ----------
    data_path : str or Path
        Path to a CSV file with columns including 'ticker', 'date', and price columns.

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame (date, ticker) with return columns added and NaN rows dropped.
    '''
    raw_data = pd.read_csv(data_path)
    raw_data['date'] = pd.to_datetime(raw_data['date'])

    return_data = raw_data.groupby('ticker', group_keys=True).apply(
        data_processor.calculate_returns, include_groups=False).reset_index('ticker').set_index(['date', 'ticker']).dropna()

    return return_data


def create_portfolio(initial_cash=1_000_000, target_gmv=500_000):
    '''Create a Portfolio instance for use in a backtest.

    Parameters
    ----------
    initial_cash : float, optional
        Starting cash balance.
    target_gmv : float, optional
        Target gross market value cap for position sizing.

    Returns
    -------
    Portfolio
    '''
    return Portfolio(initial_cash=initial_cash, target_gmv=target_gmv)


def create_trading_system(strategies, strategy_allocations, config):
    '''Create a TradingSystem wired to a new Portfolio using the provided config.

    Parameters
    ----------
    strategies : list of LongShortStrategy
        Strategies to run.
    strategy_allocations : dict of str -> float
        Mapping of strategy name -> GMV fraction. Must sum to 1.0.
    config : dict
        Must contain keys: 'initial_cash', 'target_gmv', 'start_date', 'end_date'.

    Returns
    -------
    TradingSystem
    '''
    portfolio = create_portfolio(config['initial_cash'], config['target_gmv'])

    return TradingSystem(
        portfolio=portfolio,
        strategies=strategies,
        strategy_allocations=strategy_allocations,
        initial_date=pd.to_datetime(config['start_date']),
        end_date=pd.to_datetime(config['end_date'])
    )


def plot_nav(trading_system, benchmark_data=None, save_path=None):
    '''Plot portfolio NAV over time, optionally overlaid with a benchmark.

    Parameters
    ----------
    trading_system : TradingSystem
        Completed backtest whose nav_history will be plotted.
    benchmark_data : pd.DataFrame, optional
        Multi-index (date, ticker) DataFrame with a 'return_1d' column.
        If provided, the benchmark is rebased to the portfolio's initial NAV
        and plotted alongside.
    save_path : str or Path, optional
        If given, save the figure to this path instead of displaying it.
    '''
    nav_history = trading_system.portfolio.nav_history
    dates = [entry['date'] for entry in nav_history]
    nav_series = [entry['nav'] for entry in nav_history]

    plt.figure(figsize=(12, 6))
    plt.plot(
        dates,
        nav_series,
        linewidth=2,
        label='Portfolio NAV',
        color='blue')

    if benchmark_data is not None:
        benchmark_name = benchmark_data.index.get_level_values('ticker')[0]

        start_date = dates[0]
        end_date = dates[-1]

        benchmark_window_data = benchmark_data.query(
            '@start_date <= date <= @end_date')
        benchmark_dates = benchmark_window_data.index.get_level_values(
            'date').unique()

        benchmark_returns = benchmark_window_data['return_1d'].values
        benchmark_returns[0] = 0  # Replace first entry
        benchmark_returns = benchmark_returns + 1  # Add 1 to each

        benchmark_nav_series = nav_history[0]['nav'] * \
            np.cumprod(benchmark_returns)

        plt.plot(
            benchmark_dates,
            benchmark_nav_series,
            linewidth=2,
            label=f'{benchmark_name}-Invested NAV',
            color='red')

    plt.title('Portfolio NAV Over Time')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('NAV ($)')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'NAV plot saved to {save_path}')
    else:
        plt.show()

    plt.close()


def plot_drawdown(trading_system, save_path=None):
    '''Plot the portfolio drawdown (underwater equity curve) over time.

    Parameters
    ----------
    trading_system : TradingSystem
        Completed backtest whose nav_history will be used.
    save_path : str or Path, optional
        If given, save the figure to this path instead of displaying it.
    '''
    nav_history = trading_system.portfolio.nav_history
    dates = [entry['date'] for entry in nav_history]
    nav_series = [entry['nav'] for entry in nav_history]

    cummax = np.maximum.accumulate(nav_series)
    drawdown = (nav_series - cummax) / cummax

    plt.figure(figsize=(12, 6))
    plt.fill_between(dates, drawdown * 100, 0, alpha=0.3, color='red')
    plt.plot(dates, drawdown * 100, color='red', linewidth=2)
    plt.title('Drawdown Over Time')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Drawdown plot saved to {save_path}')
    else:
        plt.show()

    plt.close()


def plot_monthly_returns_heatmap(trading_system, save_path=None):
    '''Plot a heatmap of monthly returns (year x month).

    Parameters
    ----------
    trading_system : TradingSystem
        Completed backtest whose nav_history will be used.
    save_path : str or Path, optional
        If given, save the figure to this path instead of displaying it.
    '''
    nav_history = trading_system.portfolio.nav_history
    dates = [entry['date'] for entry in nav_history]
    nav_series = [entry['nav'] for entry in nav_history]

    # Calculate daily returns
    daily_returns = np.diff(nav_series) / nav_series[:-1]

    # Create DataFrame
    returns_df = pd.DataFrame({
        'date': dates[1:],
        'return': daily_returns
    })
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    returns_df.set_index('date', inplace=True)

    # Calculate monthly returns
    monthly_returns = (1 + returns_df['return']).resample('ME').prod() - 1

    # Pivot for heatmap
    monthly_df = monthly_returns.to_frame()
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    heatmap_data = monthly_df.pivot(
        index='year', columns='month', values='return')
    heatmap_data.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Return (%)'})
    plt.title('Monthly Returns Heatmap')
    plt.xlabel('Month')
    plt.ylabel('Year')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Monthly returns heatmap saved to {save_path}')
    else:
        plt.show()

    plt.close()


def plot_returns_distribution(trading_system, save_path=None):
    '''Plot a histogram of daily portfolio returns with the mean annotated.

    Parameters
    ----------
    trading_system : TradingSystem
        Completed backtest whose nav_history will be used.
    save_path : str or Path, optional
        If given, save the figure to this path instead of displaying it.
    '''
    nav_history = trading_system.portfolio.nav_history
    nav_series = [entry['nav'] for entry in nav_history]

    # Calculate daily returns
    daily_returns = np.diff(nav_series) / nav_series[:-1]

    plt.figure(figsize=(10, 6))
    plt.hist(
        daily_returns * 100,
        bins=50,
        alpha=0.7,
        color='blue',
        edgecolor='black')
    plt.axvline(
        np.mean(daily_returns) * 100,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {
            np.mean(daily_returns) * 100:.2f}%')
    plt.axvline(0, color='black', linewidth=1)

    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Returns distribution plot saved to {save_path}')
    else:
        plt.show()

    plt.close()


def save_results(results, filepath):
    '''Serialise a backtest results dict to a JSON file.

    Parameters
    ----------
    results : dict
        Backtest performance metrics to save.
    filepath : str or Path
        Destination file path.
    '''
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'Results saved to {filepath}')


def run_standard_backtest(strategy, data, config, output_dir):
    '''Run a single-strategy backtest and save all outputs to output_dir.

    Creates the trading system, runs the backtest, generates the four standard
    plots (NAV, drawdown, monthly returns heatmap, returns distribution),
    and saves the results JSON.

    Parameters
    ----------
    strategy : LongShortStrategy
        Strategy instance to backtest.
    data : pd.DataFrame
        Multi-index (date, ticker) price data prepared by load_data.
    config : dict
        Must contain keys: 'initial_cash', 'target_gmv', 'start_date', 'end_date'.
    output_dir : str or Path
        Directory where plots and results.json will be saved.

    Returns
    -------
    dict
        Portfolio performance summary.
    '''
    system = create_trading_system(
        strategies=[strategy],
        strategy_allocations={strategy.name: 1.0},
        config=config
    )

    results = system.run_backtest(data)

    spy_data = data.xs('SPY', level='ticker', drop_level=False)

    # Generate plots
    plot_nav(system, spy_data, output_dir / 'nav.png')
    plot_drawdown(system, output_dir / 'drawdown.png')
    plot_monthly_returns_heatmap(system, output_dir / 'monthly_returns.png')
    plot_returns_distribution(system, output_dir / 'returns_distribution.png')

    # Save results
    save_results(results, output_dir / 'results.json')

    print(f'Backtest complete. Results saved to {output_dir}')

    return results
