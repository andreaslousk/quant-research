# autopep8: off
import datetime
import sys
import time
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config
from massive import RESTClient

ROOT_DIR = Path.home() / 'quant_research'
sys.path.insert(0, str(ROOT_DIR))

# isort: split
from data.config import *
from data.data_processor import *

# autopep8: on


def create_s3_session():
    '''Create and return a configured boto3 S3 client.'''
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    s3 = session.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        config=Config(signature_version='s3v4'),
    )
    return s3


def download_stock_data(start_date, end_date, output_dir=DATA_GZ_DIR):
    '''Download daily aggregate stock data from S3 for a date range.

    Parameters
    ----------
    start_date : datetime.date
        First date to download (inclusive).
    end_date : datetime.date
        Last date to download (inclusive).
    output_dir : str, optional
        Directory to save the downloaded .csv.gz files.

    Returns
    -------
    list of str
        Paths to the successfully downloaded files.
    '''
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    s3 = create_s3_session()

    downloaded_files = []
    failed_dates = []

    curr_date = start_date
    total_days = (end_date - start_date).days + 1

    print(f'Downloading data from {start_date} to {end_date}')
    print(f'Total days: {total_days}')
    print(f'Output directory: {output_path}')
    print(f'{'=' * 50}\n')

    while curr_date <= end_date:
        try:
            curr_year = f'{curr_date.year}'
            curr_month = f'{curr_date.month:02d}'
            curr_day = f'{curr_date.day:02d}'

            object_key = f'{PREFIX}/day_aggs_v1/{curr_year}/{curr_month}/{curr_year}-{curr_month}-{curr_day}.csv.gz'
            local_file_name = object_key.split('/')[-1]
            local_file_path = output_path / local_file_name

            s3.download_file(BUCKET_NAME, object_key, str(local_file_path))

            downloaded_files.append(str(local_file_path))

        except Exception as e:
            pass

        finally:
            curr_date += datetime.timedelta(days=1)

    print(f'\n{'=' * 50}')
    print(f'Summary:')
    print(f'  Successfully downloaded: {len(downloaded_files)} files')
    print(f'{'=' * 50}')

    if failed_dates:
        print(
            f'\nFailed dates: {failed_dates[:10]}{'...' if len(failed_dates) > 10 else ''}')

    return downloaded_files


def load_all_csv_files(directory, pattern='*.csv'):
    '''Load and combine all CSV files matching a glob pattern from a directory.

    Each file's stem is parsed as a date and added as a 'date' column.

    Parameters
    ----------
    directory : str or Path
        Directory to search for CSV files.
    pattern : str, optional
        Glob pattern to filter files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame from all matched CSV files, or an empty DataFrame
        if no files are found.
    '''
    directory = Path(directory)
    csv_files = sorted(directory.glob(pattern))

    if not csv_files:
        print(f'No CSV files found in {directory}')
        return pd.DataFrame()

    print(f'Found {len(csv_files)} CSV files in {directory}')
    print(f'Loading files...')

    df_list = []

    for i, file in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file)

            date_str = file.stem
            df['date'] = pd.to_datetime(date_str)

            df_list.append(df)

            if i % 50 == 0:
                print(f'  Loaded {i}/{len(csv_files)} files...')

        except Exception as e:
            pass

    print(f'Combining {len(df_list)} dataframes...')
    combined_df = pd.concat(df_list, ignore_index=True)

    print(f'\n{'=' * 50}')
    print(f'{'=' * 50}')

    return combined_df


def get_dividends_for_ticker(ticker, max_retries=API_MAX_RETRIES):
    '''Fetch dividend history for a single ticker via the REST API.

    Retries on failure with exponential back-off.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    max_retries : int, optional
        Maximum number of attempts before giving up.

    Returns
    -------
    tuple of (list, list)
        (ex_dates, dividends) — ex-dividend dates and corresponding cash amounts.
        Returns ([], []) on persistent failure.
    '''
    for attempt in range(max_retries):
        try:
            client = RESTClient(AWS_SECRET_ACCESS_KEY)

            dividends = []
            for d in client.list_stocks_dividends(
                ticker=f'{ticker}',
                limit='100',
                sort='ticker.asc',
            ):
                dividends.append(d)

            ex_dates = [div.ex_dividend_date for div in dividends]
            dividends = [div.cash_amount for div in dividends]

            return ex_dates, dividends

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(
                    f'  Retry {attempt + 1} for {ticker} in {wait_time}s... ({e})')
                time.sleep(wait_time)
            else:
                print(
                    f'✗ Failed to get dividends for {ticker} after {max_retries} attempts: {e}')
                return [], []


def get_dividends_data(tickers, batch_size=API_BATCH_SIZE,
                       batch_delay=API_BATCH_DELAY, request_delay=API_REQUEST_DELAY):
    '''Fetch dividend data for a list of tickers in rate-limited batches.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols.
    batch_size : int, optional
        Number of tickers per batch before applying a longer delay.
    batch_delay : float, optional
        Seconds to wait between batches.
    request_delay : float, optional
        Seconds to wait between individual requests.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['ticker', 'ex_dividend_date', 'dividend_amount'].
    '''
    all_divs = []

    print(f'Fetching dividends for {len(tickers)} tickers...')
    print(
        f'Batch size: {batch_size}, Batch delay: {batch_delay}s, Request delay: {request_delay}s')
    print(f'{'=' * 50}\n')

    for i, ticker in enumerate(tickers, 1):
        ex_dates, dividends = get_dividends_for_ticker(ticker)

        for ex_dividend_date, dividend in zip(ex_dates, dividends):
            all_divs.append({
                'ticker': ticker,
                'ex_dividend_date': ex_dividend_date,
                'dividend_amount': dividend
            })

        if i % 50 == 0:
            print(f'  Processed {i}/{len(tickers)} tickers...')

        time.sleep(request_delay)

        if i % batch_size == 0:
            print(f'  Batch complete, waiting {batch_delay}s...')
            time.sleep(batch_delay)

    dividends_df = pd.DataFrame(all_divs)

    if len(dividends_df) > 0:
        dividends_df['ex_dividend_date'] = pd.to_datetime(
            dividends_df['ex_dividend_date'])

    print(f'\n{'=' * 50}')
    print(
        f'✓ Collected {len(dividends_df)} dividend records from {len(tickers)} tickers')
    print(f'{'=' * 50}')

    return dividends_df


def get_splits_for_ticker_api(ticker, max_retries=API_MAX_RETRIES):
    '''Fetch stock split history for a single ticker via the REST API.

    Retries on failure with exponential back-off.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    max_retries : int, optional
        Maximum number of attempts before giving up.

    Returns
    -------
    tuple of (list, list)
        (split_dates, split_ratios) — execution dates and new/old share ratios.
        Returns ([], []) on persistent failure.
    '''
    for attempt in range(max_retries):
        try:
            client = RESTClient(AWS_SECRET_ACCESS_KEY)

            splits = []
            for s in client.list_stocks_splits(
                ticker=ticker,
                limit='100',
                sort='execution_date.desc',
            ):
                splits.append(s)

            split_dates = [pd.to_datetime(split.execution_date)
                           for split in splits]
            split_ratios = [split.split_to /
                            split.split_from for split in splits]

            return split_dates, split_ratios

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(
                    f'  Retry {attempt + 1} for {ticker} splits in {wait_time}s... ({e})')
                time.sleep(wait_time)
            else:
                print(
                    f'✗ Failed to get splits for {ticker} after {max_retries} attempts: {e}')
                return [], []


def get_splits_data(tickers, batch_size=API_BATCH_SIZE,
                    batch_delay=API_BATCH_DELAY, request_delay=API_REQUEST_DELAY):
    '''Fetch stock split data for a list of tickers in rate-limited batches.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols.
    batch_size : int, optional
        Number of tickers per batch before applying a longer delay.
    batch_delay : float, optional
        Seconds to wait between batches.
    request_delay : float, optional
        Seconds to wait between individual requests.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['ticker', 'split_date', 'split_ratio'].
    '''
    all_splits = []

    print(f'Fetching splits for {len(tickers)} tickers...')
    print(
        f'Batch size: {batch_size}, Batch delay: {batch_delay}s, Request delay: {request_delay}s')
    print(f'{'=' * 50}\n')

    for i, ticker in enumerate(tickers, 1):
        split_dates, split_ratios = get_splits_for_ticker_api(ticker)

        for split_date, split_ratio in zip(split_dates, split_ratios):
            all_splits.append({
                'ticker': ticker,
                'split_date': split_date,
                'split_ratio': split_ratio
            })

        if i % 50 == 0:
            print(f'  Processed {i}/{len(tickers)} tickers...')

        time.sleep(request_delay)

        if i % batch_size == 0:
            print(f'  Batch complete, waiting {batch_delay}s...')
            time.sleep(batch_delay)

    splits_df = pd.DataFrame(all_splits)

    print(f'\n{'=' * 50}')
    print(
        f'✓ Collected {len(splits_df)} split records from {len(tickers)} tickers')
    print(f'{'=' * 50}')

    return splits_df


def create_backtest_data(tickers, benchmark, output_path, data_csv_dir):
    '''Build a dividend-adjusted CSV dataset for backtesting.

    Loads all CSV files from data_csv_dir, filters to the requested tickers
    and benchmark, fetches and applies dividend and split adjustments, then
    saves the result to output_path.

    Parameters
    ----------
    tickers : list of str
        Tickers for the strategy universe.
    benchmark : str
        Benchmark ticker to include (e.g. 'SPY').
    output_path : str or Path
        Destination path for the output CSV file.
    data_csv_dir : str or Path
        Directory containing the source daily CSV files.
    '''
    # Load all data
    df = load_all_csv_files(data_csv_dir)

    # Combine tickers and benchmark
    all_tickers = tickers + [benchmark]
    print(df)
    df = df[df['ticker'].isin(all_tickers)]

    # Apply adjustments
    dividends_df = get_dividends_data(all_tickers)
    splits_df = get_splits_data(all_tickers)
    dividends_df = adjust_dividends_for_splits(dividends_df, splits_df)
    df = apply_dividend_adjustment(df, dividends_df)

    # Save
    df.to_csv(output_path, index=False)
    print(f'Data saved to {output_path}')
    print(f'Tickers: {', '.join(all_tickers)}')
    print(f'Total rows: {len(df)}')


if __name__ == '__main__':
    DATA_PATH = ROOT_DIR / 'data' / DATA_CSV_DIR

    OUTPUT_PATH = ROOT_DIR / \
        'portfolio_management' / 'tests' / 'test_data' / 'AAPL_MSFT_SPY.csv'

    create_backtest_data(
        tickers=['AAPL', 'MSFT'],
        benchmark='SPY',
        output_path=OUTPUT_PATH,
        data_csv_dir=DATA_PATH
    )
