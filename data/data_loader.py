import boto3
import datetime
import time
import json
import sys

import pandas as pd

from pathlib import Path
from botocore.config import Config

from data.config import *
from massive import RESTClient

from data.utils import unzip_csv_gz_to_directory
from data.data_processor import *

def create_s3_session():
    """Create and return boto3 session and s3 client"""
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
    print(f'{'='*50}\n')
    
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
    
    print(f'\n{'='*50}')
    print(f'Summary:')
    print(f'  Successfully downloaded: {len(downloaded_files)} files')
    print(f'{'='*50}')
    
    if failed_dates:
        print(f'\nFailed dates: {failed_dates[:10]}{'...' if len(failed_dates) > 10 else ''}')
    
    return downloaded_files


def load_all_csv_files(directory, pattern="*.csv"):
    directory = Path(directory)
    csv_files = sorted(directory.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files in {directory}")
    print(f"Loading files...")
    
    df_list = []
    
    for i, file in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file)
            
            date_str = file.stem
            df['date'] = pd.to_datetime(date_str)
            
            df_list.append(df)
            
            if i % 50 == 0:
                print(f"  Loaded {i}/{len(csv_files)} files...")
                
        except Exception as e:
            pass
    
    print(f"Combining {len(df_list)} dataframes...")
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print(f"\n{'='*50}")
    print(f"{'='*50}")
    
    return combined_df


def get_dividends_for_ticker(ticker, max_retries=API_MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            client = RESTClient(AWS_SECRET_ACCESS_KEY)

            dividends = []
            for d in client.list_stocks_dividends(
                ticker=f'{ticker}',
                limit="100",
                sort="ticker.asc",
                ):
                dividends.append(d)
            
            ex_dates = [div.ex_dividend_date for div in dividends]
            dividends = [div.cash_amount for div in dividends]

            return ex_dates, dividends
                    
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Retry {attempt + 1} for {ticker} in {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                print(f"✗ Failed to get dividends for {ticker} after {max_retries} attempts: {e}")
                return [], []



def get_dividends_data(tickers, batch_size=API_BATCH_SIZE, batch_delay=API_BATCH_DELAY, request_delay=API_REQUEST_DELAY):
    all_divs = []
    
    print(f"Fetching dividends for {len(tickers)} tickers...")
    print(f"Batch size: {batch_size}, Batch delay: {batch_delay}s, Request delay: {request_delay}s")
    print(f"{'='*50}\n")
    
    for i, ticker in enumerate(tickers, 1):
        ex_dates, dividends = get_dividends_for_ticker(ticker)
        
        for ex_dividend_date, dividend in zip(ex_dates, dividends):
            all_divs.append({
                'ticker': ticker,
                'ex_dividend_date': ex_dividend_date,
                'dividend_amount': dividend
            })
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(tickers)} tickers...")
        
        time.sleep(request_delay)
        
        if i % batch_size == 0:
            print(f"  Batch complete, waiting {batch_delay}s...")
            time.sleep(batch_delay)
    
    dividends_df = pd.DataFrame(all_divs)
    
    if len(dividends_df) > 0:
        dividends_df['ex_dividend_date'] = pd.to_datetime(dividends_df['ex_dividend_date'])
    
    print(f"\n{'='*50}")
    print(f"✓ Collected {len(dividends_df)} dividend records from {len(tickers)} tickers")
    print(f"{'='*50}")
    
    return dividends_df


def get_splits_for_ticker_api(ticker, max_retries=API_MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            client = RESTClient(AWS_SECRET_ACCESS_KEY)

            splits = []
            for s in client.list_stocks_splits(
                ticker=ticker,
                limit="100",
                sort="execution_date.desc",
            ):
                splits.append(s)
            
            split_dates = [pd.to_datetime(split.execution_date) for split in splits]
            split_ratios = [split.split_to / split.split_from for split in splits]
            
            return split_dates, split_ratios
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Retry {attempt + 1} for {ticker} splits in {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                print(f"✗ Failed to get splits for {ticker} after {max_retries} attempts: {e}")
                return [], []


def get_splits_data(tickers, batch_size=API_BATCH_SIZE, batch_delay=API_BATCH_DELAY, request_delay=API_REQUEST_DELAY):
    all_splits = []
    
    print(f"Fetching splits for {len(tickers)} tickers...")
    print(f"Batch size: {batch_size}, Batch delay: {batch_delay}s, Request delay: {request_delay}s")
    print(f"{'='*50}\n")
    
    for i, ticker in enumerate(tickers, 1):
        split_dates, split_ratios = get_splits_for_ticker_api(ticker)
        
        for split_date, split_ratio in zip(split_dates, split_ratios):
            all_splits.append({
                'ticker': ticker,
                'split_date': split_date,
                'split_ratio': split_ratio
            })
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(tickers)} tickers...")
        
        time.sleep(request_delay)
        
        if i % batch_size == 0:
            print(f"  Batch complete, waiting {batch_delay}s...")
            time.sleep(batch_delay)
    
    splits_df = pd.DataFrame(all_splits)
    
    print(f"\n{'='*50}")
    print(f"✓ Collected {len(splits_df)} split records from {len(tickers)} tickers")
    print(f"{'='*50}")
    
    return splits_df

if __name__ == '__main__':
    # download_files = download_stock_data(START_DATE, END_DATE, output_dir=DATA_GZ_DIR)
    # unzipped_files = unzip_csv_gz_to_directory(DATA_GZ_DIR, DATA_CSV_DIR)
    df = load_all_csv_files(DATA_CSV_DIR)

    tickers = ["AAPL", "MSFT"]
    df = df[df['ticker'].isin(tickers)]

    print(df)
    dividends_df = get_dividends_data(tickers)
    splits_df = get_splits_data(tickers)

    dividends_df = adjust_dividends_for_splits(dividends_df, splits_df)
    df = apply_dividend_adjustment(df, dividends_df)
    
    df.to_csv("AAPL_MSFT.csv", index=False)