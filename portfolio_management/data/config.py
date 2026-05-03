import datetime

AWS_ACCESS_KEY_ID = 'ff6d4f71-9a25-48aa-bb53-f44611f0f228'
AWS_SECRET_ACCESS_KEY = 'mDeRwu7LQPe1u63NIarRkOjcmWAkergT'
S3_ENDPOINT_URL = 'https://files.massive.com'
BUCKET_NAME = 'flatfiles'
PREFIX = 'us_stocks_sip'

START_DATE = datetime.date(2021, 1, 1)
END_DATE = datetime.date(2026, 1, 1)

DATA_GZ_DIR = 'data_gz'
DATA_CSV_DIR = 'data_csv'

STOCKS_METADATA_FILE = 'stocks_metadata.json'

RETURNS_PERIODS = {1: '1d', 5: '1w', 21: '1m'}
VOLS_PERIODS = {5: '1w', 21: '1m', 63: '3m'}

MOMENTUM_WINDOW = 21
BETA_WINDOW = 21

API_BATCH_SIZE = 10
API_BATCH_DELAY = 1
API_REQUEST_DELAY = 0.1
API_MAX_RETRIES = 3
