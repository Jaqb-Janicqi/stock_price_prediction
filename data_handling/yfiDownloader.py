from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import os
import multiprocessing as mp


def download_close(ticker_name: str, interval: str) -> pd.DataFrame:
    # Get today's date
    today = datetime.now().date()
    # check if ticker name is forex
    if 'USD' in ticker_name:
        ticker_name += '=X'

    # Calculate start and end dates, download data (yfinance allows at max 730 days from current date)
    start = today - timedelta(days=729)
    end = today  # - timedelta(days=1)
    while True:
        data: pd.Series = yf.download(ticker_name, start=start,
                                      end=end, interval=interval)['Close']
        if data is not None and len(data) > 0:
            break
        error_message = yf.shared._ERRORS[f'{ticker_name}=X']
        if error_message is not None:
            split_string = error_message.split()
            split_string.reverse()
            for item in split_string:
                if item.isdigit():
                    start = today - timedelta(days=int(item) - 1)

    print(f'Downloaded {len(data)} rows for {ticker_name}')
    return data.to_frame()


def download_ticker(ticker_name: str, interval: str) -> pd.DataFrame:
    # Get today's date
    today = datetime.now().date()
    # check if ticker name is forex
    if 'USD' in ticker_name:
        ticker_name += '=X'

    # Calculate start and end dates, download data (yfinance allows at max 730 days from current date)
    start = today - timedelta(days=729)
    end = today  # - timedelta(days=1)
    while True:
        data: pd.Series = yf.download(ticker_name, start=start,
                                      end=end, interval=interval)
        if data is not None and len(data) > 0:
            break
        error_message = yf.shared._ERRORS[f'{ticker_name}=X']
        if error_message is not None:
            split_string = error_message.split()
            split_string.reverse()
            for item in split_string:
                if item.isdigit():
                    start = today - timedelta(days=int(item) - 1)

    print(f'Downloaded {len(data)} rows for {ticker_name}')
    return data


def download_worker(ticker: str, interval: str, split: bool = False):
    try:
        data = download_ticker(ticker, interval)
        data.to_csv(f'data/sp500/{ticker}_{interval}.csv')
        if split:
            train = data.iloc[:int(len(data) * 0.7)]
            val = data.iloc[int(len(data) * 0.7):int(len(data) * 0.9)]
            test = data.iloc[int(len(data) * 0.9):]
            train.to_csv(f'data/sp500train/{ticker}_{interval}.csv')
            val.to_csv(f'data/sp500val/{ticker}_{interval}.csv')
            test.to_csv(f'data/sp500test/{ticker}_{interval}.csv')
    except Exception as e:
        print(f'Error downloading {ticker}: {e}')


def download_sp500(interval: str, split: bool) -> pd.DataFrame:
    # Get the list of tickers in the S&P 500
    sp500 = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].to_list()

    if not os.path.exists('data'):
        os.makedirs('data')
    # Create a directory to save the data
    if not os.path.exists('data/sp500'):
        os.makedirs('data/sp500')
    if not os.path.exists('data/sp500train'):
        os.makedirs('data/sp500train')
    if not os.path.exists('data/sp500val'):
        os.makedirs('data/sp500val')
    if not os.path.exists('data/sp500test'):
        os.makedirs('data/sp500test')

    # Download and save data for all tickers in the S&P 500
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(download_worker, [
                 (ticker, interval, split) for ticker in tickers])
    pool.close()
    pool.join()


if __name__ == '__main__':
    download_sp500(interval='1h', split=True)
