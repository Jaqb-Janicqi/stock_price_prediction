from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import os
import multiprocessing as mp


def download_ticker(ticker_name: str, interval: str, cols=['Close']) -> pd.DataFrame:
    # Get today's date
    today = datetime.now().date()
    # Check if ticker name is forex
    if 'USD' in ticker_name:
        ticker_name += '=X'

    # Calculate start and end dates, download data (yfinance allows a maximum of 730 days from the current date)
    start = today - timedelta(days=729)
    end = today  # - timedelta(days=1)
    while True:
        data: pd.DataFrame = yf.download(
            ticker_name, start=start, end=end, interval=interval)[cols]
        if data is not None and len(data) > 0:
            break
        error_message = yf.shared._ERRORS.get(f'{ticker_name}=X', None)
        if error_message is not None:
            split_string = error_message.split()
            split_string.reverse()
            for item in split_string:
                if item.isdigit():
                    start = today - timedelta(days=int(item) - 1)

    data.columns = cols
    print(f'Downloaded {len(data)} rows for {ticker_name}')
    return data


def download_worker(ticker: str, interval: str, split: bool = False, cols=['Close']):
    try:
        data = download_ticker(ticker, interval, cols)

        # Define file path
        file_path = f'data/sp500/{ticker}_{interval}.csv'

        if os.path.exists(file_path):
            # Load existing data
            existing_data = pd.read_csv(
                file_path, index_col=0, parse_dates=True)
            # Append only new rows to the existing data
            new_data = data[~data.index.isin(existing_data.index)]
            combined_data = pd.concat([existing_data, new_data])
            combined_data.to_csv(file_path)
            print(f'Appended {len(new_data)} new rows to {ticker}.')
        else:
            # If the file does not exist, save the new data
            data.to_csv(file_path)
            print(f'Saved new data for {ticker}.')

        # Handle splitting if required
        if split:
            if os.path.exists(file_path):
                # Load existing data for splitting
                combined_data = pd.read_csv(
                    file_path, index_col=0, parse_dates=True)
                train = combined_data.iloc[:int(len(combined_data) * 0.8)]
                val = combined_data.iloc[int(
                    len(combined_data) * 0.8):int(len(combined_data) * 0.9)]
                test = combined_data.iloc[int(len(combined_data) * 0.9):]

                # Save split data
                train.to_csv(f'data/sp500train/{ticker}_{interval}.csv')
                val.to_csv(f'data/sp500val/{ticker}_{interval}.csv')
                test.to_csv(f'data/sp500test/{ticker}_{interval}.csv')
            else:
                # If no existing data, split the newly downloaded data
                train = data.iloc[:int(len(data) * 0.8)]
                val = data.iloc[int(len(data) * 0.8):int(len(data) * 0.9)]
                test = data.iloc[int(len(data) * 0.9):]
                train.to_csv(f'data/sp500train/{ticker}_{interval}.csv')
                val.to_csv(f'data/sp500val/{ticker}_{interval}.csv')
                test.to_csv(f'data/sp500test/{ticker}_{interval}.csv')
    except Exception as e:
        print(f'Error downloading {ticker}: {e}')



def get_sp500_ticker_names() -> list:
    sp500 = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500['Symbol'].to_list()


def download_tickers(selected_tickers=get_sp500_ticker_names(), interval='1h', split=True) -> None:
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/sp500'):
        os.makedirs('data/sp500')
    if not os.path.exists('data/sp500train'):
        os.makedirs('data/sp500train')
    if not os.path.exists('data/sp500val'):
        os.makedirs('data/sp500val')
    if not os.path.exists('data/sp500test'):
        os.makedirs('data/sp500test')

    cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Download and save data for selected tickers
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(download_worker,
                 [(ticker, interval, split, cols) for ticker in selected_tickers])
    pool.close()
    pool.join()


if __name__ == '__main__':
    # Example usage; replace with your actual ticker selection
    selected_tickers = ['AAPL', 'MSFT']  # Add your tickers here
    download_tickers()
