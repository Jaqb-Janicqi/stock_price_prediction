from typing import List
import pandas as pd
import yfinance as yf
import talib


def get_all_candlestick_patterns() -> List[str]:
    return talib.get_function_groups()['Pattern Recognition']


def get_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    open = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)

    patterns = get_all_candlestick_patterns()
    for pattern in patterns:
        df[pattern] = getattr(talib, pattern)(open, high, low, close)
    return df


if __name__ == "__main__":
    df = yf.download("AAPL", start="2023-01-01",
                     end="2024-10-01", interval='1h')
    df = get_candlestick_patterns(df)
    print(df.head())
