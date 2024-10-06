from typing import List
import pandas as pd
import yfinance as yf
import talib


def get_all_candlestick_patterns() -> List[str]:
    return talib.get_function_groups()['Pattern Recognition']


def get_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    op = df['Open'].astype(float)
    hi = df['High'].astype(float)
    lo = df['Low'].astype(float)
    cl = df['Close'].astype(float)

    patterns = get_all_candlestick_patterns()
    for pattern in patterns:
        df[pattern] = getattr(talib, pattern)(op, hi, lo, cl)
    return df


if __name__ == "__main__":
    df = yf.download("AAPL", start="2023-01-01",
                     end="2024-10-01", interval='1h')
    df = get_candlestick_patterns(df)
    print(df.head())
