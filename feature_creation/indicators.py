from typing import List
import pandas as pd
import yfinance as yf
import talib


def get_all_candlestick_patterns() -> List[str]:
    return talib.get_function_groups()['Pattern Recognition']


def get_ohlc(df: pd.DataFrame) -> List[pd.Series]:
    open = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    return [open, high, low, close]


def get_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    patterns = get_all_candlestick_patterns()
    ohlc = get_ohlc(df)
    for pattern in patterns:
        df[pattern] = getattr(talib, pattern)(*ohlc)
    return df


def get_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA"] = talib.SMA(df['Close'].astype(float), timeperiod=3)
    df["MA"]  = talib.MA(df['Close'].astype(float), timeperiod=3)
    df["EMA"] = talib.EMA(df['Close'].astype(float), timeperiod=3)
    df["WMA"] = talib.WMA(df['Close'].astype(float), timeperiod=3)
    df["DEMA"] = talib.DEMA(df['Close'].astype(float), timeperiod=3)
    df["TEMA"] = talib.TEMA(df['Close'].astype(float), timeperiod=3)


def get_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    open, high, low, close = get_ohlc(df)
    volume = df['Volume'].astype(float)
    df['ADX'] = talib.ADX(high, low, close)
    df['ADXR'] = talib.ADXR(high, low, close)
    df['APO'] = talib.APO(close)
    df['AROON_UP'], df['AROON_DOWN'] = talib.AROON(high, low)
    df['AROONOSC'] = talib.AROONOSC(high, low)
    df['BOP'] = talib.BOP(open, high, low, close)
    df['CCI'] = talib.CCI(high, low, close)
    df['DX'] = talib.DX(high, low, close)
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(close)
    df['MFI'] = talib.MFI(high, low, close, volume)
    df['MINUS_DI'] = talib.MINUS_DI(high, low, close)
    df['MINUS_DM'] = talib.MINUS_DM(high, low)
    df['MOM'] = talib.MOM(close)
    df['PLUS_DI'] = talib.PLUS_DI(high, low, close)
    df['PLUS_DM'] = talib.PLUS_DM(high, low)
    df['PPO'] = talib.PPO(close)
    df['ROC'] = talib.ROC(close)
    df['ROCP'] = talib.ROCP(close)
    df['ROCR'] = talib.ROCR(close)
    df['RSI'] = talib.RSI(close)
    df['STOCH_SLOW_K'], df['STOCH_SLOW_D'] = talib.STOCH(high, low, close)
    df['STOCH_FAST_K'], df['STOCH_FAST_D'] = talib.STOCHF(high, low, close)
    df['STOCH_RSI_K'], df['STOCH_RSI_D'] = talib.STOCHRSI(close)
    df['TRIX'] = talib.TRIX(close)
    df['ULTOSC'] = talib.ULTOSC(high, low, close)
    df['WILLR'] = talib.WILLR(high, low, close)
    return df


if __name__ == "__main__":
    df = yf.download("AAPL", start="2023-01-01",
                     end="2024-10-01", interval='1h')
    df = get_candlestick_patterns(df)
    df = get_momentum_indicators(df)
    print(df)
