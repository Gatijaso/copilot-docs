import numpy as np
import pandas as pd


def lagged_returns(price: pd.Series, lags: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({f"ret_lag_{i}": price.pct_change().shift(i) for i in range(1, lags + 1)})
    return df


def rolling_features(price: pd.Series, window: int = 20) -> pd.DataFrame:
    ret = price.pct_change()
    df = pd.DataFrame()
    df[f"roll_vol_{window}"] = ret.rolling(window).std()
    df[f"roll_mom_{window}"] = ret.rolling(window).mean()
    df[f"roll_z_{window}"] = (ret - ret.rolling(window).mean()) / (ret.rolling(window).std() + 1e-12)
    return df


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)


def make_features(price: pd.Series, lags: int = 5, roll_window: int = 20) -> pd.DataFrame:
    df = pd.DataFrame(index=price.index)
    df = pd.concat([df, lagged_returns(price, lags)], axis=1)
    df = pd.concat([df, rolling_features(price, roll_window)], axis=1)
    df[f"rsi_{roll_window}"] = rsi(price, window=roll_window)
    df = df.dropna()
    return df
