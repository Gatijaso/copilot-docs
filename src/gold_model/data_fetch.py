import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def fetch_gld(
    ticker: str = "GLD",
    period: str = "5y",
    cache_path: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices for the ticker and compute log returns.

    Parameters:
    - ticker, period: passed to yfinance.
    - cache_path: optional path to a parquet file to save/read the resulting
      DataFrame (price + log_ret). If None a default under `.cache/` is used.
    - force_refresh: if True, ignore existing cache and re-download.

    Returns a DataFrame with columns ['price', 'log_ret'] indexed by date.
    """
    import yfinance as yf

    # resolve cache location
    if cache_path is None:
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        cache = cache_dir / f"gld-{ticker}-{period}.parquet"
    else:
        cache = Path(cache_path)

    # return cached copy when available
    if cache is not None and cache.exists() and not force_refresh:
        try:
            df_cached = pd.read_parquet(cache)
            # ensure index is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_cached.index):
                df_cached.index = pd.to_datetime(df_cached.index)
            return df_cached
        except Exception:
            # fall back to re-downloading
            pass

    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"no data for {ticker}")

    # prefer Adjusted Close, fall back to Close or the last numeric column available
    col = None
    for candidate in ("Adj Close", "Adj_Close", "Close"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            raise RuntimeError("no numeric price column found")
        col = cols[-1]

    series = df[col]
    if isinstance(series, pd.DataFrame) and series.shape[1] == 1:
        series = series.iloc[:, 0]

    # ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        series.index = pd.to_datetime(series.index)

    # infer frequency and set it to avoid ARIMA warnings; fall back to business day
    freq = pd.infer_freq(series.index)
    if freq:
        series = series.asfreq(freq)
    else:
        series = series.asfreq("B")

    s = series.rename("price").to_frame()

    # compute log returns from price (log price diff)
    s["log_ret"] = np.log(s["price"]).diff()
    s = s.dropna()

    # cache result (best-effort)
    try:
        if cache is not None:
            s.to_parquet(cache)
    except Exception:
        pass

    return s


if __name__ == "__main__":
    df = fetch_gld(period="1y")
    print(df.tail())

