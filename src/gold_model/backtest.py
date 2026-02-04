
import pandas as pd


def naive_signal_from_forecast(forecast: pd.Series) -> pd.Series:
    """Return a simple long-if-positive signal from a forecast series."""
    return (forecast > 0).astype(int)


def backtest_long_only(
    price: pd.Series,
    signal: pd.Series,
    cost: float = 0.0005,
    scale_by_vol: pd.Series | None = None,
    size_cap: float = 1.0,
) -> pd.DataFrame:
    """Compute backtest metrics for a naive long-only strategy.

    Returns a DataFrame with columns: `ret`, `position`, `pnl`, `cum`, `drawdown`.

    - `signal` is the desired position (0/1) for the day.
    - `ret` is next-day return (aligned so that today's position captures tomorrow's return).
    - `cost` is per-change transaction cost applied to absolute position changes.
    - `scale_by_vol` if provided scales the position by 1/(1+vol).
    """
    ret = price.pct_change().shift(-1)

    position = signal.reindex(price.index).fillna(0).astype(float)
    if scale_by_vol is not None:
        # ensure series-like and scale positions by inverse vol (simple heuristic)
        if not isinstance(scale_by_vol, pd.Series):
            vol = pd.Series(scale_by_vol, index=price.index)
        else:
            vol = scale_by_vol
        vol = vol.reindex(price.index).ffill().fillna(0)
        scale = 1.0 / (1.0 + vol)
        position = (position * scale).clip(0, size_cap)

    # P&L: position captures next-day returns
    pnl = position * ret
    # transaction costs applied to changes in position
    cost_series = cost * position.diff().abs().fillna(0)
    pnl = pnl - cost_series
    pnl = pnl.fillna(0)

    cum = (1 + pnl).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak

    out = pd.DataFrame({"ret": ret, "position": position, "pnl": pnl, "cum": cum, "drawdown": drawdown})
    return out


if __name__ == "__main__":
    import numpy as np

    idx = pd.date_range("2020-01-01", periods=10)
    price = pd.Series(100 + np.cumsum(np.random.randn(10)), index=idx)
    signal = pd.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], index=idx)
    print(backtest_long_only(price, signal))
