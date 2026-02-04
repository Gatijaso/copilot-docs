import numpy as np
import pandas as pd

from src.gold_model.models import fit_arima, fit_garch, garch_vol_forecast


def test_fit_arima_smoke():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    # create an upward trending series with noise
    price = pd.Series(100 + np.linspace(0, 5, len(idx)) + np.random.randn(len(idx)), index=idx)
    ar = fit_arima(price)
    assert hasattr(ar, "predict") or hasattr(ar, "get_forecast")


def test_fit_garch_smoke():
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    # simulate returns
    rets = pd.Series(np.random.randn(len(idx)) * 0.01, index=idx)
    g = fit_garch(rets)
    vf = garch_vol_forecast(g, horizon=3)
    assert len(vf) == 3
