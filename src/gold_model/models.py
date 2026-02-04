import numpy as np
import pandas as pd


def fit_arima(series: pd.Series, order=(1, 1, 1), use_log: bool = True):
    """Fit ARIMA to a series and return the fitted model object.

    If `use_log` is True, the model is fitted on `np.log(series)` and forecasts
    are returned in log-space (use `arima_price_forecast` to map back to price).
    """
    from statsmodels.tsa.arima.model import ARIMA

    if use_log:
        s = np.log(series.dropna())
    else:
        s = series.dropna()

    model = ARIMA(s, order=order).fit()
    return model


def arima_price_forecast(arima_result, steps: int = 5) -> pd.Series:
    """Return ARIMA forecast mapped back to price space (exp if log was used).

    `arima_result` should be the fitted result returned by `fit_arima`.
    """
    pred = arima_result.get_forecast(steps=steps)
    mean = pred.predicted_mean
    # assume model was fit on log prices; map back to price space
    try:
        price_forecast = np.exp(mean)
    except Exception:
        price_forecast = mean
    return pd.Series(price_forecast)


def fit_garch(returns: pd.Series, dist: str = "t"):
    """Fit a GARCH(1,1) to returns (assumes returns in decimal form).

    Returns the fitted arch model result. Uses Student-t by default.
    """
    from arch import arch_model

    # arch often expects percent-style input; scale by 100
    am = arch_model(returns.dropna() * 100, vol="Garch", p=1, q=1, dist=dist)
    res = am.fit(disp="off")
    return res


def garch_vol_forecast(garch_result, horizon: int = 5) -> pd.Series:
    """Return volatility forecast (decimal) for the next `horizon` periods.

    The returned series is in the same units as the original returns (decimal).
    """
    # res.forecast returns variances in the same scale as the model input (percent^2)
    f = garch_result.forecast(horizon=horizon)
    # take the last available variance forecast row
    var = f.variance.iloc[-1]
    vol_percent = np.sqrt(var)
    # convert back from percent to decimal
    vol_decimal = vol_percent / 100.0
    return pd.Series(vol_decimal.values, index=var.index)


def ml_baseline(X: pd.DataFrame, y: pd.Series):
    """Train a small RandomForest baseline and return the model."""
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    return model


if __name__ == "__main__":
    # quick smoke run
    import pandas as pd

    s = pd.Series(np.random.randn(500).cumsum())
    ar = fit_arima(s)
    print(ar.summary().tables[0])
