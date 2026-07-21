"""
Starter demo: synthetic data -> VolMIDAS-like HF volatility forecast -> simple Almgren-Chriss capacity sweep.
Run: python starter/volmidas_almgren_demo.py
Dependencies: numpy, pandas, scipy, statsmodels
This is a minimal, runnable skeleton for EXP-001. Replace synthetic data with real Parquet inputs.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
import datetime
import json
import os

RNG_SEED = 42
np.random.seed(RNG_SEED)

def make_synthetic_hf(n_days=90, freq_minutes=5):
    periods_per_day = int(24*60 / freq_minutes)
    n = n_days * periods_per_day
    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq=f"{freq_minutes}T")
    # simple stochastic volatility: latent vol follows AR(1)
    latent = np.zeros(n)
    latent[0] = 0.01
    for t in range(1, n):
        latent[t] = 0.98*latent[t-1] + 0.0001*np.random.randn()
    returns = np.sqrt(np.abs(latent)) * np.random.randn(n)
    df = pd.DataFrame({"timestamp": times, "return": returns})
    df.set_index("timestamp", inplace=True)
    return df

def aggregate_realized_variance(hf_df, agg_minutes=60):
    rv = hf_df['return'].pow(2).resample(f"{agg_minutes}T").sum()
    return rv.rename("realized_var")

def fit_garch_like(rv_series):
    # quick baseline: ARIMA(1,0,1) on realized variance as placeholder for GARCH/HAR
    model = ARIMA(rv_series.fillna(method='ffill'), order=(1,0,1))
    res = model.fit()
    return res

def volmidas_skeleton(hf_df, lf_series, midas_lag_days=30, agg_minutes=60):
    # Very small MIDAS-like feature: use lagged average of lf_series to predict HF realized var
    rv = aggregate_realized_variance(hf_df, agg_minutes=agg_minutes)
    # align lf_series (daily) to rv index by forward-fill
    lf_daily = lf_series.reindex(rv.index.date, method='ffill')
    lf_daily = pd.Series(lf_daily, index=rv.index)
    # simple weighted sum of last K days (uniform weights here as placeholder)
    K = midas_lag_days
    weights = np.ones(K) / K
    # build lagged feature matrix
    X = []
    for i in range(len(rv)):
        start = max(0, i-K+1)
        window = lf_daily.iloc[start:i+1].values
        if len(window) < K:
            window = np.pad(window, (K-len(window),0), 'edge')
        X.append(np.dot(weights, window))
    X = np.array(X)
    # simple linear regression to predict rv
    valid = ~np.isnan(rv.values)
    coef = np.linalg.lstsq(X[valid].reshape(-1,1), rv.values[valid], rcond=None)[0]
    preds = X.flatten() * coef[0]
    preds = pd.Series(preds, index=rv.index)
    return rv, preds

def almgren_chriss_impact(price, shares, temporary_coef, permanent_coef):
    # simple linear cost model: cost = temporary_coef * shares^2 + permanent_coef * shares
    return temporary_coef * shares**2 + permanent_coef * np.abs(shares)

def capacity_sweep(edge_series, base_aum=1e6, max_aum=1e9, steps=20, temp_coef=1e-6, perm_coef=1e-7):
    aums = np.geomspace(base_aum, max_aum, steps)
    results = []
    for aum in aums:
        # naive position sizing: allocate fixed fraction of AUM to strategy (e.g., 1%)
        position = 0.01 * aum
        # convert position to shares using synthetic price ~1
        shares = position / 1.0
        # compute slippage cost using almgren_chriss_impact
        cost = almgren_chriss_impact(1.0, shares, temp_coef, perm_coef)
        # assume expected gross edge is mean(edge_series) * position
        expected_gross = np.nanmean(edge_series) * position
        net = expected_gross - cost
        results.append({"aum": aum, "cost": cost, "expected_gross": expected_gross, "net": net})
    return pd.DataFrame(results)

def main():
    # 1) synthetic HF data
    hf = make_synthetic_hf(n_days=90, freq_minutes=5)
    # 2) synthetic LF driver (real yields) daily
    dates = pd.date_range(end=hf.index[-1].date(), periods=120, freq="D")
    real_yields = pd.Series(0.01 + 0.001*np.sin(np.linspace(0,10,len(dates))) + 0.0005*np.random.randn(len(dates)), index=dates)
    # 3) VolMIDAS skeleton
    rv, volmidas_pred = volmidas_skeleton(hf, real_yields, midas_lag_days=30, agg_minutes=60)
    # 4) Baseline ARIMA/GARCH-like
    garch_res = fit_garch_like(rv)
    garch_pred = garch_res.predict(start=rv.index[0], end=rv.index[-1])
    # 5) simple OOS metric: RMSE between preds and realized
    def rmse(a,b): return np.sqrt(np.nanmean((a-b)**2))
    rmse_volmidas = rmse(rv.values, volmidas_pred.values)
    rmse_garch = rmse(rv.values, garch_pred.values)
    print("RMSE VolMIDAS (skeleton):", rmse_volmidas)
    print("RMSE GARCH-like (ARIMA):", rmse_garch)
    # 6) capacity sweep using volmidas_pred as edge proxy (higher predicted vol -> larger scaling penalty)
    # create a synthetic edge series: negative of predicted vol (lower vol -> positive edge)
    edge_series = -volmidas_pred.fillna(method='ffill').pct_change().fillna(0).values
    cap_df = capacity_sweep(edge_series, base_aum=1e6, max_aum=1e8, steps=12,
                            temp_coef=1e-6, perm_coef=1e-7)
    print(cap_df.head())
    # 7) save artifacts
    out_dir = "experiments/EXP-001_artifacts"
    os.makedirs(out_dir, exist_ok=True)
    rv.to_csv(os.path.join(out_dir, "rv.csv"))
    volmidas_pred.to_csv(os.path.join(out_dir, "volmidas_pred.csv"))
    cap_df.to_csv(os.path.join(out_dir, "capacity_curve.csv"))
    # 8) write experiment card
    card = {
        "experiment_id": "EXP-001",
        "owner": "you",
        "description": "Starter VolMIDAS skeleton + capacity sweep on synthetic data",
        "data_hash": "synthetic",
        "config_path": "configs/default.yaml",
        "commit": "TODO:fill_commit_sha",
        "seed": RNG_SEED,
        "metrics": {
            "rmse_volmidas": float(rmse_volmidas),
            "rmse_garch": float(rmse_garch)
        },
        "artifacts": {
            "rv": os.path.join(out_dir, "rv.csv"),
            "volmidas_pred": os.path.join(out_dir, "volmidas_pred.csv"),
            "capacity_curve": os.path.join(out_dir, "capacity_curve.csv")
        }
    }
    with open(os.path.join(out_dir, "experiment_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    print("Artifacts saved to", out_dir)

if __name__ == "__main__":
    main()
