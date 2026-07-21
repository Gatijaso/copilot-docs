"""Run a small demo: fetch GLD, fit models, and print forecasts and a tiny backtest.

This demo now uses log-price ARIMA forecasts, Student-t GARCH volatility forecasts,
and plots results for quick inspection.
"""
from src.gold_model.data_fetch import fetch_gld
from src.gold_model.models import (
    fit_arima,
    arima_price_forecast,
    fit_garch,
    garch_vol_forecast,
    ml_baseline,
)
from src.gold_model.backtest import naive_signal_from_forecast, backtest_long_only
from src.gold_model.features import make_features
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    print("Fetching GLD price data...")
    df = fetch_gld(period="2y")
    price = df["price"]
    returns = df["log_ret"].dropna()

    print("Fitting ARIMA on price (log-level approximation)...")
    arima = fit_arima(price)
    ar_price_forecast = arima_price_forecast(arima, steps=5)
    print("ARIMA price forecast (next 5):")
    print(ar_price_forecast)

    print("Fitting GARCH on returns...")
    garch = fit_garch(returns)
    print(garch.summary().tables[0])
    vol_forecast = garch_vol_forecast(garch, horizon=5)
    print("GARCH vol forecast (next 5, decimal):")
    print(vol_forecast)

    # Build a simple signal from ARIMA: forecasted next price vs last price
    last_price = price.dropna().iloc[-1]
    ar_forecast_return = ar_price_forecast / last_price - 1
    # align forecast index roughly after the last date
    forecast_index = pd.date_range(price.index[-1], periods=len(ar_forecast_return) + 1, freq="B")[1:]
    ar_forecast_return.index = forecast_index

    # Signal: long if predicted next return positive
    signal = (ar_forecast_return > 0).reindex(price.index, method="ffill").fillna(0).astype(int)

    # Use GARCH vol forecast to scale positions (simple heuristic)
    vol_series = pd.Series(index=price.index)
    vol_series.iloc[-len(vol_forecast) :] = vol_forecast.values

    bt = backtest_long_only(price, signal, cost=0.0005, scale_by_vol=vol_series)
    print("Backtest summary (last rows):")
    print(bt.tail())

            # --- save artifacts to experiments/EXP-001_artifacts ---
    import os
    import json

    art_dir = "experiments/EXP-001_artifacts"
    os.makedirs(art_dir, exist_ok=True)

    # save backtest (returns + positions)
    bt.to_csv(os.path.join(art_dir, "rv.csv"))

    # save vol forecast (GARCH)
    vol_forecast.to_csv(os.path.join(art_dir, "volmidas_pred.csv"), header=True)

    # save ARIMA price forecast
    ar_price_forecast.to_csv(os.path.join(art_dir, "ar_price_forecast.csv"), header=True)

    # minimal capacity_curve placeholder
    capacity_df = pd.DataFrame({"horizon": range(1, len(vol_forecast) + 1), "vol": vol_forecast.values})
    capacity_df.to_csv(os.path.join(art_dir, "capacity_curve.csv"), index=False)

    # write a small experiment card with metadata
    experiment_card = {
        "experiment": "EXP-001",
        "script": "run_demo.py",
        "notes": "Demo run: ARIMA + GARCH + ML baseline",
        "artifacts": ["rv.csv", "volmidas_pred.csv", "ar_price_forecast.csv", "capacity_curve.csv"],
        "commit": None
    }
    with open(os.path.join(art_dir, "experiment_card.json"), "w") as f:
        json.dump(experiment_card, f, indent=2)
    # --- end save block ---

    # quick plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(price.index, price.values, label="price")
    axes[0].plot(forecast_index, ar_price_forecast.values, marker="o", linestyle="--", label="ARIMA forecast")
    axes[0].legend()
    axes[1].plot(bt.index, bt["cum"], label="cumulative")
    axes[1].legend()
    axes[2].plot(bt.index, bt["drawdown"], label="drawdown")
    axes[2].legend()
    plt.tight_layout()
    plt.show()

    # --- ML baseline using RandomForest ---
    print("Running ML baseline (RandomForest) using engineered features...")
    feats = make_features(price)
    # target: next-day return
    target = price.pct_change().shift(-1).reindex(feats.index)
    data = feats.join(pd.Series(target, name="target"))
    data = data.dropna()
    if len(data) < 20:
        print("Not enough data for ML baseline; skipping.")
        return

    # train/test split (80/20)
    n = len(data)
    split = int(n * 0.8)
    X_train = data.iloc[:split].drop(columns=["target"])
    y_train = data.iloc[:split]["target"]
    X_test = data.iloc[split:].drop(columns=["target"])
    y_test = data.iloc[split:]["target"]

    model = ml_baseline(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(((preds - y_test.values) ** 2).mean())
    print(f"ML baseline RMSE (test): {rmse:.6f}")
    # feature importances (if available)
    try:
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        print("Top features:\n", feat_imp.head())
    except Exception:
        pass


if __name__ == "__main__":
    main()


