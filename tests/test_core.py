import os
import sys
from pathlib import Path

# Ensure project root (parent of `src`) is on sys.path so `import src.*` works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.gold_model.features import make_features
from src.gold_model.backtest import backtest_long_only


def test_make_features_basic():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    price = pd.Series(100 + np.cumsum(np.random.randn(len(idx))), index=idx)
    feats = make_features(price)
    assert not feats.empty
    assert any(col.startswith("ret_lag_") for col in feats.columns)


def test_backtest_basic():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    price = pd.Series(100 + np.cumsum(np.random.randn(10)), index=idx)
    signal = pd.Series([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], index=idx)
    out = backtest_long_only(price, signal, cost=0.0)
    assert "cum" in out.columns
    assert out["cum"].iloc[0] >= 0.0
