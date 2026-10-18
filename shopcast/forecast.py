"""Forecast daily sales with TabPFN, walking forward one week per API call."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shopcast.features import BLOCK, FEATURES, build_features, training_rows

QUANTILES = [0.1, 0.5, 0.9]


def backend() -> str:
    return os.getenv("FORECAST_BACKEND", "tabpfn").lower()


def make_regressor():
    if backend() == "sklearn":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=5)

    from tabpfn_client import TabPFNRegressor, set_access_token

    token = os.getenv("TABPFN_TOKEN")
    if not token:
        raise RuntimeError("Set TABPFN_TOKEN (from platform.priorlabs.ai) or FORECAST_BACKEND=sklearn.")
    set_access_token(token)
    return TabPFNRegressor()


def predict_with_band(model, X: pd.DataFrame) -> pd.DataFrame:
    """Median plus a 10-90% band when the model supports quantiles, else a point prediction."""
    if backend() != "sklearn":
        try:
            q = model.predict(X, output_type="quantiles", quantiles=QUANTILES)
            low, mid, high = (np.asarray(v, dtype=float) for v in q)
            return pd.DataFrame({"forecast": mid, "low": low, "high": high}, index=X.index)
        except (TypeError, ValueError):
            pass
    mid = np.asarray(model.predict(X), dtype=float)
    return pd.DataFrame({"forecast": mid, "low": np.nan, "high": np.nan}, index=X.index)


@dataclass
class Forecast:
    frame: pd.DataFrame  # date, forecast, low, high, promo

    @property
    def total(self) -> float:
        return float(self.frame["forecast"].sum())


def forecast(
    history: pd.DataFrame,
    days: int = 28,
    promo_dates: set[pd.Timestamp] | None = None,
) -> Forecast:
    """Fit on all history, then predict `days` ahead in 7-day blocks.

    Each block only needs lags of 7+ days, so predictions from the previous
    block feed the next one.
    """
    origin = history["date"].min()
    feat = build_features(history, origin)
    train = training_rows(feat)
    model = make_regressor()
    model.fit(train[FEATURES], train["sales"])

    promo_dates = promo_dates or set()
    future_dates = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=days, freq="D")
    future = pd.DataFrame(
        {
            "date": future_dates,
            "sales": np.nan,
            "promo": [int(d in promo_dates) for d in future_dates],
        }
    )

    combined = pd.concat([history[["date", "sales", "promo"]], future], ignore_index=True)
    n_hist = len(history)
    parts = []
    for start in range(0, days, BLOCK):
        block = slice(n_hist + start, n_hist + min(start + BLOCK, days))
        feat = build_features(combined, origin)
        preds = predict_with_band(model, feat.iloc[block][FEATURES])
        combined.loc[preds.index, "sales"] = preds["forecast"].clip(lower=0)
        parts.append(preds)

    result = pd.concat(parts)
    result.insert(0, "date", combined.loc[result.index, "date"].values)
    result["promo"] = combined.loc[result.index, "promo"].values
    for col in ("forecast", "low", "high"):
        result[col] = result[col].clip(lower=0)
    return Forecast(result.reset_index(drop=True))


def seasonal_naive(history: pd.DataFrame, days: int) -> np.ndarray:
    """Baseline: repeat the last observed week."""
    last_week = history["sales"].to_numpy()[-BLOCK:]
    return np.resize(last_week, days)


def backtest(history: pd.DataFrame, days: int = 28) -> dict:
    """Hold out the last `days` days and compare TabPFN against the seasonal naive baseline."""
    train, test = history.iloc[:-days], history.iloc[-days:]
    fc = forecast(train, days, promo_dates=set(test.loc[test["promo"] == 1, "date"]))
    actual = test["sales"].to_numpy()
    model_pred = fc.frame["forecast"].to_numpy()
    naive_pred = seasonal_naive(train, days)

    def mae(p):
        return float(np.mean(np.abs(actual - p)))

    def mape(p):
        mask = actual > 0
        return float(np.mean(np.abs((actual[mask] - p[mask]) / actual[mask])) * 100)

    frame = pd.DataFrame(
        {"date": test["date"].values, "actual": actual, "model": model_pred, "naive": naive_pred}
    )
    model_mae, naive_mae = mae(model_pred), mae(naive_pred)
    return {
        "days": days,
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "model_mape": mape(model_pred),
        "naive_mape": mape(naive_pred),
        "improvement_pct": (1 - model_mae / naive_mae) * 100 if naive_mae else 0.0,
        "frame": frame,
    }
