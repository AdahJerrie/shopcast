"""Flag days whose sales are far from what TabPFN expected for that day."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from shopcast.features import FEATURES, build_features, training_rows
from shopcast.forecast import make_regressor


def find_anomalies(
    history: pd.DataFrame, threshold: float = 3.0, min_pct: float = 25.0, folds: int = 5
) -> pd.DataFrame:
    """Score every day with a model that did not see it, then flag days that are both
    statistically unusual (robust z-score) and big enough to matter to the owner (`min_pct`).

    Returns all scored days with `expected`, `residual`, `z` and `is_anomaly`.
    """
    rows = training_rows(build_features(history))
    X, y = rows[FEATURES], rows["sales"]
    expected = pd.Series(np.nan, index=rows.index)

    for train_idx, test_idx in KFold(folds, shuffle=True, random_state=0).split(X):
        model = make_regressor()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        expected.iloc[test_idx] = np.asarray(model.predict(X.iloc[test_idx]), dtype=float)

    residual = y - expected
    mad = float(np.median(np.abs(residual - residual.median()))) or 1.0
    z = (residual - residual.median()) / (1.4826 * mad)

    out = rows[["date", "sales", "promo"]].copy()
    out["expected"] = expected.round(0)
    out["residual"] = residual.round(0)
    out["z"] = z.round(2)
    pct = (residual / expected.where(expected > 0)).abs() * 100
    out["is_anomaly"] = (z.abs() > threshold) & (pct >= min_pct)
    return out.reset_index(drop=True)


def anomaly_table(scored: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    flagged = scored[scored["is_anomaly"]].copy()
    flagged["direction"] = np.where(flagged["residual"] > 0, "spike", "drop")
    flagged["pct_vs_expected"] = (flagged["residual"] / flagged["expected"].replace(0, np.nan) * 100).round(0)
    return (
        flagged.reindex(flagged["z"].abs().sort_values(ascending=False).index)
        .head(top_n)[["date", "sales", "expected", "pct_vs_expected", "direction", "promo"]]
        .reset_index(drop=True)
    )
