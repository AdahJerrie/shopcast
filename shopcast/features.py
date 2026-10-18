"""Load a shop's sales CSV and turn it into a tabular feature matrix for TabPFN.

Lags start at 7 days so the next week can be predicted in a single call; the
forecaster then walks forward one week at a time.
"""

from __future__ import annotations

import pandas as pd

DATE_ALIASES = ("date", "day", "sale_date", "transaction_date")
SALES_ALIASES = ("sales", "revenue", "amount", "total", "turnover")

FEATURES = [
    "trend",
    "dayofweek",
    "dayofmonth",
    "month",
    "is_weekend",
    "is_payday",
    "is_holiday",
    "is_festive",
    "promo",
    "lag7",
    "lag14",
    "roll7_lag7",
    "roll28_lag7",
]

BLOCK = 7

# Fixed-date Nigerian public holidays (month, day); movable feasts are not included.
HOLIDAYS = {(1, 1), (5, 1), (5, 27), (6, 12), (10, 1), (12, 25), (12, 26)}


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str:
    lowered = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(f"CSV needs one of these columns: {', '.join(aliases)}")


def load_sales(source) -> pd.DataFrame:
    """Read a CSV (path or file-like) into a gap-free daily frame: date, sales, promo."""
    raw = pd.read_csv(source)
    date_col = _find_column(list(raw.columns), DATE_ALIASES)
    sales_col = _find_column(list(raw.columns), SALES_ALIASES)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "sales": pd.to_numeric(raw[sales_col], errors="coerce"),
            "promo": raw["promo"].fillna(0).astype(int) if "promo" in raw.columns else 0,
        }
    ).dropna(subset=["date", "sales"])

    df["date"] = df["date"].dt.normalize()
    df = df.groupby("date", as_index=False).agg(sales=("sales", "sum"), promo=("promo", "max"))

    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full).rename_axis("date").reset_index()
    df["sales"] = df["sales"].fillna(0.0)
    df["promo"] = df["promo"].fillna(0).astype(int)

    if len(df) < 8 * BLOCK:
        raise ValueError("Need at least 8 weeks of daily sales to forecast.")
    return df


def has_promo(df: pd.DataFrame) -> bool:
    return bool(df["promo"].any())


def build_features(df: pd.DataFrame, origin: pd.Timestamp | None = None) -> pd.DataFrame:
    """Add feature columns. `df` may contain future rows whose sales are NaN."""
    out = df.copy()
    origin = origin if origin is not None else out["date"].min()
    out["trend"] = (out["date"] - origin).dt.days
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofmonth"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_payday"] = ((out["dayofmonth"] >= 25) | (out["dayofmonth"] <= 2)).astype(int)
    month_day = list(zip(out["month"], out["dayofmonth"]))
    out["is_holiday"] = [int(md in HOLIDAYS) for md in month_day]
    out["is_festive"] = (((out["month"] == 12) & (out["dayofmonth"] >= 20)) | ((out["month"] == 1) & (out["dayofmonth"] == 1))).astype(int)

    s = out["sales"]
    out["lag7"] = s.shift(7)
    out["lag14"] = s.shift(14)
    out["roll7_lag7"] = s.shift(7).rolling(7, min_periods=7).mean()
    out["roll28_lag7"] = s.shift(7).rolling(28, min_periods=14).mean()
    return out


def training_rows(feat: pd.DataFrame) -> pd.DataFrame:
    return feat.dropna(subset=FEATURES + ["sales"])
