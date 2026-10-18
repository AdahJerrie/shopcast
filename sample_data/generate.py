"""Generate a year of synthetic daily sales for a small Lagos bakery.

Patterns baked in: weekly cycle (busy Fri/Sat, quiet Sun morning), month-end
payday bump, slow upward trend, promo days, festive peaks, and a handful of
unusual days (generator outages, a bulk catering order) for anomaly detection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(26)

dates = pd.date_range("2025-10-01", "2026-10-16", freq="D")
n = len(dates)

weekday_factor = np.array([0.95, 0.9, 0.95, 1.0, 1.2, 1.35, 0.75])
base = 85_000 * (1 + np.linspace(0, 0.18, n))
sales = base * weekday_factor[dates.dayofweek]

payday = (dates.day >= 25) | (dates.day <= 2)
sales *= np.where(payday, 1.15, 1.0)

promo = np.zeros(n, dtype=int)
promo_idx = rng.choice(np.where(dates.dayofweek.isin([4, 5]))[0], size=30, replace=False)
promo[promo_idx] = 1
sales *= np.where(promo == 1, 1.22, 1.0)

festive = ((dates.month == 12) & (dates.day.isin(range(20, 32)))) | (
    (dates.month == 1) & (dates.day == 1)
)
sales *= np.where(festive, 1.5, 1.0)

sales *= rng.normal(1.0, 0.06, n)

anomalies = {
    "2026-02-11": 0.35,
    "2026-05-06": 0.4,
    "2026-07-18": 2.4,
    "2026-09-02": 0.3,
}
for day, factor in anomalies.items():
    sales[dates.get_loc(pd.Timestamp(day))] *= factor

df = pd.DataFrame({"date": dates.date, "sales": sales.round(-1).astype(int), "promo": promo})
out = Path(__file__).with_name("lagos_bakery.csv")
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
