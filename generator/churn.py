"""Churn hazard model with correlated leading indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator.config import Config


def assign_churn(cfg: Config, clientes: pd.DataFrame,
                 rng: np.random.Generator) -> pd.DataFrame:
    n = len(clientes)
    hazard_base = rng.gamma(shape=1.5, scale=cfg.churn_base_hazard, size=n)
    reasons = rng.choice(["price", "service", "competitor", "inactive"],
                         n, p=[0.35, 0.2, 0.25, 0.2])
    months = cfg.days // 30
    churn_month = np.full(n, -1, dtype=int)

    target = rng.uniform(cfg.churn_target_rate_lo, cfg.churn_target_rate_hi)
    n_churn = int(round(target * n))
    order = np.argsort(hazard_base)[::-1][:n_churn]
    for i in order:
        # monthly churn probability in (0, 1]
        p = float(np.clip(hazard_base[i] * cfg.churn_beta_revenue, 0.01, 0.9))
        t = int(rng.geometric(p)) - 1
        churn_month[i] = min(t, months - 1)

    return pd.DataFrame({
        "cliente_id": clientes["cliente_id"],
        "churn_month_index": churn_month,
        "hazard_base": np.round(hazard_base, 6),
        "churn_reason": np.where(churn_month >= 0, reasons, ""),
    })
