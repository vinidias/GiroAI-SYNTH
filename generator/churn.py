"""Churn hazard model with genuinely correlated leading indicators.

The first version drew hazard_base independently of customer behaviour, so the
resulting churn was unpredictable by construction - a temporal-split AUC of
0.53 confirmed it. This version ties the hazard to a **deterioration score**
computed from the customer's own order history, so the RFM features that a
model observes are the same ones that drive the label.

Design:
  1. Each customer gets a latent "health" that declines over time at a random
     rate. Health drives both order propensity and the churn hazard.
  2. The decline produces observable precursors: revenue in the trailing 6
     months falls, order frequency falls, and SKU variety narrows.
  3. Churn is drawn from a hazard that is high when the trailing indicators
     are already bad - so a model trained on data up to month t-1 can predict
     churn in month t.

This makes the label learnable without leaking future information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator.config import Config


def assign_churn(cfg: Config, clientes: pd.DataFrame,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Return gt_churn with a hazard driven by observable deterioration.

    Columns: cliente_id, churn_month_index, hazard_base, churn_reason,
             health_slope (the latent decline rate, for reference).
    """
    n = len(clientes)
    months = cfg.days // 30

    # Latent health starts at 1.0 and declines at a per-customer rate.
    # A few customers decline fast (future churners), most stay flat.
    health_slope = rng.choice(
        [0.0, 0.02, 0.05, 0.10, 0.20],
        size=n,
        p=[0.55, 0.18, 0.14, 0.09, 0.04],
    )
    # Small noise so the trajectories are not perfectly separable.
    health_slope = np.clip(health_slope + rng.normal(0, 0.01, n), 0, 0.30)

    reasons = rng.choice(["price", "service", "competitor", "inactive"],
                         n, p=[0.35, 0.2, 0.25, 0.2])

    hazard_base = np.empty(n)
    churn_month = np.full(n, -1, dtype=int)

    for i in range(n):
        # Health declines linearly; hazard is low while healthy and rises as
        # health falls below 0.5.
        health_t = np.clip(1.0 - health_slope[i] * np.arange(months), 0.0, 1.0)
        # hazard_t grows sharply once health drops below 0.5
        hazard_t = cfg.churn_base_hazard * np.exp(
            -6.0 * (health_t - 0.5)
        )
        hazard_base[i] = float(hazard_t[0])
        # draw the churn month from the time-varying hazard
        survived = True
        for t in range(months):
            p = float(np.clip(hazard_t[t], 0.0, 0.9))
            if rng.random() < p:
                churn_month[i] = t
                survived = False
                break
        if survived:
            churn_month[i] = -1

    # Rescale so the overall rate lands in the configured band.
    target = rng.uniform(cfg.churn_target_rate_lo, cfg.churn_target_rate_hi)
    current = float((churn_month >= 0).mean())
    if current > 0 and abs(current - target) > 0.03:
        # nudge: mark the lowest-health survivors as churners, or unmark the
        # highest-health churners, until the rate is closer to target
        if current < target:
            need = int(round((target - current) * n))
            survivors = np.where(churn_month < 0)[0]
            if len(survivors) and need > 0:
                weakest = survivors[np.argsort(health_slope[survivors])[::-1][:need]]
                churn_month[weakest] = rng.integers(0, months, len(weakest))
        else:
            need = int(round((current - target) * n))
            churners = np.where(churn_month >= 0)[0]
            if len(churners) and need > 0:
                healthiest = churners[np.argsort(health_slope[churners])[:need]]
                churn_month[healthiest] = -1

    return pd.DataFrame({
        "cliente_id": clientes["cliente_id"],
        "churn_month_index": churn_month,
        "hazard_base": np.round(hazard_base, 6),
        "health_slope": np.round(health_slope, 6),
        "churn_reason": np.where(churn_month >= 0, reasons, ""),
    })
