"""Load real CEPEA macro series; fallback to synthetic."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MACRO_CACHE = Path(__file__).resolve().parent.parent / "data" / "macro" / "cepea_monthly.csv"


def _synthetic(commodities, months, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for c in commodities:
        base = rng.uniform(50, 300)
        for i, m in enumerate(months):
            trend = 1 + 0.01 * i
            season = 1 + 0.08 * np.sin(2 * np.pi * (m.month / 12) + rng.uniform(0, 6.28))
            noise = rng.normal(1, 0.03)
            rows.append({"ano_mes": str(m), "commodity": c,
                         "preco": round(base * trend * season * noise, 2)})
    return pd.DataFrame(rows)


def load_macro(commodities, start, days, seed=42):
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(days=days)
    months = pd.period_range(start_ts, end_ts, freq="M")

    if MACRO_CACHE.exists():
        df = pd.read_csv(MACRO_CACHE)
        df = df[df["commodity"].isin(commodities)]
        df = df[(df["ano_mes"] >= str(months[0])) & (df["ano_mes"] <= str(months[-1]))]
        if not df.empty:
            log.info("Using cached CEPEA: %d rows", len(df))
            return df

    try:
        from agrobr.sync import cepea
        rows = []
        for c in commodities:
            raw = cepea.indicador(c, inicio=str(start_ts.date()))
            if raw is None or raw.empty:
                continue
            raw = raw.copy()
            date_col = "data" if "data" in raw.columns else raw.columns[0]
            value_col = "valor" if "valor" in raw.columns else raw.columns[1]
            raw["data"] = pd.to_datetime(raw[date_col], errors="coerce")
            raw["preco"] = pd.to_numeric(raw[value_col], errors="coerce")
            raw = raw.dropna(subset=["data", "preco"])
            raw["ano_mes"] = raw["data"].dt.to_period("M").astype(str)
            agg = raw.groupby("ano_mes", as_index=False)["preco"].mean()
            agg["commodity"] = c
            rows.append(agg)
        if rows:
            df = pd.concat(rows, ignore_index=True)
            MACRO_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(MACRO_CACHE, index=False)
            log.info("Fetched CEPEA: %d rows -> %s", len(df), MACRO_CACHE)
            return df
    except Exception as exc:
        log.warning("CEPEA unavailable (%s), using synthetic", exc)

    df = _synthetic(commodities, months, seed)
    MACRO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MACRO_CACHE, index=False)
    return df
