"""Causal chain: commodity -> cost -> list price."""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator.config import Config


def build_monthly_prices(cfg: Config, produtos: pd.DataFrame,
                         macro: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    macro = macro.copy()
    macro["ano_mes"] = macro["ano_mes"].astype(str)
    months = sorted(macro["ano_mes"].unique())
    macro["idx"] = macro.groupby("commodity")["preco"].transform(lambda s: s / s.iloc[0])
    idx_lookup = {(r.commodity, r.ano_mes): r.idx for r in macro.itertuples()}

    rows = []
    for _, p in produtos.iterrows():
        commodity = cfg.category_commodity.get(p["categoria"])
        base_custo = float(p["custo_referencia_brl"])
        markup = float(p["preco_lista_brl"]) / base_custo
        for m in months:
            idx = idx_lookup.get((commodity, m), 1.0)
            custo = base_custo * (idx ** cfg.macro_coupling)
            noise = rng.normal(1.0, 0.02)
            preco = custo * markup * noise
            rows.append({
                "produto_id": p["produto_id"],
                "ano_mes": m,
                "custo_brl": round(custo, 2),
                "preco_lista_brl": round(preco, 2),
            })
    return pd.DataFrame(rows)
