"""Inject controlled data-quality anomalies (catalogued in manifesto)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator.config import Config


def inject(pedidos: pd.DataFrame, itens: pd.DataFrame, cfg: Config,
           rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    counts = {}
    pedidos = pedidos.copy()
    itens = itens.copy()

    n = min(cfg.n_duplicate_orders, len(pedidos))
    dup_idx = rng.choice(len(pedidos), n, replace=False)
    pedidos = pd.concat([pedidos, pedidos.iloc[dup_idx]], ignore_index=True)
    counts["duplicate_orders"] = n

    n = min(cfg.n_bad_customer_orders, len(pedidos))
    bad_idx = rng.choice(len(pedidos), n, replace=False)
    pedidos.loc[bad_idx, "cliente_id"] = "C9999"
    counts["bad_customer_orders"] = n

    n = min(cfg.n_duplicate_items, len(itens))
    dup_idx = rng.choice(len(itens), n, replace=False)
    itens = pd.concat([itens, itens.iloc[dup_idx]], ignore_index=True)
    counts["duplicate_items"] = n

    n = min(cfg.n_junk_numeric, len(itens))
    junk_idx = rng.choice(len(itens), n, replace=False)
    itens["preco_unitario_brl"] = itens["preco_unitario_brl"].astype(object)
    itens.loc[junk_idx, "preco_unitario_brl"] = "INVALIDO"
    counts["junk_numeric"] = n

    n = min(cfg.n_cap_drift, len(pedidos))
    cap_idx = rng.choice(len(pedidos), n, replace=False)
    pedidos.loc[cap_idx, "status_pedido"] = "Faturado"
    counts["capitalization_drift"] = n

    return pedidos, itens, counts
