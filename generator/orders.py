"""Vectorised order + line-item generation.

The first version looped over customers x months x orders in pure Python and
took minutes at 5k customers. This version generates the whole order table
with numpy in one pass, then explodes line items with a vectorised repeat.

Demand model (unchanged):
    qty = base_qty * (preco_efetivo / preco_cheio) ** (-epsilon) * lognormal(0, 0.05)
    base_qty = 40 * popularity_j
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator.config import Config


def generate_orders(cfg: Config, clientes, produtos, monthly_prices,
                    affinity, propensity, propensity_gt, churn_gt,
                    price_tables, salespeople, rng):
    n_c = len(clientes)
    n_p = len(produtos)
    months = sorted(monthly_prices["ano_mes"].unique())
    n_m = len(months)

    # price matrix (n_p, n_m)
    price_pivot = (monthly_prices.pivot(index="produto_id", columns="ano_mes",
                                        values="preco_lista_brl")
                   .reindex(columns=months))
    price_mat = price_pivot.to_numpy()

    pop = rng.pareto(1.5, n_p) + 1
    pop = pop / pop.sum()

    churn_month = churn_gt["churn_month_index"].to_numpy()
    phase = propensity_gt["seasonality_phase"].to_numpy()
    amp = propensity_gt["seasonality_amplitude"].to_numpy()
    # Latent health decline drives observable deterioration before churn:
    # as health falls, order rate and SKU variety fall too, so the RFM
    # features a model sees actually precede the label.
    health_slope = churn_gt["health_slope"].to_numpy() if "health_slope" in churn_gt else np.zeros(n_c)

    table_factor = clientes["tabela_preco_id"].map(
        dict(zip(price_tables["tabela_preco_id"], price_tables["fator_sobre_lista"]))
    ).to_numpy()
    seller_ids = salespeople["vendedor_id"].to_numpy()
    seller_for_customer = rng.integers(0, len(seller_ids), n_c)

    start = pd.Timestamp(cfg.start_date)
    pid_counter = 0
    order_rows, item_rows, demand_rows = [], [], []
    item_seq = 0

    eps_sku = produtos["elasticidade_verdadeira"].to_numpy()

    for mi, month in enumerate(months):
        month_start = pd.Timestamp(month + "-01")
        if month_start < start:
            continue
        season = 1 + amp * np.sin(2 * np.pi * (mi / 12) + phase)
        active = (churn_month < 0) | (mi <= churn_month)
        # health declines linearly: 1.0 at month 0, down to (1 - slope*months)
        health = np.clip(1.0 - health_slope * mi, 0.05, 1.0)
        # order propensity scales with health, so revenue/frequency fall
        # observably in the months before churn
        lam = np.where(active, propensity * season * health, 0.0)
        n_orders = rng.poisson(np.clip(lam, 0, None))
        total_orders = int(n_orders.sum())
        if total_orders == 0:
            continue

        cust_idx = np.repeat(np.arange(n_c), n_orders)
        n_items = rng.integers(cfg.items_per_order_min,
                               cfg.items_per_order_max + 1, total_orders)
        order_ids = np.arange(pid_counter + 1, pid_counter + total_orders + 1)
        pid_counter += total_orders

        # per-order weighted SKU sampling. The loop is over orders (thousands),
        # not over items (millions), so it stays fast while preserving the
        # customer-specific affinity structure that makes the signal recoverable.
        item_cust = np.repeat(cust_idx, n_items)
        item_month = np.full(len(item_cust), mi)
        chosen_skus = np.empty(len(item_cust), dtype=np.int64)
        pos = 0
        for oi in range(total_orders):
            ci = cust_idx[oi]
            k = n_items[oi]
            # affinity dominates popularity so latent structure is the main signal.
            # As health declines the customer narrows their basket: raise the
            # affinity exponent, concentrating purchases on fewer SKUs.
            aff_power = 2.0 + (1.0 - health[ci]) * 4.0
            w = (affinity[ci] ** aff_power) * np.sqrt(pop)
            w = np.clip(w, 1e-12, None)
            w = w / w.sum()
            chosen_skus[pos:pos + k] = rng.choice(n_p, size=k, replace=False, p=w)
            pos += k

        # promo + demand
        promo = rng.choice([0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
                           size=len(item_cust),
                           p=[0.30, 0.15, 0.15, 0.15, 0.15, 0.10])
        full_price = price_mat[chosen_skus, item_month] * table_factor[item_cust]
        eff_price = full_price * (1 - promo)
        ratio = eff_price / np.clip(full_price, 1e-6, None)
        # base_qty scales with the number of SKUs so even low-popularity items
        # get a meaningful expected quantity and the price signal is not
        # rounded away by max(1, ...).
        base_qty = 40.0 * pop[chosen_skus] * n_p
        mean_qty = base_qty * (ratio ** (-eps_sku[chosen_skus]))
        noise = rng.lognormal(0, 0.05, len(item_cust))
        qty = np.maximum(1, np.round(mean_qty * noise).astype(np.int64))

        # build item rows vectorised
        item_seq_start = item_seq + 1
        item_seq += len(item_cust)
        item_ids = [f"IT{i:09d}" for i in range(item_seq_start, item_seq + 1)]
        order_id_per_item = order_ids[np.repeat(np.arange(total_orders), n_items)]

        df_items = pd.DataFrame({
            "item_pedido_id": item_ids,
            "pedido_id": [f"PD{i:08d}" for i in order_id_per_item],
            "produto_id": produtos["produto_id"].to_numpy()[chosen_skus],
            "quantidade_solicitada": qty,
            "quantidade_expedida": qty,
            "preco_unitario_brl": np.round(full_price, 2),
            "desconto_pct": np.round(promo, 4),
            "valor_mercadoria_brl": np.round(qty * eff_price, 2),
        })
        item_rows.append(df_items)

        df_demand = pd.DataFrame({
            "cliente_id": clientes["cliente_id"].to_numpy()[item_cust],
            "produto_id": produtos["produto_id"].to_numpy()[chosen_skus],
            "ano_mes": month,
            "preco_efetivo": np.round(eff_price, 2),
            "quantidade": qty,
        })
        demand_rows.append(df_demand)

        # order-level totals
        order_totals = (df_items.groupby("pedido_id")["valor_mercadoria_brl"]
                        .sum().reindex([f"PD{i:08d}" for i in order_ids],
                                       fill_value=0.0))
        days = rng.integers(0, 28, total_orders)
        dates = month_start + pd.to_timedelta(days, unit="D")
        df_orders = pd.DataFrame({
            "pedido_id": [f"PD{i:08d}" for i in order_ids],
            "cliente_id": clientes["cliente_id"].to_numpy()[cust_idx],
            "vendedor_id": seller_ids[seller_for_customer[cust_idx]],
            "data_pedido": dates,
            "status_pedido": "FATURADO",
            "canal_pedido": clientes["canal"].to_numpy()[cust_idx],
            "tabela_preco_aplicada": clientes["tabela_preco_id"].to_numpy()[cust_idx],
            "valor_mercadoria_total_brl": order_totals.to_numpy(),
        })
        order_rows.append(df_orders)

    pedidos = pd.concat(order_rows, ignore_index=True)
    itens = pd.concat(item_rows, ignore_index=True)
    gt_demand = pd.concat(demand_rows, ignore_index=True)
    return pedidos, itens, gt_demand
