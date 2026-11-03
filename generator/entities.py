"""Entities: customers, SKUs, price tables, salespeople."""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator.config import Config


def build_customers(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    seg = rng.choice(cfg.segments, cfg.n_customers)
    tables = rng.choice(list(cfg.price_tables.keys()), cfg.n_customers)
    return pd.DataFrame({
        "cliente_id": [f"C{i:05d}" for i in range(1, cfg.n_customers + 1)],
        "razao_social": [f"Cliente {i:05d} Ltda" for i in range(1, cfg.n_customers + 1)],
        "segmento": seg,
        "uf": rng.choice(["SP", "RJ", "MG", "PR", "RS", "BA", "PE", "CE"], cfg.n_customers),
        "tabela_preco_id": tables,
        "data_cadastro": pd.Timestamp(cfg.start_date)
                         - pd.to_timedelta(rng.integers(30, 2000, cfg.n_customers), unit="D"),
        "canal": rng.choice(["REPRESENTANTE", "TELEFONE", "APP", "PORTAL"],
                            cfg.n_customers, p=[0.3, 0.25, 0.25, 0.2]),
        "prazo_pagamento_dias": rng.choice([14, 21, 28, 35, 45], cfg.n_customers),
        "limite_credito_brl": np.round(rng.uniform(10_000, 500_000, cfg.n_customers), 2),
    })


def build_products(cfg: Config, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = cfg.n_skus
    cats = np.array(cfg.categories)
    sku_cat = rng.choice(cats, n)

    eps_cat = rng.uniform(cfg.epsilon_lo, cfg.epsilon_hi, len(cfg.categories))
    eps_map = dict(zip(cfg.categories, eps_cat))
    eps_sku = np.array([eps_map[c] for c in sku_cat]) * rng.lognormal(
        0, cfg.epsilon_sku_log_sd, n)
    eps_sku = np.clip(eps_sku, 0.3, 3.0)

    custo = np.round(rng.uniform(3, 80, n), 2)
    markup = rng.uniform(1.25, 2.0, n)

    produtos = pd.DataFrame({
        "produto_id": [f"P{i:04d}" for i in range(1, n + 1)],
        "descricao": [f"SKU {i:04d}" for i in range(1, n + 1)],
        "categoria": sku_cat,
        "familia": np.where(rng.random(n) < 0.5, "perecivel", "nao_perecivel"),
        "unidade_venda": rng.choice(["UN", "CX", "KG", "FD"], n),
        "custo_referencia_brl": custo,
        "preco_lista_brl": np.round(custo * markup, 2),
        "elasticidade_verdadeira": np.round(eps_sku, 4),
        "fornecedor_id": rng.integers(1, cfg.n_suppliers + 1, n),
    })

    gt_elasticity = pd.DataFrame({
        "categoria": cfg.categories,
        "epsilon": np.round(eps_cat, 4),
    })
    return produtos, gt_elasticity


def build_price_tables(cfg: Config) -> pd.DataFrame:
    return pd.DataFrame({
        "tabela_preco_id": list(cfg.price_tables.keys()),
        "fator_sobre_lista": list(cfg.price_tables.values()),
        "nome": ["Mercado", "Atacado geral", "Volume", "Distribuidor", "Contrato mensal"],
    })


def build_salespeople(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "vendedor_id": [f"VE{i:02d}" for i in range(1, cfg.n_salespeople + 1)],
        "nome": [f"Vendedor {i:02d}" for i in range(1, cfg.n_salespeople + 1)],
        "regiao": rng.choice(["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"],
                             cfg.n_salespeople),
        "comissao_pct": np.round(rng.uniform(1.0, 3.0, cfg.n_salespeople), 2),
        "meta_mensal_brl": np.round(rng.uniform(100_000, 400_000, cfg.n_salespeople), 2),
    })
