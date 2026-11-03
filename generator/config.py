"""GIRO-SYNTH configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 42
    days: int = 730
    start_date: str = "2024-09-01"

    n_customers: int = 5_000
    n_skus: int = 500
    n_suppliers: int = 25
    n_salespeople: int = 15

    items_per_order_min: int = 1
    items_per_order_max: int = 8

    epsilon_lo: float = 0.8
    epsilon_hi: float = 2.0
    epsilon_sku_log_sd: float = 0.15

    latent_dim: int = 8
    affinity_scale: float = 1.0

    churn_base_hazard: float = 0.004
    churn_beta_revenue: float = 2.5
    churn_beta_late: float = 1.8
    churn_beta_freq: float = 1.2
    churn_target_rate_lo: float = 0.12
    churn_target_rate_hi: float = 0.18

    macro_coupling: float = 0.6

    categories: list[str] = field(default_factory=lambda: [
        "MERCEARIA", "BEBIDAS", "FRIOS", "PADARIA", "LATICINIOS", "HIGIENE",
        "LIMPEZA", "CONGELADOS", "SNACKS", "HORTIFRUTI", "PET", "DESCARTAVEIS",
    ])

    segments: list[str] = field(default_factory=lambda: [
        "MERCEARIA", "PADARIA", "BAR_RESTAURANTE",
        "ATACADO_SERVICO", "LANCHONETE", "ACOUGUEIRO",
    ])

    price_tables: dict[str, float] = field(default_factory=lambda: {
        "TAB_MERCADO": 1.12,
        "TAB_ATACADO": 1.00,
        "TAB_FORCE": 0.95,
        "TAB_DISTRIBUIDOR": 0.88,
        "TAB_CONTRATO": 0.92,
    })

    category_commodity: dict[str, str] = field(default_factory=lambda: {
        "MERCEARIA": "milho",
        "BEBIDAS": "cafe",
        "LIMPEZA": "algodao",
        "HIGIENE": "algodao",
        "FRIOS": "boi_gordo",
        "LATICINIOS": "boi_gordo",
        "CONGELADOS": "frango_congelado",
        "HORTIFRUTI": "soja",
        "PADARIA": "milho",
        "SNACKS": "milho",
        "PET": "milho",
        "DESCARTAVEIS": "algodao",
    })

    n_duplicate_orders: int = 50
    n_duplicate_items: int = 60
    n_junk_numeric: int = 20
    n_cap_drift: int = 10
    n_bad_customer_orders: int = 3


CFG = Config()
