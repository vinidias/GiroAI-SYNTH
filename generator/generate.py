"""GIRO-SYNTH entry point.

Usage:
    python -m generator.generate --out data --seed 42

Writes:
    data/operational/*.csv
    data/ground_truth/*.csv
    data/macro/cepea_monthly.csv
    data/manifesto.csv, dicionario.csv, schema.json, CHECKSUMS.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generator.anomalies import inject
from generator.churn import assign_churn
from generator.config import Config
from generator.entities import (build_customers, build_price_tables,
                                build_products, build_salespeople)
from generator.latent import build_affinity, build_propensity
from generator.macro import load_macro
from generator.orders import generate_orders
from generator.pricing import build_monthly_prices


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return len(df)


DICIONARIO = [
    ("clientes", "cliente_id", "str", "PK", "Unique customer id"),
    ("clientes", "segmento", "str", "", "One of 6 wholesale segments"),
    ("clientes", "tabela_preco_id", "str", "FK tabelas_preco", "Price table assigned"),
    ("produtos", "produto_id", "str", "PK", "Unique SKU id"),
    ("produtos", "categoria", "str", "", "One of 12 food categories"),
    ("produtos", "elasticidade_verdadeira", "float", "ground truth", "True price elasticity used to generate demand"),
    ("pedidos", "pedido_id", "str", "PK", "Unique order id"),
    ("pedidos", "status_pedido", "str", "", "FATURADO | CANCELADO | SEM_EXPEDICAO | BLOQUEADO_CREDITO"),
    ("itens_pedido", "item_pedido_id", "str", "PK", "Unique line id"),
    ("itens_pedido", "preco_unitario_brl", "float", "", "Effective unit price in BRL"),
    ("gt_elasticity", "epsilon", "float", "ground truth", "Category-level true elasticity"),
    ("gt_churn", "churn_month_index", "int", "ground truth", "-1 = active, else month index of churn"),
    ("gt_affinity", "affinity", "float", "ground truth", "Latent customer-SKU affinity in [0,1]"),
    ("gt_propensity", "base_order_rate", "float", "ground truth", "Baseline orders per month"),
]


def generate(cfg: Config, out: Path) -> dict:
    rng = np.random.default_rng(cfg.seed)

    print("[1/7] entities")
    clientes = build_customers(cfg, rng)
    produtos, gt_elasticity = build_products(cfg, rng)
    price_tables = build_price_tables(cfg)
    salespeople = build_salespeople(cfg, rng)

    print("[2/7] macro (CEPEA via agrobr)")
    commodities = sorted(set(cfg.category_commodity.values()))
    macro = load_macro(commodities, cfg.start_date, cfg.days, cfg.seed)

    print("[3/7] causal pricing")
    monthly_prices = build_monthly_prices(cfg, produtos, macro, rng)

    print("[4/7] latent structure")
    affinity, gt_affinity = build_affinity(cfg, clientes, produtos, rng)
    propensity, gt_propensity = build_propensity(cfg, clientes, rng)
    gt_churn = assign_churn(cfg, clientes, rng)

    print("[5/7] orders (heavy step)")
    pedidos, itens, gt_demand = generate_orders(
        cfg, clientes, produtos, monthly_prices, affinity, propensity,
        gt_propensity, gt_churn, price_tables, salespeople, rng,
    )
    print(f"      generated {len(pedidos):,} orders / {len(itens):,} items")

    print("[6/7] controlled anomalies")
    raw_counts = {"pedidos": len(pedidos), "itens_pedido": len(itens)}
    pedidos, itens, anomaly_counts = inject(pedidos, itens, cfg, rng)

    print("[7/7] writing")
    counts = {}
    for name, df in [("clientes", clientes), ("produtos", produtos),
                     ("tabelas_preco", price_tables), ("vendedores", salespeople),
                     ("pedidos", pedidos), ("itens_pedido", itens),
                     ("precos_produto", monthly_prices)]:
        counts[name] = _write(df, out / "operational" / f"{name}.csv")

    gt_counts = {}
    for name, df in [("gt_elasticity", gt_elasticity),
                     ("gt_churn", gt_churn),
                     ("gt_affinity", gt_affinity),
                     ("gt_propensity", gt_propensity),
                     ("gt_demand", gt_demand)]:
        gt_counts[name] = _write(df, out / "ground_truth" / f"{name}.csv")

    _write(macro, out / "macro" / "cepea_monthly.csv")

    # Manifesto: raw counts + expected clean counts (after dedup) + anomaly catalogue
    manifest_rows = []
    for fname, df in [("pedidos", pedidos), ("itens_pedido", itens)]:
        raw = len(df)
        clean = len(df.drop_duplicates())
        manifest_rows.append({"arquivo": f"{fname}.csv", "linhas_raw": raw,
                              "linhas_silver_esperadas": clean})
    manifesto = pd.DataFrame(manifest_rows)
    _write(manifesto, out / "manifesto.csv")

    # Dictionary + schema
    dic = pd.DataFrame(DICIONARIO, columns=["tabela", "coluna", "tipo",
                                            "chave", "descricao"])
    _write(dic, out / "dicionario.csv")

    schema = {
        "version": "0.1.0",
        "seed": cfg.seed,
        "scale": {"customers": cfg.n_customers, "skus": cfg.n_skus,
                  "orders": int(len(pedidos)), "items": int(len(itens))},
        "tables": {
            "operational": list(counts.keys()),
            "ground_truth": list(gt_counts.keys()),
            "macro": ["cepea_monthly.csv"],
        },
        "anomalies": anomaly_counts,
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Checksums
    checks = []
    for p in sorted(out.rglob("*.csv")):
        rel = p.relative_to(out).as_posix()
        checks.append(f"{_sha256(p)}  {rel}")
    (out / "CHECKSUMS.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")

    summary = {
        "operational": counts,
        "ground_truth": gt_counts,
        "anomalies": anomaly_counts,
        "raw_counts": raw_counts,
        "out_dir": str(out),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--customers", type=int, default=None)
    ap.add_argument("--skus", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.seed:
        cfg.seed = args.seed
    if args.customers:
        cfg.n_customers = args.customers
    if args.skus:
        cfg.n_skus = args.skus
    if args.days:
        cfg.days = args.days

    generate(cfg, Path(args.out))


if __name__ == "__main__":
    main()
