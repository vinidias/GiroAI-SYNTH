"""Tests for the GIRO-SYNTH generator.

These run a small generation (500 customers, 100 SKUs) in a temp dir and assert
that the documented invariants hold: table shapes, ground-truth recoverability,
controlled anomaly counts and checksum integrity.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generator.config import Config
from generator.generate import generate


@pytest.fixture(scope="module")
def out():
    # local dir avoids a Windows permission issue on the pytest tmp root
    d = Path(__file__).resolve().parent / "_out"
    if d.exists():
        import shutil
        shutil.rmtree(d)
    cfg = Config()
    cfg.seed = 7
    cfg.n_customers = 500
    cfg.n_skus = 100
    cfg.days = 180
    generate(cfg, d)
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_operational_tables_exist(out):
    for name in ["clientes", "produtos", "pedidos", "itens_pedido",
                 "precos_produto", "tabelas_preco", "vendedores"]:
        p = out / "operational" / f"{name}.csv"
        assert p.exists(), f"missing {name}"
        assert len(pd.read_csv(p)) > 0


def test_ground_truth_tables_exist(out):
    for name in ["gt_elasticity", "gt_churn", "gt_affinity",
                 "gt_propensity", "gt_demand"]:
        p = out / "ground_truth" / f"{name}.csv"
        assert p.exists(), f"missing {name}"


def test_customer_count(out):
    clientes = pd.read_csv(out / "operational" / "clientes.csv")
    assert len(clientes) == 500
    assert clientes["cliente_id"].is_unique


def test_sku_count(out):
    produtos = pd.read_csv(out / "operational" / "produtos.csv")
    assert len(produtos) == 100
    assert produtos["produto_id"].is_unique
    # every SKU has a positive true elasticity
    assert (produtos["elasticidade_verdadeira"] > 0).all()


def test_elasticity_ground_truth(out):
    gt = pd.read_csv(out / "ground_truth" / "gt_elasticity.csv")
    assert len(gt) == 12
    assert gt["epsilon"].between(0.8, 2.0).all()


def test_churn_rate_in_range(out):
    gt = pd.read_csv(out / "ground_truth" / "gt_churn.csv")
    rate = (gt["churn_month_index"] >= 0).mean()
    assert 0.05 < rate < 0.30, f"churn rate {rate} outside expected band"


def test_churn_has_both_classes(out):
    gt = pd.read_csv(out / "ground_truth" / "gt_churn.csv")
    assert (gt["churn_month_index"] >= 0).any()
    assert (gt["churn_month_index"] < 0).any()


def test_affinity_is_probability(out):
    gt = pd.read_csv(out / "ground_truth" / "gt_affinity.csv")
    assert gt["affinity"].between(0, 1).all()
    assert len(gt) == 500 * 40  # 40 top-k per customer


def test_orders_reference_existing_customers(out):
    clientes = pd.read_csv(out / "operational" / "clientes.csv")
    pedidos = pd.read_csv(out / "operational" / "pedidos.csv")
    valid = pedidos[pedidos["cliente_id"] != "C9999"]
    assert set(valid["cliente_id"]).issubset(set(clientes["cliente_id"]))


def test_manifesto_has_checksums(out):
    manifest = pd.read_csv(out / "manifesto.csv")
    assert {"arquivo", "linhas_raw", "linhas_silver_esperadas"}.issubset(manifest.columns)
    checks = (out / "CHECKSUMS.sha256").read_text(encoding="utf-8")
    assert "operational/pedidos.csv" in checks


def test_schema_json(out):
    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    assert schema["version"] == "0.1.0"
    assert schema["scale"]["customers"] == 500
    assert "anomalies" in schema


def test_controlled_anomalies_present(out):
    pedidos = pd.read_csv(out / "operational" / "pedidos.csv")
    itens = pd.read_csv(out / "operational" / "itens_pedido.csv",
                        dtype={"preco_unitario_brl": str})
    # bad customer orders
    assert (pedidos["cliente_id"] == "C9999").any()
    # capitalization drift
    assert "Faturado" in pedidos["status_pedido"].unique()
    # junk numeric
    assert "INVALIDO" in itens["preco_unitario_brl"].unique()


def test_elasticity_signal_recoverable(out):
    """The whole point: a within-SKU regression must recover a negative slope."""
    gt = pd.read_csv(out / "ground_truth" / "gt_demand.csv")
    gt = gt.merge(pd.read_csv(out / "operational" / "produtos.csv")[
        ["produto_id", "elasticidade_verdadeira"]], on="produto_id")
    gt["log_p"] = np.log(gt["preco_efetivo"])
    gt["log_q"] = np.log(gt["quantidade"])
    grp = gt.groupby(["produto_id", "ano_mes"])
    gt["lp"] = gt["log_p"] - grp["log_p"].transform("mean")
    gt["lq"] = gt["log_q"] - grp["log_q"].transform("mean")
    slope = (gt["lp"] * gt["lq"]).sum() / (gt["lp"] ** 2).sum()
    assert slope < 0, "demand must fall when price rises"
    assert abs(slope) > 0.3, f"elasticity signal too weak: {slope}"
