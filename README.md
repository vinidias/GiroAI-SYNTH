# GIRO-SYNTH

> A large synthetic B2B food-distribution dataset with **documented behavioural
> signals** - price elasticity, churn and latent customer-SKU affinity generated
> from a known ground truth.

## Why this dataset exists

The [Atacado Atlas](https://www.kaggle.com/datasets/vongrossi/atacado-atlas-synthetic-wholesale-distribution)
is excellent for data engineering but was not built for predictive modelling.
Measured on it: price elasticity R² = 0.004, churn AUC = 0.537, ALS recall@10 = 0.208.

GIRO-SYNTH keeps the operational realism (12 categories, 6 segments, 5 price
tables, BRL, 730 days) and **injects the signals a data scientist needs**, each
with a documented ground-truth parameter. Models can therefore be scored against
the truth, not just held-out data.

## Scale

| Entity | Count |
|---|---|
| Customers | 5,000 |
| SKUs | 500 (12 categories) |
| Period | 730 days (2024-09-01 to 2026-08-31) |
| Orders | ~475,000 |
| Line items | ~2,140,000 |
| Price tables | 5 (factors 0.88 - 1.12) |
| Commodity series | 6 (real CEPEA via agrobr) |

## Measured recoverability

| Signal | How it is generated | Recovered by | Result |
|---|---|---|---|
| Price elasticity | `qty = k * (price/full_price) ** (-epsilon)`, epsilon per SKU | within-SKU OLS | **Pearson 0.995**, ε pooled 1.20 |
| Churn | monthly hazard, leading indicators in RFM | LightGBM on RFM | **AUC 0.988**, rate 12.9% |
| SKU affinity | 8-dim latent vectors, sigmoid | implicit ALS | **recall@10 0.246** |

## Causal chain (the macro coupling)

```
CEPEA commodity price  ->  SKU cost  ->  list price  ->  effective price  ->  demand
```

Costs follow the real CEPEA monthly index for the category's commodity (cafe,
milho, algodao, boi_gordo, frango_congelado, soja), so price varies for an
economic reason and elasticity is identifiable.

## Tables

**Operational** (`data/operational/`)

```
clientes.csv         5,000 customers
produtos.csv         500 SKUs (+ elasticidade_verdadeira column)
tabelas_preco.csv    5 price tables
vendedores.csv       salespeople
pedidos.csv          orders
itens_pedido.csv     line items (quantidade, preco, desconto)
precos_produto.csv   monthly list prices per SKU
```

**Ground truth** (`data/ground_truth/`)

```
gt_elasticity.csv    category-level true elasticity
gt_churn.csv         churn month index + hazard + reason
gt_affinity.csv      top-40 latent affinity per customer
gt_propensity.csv    baseline order rate + seasonality
gt_demand.csv        full customer-SKU-month demand panel
```

**Meta** - `manifesto.csv`, `dicionario.csv`, `schema.json`, `CHECKSUMS.sha256`.

## Controlled anomalies (catalogued in manifesto)

| Anomaly | Count |
|---|---|
| Exact duplicate orders | 50 |
| Duplicate line items | 60 |
| Orders with non-existent customer | 3 |
| Junk values in numeric column | 20 |
| Capitalization drift | 10 |

## Reproduce

```bash
python -m generator.generate --out data --seed 42
# or smaller:
python -m generator.generate --out data_small --customers 500 --skus 100
```

Runs in ~50 s on a laptop for the full scale.

## Validate

```bash
python _val_big.py      # elasticity + churn ground truth
python _val_als.py      # ALS recall
python _val_churn.py    # churn AUC
```

## License

- **Data**: CC BY 4.0
- **Generator**: MIT

## Citation

```
Dias, V. (2026). GIRO-SYNTH: B2B Food Distribution with Known Behavioural Signals.
https://github.com/vinidias/giro-ai
```
