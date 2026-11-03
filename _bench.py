import time
from generator.config import Config
from generator.entities import build_customers, build_products, build_price_tables, build_salespeople
from generator.latent import build_affinity, build_propensity
from generator.churn import assign_churn
from generator.macro import load_macro
from generator.pricing import build_monthly_prices
import numpy as np

cfg = Config()
rng = np.random.default_rng(42)
t0 = time.time()
clientes = build_customers(cfg, rng); print(f"clientes: {time.time()-t0:.2f}s")
produtos, _ = build_products(cfg, rng); print(f"produtos: {time.time()-t0:.2f}s")
tables = build_price_tables(cfg)
sellers = build_salespeople(cfg, rng)
commodities = sorted(set(cfg.category_commodity.values()))
macro = load_macro(commodities, cfg.start_date, cfg.days, cfg.seed); print(f"macro: {time.time()-t0:.2f}s")
prices = build_monthly_prices(cfg, produtos, macro, rng); print(f"prices: {time.time()-t0:.2f}s ({len(prices):,} rows)")
aff, _ = build_affinity(cfg, clientes, produtos, rng); print(f"affinity: {time.time()-t0:.2f}s")
prop, prop_gt = build_propensity(cfg, clientes, rng)
churn = assign_churn(cfg, clientes, rng); print(f"churn: {time.time()-t0:.2f}s")