import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

d = "D:/Dev/giro-ai/giro-synth/data"
gt = pd.read_csv(f"{d}/ground_truth/gt_demand.csv")
prod = pd.read_csv(f"{d}/operational/produtos.csv")
print(f"gt_demand: {len(gt):,} linhas")
df = gt.merge(prod[["produto_id","elasticidade_verdadeira","categoria"]], on="produto_id")

# amostra para velocidade
df = df.sample(min(300_000, len(df)), random_state=42)
df["log_preco"] = np.log(df["preco_efetivo"])
df["log_qty"] = np.log(df["quantidade"])
grp = df.groupby(["produto_id","ano_mes"])
df["lp"] = df["log_preco"] - grp["log_preco"].transform("mean")
df["lq"] = df["log_qty"] - grp["log_qty"].transform("mean")

m = smf.ols("lq ~ lp - 1", data=df).fit()
print(f"\n=== ELASTICIDADE ===")
print(f"Pooled within-SKU: eps={-m.params['lp']:.3f}, R2={m.rsquared:.3f}")

truths, ests = [], []
for pid in df.produto_id.unique()[:100]:
    s = df[df.produto_id == pid]
    if s["lp"].std() < 1e-6: continue
    mm = smf.ols("lq ~ lp - 1", data=s).fit()
    truths.append(s.elasticidade_verdadeira.iloc[0])
    ests.append(-mm.params["lp"])
print(f"Correlacao Pearson:  {pearsonr(truths, ests)[0]:.3f}")
print(f"Correlacao Spearman: {spearmanr(truths, ests)[0]:.3f}")

# churn
gt_ch = pd.read_csv(f"{d}/ground_truth/gt_churn.csv")
print(f"\n=== CHURN ===")
churn_rate = (gt_ch.churn_month_index >= 0).mean()
print(f"Churn rate: {churn_rate:.3f} ({int((gt_ch.churn_month_index >= 0).sum()):,}/{len(gt_ch):,})")

# recomendacao: esparsidade da matriz
print(f"\n=== RECOMENDACAO ===")
n_users = gt.cliente_id.nunique()
n_items = gt.produto_id.nunique()
n_inter = len(gt)
print(f"Matriz: {n_users} x {n_items}")
print(f"Interacoes: {n_inter:,}")
print(f"Esparsidade: {1 - n_inter/(n_users*n_items):.4f}")
print(f"Media interacoes/usuario: {n_inter/n_users:.1f}")