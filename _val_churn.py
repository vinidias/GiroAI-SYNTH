import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

d = 'D:/Dev/giro-ai/giro-synth/data'
gt_ch = pd.read_csv(f'{d}/ground_truth/gt_churn.csv')
clientes = pd.read_csv(f'{d}/operational/clientes.csv')
pedidos = pd.read_csv(f'{d}/operational/pedidos.csv', parse_dates=['data_pedido'])

valid = pedidos[pedidos.status_pedido=='FATURADO'].copy()
ref = valid.data_pedido.max()
g = valid.groupby('cliente_id')
rfm = pd.DataFrame({
    'recencia': (ref - g.data_pedido.max()).dt.days,
    'frequencia': g.pedido_id.nunique(),
    'receita': g.valor_mercadoria_total_brl.sum(),
}).reset_index()
rfm = rfm.merge(clientes[['cliente_id','segmento','uf']], on='cliente_id')
rfm = rfm.merge(gt_ch[['cliente_id','churn_month_index']], on='cliente_id')
rfm['churn'] = (rfm.churn_month_index >= 0).astype(int)

X = pd.get_dummies(rfm[['recencia','frequencia','receita','segmento','uf']])
y = rfm.churn
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
clf = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1).fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:,1]
print(f'CHURN AUC: {roc_auc_score(yte, proba):.3f}')
print(f'Churn rate: {y.mean():.3f}')