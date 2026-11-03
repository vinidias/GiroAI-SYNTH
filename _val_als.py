import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sp

d = 'D:/Dev/giro-ai/giro-synth/data'
gt = pd.read_csv(f'{d}/ground_truth/gt_demand.csv')
inter = gt.groupby(['cliente_id','produto_id'])['quantidade'].sum().reset_index()
users = inter.cliente_id.unique()
items = inter.produto_id.unique()
ui = {u:i for i,u in enumerate(users)}
ii = {p:i for i,p in enumerate(items)}
rows = inter.cliente_id.map(ui).to_numpy()
cols = inter.produto_id.map(ii).to_numpy()
vals = inter.quantidade.astype(float).to_numpy()
mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))

model = AlternatingLeastSquares(factors=64, iterations=15, random_state=42, use_gpu=False)
model.fit(mat)

# Recall@10: split determinístico 80/20 das interações por usuário
rng = np.random.default_rng(42)
hits, total = 0, 0
sample_users = rng.choice(len(users), size=min(500, len(users)), replace=False)
for u in sample_users:
    row = mat.getrow(u).toarray().ravel()
    nz = np.where(row > 0)[0]
    if len(nz) < 2: continue
    held = rng.choice(nz)
    train_row = row.copy(); train_row[held] = 0
    train_mat = sp.csr_matrix(train_row.reshape(1, -1))
    ids, _ = model.recommend(u, train_mat, N=10, filter_already_liked_items=True)
    hits += int(held in ids)
    total += 1
print(f'ALS recall@10: {hits/total:.3f} ({hits}/{total})')
print(f'Matriz: {len(users)} users x {len(items)} items, {mat.nnz:,} interacoes')
print(f'Esparsidade: {1 - mat.nnz/(len(users)*len(items)):.4f}')