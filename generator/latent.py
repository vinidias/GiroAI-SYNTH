"""Ground-truth latent structure: affinity and propensity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from generator.config import Config


def build_affinity(cfg: Config, clientes: pd.DataFrame,
                   produtos: pd.DataFrame, rng: np.random.Generator):
    d = cfg.latent_dim
    n_c = len(clientes)
    n_p = len(produtos)
    u = rng.normal(0, 1, (n_c, d))
    v = rng.normal(0, 1, (n_p, d))
    # Sharper logits -> more concentrated affinity, so ALS recovers structure
    # instead of only popularity.
    logits = (u @ v.T) / np.sqrt(d) * cfg.affinity_scale * 4.0
    affinity = 1 / (1 + np.exp(-logits))

    top_k = 40
    # vectorised top-k: argpartition is O(n) per row, no Python loop over pairs
    top_idx = np.argpartition(-affinity, top_k, axis=1)[:, :top_k]
    row_idx = np.repeat(np.arange(n_c), top_k)
    flat_affinity = affinity[row_idx, top_idx.ravel()]
    gt = pd.DataFrame({
        "cliente_id": clientes["cliente_id"].to_numpy()[row_idx],
        "produto_id": produtos["produto_id"].to_numpy()[top_idx.ravel()],
        "affinity": np.round(flat_affinity, 4),
    })
    return affinity, gt


def build_propensity(cfg: Config, clientes: pd.DataFrame, rng: np.random.Generator):
    base = rng.gamma(shape=2.0, scale=2.0, size=len(clientes))
    base = np.clip(base, 0.5, 20)
    phase = rng.uniform(0, 2 * np.pi, len(clientes))
    amp = rng.uniform(0.05, 0.25, len(clientes))
    gt = pd.DataFrame({
        "cliente_id": clientes["cliente_id"],
        "base_order_rate": np.round(base, 4),
        "seasonality_phase": np.round(phase, 4),
        "seasonality_amplitude": np.round(amp, 4),
    })
    return base, gt
