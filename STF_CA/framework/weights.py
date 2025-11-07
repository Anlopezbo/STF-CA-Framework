from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any


def shap_kernel_weight(M: int, s: int) -> float:
    """
    Classic SHAP kernel weight for coalition size s (excluding the target feature j).
      weight(s) = (M-1) / [ C(M-1, s) * s * (M-1 - s) ]  for s in {1, ..., M-2}
      weight(0) or weight(M-1) = 0
    """
    M1 = M - 1
    if s <= 0 or s >= M1:
        return 0.0
    return (M1) / (math.comb(M1, s) * s * (M1 - s))


def aggregate_subject_phi(subj: Dict[str, Any], split: str) -> Dict[str, np.ndarray]:
    """
    Applies both kernel SHAP and uniform coalition weighting to per-cell φ values, then:
      - averages across trials (N) → (M, C)
      - min-shifts and max-normalizes each class pair jointly

    Inputs (from load_subject_phi):
      subj["coalitions_per_cell"]  : dict with 'S_idx' per cell (coalitions' sets S)
      subj["phi_cls0"][split][j]   : array (K, N0, C) for cell j
      subj["phi_cls1"][split][j]   : array (K, N1, C) for cell j

    Returns dict of arrays with shape (M, C):
      {
        "kernel_cls0",  "kernel_cls1",    # kernel-weighted, normalized
        "uniform_cls0", "uniform_cls1",   # simple mean over coalitions, normalized
      }
    """
    M = int(subj["M"])
    coalitions_per_cell = subj["coalitions_per_cell"]
    phi_cls0 = subj["phi_cls0"][split]   # list length M, each (K, N0, C)
    phi_cls1 = subj["phi_cls1"][split]   # list length M, each (K, N1, C)

    # Pre-compute weights per cell j
    weights_per_j = {}
    for j in range(M):
        s_sizes = [len(idx) for idx in coalitions_per_cell[j]['S_idx']]
        w = np.array([shap_kernel_weight(M, s) for s in s_sizes], dtype=np.float64)
        w /= w.sum()
        weights_per_j[j] = w  # shape (K,)

    def weighted_phi(phi_KNC: np.ndarray, w: np.ndarray) -> np.ndarray:
        # phi_KNC: (K, N, C) ; w: (K,)
        # tensordot over K → (N, C)
        return np.tensordot(w, phi_KNC, axes=(0, 0))

    phi_cells_kernel_cls0, phi_cells_kernel_cls1 = [], []
    phi_cells_uniform_cls0, phi_cells_uniform_cls1 = [], []

    for j in range(M):
        phi0, phi1 = phi_cls0[j], phi_cls1[j]          # (K,N0,C), (K,N1,C)
        wj = weights_per_j[j]                          # (K,)
        phi_cells_kernel_cls0.append(weighted_phi(phi0, wj))   # (N0,C)
        phi_cells_kernel_cls1.append(weighted_phi(phi1, wj))   # (N1,C)
        phi_cells_uniform_cls0.append(phi0.mean(axis=0))       # (N0,C)
        phi_cells_uniform_cls1.append(phi1.mean(axis=0))       # (N1,C)

    # Stack over cells j → (M, N, C)
    phi_cells_kernel_cls0  = np.stack(phi_cells_kernel_cls0,  axis=0)
    phi_cells_kernel_cls1  = np.stack(phi_cells_kernel_cls1,  axis=0)
    phi_cells_uniform_cls0 = np.stack(phi_cells_uniform_cls0, axis=0)
    phi_cells_uniform_cls1 = np.stack(phi_cells_uniform_cls1, axis=0)

    # Average over trials N → (M, C)
    phi_kernel_cls0_MC  = phi_cells_kernel_cls0.mean(axis=1)
    phi_kernel_cls1_MC  = phi_cells_kernel_cls1.mean(axis=1)
    phi_uniform_cls0_MC = phi_cells_uniform_cls0.mean(axis=1)
    phi_uniform_cls1_MC = phi_cells_uniform_cls1.mean(axis=1)

    def normalize_pair(phi0: np.ndarray, phi1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # joint min-shift and max-normalization across the pair
        gmin = np.min([phi0, phi1])
        phi0, phi1 = phi0 - gmin, phi1 - gmin
        gmax = np.max([phi0, phi1])
        return (phi0 / gmax, phi1 / gmax) if gmax > 0 else (phi0, phi1)

    phi_norm0,  phi_norm1  = normalize_pair(phi_kernel_cls0_MC,  phi_kernel_cls1_MC)
    phi_norm0u, phi_norm1u = normalize_pair(phi_uniform_cls0_MC, phi_uniform_cls1_MC)

    return {
        "kernel_cls0":  phi_norm0,
        "kernel_cls1":  phi_norm1,
        "uniform_cls0": phi_norm0u,
        "uniform_cls1": phi_norm1u,
    }
