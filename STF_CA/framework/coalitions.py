from __future__ import annotations
import numpy as np
from typing import Dict, List, Any


class CoalitionGenerator:
    """
    Generates SHAP-style coalition pairs per target cell j, preserving the original logic:
    - For each cell j, produce K pairs (S, S ∪ {j}) where S is non-empty and not "all except j".
    - Enforce z[j] = 0 in S and z'[j] = 1 in S ∪ {j}.
    - Remove duplicates via a set().
    """

    def __init__(self, M: int = 16, K: int = 64, seed: int = 123):
        self.M = M
        self.K = K
        self.seed = seed

    def generate_shap_coalitions_full(self) -> Dict[int, Dict[str, List[Any]]]:
        rng = np.random.default_rng(self.seed)
        coalitions_per_cell: Dict[int, Dict[str, List[Any]]] = {}

        for j in range(self.M):
            z_S, z_SUJ, S_idx, SUJ_idx = [], [], [], []
            seen = set()

            while len(z_S) < self.K:
                z = rng.integers(0, 2, size=self.M)
                z[j] = 0  # enforce j OFF in S
                active = int(np.sum(z))
                if 0 < active < self.M - 1:
                    key = tuple(z.tolist())
                    if key not in seen:
                        seen.add(key)
                        z_S.append(z.copy())
                        z_prime = z.copy()
                        z_prime[j] = 1  # S ∪ {j}
                        z_SUJ.append(z_prime)
                        S_idx.append(np.where(z == 1)[0].tolist())
                        SUJ_idx.append(np.where(z_prime == 1)[0].tolist())

            coalitions_per_cell[j] = {
                "S": z_S,
                "S_U_j": z_SUJ,
                "S_idx": S_idx,
                "S_U_j_idx": SUJ_idx,
            }
        return coalitions_per_cell
