from __future__ import annotations
import numpy as np


class PhiComputer:
    """
    Implements pairwise φ (phi) computation with channel-wise replacement:
    φ_pair[k, n, c, :] = |F(X_SUJ) - F(X_S)|, where X_* differ only at channel c.
    """

    def __init__(self, tf_model):
        """
        tf_model: pre-loaded Keras/TF model. Not recompiled (compile=False).
        """
        self.model = tf_model

    def compute_phi_shap_pairwise(
        self,
        X_orig: np.ndarray,       # (N, C, T)
        X_rec_S: np.ndarray,      # (K, N, C, T)
        X_rec_SUJ: np.ndarray,    # (K, N, C, T)
        verbose: bool = True,
    ) -> np.ndarray:
        K, N, C, T = X_rec_S.shape
        num_classes = int(self.model.output_shape[-1])
        phi_pair = np.zeros((K, N, C, num_classes), dtype=np.float32)

        for k in range(K):
            if verbose:
                print(f"Computing φ for coalition pair {k + 1}/{K}")
            for c in range(C):
                X_S = X_orig.copy()
                X_SUJ = X_orig.copy()
                X_S[:, c, :] = X_rec_S[k, :, c, :]
                X_SUJ[:, c, :] = X_rec_SUJ[k, :, c, :]

                F_S = self.model.predict(np.expand_dims(X_S, -1), verbose=0)
                F_SUJ = self.model.predict(np.expand_dims(X_SUJ, -1), verbose=0)

                phi_pair[k, :, c, :] = np.abs(F_SUJ - F_S)
        return phi_pair

    @staticmethod
    def aggregate_phi_L1_over_classes(phi_pair: np.ndarray) -> np.ndarray:
        """
        Mean across classes (unchanged from original).
        """
        return np.mean(phi_pair, axis=-1)