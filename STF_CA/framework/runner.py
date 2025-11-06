from __future__ import annotations
import os
import gc
import numpy as np
from typing import List

from STF_CA.utils.config import (
    SAMPLING_RATE,
    CUT_TIMES,
    BAND_LIMITS,
    M_CELLS,
)
from STF_CA.framework.coalitions import CoalitionGenerator
from STF_CA.framework.fft_backend import (
    build_time_windows_indices,
    freq_bins_per_window,
    cell_map_and_freq_masks,
    sfft_windows_trials,
    reconstruct_trials_from_coalitions_stft,
)
from STF_CA.framework.phi import PhiComputer
from STF_CA.utils.io_utils import load_subject_artifacts, save_numpy


class PipelineRunner:
    """
    Orchestrates the pipeline over subject(s) and split(s), preserving names,
    steps and outputs from the original script.
    """

    def __init__(
        self,
        input_base_dir: str = "./data",          # generic default
        save_base_dir: str = "./outputs",        # generic default
        M: int = M_CELLS,
        K: int = 10,
        seed: int = 23,
        split: str = "train",  # 'train' | 'test' | 'both'
        subject_dir_template: str = "subject_{id}_results_0_7",
        model_filename_template: str = "model_subject_{id}.h5",
    ):
        self.input_base_dir = input_base_dir
        self.save_base_dir = save_base_dir
        self.M = M
        self.K = K
        self.seed = seed
        self.split = split
        self.subject_dir_template = subject_dir_template
        self.model_filename_template = model_filename_template

    def _get_splits(self) -> List[str]:
        return [self.split] if self.split in ("train", "test") else ["train", "test"]

    def run_on_subjects(self, subject_list: List[int]) -> None:
        os.makedirs(self.save_base_dir, exist_ok=True)

        # Generate coalitions once
        coal_gen = CoalitionGenerator(M=self.M, K=self.K, seed=self.seed)
        coalitions_per_cell = coal_gen.generate_shap_coalitions_full()

        for sbj in subject_list:
            print(f"\n\n========== Subject {sbj} ==========")
            out_dir = os.path.join(self.save_base_dir, f"subject_{sbj}_results")
            os.makedirs(out_dir, exist_ok=True)

            # Save coalitions immediately (as in the original script)
            save_numpy(os.path.join(out_dir, "coalitions_per_cell.npy"), coalitions_per_cell, allow_pickle=True)

            # Run per split (model + data per split)
            for sp in self._get_splits():
                try:
                    model, X, y = load_subject_artifacts(
                        base_dir=self.input_base_dir,
                        subject_id=sbj,
                        split=sp,
                        subject_dir_template=self.subject_dir_template,
                        model_filename_template=self.model_filename_template,
                    )
                except FileNotFoundError as e:
                    print(f"[WARN] {e}; continue.")
                    continue

                classes = np.unique(y)
                if len(classes) != 2:
                    print(f"[WARN] Expected 2 classes, found {len(classes)} in subject {sbj} ({sp}). Classes: {classes}")
                    del X, y, model
                    gc.collect()
                    continue

                cls0, cls1 = int(classes[0]), int(classes[1])
                X_class_0 = X[y == cls0]
                X_class_1 = X[y == cls1]

                # FFT backend (per split, in case T differs)
                T = X.shape[-1]
                win_slices, win_lengths = build_time_windows_indices(CUT_TIMES, SAMPLING_RATE, T)
                rfftfreqs, _ = freq_bins_per_window(win_lengths, SAMPLING_RATE)
                freq_mask_per_cell = cell_map_and_freq_masks(BAND_LIMITS, rfftfreqs)

                # Auxiliary saves
                save_numpy(os.path.join(out_dir, "win_lengths.npy"), np.array(win_lengths))
                save_numpy(os.path.join(out_dir, "rfftfreqs.npy"), np.array(rfftfreqs, dtype=object), allow_pickle=True)

                print(f"SFFT windows class 0 ({sp})...")
                specs_trials_0 = sfft_windows_trials(X_class_0, SAMPLING_RATE, win_slices)
                print(f"SFFT windows class 1 ({sp})...")
                specs_trials_1 = sfft_windows_trials(X_class_1, SAMPLING_RATE, win_slices)

                # φ computation
                phi_comp = PhiComputer(model)

                # Loop over target cell j (0..M-1)
                for j in range(self.M):
                    print(f"\n--- Target cell j={j}/{self.M - 1} ({sp}) ---")
                    z_S = coalitions_per_cell[j]["S"]
                    z_SUJ = coalitions_per_cell[j]["S_U_j"]

                    # Reconstructions per coalition (per class)
                    X_rec_S_0 = reconstruct_trials_from_coalitions_stft(
                        specs_trials_0, z_S, SAMPLING_RATE, win_lengths, freq_mask_per_cell
                    )
                    X_rec_SUJ_0 = reconstruct_trials_from_coalitions_stft(
                        specs_trials_0, z_SUJ, SAMPLING_RATE, win_lengths, freq_mask_per_cell
                    )
                    X_rec_S_1 = reconstruct_trials_from_coalitions_stft(
                        specs_trials_1, z_S, SAMPLING_RATE, win_lengths, freq_mask_per_cell
                    )
                    X_rec_SUJ_1 = reconstruct_trials_from_coalitions_stft(
                        specs_trials_1, z_SUJ, SAMPLING_RATE, win_lengths, freq_mask_per_cell
                    )

                    # Pairwise φ (channel-wise replacement)
                    phi_pair_0 = phi_comp.compute_phi_shap_pairwise(
                        X_class_0, X_rec_S_0, X_rec_SUJ_0
                    )  # (K, N0, C, num_classes)
                    phi_pair_1 = phi_comp.compute_phi_shap_pairwise(
                        X_class_1, X_rec_S_1, X_rec_SUJ_1
                    )  # (K, N1, C, num_classes)

                    # Scalar φ mean across classes
                    phi_pair_0_L1mean = PhiComputer.aggregate_phi_L1_over_classes(phi_pair_0)  # (K, N0, C)
                    phi_pair_1_L1mean = PhiComputer.aggregate_phi_L1_over_classes(phi_pair_1)  # (K, N1, C)

                    # Save (add _test suffix if needed)
                    suf = "" if sp == "train" else "_test"
                    save_numpy(os.path.join(out_dir, f"phi_pair_cls0_j{j}{suf}.npy"), phi_pair_0)
                    save_numpy(os.path.join(out_dir, f"phi_pair_cls1_j{j}{suf}.npy"), phi_pair_1)
                    save_numpy(os.path.join(out_dir, f"phi_pair_L1mean_cls0_j{j}{suf}.npy"), phi_pair_0_L1mean)
                    save_numpy(os.path.join(out_dir, f"phi_pair_L1mean_cls1_j{j}{suf}.npy"), phi_pair_1_L1mean)

                    # Memory hygiene for this cell
                    del X_rec_S_0, X_rec_SUJ_0, X_rec_S_1, X_rec_SUJ_1
                    del phi_pair_0, phi_pair_1, phi_pair_0_L1mean, phi_pair_1_L1mean
                    gc.collect()

                # Clean per split
                del X, y, X_class_0, X_class_1
                del specs_trials_0, specs_trials_1, freq_mask_per_cell, rfftfreqs, win_lengths, win_slices
                del model
                gc.collect()