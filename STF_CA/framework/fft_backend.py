from __future__ import annotations
import gc
import numpy as np
from typing import Dict, List, Tuple


def build_time_windows_indices(cut_times: List[float], fs: float, T: int) -> Tuple[List[slice], List[int]]:
    """
    Build time window slices and lengths (in samples).
    """
    idx_edges = [int(t * fs) for t in cut_times]
    idx_edges[-1] = T  # ensure last edge equals provided T
    win_slices = [slice(idx_edges[i], idx_edges[i + 1]) for i in range(len(idx_edges) - 1)]
    win_lengths = [s.stop - s.start for s in win_slices]
    return win_slices, win_lengths


def freq_bins_per_window(win_lengths: List[int], fs: float) -> Tuple[List[np.ndarray], List[int]]:
    """
    Compute rFFTFreqs per window and number of bins per window.
    """
    rfftfreqs = [np.fft.rfftfreq(Nw, d=1.0 / fs) for Nw in win_lengths]
    nfft_bins = [len(f) for f in rfftfreqs]
    return rfftfreqs, nfft_bins


def cell_map_and_freq_masks(
    band_limits: Dict[str, Tuple[float, float]],
    rfftfreqs: List[np.ndarray],
) -> List[Dict[int, np.ndarray]]:
    """
    Cell ordering: band_order=['Theta','Alpha','Beta','Gamma'], j = i_band*4 + i_win.
    Returns freq_mask_per_cell[j] -> dict {win_idx: boolean mask over Fw}.
    """
    band_order = ["Theta", "Alpha", "Beta", "Gamma"]
    freq_mask_per_cell: List[Dict[int, np.ndarray]] = []
    for i_band, bname in enumerate(band_order):
        low, high = band_limits[bname]
        for i_win in range(4):
            freqs = rfftfreqs[i_win]
            mask = (freqs >= low) & (freqs < high)
            freq_mask_per_cell.append({i_win: mask})
    return freq_mask_per_cell


def sfft_windows_trial(trial_data: np.ndarray, fs: float, win_slices: List[slice]) -> List[np.ndarray]:
    """
    trial_data: (C, T) -> list of 4 arrays (C, Fw) with rFFT per window.
    """
    C, _T = trial_data.shape
    specs_win: List[np.ndarray] = []
    for slc in win_slices:
        seg = trial_data[:, slc]                   # (C, Nw)
        spec = np.fft.rfft(seg, axis=-1)          # (C, Nw//2+1)
        specs_win.append(spec)
    return specs_win


def sfft_windows_trials(X: np.ndarray, fs: float, win_slices: List[slice]) -> List[List[np.ndarray]]:
    """
    X: (N, C, T) -> list length N; each entry is a list of length 4 with (C, Fw).
    """
    return [sfft_windows_trial(X[n], fs, win_slices) for n in range(X.shape[0])]


def isfft_concat_trial(specs_win: List[np.ndarray], fs: float, win_lengths: List[int]) -> np.ndarray:
    """
    Window-wise iRFFT reconstruction followed by temporal concatenation: returns (C, T).
    """
    C = specs_win[0].shape[0]
    rec_segs: List[np.ndarray] = []
    for w, spec in enumerate(specs_win):
        Nw = win_lengths[w]
        rec = np.fft.irfft(spec, n=Nw, axis=-1)  # (C, Nw) real
        rec_segs.append(rec)
    x_rec = np.concatenate(rec_segs, axis=-1)    # (C, T)
    return x_rec.astype(np.float32)


def reconstruct_trials_from_coalitions_stft(
    specs_trials: List[List[np.ndarray]],
    z_list: List[np.ndarray],
    fs: float,
    win_lengths: List[int],
    freq_mask_per_cell: List[Dict[int, np.ndarray]],
) -> np.ndarray:
    """
    specs_trials: length N, each = list of 4 windows, (C, Fw)
    z_list: length K, each z shape=(M=16,)
    Returns: X_rec (K, N, C, T)
    """
    N = len(specs_trials)
    C = specs_trials[0][0].shape[0]
    T = sum(win_lengths)
    K = len(z_list)
    X_rec = np.zeros((K, N, C, T), dtype=np.float32)

    for k, z in enumerate(z_list):
        # Union of OFF masks per window
        per_win_union = [None, None, None, None]
        for j, zj in enumerate(z):
            if zj == 0:  # OFF
                for win_idx, mask in freq_mask_per_cell[j].items():
                    per_win_union[win_idx] = mask if per_win_union[win_idx] is None else (per_win_union[win_idx] | mask)

        for n in range(N):
            masked_specs: List[np.ndarray] = []
            for w in range(4):
                spec = specs_trials[n][w].copy()     # (C, Fw)
                Fw = spec.shape[-1]
                if per_win_union[w] is None:
                    mask_w = np.zeros(Fw, dtype=bool)
                else:
                    mask_w = per_win_union[w]
                    # Safety: match length
                    if mask_w.shape[0] != Fw:
                        mask_w = mask_w[:Fw] if mask_w.shape[0] > Fw else np.pad(
                            mask_w, (0, Fw - mask_w.shape[0]), 'constant', constant_values=False
                        )
                spec[:, mask_w] = 0 + 0j
                masked_specs.append(spec)
            X_rec[k, n] = isfft_concat_trial(masked_specs, fs, win_lengths)
        gc.collect()
    return X_rec