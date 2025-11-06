# EEG SHAP-style φ Pipeline (windowed FFT, band × time cells)

This repository computes **φ (phi) attributions** for EEG using a **windowed FFT** backend. Each attribution targets one of **16 cells** (4 frequency bands × 4 time windows). The math/logic is **unchanged** from the original script:

- Windows: **CUT_TIMES = [0, 2, 2.5, 5, 7]** seconds (4 segments)
- Bands: **Theta [4–8), Alpha [8–13), Beta [13–30), Gamma [30–40)**
- Reconstruction: rFFT/iRFFT **per window**, with **cell-wise frequency masking**
- Attribution: **pairwise φ** per coalition, **channel-wise replacement**:
  |F(X_{S∪{j}}) − F(X_S)|, then mean across classes if needed

---

## Features

- Coalition generation per target cell **j** (S vs S ∪ {j}), no duplicates
- Exact band×window masks (16 cells total)
- Windowed rFFT/iRFFT reconstruction with OFF masks
- Channel-wise φ computation; helper to average across classes
- CLI runner with **generic, reusable paths** and **naming templates**

---

## Repository layout

```
eeg_shap_pipeline/
├─ eeg_shap_pipeline/
│  ├─ framework/       # domain core: coalitions, FFT backend, phi, pipeline runner
│  ├─ utils/           # config constants and IO helpers (load/save)
│  └─ py.typed
│
├─ scripts/
│  └─ run_pipeline.py  # CLI entrypoint
│
└─ README.md
```

---

## Data layout (default expectation)

Place your data under `--input_base_dir` (default `./data`) like this:

```
./data/
  subject_{ID}_results_0_7/
    model_subject_{ID}.h5
    train_data.npz   # X_train, y_train
    test_data.npz    # X_test,  y_test
```

You can change the folder and filename patterns via CLI:
- `--subject_dir_template` (default: `subject_{id}_results_0_7`)
- `--model_filename_template` (default: `model_subject_{id}.h5`)

Example for:
```
./data/
  S{ID}/
    net_{ID}.h5
    train_data.npz
    test_data.npz
```

Run with:
```bash
--subject_dir_template "S{id}" --model_filename_template "net_{id}.h5"
```

---

## Environment

This is a pure-Python repo; ensure:

- Python 3.10+
- `tensorflow` / `tf_keras`
- `numpy`, `tqdm`

(Other libs from your workflow like `mne`, `pywt` are optional for this pipeline.)

---

## Quick start

1) **Put your data in place** (see *Data layout*).

2) **Run the pipeline**:
```bash
python scripts/run_pipeline.py   --input_base_dir ./data   --save_base_dir ./outputs   --subjects 1 2 3 4 5   --K 10   --split test
```

Key args:
- `--subjects`: list of subject IDs (required)
- `--split`: `train`, `test`, or `both` (default: `test`)
- `--K`: coalitions per cell (default: 10; try 64 for denser sampling)
- `--M`: number of cells (default: 16; i.e., 4 bands × 4 windows)

3) **Outputs** (per subject **S**):
```
./outputs/subject_S_results/
  coalitions_per_cell.npy
  win_lengths.npy
  rfftfreqs.npy
  phi_pair_cls0_j{j}[_test].npy
  phi_pair_cls1_j{j}[_test].npy
  phi_pair_L1mean_cls0_j{j}[_test].npy
  phi_pair_L1mean_cls1_j{j}[_test].npy
```

`_test` suffix appears when `--split test` or `--split both`.

---

