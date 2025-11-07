# STF-CA-FRAMEWORK Pipeline (windowed FFT, band × time cells)

> Developed by [Andrés Camilo López Boscán, MSc.](https://github.com/Anlopezbo)
[Diego Armando Perez Rosero, PhD.](https://github.com/Daprosero)  
[Andrés Marino Álvarez Meza, PhD.](https://github.com/amalvarezme)  
César Germán Castellanos Dominguez, PhD.  
> _Digital Signal Processing and Control Group_  | _Grupo de Control y Procesamiento Digital de Señales ([GCPDS](https://github.com/UN-GCPDS/))_  
> _Universidad Nacional de Colombia sede Manizales_

----

This repository computes **φ (phi) attributions** for EEG using a **windowed FFT** backend. Each attribution targets one of **16 cells** (4 frequency bands × 4 time windows). The math/logic is **unchanged** from the original script:

- Windows: **CUT_TIMES = [0, 2, 2.5, 5, 7]** seconds (4 segments)  
- Bands: **Theta [4–8), Alpha [8–13), Beta [13–30), Gamma [30–40)**  
- Reconstruction: rFFT/iRFFT **per window**, with **cell-wise frequency masking**  
- Attribution: **pairwise φ** per coalition, **channel-wise replacement**:  
  \|F(X_{S∪{j}}) − F(X_S)\|, then mean across classes if needed

---

## Features

- Coalition generation per target cell **j** (S vs S ∪ {j}), no duplicates
- Exact band×window masks (16 cells total)
- Windowed rFFT/iRFFT reconstruction with OFF masks
- Channel-wise φ computation; helper to average across classes
- **Weighting step (DELETION notebook logic)**: kernel-SHAP vs uniform coalition weighting
- CLI runner with **generic, reusable paths** and **naming templates**

---

## Repository layout

```
STF-CA-FRAMEWORK/
├─ STF_CA/
│  ├─ framework/          # domain core
│  │  ├─ coalitions.py        # CoalitionGenerator (S vs S∪{j})
│  │  ├─ fft_backend.py       # windows, masks, sFFT/iFFT, recon
│  │  ├─ phi.py               # PhiComputer (pairwise φ)
│  │  └─ weights.py           # kernel/uniform weighting & normalization
│  │
│  ├─ utils/              # cross-cutting helpers
│  │  ├─ config.py            # constants: bands, cut times, etc.
│  │  ├─ io_utils.py          # load/save for pipeline
│  │  ├─ subject_discovery.py # find subject_*_results dirs
│  │  └─ subject_loader.py    # load φ artifacts per subject / multi-subject
│  │
│  ├─ __init__.py
│  └─ py.typed
│
├─ scripts/
│  ├─ run_pipeline.py     # run the FFT-mask pipeline (produces φ artifacts)
│  └─ compute_weights.py  # apply kernel/uniform weights and save (M×C) maps
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
- (Optional for your workflow) `mne`, `pywt`, `ssqueezepy`

A sample `requirements.txt` is provided.

---

## Quick start (pipeline)

1) **Put your data in place** (see *Data layout*).

2) **Run the pipeline** to produce φ artifacts per subject:
```bash
python scripts/run_pipeline.py \
  --input_base_dir ./data \
  --save_base_dir ./outputs \
  --subjects 1 2 3 4 5 \
  --K 10 \
  --split test
```

Key args:
- `--subjects`: list of subject IDs (required)
- `--split`: `train`, `test`, or `both` (default: `test`)
- `--K`: coalitions per cell (default: 10; try 64 for denser sampling)
- `--M`: number of cells (default: 16; i.e., 4 bands × 4 windows)

3) **Pipeline outputs** (per subject **S**):
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

## Weighted aggregation (kernel-SHAP vs uniform)

After the pipeline, you can aggregate φ across coalitions using **kernel-SHAP weights** or a simple **uniform mean**, then average across trials **per cell** to obtain (M × C) maps that are **jointly min–max normalized** by class pair.

### Kernel weight
For total cells **M** and coalition size **s** (excluding the target cell j):
\[
w(s) = \frac{M-1}{\binom{M-1}{s}\; s \; (M-1-s)}, \quad s \in \{1,\dots,M-2\}
\]
and \(w(0) = w(M-1) = 0\).

### CLI
```bash
python scripts/compute_weights.py \
  --roots "./outputs" \
  --split test \
  --M 16 \
  --save_suffix weights_test
```

- `--roots` accepts one or multiple glob patterns (e.g., `"./outputs"`, `"./outputs/*"`).
- You may specify exact subject directories via `--subjects` (absolute or relative to each root).

### Weighting outputs
For each subject **S**, the script saves a single NPZ alongside the φ files:
```
./outputs/subject_S_results/weights_test.npz
  kernel_cls0   # (M, C)  kernel-weighted, normalized
  kernel_cls1   # (M, C)
  uniform_cls0  # (M, C)  uniform-mean, normalized
  uniform_cls1  # (M, C)
  split         # e.g., "test"
  M             # e.g., 16
```

### Programmatic API
```python
from STF_CA.utils.subject_loader import load_multi_subjects
from STF_CA.framework.weights import aggregate_subject_phi

loaded = load_multi_subjects(roots="./outputs", M=16, verbose=True)
S0 = loaded["subjects_raw"][0]       # dict for one subject
res = aggregate_subject_phi(S0, split="test")
phi_kernel_cls0 = res["kernel_cls0"] # (M, C)
```

---


