"""
Compute per-cell (M × C) φ maps using:
  - Kernel SHAP coalition weights
  - Uniform averaging (baseline)
on top of the saved phi_pair_L1mean_* artifacts produced by the pipeline.

Usage examples:
  python scripts/compute_weights.py \
    --roots "./outputs" \
    --split test \
    --M 16 \
    --save_suffix weights_test

  python scripts/compute_weights.py \
    --roots "./outputs/*" "./more_outputs/*" \
    --split train \
    --subjects "subject_1_results" "subject_5_results" \
    --save_suffix weights_train
"""

from __future__ import annotations
import os
import argparse
import numpy as np

from STF_CA.utils.subject_loader import load_multi_subjects
from STF_CA.framework.weights import aggregate_subject_phi


def parse_args():
    p = argparse.ArgumentParser(description="Compute kernel/uniform-weighted φ maps per subject.")
    p.add_argument("--roots", type=str, nargs="+", required=True,
                   help="One or more glob patterns pointing to results roots (each containing subject_*_results dirs).")
    p.add_argument("--split", type=str, required=True,
                   help="Which split to use (e.g., 'test' or 'train'). Must match suffix present in saved phi files.")
    p.add_argument("--M", type=int, default=16, help="Number of cells (default: 16).")
    p.add_argument("--subjects", type=str, nargs="*", default=None,
                   help="Optional explicit subject paths (absolute or relative to each root).")
    p.add_argument("--save_suffix", type=str, default=None,
                   help="Optional suffix for the saved .npz file (default: inferred from split, e.g., 'weights_test').")
    return p.parse_args()


def main():
    args = parse_args()

    loaded = load_multi_subjects(
        roots=args.roots,
        M=args.M,
        subjects=args.subjects,
        verbose=True,
    )
    subjects_raw = loaded["subjects_raw"]

    if args.save_suffix is None:
        save_suffix = f"weights_{args.split}"
    else:
        save_suffix = args.save_suffix

    for s in subjects_raw:
        subj_id = s["subject_id"]
        subj_dir = None
        # We need to reconstruct the path from any phi file we have, but loader didn't store it.
        # Easiest reliable approach: ask user to pass actual dirs in --subjects or use consistent roots.
        # Here we infer from coalitions file as a fallback:
        # Try common locations under roots to find subject folder again.
        # Simpler: we re-scan based on typical name.
        # However, since we only need to save next to existing files, we can store next to coalitions_per_cell.npy path.
        # In our loader we didn't keep path, so we compute from first available split artifact.

        # We'll reconstruct by searching up the tree of coalitions (cheap):
        # NOT ideal to rescan; instead, ask users to run from the subject dir.
        # Pragmatic: Accept that 'coalitions_per_cell' was loaded from some path, but we didn't retain it.
        # We'll choose a conventional output path using the convention './outputs/subject_{id}_results'.
        # For reliability, recommend running with --subjects providing full paths.

        # Safer approach: recompute the subject_dir by searching roots.
        found_dir = None
        for r in args.roots:
            import glob
            cands = glob.glob(os.path.join(r, f"subject_{subj_id}_results"))
            if cands:
                found_dir = cands[0]
                break
            cands = glob.glob(os.path.join(r, "*", f"subject_{subj_id}_results"))
            if cands:
                found_dir = cands[0]
                break

        if found_dir is None:
            print(f"[WARN] Could not locate folder for subject {subj_id} under roots={args.roots}. "
                  f"Use --subjects to pass explicit paths. Skipping.")
            continue

        subj_dir = found_dir

        if args.split not in s["splits"]:
            print(f"[WARN] Subject {subj_id}: split '{args.split}' not found in available {sorted(s['splits'])}. Skipping.")
            continue

        res = aggregate_subject_phi(s, args.split)  # dict with 4 arrays (M,C)

        # Save as a single NPZ next to original files
        out_path = os.path.join(subj_dir, f"{save_suffix}.npz")
        np.savez_compressed(
            out_path,
            kernel_cls0=res["kernel_cls0"],
            kernel_cls1=res["kernel_cls1"],
            uniform_cls0=res["uniform_cls0"],
            uniform_cls1=res["uniform_cls1"],
            split=args.split,
            M=args.M,
        )
        print(f"[S{subj_id}] saved: {out_path}")


if __name__ == "__main__":
    main()
