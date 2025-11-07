from __future__ import annotations
import os, re, glob
import numpy as np
from typing import Dict, List, Optional, Any
from .subject_discovery import parse_subject_id, find_subject_dirs_from_globs


def load_subject_phi(subject_dir: str, M: int = 16, verbose: bool = True) -> Dict[str, Any]:
    """
    Load per-subject artifacts needed for weighting/aggregation (from outputs folder).

    Expected files inside subject_dir (produced by your pipeline):
      - coalitions_per_cell.npy
      - phi_pair_L1mean_cls{0|1}_j{j}_{split}.npy  (for j in 0..M-1)

    Returns:
      dict with:
        subject_id: int
        M: int
        coalitions_per_cell: dict
        phi_cls0: {split: list[M] of np.ndarray(K, N0, C)}
        phi_cls1: {split: list[M] of np.ndarray(K, N1, C)}
        splits: set[str]
    """
    subj_id = parse_subject_id(subject_dir)
    out: Dict[str, Any] = {
        "subject_id": subj_id,
        "M": M,
        "coalitions_per_cell": None,
        "phi_cls0": {},
        "phi_cls1": {},
        "splits": set(),
    }

    # coalitions_per_cell
    cpath = os.path.join(subject_dir, "coalitions_per_cell.npy")
    if os.path.exists(cpath):
        out["coalitions_per_cell"] = np.load(cpath, allow_pickle=True).item()
        if verbose:
            print(f"[S{subj_id}] coalitions_per_cell loaded ({len(out['coalitions_per_cell'])} cells)")
    else:
        if verbose:
            print(f"[S{subj_id}] WARNING: coalitions_per_cell.npy not found")

    # Files like: phi_pair_L1mean_cls0_j2_test.npy
    pattern = os.path.join(subject_dir, "phi_pair_L1mean_cls*_j*_*[.]npy")
    files = glob.glob(pattern)
    rgx = re.compile(r"phi_pair_L1mean_cls([01])_j(\d+)_([A-Za-z0-9\-]+)\.npy$")

    for f in files:
        m = rgx.search(os.path.basename(f))
        if not m:
            continue
        cls = int(m.group(1))
        j = int(m.group(2))
        split = m.group(3)
        if j >= M:
            continue
        out["splits"].add(split)
        if cls == 0:
            out["phi_cls0"].setdefault(split, [None]*M)
            out["phi_cls0"][split][j] = np.load(f)
        else:
            out["phi_cls1"].setdefault(split, [None]*M)
            out["phi_cls1"][split][j] = np.load(f)

    if verbose:
        print(f"[S{subj_id}] detected splits: {sorted(out['splits'])}")

    return out


def load_multi_subjects(
    roots=None,
    M: int = 16,
    subjects: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Multi-subject loader.

    Args:
      roots: str or list[str] of glob patterns (e.g., './outputs/*/subject_*_results' or './outputs/*')
      subjects: specific subject paths (absolute or relative to each root). If provided, we respect them.

    Returns:
      {
        "subjects_raw": [load_subject_phi(...) dicts],
        "all_splits":  sorted list of all splits found across subjects
      }
    """
    if roots is None:
        raise ValueError("You must provide 'roots' as a glob or a list of globs to discover multiple datasets.")

    # Use explicit subject paths if provided
    if subjects is not None:
        subject_dirs = []
        for p in subjects:
            if os.path.isabs(p):
                subject_dirs.append(p)
            else:
                # Try match under each root
                roots_iter = [roots] if isinstance(roots, (str, bytes)) else roots
                found = False
                for r in roots_iter:
                    import glob as _glob
                    cand = _glob.glob(os.path.join(r, p))
                    if cand:
                        subject_dirs.append(cand[0])
                        found = True
                        break
                if not found and verbose:
                    print(f"[WARN] Relative path not found: {p}")
        subject_dirs = find_subject_dirs_from_globs(subject_dirs)
    else:
        # Discover from patterns
        subject_dirs = find_subject_dirs_from_globs(roots)

    subjects_raw = [load_subject_phi(sd, M=M, verbose=verbose) for sd in subject_dirs]

    all_splits = set()
    for s in subjects_raw:
        all_splits |= s["splits"]

    return {"subjects_raw": subjects_raw, "all_splits": sorted(all_splits)}
