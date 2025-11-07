from __future__ import annotations
import os, re, glob
from typing import List, Optional, Tuple

# Matches .../subject_{id}_results
SUBJ_DIR_NAME_RGX = re.compile(r"subject_(\d+)_results$")


def parse_subject_id(subject_dir: str) -> Optional[int]:
    """
    Extract subject ID from a directory name like 'subject_{id}_results'.
    """
    m = SUBJ_DIR_NAME_RGX.search(os.path.basename(subject_dir))
    return int(m.group(1)) if m else None


def _sort_by_subject_id(paths: List[str]) -> List[str]:
    """
    Sort paths by subject id, with non-parsable dirs at the end.
    """
    def _key(p: str):
        sid = parse_subject_id(p)
        return (sid if sid is not None else 10**9, p)
    return sorted(paths, key=_key)


def find_subject_dirs(root: str) -> List[str]:
    """
    Single-root discovery. Looks for '{root}/subject_*_results'.
    """
    cand = glob.glob(os.path.join(root, "subject_*_results"))
    return _sort_by_subject_id(cand)


def find_subject_dirs_from_globs(globs_or_str) -> List[str]:
    """
    Discover subject directories from one or many glob patterns.

    Accepts either:
      - A single glob string, or
      - A list/tuple of glob strings

    Each pattern may target either:
      a) .../eegnet_phi_results
         (we append 'subject_*_results' internally), or
      b) .../eegnet_phi_results/subject_*_results
         (used as-is)
    """
    if isinstance(globs_or_str, (str, bytes)):
        patterns = [globs_or_str]
    else:
        patterns = list(globs_or_str)

    found: List[str] = []
    for pat in patterns:
        if pat.rstrip("/").endswith("subject_*_results"):
            found.extend(glob.glob(pat))
        else:
            found.extend(glob.glob(os.path.join(pat, "subject_*_results")))

    # Order & deduplicate by subject id
    found = _sort_by_subject_id(found)
    uniq = {}
    for p in found:
        sid = parse_subject_id(p)
        if sid is not None and sid not in uniq:
            uniq[sid] = p
    return [uniq[k] for k in sorted(uniq.keys())]
