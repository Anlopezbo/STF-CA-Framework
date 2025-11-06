from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

# ================== Global parameters (unchanged logic) ==================
SAMPLING_RATE: float = 128.0
N_TIMES: int = 896
CUT_TIMES: List[float] = [0.0, 2.0, 2.5, 5.0, 7.0]   # 4 time windows
RHYTHMS: List[str] = ["Theta", "Alpha", "Beta", "Gamma"]
BAND_LIMITS: Dict[str, Tuple[float, float]] = {
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40),
}
DTYPE_COMPLEX = np.complex64

# 16-cell grid: 4 bands × 4 time windows
M_CELLS: int = 16

# Default time vector (optional)
TIME_VEC = np.arange(N_TIMES) / SAMPLING_RATE
