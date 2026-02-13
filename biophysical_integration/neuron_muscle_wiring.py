"""
Neuron-muscle wiring and muscle ordering.

Different muscle orderings exist in C. elegans models:
- Standard biological order: DR0-23, VR0-23, DL0-23, VL0-23 (dorsal right, ventral right, dorsal left, ventral left)
- worm_sim (MuJoCo) order: DR[0-23], DL[0-23], VR[0-23], VL[0-23]

This module provides:
- Permutation to convert between muscle orderings
- Optional load of neuron_muscle.xlsx for sparse readout mask (biological connectivity)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

# Default path to neuron_muscle.xlsx (biological connectivity matrix)
DEFAULT_NEURON_MUSCLE_PATH = (
    PROJECT_ROOT / "BAAIWorm" / "eworm_learn" / "components" / "param" / "connection" / "neuron_muscle.xlsx"
)

# Muscle ordering permutation:
# Standard biological order: DR(0-23), VR(24-47), DL(48-71), VL(72-95)
# worm_sim (MuJoCo) order:   DR(0-23), DL(24-47), VR(48-71), VL(72-95)
#
# This permutation converts from standard biological order to worm_sim order:
# worm_sim[0:24] = bio[0:24] (DR unchanged)
# worm_sim[24:48] = bio[48:72] (DL from bio position 48-71)
# worm_sim[48:72] = bio[24:48] (VR from bio position 24-47)
# worm_sim[72:96] = bio[72:96] (VL unchanged)
BIOLOGICAL_TO_WORM_SIM = np.array(
    list(range(24)) + list(range(48, 72)) + list(range(24, 48)) + list(range(72, 96)),
    dtype=np.int64,
)

# Keep old name for backwards compatibility
BAAIWORM_TO_WORM_SIM = BIOLOGICAL_TO_WORM_SIM


def apply_muscle_order_permutation(muscles: np.ndarray, from_biological: bool = True) -> np.ndarray:
    """
    Convert muscle array between biological ordering and worm_sim (MuJoCo) ordering.

    Args:
        muscles: (96,) or (T, 96)
        from_biological: If True, input is biological order (DR,VR,DL,VL), output worm_sim order (DR,DL,VR,VL).
                         If False, input is worm_sim order, output biological order.
    """
    muscles = np.asarray(muscles)
    if from_biological:
        # bio → worm_sim: apply inverse permutation
        inv = np.argsort(BIOLOGICAL_TO_WORM_SIM)
        return muscles[..., inv] if muscles.ndim > 1 else muscles[inv]
    else:
        # worm_sim → bio: apply forward permutation
        return muscles[..., BIOLOGICAL_TO_WORM_SIM] if muscles.ndim > 1 else muscles[BIOLOGICAL_TO_WORM_SIM]


def load_neuron_muscle_mask(
    neuron_names: list,
    muscle_order: str = "worm_sim",
    path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Load neuron-muscle connectivity mask from neuron_muscle.xlsx.

    Returns (n_neurons, 96) boolean mask: True where connection exists.
    Returns None if file not found or read fails.

    Args:
        neuron_names: List of neuron names (e.g. motor neuron names in our order)
        muscle_order: "worm_sim" (default) or "biological" - output mask matches this order
        path: Path to neuron_muscle.xlsx
    """
    p = Path(path) if path else DEFAULT_NEURON_MUSCLE_PATH
    if not p.exists():
        return None

    try:
        import pandas as pd
    except ImportError:
        return None

    try:
        df = pd.read_excel(p, sheet_name="Sheet1", header=0)
        matrix_neuron_names = df.iloc[:, 0].astype(str).tolist()

        # Build mask: for each neuron in neuron_names, for each muscle 0-95
        n_neurons = len(neuron_names)
        n_muscles = min(96, df.shape[1] - 1)
        mask = np.zeros((n_neurons, 96), dtype=bool)

        name_to_row = {str(n): i for i, n in enumerate(matrix_neuron_names)}
        for j, nn in enumerate(neuron_names):
            if nn not in name_to_row:
                continue
            row_idx = name_to_row[nn]
            for m in range(n_muscles):
                val = df.iloc[row_idx, m + 1]
                if pd.notna(val) and str(val).strip():
                    mask[j, m] = True

        # Excel columns are in biological order (DR, VR, DL, VL)
        # Convert to worm_sim order if requested
        if muscle_order == "worm_sim":
            mask = mask[:, BIOLOGICAL_TO_WORM_SIM]
        return mask
    except Exception:
        return None


def get_wiring_mask_for_motor_neurons(
    motor_neuron_names: list,
    path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Convenience: load mask for our 27 motor neurons, worm_sim muscle order."""
    return load_neuron_muscle_mask(motor_neuron_names, muscle_order="worm_sim", path=path)
