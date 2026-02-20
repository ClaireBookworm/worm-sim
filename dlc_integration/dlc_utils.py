"""
Load DeepLabCut outputs and convert keypoints to MuJoCo joint angles.

DLC bodyparts (from config): head-1, head-2, 1, 2, ..., 23, tail-1, tail-2 (27 total)
MuJoCo: 24 segments, 23 joints (dv + lr). For 2D top-down: use dorsoventral curvature.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List

# DLC bodypart order from config.yaml
DLC_BODYPARTS = [
    "head-1", "head-2",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23",
    "tail-1", "tail-2",
]

# Indices for 24 points along body (head → tail) for MuJoCo mapping
# Use: head-1, 1, 2, ..., 22, tail-1  (24 points)
DLC_TO_24_INDICES = [0] + list(range(2, 25)) + [25]  # head-1, 1..22, tail-1


def load_dlc_keypoints(
    csv_path: str,
    bodyparts: Optional[List[str]] = None,
    min_likelihood: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DLC output CSV and return keypoints.

    Args:
        csv_path: Path to DLC CSV (raw or filtered)
        bodyparts: Override bodypart list (default: DLC_BODYPARTS)
        min_likelihood: Replace (x,y) with NaN where likelihood < this

    Returns:
        keypoints: (T, N, 2) array of (x, y) positions, N=27
        likelihoods: (T, N) array
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

    # Handle multi-level columns: (scorer, bodypart, coords)
    if isinstance(df.columns, pd.MultiIndex):
        # Build (bodypart, coord) -> column index
        bp_to_cols = {}
        for col in df.columns:
            if len(col) >= 3:
                scorer, bp, coord = col[0], col[1], col[2]
            elif len(col) == 2:
                bp, coord = col[0], col[1]
            else:
                continue
            if bp not in bp_to_cols:
                bp_to_cols[bp] = {}
            bp_to_cols[bp][coord] = col
    else:
        raise ValueError("Expected DLC CSV with MultiIndex columns (scorer, bodypart, coords)")

    bodyparts = bodyparts or DLC_BODYPARTS
    T = len(df)
    N = len(bodyparts)
    keypoints = np.full((T, N, 2), np.nan, dtype=np.float64)
    likelihoods = np.full((T, N), np.nan, dtype=np.float64)

    for i, bp in enumerate(bodyparts):
        if bp not in bp_to_cols:
            continue
        cols = bp_to_cols[bp]
        if "x" in cols:
            keypoints[:, i, 0] = df[cols["x"]].values
        if "y" in cols:
            keypoints[:, i, 1] = df[cols["y"]].values
        if "likelihood" in cols:
            likelihoods[:, i] = df[cols["likelihood"]].values

    # Mask low-confidence points
    keypoints[likelihoods < min_likelihood] = np.nan

    return keypoints, likelihoods


def keypoints_to_joint_angles(
    keypoints: np.ndarray,
    n_output_points: int = 24,
    use_dlc_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Convert (T, N, 2) keypoints to (T, n_joints) joint angles.

    Joint angle = change in tangent direction between consecutive segments.
    For 24 points → 23 joint angles (curvature between segments).

    For 2D top-down view: bending is in the image plane (xy).
    MuJoCo joint_*_lr (axis z) bends in xy plane → use these for 2D.

    Args:
        keypoints: (T, N, 2) array of (x, y)
        n_output_points: Number of points along body (24 for MuJoCo)
        use_dlc_indices: Which DLC indices to use for the n points (default: head, 1-22, tail)

    Returns:
        joint_angles: (T, n_output_points-1) curvature angles in radians
    """
    T, N, _ = keypoints.shape

    if use_dlc_indices is None:
        # Map 27 DLC points to 24: head-1, 1..22, tail-1
        use_dlc_indices = [0] + list(range(2, 24)) + [25]

    # Extract and possibly interpolate to get exactly n_output_points
    pts = keypoints[:, use_dlc_indices, :]  # (T, n_pts, 2)
    n_pts = pts.shape[1]

    if n_pts != n_output_points:
        # Interpolate along body to get n_output_points
        s_out = np.linspace(0, 1, n_output_points, endpoint=True)
        s_in = np.linspace(0, 1, n_pts, endpoint=True)
        pts_new = np.zeros((T, n_output_points, 2))
        for t in range(T):
            for dim in range(2):
                vals = pts[t, :, dim]
                if np.isnan(vals).all():
                    pts_new[t, :, dim] = np.nan
                else:
                    valid = ~np.isnan(vals)
                    if valid.any():
                        pts_new[t, :, dim] = np.interp(s_out, s_in[valid], vals[valid])
        pts = pts_new
        n_pts = n_output_points

    # Forward-fill then backward-fill any remaining NaNs
    for t in range(T):
        for dim in range(2):
            col = pts[t, :, dim]
            if np.isnan(col).any():
                mask = np.isnan(col)
                col = col.copy()
                col[mask] = np.interp(
                    np.flatnonzero(mask),
                    np.flatnonzero(~mask),
                    col[~mask]
                )
                pts[t, :, dim] = col

    # Tangent angles: theta_i = atan2(dy, dx) for segment i to i+1
    # joint_i = theta_{i+1} - theta_i (curvature)
    angles = np.zeros((T, n_pts - 1))
    for i in range(n_pts - 1):
        dx = pts[:, i + 1, 0] - pts[:, i, 0]
        dy = pts[:, i + 1, 1] - pts[:, i, 1]
        angles[:, i] = np.arctan2(dy, dx)

    joint_angles = np.diff(angles, axis=1)  # (T, n_pts-2) curvature at each joint
    if joint_angles.shape[1] == 22:
        joint_angles = np.pad(joint_angles, ((0, 0), (0, 1)), constant_values=0)

    # Wrap to [-pi, pi]
    joint_angles = np.arctan2(np.sin(joint_angles), np.cos(joint_angles))

    return joint_angles


def keypoints_to_joint_angles_24_segments(
    keypoints: np.ndarray,
) -> np.ndarray:
    """
    Convert DLC keypoints to 23 joint angles for MuJoCo (24 segments).

    Uses 25 points from DLC: head-1, 1..23, tail-1.
    """
    use_dlc_indices = [0] + list(range(2, 25)) + [25]  # 25 points
    return keypoints_to_joint_angles(
        keypoints,
        n_output_points=25,
        use_dlc_indices=use_dlc_indices,
    )
