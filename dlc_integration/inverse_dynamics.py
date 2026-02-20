"""
Step 2a: Inverse dynamics - find muscle activations that produce a target pose.

Given target joint angles (from DLC), optimize muscle activations so that
running MuJoCo to equilibrium yields that pose.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy.optimize import minimize

from .dlc_utils import load_dlc_keypoints, keypoints_to_joint_angles_24_segments

try:
    import mujoco
except ImportError:
    mujoco = None

N_JOINTS = 23
N_MUSCLES = 96

# Simplified: 24 "net bend" muscles (one per segment)
# bend_i = (DR_i + DL_i) - (VR_i + VL_i) → map to 24 values in [0,1], 0.5=neutral
N_SIMPLIFIED_MUSCLES = 24


def _simplified_to_full_muscles(simplified: np.ndarray) -> np.ndarray:
    """Convert 24 simplified bend signals to 96 full muscle activations."""
    simplified = np.clip(np.asarray(simplified).flatten(), 0, 1)
    full = np.zeros(96)
    for i in range(24):
        bend = simplified[i]  # 0=ventral, 0.5=neutral, 1=dorsal
        dorsal = max(bend - 0.5, 0) * 2
        ventral = max(0.5 - bend, 0) * 2
        full[i] = dorsal      # DR
        full[i + 24] = dorsal  # DL
        full[i + 48] = ventral  # VR
        full[i + 72] = ventral  # VL
    return np.clip(full, -1, 1)  # MuJoCo motors use [-1, 1]


def _get_joint_qpos_indices(model, use_lr: bool = True):
    """Return list of qpos indices for joint angles (lr for 2D, dv for 3D)."""
    adr = []
    for i in range(1, N_JOINTS + 1):
        jname = f"joint_{i}_lr" if use_lr else f"joint_{i}_dv"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr.append(model.jnt_qposadr[jid])
    return adr


def find_muscles_for_pose(
    target_angles: np.ndarray,
    model_path: Optional[str] = None,
    n_muscles: int = 24,
    n_steps_equilibrium: int = 50,
    use_lr: bool = True,
) -> np.ndarray:
    """
    Optimize muscle activations to achieve target joint angles.

    Args:
        target_angles: (23,) target curvature angles
        model_path: Path to worm_model.xml
        n_muscles: 24 (simplified) or 96 (full)
        n_steps_equilibrium: Physics steps to reach equilibrium
        use_lr: Use joint_*_lr for 2D (default True)

    Returns:
        muscle_activations: (24,) or (96,)
    """
    if mujoco is None:
        raise ImportError("mujoco required")

    if model_path is None:
        model_path = str(Path(__file__).parent.parent / "worm_model.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    jnt_adr = _get_joint_qpos_indices(model, use_lr=use_lr)
    target_angles = np.asarray(target_angles).flatten()[:N_JOINTS]

    def pose_error(x: np.ndarray) -> float:
        if n_muscles == 24:
            ctrl = _simplified_to_full_muscles(x)
        else:
            ctrl = np.clip(x, -1, 1)
        data.ctrl[:] = ctrl

        mujoco.mj_resetData(model, data)
        data.qvel[:] = 0

        for _ in range(n_steps_equilibrium):
            mujoco.mj_step(model, data)

        current = np.array([data.qpos[adr] for adr in jnt_adr])
        err = np.mean((current - target_angles) ** 2)
        return float(err)

    if n_muscles == 24:
        x0 = np.ones(24) * 0.5
        bounds = [(0, 1)] * 24
    else:
        x0 = np.zeros(96)
        bounds = [(-1, 1)] * 96

    result = minimize(
        pose_error,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100, "disp": False},
    )

    if n_muscles == 24:
        return np.clip(result.x, 0, 1)
    return np.clip(result.x, -1, 1)


def collect_pose_muscle_pairs(
    dlc_csv_path: str,
    model_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    stride: int = 1,
    n_muscles: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each DLC frame, compute joint angles and find optimal muscles.

    Returns:
        poses: (T, 23) joint angles
        muscles: (T, 24) or (T, 96) muscle activations
    """
    keypoints, _ = load_dlc_keypoints(dlc_csv_path)
    joint_angles = keypoints_to_joint_angles_24_segments(keypoints)

    if max_frames:
        joint_angles = joint_angles[:max_frames]
    if stride > 1:
        joint_angles = joint_angles[::stride]

    poses = []
    muscles = []
    T = len(joint_angles)
    for t, angles in enumerate(joint_angles):
        if t % 50 == 0:
            print(f"  Inverse dynamics frame {t}/{T}")
        m = find_muscles_for_pose(angles, model_path=model_path, n_muscles=n_muscles)
        poses.append(angles)
        muscles.append(m)

    return np.array(poses), np.array(muscles)
