"""
DLC ↔ MuJoCo integration for C. elegans pose replay and muscle controller learning.

Pipeline:
  Step 1: Video → DLC keypoints → joint angles → MuJoCo kinematic replay
  Step 2: For each pose: inverse dynamics → muscle activations → train NN controller
  Step 3: Closed-loop: current_pose → NN → muscles → MuJoCo
"""

from .dlc_utils import (
    load_dlc_keypoints,
    keypoints_to_joint_angles,
    keypoints_to_joint_angles_24_segments,
)
from .mujoco_replay import MujocoDLCReplay
from .inverse_dynamics import find_muscles_for_pose, collect_pose_muscle_pairs

try:
    from .pose_muscle_controller import MuscleController, train_pose_muscle_controller
except ImportError:
    MuscleController = None
    train_pose_muscle_controller = None

__all__ = [
    "load_dlc_keypoints",
    "keypoints_to_joint_angles",
    "keypoints_to_joint_angles_24_segments",
    "MujocoDLCReplay",
    "find_muscles_for_pose",
    "collect_pose_muscle_pairs",
    "MuscleController",
    "train_pose_muscle_controller",
]
