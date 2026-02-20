"""
Step 1: Replay DLC-derived poses in MuJoCo (kinematic).

Sets joint positions directly from DLC keypoints → joint angles.
For 2D top-down: bending is in xy plane → use joint_*_lr (axis z).
"""

import numpy as np
from pathlib import Path
from typing import Optional

from .dlc_utils import load_dlc_keypoints, keypoints_to_joint_angles_24_segments

try:
    import mujoco
except ImportError:
    mujoco = None


N_JOINTS = 23


def _get_joint_qpos_indices(model) -> tuple:
    """Return (dv_indices, lr_indices) for joint positions in qpos."""
    dv_adr = []
    lr_adr = []
    for i in range(1, N_JOINTS + 1):
        jid_dv = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}_dv")
        jid_lr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}_lr")
        dv_adr.append(model.jnt_qposadr[jid_dv])
        lr_adr.append(model.jnt_qposadr[jid_lr])
    return dv_adr, lr_adr


class MujocoDLCReplay:
    """
    Replay DLC poses in MuJoCo by setting joint positions.

    For 2D top-down view: bending in image plane = rotation around z = joint_*_lr.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_dorsoventral: bool = False,
    ):
        """
        Args:
            model_path: Path to worm_model.xml (default: worm-sim/worm_model.xml)
            use_dorsoventral: If True, use joint_*_dv (bends in xz). Default False = use _lr (xy).
        """
        if mujoco is None:
            raise ImportError("mujoco is required. pip install mujoco")

        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "worm_model.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.dv_adr, self.lr_adr = _get_joint_qpos_indices(self.model)
        self.use_dv = use_dorsoventral

    def set_pose_from_joint_angles(
        self,
        joint_angles: np.ndarray,
        root_pos: Optional[np.ndarray] = None,
        root_quat: Optional[np.ndarray] = None,
    ):
        """
        Set MuJoCo state from 23 joint angles (curvature).

        Args:
            joint_angles: (23,) curvature at each joint
            root_pos: (3,) position of head. Default [0,0,0]
            root_quat: (4,) quaternion. Default identity
        """
        joint_angles = np.asarray(joint_angles).flatten()
        if len(joint_angles) != N_JOINTS:
            joint_angles = np.pad(joint_angles, (0, max(0, N_JOINTS - len(joint_angles))))

        # Zero velocities for kinematic replay
        self.data.qvel[:] = 0

        # Root
        if root_pos is not None:
            self.data.qpos[0:3] = root_pos
        else:
            self.data.qpos[0:3] = [0, 0, 0]
        if root_quat is not None:
            self.data.qpos[3:7] = root_quat
        else:
            self.data.qpos[3:7] = [1, 0, 0, 0]

        # Joints: for 2D use _lr (xy plane), _dv = 0
        for i in range(N_JOINTS):
            if self.use_dv:
                self.data.qpos[self.dv_adr[i]] = joint_angles[i]
                self.data.qpos[self.lr_adr[i]] = 0
            else:
                self.data.qpos[self.dv_adr[i]] = 0
                self.data.qpos[self.lr_adr[i]] = joint_angles[i]

        mujoco.mj_forward(self.model, self.data)

    def replay_from_dlc_csv(
        self,
        csv_path: str,
        scale: float = 0.001,
        fps: float = 30,
        render: bool = False,
        max_frames: Optional[int] = None,
    ):
        """
        Load DLC CSV and replay poses frame by frame.

        Args:
            csv_path: DLC output CSV
            scale: Pixel-to-meter scale (default 1mm per 1000px)
            fps: Frames per second for timing
            render: Whether to show viewer
            max_frames: Limit frames (default: all)
        """
        keypoints, _ = load_dlc_keypoints(csv_path)
        joint_angles = keypoints_to_joint_angles_24_segments(keypoints)

        if max_frames:
            joint_angles = joint_angles[:max_frames]
            keypoints = keypoints[:max_frames]

        viewer = None
        if render:
            try:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except (ImportError, AttributeError):
                try:
                    viewer = mujoco.launch_passive(self.model, self.data)
                except (ImportError, AttributeError):
                    print(
                        "MuJoCo viewer not available. Install with: pip install mujoco[viewer]\n"
                        "On macOS, use: mjpython -m dlc_integration.csv_to_mujoco_replay ... --render"
                    )
                    viewer = None

        dt = 1.0 / fps
        for t, angles in enumerate(joint_angles):
            # Root position from first keypoint (head) - use head-1
            head_pt = keypoints[t, 0, :]  # head-1
            if not np.isnan(head_pt).any():
                root_pos = np.array([head_pt[0] * scale, head_pt[1] * scale, 0])
            else:
                root_pos = np.array([0, 0, 0])

            self.set_pose_from_joint_angles(angles, root_pos=root_pos)

            if viewer is not None:
                viewer.sync()
                import time
                time.sleep(dt)

        if viewer is not None:
            viewer.close()


def run_replay_cli():
    """CLI for replaying DLC poses in MuJoCo."""
    import argparse
    parser = argparse.ArgumentParser(description="Replay DLC poses in MuJoCo")
    parser.add_argument("csv", help="DLC output CSV (filtered)")
    parser.add_argument("--model", default=None, help="worm_model.xml path")
    parser.add_argument("--scale", type=float, default=0.001, help="Pixel to meter scale")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    replay = MujocoDLCReplay(model_path=args.model)
    replay.replay_from_dlc_csv(
        args.csv,
        scale=args.scale,
        fps=args.fps,
        render=args.render,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    run_replay_cli()
