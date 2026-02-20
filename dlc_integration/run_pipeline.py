#!/usr/bin/env python3
"""
Run the full DLC → MuJoCo pipeline.

Use --output-dir to specify where to save. Overwrites existing files.

Usage (run from worm-sim):
  python -m dlc_integration.run_pipeline collect DLC_filtered.csv --output-dir my_run
  python -m dlc_integration.run_pipeline train my_run/poses_muscles.npz --output-dir my_run
"""

import argparse
from pathlib import Path

import numpy as np

from .paths import output_dir_from_input_path


def cmd_replay(args):
    from .mujoco_replay import MujocoDLCReplay
    replay = MujocoDLCReplay(model_path=args.model)
    replay.replay_from_dlc_csv(
        args.csv,
        scale=args.scale,
        fps=args.fps,
        render=args.render,
        max_frames=args.max_frames,
    )


def cmd_collect(args):
    from .inverse_dynamics import collect_pose_muscle_pairs
    poses, muscles = collect_pose_muscle_pairs(
        args.csv,
        model_path=args.model,
        max_frames=args.max_frames,
        stride=args.stride,
        n_muscles=24,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "poses_muscles.npz"
    np.savez(out, poses=poses, muscles=muscles)
    print(f"Saved {len(poses)} pairs to {out}")


def cmd_train(args):
    from .pose_muscle_controller import MuscleController, train_pose_muscle_controller
    data = np.load(args.data)
    poses = data["poses"]
    muscles = data["muscles"]
    print(f"Training on {len(poses)} samples")
    print(f"  Input: poses {poses.shape} -> muscles {muscles.shape}")
    if len(poses) < 50:
        print(f"  Note: <50 samples may overfit. Consider: collect ... --max-frames 500 --stride 2")
    model = train_pose_muscle_controller(
        poses, muscles,
        epochs=args.epochs,
        lr=args.lr,
    )
    out_dir = Path(args.output_dir) if args.output_dir else output_dir_from_input_path(args.data)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "controller.pt"
    import torch
    torch.save(model.state_dict(), out)
    print(f"Saved controller to {out}")


def main():
    parser = argparse.ArgumentParser(description="DLC → MuJoCo pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # replay
    p_replay = sub.add_parser("replay", help="Step 1: Replay DLC poses in MuJoCo")
    p_replay.add_argument("csv", help="DLC filtered CSV")
    p_replay.add_argument("--model", default=None)
    p_replay.add_argument("--scale", type=float, default=0.001)
    p_replay.add_argument("--fps", type=float, default=30)
    p_replay.add_argument("--max-frames", type=int, default=None)
    p_replay.add_argument("--render", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    # collect
    p_collect = sub.add_parser("collect", help="Step 2: Inverse dynamics, collect pose-muscle pairs")
    p_collect.add_argument("csv", help="DLC filtered CSV")
    p_collect.add_argument("--output-dir", required=True, help="Folder for poses_muscles.npz (overwrites if exists)")
    p_collect.add_argument("--model", default=None)
    p_collect.add_argument("--max-frames", type=int, default=None)
    p_collect.add_argument("--stride", type=int, default=1)
    p_collect.set_defaults(func=cmd_collect)

    # train
    p_train = sub.add_parser("train", help="Step 3: Train pose → muscle controller")
    p_train.add_argument("data", help="poses_muscles.npz from collect")
    p_train.add_argument("--output-dir", help="Folder for controller.pt (default: same dir as input file)")
    p_train.add_argument("--epochs", type=int, default=1000)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
