#!/usr/bin/env python3
"""
Read DLC inference CSV files and convert to MuJoCo replay.

Usage (from worm-sim, with venv activated):
  cd worm-sim && source .venv/bin/activate

  # Replay a single CSV
  python -m dlc_integration.csv_to_mujoco_replay \
    ../relabeled-celegans-claire-2025-10-26/videos/inference_outputs/nir_video_2023_03_07DLC_Resnet50_relabeled-celegansOct26shuffle6_snapshot_best-470_filtered.csv \
    --render

  # Process all CSVs in inference_outputs folder
  python -m dlc_integration.csv_to_mujoco_replay \
    ../relabeled-celegans-claire-2025-10-26/videos/inference_outputs \
    --render --max-frames 100

  # Save segments (joint angles) to a folder
  python -m dlc_integration.csv_to_mujoco_replay INPUT.csv --output --output-dir my_run
"""

import argparse
from pathlib import Path
import sys

from .dlc_utils import load_dlc_keypoints, keypoints_to_joint_angles_24_segments
from .mujoco_replay import MujocoDLCReplay


def find_dlc_csvs(path: Path) -> list[Path]:
    """Find DLC CSV files. Prefer *_filtered.csv, else any *.csv."""
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []
    csvs = list(path.glob("*_filtered.csv"))
    if not csvs:
        csvs = list(path.glob("*.csv"))
    return sorted(csvs)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DLC inference CSVs to MuJoCo segments and replay"
    )
    parser.add_argument(
        "input",
        help="Path to a DLC CSV file or folder (e.g. inference_outputs/)",
    )
    parser.add_argument(
        "--output", "-o",
        action="store_true",
        help="Save joint angles to segments.npy",
    )
    parser.add_argument(
        "--output-dir",
        help="Folder for outputs (required with --output). Overwrites if exists.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Replay in MuJoCo viewer",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to process",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.001,
        help="Pixel to meter scale (default 0.001 = 1mm per 1000px)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Frames per second for replay",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to worm_model.xml (default: worm-sim/worm_model.xml)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    csvs = find_dlc_csvs(input_path)

    if not csvs:
        print(f"No DLC CSV files found at {input_path}")
        sys.exit(1)

    print(f"Found {len(csvs)} CSV file(s)")

    for csv_path in csvs:
        print(f"\n--- {csv_path.name} ---")
        keypoints, likelihoods = load_dlc_keypoints(str(csv_path))
        joint_angles = keypoints_to_joint_angles_24_segments(keypoints)

        if args.max_frames:
            joint_angles = joint_angles[: args.max_frames]
            keypoints = keypoints[: args.max_frames]

        print(f"  Frames: {len(joint_angles)}, Joint angles shape: {joint_angles.shape}")

        if args.output:
            if not args.output_dir:
                sys.exit("--output requires --output-dir")
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if len(csvs) > 1:
                stem = csv_path.stem.replace("_filtered", "")
                out_path = out_dir / f"segments_{stem}.npy"
            else:
                out_path = out_dir / "segments.npy"
            import numpy as np
            np.save(out_path, joint_angles)
            print(f"  Saved segments to {out_path}")

        if args.render:
            replay = MujocoDLCReplay(model_path=args.model)
            replay.replay_from_dlc_csv(
                str(csv_path),
                scale=args.scale,
                fps=args.fps,
                render=True,
                max_frames=args.max_frames,
            )


if __name__ == "__main__":
    main()
