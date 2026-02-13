#!/usr/bin/env python3
"""
Run closed-loop simulation with biophysical RNN integration.

Run with: source .venv/bin/activate  (then run this script)

Usage:
  # With mock model (no checkpoint needed):
  python run_biophysical_sim.py --mock --render

  # With trained muscle readout (still mock neural model until you integrate Jaxley):
  python run_biophysical_sim.py --mock --readout biophysical_integration/checkpoints/muscle_readout.npz --render

  # With real biophysical model (once integrated):
  python run_biophysical_sim.py --checkpoint /path/to/ckpt_epoch_5000.pkl --readout /path/to/muscle_readout.npz\

python run_biophysical_sim.py \
  --checkpoint models/Onewindow_polyak_trainableinput_tbptt_voltage_corr_new2/input_trainable_voltage_biophys \
  --net-cache models/Onewindow_polyak_trainableinput_tbptt_voltage_corr_new2/input_trainable_voltage_biophys/network_synapse_location_always_soma.pkl \
  --checkpoint-epoch 5000 \
  --readout biophysical_integration/checkpoints/muscle_readout.npz \
  --render

"""

import argparse
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))

N_NEURONS = 58
N_MOTOR_NEURONS = 27
N_MUSCLES = 96

from worm_sim import WormConfig
from biophysical_integration.biophysical_wrapper import (
    BiophysicalRNNWrapper,
    BiophysicalConfig,
    MockBiophysicalRNN,
    run_closed_loop,
)
from biophysical_integration.muscle_readout import LinearMuscleReadout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock biophysical model (no checkpoint)")
    parser.add_argument("--checkpoint", type=str, help="Directory with ckpt_epoch_*.pkl (e.g. models/)")
    parser.add_argument("--checkpoint-epoch", type=int, default=None, help="Load specific epoch (e.g. 5000). Default: latest.")
    parser.add_argument("--net-cache", type=str, help="Path to network pickle (required for --jaxley)")
    parser.add_argument("--readout", type=str, help="Path to trained muscle readout .npz")
    parser.add_argument("--render", action="store_true",
                        help="Render simulation (on macOS use: mjpython run_biophysical_sim.py ... --render)")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--ca-cell-names", type=str, default=None,
                        help="Path to Ca_traces_cell_name.txt (default: BAAIWorm/.../Ca_traces_cell_name.txt)")
    parser.add_argument("--motor-only", action="store_true",
                        help="Use only motor neurons (~28) for muscle readout (WormAtlas/MoW classification)")
    parser.add_argument("--save-output", type=str, default=None,
                        help="Save trajectory, muscles, food_concentration to this directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print muscle stats (mean, std, min, max) every 100 steps")
    parser.add_argument("--food-x", type=float, default=2.0, help="Food position x")
    parser.add_argument("--food-y", type=float, default=0.0)
    parser.add_argument("--food-z", type=float, default=0.0)
    parser.add_argument("--save-video", type=str, default=None,
                        help="Save simulation to MP4 video (e.g. output.mp4). Needs: pip install imageio imageio-ffmpeg")
    args = parser.parse_args()

    config = BiophysicalConfig()

    # Muscle readout (n_neurons = 28 if motor_only else 58)
    n_readout_neurons = N_MOTOR_NEURONS if args.motor_only else N_NEURONS
    muscle_readout = None
    if args.readout and Path(args.readout).exists():
        muscle_readout = LinearMuscleReadout(n_readout_neurons, N_MUSCLES)
        muscle_readout.load(args.readout)
        print(f"Loaded muscle readout from {args.readout} (expects {muscle_readout.n_neurons} neurons)")

    # Biophysical model and sensory mapping
    sensory_mapping = None
    biophysical_model = None

    if args.mock:
        biophysical_model = MockBiophysicalRNN(n_neurons=N_NEURONS)
        print("Using mock biophysical model")
    elif args.checkpoint and args.net_cache:
        try:
            from biophysical_integration.jaxley_loader import JaxleyBiophysicalRNN
            from biophysical_integration.sensory_mapping import HeuristicSensoryMapping
            biophysical_model = JaxleyBiophysicalRNN(
                checkpoint_dir=args.checkpoint,
                net_cache=args.net_cache,
                checkpoint_epoch=args.checkpoint_epoch,
                ca_cell_names_path=args.ca_cell_names,
                motor_only=args.motor_only,
            )
            sensory_mapping = HeuristicSensoryMapping()
            print(f"Loaded Jaxley model (jaxley_worm, {biophysical_model.n_sensory} sensory neurons)")
        except Exception as e:
            raise RuntimeError(f"Failed to load Jaxley model: {e}") from e
    else:
        biophysical_model = MockBiophysicalRNN(n_neurons=N_NEURONS)
        print("Using mock biophysical model (default)")

    wrapper = BiophysicalRNNWrapper(
        config=config,
        sensory_mapping=sensory_mapping,
        muscle_readout=muscle_readout,
        biophysical_model=biophysical_model,
    )

    worm_config = WormConfig()
    worm_config.food_position = np.array([args.food_x, args.food_y, args.food_z])

    print(f"\nRunning closed-loop simulation for {args.n_steps} steps...")
    results = run_closed_loop(
        wrapper,
        n_steps=args.n_steps,
        render=args.render,
        config=worm_config,
        verbose=args.verbose,
        save_video_path=args.save_video,
    )

    # Save outputs if requested
    if args.save_output:
        out_dir = Path(args.save_output)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "trajectory.npy", results["trajectory"])
        np.save(out_dir / "muscles.npy", results["muscles"])
        np.save(out_dir / "food_concentration.npy", results["food_concentration"])
        print(f"\nSaved outputs to {out_dir}/")

    traj = results["trajectory"]
    total_dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    net_disp = np.linalg.norm(traj[-1] - traj[0])
    print("\n=== Results ===")
    print(f"Total path length: {total_dist:.3f}")
    print(f"Net displacement: {net_disp:.3f}")
    print(f"Straightness: {net_disp / total_dist if total_dist > 0 else 0:.3f}")
    print(f"Final food concentration: {results['food_concentration'][-1]:.3f}")


if __name__ == "__main__":
    main()
