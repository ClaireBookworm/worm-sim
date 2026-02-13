#!/usr/bin/env python3
"""
Collect (voltages, muscles) training data for the muscle readout.

Runs closed-loop: Jaxley biophysical RNN + physics sim with sine-wave reference
muscles. Records voltages from named neurons (Ca dataset) and muscles at each step.

Usage:
  python -m biophysical_integration.collect_muscle_training_data \
    --checkpoint models/.../input_trainable_voltage_biophys \
    --net-cache models/.../network_synapse_location_always_soma.pkl \
    --output-dir biophysical_integration/data \
    --n-steps 2000

Then train:
  python -m biophysical_integration.muscle_readout train \
    --voltages biophysical_integration/data/voltages.npy \
    --muscles biophysical_integration/data/muscles.npy \
    --save biophysical_integration/checkpoints/muscle_readout.npz
"""

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

N_MUSCLES = 96
N_MOTOR_NEURONS = 27

from biophysical_integration.jaxley_loader import JaxleyBiophysicalRNN
from biophysical_integration.muscle_readout import LinearMuscleReadout
from biophysical_integration.sensory_mapping import HeuristicSensoryMapping
from worm_sim import WormSimulation, WormConfig, generate_sine_wave_muscles


def collect_training_data(
    checkpoint_dir: str,
    net_cache: str,
    output_dir: str,
    n_steps: int = 2000,
    checkpoint_epoch: int = None,
    ca_cell_names_path: str = None,
    motor_only: bool = False,
    frequency: float = 1.5,
    amplitude: float = 0.8,
    render: bool = False,
    readout_path: str = None,
) -> tuple:
    """
    Run closed-loop with Jaxley + muscles, record (voltages, muscles).

    Muscles: sine wave (default) or from trained readout if readout_path given.
    When using readout_path, pass --motor-only to match readout's expected input.

    Returns:
        (voltages, muscles): (T, n_neurons), (T, 96)
    """
    # Load Jaxley (obs_idx from Ca_traces_cell_name.txt; motor_only => ~28 motor neurons)
    biophysical_model = JaxleyBiophysicalRNN(
        checkpoint_dir=checkpoint_dir,
        net_cache=net_cache,
        checkpoint_epoch=checkpoint_epoch,
        ca_cell_names_path=ca_cell_names_path,
        motor_only=motor_only,
    )
    sensory_mapping = HeuristicSensoryMapping()

    sim = WormSimulation(WormConfig())
    biophysical_model.reset()
    state = sim.reset()

    readout = None
    if readout_path and Path(readout_path).exists():
        if not motor_only:
            raise ValueError("--readout requires --motor-only (readout expects motor neuron voltages)")
        readout = LinearMuscleReadout(N_MOTOR_NEURONS, N_MUSCLES)
        readout.load(readout_path)
        print(f"Using readout from {readout_path} for muscles (closed-loop)")
    elif readout_path:
        raise FileNotFoundError(f"Readout path not found: {readout_path}")

    voltages_list = []
    muscles_list = []

    for step in range(n_steps):
        # Sensory drive from current physics state
        S = sensory_mapping(state)

        # Run Jaxley for steps_per_physics (6) neural steps
        voltages = biophysical_model.step(S)

        # Muscles: from readout (closed-loop) or sine wave (reference)
        if readout is not None:
            muscles = readout.predict(voltages[np.newaxis, :])[0]
            muscles = amplitude * muscles  # scale [0,1] to [0, amplitude]
        else:
            t = state.time
            muscles = generate_sine_wave_muscles(t, frequency=frequency, amplitude=amplitude)

        # Step physics with reference muscles
        state = sim.step(muscles)

        # Record: voltages (n_named,), muscles (96,)
        voltages_list.append(voltages)
        muscles_list.append(muscles)

        if render and step % 50 == 0:
            sim.render("human")

        if step % 200 == 0:
            print(f"  Step {step}/{n_steps}: pos={state.head_position[:2].round(3)}, "
                  f"n_voltages={len(voltages)}, n_muscles={len(muscles)}")

    sim.close()

    voltages_arr = np.array(voltages_list, dtype=np.float32)
    muscles_arr = np.array(muscles_list, dtype=np.float32)

    # Normalize muscles to [0, 1] for readout training (sine gives [0, amplitude])
    muscles_arr = np.clip(muscles_arr / amplitude, 0.0, 1.0).astype(np.float32)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "voltages.npy", voltages_arr)
    np.save(out / "muscles.npy", muscles_arr)

    print(f"\nSaved to {out}:")
    print(f"  voltages.npy: {voltages_arr.shape}")
    print(f"  muscles.npy: {muscles_arr.shape}")

    return voltages_arr, muscles_arr


def main():
    parser = argparse.ArgumentParser(description="Collect muscle readout training data")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--net-cache", required=True, help="Network pickle path")
    parser.add_argument("--output-dir", default="biophysical_integration/data",
                        help="Output directory for voltages.npy, muscles.npy")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument("--ca-cell-names", type=str, default=None,
                        help="Path to Ca_traces_cell_name.txt (default: BAAIWorm/.../Ca_traces_cell_name.txt)")
    parser.add_argument("--motor-only", action="store_true",
                        help="Use only motor neurons (~28) for muscle readout")
    parser.add_argument("--frequency", type=float, default=1.5)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--readout", type=str, default=None,
                        help="Path to trained muscle_readout.npz; use for closed-loop data (requires --motor-only)")
    args = parser.parse_args()

    collect_training_data(
        checkpoint_dir=args.checkpoint,
        net_cache=args.net_cache,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        checkpoint_epoch=args.checkpoint_epoch,
        ca_cell_names_path=args.ca_cell_names,
        motor_only=args.motor_only,
        frequency=args.frequency,
        amplitude=args.amplitude,
        render=args.render,
        readout_path=args.readout,
    )


if __name__ == "__main__":
    main()
