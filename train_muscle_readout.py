#!/usr/bin/env python3
"""
Train muscle readout: voltages → 96 muscles.

Run with: source .venv/bin/activate  (then run this script)

Usage:
  # With synthetic data (no biophysical model yet):
  python train_muscle_readout.py --synthetic --output checkpoints/muscle_readout.npz

  # With real voltage data from your biophysical model:
  python train_muscle_readout.py --voltages voltages.npy --muscles muscles.npy --output checkpoints/muscle_readout.npz

The synthetic mode generates (voltages, muscles) from sine wave locomotion
so you can test the pipeline. Replace with real data once the biophysical
model is integrated.
"""

import argparse
from pathlib import Path

import numpy as np

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent))

N_NEURONS = 58

from biophysical_integration.muscle_readout import (
    train_muscle_readout,
    generate_training_data_with_sine_wave,
    MuscleReadoutConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic training data")
    parser.add_argument("--voltages", type=str, help="Path to voltages .npy (T, n_neurons)")
    parser.add_argument("--muscles", type=str, help="Path to muscles .npy (T, 96)")
    parser.add_argument("--output", type=str, default="biophysical_integration/checkpoints/muscle_readout.npz")
    parser.add_argument("--readout", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs for MLP")
    parser.add_argument("--n-steps", type=int, default=2000, help="Steps for synthetic data")
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic training data (sine wave + placeholder voltages)...")
        voltages, muscles = generate_training_data_with_sine_wave(
            n_steps=args.n_steps,
            n_neurons=N_NEURONS,
        )
        print(f"  Voltages: {voltages.shape}, Muscles: {muscles.shape}")
    elif args.voltages and args.muscles:
        voltages = np.load(args.voltages)
        muscles = np.load(args.muscles)
        assert voltages.shape[0] == muscles.shape[0], "Length mismatch"
        print(f"Loaded: voltages {voltages.shape}, muscles {muscles.shape}")
    else:
        parser.error("Use --synthetic or provide --voltages and --muscles")

    config = MuscleReadoutConfig(
        readout_type=args.readout,
        epochs=args.epochs,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {args.readout} muscle readout...")
    readout = train_muscle_readout(
        voltages, muscles,
        config=config,
        save_path=str(out_path),
    )

    # Quick validation
    pred = readout.predict(voltages[:100])
    mse = np.mean((pred - muscles[:100]) ** 2)
    print(f"\nValidation MSE (first 100 steps): {mse:.6f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
