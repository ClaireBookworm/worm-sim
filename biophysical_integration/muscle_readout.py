"""
Muscle readout: Motor neuron voltages → 96 muscle activations.

Train a readout layer to map biophysical RNN voltages to muscle commands.
Supports:
- Linear (reservoir-style, like BAAIWorm)
- MLP (small feedforward network)

Training data: (voltages, muscles) pairs from running the simulation
with the biophysical model + a reference controller (e.g. sine wave).

Run directly (avoids package import warning):
  python biophysical_integration/muscle_readout.py --voltages ... --muscles ... --save ...
Or as module:
  python -m biophysical_integration.muscle_readout --voltages ... --muscles ... --save ...
"""

import sys
from pathlib import Path

# Allow running as script: python biophysical_integration/muscle_readout.py
if __name__ != "__main__":
    pass  # Normal import
elif str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

N_NEURONS = 58
N_MUSCLES = 96
N_MOTOR_NEURONS = 27
from pathlib import Path
from typing import Optional, List, Tuple

from biophysical_integration.recorded_neurons import load_recorded_neuron_indices
from biophysical_integration.neuron_muscle_wiring import get_wiring_mask_for_motor_neurons, load_neuron_muscle_mask
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MuscleReadoutConfig:
    """Configuration for muscle readout training."""
    # Which neurons to use (indices or names)
    # If your biophysical model has different neuron order, provide mapping
    motor_neuron_indices: Optional[List[int]] = None  # None = use all

    # Readout type
    readout_type: str = "linear"  # "linear" or "mlp"

    # MLP config (if readout_type == "mlp")
    mlp_hidden: int = 128
    mlp_layers: int = 2

    # Training
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 64
    val_split: float = 0.1

    # Voltage normalization (model outputs ~[-80, 20] mV)
    voltage_min: float = -80.0
    voltage_max: float = 20.0

    # Linear readout: ridge regularization (L2)
    ridge: float = 0.1

    # NaN handling for training
    nan_threshold: float = 0.5  # Filter rows with > this fraction NaN
    resting_potential: float = -65.0  # mV, for replacing remaining NaNs

    # Biological wiring: path to neuron_muscle.xlsx (sparse readout mask)
    wiring_mask_path: Optional[str] = None
    ca_cell_names_path: Optional[str] = None  # For neuron name order when loading mask


class LinearMuscleReadout:
    """
    Linear readout: muscles = W @ voltages + b, then clip to [0, 1].

    Same style as BAAIWorm's reservoir readout (n_neurons×96 matrix).
    """

    def __init__(self, n_neurons: int = None, n_muscles: int = None):
        self.n_neurons = n_neurons if n_neurons is not None else N_NEURONS
        self.n_muscles = n_muscles if n_muscles is not None else N_MUSCLES
        self.W = np.zeros((n_muscles, n_neurons), dtype=np.float32)
        self.b = np.zeros(n_muscles, dtype=np.float32)

    def fit(
        self,
        voltages: np.ndarray,  # (T, n_neurons)
        muscles: np.ndarray,   # (T, 96)
        ridge: float = 0.1,
        wiring_mask: Optional[np.ndarray] = None,  # (n_neurons, 96) bool, True = connected
        nan_threshold: float = 0.5,  # Filter rows with > this fraction NaN
        resting_potential: float = -65.0,  # mV, for replacing remaining NaNs
    ) -> dict:
        """
        Fit linear readout with ridge regression (L2-regularized least squares).

        Voltages are scaled to [0,1] for numerical stability (raw ~[-80,20] mV).

        Handles NaN voltages by:
        1. Filtering out timesteps where > nan_threshold of neurons have NaN
        2. Replacing remaining NaNs with resting_potential (-65 mV)
        3. Clipping extreme values to [-100, 50] mV range
        """
        V_raw = voltages.astype(np.float64)
        M_raw = muscles.astype(np.float64)

        # Count NaN per row
        nan_frac = np.isnan(V_raw).mean(axis=1)
        valid_rows = nan_frac <= nan_threshold
        n_valid = valid_rows.sum()
        n_total = len(valid_rows)

        if n_valid < 10:
            raise ValueError(
                f"Only {n_valid}/{n_total} rows have <= {nan_threshold*100:.0f}% NaN. "
                "Check Jaxley model output or lower nan_threshold."
            )

        print(f"[LinearMuscleReadout] Using {n_valid}/{n_total} rows ({100*n_valid/n_total:.1f}%) with <= {nan_threshold*100:.0f}% NaN")

        V = V_raw[valid_rows]
        M = M_raw[valid_rows]

        # Replace remaining NaNs with resting potential, clip extreme values
        V = np.nan_to_num(V, nan=resting_potential, posinf=50.0, neginf=-100.0)
        V = np.clip(V, -100.0, 50.0)  # Clip unrealistic values
        M = np.nan_to_num(M, nan=0.5, posinf=1.0, neginf=0.0)

        # Scale voltages to [0,1] for numerical stability (raw ~[-80, 20] mV)
        V_scaled = np.clip((V + 80.0) / 100.0, 0.0, 1.0)

        ones = np.ones((V_scaled.shape[0], 1))
        X = np.hstack([V_scaled, ones])
        n_features = X.shape[1]
        XtX = X.T @ X
        XtM = X.T @ M
        reg = np.eye(n_features, dtype=np.float64) * ridge
        reg[-1, -1] = 1e-6
        try:
            theta = np.linalg.solve(XtX + reg, XtM)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(X, M, rcond=1e-10)[0]
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            theta = np.linalg.pinv(X) @ M  # Most robust fallback

        self.W = theta[:-1, :].T.astype(np.float32)
        self.b = theta[-1, :].astype(np.float32)

        # Apply biological wiring mask (zero non-connected weights)
        if wiring_mask is not None and wiring_mask.shape == (self.n_neurons, self.n_muscles):
            self.W = self.W * wiring_mask.T.astype(np.float32)

        # Compute MSE on filtered training data
        pred = np.clip(V_scaled @ self.W.T + self.b, 0.0, 1.0)
        mse = float(np.nanmean((pred.astype(np.float64) - M) ** 2))
        if np.isnan(mse):
            mse = 0.0  # Fallback if all-NaN
        if np.any(np.isnan(self.W)) or np.any(np.isnan(self.b)):
            raise RuntimeError("Fit produced NaN weights; try larger --ridge or check data")
        return {"mse": mse, "n_train_rows": n_valid}

    def predict(self, voltages: np.ndarray, resting_potential: float = -65.0) -> np.ndarray:
        """Predict muscle activations from voltages (raw mV, ~[-80, 20]).

        NaN/inf values are replaced with resting_potential before prediction.
        Extreme voltages are clipped to [-100, 50] mV.
        """
        V = voltages.astype(np.float64)
        V = np.nan_to_num(V, nan=resting_potential, posinf=50.0, neginf=-100.0)
        V = np.clip(V, -100.0, 50.0)
        V_scaled = np.clip((V + 80.0) / 100.0, 0.0, 1.0).astype(np.float32)
        out = V_scaled @ self.W.T + self.b
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b)

    def load(self, path: str):
        data = np.load(path)
        self.W = data["W"]
        self.b = data["b"]
        self.n_neurons = self.W.shape[1]
        self.n_muscles = self.W.shape[0]


class MLPMuscleReadout:
    """
    Small MLP readout: voltages → hidden → 96 muscles.

    Can capture nonlinearities if the linear readout is insufficient.
    """

    def __init__(
        self,
        n_neurons: int,
        n_muscles: int = 96,
        hidden: int = 128,
        n_layers: int = 2,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MLPMuscleReadout")
        self.n_neurons = n_neurons
        self.n_muscles = n_muscles
        layers = []
        in_dim = n_neurons
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_muscles))
        self.net = nn.Sequential(*layers)

    def fit(
        self,
        voltages: np.ndarray,
        muscles: np.ndarray,
        epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 64,
        val_split: float = 0.1,
    ) -> dict:
        """Train MLP with MSE loss."""
        n = len(voltages)
        idx = np.random.permutation(n)
        nval = int(n * val_split)
        val_idx, train_idx = idx[:nval], idx[nval:]

        V_train = torch.FloatTensor(voltages[train_idx])
        M_train = torch.FloatTensor(muscles[train_idx])
        V_val = torch.FloatTensor(voltages[val_idx])
        M_val = torch.FloatTensor(muscles[val_idx])

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        losses = []
        for ep in range(epochs):
            perm = np.random.permutation(len(train_idx))
            for i in range(0, len(perm), batch_size):
                batch = perm[i : i + batch_size]
                pred = self.net(V_train[batch])
                loss = ((pred - M_train[batch]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                vloss = ((self.net(V_val) - M_val) ** 2).mean().item()
            losses.append(vloss)
            if (ep + 1) % 100 == 0:
                print(f"  Epoch {ep+1}/{epochs} val MSE: {vloss:.6f}")

        return {"val_mse": losses[-1], "losses": losses}

    def predict(self, voltages: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.net(torch.FloatTensor(voltages))
        return np.clip(out.numpy(), 0.0, 1.0).astype(np.float32)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location="cpu"))


def train_muscle_readout(
    voltages: np.ndarray,   # (T, n_neurons)
    muscles: np.ndarray,    # (T, 96)
    config: Optional[MuscleReadoutConfig] = None,
    save_path: Optional[str] = None,
) -> object:
    """
    Train muscle readout from (voltages, muscles) data.

    Returns trained readout object (LinearMuscleReadout or MLPMuscleReadout).
    """
    config = config or MuscleReadoutConfig()
    n_neurons = voltages.shape[1]
    n_muscles = muscles.shape[1]

    if config.motor_neuron_indices is not None:
        voltages = voltages[:, config.motor_neuron_indices]
        n_neurons = voltages.shape[1]

    wiring_mask = None
    if config.wiring_mask_path and config.readout_type == "linear":
        _, neuron_names = load_recorded_neuron_indices(
            ca_cell_names_path=config.ca_cell_names_path,
            subset="motor" if n_neurons == N_MOTOR_NEURONS else "all",
        )
        if n_neurons == N_MOTOR_NEURONS:
            wiring_mask = get_wiring_mask_for_motor_neurons(neuron_names, path=config.wiring_mask_path)
        else:
            wiring_mask = load_neuron_muscle_mask(neuron_names, muscle_order="worm_sim", path=config.wiring_mask_path)
        if wiring_mask is not None:
            print(f"Applied wiring mask from {config.wiring_mask_path} ({wiring_mask.sum()} connections)")
        else:
            print(f"Wiring mask path given but file not found or unreadable: {config.wiring_mask_path}")

    if config.readout_type == "linear":
        readout = LinearMuscleReadout(n_neurons, n_muscles)
        metrics = readout.fit(
            voltages, muscles,
            ridge=config.ridge,
            wiring_mask=wiring_mask,
            nan_threshold=config.nan_threshold,
            resting_potential=config.resting_potential,
        )
        print(f"Linear readout train MSE: {metrics['mse']:.6f}")
    elif config.readout_type == "mlp":
        readout = MLPMuscleReadout(
            n_neurons, n_muscles,
            hidden=config.mlp_hidden,
            n_layers=config.mlp_layers,
        )
        metrics = readout.fit(
            voltages, muscles,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            val_split=config.val_split,
        )
    else:
        raise ValueError(f"Unknown readout_type: {config.readout_type}")

    if save_path:
        readout.save(save_path)
        print(f"Saved readout to {save_path}")

    return readout


def _sine_wave_muscles(t: float, frequency: float = 1.5, amplitude: float = 0.7, n_segments: int = 24) -> np.ndarray:
    """Inline sine wave muscle pattern (avoids importing worm_sim/mujoco)."""
    phases = np.linspace(0, 2 * np.pi, n_segments)
    omega = 2 * np.pi * frequency
    dv_activation = amplitude * np.sin(omega * t - phases)
    dr = np.maximum(dv_activation, 0)
    dl = np.maximum(dv_activation, 0)
    vr = np.maximum(-dv_activation, 0)
    vl = np.maximum(-dv_activation, 0)
    return np.concatenate([dr, dl, vr, vl])


def generate_training_data_with_sine_wave(
    n_steps: int = 2000,
    dt: float = 0.01,
    frequency: float = 1.5,
    amplitude: float = 0.7,
    n_neurons: int = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate placeholder training data: sine wave muscles + synthetic voltages.

    Use this when you don't yet have the biophysical model running.
    The "voltages" are synthetic (sine-based) so the readout learns a simple
    mapping. Replace with real voltages from your biophysical RNN once integrated.

    Returns:
        voltages: (T, n_neurons) - synthetic for now
        muscles: (T, 96) - from sine wave pattern
    """
    if n_neurons is None:
        n_neurons = N_NEURONS
    np.random.seed(seed)
    muscles_list = []
    for step in range(n_steps):
        t = step * dt
        m = _sine_wave_muscles(t, frequency=frequency, amplitude=amplitude)
        muscles_list.append(m)
    muscles = np.array(muscles_list, dtype=np.float32)

    # Synthetic voltages: correlate with muscles via random projection
    # In reality, these come from the biophysical model
    W_synth = np.random.randn(n_neurons, 96) * 0.1
    voltages = muscles @ W_synth.T
    voltages = np.clip(voltages * 50 - 30, -80, 20)  # Roughly [-80, 20] mV range
    voltages = voltages.astype(np.float32)

    return voltages, muscles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train muscle readout from collected data")
    parser.add_argument("--voltages", required=True, help="voltages.npy from collect_muscle_training_data")
    parser.add_argument("--muscles", required=True, help="muscles.npy from collect_muscle_training_data")
    parser.add_argument("--save", default="biophysical_integration/checkpoints/muscle_readout.npz")
    parser.add_argument("--readout-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--ridge", type=float, default=0.1, help="Ridge regularization for linear readout")
    parser.add_argument("--epochs", type=int, default=500, help="For MLP only")
    parser.add_argument("--wiring-mask", type=str, default=None,
                        help="Path to neuron_muscle.xlsx for sparse readout (linear only)")
    parser.add_argument("--ca-cell-names", type=str, default=None,
                        help="Path to Ca_traces_cell_name.txt (for neuron order when using --wiring-mask)")
    parser.add_argument("--nan-threshold", type=float, default=0.5,
                        help="Filter rows with > this fraction of NaN voltages (default 0.5)")
    parser.add_argument("--resting-potential", type=float, default=-65.0,
                        help="Replace remaining NaN voltages with this value in mV (default -65.0)")
    parser.add_argument("--early-steps", type=int, default=None,
                        help="Use only first N steps for training (model saturates after ~100 steps; use 100-150)")
    args = parser.parse_args()

    voltages = np.load(args.voltages)
    muscles = np.load(args.muscles)
    if args.early_steps is not None:
        n_use = min(args.early_steps, len(voltages))
        voltages = voltages[:n_use]
        muscles = muscles[:n_use]
        print(f"Using first {n_use} steps only (--early-steps={args.early_steps})")
    print(f"Loaded voltages {voltages.shape}, muscles {muscles.shape}")

    config = MuscleReadoutConfig(
        readout_type=args.readout_type,
        ridge=args.ridge,
        epochs=args.epochs,
        wiring_mask_path=args.wiring_mask,
        ca_cell_names_path=args.ca_cell_names,
        nan_threshold=args.nan_threshold,
        resting_potential=args.resting_potential,
    )
    train_muscle_readout(voltages, muscles, config=config, save_path=args.save)
