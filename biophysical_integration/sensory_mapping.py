"""
Sensory mapping: SensoryState → 15 sensory neuron drives for jaxley_worm.

The biophysical RNN expects input S of shape (15,) for sensory neurons:
  AWAL, AWAR, AWCL, AWCR, ASKL, ASKR, ALNL, ALNR, PLML, PHAL, PHAR, URYDL, URYDR, URYVL, URYVR

S is bounded by SigmoidTransform(-0.005, 0.005) — i.e. current injection in nA range.

Key insight from BAAIWorm paper: They use the DERIVATIVE of food concentration
(change over time) as the sensory signal, not just the gradient magnitude.
This captures whether the worm is moving toward or away from food.

Maps simulation SensoryState to these 15 drives. Can use heuristics or a trained model.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from worm_sim import SensoryState


# Sensory neuron order for jaxley_worm (15 neurons)
SENSORY_NEURON_ORDER = [
    "AWAL", "AWAR", "AWCL", "AWCR", "ASKL", "ASKR",
    "ALNL", "ALNR", "PLML", "PHAL", "PHAR",
    "URYDL", "URYDR", "URYVL", "URYVR",
]

# Input bounds - MUST match training range from run_jaxley_worm.py:
#   S0 = 0.002 * (2 * jax.random.uniform(...) - 1)  => S in [-0.002, 0.002]
# Using larger values (e.g. ±0.005) causes model instability!
S_INPUT_LOW = -0.002
S_INPUT_HIGH = 0.002


@dataclass
class SensoryMappingConfig:
    """Configuration for sensory mapping."""
    # Chemosensory scaling for concentration derivative (dC/dt)
    # Positive dC/dt = moving toward food = positive drive
    concentration_derivative_scale: float = 0.01

    # History length for computing derivative (in steps)
    history_len: int = 11

    # Normalization range for derivative - scaled down to match ±0.002 training range
    # Original was [-0.08, 0.02] but that's way outside ±0.002
    derivative_min: float = -0.002  # nA (matches training range)
    derivative_max: float = 0.002   # nA (matches training range)

    # URY neurons: mechanosensory / proprioceptive
    # Dorsal URY: driven by dorsal curvature, Ventral URY: driven by ventral curvature
    # Scaled down to stay within ±0.002 range
    curvature_scale: float = 0.001
    velocity_scale: float = 0.0005

    # Baseline drive (when no stimulus)
    baseline: float = 0.0

    # Smoothing: exponential moving average factor (0 = no smoothing, 0.9 = heavy smoothing)
    # Prevents sharp jumps that can destabilize the model
    smoothing_alpha: float = 0.3

    # Sensory noise: Jaxley was trained with white noise S ~ uniform[-0.002, 0.002].
    # Without noise, constant S causes model to saturate to fixed point after ~100 steps.
    # Add small noise to keep model in dynamic regime (0 = off).
    # TODO: maybe delete if this isn't accurate 
    sensory_noise_scale: float = 0.0015


class HeuristicSensoryMapping:
    """
    Heuristic mapping from SensoryState to 15 sensory drives (jaxley_worm).

    Based on BAAIWorm approach:
    - Chemosensory neurons: driven by the DERIVATIVE of food concentration (dC/dt)
      This captures whether the worm is moving toward (positive) or away (negative) from food.
    - URY neurons: mechanosensory/proprioceptive — body curvature and angular velocity
      URYDL/URYDR: dorsal curvature, URYVL/URYVR: ventral curvature
    """

    N_SENSORY = 15

    def __init__(self, config: Optional[SensoryMappingConfig] = None):
        self.config = config or SensoryMappingConfig()
        # History of concentration values for computing derivative
        self._concentration_history = np.zeros(self.config.history_len, dtype=np.float32)
        self._initialized = False
        # Smoothed output for temporal filtering
        self._S_smoothed = np.zeros(self.N_SENSORY, dtype=np.float32)

    def reset(self):
        """Reset concentration history and smoothing state (call when resetting simulation)."""
        self._concentration_history[:] = 0.0
        self._initialized = False
        self._S_smoothed[:] = 0.0

    def _update_concentration_history(self, concentration: float) -> float:
        """
        Update history and compute normalized concentration derivative.

        Returns derivative scaled to [derivative_min, derivative_max] range.
        """
        # Shift history and add new value
        self._concentration_history[:-1] = self._concentration_history[1:]
        self._concentration_history[-1] = concentration

        if not self._initialized:
            # Fill history with initial value on first call
            self._concentration_history[:] = concentration
            self._initialized = True
            return 0.0

        # Compute derivative: current - previous
        delta_concentration = concentration - self._concentration_history[-2]

        # Normalize based on history range (like BAAIWorm)
        hist_min = np.min(self._concentration_history)
        hist_max = np.max(self._concentration_history)
        hist_range = hist_max - hist_min

        if hist_range < 1e-8:
            # No change in concentration over history window
            normalized = 0.0
        else:
            # Normalize to [-1, 1] range based on history, then scale
            normalized = delta_concentration / (hist_range + 1e-8)
            normalized = np.clip(normalized, -1.0, 1.0)

        # Scale to the derivative range (like BAAIWorm's [-80, 20] mV scaled to nA)
        cfg = self.config
        derivative_scaled = normalized * (cfg.derivative_max - cfg.derivative_min) / 2

        return float(derivative_scaled)

    def __call__(self, state: "SensoryState") -> np.ndarray:
        """
        Map SensoryState to 15 sensory drives, shape (15,).

        Chemosensory neurons (indices 0-10): driven by concentration derivative
        URY neurons (indices 11-14): driven by body curvature/velocity
        """
        S = np.zeros(self.N_SENSORY, dtype=np.float32)
        cfg = self.config

        # === Chemosensory: concentration derivative (dC/dt) ===
        # Positive derivative = moving toward food = excitatory input
        chem_drive = self._update_concentration_history(state.food_concentration)

        # All chemosensory neurons get the same drive (like BAAIWorm)
        # AWAL, AWAR, AWCL, AWCR, ASKL, ASKR, ALNL, ALNR, PLML, PHAL, PHAR
        S[:11] = chem_drive

        # === Mechanosensory / Proprioceptive: URY neurons ===
        # These sense head position and body bending
        # URYDL, URYDR: dorsal (respond to dorsal bending)
        # URYVL, URYVR: ventral (respond to ventral bending)

        curvature = state.curvature  # (23,) body curvature along segments
        velocities_dv = state.joint_velocities_dv  # (23,) dorsoventral joint velocities

        # Head curvature (first few segments)
        head_curv = np.mean(curvature[:5]) if len(curvature) >= 5 else np.mean(curvature)
        head_vel = np.mean(velocities_dv[:5]) if len(velocities_dv) >= 5 else np.mean(velocities_dv)

        # Dorsal URY neurons: activated by positive (dorsal) curvature
        dorsal_drive = cfg.curvature_scale * max(head_curv, 0) + cfg.velocity_scale * max(head_vel, 0)

        # Ventral URY neurons: activated by negative (ventral) curvature
        ventral_drive = cfg.curvature_scale * max(-head_curv, 0) + cfg.velocity_scale * max(-head_vel, 0)

        S[11] = dorsal_drive   # URYDL
        S[12] = dorsal_drive   # URYDR
        S[13] = ventral_drive  # URYVL
        S[14] = ventral_drive  # URYVR

        # Clip to valid input range (MUST match training: ±0.002)
        S = np.clip(S, S_INPUT_LOW, S_INPUT_HIGH).astype(np.float32)

        # Apply temporal smoothing to prevent sharp jumps
        alpha = cfg.smoothing_alpha
        if alpha > 0:
            self._S_smoothed = alpha * self._S_smoothed + (1 - alpha) * S
            S = self._S_smoothed.copy()

        # Add sensory noise to match training regime (run_jaxley_worm uses S ~ uniform[-0.002, 0.002])
        # Prevents model from saturating to fixed point when S becomes constant
        if cfg.sensory_noise_scale > 0:
            noise = cfg.sensory_noise_scale * (2 * np.random.rand(self.N_SENSORY).astype(np.float32) - 1)
            S = S + noise

        # Clip to valid input range (MUST match training: ±0.002)
        S = np.clip(S, S_INPUT_LOW, S_INPUT_HIGH).astype(np.float32)

        return S


class MLPSensoryMapping:
    """
    Learned MLP mapping: SensoryState → 15 sensory drives.

    Train this on (SensoryState, target_S) pairs. Target S can come from:
    - The trained checkpoint's opt_params["inputs"] (if you have matching states)
    - Or hand-designed targets for desired behaviors
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.W = None
        self.b = None
        if weights_path:
            self.load(weights_path)

    def load(self, path: str):
        """Load weights from file (e.g. .npy or .pkl)."""
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        if isinstance(data, dict):
            self.W = data["W"]
            self.b = data["b"]
        else:
            self.W = data
            self.b = np.zeros(15)

    def __call__(self, state: "SensoryState") -> np.ndarray:
        """Map SensoryState to 15 drives."""
        if self.W is None:
            raise RuntimeError("MLPSensoryMapping not loaded or trained")
        x = state.to_flat_array()
        S = np.tanh(self.W @ x + self.b) * (S_INPUT_HIGH - S_INPUT_LOW) / 2
        S = np.clip(S, S_INPUT_LOW, S_INPUT_HIGH).astype(np.float32)
        return S


def get_sensory_mapping(mode: str = "heuristic", **kwargs) -> callable:
    """
    Factory for sensory mapping.

    Args:
        mode: "heuristic" or "mlp"
        **kwargs: Passed to HeuristicSensoryMapping or MLPSensoryMapping
    """
    if mode == "heuristic":
        return HeuristicSensoryMapping(**kwargs)
    elif mode == "mlp":
        return MLPSensoryMapping(**kwargs)
    else:
        raise ValueError(f"Unknown sensory mapping mode: {mode}")
