"""
Biophysical RNN wrapper for closed-loop simulation.

Integrates the Jaxley/NEURON biophysical RNN checkpoint with:
- Sensory mapping (SensoryState → 15 sensory drives S)
- Muscle readout (voltages → 96 muscles)

The actual model loading requires the NEURON repo. This module provides:
1. BiophysicalRNNWrapper: Interface that you complete with your model loader
2. MockBiophysicalRNN: Placeholder for testing the pipeline without the real model
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

N_NEURONS = 58
N_SENSORY = 15
N_MUSCLES = 96

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from worm_sim import SensoryState, WormSimulation, generate_sine_wave_muscles


@dataclass
class BiophysicalConfig:
    """Configuration for biophysical RNN integration."""
    # Paths
    checkpoint_path: str = ""  # ckpt_epoch_5000.pkl
    neuron_repo_path: str = ""  # Path to NEURON/components for Jaxley imports
    muscle_readout_path: str = ""
    sensory_mapping: str = "heuristic"  # "heuristic" or "mlp"

    # Timing: physics runs at 10ms, neural at ~0.5ms
    physics_dt: float = 0.01   # 10ms control period
    neural_dt: float = 5/3000  # ~1.67ms (5/3 ms from BAAIWorm)
    steps_per_physics: int = 6  # Run this many neural steps per physics step

    # Neuron indices (if your model has different order than BAAIWorm)
    motor_neuron_indices: Optional[list] = None


class MockBiophysicalRNN:
    """
    Mock biophysical RNN for testing the pipeline without the real model.

    Uses sine-wave-like dynamics to produce voltage-like outputs.
    Replace with actual Jaxley model when integrating.
    """

    def __init__(self, n_neurons: int = None, dt: float = 5/3000):
        self.n_neurons = n_neurons if n_neurons is not None else N_NEURONS
        self.dt = dt
        self.t = 0.0
        self.state = np.zeros(n_neurons, dtype=np.float32)
        # Random coupling for "recurrent" dynamics
        np.random.seed(42)
        self.W_rec = np.random.randn(n_neurons, n_neurons) * 0.02

    def reset(self):
        self.t = 0.0
        self.state = np.zeros(self.n_neurons, dtype=np.float32)

    def step(self, S: np.ndarray) -> np.ndarray:
        """
        One step: S (n_sensory,) → voltages (n_neurons,).

        Mock: inject S into first n_sensory neurons, add recurrent dynamics.
        """
        n_sens = min(len(S), N_SENSORY)
        self.state[:n_sens] += S[:n_sens] * 10 * self.dt
        # Recurrent
        self.state += self.dt * (self.W_rec @ self.state)
        # Decay + bound
        self.state *= 0.99
        self.state = np.clip(self.state, -80, 20)
        self.t += self.dt
        return self.state.copy()


class BiophysicalRNNWrapper:
    """
    Full wrapper: SensoryState → muscles for closed-loop simulation.

    Components:
    1. sensory_mapping: state → S (n_sensory,)
    2. biophysical_model: S → voltages (run multiple steps per physics step)
    3. muscle_readout: voltages → muscles (96,)

    To use with the real checkpoint:
    - Set load_model_fn to a function that loads your Jaxley model from checkpoint
    - The function should return an object with: step(S) -> voltages, reset()
    """

    def __init__(
        self,
        config: Optional[BiophysicalConfig] = None,
        sensory_mapping: Optional[Callable[[SensoryState], np.ndarray]] = None,
        muscle_readout: Optional[object] = None,
        biophysical_model: Optional[object] = None,
    ):
        self.config = config or BiophysicalConfig()

        # Sensory mapping
        if sensory_mapping is not None:
            self.sensory_mapping = sensory_mapping
        else:
            from .sensory_mapping import HeuristicSensoryMapping
            self.sensory_mapping = HeuristicSensoryMapping()

        # Muscle readout
        if muscle_readout is not None:
            self.muscle_readout = muscle_readout
        elif self.config.muscle_readout_path and Path(self.config.muscle_readout_path).exists():
            from .muscle_readout import LinearMuscleReadout
            self.muscle_readout = LinearMuscleReadout(N_NEURONS, N_MUSCLES)
            self.muscle_readout.load(self.config.muscle_readout_path)
        else:
            # No readout: use sine wave as fallback (for testing)
            self.muscle_readout = None

        # Biophysical model
        if biophysical_model is not None:
            self.biophysical_model = biophysical_model
        else:
            # Default: mock for testing
            self.biophysical_model = MockBiophysicalRNN(
                n_neurons=N_NEURONS,
                dt=self.config.neural_dt,
            )

    def reset(self):
        """Reset internal state (e.g. RNN hidden state, carry, sensory history)."""
        if hasattr(self.biophysical_model, "reset"):
            self.biophysical_model.reset()
        if hasattr(self.sensory_mapping, "reset"):
            self.sensory_mapping.reset()

    def forward(self, state: SensoryState) -> np.ndarray:
        """
        Full forward pass: SensoryState → 96 muscle activations.

        1. state → S (7,)
        2. Run biophysical model for steps_per_physics steps with S
        3. voltages → muscles
        """
        S = self.sensory_mapping(state)

        # Run neural model for multiple steps (one physics step)
        n_steps = self.config.steps_per_physics
        voltages_list = []
        for _ in range(n_steps):
            v = self.biophysical_model.step(S)
            voltages_list.append(v)
        voltages = np.mean(voltages_list, axis=0)  # Average over sub-steps

        # Muscle readout
        if self.muscle_readout is not None:
            muscles = self.muscle_readout.predict(voltages[np.newaxis, :])[0]
            # Convert [0, 1] to [-1, 1] for sim if needed (worm_sim uses [-1, 1])
            muscles = 2 * muscles - 1
        else:
            # Fallback: sine wave
            muscles = generate_sine_wave_muscles(
                state.time, frequency=1.5, amplitude=0.7
            )

        return np.clip(muscles, -1, 1).astype(np.float32)


def run_closed_loop(
    wrapper: BiophysicalRNNWrapper,
    n_steps: int = 500,
    render: bool = False,
    config=None,
    verbose: bool = False,
    save_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run closed-loop simulation with biophysical wrapper.

    Returns dict with trajectory, muscles, food_concentration.
    """
    from worm_sim import WormConfig
    sim = WormSimulation(config or WormConfig())
    wrapper.reset()
    state = sim.reset()

    trajectory = []
    muscle_history = []
    food_history = []

    video_writer = None
    if save_video_path:
        try:
            import imageio
            video_writer = imageio.get_writer(save_video_path, fps=50)
        except ImportError:
            print("Warning: imageio not installed. Run: pip install imageio imageio-ffmpeg")
            save_video_path = None

    for step in range(n_steps):
        muscles = wrapper.forward(state)
        state = sim.step(muscles)

        trajectory.append(state.head_position.copy())
        muscle_history.append(muscles.copy())
        food_history.append(state.food_concentration)

        if render and step % 10 == 0:
            sim.render("human")

        if video_writer is not None:
            frame = sim.render("rgb_array")
            if frame is not None:
                video_writer.append_data(frame)

        if step % 100 == 0:
            msg = f"  Step {step}/{n_steps}: pos={state.head_position[:2].round(3)}, food={state.food_concentration:.3f}"
            if verbose and muscles.size > 0:
                msg += f" | muscles: mean={muscles.mean():.3f} std={muscles.std():.3f} min={muscles.min():.3f} max={muscles.max():.3f}"
            print(msg)

    if video_writer is not None:
        video_writer.close()
        print(f"Saved video to {save_video_path}")

    sim.close()

    return {
        "trajectory": np.array(trajectory),
        "muscles": np.array(muscle_history),
        "food_concentration": np.array(food_history),
        "final_state": state,
    }


# --- Integration instructions for real Jaxley model ---
"""
To integrate the real biophysical RNN checkpoint:

1. Add the NEURON repo to your path:
   sys.path.insert(0, "/orcd/data/edboyden/002/davy/NEURON/components")

2. Load checkpoint and build model (pseudocode from your setup):
   import pickle
   ckpt = pickle.load(open("ckpt_epoch_5000.pkl", "rb"))
   opt_params = ckpt["opt_params"]
   # Use tf.forward() to map opt_params to physical params
   # Build Jaxley model with those params

3. Create a step function that:
   - Takes S (7,) for current timestep
   - Runs model for one neural step (dt ~ 5/3 ms)
   - Returns voltages for all cells (or at least motor neurons)

4. Pass to BiophysicalRNNWrapper:
   wrapper = BiophysicalRNNWrapper(
       biophysical_model=YourJaxleyModel(ckpt, ...),
       muscle_readout=trained_readout,
       sensory_mapping=HeuristicSensoryMapping(),
   )
"""
