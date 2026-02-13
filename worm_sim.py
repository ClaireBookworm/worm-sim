"""
C. elegans MuJoCo Simulation

Full sensory-motor loop interface for neural model integration.
Based on BAAI WORM paper architecture.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import mujoco


@dataclass
class SensoryState:
    """Complete sensory information available to the neural model."""

    # Proprioception: joint angles (23 joints x 2 DOF = 46 values)
    joint_angles_dv: np.ndarray  # Dorsoventral angles, shape (23,)
    joint_angles_lr: np.ndarray  # Left-right angles, shape (23,)

    # Proprioception: joint velocities (subset for efficiency)
    joint_velocities_dv: np.ndarray  # shape (23,)
    joint_velocities_lr: np.ndarray  # shape (23,)

    # Body pose
    head_position: np.ndarray  # shape (3,) - xyz
    head_orientation: np.ndarray  # shape (4,) - quaternion
    tail_position: np.ndarray  # shape (3,)

    # Segment positions (all 24 segments)
    segment_positions: np.ndarray  # shape (24, 3)

    # Body curvature (derived from joint angles)
    curvature: np.ndarray  # shape (23,)

    # Environment sensing
    food_concentration: float  # Concentration at head
    food_gradient: np.ndarray  # Gradient vector at head, shape (3,)

    # Time
    time: float

    def to_flat_array(self) -> np.ndarray:
        """Flatten all sensory data into a single array for neural input."""
        return np.concatenate([
            self.joint_angles_dv,
            self.joint_angles_lr,
            self.joint_velocities_dv,
            self.joint_velocities_lr,
            self.head_position,
            self.head_orientation,
            self.tail_position,
            self.curvature,
            [self.food_concentration],
            self.food_gradient,
            [self.time]
        ])

    @property
    def flat_size(self) -> int:
        """Size of flattened sensory array."""
        return (23 + 23 + 23 + 23 +  # joint angles and velocities
                3 + 4 + 3 +  # head pos, quat, tail pos
                23 +  # curvature
                1 + 3 +  # food concentration and gradient
                1)  # time


@dataclass
class WormConfig:
    """Configuration for the worm simulation."""

    # Model parameters
    model_path: str = None  # Will default to worm_model.xml

    # Simulation parameters
    timestep: float = 0.001  # 1ms timestep
    frame_skip: int = 10  # Steps per control step (10ms control period)

    # Environment parameters
    food_position: np.ndarray = field(default_factory=lambda: np.array([2.0, 0.0, 0.0]))
    food_decay_rate: float = 0.5  # Concentration decay with distance

    # Fluid dynamics (simplified) - disabled by default, causes instability
    drag_coefficient: float = 0.01
    enable_fluid_drag: bool = False

    # Muscle parameters
    muscle_gain: float = 1.0
    muscle_damping: float = 0.1


class WormSimulation:
    """
    MuJoCo simulation of C. elegans with full sensory-motor interface.

    Provides:
    - 96 muscle actuators (4 strings x 24 segments)
    - Proprioceptive feedback (joint angles, velocities)
    - Environmental sensing (food concentration gradient)
    - Body pose tracking
    """

    N_SEGMENTS = 24
    N_JOINTS = 23  # One less than segments
    N_MUSCLES = 96  # 4 strings x 24 muscles
    N_MUSCLE_STRINGS = 4  # DR, DL, VR, VL

    # Muscle string indices
    MUSCLE_DR = 0  # Dorsal Right
    MUSCLE_DL = 1  # Dorsal Left
    MUSCLE_VR = 2  # Ventral Right
    MUSCLE_VL = 3  # Ventral Left

    def __init__(self, config: Optional[WormConfig] = None):
        self.config = config or WormConfig()

        # Load model
        if self.config.model_path is None:
            self.config.model_path = str(Path(__file__).parent / "worm_model.xml")

        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)

        # Set timestep
        self.model.opt.timestep = self.config.timestep

        # Cache body and site IDs for fast lookup
        self._cache_ids()

        # Renderer (lazy initialization)
        self._renderer = None
        self._viewer = None

        # Reset to initial state
        self.reset()

    def _cache_ids(self):
        """Cache MuJoCo IDs for bodies, sites, joints, actuators."""
        # Segment body IDs
        self.segment_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"seg_{i}")
            for i in range(self.N_SEGMENTS)
        ]

        # Joint IDs (dv and lr for each)
        self.joint_ids_dv = []
        self.joint_ids_lr = []
        for i in range(1, self.N_SEGMENTS):
            self.joint_ids_dv.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}_dv")
            )
            self.joint_ids_lr.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}_lr")
            )

        # Site IDs
        self.head_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head")
        self.tail_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tail")

        # Actuator IDs (96 total: 24 each for DR, DL, VR, VL)
        self.actuator_ids = {
            'DR': [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"DR_{i}")
                   for i in range(24)],
            'DL': [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"DL_{i}")
                   for i in range(24)],
            'VR': [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"VR_{i}")
                   for i in range(24)],
            'VL': [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"VL_{i}")
                   for i in range(24)],
        }

    def reset(self, seed: Optional[int] = None) -> SensoryState:
        """Reset simulation to initial state."""
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Small random perturbation to break symmetry
        self.data.qpos[0:3] = [0, 0, 0]  # Slight elevation
        self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)

        return self.get_sensory_state()

    def step(self, muscle_activations: np.ndarray) -> SensoryState:
        """
        Step the simulation with given muscle activations.

        Args:
            muscle_activations: Array of shape (96,) with values in [-1, 1]
                               Order: DR[0-23], DL[0-23], VR[0-23], VL[0-23]

        Returns:
            SensoryState with updated sensory information
        """
        if muscle_activations.shape != (self.N_MUSCLES,):
            raise ValueError(f"Expected muscle_activations shape (96,), got {muscle_activations.shape}")

        # Clip activations to valid range
        muscle_activations = np.clip(muscle_activations, -1, 1)

        # Apply muscle gain
        muscle_activations = muscle_activations * self.config.muscle_gain

        # Set control signals
        self.data.ctrl[:] = muscle_activations

        # Apply fluid drag if enabled
        if self.config.enable_fluid_drag:
            self._apply_fluid_drag()

        # Step simulation
        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        return self.get_sensory_state()

    def _apply_fluid_drag(self):
        """Apply simplified fluid drag forces to simulate swimming."""
        drag = self.config.drag_coefficient

        # Apply drag to each segment based on velocity
        for i, body_id in enumerate(self.segment_body_ids):
            # Get body velocity
            vel = self.data.cvel[body_id, 3:6]  # Linear velocity

            # Drag force proportional to velocity squared, opposite direction
            drag_force = -drag * vel * np.abs(vel)

            # Apply as external force
            self.data.xfrc_applied[body_id, :3] = drag_force

    def get_sensory_state(self) -> SensoryState:
        """Get complete sensory state for neural model input."""

        # Joint angles
        joint_angles_dv = np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                                     for jid in self.joint_ids_dv])
        joint_angles_lr = np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                                     for jid in self.joint_ids_lr])

        # Joint velocities
        joint_velocities_dv = np.array([self.data.qvel[self.model.jnt_dofadr[jid]]
                                         for jid in self.joint_ids_dv])
        joint_velocities_lr = np.array([self.data.qvel[self.model.jnt_dofadr[jid]]
                                         for jid in self.joint_ids_lr])

        # Head and tail positions
        head_pos = self.data.site_xpos[self.head_site_id].copy()
        tail_pos = self.data.site_xpos[self.tail_site_id].copy()

        # Head orientation (from body quaternion)
        head_body_id = self.segment_body_ids[0]
        head_quat = self.data.xquat[head_body_id].copy()

        # Segment positions
        segment_positions = np.array([self.data.xpos[bid].copy()
                                       for bid in self.segment_body_ids])

        # Curvature (second derivative of position, approximated by joint angles)
        curvature = joint_angles_dv  # Simplified: use DV angles as curvature proxy

        # Food sensing
        food_conc, food_grad = self._compute_food_signal(head_pos)

        return SensoryState(
            joint_angles_dv=joint_angles_dv,
            joint_angles_lr=joint_angles_lr,
            joint_velocities_dv=joint_velocities_dv,
            joint_velocities_lr=joint_velocities_lr,
            head_position=head_pos,
            head_orientation=head_quat,
            tail_position=tail_pos,
            segment_positions=segment_positions,
            curvature=curvature,
            food_concentration=food_conc,
            food_gradient=food_grad,
            time=self.data.time
        )

    def _compute_food_signal(self, head_pos: np.ndarray) -> tuple:
        """Compute food concentration and gradient at head position."""
        food_pos = self.config.food_position
        decay = self.config.food_decay_rate

        # Distance to food
        diff = food_pos - head_pos
        dist = np.linalg.norm(diff)

        # Exponential decay of concentration
        concentration = np.exp(-decay * dist)

        # Gradient points toward food, magnitude decreases with distance
        if dist > 1e-6:
            gradient = (diff / dist) * decay * concentration
        else:
            gradient = np.zeros(3)

        return concentration, gradient

    def get_body_state(self) -> Dict[str, Any]:
        """Get detailed body state for analysis/visualization."""
        return {
            'segment_positions': np.array([self.data.xpos[bid].copy()
                                            for bid in self.segment_body_ids]),
            'segment_orientations': np.array([self.data.xquat[bid].copy()
                                               for bid in self.segment_body_ids]),
            'joint_angles': {
                'dv': np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                                for jid in self.joint_ids_dv]),
                'lr': np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                                for jid in self.joint_ids_lr])
            },
            'center_of_mass': self._compute_com(),
            'time': self.data.time
        }

    def _compute_com(self) -> np.ndarray:
        """Compute center of mass of the worm."""
        total_mass = 0
        com = np.zeros(3)
        for body_id in self.segment_body_ids:
            mass = self.model.body_mass[body_id]
            pos = self.data.xpos[body_id]
            com += mass * pos
            total_mass += mass
        return com / total_mass if total_mass > 0 else com

    def set_food_position(self, position: np.ndarray):
        """Update food/attractor position."""
        self.config.food_position = np.array(position)

        # Also update the visual if it exists in the model
        food_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "food")
        if food_body_id >= 0:
            # Note: Can't directly move bodies, would need to modify mocap or recreate
            pass

    def render(self, mode: str = 'human', camera: str = None) -> Optional[np.ndarray]:
        """
        Render the simulation.

        Args:
            mode: 'human' for interactive viewer, 'rgb_array' for image
            camera: Camera name or None for default

        Returns:
            RGB array if mode='rgb_array', None otherwise
        """
        if mode == 'human':
            if self._viewer is None:
                try:
                    import mujoco.viewer as mj_viewer
                    self._viewer = mj_viewer.launch_passive(self.model, self.data)
                except (ImportError, AttributeError):
                    try:
                        self._viewer = mujoco.launch_passive(self.model, self.data)
                    except AttributeError:
                        print("Warning: MuJoCo viewer not available. Try: pip install mujoco[viewer]")
                        print("Falling back to rgb_array mode for headless rendering.")
                        return self.render('rgb_array')
                except RuntimeError as e:
                    if "mjpython" in str(e).lower():
                        print("Warning: On macOS, run with `mjpython` for live viewer: mjpython run_biophysical_sim.py ... --render")
                        print("Continuing without viewer.")
                        self._viewer = False  # Sentinel: don't retry
                    else:
                        raise
            if self._viewer is not None and self._viewer is not False:
                self._viewer.sync()
            return None

        elif mode == 'rgb_array':
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        """Clean up resources."""
        if self._viewer is not None and self._viewer is not False:
            self._viewer.close()
        self._viewer = None
        if self._renderer is not None:
            self._renderer = None

    # Convenience methods for muscle organization
    @staticmethod
    def organize_muscles(
        dr: np.ndarray,
        dl: np.ndarray,
        vr: np.ndarray,
        vl: np.ndarray
    ) -> np.ndarray:
        """
        Combine muscle activations from 4 strings into flat array.

        Args:
            dr, dl, vr, vl: Each shape (24,)

        Returns:
            Combined array shape (96,)
        """
        return np.concatenate([dr, dl, vr, vl])

    @staticmethod
    def split_muscles(activations: np.ndarray) -> tuple:
        """
        Split flat muscle array into 4 strings.

        Args:
            activations: Shape (96,)

        Returns:
            Tuple of (dr, dl, vr, vl), each shape (24,)
        """
        return (
            activations[0:24],   # DR
            activations[24:48],  # DL
            activations[48:72],  # VR
            activations[72:96],  # VL
        )


def generate_sine_wave_muscles(
    t: float,
    frequency: float = 1.0,
    wavelength: float = 1.0,
    amplitude: float = 0.5,
    n_segments: int = 24
) -> np.ndarray:
    """
    Generate sinusoidal muscle activation pattern for forward locomotion.

    This produces the traveling wave pattern seen in real C. elegans.

    Args:
        t: Current time
        frequency: Wave frequency (Hz)
        wavelength: Spatial wavelength (fraction of body length)
        amplitude: Activation amplitude [0, 1]
        n_segments: Number of segments

    Returns:
        Muscle activations shape (96,)
    """
    # Phase offset for each segment (wave travels from head to tail)
    phases = np.linspace(0, 2 * np.pi / wavelength, n_segments)

    # Temporal phase
    omega = 2 * np.pi * frequency

    # Dorsoventral activation (main locomotion)
    dv_activation = amplitude * np.sin(omega * t - phases)

    # DR and DL are in phase (both dorsal)
    dr = np.maximum(dv_activation, 0)  # Positive phase
    dl = np.maximum(dv_activation, 0)

    # VR and VL are anti-phase (ventral)
    vr = np.maximum(-dv_activation, 0)  # Negative phase
    vl = np.maximum(-dv_activation, 0)

    return WormSimulation.organize_muscles(dr, dl, vr, vl)


# Example usage and testing
if __name__ == "__main__":
    print("Initializing C. elegans simulation...")

    sim = WormSimulation()
    print(f"Model loaded: {sim.N_SEGMENTS} segments, {sim.N_MUSCLES} muscles")

    # Test with sine wave locomotion
    print("\nRunning test simulation with sine wave locomotion...")

    trajectory = []
    for step in range(500):
        t = step * sim.config.timestep * sim.config.frame_skip

        # Generate muscle commands
        muscles = generate_sine_wave_muscles(t, frequency=1.5, amplitude=0.8)

        # Step simulation
        state = sim.step(muscles)

        # Record trajectory
        if step % 50 == 0:
            print(f"  Step {step}: head at {state.head_position}, food_conc={state.food_concentration:.3f}")
            trajectory.append(state.head_position.copy())

    print(f"\nSimulation complete. Head moved from {trajectory[0]} to {trajectory[-1]}")
    print(f"Total distance: {np.linalg.norm(trajectory[-1] - trajectory[0]):.3f}")

    # Show sensory state structure
    final_state = sim.get_sensory_state()
    print(f"\nSensory state flat size: {final_state.flat_size}")
    print(f"Sensory array shape: {final_state.to_flat_array().shape}")

    sim.close()
    print("\nDone!")
