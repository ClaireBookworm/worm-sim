"""
Load and run the Jaxley biophysical RNN for closed-loop inference.

Uses neuron/jaxley_worm: 136 total neurons, ~58 named (from Ca_traces_cell_name.txt).
Requires:
  - checkpoint_dir: directory with ckpt_epoch_*.pkl
  - net_cache: path to pre-built network pickle (from a prior --build-net run)
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

N_NEURONS = 58

from biophysical_integration.recorded_neurons import load_recorded_neuron_indices

PROJECT_ROOT = Path(__file__).parent.parent
NEURON_DIR = PROJECT_ROOT / "neuron"
JAXLEY_MODULE = "jaxley_worm"


def _add_jaxley_path():
    """Add neuron/jaxley_worm to path."""
    p = NEURON_DIR / JAXLEY_MODULE
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
        sys.path.insert(0, str(NEURON_DIR))
    return str(p)


def load_jaxley_model(
    checkpoint_dir: str,
    net_cache: str,
    checkpoint_epoch: Optional[int] = None,
    ca_cell_names_path: Optional[str] = None,
    motor_only: bool = False,
):
    """
    Load Jaxley network + checkpoint for inference.

    Args:
        checkpoint_dir: Directory with ckpt_epoch_*.pkl (e.g. "models/")
        net_cache: Path to network pickle (e.g. "models/network_synapse_location_always_soma.pkl")
        checkpoint_epoch: If set, load this specific epoch (e.g. 5000). Otherwise load latest.

    Returns:
        (simulate_fn, params_phys, inputs_phys, n_sensory, n_cells)
    """
    _add_jaxley_path()

    from utils import load_checkpoint
    from simulate import make_simulate
    from network import load_cb2022, create_transform
    import jaxley.optimize.transforms as jt
    import pickle

    input_cell_names = [
        "AWAL", "AWAR", "AWCL", "AWCR", "ASKL", "ASKR",
        "ALNL", "ALNR", "PLML", "PHAL", "PHAR",
        "URYDL", "URYDR", "URYVL", "URYVR",
    ]
    total_network_dict = load_cb2022()
    n_cells = len(total_network_dict)

    # Load pre-built network (from prior --build-net run)
    with open(net_cache, "rb") as f:
        net = pickle.load(f)

    # Must match run_jaxley_worm.py: setup recordings and trainables before get_parameters()
    net.delete_recordings()
    for cell in range(n_cells):
        net.cell(cell).branch(0).loc(0.5).record("v", verbose=False)
    net.delete_trainables()
    net.ElectricalSynapse.edge("all").make_trainable("ElectricalSynapse_g_gap", verbose=False)
    net.Excitatory_chemical_synapse.edge("all").make_trainable("Excitatory_chemical_synapse_gS", verbose=False)
    net.Inhibitory_chemical_synapse.edge("all").make_trainable("Inhibitory_chemical_synapse_gS", verbose=False)
    channel_names = ["kcnl", "kqt3", "kvs1", "cca1", "egl2", "egl19", "egl36", "irk",
        "nca", "shk1", "shl1", "slo1_egl19", "slo1_unc2", "slo2_egl19",
        "slo2_unc2", "unc2"]
    for ch in channel_names:
        net.cell("all").branch(0).loc("all").make_trainable(f"{ch}_w", verbose=False)

    remapped_network_dict = {i: v for i, v in enumerate(total_network_dict.values())}
    sensory_network_dict = {v: k for k, v in remapped_network_dict.items() if v in input_cell_names}
    n_sensory = len(input_cell_names)

    # Transforms (must match network.py create_transform)
    parameters = net.get_parameters()
    transforms = [{k: create_transform(k) for k in param} for param in parameters]
    tf = jt.ParamTransform(transforms)
    tf_inputs = jt.ParamTransform({"S": jt.SigmoidTransform(-0.005, 0.005)})

    if checkpoint_epoch is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch_{checkpoint_epoch}.pkl")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        print(f"Restored checkpoint from {ckpt_path}")
    else:
        ckpt = load_checkpoint(checkpoint_dir)
    if ckpt is None or "opt_params" not in ckpt:
        raise FileNotFoundError(f"No valid checkpoint in {checkpoint_dir}")

    opt_params = ckpt["opt_params"]
    params_phys = tf.forward(opt_params["net"])
    inputs_opt = opt_params.get("inputs")
    if inputs_opt is not None:
        inputs_phys = tf_inputs.forward(inputs_opt)
    else:
        inputs_phys = {"S": np.zeros((n_sensory, 1000), dtype=np.float64)}

    DT_MS = 5.0 / 3.0
    WINDOW_T = 6

    vmapped_sim = make_simulate(
        net,
        window_T=WINDOW_T,
        dt_ms=DT_MS,
        n_cells=n_sensory,
        record_vars=("v",),
        use_external_input=True,
        input_cell_names=input_cell_names,
        sensory_network_dict=sensory_network_dict,
    )

    # Indices of recorded neurons (named subset from Ca dataset) for muscle readout
    # Prefer: checkpoint extra, then recorded_neuron_indices.npy, else derive from Ca_traces_cell_name.txt
    obs_idx = None
    if ckpt.get("obs_idx") is not None:
        obs_idx = np.asarray(ckpt["obs_idx"], dtype=np.int64)
        print(f"Using obs_idx from checkpoint ({len(obs_idx)} neurons)")
    else:
        idx_path = os.path.join(checkpoint_dir, "recorded_neuron_indices.npy")
        if os.path.exists(idx_path):
            obs_idx = np.load(idx_path).astype(np.int64)
            print(f"Loaded obs_idx from {idx_path} ({len(obs_idx)} neurons)")
        else:
            try:
                subset = "motor" if motor_only else "all"
                obs_idx, labels = load_recorded_neuron_indices(ca_cell_names_path, subset=subset)
                print(f"Derived obs_idx from Ca_traces_cell_name.txt ({len(obs_idx)} {subset} neurons)")
            except FileNotFoundError:
                obs_idx = np.arange(min(N_NEURONS, n_cells), dtype=np.int64)
                print(f"Ca file not found: using first {len(obs_idx)} neurons as fallback")

    return vmapped_sim, params_phys, inputs_phys, n_sensory, n_cells, obs_idx


class JaxleyBiophysicalRNN:
    """
    Wrapper: step(S) -> voltages for BiophysicalRNNWrapper.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        net_cache: str,
        checkpoint_epoch: Optional[int] = None,
        ca_cell_names_path: Optional[str] = None,
        motor_only: bool = False,
        max_steps: int = 200000,
    ):
        out = load_jaxley_model(checkpoint_dir, net_cache, checkpoint_epoch, ca_cell_names_path, motor_only)
        self.vmapped_sim, self.params_phys, self.inputs_phys, self.n_sensory, self.n_cells, self.obs_idx = out
        self.max_steps = max_steps
        self.S_buffer = np.zeros((self.n_sensory, max_steps), dtype=np.float64)
        self.t0 = 0
        self.carry = None

    def reset(self):
        self.t0 = 0
        self.carry = None
        self.S_buffer = np.zeros((self.n_sensory, self.max_steps), dtype=np.float64)

    def step(self, S: np.ndarray) -> np.ndarray:
        """S (n_sensory,) -> mean voltage (n_recorded,) over 6 neural steps."""
        if self.t0 + 6 > self.max_steps:
            raise RuntimeError(f"Exceeded max_steps={self.max_steps}")

        S = np.asarray(S, dtype=np.float64).flatten()[: self.n_sensory]
        for j in range(6):
            self.S_buffer[:, self.t0 + j] = S

        self.inputs_phys["S"] = self.S_buffer
        traces, self.carry = self.vmapped_sim(
            self.params_phys, self.inputs_phys, self.t0, self.carry
        )
        V = traces["v"]
        self.t0 += 6
        V_mean = np.mean(np.asarray(V), axis=1)
        # Select recorded neurons (named subset) for muscle readout
        if self.obs_idx is not None:
            V_mean = V_mean[self.obs_idx]
        return V_mean.astype(np.float32)
