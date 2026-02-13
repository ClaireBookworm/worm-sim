from utils import *
from network import *
import optax
import jaxley as jx
import numpy as np
from jax import jit, value_and_grad
import jax.numpy as jnp
import jax
import jax.random as jr
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
import jax.scipy as jsp
from jax import lax
from utils import _corr_loss

def make_simulate(
    net,
    window_T,
    *,
    dt_ms=5.0 / 3.0,
    n_cells=None,
    record_vars=("v",),
    use_external_input: bool = True,
    input_cell_names=None,
    sensory_network_dict = None
):

    n_vars = int(len(record_vars))
    
    # Make it static for JIT (important).
    input_cell_names = tuple(input_cell_names)

    @jax.jit
    def simulate_traces(params, inputs, t0, carry_states=None):
        
        """Simulate one window and return a dict of recorded traces + carry."""
        data_stimuli = None
        if n_cells is None:
            raise ValueError("make_simulate requires n_cells")

        S_full = inputs['S']  # (N_src, T)
        if not use_external_input:
            S_full = jnp.zeros_like(S_full) ### disable input
        
        S_win  = lax.dynamic_slice_in_dim(S_full, t0, window_T, axis=1)  # (N_src, W)

        # Stimulate every cell with its own row of S_win (white noise drive).
        for k, cell in enumerate(input_cell_names):
            series = S_win[k]
            cid = sensory_network_dict[cell]
            data_stimuli = net.cell(cid).branch(0).loc(0.5).data_stimulate(series, data_stimuli)

        levels = 1 if window_T <= 64 else 3
        checkpoints = [int(np.ceil(window_T**(1 / levels))) for _ in range(levels)]
        recs, carry_states = jaxley.integrate(
            net,
            params=params,
            data_stimuli=data_stimuli,
            delta_t=dt_ms,
            return_states=True,
            solver="bwd_euler",
            voltage_solver="jaxley.dhs",
            all_states=carry_states,
            checkpoint_lengths=checkpoints,
        )


        recs_win = recs[:, :-1] if recs.shape[1] == window_T + 1 else recs

        # Split recordings into a dict keyed by variable name.
        traces = {}
        if n_vars == 1:
            traces[record_vars[0]] = recs_win
        else:
            # Assumes you record per-cell in this order: var0, var1, var0, var1, ...
            for i, var in enumerate(record_vars):
                traces[var] = recs_win[i::n_vars]

        return traces, carry_states


    return simulate_traces


def make_loss_fn(
    vmapped_sim,
    tf_net,
    tf_inputs,
    BURN,
    range_penalty=0.05,
    input_l2: float = 0.0,
    loss: str = "mse",
    *,
    fit_target: str = "voltage_proxy",
    dt_ms=5.0 / 3.0,
    obs_params=None,
    burn_only_if_carry_none: bool = True,
    obs_idx=None,
    # ---------------- Multiple-shooting (optional) ----------------
    # If `carry_unravel_fn` is provided and `carry_penalty_weight > 0`, the
    # loss will:
    #   1) Use opt_params_all['carry_inits'][win_idx] as the initial solver state
    #      for each segment (window).
    #   2) Add a continuity penalty between carry_end(win_idx) and
    #      carry_init(win_idx+1).
    #
    # This implements classic optimal-control multiple shooting (soft constraints).
    carry_unravel_fn=None,
    carry_flatten_fn=None,
    carry_penalty_weight: float = 0.0,
    ms_stride: int = None,
):

    obs_idx_arr = jnp.asarray(obs_idx, dtype=jnp.int64)

    # Default flattening helper (JIT-safe) if none provided.
    if carry_flatten_fn is None:
        carry_flatten_fn = lambda c: ravel_pytree(c)[0]

    using_ms = (
        (carry_unravel_fn is not None)
        and (carry_penalty_weight is not None)
        and (carry_penalty_weight > 0.0)
    )

    def loss_fn(opt_params_all, labels_win, t0, carry):
        net_opt = opt_params_all["net"]
        inp_opt = opt_params_all["inputs"]

        params_phys = tf_net.forward(net_opt)
        if tf_inputs is None:
            inputs_phys = inp_opt
        else:
            inputs_phys = tf_inputs.forward(inp_opt)

        assert_finite_tree_or_raise_detailed(params_phys, "params_phys")

        # ---------------- Multiple-shooting carry selection ----------------
        carry_used = carry
        win_idx = None
        if using_ms:
            if (ms_stride is None) or (ms_stride <= 0):
                raise ValueError("make_loss_fn: ms_stride must be set to WINDOW_T when using multiple shooting")
            if "carry_inits" not in opt_params_all:
                raise KeyError("opt_params_all must contain key 'carry_inits' when using multiple shooting")

            win_idx = t0 // int(ms_stride)
            carry_init_flat = opt_params_all["carry_inits"][win_idx]
            carry_used = carry_unravel_fn(carry_init_flat)

        sim_out, carry_next = vmapped_sim(params_phys, inputs_phys, t0, carry_used)

        # vmapped_sim may return either (V, carry) or (traces_dict, carry)
        if isinstance(sim_out, dict):
            traces = sim_out
        else:
            traces = {"v": sim_out}

        # Burn-in logic: decide burn based on the carry actually used.
        burn_here = BURN
        if burn_only_if_carry_none:
            burn_here = (BURN if (carry_used is None) else 0)

        V = traces.get("v", None)
        if V is None:
            raise ValueError("Voltage ('v') must be recorded (used for stability penalty).")

        # Select supervised prediction
        if fit_target in ("voltage", "voltage_proxy", "v"):
            pred = V
        elif fit_target in ("fluorescence", "fluor", "f"):
            if "CaCon_i" not in traces:
                raise ValueError("fit_target='fluorescence' requires recording 'CaCon_i'.")
            ca = traces["CaCon_i"]
            p = {} if (obs_params is None) else dict(obs_params)

            # If obs params are trainable, override gain/f0 here.
            if "obs" in opt_params_all:
                p["gain"] = jnp.exp(opt_params_all["obs"]["log_gain"])
                p["f0"]   = opt_params_all["obs"]["bias"]

            if "obs" in opt_params_all:
                s = jnp.exp(opt_params_all["obs"]["log_gain"])
                b = opt_params_all["obs"]["bias"]
            else:
                s = jnp.array(float(p.get("gain", 1.0)), dtype=ca.dtype)
                b = jnp.array(float(p.get("f0", 0.0)), dtype=ca.dtype)

            pred = jax.nn.sigmoid(b + s * ca)
        else:
            raise ValueError(f"Unknown fit_target={fit_target}")

        assert_finite_tree_or_raise_detailed(pred, "pred")

        pred_sup = jnp.take(pred, obs_idx_arr, axis=0)
        pred_eff = pred_sup[:, burn_here:]
        Y_eff = labels_win[:, burn_here:]

        # Only clip targets if training against voltage proxy
        if fit_target in ("voltage", "voltage_proxy", "v"):
            Y_eff = jnp.clip(Y_eff, -80.0, 20.0)
            pred_eff = jnp.clip(pred_eff, -80.0, 20.0)

        if loss == "corr":
            #Y_eff = jnp.clip(Y_eff, -80.0, 20.0)
            #pred_eff = jnp.clip(pred_eff, -80.0, 20.0)
            data_loss = _corr_loss(pred_eff, Y_eff)
        elif loss == "mse":
            diff = pred_eff - Y_eff
            data_loss = jnp.mean(optax.huber_loss(diff, delta=25.0))
        elif loss == "mse_corr":
            data_loss = _corr_loss(pred_eff, Y_eff) + jnp.mean(optax.huber_loss((pred_eff - Y_eff), delta=25.0))
        else:
            raise ValueError(f"Unknown loss={loss}")
        
        # Voltage stability penalty (always applied)
        V_eff = V[:, burn_here:]
        over = jax.nn.softplus(V_eff - 20.0)
        under = jax.nn.softplus(-80.0 - V_eff)
        bound_pen = jnp.mean(over**2 + under**2)

        total = data_loss + range_penalty * bound_pen

        # ---------------- Multiple-shooting continuity penalty ----------------
        cont_pen = jnp.array(0.0, dtype=total.dtype)
        if using_ms:
            carry_inits = opt_params_all["carry_inits"]
            num_windows = int(carry_inits.shape[0])

            def _cont(_):
                flat_next = carry_flatten_fn(carry_next)
                next_init = carry_inits[win_idx + 1]
                return jnp.mean((flat_next - next_init) ** 2)

            cont_pen = lax.cond(
                win_idx < (num_windows - 1),
                _cont,
                lambda _: jnp.array(0.0, dtype=total.dtype),
                operand=None,
            )
            total = total + jnp.asarray(carry_penalty_weight, dtype=total.dtype) * cont_pen

        # L2 on inputs (legacy)
        if input_l2 > 0.0:
            leaves = jtu.tree_leaves(inputs_phys)
            if leaves:
                sz = sum(x.size for x in leaves)
                total = total + input_l2 * (sum(jnp.sum(x * x) for x in leaves) / (sz + 1e-8))

        aux = {
            "pred": pred_sup,
            "labels": labels_win,
            "burn": burn_here,
            "traces": traces,
            "carry_next": carry_next,
            "cont_pen": cont_pen,
        }
        return total, aux

    return jit(value_and_grad(loss_fn, argnums=0, has_aux=True))




def build_param_labels(params):
    """Return a label pytree matching `params` (same structure), using only Python recursion."""
    def walk(node, path):
        if isinstance(node, dict):
            return {k: walk(v, path + (k,)) for k, v in node.items()}
        elif isinstance(node, (list, tuple)):
            t = [walk(v, path + (str(i),)) for i, v in enumerate(node)]
            return tuple(t) if isinstance(node, tuple) else t
        else:
            # leaf: decide group from the path
            p = "/".join(path)
            if p.startswith("inputs/"):
                return "inputs"

            return "net"
        
    return walk(params, ())

