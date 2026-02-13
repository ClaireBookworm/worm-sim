import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import jaxley
from jaxley.synapses.synapse import Synapse
from jaxley.connect import connect
from typing import Dict, Optional, Tuple
from jaxley.solver_gate import save_exp
import matplotlib
matplotlib.use('Agg')  # For headless save-to-file
import matplotlib.pyplot as plt
import re
import argparse
from channel_mechanisms import *
import pickle
from jax import lax
from dataclasses import dataclass
import datetime, pathlib
from pathlib import Path
from jax import config
import jax.tree_util as jtu
from contextlib import contextmanager
import optax

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

# ---------- W&B logging helpers ----------
def _flatten_key(path_str: str, prefix: str = "") -> str:
    # 'net/gap/g' -> 'net.gap.g' (with optional prefix)
    return (prefix + path_str.replace("/", ".")).strip(".")

def _iter_with_paths(tree):
    """(re)define if not in scope: yield (path_str, leaf) over nested pytrees."""
    def walk(node, path):
        if isinstance(node, dict):
            for k in node.keys():
                yield from walk(node[k], path + (str(k),))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                yield from walk(v, path + (str(i),))
        else:
            yield "/".join(path), node
    yield from walk(tree, ())

def tree_stats_dict(tree, *, prefix: str = "") -> dict:
    """Compute per-leaf stats suitable for wandb.log."""
    out = {}
    for path, leaf in _iter_with_paths(tree):
        if isinstance(leaf, (jnp.ndarray, np.ndarray)):
            arr = np.asarray(leaf)
            key = _flatten_key(path, prefix)
            # basic size + numerics
            out[f"{key}/size"]  = int(arr.size)
            out[f"{key}/n_nan"] = int(np.isnan(arr).sum())
            out[f"{key}/n_inf"] = int(np.isinf(arr).sum())
            # safe nan-aware stats
            with np.errstate(all="ignore"):
                out[f"{key}/mean"] = float(np.nanmean(arr))
                out[f"{key}/std"]  = float(np.nanstd(arr))
                out[f"{key}/min"]  = float(np.nanmin(arr))
                out[f"{key}/max"]  = float(np.nanmax(arr))
                out[f"{key}/l2"]   = float(np.sqrt(np.nansum(arr.astype(np.float64)**2)))
    return out

# utils.py  — replace log_tree_to_wandb with this richer version
def log_tree_to_wandb(tree, prefix="", step=None, log_hist=False):
    import numpy as np, wandb, jax.tree_util as jtu

    def _as_np(x):
        try:
            return np.asarray(x).astype(np.float64).ravel()
        except Exception:
            return np.array([], dtype=np.float64)

    payload = {}
    for path, x in jtu.tree_leaves_with_path(tree):
        key = prefix + "/".join([str(k) for k in path])
        arr = _as_np(x)
        if arr.size == 0:
            continue

        finite = np.isfinite(arr)
        n_finite = int(finite.sum())

        if n_finite > 0:
            af = arr[finite]
            with np.errstate(all="ignore"):
                payload[key + "/mean"] = float(af.mean())
                payload[key + "/std"]  = float(af.std())
                payload[key + "/min"]  = float(af.min())
                payload[key + "/max"]  = float(af.max())
                payload[key + "/l2"]   = float(np.sqrt((af ** 2).sum()))


            if log_hist and af.size >= 2:
                try:
                    payload[key + "/hist"] = wandb.Histogram(af)
                except Exception:
                    pass

        elif n_finite == 1:  # (kept from your original for scalars)
            payload[key + "/value"] = float(arr[finite][0])

    if payload:
        wandb.log(payload, step=step, commit=False)



def dump_tree_npz(tree, out_path: str) -> str:
    """Serialize all ndarray leaves into a single .npz (flat dict keyed by path)."""
    flat = {}
    for path, leaf in _iter_with_paths(tree):
        if isinstance(leaf, (jnp.ndarray, np.ndarray)):
            flat[_flatten_key(path)] = np.asarray(leaf)
    np.savez_compressed(out_path, **flat)
    return out_path
# ---------- end W&B helpers ----------


def tree_all_finite(t):
    leaves, _ = jtu.tree_flatten(t)
    return all((not isinstance(x, jnp.ndarray)) or bool(jnp.isfinite(x).all())
               for x in leaves)

def _iter_with_paths(tree):
    """Yield (path_str, leaf) over arrays in a nested dict/list/tuple tree."""
    def walk(node, path):
        if isinstance(node, dict):
            for k in node.keys():  # keep original order
                yield from walk(node[k], path + (str(k),))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                yield from walk(v, path + (str(i),))
        else:
            yield "/".join(path), node
    yield from walk(tree, ())

def assert_finite_tree_or_raise_detailed(tree, tag: str):
    """
    Installs per-leaf checks that raise with a precise path and diagnostics
    if *any* ndarray leaf contains a non-finite value.
    """
    for path, leaf in _iter_with_paths(tree):
        if isinstance(leaf, jnp.ndarray):
            x = leaf

            isfinite = jnp.isfinite(x)
            ok       = jnp.all(isfinite)

            n_nan    = jnp.sum(jnp.isnan(x))
            n_inf    = jnp.sum(jnp.isinf(x))
            n_bad    = jnp.size(x) - jnp.sum(isfinite)

            # stats (nan-aware so they still compute on mixed arrays)
            x_min    = jnp.nanmin(x)
            x_max    = jnp.nanmax(x)
            x_mean   = jnp.nanmean(x)

            # find first bad flat index for a concrete coordinate report
            bad_mask = jnp.logical_not(isfinite).reshape(-1)
            # arange + where to avoid argmax-on-all-false issues; we only use it when !ok
            flat_idx = jnp.where(bad_mask, jnp.arange(bad_mask.size), -1).max()

            shape = x.shape  # Python object; safe to close over

            def _cb(ok_val, n_nan_val, n_inf_val, n_bad_val, flat_idx_val,
                    x_min_val, x_max_val, x_mean_val):
                if not bool(ok_val):
                    # unravel the first bad index using the known static shape
                    bad_flat = int(flat_idx_val)
                    try:
                        bad_idx = np.unravel_index(bad_flat, shape)
                    except Exception:
                        bad_idx = (bad_flat,)
                    raise FloatingPointError(
                        f"{tag}: non-finite at '{path}' "
                        f"shape={shape} "
                        f"n_nan={int(n_nan_val)} n_inf={int(n_inf_val)} n_bad={int(n_bad_val)} "
                        f"stats[min={float(x_min_val):.6g}, max={float(x_max_val):.6g}, mean={float(x_mean_val):.6g}] "
                        f"first_bad_idx={bad_idx}"
                    )

            jax.debug.callback(
                _cb, ok, n_nan, n_inf, n_bad, flat_idx, x_min, x_max, x_mean
            )
    return tree

def tree_all_finite_np(tree):
    """Host-side check: True iff every ndarray leaf is finite."""
    leaves, _ = jtu.tree_flatten(tree)
    for x in leaves:
        if isinstance(x, (jnp.ndarray, np.ndarray)):
            if not np.isfinite(np.asarray(x)).all():
                return False
    return True

def snapshot_pytree(tree):
    """Make an immutable snapshot (structure + array references are fine)."""
    return jtu.tree_map(lambda x: x, tree)

def revert_to_last_good(last_good):
    print("Reverting to last known good params/state.")
    return snapshot_pytree(last_good["params"]), snapshot_pytree(last_good["opt_state"])


def save_checkpoint(epoch, opt_params_all, opt_state, loss, path, extra=None):
    if not tree_all_finite(opt_params_all):
        print("Refusing to save checkpoint: non-finite parameters.")
        return
    ckpt = {"epoch": epoch, "opt_params": opt_params_all, "opt_state": opt_state, "loss": float(loss)}
    if extra is not None:
        ckpt.update(extra)
    fname = os.path.join(path, f"ckpt_epoch_{epoch:04d}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"Saved checkpoint to {fname}")


import os, re, pickle

def load_checkpoint(path):
    # Only accept files saved by save_checkpoint()
    pat = re.compile(r"^ckpt_epoch_(\d+)\.pkl$")
    matches = []
    for f in os.listdir(path):
        m = pat.match(f)
        if m:
            matches.append((int(m.group(1)), f))

    if not matches:
        return None  # or raise FileNotFoundError(...) if you prefer

    _, latest = max(matches, key=lambda x: x[0])
    with open(os.path.join(path, latest), "rb") as f:
        ckpt = pickle.load(f)
    print(f"Restored checkpoint from {os.path.join(path, latest)}")
    return ckpt



def _prep_for_metrics(voltages_NT, labels_TN, burn_in: int = 0, align_steps: int = 0):
    """
    Returns aligned (sim_NT, tgt_NT) after optional burn-in drop and DC alignment
    using the first `align_steps` timepoints.
    """
    tgt_NT = labels_TN.T
    T = jnp.minimum(voltages_NT.shape[1], tgt_NT.shape[1])
    # burn-in: drop first B steps, keep overlap
    B = jnp.minimum(burn_in, T - 1)
    T_eff = T - B
    sim = voltages_NT[:, B:B+T_eff]
    tgt = tgt_NT[:,     B:B+T_eff]

    if align_steps and align_steps > 0:
        K = jnp.minimum(align_steps, T_eff)
        sim0 = jnp.mean(sim[:, :K], axis=1, keepdims=True)
        tgt0 = jnp.mean(tgt[:, :K], axis=1, keepdims=True)
        sim = sim + (tgt0 - sim0)
    return sim, tgt  # both (N, T_eff)


def _per_neuron_pearson(sim_NT, tgt_NT, eps=1e-8):
    x = sim_NT - jnp.mean(sim_NT, axis=1, keepdims=True)
    y = tgt_NT - jnp.mean(tgt_NT, axis=1, keepdims=True)
    num = jnp.sum(x * y, axis=1)
    den = jnp.sqrt(jnp.sum(x * x, axis=1) * jnp.sum(y * y, axis=1)) + eps
    return num / den  # (N,)


def _corr_matrix_from_traces(traces_NT, eps=1e-8):
    X = traces_NT - jnp.mean(traces_NT, axis=1, keepdims=True)
    std = jnp.sqrt(jnp.sum(X * X, axis=1, keepdims=True) + eps)
    C = X / std
    return C @ C.T


def _corr_loss(sim_NT, tgt_NT, eps: float = 1e-8):
    C_sim = _corr_matrix_from_traces(sim_NT, eps=eps)
    C_tgt = _corr_matrix_from_traces(tgt_NT, eps=eps)
    n = C_sim.shape[0]
    iu = jnp.triu_indices(n, k=1)
    diff = C_sim[iu] - jax.lax.stop_gradient(C_tgt[iu])
    return jnp.mean(diff * diff)

def _matrix_correlation(C1, C2, eps=1e-8):
    # Pearson r between upper triangles (excluding diagonal)
    n = C1.shape[0]
    iu = jnp.triu_indices(n, k=1)
    a = C1[iu]
    b = C2[iu]
    a = a - jnp.mean(a); b = b - jnp.mean(b)
    return jnp.sum(a * b) / (jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b)) + eps)


def compute_metrics(voltages_NT, labels_TN, burn_in=20, align_steps=80):
    """
    Returns:
      r_per_neuron: (N,) Pearson r for each neuron’s time series
      r_mean: scalar, mean over neurons
      C_sim, C_tgt: (N,N) correlation matrices
      r_struct: scalar, Pearson r between upper triangles of C_sim and C_tgt
      r_global: scalar, Pearson r after flattening all (N,T) timepoints
    """
    sim, tgt = _prep_for_metrics(voltages_NT, labels_TN, burn_in=burn_in, align_steps=align_steps)

    # per-neuron r
    r_per_neuron = _per_neuron_pearson(sim, tgt)
    r_mean = jnp.mean(r_per_neuron)

    # correlation matrices across neurons
    C_sim = _corr_matrix_from_traces(sim)
    C_tgt = _corr_matrix_from_traces(tgt)
    r_struct = _matrix_correlation(C_sim, C_tgt)

    # global r across all timepoints & neurons (sanity check)
    x = (sim - jnp.mean(sim)).ravel()
    y = (tgt - jnp.mean(tgt)).ravel()
    r_global = jnp.sum(x * y) / (jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y)+ 1e-8))

    return {
        "r_per_neuron": r_per_neuron,
        "r_mean": r_mean,
        "C_sim": C_sim,
        "C_tgt": C_tgt,
        "r_struct": r_struct,
        "r_global": r_global,
    }


def save_voltage_plot(
    V_win,
    labels_win,
    names,
    *,
    epoch,
    outdir,
    mb=None,
    t0=None,
    burn=0,
    dt_ms=5.0/3.0,
    max_traces=8,
    ylabel: str = "membrane V (mV)",
):
    V = np.asarray(V_win); Y = np.asarray(labels_win)
    N, W = V.shape
    Path(outdir).mkdir(parents=True, exist_ok=True)

    idx = np.arange(min(N, max_traces))
    sel_names = [names[i] if i < len(names) else f"n{i}" for i in idx]
    t = np.arange(W - burn) * (dt_ms / 1000.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # One distinct color per neuron
    colors = plt.cm.tab20(np.linspace(0, 1, len(idx)))

    for j, (i, nm) in enumerate(zip(idx, sel_names)):
        c = colors[j]
        # prediction: solid
        ax.plot(t, V[i, burn:], lw=1.6, color=c, label=f"{nm} pred")
        # ground truth: dashed, same color
        ax.plot(t, Y[i, burn:], lw=1.0, ls="--", color=c, label=f"{nm} gt")
        
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)
    ttl = f"Eval window @ t0={t0}"
    if epoch is not None: ttl += f" — epoch {epoch}"
    if mb is not None:    ttl += f", mb {mb}"
    ax.set_title(ttl); ax.grid(True, alpha=0.3)
    leg = ax.legend(ncol=2, fontsize=8, framealpha=0.9); leg.get_frame().set_linewidth(0.0)

    fname = f"eval_ep{epoch:05d}" if epoch is not None else "eval"
    if mb is not None: fname += f"_mb{mb:05d}"
    if t0 is not None: fname += f"_t0{t0:06d}"
    fig.savefig(Path(outdir) / f"{fname}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_ground_truth_reference(
    labels_win,
    names,
    *,
    t0,
    outdir,
    burn=0,
    dt_ms=5.0/3.0,
    max_traces=8,
    ylabel: str = "membrane V (mV)",
):

    Y = np.asarray(labels_win)
    N, W = Y.shape
    Path(outdir).mkdir(parents=True, exist_ok=True)

    idx = np.arange(min(N, max_traces))
    sel_names = [names[i] if i < len(names) else f"n{i}" for i in idx]
    t = np.arange(W - burn) * (dt_ms / 1000.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, nm in zip(idx, sel_names):
        ax.plot(t, Y[i, burn:], lw=1.4, label=f"{nm} gt")
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)

    ax.set_title(f"Ground truth (one-time reference) — t0={t0}")
    ax.grid(True, alpha=0.3)
    leg = ax.legend(ncol=2, fontsize=8, framealpha=0.9); leg.get_frame().set_linewidth(0.0)

    fig.savefig(Path(outdir) / f"GT_t0{t0:06d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_simulation_reference(
    V_full,
    names,
    *,
    outdir,
    burn=0,
    dt_ms=5.0/3.0,
    max_traces=8,
    epoch=None,
    ylabel: str = "membrane V (mV)",
):

    V = np.asarray(V_full)
    N, W = V.shape
    Path(outdir).mkdir(parents=True, exist_ok=True)

    idx = np.arange(min(N, max_traces))
    sel_names = [names[i] if i < len(names) else f"n{i}" for i in idx]
    t = np.arange(W - burn) * (dt_ms / 1000.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, nm in zip(idx, sel_names):
        ax.plot(t, V[i, burn:], lw=1.4, label=f"{nm} sim")
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)
    ttl = "Simulation (full length)" + (f" — epoch {epoch}" if epoch is not None else "")
    ax.set_title(ttl); ax.grid(True, alpha=0.3)
    leg = ax.legend(ncol=2, fontsize=8, framealpha=0.9); leg.get_frame().set_linewidth(0.0)

    fname = ("SIM_full.png" if epoch is None else f"SIM_full_ep{epoch:05d}.png")
    fig.savefig(Path(outdir) / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _flatten_with_paths(node, prefix=()):
    # Walk mappings (dict, FrozenDict, etc.) and sequences (list/tuple),
    # and yield (path, jnp.ndarray) leaves. Ignores None/unsupported leaves.
    from collections.abc import Mapping, Sequence
    import jax.numpy as jnp

    if isinstance(node, Mapping):
        for k, v in node.items():
            yield from _flatten_with_paths(v, prefix + (str(k),))
        return

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for i, v in enumerate(node):
            yield from _flatten_with_paths(v, prefix + (str(i),))
        return

    if node is None:
        return

    try:
        arr = jnp.asarray(node)
    except TypeError:
        # Unknown/unsupported object (e.g., optimizer sentinel) — skip
        return

    yield ("/".join(prefix), arr)

def avg_abs_grad_by_param(grad_tree, group_top: str | None = None):
    """
    Return {param_name: mean(|grad|)} aggregated over leaves that share the same last path segment.
    Example: .../Leak_gLeak and .../AWAL/Leak_gLeak both contribute to key 'Leak_gLeak'.
    """
    import jax.numpy as jnp

    if group_top is not None and isinstance(grad_tree, dict) and group_top in grad_tree:
        grad_tree = grad_tree[group_top]

    buckets, counts = {}, {}
    for path, arr in _flatten_with_paths(grad_tree):
        key = path.split("/")[-1]
        val = jnp.mean(jnp.abs(arr))
        buckets[key] = buckets.get(key, 0.0) + float(val)
        counts[key] = counts.get(key, 0) + 1

    if not buckets:
        return {}

    return {k: (buckets[k] / counts[k]) for k in buckets}



def _finite_1d(x, *, max_elems=None, rng=None):
    """Return finite values as 1D float64 array, optionally subsampled."""
    try:
        a = np.asarray(x, dtype=np.float64).ravel()
    except Exception:
        return np.array([], dtype=np.float64)
    m = np.isfinite(a)
    if not np.any(m):
        return np.array([], dtype=np.float64)
    af = a[m]
    if (max_elems is not None) and (af.size > max_elems):
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(af.size, size=max_elems, replace=False)
        af = af[idx]
    return af

def bucket_stats_by_paramname(tree, *, prefix="", q=(5, 95), absval=False, max_elems_per_bucket=200_000):
    """
    Aggregate stats over leaves that share the same last path segment.
    Returns flat dict suitable for wandb.log, e.g.:
      phys/net/Leak_gLeak/mean, phys/net/Leak_gLeak/p05, phys/net/Leak_gLeak/p95, ...
    """
    from utils import _flatten_with_paths  # you already have this in-file :contentReference[oaicite:10]{index=10}
    rng = np.random.default_rng(0)

    buckets = {}
    for path, arr in _flatten_with_paths(tree):
        key = path.split("/")[-1]
        af = _finite_1d(arr, max_elems=max_elems_per_bucket, rng=rng)
        if af.size == 0:
            continue
        if absval:
            af = np.abs(af)
        buckets.setdefault(key, []).append(af)

    out = {}
    for k, parts in buckets.items():
        v = np.concatenate(parts, axis=0)
        if v.size == 0:
            continue
        p_lo, p_hi = np.percentile(v, q)
        base = f"{prefix}/{k}".strip("/")
        out[f"{base}/mean"] = float(v.mean())
        out[f"{base}/p{q[0]:02d}"] = float(p_lo)
        out[f"{base}/p{q[1]:02d}"] = float(p_hi)
        out[f"{base}/min"] = float(v.min())
        out[f"{base}/max"] = float(v.max())
        out[f"{base}/n"]   = int(v.size)
    return out

def global_stats(tree, *, prefix="", q=(5, 95), absval=False, max_elems=500_000):
    """Same stats, but pooled over *all* leaves."""
    from utils import _flatten_with_paths
    rng = np.random.default_rng(0)
    parts = []
    for _, arr in _flatten_with_paths(tree):
        af = _finite_1d(arr, max_elems=max_elems, rng=rng)
        if af.size:
            parts.append(np.abs(af) if absval else af)
    if not parts:
        return {}
    v = np.concatenate(parts, axis=0)
    p_lo, p_hi = np.percentile(v, q)
    base = prefix.strip("/")
    return {
        f"{base}/mean": float(v.mean()),
        f"{base}/p{q[0]:02d}": float(p_lo),
        f"{base}/p{q[1]:02d}": float(p_hi),
        f"{base}/min": float(v.min()),
        f"{base}/max": float(v.max()),
        f"{base}/n": int(v.size),
    }

# --- extend your existing log_tree_to_wandb to include percentiles ---
def log_tree_to_wandb(tree, prefix="", step=None, log_hist=False):
    import numpy as np, wandb, jax.tree_util as jtu

    def _as_np(x):
        try:
            return np.asarray(x).astype(np.float64).ravel()
        except Exception:
            return np.array([], dtype=np.float64)

    payload = {}
    for path, x in jtu.tree_leaves_with_path(tree):
        key = prefix + "/".join([str(k) for k in path])
        arr = _as_np(x)
        if arr.size == 0:
            continue

        finite = np.isfinite(arr)
        n_finite = int(finite.sum())

        if n_finite > 0:
            af = arr[finite]
            with np.errstate(all="ignore"):
                payload[key + "/mean"] = float(af.mean())
                payload[key + "/std"]  = float(af.std())
                payload[key + "/min"]  = float(af.min())
                payload[key + "/max"]  = float(af.max())
                payload[key + "/l2"]   = float(np.sqrt((af ** 2).sum()))
                # NEW:
                p05, p95 = np.percentile(af, [5, 95])
                payload[key + "/p05"] = float(p05)
                payload[key + "/p95"] = float(p95)

            if log_hist and af.size >= 2:
                try:
                    payload[key + "/hist"] = wandb.Histogram(af)
                except Exception:
                    pass

        elif n_finite == 1:
            payload[key + "/value"] = float(arr[finite][0])

    if payload:
        wandb.log(payload, step=step, commit=False)


def tree_vdot(a, b):
    return float(sum(jnp.vdot(x, y) for x, y in zip(jtu.tree_leaves(a), jtu.tree_leaves(b))))
def tree_norm(t):
    return float(optax.global_norm(jtu.tree_leaves(t)))