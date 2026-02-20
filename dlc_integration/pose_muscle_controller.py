"""
Step 2b/c: Train NN controller: pose → muscle activations.

Given (pose, muscles) pairs from inverse dynamics, train a network to predict
muscles from pose. Use for closed-loop control.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_JOINTS = 23
N_MUSCLES = 96
N_SIMPLIFIED = 24


class MuscleController(nn.Module if TORCH_AVAILABLE else object):
    """
    NN: target pose (23 joint angles) → muscle activations (24 or 96).
    """

    def __init__(
        self,
        n_joints: int = N_JOINTS,
        n_muscles: int = N_SIMPLIFIED,
        hidden: int = 128,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MuscleController")
        super().__init__()
        self.n_joints = n_joints
        self.n_muscles = n_muscles
        self.net = nn.Sequential(
            nn.Linear(n_joints, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_muscles),
            nn.Sigmoid() if n_muscles == N_SIMPLIFIED else nn.Tanh(),
        )

    def forward(self, pose):
        return self.net(pose)

    def predict(self, pose: np.ndarray) -> np.ndarray:
        """NumPy interface."""
        with torch.no_grad():
            x = torch.FloatTensor(np.asarray(pose))
            if pose.ndim == 1:
                x = x.unsqueeze(0)
            out = self.forward(x)
            return out.numpy().squeeze()


def train_pose_muscle_controller(
    poses: np.ndarray,
    muscles: np.ndarray,
    epochs: int = 1000,
    lr: float = 1e-3,
    val_split: float = 0.1,
) -> "MuscleController":
    """
    Train pose → muscle controller.

    Args:
        poses: (T, 23) joint angles
        muscles: (T, 24) or (T, 96) - should be in [0,1] for 24 or [-1,1] for 96
        epochs: Training epochs
        lr: Learning rate
        val_split: Fraction for validation

    Returns:
        Trained MuscleController
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    n_muscles = muscles.shape[1]
    model = MuscleController(n_muscles=n_muscles)

    # Split (ensure at least 1 train sample; val can be 0 for tiny datasets)
    n = len(poses)
    perm = np.random.permutation(n)
    n_val = max(0, min(int(n * val_split), n - 1))  # keep at least 1 for train
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train = torch.FloatTensor(poses[train_idx])
    y_train = torch.FloatTensor(muscles[train_idx])
    has_val = len(val_idx) > 0
    if has_val:
        X_val = torch.FloatTensor(poses[val_idx])
        y_val = torch.FloatTensor(muscles[val_idx])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        pred = model(X_train)
        loss = ((pred - y_train) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            val_str = ""
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    val_loss = ((val_pred - y_val) ** 2).mean().item()
                val_str = f" val MSE={val_loss:.6f}"
            else:
                val_str = " (no val set, n<10)"
            print(f"Epoch {epoch}: train MSE={loss.item():.6f}{val_str}")

    return model
