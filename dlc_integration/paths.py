"""Path helpers for DLC integration (minimal)."""

from pathlib import Path


def output_dir_from_input_path(input_path: Path) -> Path:
    """Return parent dir of input file (e.g. for train: poses_muscles.npz -> parent)."""
    return Path(input_path).resolve().parent
