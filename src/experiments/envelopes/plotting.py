"""Shared standard Matplotlib sizes and deterministic diagnostic slice selection."""
import numpy as np

_STYLE = {
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
}


def _slice_indices(X):
    order = np.argsort(np.linalg.norm(X, axis=1), kind="stable")
    return np.array(list(dict.fromkeys(order[[0, (len(order)-1)//2, -1]])))
