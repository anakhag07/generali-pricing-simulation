"""Plot optimization iterates on a flipped quadratic surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def loss(theta: np.ndarray) -> float:
    """Quadratic loss minimized by the three methods."""
    return float(np.dot(theta, theta))


def surface_value(theta: np.ndarray) -> float:
    """Flipped quadratic height used in the 3D plot."""
    return -loss(theta)


def full_gradient(theta: np.ndarray) -> np.ndarray:
    return 2.0 * theta


def finite_difference_gradient(theta: np.ndarray, radius: float) -> np.ndarray:
    """Coordinate-wise two-sided finite-difference gradient."""
    gradient = np.empty_like(theta)
    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = 1.0
        gradient[index] = (
            loss(theta + radius * direction) - loss(theta - radius * direction)
        ) / (2.0 * radius)
    return gradient


def stein_difference_gradient(
    theta: np.ndarray,
    radius: float,
    n_directions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Two-sided Gaussian Stein-difference gradient estimate."""
    directions = rng.normal(size=(n_directions, theta.size))
    differences = np.asarray(
        [
            loss(theta + radius * direction) - loss(theta - radius * direction)
            for direction in directions
        ]
    )
    return np.mean((differences / (2.0 * radius))[:, None] * directions, axis=0)


def optimize(
    theta0: np.ndarray,
    gradient_fn,
    *,
    step_size: float,
    n_steps: int,
) -> np.ndarray:
    theta = np.asarray(theta0, dtype=float).copy()
    iterates = [theta.copy()]
    for _ in range(n_steps):
        theta -= step_size * gradient_fn(theta)
        iterates.append(theta.copy())
    return np.asarray(iterates)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("flipped_quadratic_optimization"),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    theta0 = np.asarray([2.4, -2.0])
    radius = 0.20
    n_steps = 14
    rng = np.random.default_rng(args.seed)

    trajectories = {
        "full_gd": optimize(theta0, full_gradient, step_size=0.18, n_steps=n_steps),
        "stein_difference_zo": optimize(
            theta0,
            lambda theta: stein_difference_gradient(theta, radius, 8, rng),
            step_size=0.10,
            n_steps=n_steps,
        ),
        "finite_difference_zo": optimize(
            theta0,
            lambda theta: finite_difference_gradient(theta, radius),
            step_size=0.12,
            n_steps=n_steps,
        ),
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "flipped_quadratic_iterates.png"
    plot_trajectories(plot_path, trajectories)
    print(f"Wrote: {plot_path}")


def plot_trajectories(path: Path, trajectories: dict[str, np.ndarray]) -> None:
    grid = np.linspace(-3.0, 3.0, 151)
    x_grid, y_grid = np.meshgrid(grid, grid)
    z_grid = -(x_grid**2 + y_grid**2)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", alpha=0.62, linewidth=0)

    styles = {
        "full_gd": {"color": "C0", "marker": "o", "linestyle": "-"},
        "stein_difference_zo": {"color": "C1", "marker": "^", "linestyle": "-"},
        "finite_difference_zo": {"color": "C2", "marker": "s", "linestyle": "--"},
    }
    for name, iterates in trajectories.items():
        heights = -(iterates[:, 0] ** 2 + iterates[:, 1] ** 2)
        ax.plot(
            iterates[:, 0],
            iterates[:, 1],
            heights,
            label=name,
            markersize=5,
            linewidth=2,
            **styles[name],
        )

    ax.scatter([0.0], [0.0], [0.0], color="black", marker="*", s=100, label="maximum")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title(r"Optimization on $f(x,y)=-(x^2+y^2)$")
    ax.view_init(elev=28, azim=-55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
