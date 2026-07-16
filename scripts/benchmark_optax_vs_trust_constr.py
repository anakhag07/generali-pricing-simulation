"""Benchmark SciPy minimize (L-BFGS-B / trust-constr) against optax step rules.

Two benchmark groups:

* ``planted_logistic``: NumPy planted-logistic objective with a LinearPolicy
  sized to ``--state-dim`` (default theta dim 200), comparing SciPy L-BFGS-B
  against ``optax-adam`` / ``optax-sgd``.
* ``glm_jax``: JAX prepared GLM objective on sampled real-data rows, comparing
  SciPy trust-constr with the observed acceptance floor against ``optax-adam``
  on the smooth acceptance-penalty formulation of the same floor.

Writes ``benchmark.csv`` under ``results/optax-benchmark/benchmark_<ts>/`` and
prints a summary table. Solver wall time is measured around
``Optimization.solve`` only; JAX objectives are warmed up (compiled) first.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.configs import get_config  # noqa: E402
from experiments.paths import results_root  # noqa: E402
from objective.base import sample_states  # noqa: E402
from objective.objectives.generali.jax_prepared_glm import JaxPreparedGLMObjective  # noqa: E402
from objective.objectives.synthetic.planted_logistic import PlantedLogisticObjective  # noqa: E402
from objective.objectives.generali.prepared_glm import prepare_glm_batch  # noqa: E402
from objective.policy import LinearPolicy  # noqa: E402
from optimization import FirstOrderGradient, Optimization  # noqa: E402

FIELDNAMES = [
    "group",
    "algorithm",
    "jax_backend",
    "theta_dim",
    "n_rows",
    "wall_time_s",
    "n_steps",
    "time_per_step_s",
    "final_value",
    "final_raw_value",
    "mean_u_gap",
    "mean_acceptance",
    "constraint_violation",
    "penalty_weight",
]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dim", type=int, default=199, help="planted-logistic state dim (theta dim = state_dim + 1)")
    parser.add_argument("--logistic-rows", type=int, default=4096)
    parser.add_argument("--logistic-steps", type=int, default=300)
    parser.add_argument("--logistic-lr", type=float, default=0.05)
    parser.add_argument("--glm-rows", type=int, default=20000, help="sampled real-data rows; <= 0 uses all complete eligible rows")
    parser.add_argument("--glm-steps", type=int, default=100)
    parser.add_argument("--glm-adam-lr", type=float, default=0.02)
    parser.add_argument(
        "--glm-policy-kind",
        type=str,
        default="softmax",
        choices=("constant", "linear", "softmax", "mlp"),
        help="policy class for the GLM group (mlp uses the JAX MLP backend)",
    )
    parser.add_argument(
        "--glm-penalty-weights",
        type=str,
        default="1e4",
        help="comma-separated acceptance_penalty_weight values; optax-adam runs once per weight",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-glm", action="store_true", help="skip the real-data GLM group (no artifacts needed)")
    parser.add_argument("--skip-logistic", action="store_true", help="skip the planted-logistic group (e.g. GLM-only GPU scaling runs)")
    return parser.parse_args(argv)


def _timed_solve(
    objective: object,
    x_samples: object,
    theta0: np.ndarray,
    *,
    algorithm: str,
    t_steps: int,
    step_size: float,
    initial_constr_penalty: float | None = None,
) -> tuple[np.ndarray, object, float]:
    optimizer = Optimization(
        objective,
        x_samples,
        FirstOrderGradient(),
        algorithm=algorithm,
        t_steps=t_steps,
        n_grad_samples=1,
        sigma=0.1,
        step_size=step_size,
        initial_constr_penalty=initial_constr_penalty,
        batch_rng=np.random.default_rng(1),
        gradient_rng=np.random.default_rng(2),
    )
    start = time.perf_counter()
    theta_final, trace = optimizer.solve(theta0)
    elapsed = time.perf_counter() - start
    return theta_final, trace, elapsed


def _jax_backend() -> str:
    import jax

    return jax.default_backend()


def _base_row(group: str, algorithm: str, theta_dim: int, n_rows: int, trace: object, elapsed: float) -> dict:
    n_steps = max(len(trace.steps) - 1, 1)
    return {
        "group": group,
        "algorithm": algorithm,
        "jax_backend": _jax_backend(),
        "theta_dim": theta_dim,
        "n_rows": n_rows,
        "wall_time_s": round(elapsed, 4),
        "n_steps": n_steps,
        "time_per_step_s": round(elapsed / n_steps, 6),
        "final_value": trace.objective_values[-1],
        "final_raw_value": "",
        "mean_u_gap": "",
        "mean_acceptance": "",
        "constraint_violation": "",
        "penalty_weight": "",
    }


def run_planted_logistic_group(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    objective = PlantedLogisticObjective.from_parameters(
        policy=LinearPolicy(),
        alpha=2.0,
        beta=0.3 * rng.normal(size=args.state_dim),
        bias=-0.5,
        u_star=1.1,
    )
    x_samples = sample_states(rng, args.logistic_rows, args.state_dim)
    theta_dim = args.state_dim + 1
    theta0 = np.zeros(theta_dim, dtype=float)

    rows: list[dict] = []
    for algorithm, step_size in (
        ("l-bfgs-b", 0.01),
        ("optax-adam", args.logistic_lr),
        ("optax-sgd", args.logistic_lr),
    ):
        theta_final, trace, elapsed = _timed_solve(
            objective,
            x_samples,
            theta0,
            algorithm=algorithm,
            t_steps=args.logistic_steps,
            step_size=step_size,
        )
        row = _base_row("planted_logistic", algorithm, theta_dim, args.logistic_rows, trace, elapsed)
        u_final = objective.policy.value(theta_final, x_samples)
        row["mean_u_gap"] = abs(float(np.mean(u_final)) - objective.optimal_u())
        rows.append(row)
    return rows


def run_glm_jax_group(args: argparse.Namespace) -> list[dict]:
    n_samples = None if args.glm_rows <= 0 else args.glm_rows
    config = get_config(
        "real_data_glm_base",
        overrides={
            "n_samples": n_samples,
            "constraint_mode": "trust_constr",
            "policy_kind": args.glm_policy_kind,
            "seed": args.seed,
        },
    )
    source = config.objective
    floor_value = config.acceptance_floor
    if floor_value is None:
        floor_value = getattr(source, "acceptance_floor", None)
    floor = float(floor_value)
    batch = prepare_glm_batch(source, config.x_fixed, row_indices=config.x_fixed_row_indices)
    n_rows = int(batch.x_array.shape[0])
    shared_kwargs = dict(
        policy=source.policy,
        x_array=batch.x_array,
        u_coef=batch.u_coef,
        probability_target=batch.probability_target,
        u_bounds=getattr(source, "u_bounds", None),
        acceptance_floor=floor,
    )
    penalty_weights = [float(w) for w in args.glm_penalty_weights.split(",")]
    constrained_objective = JaxPreparedGLMObjective(**shared_kwargs)
    penalty_objectives = {
        weight: JaxPreparedGLMObjective(**shared_kwargs, acceptance_penalty_weight=weight)
        for weight in penalty_weights
    }
    theta0 = np.asarray(config.theta0, dtype=float)
    constrained_objective.warmup(theta0)
    for penalty_objective in penalty_objectives.values():
        penalty_objective.warmup(theta0)
    theta_dim = constrained_objective.policy_theta_dim()

    specs = [
        ("trust-constr", constrained_objective, config.step_size, config.initial_constr_penalty, ""),
    ]
    for weight in penalty_weights:
        specs.append(("optax-adam", penalty_objectives[weight], args.glm_adam_lr, None, weight))
    # sgd is a reference point only; run it once at the first penalty weight.
    specs.append(("optax-sgd", penalty_objectives[penalty_weights[0]], args.glm_adam_lr, None, penalty_weights[0]))

    rows: list[dict] = []
    for algorithm, objective, step_size, constr_penalty, penalty_weight in specs:
        theta_final, trace, elapsed = _timed_solve(
            objective,
            batch.x_array,
            theta0,
            algorithm=algorithm,
            t_steps=args.glm_steps,
            step_size=step_size,
            initial_constr_penalty=constr_penalty,
        )
        row = _base_row("glm_jax", algorithm, theta_dim, n_rows, trace, elapsed)
        mean_acceptance = float(objective.mean_acceptance(theta_final, batch.x_array))
        row["penalty_weight"] = penalty_weight
        row["final_raw_value"] = float(objective.base_value(theta_final, batch.x_array))
        row["mean_acceptance"] = mean_acceptance
        row["constraint_violation"] = max(0.0, floor - mean_acceptance)
        rows.append(row)
    return rows


def _write_outputs(rows: list[dict]) -> Path:
    out_dir = results_root() / "optax-benchmark" / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _print_table(rows: list[dict]) -> None:
    header = f"{'group':<18}{'algorithm':<14}{'dim':>5}{'rows':>8}{'time(s)':>10}{'steps':>7}{'s/step':>10}{'final_value':>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['group']:<18}{row['algorithm']:<14}{row['theta_dim']:>5}{row['n_rows']:>8}"
            f"{row['wall_time_s']:>10.3f}{row['n_steps']:>7}{row['time_per_step_s']:>10.5f}"
            f"{row['final_value']:>16.6f}"
        )
        extras = []
        if row["mean_u_gap"] != "":
            extras.append(f"|mean_u - u*|={row['mean_u_gap']:.4f}")
        if row["mean_acceptance"] != "":
            extras.append(f"mean_acceptance={row['mean_acceptance']:.4f}")
            extras.append(f"violation={row['constraint_violation']:.5f}")
            extras.append(f"raw_value={row['final_raw_value']:.4f}")
        if row["penalty_weight"] != "":
            extras.append(f"penalty_weight={row['penalty_weight']:.0e}")
        if extras:
            print(f"{'':<32}{'  '.join(extras)}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    print(f"jax default backend: {_jax_backend()}")
    rows = [] if args.skip_logistic else run_planted_logistic_group(args)
    if not args.skip_glm:
        rows.extend(run_glm_jax_group(args))
    out_path = _write_outputs(rows)
    _print_table(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
