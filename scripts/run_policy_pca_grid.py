"""Run the GLM policy PCA-dimension by policy-class experiment grid."""

from __future__ import annotations

import argparse

from data.loader import load_mean_observed_acceptance
from experiments.policy_pca_grid import PCA_DIMS, POLICY_CLASSES, PolicyPcaGridSpec, run_policy_pca_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--estimator", default="first_order")
    parser.add_argument("--t-steps", type=int, default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--acceptance-floor", type=float, default=None)
    parser.add_argument("--initial-constr-penalty", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true", help="Disable per-condition progress output.")
    args = parser.parse_args()

    step_rule = "trust-constr" if args.constrained else "l-bfgs-b"
    t_steps = args.t_steps if args.t_steps is not None else (500 if args.constrained else 1000)
    project_name = args.project_name or (
        "policy-pca-grid-constrained" if args.constrained else "policy-pca-grid"
    )
    acceptance_floor = (
        args.acceptance_floor
        if args.acceptance_floor is not None
        else load_mean_observed_acceptance("glm") if args.constrained else None
    )

    spec = PolicyPcaGridSpec(
        pca_dims=PCA_DIMS,
        policy_classes=POLICY_CLASSES,
        seeds=tuple(args.seeds),
        n_samples=args.n_samples,
        estimator=args.estimator,
        step_rule=step_rule,
        t_steps=t_steps,
        initial_constr_penalty=args.initial_constr_penalty if args.constrained else None,
        acceptance_floor=acceptance_floor,
        project_name=project_name,
        verbose=not args.quiet,
    )
    output = run_policy_pca_grid(spec)
    print(f"Wrote policy PCA grid outputs to {output.output_dir}")


if __name__ == "__main__":
    main()
