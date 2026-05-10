"""Run the GLM policy PCA-dimension by policy-class experiment grid."""

from __future__ import annotations

import argparse

from experiments.policy_pca_grid import PCA_DIMS, POLICY_CLASSES, PolicyPcaGridSpec, run_policy_pca_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--estimator", default="first_order")
    parser.add_argument("--t-steps", type=int, default=1000)
    parser.add_argument("--project-name", default="policy-pca-grid")
    parser.add_argument("--quiet", action="store_true", help="Disable per-condition progress output.")
    args = parser.parse_args()

    spec = PolicyPcaGridSpec(
        pca_dims=PCA_DIMS,
        policy_classes=POLICY_CLASSES,
        seeds=tuple(args.seeds),
        n_samples=args.n_samples,
        estimator=args.estimator,
        t_steps=args.t_steps,
        project_name=args.project_name,
        verbose=not args.quiet,
    )
    output = run_policy_pca_grid(spec)
    print(f"Wrote policy PCA grid outputs to {output.output_dir}")


if __name__ == "__main__":
    main()
