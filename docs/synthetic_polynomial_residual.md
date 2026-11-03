# Fixed quartic residual versus additive polynomial features

This is the `polynomial_residual` mode of `wild_bootstrap_envelopes`, using
`manifests/synthetic-polynomial-residual-degree4.json` through the existing
`scripts/run_experiment_manifest.py` launcher. The approved first stage uses
one fixed residual and six feature degrees, not a residual-degree sweep.

## Implementation checklist

- [x] Reuse frozen-feature QR/bootstrap mathematics; add exact polynomial queries,
  Gaussian moments, asymmetric containment, and saved-state replay.
- [x] Extend the existing family dispatch without changing historical manifests.
- [x] Freeze one Gaussian design and one exact squared-exponential GP restriction.
  Separate context, action, profile, observation, method, and optimizer streams;
  pair observations/multipliers/optimizer perturbations across feature degrees.
- [x] Check noiseless recovery, independent bootstrap refits, all-real tail and
  interior certificates, moments, query parity, and repository optimizer use.
- [x] Pilot the headline repetition, freeze production optimizer settings, run
  100 coverage repetitions, then collect exactly four vector PDFs and one table.
- [x] Update MATH, README, AGENTS, exports, and run relevant tests and a demo.
- [x] Preserve source/input hashes and saved solver traces; reporting reads saved
  state only. Archive superseded attempts with relocation checksums.

## Scientific contract

See MATH section 7.22. The true contextual quadratic is unchanged. Its fixed
surrogate subtracts the explicit quartic residual. No predictor training or
feature learning occurs. Contexts stay linear and additive. The action domain
is R; numerical plotting windows do not limit queries, integration or coverage.
Noise uses the exact 300-dimensional restriction of the stated GP, with
lengthscales 1 and sqrt(0.45), not the older finite-Fourier approximation.
The GP is needed only at the saved design points. No off-grid noise process is
queried or interpolated. All off-grid residual/band queries use exact analytic
polynomial formulas. Observations are independent conditional on the frozen
design/profile; shared bootstrap coefficient draws couple every band query.

Use raw-residual coefficient-ellipsoid calibration at delta=0.05, B=1999.
Transfer with separate positive-part lower and upper radii. Whole-line coverage
is conditional on the fixed environment and estimated from 100 independent
observation/multiplier repetitions. It is not an exact finite-sample guarantee.
R2, RMSE, widths and optimization use predetermined repetition 0. Metrics use
the empirical distribution over all saved contexts and A~N(0.5,0.45).

All reported actions are repository solver endpoints. Stein differences query
the negative lower endpoint with one free action per context, full batch and
unrestricted Gaussian probes. A repository first-order solve supplies the
truth reference; 0.5 is its analytical check. Termination and any global-gap
certificates are reported separately. Failed/unknown certificates are retained.
Coverage certificates refer to exact rational interpretations of the saved
floating coefficients, not enclosures of statistical fitting roundoff.

## Deliverables

The pilot retained the inherited constant-step settings and reached the 1,000
step cap on 17/18 starts. Global gap certificates at 1e-4 covered 236–269 of
300 contexts per basis. Production uses the same repository action-space Stein
estimator with its L-BFGS-B step rule, smoothing SD 0.001, 64 perturbations,
three unchanged starts, a 1,000-step cap, and ftol=1e-14. Pilot checks on degrees
1 and 6 converged and certified all 300 endpoints at tolerance 1e-4. This is
an optimizer-setting revision; truth, observations, residual, features and bands
are unchanged. The initial pilot and solver-setting diagnostics are archived
with checksums in `scratch/retired_experiments/synthetic_polynomial_pilot_20261103.json`.

The new shared result folder is `synthetic-polynomial-residual-degree4/`.
It contains four PDFs: `01_residual_recovery.pdf` (six panels, R2/RMSE),
`02_simultaneous_coverage.pdf` (six bars and Wilson intervals),
`03_average_envelope_width.pdf` (mean full transferred width versus action),
and `04_policy_regret.pdf` (regret and RMS action error).
Frozen inputs, coefficients, bootstrap statistics, per-repetition decisions,
solver traces, plotted coordinates, summary CSV/JSON and provenance remain
alongside the figures. Report collection never fits or optimizes.

## Commands

```bash
conda activate simulation_env
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-degree4.json --launch local
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-degree4.json --launch local --collect
```

Local/Slurm launch uses the existing infrastructure. Each task owns a group of
coverage repetitions; task zero also owns the headline policies. Complete
artifacts are immutable and resumable only with matching sources and inputs.

Later stages can supply different explicit residuals and feature maps. Learned
features will require a frozen representation trained separately from beta
calibration. That extension is not implemented in this first stage.

## Completed first run

All 100 repetitions and 18 Stein solver starts completed. Every start converged;
all 1,800 selected actions have an unsmoothed global LCB gap at most 1e-4.
Transferred coverage counts by degree 1–6 are 0, 0, 0, 98, 100, 100. These are
conditional empirical counts; all-real misspecification explains degrees 1–3.
The matched quartic headline R2 is 0.90156 and RMSE is 0.09161. The quadratic
fit has smaller headline RMSE, 0.08551, despite failing whole-line coverage.
No seed or function coefficients were selected to favor a representation.

Numerical provenance is pinned to commit `f5b7781` in the dedicated worktree,
with original runtime files copied to the result folder's `source/`. Subsequent
MATH formatting and completion-document updates do not rewrite that contract.
Use `--collect` for current report replay. Reproducing numerical tasks under
the original contract requires the recorded source revision; a changed setup
requires a new result folder. All pilot/production environment and first-fit
arrays were verified bit-for-bit identical before handoff.

Validation includes 136 focused/neighboring tests, the manifest/launcher/solver
checks and all mathematical-document rendering checks. The `main.py` entry
point was also exercised with a small NumPy synthetic preset and both first-order
and Stein estimators; the default full-data GPU Generali configuration was not
launched. Its saved validation outputs are indexed by
`scratch/retired_experiments/synthetic_polynomial_demo_20261103.json`.
