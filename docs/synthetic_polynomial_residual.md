# Fixed quartic residual versus additive polynomial features

This is the `polynomial_residual` mode of `wild_bootstrap_envelopes`, using
`manifests/synthetic-polynomial-residual-degree4.json` through the existing
`scripts/run_experiment_manifest.py` launcher. The approved first stage uses
one fixed residual and six feature degrees, not a residual-degree sweep.

## Implementation checklist

- [ ] Reuse frozen-feature QR/bootstrap mathematics; add exact polynomial queries,
  Gaussian moments, asymmetric containment, and saved-state replay.
- [ ] Extend the existing family dispatch without changing historical manifests.
- [ ] Freeze one Gaussian design and one exact squared-exponential GP restriction.
  Separate context, action, profile, observation, method, and optimizer streams;
  pair observations/multipliers/optimizer perturbations across feature degrees.
- [ ] Check noiseless recovery, independent bootstrap refits, all-real tail and
  interior certificates, moments, query parity, and repository optimizer use.
- [ ] Pilot the headline repetition, freeze production optimizer settings, run
  100 coverage repetitions, then collect exactly four vector PDFs and one table.
- [ ] Update MATH, README, AGENTS, exports, and run relevant tests and a demo.
- [ ] Preserve source/input hashes and saved solver traces; reporting reads saved
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
