# Interaction and nonlinear synthetic residual experiments

## Implementation checklist

- [x] Extend additive polynomial queries with original-action interaction columns.
- [x] Preserve the existing family, launcher, frozen GP design and paired streams.
- [x] Keep the trained MLP and its original head fixed during calibration.
- [x] Fit polynomial nonlinear controls on the same first 150 rows.
- [x] Bootstrap head perturbations on the other 150; never fit observed calibration targets.
- [x] Retain whole-line interaction coverage and label nonlinear finite-grid coverage.
- [x] Use repository solvers for MLP fitting and every reported policy endpoint.
- [x] Validate independent bootstrap refits, gradients, leakage, replay and domain labels.
- [x] Produce the five established PDF reports, including context slices and a single-panel figure 05.
- [x] Preserve historical experiments and archive useful pilot diagnostics with checksums.

## Resolved contract

Keep the original contextual quadratic, n=300, exact squared-exponential GP
restriction, observation/multiplier/optimizer seed roots, B=1999, S=100 and
asymmetric transfer. The new manifests remain in `polynomial_residual` mode.
The interaction residual adds 0.35*x1*a + 0.20*x2*a^2 to g4; compare action
degree six with interaction orders 0, 1 and 2, using all 300 rows for OLS.

The nonlinear residual is 0.2+0.1*x1+0.25*a+0.35*x1*a plus
0.8*tanh(x2)*exp(-(a-0.5)^2/(2*0.4^2)). Compare the no-interaction and
order-two polynomial bases and a two-hidden-layer, width-32 SiLU MLP.
Rows 0:150 fit each point predictor. Rows 150:300 calibrate head perturbations
and define the evaluation contexts. They are calibration data, not an independent
test set. MLP input standardization uses only training rows. Full-batch repository
L-BFGS-B fits all network weights once per repetition, with fixed initialization,
L2 weight penalty 0.001, a 1,000-step cap, gradient tolerance 1e-6 and ftol=1e-12.
These are declared first-version settings, not tuned to true residual scores.
The dedicated initialization seed is 20261104, held fixed across repetitions.
The entire training network is fitted anew for each independent observation-noise
repetition. Within that repetition, the 1,999 bootstrap draws change only copies
of the output head. Hidden weights and the original prediction remain fixed.

For the frozen feature design P on calibration rows, retain the training head b0
and errors e=r-P*b0. Bootstrap perturbations are db=(P'P)^(-1)P'(signs*e).
Bootstrap head copies are b0+db; b0 is never replaced by a fit to the observed
calibration responses or an average bootstrap head. Use the existing QR
coefficient-ellipsoid quantile and asymmetric transfer. This is empirical
calibration around a frozen predictor, not an asserted OLS confidence theorem.
Record rank/conditioning and training/calibration errors; fail explicitly on
rank deficiency rather than silently applying a pseudoinverse or ridge.

Nonlinear coverage checks 150 contexts times 401 action points on
[0.5-3*sqrt(0.45),0.5+3*sqrt(0.45)], then a nested 1601-point grid with the same
band. Neither is a continuous-domain certificate. True-residual R2/RMSE and
average width integrate over the full Gaussian action distribution. Policy
optimization is continuous and unconstrained; no grid chooses an action.
Report direct J(0.5)-J(converged actions), repository convergence and any actions
outside the coverage window. Only polynomial interaction objectives receive
the existing exact polynomial global-gap check.

## Module responsibilities

`polynomial.py` owns interaction features, polynomial fibers and certificates.
`mlp_head.py` owns SiLU fitting, frozen predictions and calibration-only head
bootstrap. `structured_residual.py` owns the explicit nonlinear function,
model specifications, saved-state restoration and finite-grid evaluation.
The existing `polynomial_study.py` continues to own manifests, immutable source
contracts, repetitions and repository policies. `structured_reporting.py`
replays the new models through the established five-figure reporting pattern.
Existing manifests keep their original behavior and saved key conventions.

## Running and outputs

Activate `simulation_env`, then run the existing launcher from this worktree:

```bash
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-interactions.json --launch local
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-nonlinear.json --launch local
```

Each manifest defaults to 100 independent noise repetitions and 1,999 bootstrap
draws. The launcher supports the existing `--task-index`, `--sweep-id`, and
`--collect` flags for task distribution and saved-state reporting. Prepare the
environment once before distributing local tasks. Collection never fits or
optimizes. Runtime sources and input hashes must match the saved contract.

The shared results root contains one current folder per experiment:
`synthetic-polynomial-residual-interactions/` and
`synthetic-polynomial-residual-nonlinear/`. Each contains:

| Artifact | Contents |
| --- | --- |
| `01_residual_recovery.pdf` | True and fitted residuals at three actual contexts; full-Gaussian R²/RMSE; unsuccessful point-fit status |
| `02_simultaneous_coverage.pdf` | Coverage across repetitions with Wilson intervals; nonlinear grid size, action limits and refinement counts |
| `03_average_envelope_width.pdf` | Mean full asymmetric width against action and its full-Gaussian weighted average |
| `04_policy_regret.pdf` | Direct true-objective difference at repository policy endpoints, with RMS action error from 0.5 |
| `05_truth_surrogate_and_observations.pdf` | Single-panel true function, fixed surrogate and saved context-aligned noisy observations |
| `summary.csv`, `summary.json` | Accuracy, separate radii, residual magnitude, coverage, solver diagnostics and provenance |
| `inputs.npz`, `repetitions/`, `fits/`, `policies/` | Frozen design/GP, paired samples, trained networks, calibration/head perturbations and repository solver endpoints |

The residual recovery figure intentionally shows curves without the uncertainty
fill: the large tail bands obscured residual shape in the pilot. Band widths are
reported separately in figure 03 and in the numerical tables. The three context
slices are selected by input coordinates alone, not by model accuracy.

Calibration RMSE is error against noisy held-out residual observations. The R²
and residual RMSE compare predictions against the known noiseless function over
the full Gaussian action distribution. Calibration observations also determine
the bands, so their fit score is not an independent final test score.

Both production studies completed 100/100 repetitions. Interaction coverage
for q=0,1,2 is 0/100, 50/100 and 100/100. Nonlinear coverage for q=0, q=2 and
MLP is 2/100, 98/100 and 98/100, unchanged on the refined grid. The MLP reaches
the training iteration cap in every repetition; the headline run has R²=-6.643,
training/calibration noisy-target RMSE 0.160/1.355 and mean full width 10.539.
These are the declared first-version settings, retained without tuning after
observing the results. All five PDFs per study were opened, checked for vector
content and visually inspected. Each production folder includes `validation.json`.

Policy solver convergence and a polynomial global-gap certificate are distinct
diagnostics. The interaction collector reports how many returned actions pass
the 1e-4 gap certificate; it does not silently replace uncertified endpoints.
Nonlinear global gaps remain unknown. High simultaneous coverage can reflect
wide bands, even for a misspecified or poorly generalizing model.

## Validation and retained diagnostics

The focused mathematical, split/replay, manifest and launcher suites pass 136 distinct tests. A small `main.py` synthetic quadratic demo also passes; the default real-data GPU experiment was not launched. Public API pages were rebuilt with pdoc and their local links checked.

The one-repetition pilots and demo artifacts are retained in shared `scratch-results/synthetic-residual-extensions-validation/`, indexed with SHA-256 checksums by `scratch/retired_experiments/synthetic_residual_extensions_validation_20261103.json`. The original polynomial results are protected by `scratch/synthetic_residual_extensions_inheritance.json` (1,896 artifacts). The production nonlinear folder also records exact split, original-head and paired-draw checks in `split_and_pairing_validation.json`.
