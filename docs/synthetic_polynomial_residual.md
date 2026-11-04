# Explicit polynomial residuals versus additive polynomial features

This is the `polynomial_residual` mode of `wild_bootstrap_envelopes`, using
`manifests/synthetic-polynomial-residual-degree4.json` through the existing
`scripts/run_experiment_manifest.py` launcher. The approved first stage uses
one fixed quartic residual and six feature degrees. The approved continuation
adds the original writeup's g1, g2 and g6 using the same settings and draws.

## Full polynomial continuation

Use `manifests/synthetic-polynomial-residual-degree{d}.json` for d=1,2,6;
the existing d=4 run is retained without rerunning or altering its artifacts.
The residual action coefficients, in increasing powers of the original action,
are `[0.2,0.25]`, `[0.2,0.25,0.15]`, and
`[0.2,0.25,0.15,-0.08,0.04,-0.012,0.006]`, respectively. All retain
the additive context term 0.1*x1. No centering or interactions are added.

Each new manifest runs all six feature bases, n=300, S=100, B=1999 and
nominal 95% coefficient-ellipsoid calibration with asymmetric transfer.
Design, GP, observation, multiplier and optimizer seed roots are unchanged.
This pairs each new experiment's inputs and noise with the retained quartic
run. Coverage remains simultaneous over the saved contexts and the whole real
action line. Regret remains direct J(theta_star)-J(theta_hat), using repository
solver endpoints and the original optimizer settings.

Each `synthetic-polynomial-residual-degree{d}/` folder contains the same five
PDFs and saved-state artifacts described below; figure 05 has only the requested
truth/surrogate/observation panel. The old quartic source worktree stays pinned
at 615c053; the new worktree is `feature-synthetic-polynomial-sweep`.
`scratch/synthetic_polynomial_sweep_inheritance.json` records retained inputs,
source identity and original artifact checksums. The runtime mathematics and
optimizer are unchanged; residual-recovery titles now reflect each degree.

For every degree, execute the corresponding manifest with the existing launcher:

```bash
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-degree1.json --launch local
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-degree2.json --launch local
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-degree6.json --launch local
```

Tasks can be distributed with `--task-index` and a shared `--sweep-id`; prepare
the environment once before parallel local tasks, then use `--collect` after
all 100 tasks complete. Execution logs retain the exact commands and statuses.
Representability starts at k=d. It removes approximation error, but does not
make noisy fitted coefficients exact or make empirical coverage identically one.

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

## Shared scientific contract

See MATH section 7.22. The true contextual quadratic is unchanged. Each fixed
surrogate subtracts its explicit residual g_d; the initial run used g4. No predictor training or
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

The shared result folder is `synthetic-polynomial-residual-degree4/`.
It contains four comparison PDFs: `01_residual_recovery.pdf` (six panels, R2/RMSE),
`02_simultaneous_coverage.pdf` (six bars and Wilson intervals),
`03_average_envelope_width.pdf` (mean full transferred width versus action),
and `04_policy_regret.pdf` (regret and RMS action error).
Regret is calculated directly as `objective_at_theta_star - objective_at_theta_hat`,
where each objective value is the mean true function over all saved contexts.
Both values are included in the summary CSV/JSON. For this quadratic the result
equals half the mean squared action error; that identity is only a validation
check. RMS action error remains a secondary diagnostic in the figure.
Frozen inputs, coefficients, bootstrap statistics, per-repetition decisions,
solver traces, plotted coordinates, summary CSV/JSON and provenance remain
alongside the figures. Report collection never fits or optimizes.

The requested supplementary `05_truth_surrogate_and_observations.pdf` shows
the true function and fixed surrogate at the mean calibration context, with
the original observations aligned by subtracting their known additive context
offsets. This single-panel view colors observations by their saved noise SD.
These transformations retain
all 300 observations and their original heteroskedastic noise; they do not
simulate new responses at the mean context or force the sample noise mean to
zero. `observation_data.csv`, `observation_curves.csv`, and
`observation_report.json` retain plotted coordinates, source hashes and checks.

R2 integrates true-residual error over every real action with Gaussian weights,
using the same 300 contexts. It is not computed only at the logged actions or
on the displayed mean curve. For phi_1, MSE is 0.0205538 and residual variance
is 0.0852472, giving R2=0.758892. The 0.27% action probability outside the
three-SD display window contributes 26.1% of its squared error. Whole-line
coverage detects tail failure, but must be read alongside band width: wide
bands can cover despite poor point estimates. The guarantee domain concerns
the saved contexts, not extrapolation to new contexts.

## Commands

```bash
conda activate simulation_env
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-degree4.json --launch local
python scripts/run_experiment_manifest.py manifests/synthetic-polynomial-residual-degree4.json --launch local --collect
```

Local/Slurm launch uses the existing infrastructure. Each task owns a group of
coverage repetitions; task zero also owns the headline policies. Complete
artifacts are immutable and resumable only with matching sources and inputs.

For a regret-only report replay from the existing numerical artifacts:

```python
from experiments.envelopes.polynomial_reporting import update_regret_report
update_regret_report("/home/anakhag/projects/generali-pricing/results/synthetic-polynomial-residual-degree4")
```

This updates only the regret PDF and summary tables. The original regret report
is archived through `scratch/retired_experiments/synthetic_polynomial_regret_report_20261103.json`.

To replay only the supplementary observation figure and its coordinate files:

```python
from experiments.envelopes.polynomial_reporting import write_observation_report
write_observation_report("/home/anakhag/projects/generali-pricing/results/synthetic-polynomial-residual-degree4")
```

Full `--collect` now includes this fifth PDF. Prior report metadata and the
previous reporting source are indexed by
`scratch/retired_experiments/synthetic_polynomial_observation_addition_20261103.json`.
The former two-panel observation PDF and its source/metadata are archived via
`scratch/retired_experiments/synthetic_polynomial_observation_top_only_20261103.json`.

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

## Completed full polynomial continuation

The three additional runs completed 100 repetitions each. The table reports
transferred whole-line coverage counts out of 100; the g4 row is the retained
original result.

| Residual | phi1 | phi2 | phi3 | phi4 | phi5 | phi6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| g1 | 98 | 98 | 97 | 98 | 100 | 100 |
| g2 | 0 | 98 | 97 | 98 | 100 | 100 |
| g4 | 0 | 0 | 0 | 98 | 100 | 100 |
| g6 | 0 | 0 | 0 | 0 | 0 | 100 |

Every misspecified degree fails on the full real line. Representable cases
cover 97–100 repetitions, rather than automatically achieving perfect coverage.
For g6, phi4 has weighted R2=0.91374 and zero whole-line coverage; phi6 has
R2=0.89935 and coverage 100/100. Coverage and Gaussian-weighted approximation
therefore answer different questions. Read coverage with width: higher-degree
bands expand strongly in the tails. Counts of 100/100 are empirical outcomes,
not a finite-sample guarantee. The separate coefficient and auxiliary coverage
events remain in the summary tables.

All 54 new Stein solver starts converged, and all 5,400 selected actions passed
the unsmoothed global LCB gap check at tolerance 1e-4. Direct true-objective
regret was checked independently against the quadratic action-error identity.
All 100 observation, innovation and multiplier arrays per residual exactly match
the corresponding original quartic arrays, as do all frozen environment arrays.
The represented coefficient errors match the paired quartic errors within
1.3e-15. All 426 original quartic artifact hashes remain unchanged.

Numerical source contracts for the continuation are pinned at `d82744e`;
the 18 numerical runtime source files match the quartic run. Validation includes
15 polynomial/bootstrap/solver-interface and math-document checks, independent
feature-span identities, successful task statuses, all saved source/data hashes,
and visual/vector checks of all 15 new PDFs. Each result folder's `validation.json`
records these checks. Preparation-only contracts preceding a documentation
rendering correction are archived via
`scratch/retired_experiments/synthetic_polynomial_sweep_preflight_20261103.json`.

## Metric comparison plots

The shared `synthetic-polynomial-residual-degree4/metric-comparison/` folder
contains one three-panel vector PDF each for degree 4, interactions and the
nonlinear residual, named `*_r_squared_radius_regret_bars.pdf`. The aligned
horizontal bars show R² (higher desired), mean radius (lower desired) and regret
(lower desired), with the same model rows/colors and numerical value labels
across columns. Coverage percentages from 100 repetitions appear beside model
names so narrow, unreliable bands are identifiable. Each metric retains its own
units and axis; there is no normalized composite score.
`metric_comparison.csv` contains all 12 rows and the added coverage fields.

All three coordinates use the saved first noise realization. R² is measured
against the known noiseless residual over the full Gaussian action distribution.
Mean radius is `(mean_lower_radius + mean_upper_radius)/2`, half the transferred
envelope's full width; the two asymmetric radii remain separate columns in the
CSV. Regret is the existing direct true-objective difference at saved repository
optimizer actions. There is no refitting, resampling, integration or optimization.
The quartic/interaction figures evaluate 300 contexts; nonlinear evaluates its
150 held-out contexts, so comparisons are primarily within each experiment.

The nonlinear figure includes both polynomial controls and only the 2×2 MLP,
as requested. It has the best mean held-out noisy R² across the 100 capacity
repetitions (0.2369); this model-selection score differs from its plotted
headline full-Gaussian noiseless R² (0.6116). The calibration half was reused for
comparison and bootstrap, not an untouched test set. All displayed policy actions
come from converged repository solves. Prediction/radius/regret plots alone do
not establish coverage; the selected MLP's simultaneous grid coverage is 0/100.

From the `feature-synthetic-residual-tradeoffs` worktree, replay the saved request:

```bash
conda activate simulation_env
PYTHONPATH=src python - <<'PY'
from experiments.paths import results_root
from experiments.provenance import read_json
from experiments.envelopes.polynomial_reporting import write_metric_comparison
folder = results_root() / 'synthetic-polynomial-residual-degree4/metric-comparison'
write_metric_comparison(**read_json(folder / 'request.json'))
PY
```

`provenance.json` records input summaries, contracts, selected policy arrays,
source snapshots, output checksums and metric definitions. A changed model
selection or rendering source requires archiving the previous report first.
Validation checked all 12 plotted rows against their original summaries and 75
input records against saved hashes; all three PDFs were visually inspected and
contain vector graphics only. The 21 polynomial/structured-residual tests pass.
The superseded scatter presentation, original request, source and provenance
are retained verbatim under shared
`scratch-results/synthetic-residual-metric-comparison/scatter-summary/`.
Its old/new paths and hashes are indexed by
`scratch/retired_experiments/synthetic_residual_metric_scatter_20261103.json`.
