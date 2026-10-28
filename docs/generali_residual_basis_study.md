# Generali residual-basis comparison

This extends `glm-spline-profit-dispersion`; the original affine/full-data run
remains an historical reference. The user approved varying only the residual
approximation, retaining the saved 20,000 policy rows, a customer-disjoint
40/40/20 split, and common coefficient-ellipsoid wild calibration.

## Implementation and execution plan

- Extend the existing `scratch/plot_generali_profit_residuals.py` entry point;
  use a JSON study configuration and a study-specific companion adapter.
- Put reusable frozen-design coefficient-ellipsoid mathematics in
  `experiments.envelopes.feature_wild`. Reuse repository solvers, responses,
  policy preprocessing, signed transfer, launch and provenance helpers.
- Preserve the saved 20% residual-validation customer IDs. Split the existing
  80% by customer ID into representation-learning and coefficient-fitting halves.
  New named split and MLP-initialization streams are independent of the saved
  wild-multiplier stream. No randomized mini-batching is needed.
- Compare nested total-degree 1/2/3 polynomials in the 11 saved numeric contexts
  and action (13/91/455 terms), and a tanh MLP with two 16-unit hidden layers.
  Train the MLP with a temporary linear regression head on the first 40%, freeze
  its hidden layers, discard that head and refit an intercept plus 16 output
  features on the second 40%. Train one representation per frozen predictor.
- Fit all four beta models on exactly the same second 40%; evaluate residual
  error on the original 20% without refitting on validation responses. Existing
  predictors and historical preprocessing remain frozen; their original training
  membership is unknown, so this is conditional residual validation only.
- Use the same 1,999 saved draw seeds/Rademacher signs, restricted to beta rows,
  for every basis and predictor. Keep nominal delta=.05. All variants, including
  the new affine control, use the common calibration below.
- Optimize personalized lower endpoints on the unchanged saved 20k rows using
  the repository first-order trust-constr solver, existing 19 policy inputs,
  softmax-linear policy, [-.1,.2] action bounds, .08 initialization and recorded
  acceptance floor. Replay the saved nominal-policy endpoints. No grid selects
  an action. Report local solver status; global optimality is not certified.
- Evaluate personalized actions on all 715,023 rows and report f-hat, corrected
  profit, half-width, both transfer penalties/endpoints, acceptance and action
  changes. Score both each variant's own policy and the same nominal policy to
  separate band changes from policy adaptation. Also report holdout residual
  RMSE/MAE/R2, leverage, conditioning and parameter counts.
- Benchmark fit, bootstrap batch, and policy iterations independently. Save
  accepted optimizer iterates and completed bootstrap batches atomically.
  Resume with explicit solver restart metadata; never call an interrupted or
  iteration-limited solution converged. Stop before scheduler walltime and keep
  checkpoints. Fail visibly on rank deficiency/nonfinite values; no implicit
  ridge, dropped terms, altered calibration or reduced final bootstrap count.
- Validate independent least-squares bootstrap parity, coefficient-ball bounds,
  feature/action derivatives, polynomial-cache parity, customer disjointness,
  exact full-row coverage, checkpoints and source/input hashes. Run focused
  tests and the repository demo; regenerate pdoc for the public addition.

## Mathematical and representation contract

Condition on each frozen feature map. With beta-training matrix P=QR,
beta=R^{-1}Q' r and residual e=r-P beta, use bootstrap errors
d_b=R^{-1}Q'(e*xi_b). Calibrate T_b=||R d_b|| and its ceil(.95 B)-th order
statistic c. At a query, g=phi' beta and w=c ||R^{-T}phi||. MATH section 7.20
derives the frozen-feature coefficient ellipsoid, its signed residual transfer,
and exact action derivatives. Cauchy-Schwarz gives a simultaneous bound over every feature
vector conditional on the coefficient-ball event. It is not the original
affine action-supremum calibration and can be more conservative.

The mathematical query domain is the saved customer contexts crossed with all
real actions; actual policy decisions stay in [-.1,.2]. Polynomial and frozen
tanh evaluations are exact off-grid formulas. Shared bootstrap beta errors
couple all queries; conditional covariance is phi(z)' V P' diag(e^2) P V phi(z'),
V=(P'P)^{-1}. Plotted action samples are visualization only. There is no known
conditional-mean truth or empirical coverage estimate. The coefficient-ball
implication does not establish bootstrap validity or remove approximation bias;
inference is conditional on the learned representation. Existing high-leverage,
raw-residual and within-customer dependence limitations remain explicit.

Transfer before averaging: lower=f-hat-max(w-g,0),
upper=f-hat+max(w+g,0). The corrected prediction f-hat+g is reported separately.
No numerical global-reference optimization or certification is introduced.

## Runtime contingencies

The prior 20k solves took about 1 second (GLM) and 35 seconds (spline); spline
exhausted 1,000 iterations. These are historical measurements, not runtime
promises. At the new beta sample size the cubic design alone is about 0.97 GiB,
before QR and workspace. Batch full-population queries and bootstrap signs;
cache exact polynomial action coefficients for repeated policy evaluations.
MLP training occurs once per predictor, not once per bootstrap draw. A bounded
initial solve and recorded continuation budgets distinguish slow convergence
from completion. Slurm tasks must preserve useful diagnostics on interruption.

Execution uses the same experiment family and one current
`residual-basis-study` deliverables folder; named basis variants are scientific
comparisons, not redundant attempts.

```bash
conda activate simulation_env
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
python scratch/plot_generali_profit_residuals.py \
  --fit-wild-bases scratch/generali_residual_wild_basis_study.json \
  --replay /home/anakhag/projects/generali-pricing/results/glm-spline-profit-dispersion/residuals-20261026 \
  --output-dir /home/anakhag/projects/generali-pricing/results/glm-spline-profit-dispersion/residual-basis-study \
  --launch slurm --array --array-max-parallel 2
```

Use `--launch local --task-index 0` for one task, or `--launch local --collect`
to collect existing summaries. Task order is basis-major and then GLM/spline.
Repeat an interrupted task with exactly the same config and sources to resume.
An accepted-iterate restart resets the solver's internal memory and is recorded.
Wall-time stops save the solver checkpoint and `*.interrupted.json` diagnostics,
then stop the task before producing a completed fit, policy, or task summary.
Rerunning the same task resumes from that checkpoint for both MLP learning and
policy optimization. A legacy solver JSON with `status: wall_time_limit` is
reclassified as interruption diagnostics before resuming; completed historical
study artifacts and their original source contracts are not rewritten.
The repository callback recorder reevaluates the accepted point before step
logging, so first-order checkpoint theta agrees with the callback even after a
rejected trial evaluation. A regression test covers both supported solvers.
Completed bounded solves, including unconverged ones, are replayed; additional
budgets require an explicit revision, not overwriting their original outcomes.

## Completed 2026-10-27 run

All eight tasks completed under the stated configuration. The realized row
counts are 286,021 representation / 286,001 beta / 143,001 validation. Each
task retains 1,999 paired multiplier draws and all 715,023 evaluation rows.
All policy solves met their convergence criterion. Both neural representation
fits reached the 1,000-iteration cap; they are budget-limited representations,
not claimed fully converged neural fits. End-to-end task times ranged from
19.5 to 214.8 seconds, excluding queue time.

Current results are in `glm-spline-profit-dispersion/residual-basis-study/`
under the shared results root: `policy_metrics.csv` separates personalized
full-row and optimization-cohort scores, `residual_fit_metrics.csv` reports
the three residual splits, and `solver_and_geometry.csv` records convergence,
conditioning and leverage. `validation.json` records the source/output hashes,
split/draw/action checks and resolved new seed streams. Sixty-five focused
checks and the small repository demo passed; API documentation was regenerated.

The affine controls have the highest full-population lower-endpoint means in
this run. Quadratic/cubic extrapolation is unstable on extreme held-out
customers. The frozen MLP improves holdout RMSE (GLM 113.15 to 105.64;
spline 95.76 to 95.20), but its wider bands reduce the optimized mean lower
scores. Do not interpret residual-corrected extrapolations as observed profits.

Completed intermediate bootstrap checkpoints were archived, without changing
their bytes or the completed band/policy/network artifacts. Their relocation
index is `scratch/retired_experiments/generali_residual_basis_checkpoints_20261027.json`;
the results folder contains a copy as `checkpoint_relocations.json`.

## Follow-up audit: why the wider bases behave differently

Replay of the saved cubic policies identifies source rows 356127, 276076 and
521036 as the three largest absolute GLM residual corrections. Each has
`X_vehicle_age = -7975` in the unchanged source dataset: an implausible age,
not simply a typical high-risk customer. None belongs to the beta-fitting or
20k policy cohort. Their cubic GLM corrections are approximately 1.324 billion
each and their half-widths approximately 3.76 billion. The median full-cohort
correction is only 1.1393. The ten largest absolute corrections contribute
99.957% of the signed correction sum and 99.136% of the half-width sum.
Thus the huge averages are dominated by polynomial extrapolation on anomalous
inputs. This audit does not change the data, actions, or completed results.
Resolve the age encoding before interpreting this as an inherent disadvantage
of polynomial features; removing anomalous rows post hoc is not the same study.

The MLP result is a separate issue. At the identical nominal policy actions,
GLM's mean half-width increases from 8.9353 (affine) to 15.8690 (MLP): the
bootstrap critical value rises from 1435.69 to 2163.90, while mean query
geometry increases from 0.00622370 to 0.00733350. Its beta-training mean squared
residual decreases from 11842.49 to 10077.18, but the expected squared bootstrap
coefficient-ball statistic, sum_i leverage_i * residual_i^2, rises from
812859.39 to 1830682.58. Bootstrap empirical means agree (818935.77 and
1843514.66). Ordinary prediction error and joint coefficient uncertainty are
different quantities. The learned features change which errors influence each
coefficient, and the conservative joint calibration propagates this uncertainty
to all query directions.

The reported validation RMSE is error against noisy observed residuals at logged
actions, not error against a known true beta or conditional-mean function. It
does not measure uniform accuracy at candidate policy actions. These bands do
not automatically include residual-model approximation bias, so a narrower
affine band and a higher affine lower score do not establish higher true profit.


## Ten-record exclusion sensitivity analysis (2026-10-27)

The user requested removal of all ten records flagged by the original cubic
GLM audit, then refitting and reoptimizing the richer residual bases. The new
`residual-basis-study-trimmed` revision includes a matched affine control.
Its configuration is `scratch/generali_residual_wild_basis_study_trimmed.json`,
through the same `--fit-wild-bases` entry point. Results use the same family at
`glm-spline-profit-dispersion/residual-basis-study-trimmed/`.

`scratch/generali_residual_basis_flagged_rows.json` fixes the ten source row
IDs, their original positions, split memberships, saved corrections, and the
SHA-256 records for the original cubic GLM band and repository-optimized policy.
The selection is the ten largest absolute fitted corrections at that saved
policy's personalized actions; the same ten exclusions apply to both response
models and every residual basis. This is a post-selection sensitivity analysis,
not an independently validated outlier rule or untouched holdout evaluation.
Only three of these rows have the established invalid vehicle age -7975;
we do not assume the other seven are erroneous observations.

Remove these rows from every residual-learning, validation and scoring stage,
without changing the raw dataset or frozen GLM/spline predictors. Six rows are
in the representation pool and four in validation. None is in the beta pool
or original policy optimization cohort: retained counts are 286,015 / 286,001 /
142,997, with 715,013 full evaluation rows and exactly 20,000 policy rows.
Replay original split memberships before filtering; never reshuffle the shorter
cohort. Generate each multiplier on the original 715,023 coordinates and then
select the retained beta coordinates. This preserves the original draw hashes.
Initialization, budgets, solvers, acceptance constraint, action domain, exact
query rules and coefficient-ellipsoid calibration remain as specified above.
No new stochastic process or seed stream is introduced.

Refit all residual models and bootstrap calibrations and rerun each policy
through the repository solver. Retrain both MLP representations on their
retained first pools, then freeze them for beta fitting and all bootstrap draws.
The architecture is 12 standardized inputs (11 numeric customer covariates
plus action), 16 tanh units, 16 tanh units, then a temporary linear scalar head:
497 trainable parameters. Discard that temporary head; the final linear beta
fit uses the 16 frozen last-layer outputs plus an intercept (17 coefficients).
Each predictor gets its own network trained on that predictor's profit residuals.

The original worktree and completed artifacts are preserved because historical
contracts include absolute source paths and hashes. The filtering extension
lives in `worktrees/feature-generali-residual-bases-trimmed`. Run the command above
with the `_trimmed.json` configuration and `residual-basis-study-trimmed` output.


### Completed exclusion revision

All eight refits and policy solves completed. The Slurm controller temporarily
failed status requests, but the remaining outputs and the 8/8 collector arrived;
subsequent queue inspection found neither study job live. Array 27050829 and
collector 27050830 used the configuration above. Task compute times were
21.2–174.4 seconds, excluding queue delays. All policy solves reported success;
both neural representation fits stopped at their 1,000-iteration limit.

Every profit and band mean below is an equal-row average at that basis's own
saved repository-optimized policy actions over all 715,013 retained rows.
Validation RMSE uses the 142,997 retained original holdout rows at logged actions.

| Predictor | Basis | Validation RMSE | Mean residual correction | Mean half-width | Mean fhat profit | Mean lower profit |
|---|---|---:|---:|---:|---:|---:|
| glm | affine | 112.5265 | 1.5685 | 8.7584 | 145.4384 | 135.8551 |
| glm | quadratic | 103.6264 | 5.7735 | 57.8281 | 142.7920 | 90.0637 |
| glm | cubic | 104.3756 | 2.3715 | 144.3668 | 141.3915 | -0.6114 |
| glm | mlp | 102.7253 | 2.6253 | 14.3646 | 142.0005 | 123.3720 |
| spline_xgb | affine | 95.7368 | -9.0831 | 6.8512 | 159.3163 | 143.3821 |
| spline_xgb | quadratic | 95.7713 | -8.8464 | 48.8410 | 157.6013 | 99.9140 |
| spline_xgb | cubic | 100.1591 | -8.9979 | 132.3326 | 152.9314 | 11.6008 |
| spline_xgb | mlp | 95.0067 | -9.5772 | 14.6946 | 154.7039 | 130.4316 |

The original cubic mean corrections fall from 5542.66 to 2.37 (GLM) and 248.18
to -9.00 (spline). The corresponding lower means improve from -11019.03 to -0.61
and from -14894.89 to 11.60. Polynomial bootstrap critical values are unchanged
to relative numerical tolerance 1.2e-13: exclusions do not touch beta fitting,
and full total-degree polynomial spaces are invariant to the changed input
centering/scaling. Replaying original policies on the retained cohort agrees
with their refitted results to reporting precision. The large mean changes
therefore primarily remove extreme evaluation queries rather than change the
polynomial fitted functions. No additional observations were trimmed.

The refitted MLP has the lowest retained holdout RMSE for each predictor, while
its wider bands still yield lower bound scores than affine. Relative to each
original MLP policy evaluated on exactly the retained cohort, refitting changes
the lower mean from 124.37 to 123.37 (GLM) and 133.95 to 130.43 (spline).
Predictive RMSE and joint coefficient uncertainty are distinct; this result does
not establish realized profit rankings or eliminate approximation bias.
The ten-record rule was selected from the original cubic outputs and remains
an explicitly labeled sensitivity analysis, not a new independent holdout study.

The results directory contains `before_after_policy_metrics.csv` (original full
cohort, original policies on retained rows, and refitted policies on retained
rows), the directly evaluated excluded-row scores and replay provenance,
`residual_fit_metrics.csv`, `solver_and_geometry.csv`, and `validation.json`.
Forty-six focused tests and a small repository demo passed. The artifact audit
verified all source/output hashes, retained customer-disjoint splits, identical
20k policy rows and beta rows, original draw seeds and full-coordinate multiplier
hashes, signed envelope accounting and policy constraints.

To repeat the saved-artifact checks from this worktree without fitting:

```bash
conda activate simulation_env
PYTHONPATH=src:. python /home/anakhag/projects/generali-pricing/results/glm-spline-profit-dispersion/residual-basis-study-trimmed/validate_saved_results.py
```

Completed intermediate bootstrap checkpoints were moved unchanged to
`scratch-results/glm-spline-profit-dispersion/residual-basis-study-trimmed-checkpoints/`.
Their checksum/relocation index is
`scratch/retired_experiments/generali_residual_basis_trimmed_checkpoints_20261027.json`,
with a copy in the result directory. All final fit, network and policy artifacts
remain current, and the original untrimmed study and source worktree are intact.
