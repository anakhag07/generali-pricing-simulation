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
statistic c. At a query, g=phi' beta and w=c ||R^{-T}phi||. This is the existing
coefficient-ellipsoid construction in MATH section 7.18, generalized to a fixed
feature matrix. Cauchy-Schwarz gives a simultaneous bound over every feature
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
