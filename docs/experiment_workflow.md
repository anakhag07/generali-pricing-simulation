# Settings, bands and optimization

The existing manifest runner now supports `task: "optimize"` (the default),
`"band"`, and `"optimize_with_band"`. There is one execution pipeline and no
natural-language parser. An agent translates a request into an existing family
manifest after resolving consequential unknowns; see
[AGENTS.md](../AGENTS.md#natural-language-experiment-requests).

Keep the family name when changing settings. Use variants, matrix axes, or a
fresh run root for a revision. Saved runs are immutable: changing data, sources,
parameters or seed pairing requires a fresh location within that family.

## Where choices live

| Choice | Source of truth |
|---|---|
| Family, variants, launch resources, reporting | Existing manifest `name`, `variants`/`matrix`, `launch`, `reporting` |
| Optimization-only function, data/model, policy | Registered `objective.preset` and `objective.overrides` |
| Band inputs | Manifest `setting`: an existing synthetic study, supplied observations, or explicit Generali rows |
| Construction and assumptions | Manifest `band`: method, calibration parameters, domain; or one saved `artifact` |
| Band optimization target | `objective.preset: "band_policy"`; overrides specify `endpoint`, `sense`, `policy_kind`, bounds |
| Solver, first-/zeroth-order estimator, budget | Existing `optimizer` fields, including `step_rule`, `enabled_estimators`, `perturbation_space` |
| Randomness | Existing optimizer `seeds`; synthetic study streams or supplied bootstrap multipliers for band fitting |
| Executed configuration and evidence | `EXPERIMENT.md`, `run-contract.json`, fitted state, input/source hashes and existing per-seed reporters |

## Source layout and interfaces

- `experiments/settings.py`: `prepare_setting(spec, base_path=...)` returns
  observations, frozen predictor, contexts, provenance and (only for synthetic
  evaluation) truth. `repetition_draws` reuses the retained study's exact stream
  derivations, full draw shapes and prefix pairing.
- `experiments/envelopes/`: `fit_band(data, predictor, target, settings,
  multipliers=...)`, `fit_support_band(...)`, `save_band(...)`, `load_band(...)`.
  A `FittedEnvelope.evaluate(X, actions)` returns `center`, `lower`, `upper`, and
  diagnostic components. Asymmetric bands have `radius=None` and separate lower
  and upper radii; never silently average them for coverage or optimization.
- `objective/banded.py`: `BandObjective` composes the selected endpoint and
  policy through the existing `Objective.value/grad` interface. Maximization
  negates the objective for repository minimizers; saved endpoints retain the
  original units and sign. Acceptance constraints and scalar penalties use the
  existing objective/optimizer machinery.
- `experiments/configs/band_policy.py`: builds `ExperimentConfig` and validates
  domain/gradient capabilities. `manifest.py` and `execution.py` own orchestration;
  `run.py`, reporters and `src/optimization/` still execute optimization.
- `envelopes/quadratic.py`, `envelopes/linear.py`, `envelopes/wild.py`,
  `offline_bootstrap_math.py`, and `synthetic.py` retain shared mathematics.
  `wild_study.py`/`wild_reporting.py` retain the current study runner and reports.
  Concluded orchestration is in `scratch/retired_experiments/studies/`.

## Three flows

**Optimization only:** use the existing manifest unchanged. Its preset prepares
inputs and objective; the requested repository optimizer returns the policy;
existing reporters save theta, traces, metrics and seed metadata. To suppress
truth-derived metrics explicitly, use `truth: {"source": "none"}`.

**Band only:** set `task: "band"`, `setting`, `band`, `name`, and `launch`.
Omit optimizer and optimizer seeds. The setting supplies observations and any
bootstrap draws; fitting never receives synthetic truth. Saved state can be
loaded and evaluated without fitting, sampling or optimizing.

**Optimize with a band:** use the same setting/band and add
`objective.preset: "band_policy"`, explicit target/policy overrides,
`objective_modifications`, optimizer, and seeds. Fitting occurs once per variant;
optimizer repetitions share that exact fitted band. To vary calibration draws,
use a setting variant with a different explicit repetition. Band-based runs use
`truth.source: "none"`; known-truth coverage stays in the retained study runner.

This example selects an existing residual family setup. Values are illustrative
choices, not new scientific defaults; the source manifest still defines the
predictor, noise law, seeds and paired schedule:

```json
{
  "name": "uncertainty-envelopes-residual-paired-nb",
  "task": "band",
  "setting": {
    "source": "wild_study",
    "manifest": "uncertainty-envelopes-residual-paired-nb.json",
    "n": 30,
    "repetition": 0
  },
  "band": {"method": "residual_wild", "B": 99, "delta": 0.05},
  "launch": {"mode": "local", "array": "variant"}
}
```

Choose `n` from the referenced setup's `n_values` and `B` from the available
paired multiplier prefix. Relative input paths resolve beside the requesting
manifest. For band optimization, additionally supply (with task changed):

```json
{
  "objective": {
    "preset": "band_policy",
    "overrides": {
      "endpoint": "lower", "sense": "maximize",
      "policy_kind": "softmax", "softmax_action_bounds": [-1, 1]
    }
  },
  "objective_modifications": [],
  "optimizer": {
    "step_rule": "l-bfgs-b", "enabled_estimators": ["first_order"],
    "perturbation_space": "theta", "t_steps": 100
  },
  "seeds": {"run_seeds": [7], "anchor_seed": 7, "vary": ["theta"]}
}
```

Launch through `python scripts/run_experiment_manifest.py <manifest.json>
--runs-root <fresh-root>`. Existing local/Slurm array behavior remains available.

## Settings and method boundaries

`setting.source: "wild_study"` references the retained, validated study manifest
and an explicit `n` and `repetition`. It reproduces that design and frozen
predictor, including fixed moved-row noise. No new random process is introduced.
The actual setup is saved as `resolved_source` alongside its hash and seed roots.

`setting.source: "observations"` reads an explicit NPZ with `X`, `actions`, `y`
and optional Rademacher `multipliers`. Supply `path`, `quantity`, a `provenance`
description and frozen `predictor: {"type": "linear"|"quadratic",
"coefficients": [...]}`. Quadratic means the established action-only quadratic
basis. The caller must justify the response and calibration assumptions; an
arbitrary trained Generali model is not treated as truth or replaced by a
polynomial automatically.

`setting.source: "generali_support"` requires `model`, explicit `row_indices`,
`quantity: "profit"|"cost"|"acceptance"`, `action_grid`, `n_neighbors`, and
`action_bandwidth`. It retains source row IDs, dataset/model hashes, artifact
preprocessing and exact-spline settings where selected. No cohort is sampled.
Use `band: {"method": "support_grid", "construction": "normalized_local",
"scale": ..., "quantity": ...}` with the same quantity. This is a descriptive
support penalty, with **no probability guarantee**. The direct Python adapter
also accepts the existing `absolute_action` width construction.

Polynomial methods are `ols_transfer`, `residual_ols`, and `residual_wild`.
All require `delta`; OLS requires `sigma_bound`; manifest wild fitting requires
`B` and supplied/paired multipliers. The known-variance OLS bound and approximate
wild calibration remain distinct. Supported context domains are the calibration
contexts crossed with all real actions. Support bands use saved contexts and
the bounded action interval; local curves interpolate linearly, absolute-action
widths use the established natural cubic. Duplicate ambiguous support contexts,
unknown rows, extrapolation, unsupported methods and JAX adapters fail explicitly.

First-order evaluation uses exact represented derivatives, with zero selected
at absolute-value/positive-part kinks; this is not a smoothness claim. Zeroth-order
methods query the same band. Support optimization requires an explicitly bounded
softmax policy and theta-space perturbations to keep every query in-domain.
Automatic `constant` estimator copies and legacy `per_seed_plots` are unsupported
for band tasks; select a constant policy with the requested gradient estimator
explicitly, and use saved endpoint arrays or aggregate PDFs.
Ask about solver, bounds, confidence domain, quantity/cohort or budget whenever
these cannot be inherited from the prompt or the identified experiment.

## Outputs and replay

Each band variant saves:

- `band.json` + checksummed `band.npz`: fitted coefficients/geometry or support
  curves, observations, calibration state, resolved setting and provenance.
- `band-evaluation.npz` + `band-summary.json`: endpoint values at logged actions
  or the supplied support grid. These points do not select an optimum.
- `band-binding.json` per-seed `band-policy-seed-*.json`, and `run-contract.json`: artifact binding, executable
  inputs, source/package identity, optimizer choices and seeds.
- For optimization: ordinary `summary-seed-*.json`, per-seed theta/traces/config,
  and `seeds/seed-*/band-policy-<estimator>.npz`, evaluating the exact returned
  policy on the saved cohort. No secondary optimizer runs during this evaluation. Interrupted endpoint exports
  resume from saved theta without rerunning optimization, including when their
  provenance marker is still present. Missing evaluation/summary exports can
  likewise be recovered from the saved fitted state. Existing recorded files
  are checked even while other seeds are unfinished; changed bytes are rejected.

Set `band: {"artifact": "path/to/band.json"}` and omit `setting` to reuse a
saved fit. `load_band(path).evaluate(X, actions)` handles standalone queries or
actions from a saved optimized policy; preserve that policy's artifact reference
in downstream reports. Evaluation does not rerun optimization. Report collection
reads saved summaries; it does not refit bands. Execution refuses a changed
contract, and generic `--force` cannot overwrite completed runs. Legacy runs
without contracts remain readable for reporting but cannot be resumed as newly
verified runs; recover their original source for exact replay.

The source contract intentionally binds every `src/**/*.py` file, including
comments and docstrings. An unrelated fix still requires a fresh output location
or the original source checkout to resume; it must not silently mix revisions
within a seed sweep. Per-process source digest caching avoids rereading unchanged
files and checks filesystem identity on every request, including additions and
removals. Input artifacts remain uncached. Execution validates the contract once
per variant call; reporting checks saved configuration identity without hashing
the current source tree.

`experiments.band_workflow` owns band fit/load, output integrity and saved-policy
endpoint replay. `experiments.execution` owns the generic run lifecycle and the
public `seed_reporter_stack_factory` shared with ordinary sweeps. Band and
predictor configurations share `configs.real_data_factory.make_policy`; JSON
conversion and cached source digests live in `experiments.provenance`.

## Generali residual diagnostics

The follow-up [residual-basis study](generali_residual_basis_study.md) varies the
residual representation alone with a customer-disjoint 40/40/20 split and common
coefficient-ellipsoid wild calibration. Its `--fit-wild-bases` mode retains this
family and the saved policy cohort, and supports resumable Slurm tasks. The
`generali_residual_wild_basis_study_trimmed.json` revision removes a fixed ten-row
selection while preserving split memberships and original multiplier coordinates;
its selection provenance and original study remain available.

This descriptive report precedes choosing a residual basis or calibrating a
band. It extends the GLM/spline comparison family without running an optimizer
or a band task. The existing manifest report recipe evaluates action-grid profit
dispersion, not observed residuals; this narrow diagnostic uses the shared data,
model, cache and provenance helpers through a scratch entry point:

```bash
conda activate simulation_env
python scratch/plot_generali_profit_residuals.py \
  --output-dir ../results/glm-spline-profit-dispersion/residuals-20261026
```

All common eligible source rows are evaluated at historical `a = U`; no new seed
is needed. Actual profit is `(1-is_churn)*(X_upcoming_premium-Y_G_Loss)` and
predicted profit is `acceptance(X,a)*((1+a)*X_policy_premium-predicted_loss(X))`.
The report checks the premium identity and writes exclusion counts, row-level
inputs/predictions/residuals, model/source hashes, summary statistics, and two
vector PDFs with common axes and no clipping or smoothing. See MATH section 6.

GLM uses the canonical logistic/Ridge pair. Spline/XGBoost replays the full
715,023-customer cache with its original constant-left/clipped-tangent-right
boundary behavior. `--spline-cache` overrides its location; source hashes,
all payload checksums, customer IDs, anchor weights, and direct spline parity
must match. This is descriptive saved-model analysis, not held-out performance
or known conditional-mean error. Missing cache rows fail without raw-XGB fallback.

To regenerate only the PDFs and summaries from checksummed residuals, supply
`--replay <completed-report>` and a fresh `--output-dir`. Previous report folders
are never overwritten. By default figures retain the full action and residual
ranges. If the all-row scatter is too heavy for a PDF viewer, add
`--action-renderer density` for a compact vector density plot counting every row.
Use `--central-fraction .95` to retain the central residual percentiles, and
optionally `--trim-actions` to intersect these with central action percentiles.
The latter retains slightly less than 95% jointly: `plot_selection.json` records
the exact counts/bounds and `plot_rows.npz` stores the displayed row IDs. Both
models share axes, bin grids and density color scaling. Raw residuals and
`summary.csv` always retain the full original population.

The retained 2026-10-26 reports under shared
`results/glm-spline-profit-dispersion/` are `residuals-20261026/` (the full
715,023-row density plot and histogram) and `central95/` (679,271 rows per model,
residual tails only, rectangular density and histogram). Their saved residual
arrays and full-population summaries are byte-identical. The central plot uses
each model's own 2.5th and 97.5th percentiles; the selected customers can differ.
Superseded rendering trials and the early action-trimmed trial are archived in
`scratch-results/glm-spline-profit-dispersion/rendering-trials-20261026/`.
Saved absolute paths remain original; the relocation record is
`scratch/retired_experiments/generali_residual_outputs_20261026.json`.
Use the current report directory for `--replay`.

At the user's subsequent request, the full-data action PDF was replaced at its
original path with a smaller vector density rendering. The complete original
scatter report is archived at
`rendering-trials-20261026/residuals-20261026-original-scatter/` alongside the
other trials. Data, full-population summaries and the histogram are unchanged;
only the action rendering and its documentation/provenance were revised. See
`scratch/retired_experiments/generali_residual_compact_20261026.json`.

For the follow-up exploration before selecting a residual basis:

```bash
python scratch/plot_generali_profit_residuals.py --explore \
  --replay ../results/glm-spline-profit-dispersion/residuals-20261026 \
  --output-dir ../results/glm-spline-profit-dispersion/customer-exploration-20261026
```

The wild-report `_slice_indices` helper selects smallest, median and largest
Euclidean norms of the saved GLM preprocessor's un-clipped whitened numeric
features. Categorical label codes, historical actions and outcomes are excluded
from this distance, but predictors still use their full feature sets. Both
models are plotted on the same three customers. Only the historical action has
an observed profit; the curves are predictions, not true counterfactual curves.
The slide action interval is extended to contain the three historical actions.
The unchanged spline boundary behavior is explicitly recorded.

The report saves selected source rows/features, numerical curves, Q-Q coordinates,
residual moments (full, trimmed, renewed and churned), and action-bin counts and
statistics. Its action plot is an explicit [-0.1, 0.3] zoom with outside counts;
the 10th--90th percentile ribbon is descriptive spread, not a confidence band.
No preprocessing, residual basis, band, or optimizer is fitted. Existing report
versions are unchanged; all outputs remain in the same comparison family.

To inspect residuals by the actual renewal outcome:

```bash
python scratch/plot_generali_profit_residuals.py --split-churn \
  --replay ../results/glm-spline-profit-dispersion/residuals-20261026 \
  --output-dir ../results/glm-spline-profit-dispersion/churn-split-20261026
```

The script verifies saved `is_churn` against the canonical CSV's original rows,
then produces full and central-95% 2×2 histograms for both models and both groups.
Each PDF uses common bins and axes, with unit-area density per displayed subset.
The central view trims residual tails within each group/model, without trimming
actions. `churn_split_summary.csv` and dashed means always use full groups;
`plot_selection.json` records exact bounds and retained counts. Group differences
condition on the realized outcome and do not imply expected-profit model bias
by themselves. Wild mean bands and prediction intervals for individual outcomes
have different targets; no bootstrap is fitted or coverage asserted here.

To fit the first affine residual basis and construct its wild bands:

```bash
MPLCONFIGDIR=/tmp/generali-residuals-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scratch/plot_generali_profit_residuals.py \
  --fit-wild-linear scratch/generali_residual_wild_linear.json \
  --replay ../results/glm-spline-profit-dispersion/residuals-20261026 \
  --output-dir ../results/glm-spline-profit-dispersion/wild-linear-20261026
```

This explicit fitting mode extends the same comparison family. The generic
manifest band adapter currently requires polynomial predictors; the companion
`scratch/generali_residual_wild.py` uses `WildLinearDesign` directly with the
verified frozen GLM/spline residuals and existing saved customer curves. It does
not substitute a polynomial for either Generali predictor or change shared
mathematics. No optimizer is run or action selected.

The JSON setup declares the affine numeric basis (intercept, 11 saved whitened
numeric features, action), an 80/20 validation split, nominal 95% calibration,
1,999 bootstrap draws and root seed 20261026. Split and method streams use
`derive_seed(root, "generali-residual:validation-split")` and
`derive_seed(root, "generali-residual:wild-multipliers")`; method child
`draw:<index>` seeds generate one Rademacher row each. Both predictors and future
bases reuse these draws and the ordered cohort. Changing batching preserves
draws; the complete sign matrix's SHA-256 is saved without storing a huge matrix.

Validation keeps repeated customer IDs together. The final band refits all
rows using the existing raw post-fit residual bootstrap with row-wise signs.
Its statistic covers observed contexts crossed with all real actions; the
unchanged saved slice grid is for rendering. Exact affine and QR formulas give
off-grid residual evaluations, and common coefficient draws induce dependence
across queries. Frozen predictor curves retain their original boundary behavior.

Outputs include train/holdout/full-refit metrics (also by observed churn), beta
coefficients, leverage diagnostics, full numeric fit state and split indices,
draw seeds/statistics/critical values, plotted endpoints, and three vector PDFs:
holdout calibration, residual-mean bands, and transferred profit envelopes.
The latter also shows the corrected mean prediction, which is distinct from
the transferred envelope's frozen center. Source/data/model hashes and resolved
configuration make the run reproducible. Original residual reports remain intact.

Coverage is unknown: this dataset supplies no conditional-mean truth, and the
linear residual specification, independent-row errors and low-leverage conditions
are unverified. The row-wise bootstrap does not address within-ID dependence.
The synthetic Gaussian finite-sample bound is not asserted for this data.
Held-out correction fit is not held-out evaluation of the original predictors.
Future basis comparisons should reuse this split and draws and consider both
approximation error and variance/leverage; better fit need not yield narrower bands.

For figure-only revisions, use `--render-wild <completed-wild-run> --output-dir
<fresh-directory>`. This checks input hashes and renders the saved endpoint
arrays without reconstructing a fit, generating bootstrap draws, or evaluating
models. The baseline's final band PDFs are in `wild-linear-20261026/plots/`:
outside legends keep observed markers visible. Its original root PDFs and all
numeric artifacts retain their original provenance; the rendering revision has
its own source/input/output hashes.

The downstream policy analysis keeps the same comparison family:

```bash
MPLCONFIGDIR=/tmp/generali-residuals-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scratch/plot_generali_profit_residuals.py \
  --optimize-wild scratch/generali_wild_policy.json \
  --replay ../results/glm-spline-profit-dispersion/residuals-20261026 \
  --output-dir ../results/glm-spline-profit-dispersion/wild-policy
```

The saved beta, QR factor and critical value are replayed without fitting or
bootstrap draws. For plots, evaluate each lower/upper endpoint and then average
over all 715,023 contexts at each common action. Optimization instead maximizes
the mean of those **customer-specific lower endpoints at personalized policy
actions**. Never optimize the averaged plotting curve or substitute a band
at the average context. Report policy profit by evaluating each selected action,
not by reading the common-action curve at the policy's mean action.

Keep the three cohorts distinct: the saved beta-validation split has 572,022
training and 143,001 holdout rows (grouped by customer ID); the final beta refit
and bootstrap calibration use all 715,023 rows; only downstream policy
optimization uses the saved 20,000-row presentation cohort. Full-population
policy evaluation is descriptive, not an independent residual-fit holdout.

The presentation setup supplies saved 20,000-row indices, 19-feature policy
preprocessing, a bounded softmax-linear policy on [-0.1,0.2], initial action
0.08, acceptance floor 0.8787745289312372, and first-order trust-constr settings.
All policy endpoints come from `optimization.solvers.run_first_order_minimize`
through the shared `BandObjective`; there is no grid-selected optimum or direct
SciPy solver. A matched nominal-profit optimization measures the change caused
by the lower-band objective. The GLM reference's synthetic-tail directory is
used only for its saved cohort/preprocessing/constraint; no synthetic penalty
is carried into this experiment.

The study-specific adapter in `scratch/generali_wild_policy.py` joins the frozen
band and Generali responses with the existing policy/optimizer interfaces. It
uses exact saved affine QR norms off-grid, exact GLM logits, and the validated
full-population monotone-spline cache with its analytic derivatives. A collapsed
QR action-fiber cache is checked against the shared saved-band evaluator.
Common-action spline plots use the same pairwise evaluator with a repeated
action, so each query reads only the required two knots per customer. The
original `wild-linear-policy-20261026` attempt was interrupted during the
full-knot display evaluation; its two completed GLM policy artifacts and
`interrupted.json` are retained. The `-r2` attempt was interrupted during spline
optimization because repeated file opens dominated evaluation time; its GLM
artifacts and interruption record are also retained. The `-r3` run retains the
fixed 20k cohort's exact spline knots in memory and evaluates them through the
same shared Hermite primitive, checking values and derivatives against the
file-backed evaluator on all 20k rows. It records complete optimizer outcomes.
The action grid only renders curves. Native positive-part derivative conventions
are preserved; optimizer convergence is local and global optimality is unknown.

No new random streams are introduced: saved rows and policy preprocessing are
replayed, and first-order optimization uses the entire fixed 20k batch. The
acceptance floor constrains that optimization cohort; full-population acceptance
is reported separately. Outputs retain policy coefficients/actions/traces,
full-context curve coordinates, nominal/corrected/lower-bound profit summaries,
histogram counts and three vector PDFs. New-price actual profit and statistical
band coverage remain unknown; original runs are preserved.

The completed `wild-policy` baseline keeps full-data
calibration. Both GLM fits satisfy the solver's gradient termination criterion;
both spline fits exhaust the inherited 1,000-step budget with zero sample
constraint violation, so their saved policies remain unconverged candidates.
Their objective changes over the last 100 steps are below 0.00004 per row;
this is not a convergence or global-optimality certificate. The full-population
GLM nominal policy's predicted acceptance (0.878663674) is slightly below the
sample floor, while its lower-band policy satisfies it (0.878786783).
All three PDFs were opened and visually inspected; saved solver endpoints,
histogram counts, action bounds, sample constraints, and source/output hashes
were verified. Eight focused tests and the small repository-optimizer demo pass.

### Saved Generali wild-band slices and historical-action cohorts

Use the existing residual driver to evaluate the saved linear band without any
new fit, bootstrap draw or optimization:

```bash
MPLCONFIGDIR=/tmp/generali-residuals-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scratch/plot_generali_profit_residuals.py \
  --explore-wild scratch/generali_wild_slices.json \
  --replay ../results/glm-spline-profit-dispersion/residuals-20261026 \
  --wild-action-bounds -1 2 \
  --output-dir ../results/glm-spline-profit-dispersion/wild-band-exploration
```

This revision uses the approved inclusive historical-action groups [-.005,.005],
[.095,.105], and the highest 5% (a >= 0.14428286142004157). All qualifying rows
contribute to group-average endpoints. A customer's historical action determines
group membership; the plotted horizontal axis queries different actions at the
same contexts. It is not a local-action recalibration or causal group comparison.
The individual panels use the original stable median-norm helper within each
group. Without a display override, norm panels replay the original coordinates.
With `--wild-action-bounds LOW HIGH`, both norm and historical-action plots query
the frozen functions across that interval and set matching axis limits. The
original norm-grid points inside the interval are retained and checked against
the saved band. No beta fit, bootstrap recalibration or policy-bound change occurs.

The common-action table compares half-width and transferred radii at a=.1;
logged-action values are also retained. The saved whitened numeric representation
defines norms and nearest-other-customer distances, excluding all same-ID rows.
Raw-feature ranks and optimizer-cohort membership provide descriptive context.
No trained-model OOD claim follows: these rows are in the calibration dataset,
predictor training membership is unknown, and categorical support is not tested.
CSV coordinates, cohort row IDs, selected features, width/penalty diagnostics,
four vector PDFs, and checked input/output provenance are saved together.
All current figures are directly in `wild-band-exploration/`, with descriptive
names: `customer_norm_profit_bands.pdf`,
`historical_action_customer_profit_bands.pdf`, `historical_action_mean_bands.pdf`,
and `historical_action_widths_and_penalties.pdf`. Regenerate only the two group
figures with `--render-wild-slices <saved-slice-report> --output-dir
<fresh-directory>`; this preserves recorded axis limits without reevaluation.

The requested cleanup moved interrupted attempts, narrow-range slice reports
and duplicate calibration renders to
`scratch-results/glm-spline-profit-dispersion/superseded-wild-runs-20261026/`.
The completed policy result was renamed from its retry-labelled directory to
`wild-policy`. Original files/provenance were preserved byte-for-byte; resolve
historical paths using
`scratch/retired_experiments/generali_wild_outputs_20261026.json`. Required
residual/customer inputs and the full fitted/bootstrap state remain available.
The expanded run passes the ten focused checks and the small optimizer demo.
Both models' saved endpoints at -1 and +2 agree with the native band evaluator;
cohort IDs and common-action/historical-action diagnostics are byte-identical
to the narrow study. Current PDFs were opened and visually inspected, and all
archived/renamed files were verified against the relocation checksums.

Baseline findings: source row 715959 has the largest norm (3088.50) and recorded
`X_claim_tot_value` 1,001,562.35 (population 99th percentile 2,100). At a=.1 its
GLM/spline residual half-widths are 811.03/489.95, but the GLM lower penalty is
zero because its fitted residual is positive and larger than the half-width.
It is in calibration and absent from the saved 20k optimizer cohort. High-action
customers are enriched for top-1%-norm contexts (8.00%, versus 0.34% near a=.1),
while their median-norm representative is only at the 76.5th norm percentile.
Ten focused tests and the small repository-optimizer demo pass. Cohort row IDs,
mean endpoint identities, unchanged original artifacts and report checksums were
verified; all four final PDFs were opened and visually inspected.

## Retirement and recovery

Generic uncertainty-envelope, historical pairs-bootstrap/OLS, offline-bootstrap,
policy-LCB and coverage-kernel study drivers moved to tracked scratch, together
with their reporting orchestration. Current terminal/residual/fixed-noise studies,
illustration drivers, Generali models/data/configurations and shared mathematics
remain available. Recipes and results were already moved in the previous cleanup;
this source refactor does not move or delete further results.

See [archive instructions](../scratch/retired_experiments/README.md) and
[source relocation records](../scratch/retired_experiments/source_relocations_20261023.json).
Historical artifact hashes and absolute paths are unchanged. The archived runner
is for new exploratory runs at a fresh location; exact historical replay requires
the recorded source checkout and inputs.

Compatibility note: policy-LCB, offline-bootstrap and finite-policy study exports
were removed from `experiments.__init__`. External notebooks using those imports
must use the archived study modules under `scratch.retired_experiments.studies`,
or the original source checkout for exact replay. The relocation records above
map each former module to its archive path; this is not a drop-in import alias.

## Refactor validation

Validation on `refactor/experiment-band-workflow` (2026-10-23):

- PR review follow-up: 63 focused workflow/manifest/provenance/execution/sweep
  checks passed. The full suite completed with 1,292 passes and one failure:
  the existing legacy proof-validation NaN plotting case. The declared Markdown
  dev dependency was available via the temporary test directory. The small
  repository-optimizer demo and CLI help check passed; public docs regenerated.
  Regression coverage includes source-cache invalidation (including timestamp
  collisions), changed input artifacts, partial-run integrity, missing/mismatched
  endpoint exports, completed optimization `--force`, and the empty-seed guard.
- 117 focused retained-study, archive, band and Generali preparation checks passed;
  31 final workflow/manifest/sweep checks passed after the resume/PDF updates.
- The full suite completed with 1,268 passes and two failures. One is the already
  documented legacy proof-validation NaN plotting failure. The other was the
  environment's missing declared `Markdown` dev dependency; after installing it
  into a temporary test-only directory, all seven math-document checks passed.
- JAX initially aborted with unrestricted CPU affinity on this host. The full
  suite completed using `JAX_PLATFORMS=cpu` and `taskset -c 0-3`.
- A small `main.main` demo passed with 32 samples, three steps, seed 7, and the
  first-order, finite-difference and Stein estimators. An optimization-only CLI
  run produced valid PDFs with no raster copies. Band-only, band optimization,
  saved-fit replay and interrupted endpoint-export recovery are covered by tests.
- All 30 source relocation records match the original Git objects. Active
  manifests are unchanged; shared results and data were not modified. Both
  launchers import and expose help, and API documentation was regenerated.

The reusable band adapter is NumPy-only and intentionally validates its current
polynomial/support method and domain capabilities. New Generali response-based
calibration or a different stochastic representation still needs an explicit
scientific setup; this refactor does not supply arbitrary assumptions.
