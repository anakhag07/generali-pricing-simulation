# Generali Pricing Simulation

Pricing simulation and black-box optimization demo with pluggable objectives,
policies, and gradient estimators. More detailed docs can be viewed [here](https://anakhag07.github.io/generali-pricing-simulation/index.html).

## Quickstart

```bash
# Conda
conda create -n simulation_env python=3.11
conda activate simulation_env
pip install -e .
python main.py
```

Runtime dependencies live in `requirements.txt` and mirror `pyproject.toml`
(including CPU JAX). Install `.[jax-cuda12]` only when you need CUDA-specific
JAX wheels.

On ORCD, launch-aware entry points can run locally or submit themselves to
Slurm. `python main.py` defaults to `--launch auto`, submitting when outside a
Slurm allocation and running directly inside one. Use `--launch local` (or the
legacy alias `--no-sbatch`) to run in the current process, and `--launch slurm`
to force submission when outside an allocation. Runs with
`compute_backend="jax"` use `mit_normal_gpu` with one L40S GPU by default;
NumPy-only runs use the CPU `mit_normal` profile. Slurm logs are written under
`results/slurm/%x-%j.out`.

Sweep entry points that opt into the shared launcher can split task-level work
across Slurm arrays:

```bash
python main.py --launch slurm --array
python scripts/run_experiment_manifest.py path/to/manifest.json
python scripts/run_sweep.py fixed_regression_base --launch slurm
python scripts/run_sweep.py real_data_glm_base --requires-jax --launch slurm
python scripts/run_noisy_glm_theta_variance_sweep.py --launch slurm --array --array-max-parallel 6
```

Array tasks write task records under
`results/<project>/sweeps/<sweep-id>/tasks/`; sweep collectors run after the
array and rebuild aggregate CSVs/plots from completed task summaries.

Run artifacts land in a shared external results root at
`~/projects/generali-pricing/results` by default, independent of the active
checkout or worktree. Override this with `GENERALI_RESULTS_ROOT=/path/to/results`.
Normal run directories use readable flat names like `<slug>__<timestamp>`.

To run tests:

```bash
pip install -e ".[dev]"
pytest -q
```

## What This Does

Most objectives optimize a parameterized policy over state vectors:

$$
x \sim \mathcal{N}(0, I),\quad \theta \in \mathbb{R}^p,\quad u = \pi_\theta(x)
$$

The optimizer solves the theta-space objective:

$$
\min_{\theta} J(\theta),\qquad
J(\theta) = \mathbb{E}_x\big[f(\pi_\theta(x); x)\big]
$$

Direct theta-space objectives may instead define $$J(\theta)$$ without a policy.
`StronglyConvexQuadratic.isotropic(d)` provides the strongly convex benchmark
$$J(\theta)=\frac12\|\theta\|_2^2$$ with unique minimizer $$\theta^*=0$$, and
`StronglyConvexQuadratic.from_seed(...)` generalizes it to a seeded rotated
spectrum with a configurable condition number. `NoisyObjective` can apply
reproducible homoskedastic Gaussian noise keyed by the exact parameter vector to
these policy-free objectives.

Objectives are split by provenance under `src/objective/objectives/`:
`generali/` needs the real dataset and trained artifacts, `synthetic/` is
self-contained with analytically known optima. Both are re-exported, so
`from objective import X` works regardless of which side a class lives on.

Pluggable components:
- **Objectives**: synthetic — the ladder (`StronglyConvexQuadratic`, `SmoothedNonconvex`), `FixedRegressionObjective`, `PlantedLogisticObjective`; generali — `ModelBasedObjective`, `PreparedGLMObjective`, `JaxPreparedGLMObjective`; plus `NoisyObjective` and `BiasedObjective` wrappers
- **Policies**: `ConstantPolicy`, `LinearPolicy`, `SoftmaxPolicy`, `MLPPolicy` (2-layer, default hidden=16)
- **Gradient estimators**: `constant`, `first_order`, `finite_difference`, `gauss_stein`, `stein_difference`, `spsa`

The default bounded policy is `SoftmaxPolicy`, which maps
`u = action_low + (action_high - action_low) * sigma(theta^T phi(x))`.
The default action range is `(-0.5, 0.5)`; pass custom bounds such as
`SoftmaxPolicy(action_low=-0.1, action_high=0.2)` or the real-data override
`softmax_action_bounds=(-0.1, 0.2)` to restrict proposed actions.
`LinearPolicy` and `SoftmaxPolicy` support configurable state feature maps
`varphi(x)`. The policy prepends the intercept internally, so
`phi(x) = [1, varphi(x)]` and custom feature maps should not include the
leading `1`. The default `IdentityFeatureMap` gives the previous behavior
`phi(x) = [1, x]`; `QuadraticFeatureMap` expands the state with linear,
square, and pairwise interaction terms. `CubicFeatureMap` and
`QuarticFeatureMap` follow the same pattern with linear terms plus exact
degree-3 or degree-4 monomials.

Policy replay separates input preprocessing from policy feature mapping:

- **Policy input preprocessing** maps raw source rows to numeric policy inputs
  `z`. For real-data objectives this may include artifact preprocessing plus
  optional policy-side standardization, whitening/sphering, and PCA.
- **Policy feature mapping** maps `z` to `varphi(z)` and, for linear/softmax
  heads, the policy internally builds `phi(z) = [1, varphi(z)]`. PCA and
  whitening are not `phi`; they are upstream preprocessing.

For the real-data model-based objective, this `u` remains centered and the
revenue term uses premium multiplier `u + 1`.

`finite_difference` is a deterministic coordinate-wise central-difference baseline
that uses `2 * dim(theta)` objective evaluations per gradient call.

Core API convention:
- `sample_states(rng, n, dim)` produces state batches with shape `(n, dim)`.
- `Policy.value/grad` and `Objective.value/grad` operate on 2D `x_batch` arrays.
- `Policy.weighted_grad(theta, x_batch, weights)` is the preferred VJP-style
  path for chain-rule gradients because it avoids materializing full
  `(n_samples, theta_dim)` policy Jacobians.

Policy-free runs report theta/objective diagnostics normally and leave action
metrics such as `final_u` and `mean_u` null.

The synthetic ladder provides the policy-free benchmarks: seeded functions whose
global minimizers are known exactly by construction, so true-gap metrics need no
reference run (see MATH.md 3.7). `synthetic_quadratic_base` is a rotated
ill-conditioned quadratic and `synthetic_smoothed_nonconvex_base` a quadratic with
compactly supported traps, with piecewise convex/nonconvex rungs stubbed for later.
`dimension` and rung parameters are factory and sweep axes:

```python
config = get_config(
    "synthetic_quadratic_base",
    overrides={"dimension": 50, "function_params": {"condition_number": 1000.0}},
)
```

Ladder presets default to `l-bfgs-b` with estimators `first_order`,
`finite_difference`, `spsa`, and `stein_difference` (theta-space two-sided
mode); there is no action space, so `perturbation_space="u"` is rejected.

Ladder runs record their construction in `summary.json` as
`{rung, spec, w_star, fingerprint}`, so a saved run can be replayed with
`SyntheticFunction.from_dict(...)`. The fingerprint makes a changed `from_seed`
construction fail loudly rather than silently rebuild a different function.

For `synthetic_smoothed_nonconvex_base`, `function_params={"depth_fraction": ...}`
controls whether the traps are genuine local minima. The default `0.9` traps every
basin; lowering it toward `0.1` makes descent roll straight through, leaving the
rung unimodal despite still being the nonconvex rung.

`NoisyObjective` wraps an existing objective with additive deterministic
action-level noise $$\hat{M}(x,u)=M(x,u)+\delta(x,u)$$. The
`HomoskedasticGaussianNoise` adapter (constant std) is keyed by exact
`(x, u, seed)`, so the same row/action pair has the same noise on every call.
`HeteroskedasticGaussianNoise` scales the same keyed unit-normal field by
$$\sigma_0 + \gamma\,|u - u_c|$$, so noise grows with action distance from
`u_center` (typically the planted optimum) and vanishes near the global
minimum when $$\sigma_0 = 0$$. It exposes noisy value
oracles for zeroth-order optimization and intentionally has no analytical
gradient; use `base_objective.grad(...)` when inspecting the true non-noisy
objective gradient. Summaries and policy evaluations use the wrapped clean
objective when it exposes `base_value(...)` / `base_value_at_u(...)`, while
optimizer traces still record the noisy oracle values. For trace diagnostics on noisy objectives,
`CorrectnessSpec(gradient_source="denoised_exact")` uses the exact gradient of
the wrapped clean objective, while `gradient_source="exact"` still refers to the
optimizer-facing noisy objective.

`BiasedObjective` wraps an existing objective with deterministic action bias
$$\hat{M}(x,u)=M(x,u)+b(x,u)$$. `LinearActionBias` uses
$$b(u)=-\lambda_{bias}u$$, so positive `lambda_bias` makes larger actions look
artificially better for minimization. `UpperSupportHingeBias` uses
$$b(u)=-\lambda_{bias}(u-h)_+$$ with `h = support_center + support_radius`, so
the surrogate is exact inside support and optimistic only above support. The
wrapper exposes the biased surrogate to the optimizer while `base_value(...)` /
`base_value_at_u(...)` preserve true-objective reporting. The focused
planted-logistic experiments are `scripts/run_planted_logistic_action_bias_sweep.py`
and `scripts/run_planted_logistic_support_bias_sweep.py`.

Policy-free objectives can instead use `ThetaBiasModification`. The built-in
one-dimensional fields are `LinearThetaBias` ($$\alpha x$$),
`ArctanThetaBias` ($$\alpha\arctan x$$), and
`ArctanRemainderThetaBias` ($$\alpha(x-\arctan x)$$).

### Zeroth-Order Proof Validation

The `zeroth_order_proof_base` preset implements the globally strongly convex
objective $$f(x)=x^2+\frac12(\sin x-x)$$ with known $$x^*=0$$. Two committed
manifests check the predicted finite-difference and Stein-difference dependence
on perturbation radius, Stein sample count, and deterministic functional bias:

```bash
python scripts/run_experiment_manifest.py manifests/zeroth_order_baseline.json
python scripts/run_experiment_manifest.py manifests/zeroth_order_functional_bias.json
python scripts/analyze_zeroth_order_proof_validation.py \
  --baseline-manifest manifests/zeroth_order_baseline.json \
  --bias-manifest manifests/zeroth_order_functional_bias.json
```

The analyzer records $$x_0$$, the clean and biased optima, each estimator's
population fixed point, and $$x_K$$. It writes sweep-level displacement,
bias/variance/MSE, theorem-bound, and fitted-scaling tables and plots under
`results/zeroth-order-proof-validation-analysis/`.

### Zeroth-Order Support Envelopes

The committed support-envelope sweep reuses the same strongly convex proof
objective and compares a constant control, a constant-derivative distance
envelope, and a smooth monotone-but-nonconvex saturating envelope outside the
covered interval $$[0.75,1.25]$$:

```bash
python scripts/run_experiment_manifest.py manifests/zeroth_order_envelopes.json
python scripts/analyze_zeroth_order_envelopes.py \
  --manifest manifests/zeroth_order_envelopes.json
```

The 54 variants vary matched envelope strength, initialization, and (for the
nonconvex form) zeroth-order radius. The analyzer writes exact and
population-smoothed stationary points, final-iterate/regret tables, basin-rate
and finite-run target-error plots, and `envelope_diagnostics.png`, which shows
each envelope together with the true and upper objectives, under
`results/zeroth-order-envelope-analysis/`.

Optimization step rules:
- `l-bfgs-b` uses `scipy.minimize(method="L-BFGS-B")`.
- `trust-constr` uses `scipy.minimize(method="trust-constr")` and adds the
  acceptance-floor equation directly as a nonlinear constraint. Optional
  `initial_constr_penalty` is passed through to SciPy when configured.
- `constant` uses the repo's manual gradient loop with fixed `step_size`.
- `armijo` uses the same manual loop with Armijo backtracking seeded by `step_size`.
- `optax-adam` / `optax-sgd` run an optax update loop with `step_size` as the
  learning rate. Gradients come from the configured estimator, so all gradient
  methods work; acceptance floors use the penalty or Lagrangian objective paths
  (there is no constrained optax driver). Compare against SciPy with
  `scripts/benchmark_optax_vs_trust_constr.py`.

`enabled_estimators=("constant", ...)` optimizes a one-parameter
`ConstantPolicy` on the same objective/data as the configured policy. This is
separate from `ExperimentConfig.constant_u_baselines`, which only evaluates
fixed-action reference points such as `(-0.3, 0.0, 0.2)` and shows them in
`summary.json`, the console summary, and `plots/optimization/loss_curves.png`.

## Documentation

Full API documentation, objective formulas, and configuration reference are
available in `docs/` (generated via pdoc). `MATH.md` provides a central
reference for all mathematical formulas implemented in `src/`, organized by
module layer.

## Data Sources

Available base presets include:

| Preset | State source | Objective |
|---|---|---|
| `synthetic_quadratic_base` | Fixed dummy batch (ignored) | `StronglyConvexQuadratic` (seeded rotation, configurable condition number) |
| `synthetic_smoothed_nonconvex_base` | Fixed dummy batch (ignored) | `SmoothedNonconvex` (known global minimum, local-minima traps) |
| `zeroth_order_proof_base` | Fixed dummy batch (ignored) | One-dimensional strongly convex sine-perturbed quadratic with known minimum |
| `fixed_regression_base` | Synthetic N(0, I) | `FixedRegressionObjective` |
| `real_data_glm_base` | All complete eligible raw acceptance CSV rows by default; seeded `n_samples` draw when set | `ModelBasedObjective` (GLM bundle, analytical grad when supported) |
| `real_data_xgb_base` | All complete eligible raw acceptance CSV rows by default; seeded `n_samples` draw when set | `ModelBasedObjective` (XGBoost bundle, FD acceptance gradient) |
| `real_data_xgb_logit_spline_base` | 200 canonical rows covered by per-policy splines; seeded `n_samples` subset when set | `ModelBasedObjective` (XGBoost-derived logit splines, analytical acceptance gradient) |
| `real_data_glm_glm_20260728_base` | Complete canonical rows | 20260527 GLM acceptance + 20260527 GLM loss |
| `real_data_smoothed_glm_20260728_base` | Exact 200 rows covered by the supplied sigmoids | 20260728 shifted-sigmoid acceptance + 20260527 GLM loss |
| `real_data_xgb_glm_20260728_base` | Complete canonical rows | 20260728 XGB acceptance + 20260527 GLM loss |
| `real_data_xgb_xgb_20260728_base` | Complete canonical rows | 20260728 XGB acceptance + 20260728 XGB loss |

Real-data overrides can select policy, feature order, preprocessing, loss source,
constraint mode, and runtime knobs without adding a new preset module. Example:

```python
config = get_config(
    "real_data_glm_base",
    overrides={
        "policy_kind": "softmax",
        "softmax_action_bounds": (-0.1, 0.2),
        "initial_u": 0.0,
        "feature_order": "quartic",
        "policy_preprocessing": "no_pca",
        "loss_source": "observed",
        "constraint_mode": "trust_constr",
        "enabled_estimators": ("first_order", "constant"),
    },
)
```

Supported policy axes are `policy_kind in {"constant", "linear", "softmax",
"mlp"}`, `feature_order in {"linear", "quadratic", "cubic", "quartic"}`,
`policy_preprocessing in {"artifact", "no_pca"}`, and `constraint_mode in
{"none", "trust_constr", "penalty", "lagrangian"}`. `compute_backend` defaults
to `"numpy"`; GLM `trust_constr` fixed-full-batch runs may use `"jax"`.
Softmax real-data runs also accept `softmax_action_bounds=(low, high)`. `loss_source` defaults to
`"predicted"`; setting `loss_source="observed"` keeps model-predicted acceptance
but uses row-aligned historical `Y_G_Loss` as the loss term. GLM real-data runs
also accept a `u_coef` override for counterfactual acceptance sensitivity sweeps;
it changes only the logistic acceptance coefficient on generated policy `u`, not
the loss term.

The objective for real-data configs is $$f(u; x) = a(x,u)(L(x) - (u + 1) \cdot p(x))$$
where $$a$$ is acceptance probability, $$L$$ is either expected financial loss
$$\hat{Y}$$ or observed `Y_G_Loss`, and $$p$$ is policy premium.

Real-data source rows now live in the canonical `src/data/dataset.csv` file,
with schema/path metadata tracked in `src/data/dataset_metadata.py`. The current
canonical CSV is the 052726 raw single-year export; both GLM/linear and XGB
real-data loaders sample complete eligible rows from it.
Model artifacts live under `src/data/models/linear/`, `src/data/models/xgb/`,
`src/data/models/xgb_logit_spline/`, and `src/data/models/xgb_sigmoid/`.
Rebuild the portable spline artifact
from the trusted legacy smoothing bundle with
`python scripts/prepare_xgb_logit_spline_artifact.py`.
Convert the supplied 20260728 shifted-sigmoid wrapper with
`python scripts/prepare_xgb_sigmoid_artifact.py`; runtime loads only the
validated NPZ and never invokes the legacy pickle. The loader exposes independent
versioned `acceptance_model_type` and `loss_model_type` selectors. Historical
presets retain their original paired artifacts, while the 20260728 best-fold
bundles are opt-in.
The objective keeps raw CSV X rows at the optimization boundary and reuses each
artifact's saved `FeatureProcessor` internally. The 052726 classifiers expose
class-1 probability as direct `p_accept(x, u)`, not churn probability.
The `xgb_logit_spline` option instead uses 200 customer-specific acceptance
curves derived from the five-fold XGBoost ensemble. Each curve freezes one
covered insurance-policy profile and varies `U` over its fitted `[0, 0.16]`
support. The preset includes only those 200 canonical rows, carries `id` solely
for curve lookup, defaults bounded policies to the fitted support, and exposes
an analytical action derivative for NumPy first-order optimization. Unknown IDs
raise instead of silently falling back to non-differentiable trees.
Only the model artifact X covariates are numerical objective/policy inputs.
Historical `U`, `Y_G_Loss`, `is_churn`, IDs/dates, and the lookahead
`X_upcoming_premium` column remain excluded from those inputs; the spline preset
carries `id` only as a curve lookup key. Observed `U`/churn are retained for
diagnostics and acceptance-floor summaries.
For GLM/linear artifacts, extractable first-fold coefficients are used for
array-native acceptance and loss predictions, avoiding repeated sklearn
prediction calls in value-query gradient estimators. XGBoost and unsupported
artifacts fall back to their bundled estimator prediction methods.
`PreparedGLMObjective` can further move the GLM hot path onto a compact numeric
batch with columns `[base_logit, loss, premium, policy_features...]`. Use
`prepare_glm_objective(model_based_objective, x_frame)` to materialize the
numeric objective/batch pair after raw pandas/artifact preprocessing has run once.
For GLM `trust-constr` parity experiments, `compute_backend="jax"` keeps SciPy's
constrained optimizer but evaluates the prepared GLM objective, gradients,
zeroth-order value queries, mean acceptance, and constraint Jacobian through
JIT-compiled JAX callbacks. The same prepared JAX objective also runs under the
`optax-adam` / `optax-sgd` step rules, replacing the SciPy driver with an optax
update loop (acceptance floors via the penalty/Lagrangian paths).
The JAX backend requires `batch_size=None` and
supports fixed full-batch GLM runs for `first_order`, `finite_difference`,
`gauss_stein`, `spsa`, and `stein_difference` with constant policies plus
linear or softmax policies over finite materializable feature maps, including
the built-in linear, quadratic, cubic, and quartic maps and `CallableFeatureMap`.
`MLPPolicy` is also supported: theta is unpacked into layer weights inside a
jitted forward pass and value/grad/mean_acceptance flow through autodiff, so MLP
GLM runs work with every estimator except the action-space `stein_difference`
(whose policy Jacobian is not materialized for MLP).
The expanded policy design matrix is materialized once before transfer to JAX,
so high-order or callable maps increase fixed-batch device memory use. When
launched through `main.py`, JAX configs are submitted to ORCD GPU Slurm and fail
fast if JAX reports only a CPU backend, preventing silent CPU fallback.
`NoisyObjective(ModelBasedObjective(...))` is also supported on this fixed-batch
JAX path for zeroth-order trust-constr sweeps: the clean GLM base is prepared on
device first and then re-wrapped with the deterministic noise field.
For policy-feature experiments, `ModelBasedObjective` can instead take a
separate fitted policy-side preprocessor. In that mode the policy sees the
configured policy features, while the sealed acceptance and loss model paths
still receive raw `x` and apply their saved artifact preprocessing internally.
Real-data configs use all complete eligible acceptance CSV rows when `n_samples`
is omitted or set to `None`. Setting an integer `n_samples` samples that many
complete eligible rows with the experiment seed. The selected row indices are
stored so observed-`U` diagnostics use the same source rows.
Set `train_fraction` and `test_fraction` to split the selected rows for a run;
they must sum to `1.0`, with `train_fraction > 0`. Optimizers fit only on the
training rows, while final policies are evaluated on both train and test rows in
`summary.json` and the policy diagnostic folders.
Normal runs also write reloadable trained-policy artifacts under
`policies/<estimator>/policy.json` with sidecar arrays in `arrays.npz`. These
artifacts save theta, source CSV row bindings, model/objective metadata, and the
full fitted policy-side preprocessing state so validation can be rerun without
retraining the optimizer:

```python
from experiments.policy_artifacts import load_policy_artifact

artifact = load_policy_artifact("results/.../policies/first_order/policy.json")
u_train = artifact.predict_u(split="train")
train_metrics = artifact.evaluate(split="train")
```
The policy-artifact CLI can evaluate either the trained model objective or an
observed historical diagnostic on saved run rows:

```bash
python scripts/evaluate_historical_policy_objective.py \
  --policy-artifact results/.../policies/first_order/policy.json \
  --objective model \
  --split train

python scripts/evaluate_historical_policy_objective.py \
  --policy-artifact results/.../policies/first_order/policy.json \
  --objective historical \
  --split all

python scripts/evaluate_historical_policy_objective.py \
  --u-source historical \
  --model-type glm \
  --acceptance-source model \
  --technical-price-source historical
```

`--objective model` replays
`p_accept_model(x,u) * (loss_hat_model(x) - revenue)`, matching training
metrics for the same split. `--objective historical` uses observed outcomes
`(1 - is_churn) * (Y_G_Loss - revenue)` with the learned policy `u`, so it is
an observed-outcome diagnostic and need not match the training objective.
`--u-source historical` does not use the optimized policy action; it evaluates
historical CSV `U` with independently selected `--acceptance-source` and
`--technical-price-source` values, each one of `historical` or `model`.
Supported splits are `train`, `test`, and `all`, where `all` means all selected
rows from the run before train/test splitting.
To inspect client-level counterfactual acceptance curves for a saved policy,
sample clients from low/medium/high mean-sensitivity and predicted-loss tertiles:

```bash
python scripts/plot_policy_acceptance_grid.py \
  --policy-artifact results/.../policies/first_order/policy.json \
  --split all \
  --u-min 0 \
  --u-max 0.15 \
  --u-count 61 \
  --n-clients 10
```

This writes separate three-panel acceptance-curve plots by sensitivity and by
predicted loss under `results/policy-acceptance-grid/`, with each sampled
client's artifact policy action overlaid on its predicted acceptance curve.
When plotting is enabled, real-data runs write optimization plots under
`plots/optimization/` and final customer-level policy diagnostics under
`plots/policy_train/` and `plots/policy_test/`. Policy diagnostics include final
metric bars with 25-75% customer ranges, observed-vs-policy `u` and acceptance
histograms, `delta_u_histogram.png` showing `Δu = optimized customer u - historical u`,
`delta_u_by_sensitivity.png` showing the same `Δu` against absolute acceptance
sensitivity at `u=0.08`, `objective_contribution_summary.png` showing
customer-level expected profit spread and expected profit vs predicted
acceptance, and one
`u_acceptance/<estimator>.png` file per estimator showing the final action
histogram, customer acceptance-vs-`u` scatter, and raw objective contribution
histogram.
Plot generation writes per-plot wall-clock diagnostics to `plots/plot_timings.json`.
For model-based objectives, theta contour plots are evaluated on a deterministic
subsample of at most 200 rows so large real-data experiments do not spend most
of their time rendering diagnostics; their contour grid is also capped at 20x20
instead of the default 60x60 used for cheaper synthetic objectives.

`ExperimentConfig` supports two acceptance-floor paths for objectives exposing
`mean_acceptance(theta, x_batch)`:

- `step_rule="trust-constr"` enforces `mean_acceptance >= acceptance_floor`
  directly inside SciPy as a nonlinear constraint. This is the constrained GLM
  path used by the softmax and linear trust-region presets.
- penalty-based step rules such as `l-bfgs-b` use
  `acceptance_penalty_weight` and `acceptance_penalty_temperature`, which add a
  differentiable penalty to `ModelBasedObjective`. This remains the XGBoost
  floor path.
- lagrangian scalarization uses `lagrangian_lambda` together with
  `acceptance_floor`, optimizing
  `J(theta) + lagrangian_lambda * (acceptance_floor - mean_acceptance(theta))`.
  This path is available on unconstrained step rules and keeps experiment
  summaries on the raw objective `J(theta)` so lambda sweeps can be compared on
  the same frontier.

`scripts/run_sweep.py` is a generic seed-aware preset sweep launcher around
`experiments.sweep_utils.run_sweep(...)`. Pass a base preset plus JSON overrides,
override lists, or override grids; use `--requires-jax` when a sweep should submit
to the GPU Slurm profile.

`scripts/run_experiment_manifest.py` is the preferred reusable entry point for
new sweeps. The JSON manifest must explicitly specify the objective preset,
objective modifications, optimizer, seed policy, truth source, launch mode, and
array structure. Reruns skip variants whose requested `summary-seed-<seed>.json`
files already exist unless `--force` is passed. Example:

```json
{
  "name": "quadratic-noise-grid",
  "objective": {"preset": "synthetic_quadratic_base", "overrides": {"dimension": 10}},
  "objective_modifications": [],
  "optimizer": {
    "step_rule": "l-bfgs-b",
    "compute_backend": "numpy",
    "t_steps": 100,
    "step_size": 0.01,
    "n_grad_samples": 64,
    "enabled_estimators": ["first_order", "finite_difference"],
    "plot": true
  },
  "seeds": {"run_seeds": [7, 8, 9], "anchor_seed": 7, "vary": ["theta"]},
  "truth": {"source": "clean_base_objective"},
  "launch": {"mode": "slurm", "array": "variant", "array_max_parallel": 6},
  "matrix": {"dimension": [10, 25, 50]}
}
```

To run the dedicated XGB logit-spline convergence and policy experiment, use:

```bash
python scripts/run_xgb_logit_spline_experiment.py --launch local
```

By default the script selects all 200 covered profiles, uses a deterministic
80/20 train/test split, and compares analytical first-order gradients with
action-space central finite differences from the same bounded softmax start.
Exact spline gradients are recorded as correctness diagnostics. Normal run
outputs, including optimizer status, `summary.json`, `steps.csv`, reloadable
policy artifacts, convergence plots, and train/test policy plots, are written
under `results/xgb-logit-spline-experiment/`. Use `--help` for sample-count,
split, seed, iteration, finite-difference, stochastic-gradient sample-budget,
estimator, and launch overrides.

To compare all four 20260728 hierarchy choices on the exact same 200-policy
sigmoid cohort, run:

```bash
python scripts/run_experiment_manifest.py \
  manifests/real_data_model_hierarchy_200.json
```

The manifest fixes the data, split, bounded softmax policy, `no_pca`
preprocessing policy, optimizer seed, and action-space finite-difference
estimator. Only the acceptance/loss artifact pair changes between variants.
Outputs land under `results/real-data-model-hierarchy-200/`.

`scripts/run_fixed_regression_noise_offset_grid.py` runs the synthetic
fixed-regression homoskedastic/heteroskedastic noise x theta-offset grid. It
computes a clean first-order reference, centers heteroskedastic noise at that
reference policy's mean `u`, varies `theta0 = theta_clean + delta * 1`, and uses
the canonical seed-sweep folder convention with warm `(family, noise-level)`
launch tasks. Outputs land under `results/fixed-regression-*-noise-offset-grid/`.

`scripts/run_noisy_glm_theta_variance_sweep.py` runs the all-data trust-constrained
GLM noisy-objective sweeps on GPU/JAX. By default it uses the saved no-noise
first-order run at
`/home/anakhag/projects/generali-pricing/results/real_data_glm_base__20260706_124627/summary.json`
as truth, centers heteroskedastic noise at that run's final mean `u`, sweeps
theta starts along the real initialization-to-truth line by L2 distance, and
sweeps homoskedastic/heteroskedastic noise variance for zeroth-order estimators.
Outputs land under `results/noisy-glm-*-sweep/` with final CSVs and theta-distance
diagnostic plots.

`scripts/run_glm_softmax_alpha_sweep.py` runs the trust-constrained softmax /
no-PCA / linear-feature GLM setup over symmetric action bounds
`[-alpha, alpha]` for `alpha in {0.5, 0.4, 0.3, 0.2, 0.15, 0.125, 0.1, 0.075}`.
It writes normal per-alpha runs and policy artifacts under
`results/glm-softmax-alpha-sweep/alpha_<value>__<timestamp>/`, then writes aggregate outputs
under `results/glm-softmax-alpha-sweep/alpha_sweep_<timestamp>/`: final
objective/profit CSVs and plots, acceptance-threshold profit summaries, and one
expected-profit-by-`u`-bin diagram per alpha. It accepts the shared launch flags;
`--launch slurm --array` runs one alpha value per array task.

`scripts/run_glm_u_coef_sweep.py` runs the softmax/no-PCA/trust-constr GLM setup
over `200000` sampled rows and direct acceptance coefficients
`u_coef in {-4, -5, -8, -10, -20}` with per-run policy distribution plots
enabled. It writes per-run outputs under `results/glm-u-coef-sweep/<u_coef-run>__<timestamp>/`
plus aggregate `glm_u_coef_sweep.csv` and frontier plots under
`results/glm-u-coef-sweep/u_coef_frontier_<timestamp>/`. It accepts the shared
launch flags; `--launch slurm --array` runs one `u_coef` value per Slurm array
task and submits a collector to write the same aggregate outputs.

`scripts/run_glm_sensitivity_bucket_experiment.py` buckets all complete GLM rows
into low/medium/high local price-sensitivity tertiles using
`|d p_accept(x, u_ref) / du|` at the median observed historical `U`, then runs
the same softmax/no-PCA/trust-constr GLM setup on all rows in each bucket. It
writes per-bucket policy distribution plots under `results/glm-sensitivity-buckets/`
and an aggregate `glm_sensitivity_bucket_experiment.csv` plus comparison plots
under `sensitivity_bucket_summary_<timestamp>/`. It accepts the shared launch
flags; `--launch slurm --array` runs one sensitivity bucket per array task.

`scripts/run_glm_reference_elasticity_bucket_experiment.py` repeats the bucketed
GLM experiment for reference actions `u_ref in {-0.1, 0.1, 0.2, 0.3}`, ranking
customers into low/medium/high buckets by elasticity magnitude at each reference
action. It runs only `first_order`, annotates summary charts with average bucket
elasticity magnitude, and writes per-reference summaries under
`results/glm-reference-elasticity-buckets/`. It accepts the shared launch flags;
`--launch slurm --array` runs one `(u_ref, bucket)` pair per array task.

`scripts/plot_glm_sensitivity_distribution.py` computes GLM customer
elasticities `d p_accept / du` across a default `u in [-0.3, 0.3]` grid. It
writes a mean/quantile elasticity-by-`u` curve, selected-`u` customer elasticity
histograms with default `0.5-99.5%` x-axis clipping marked on the chart, and CSV
summaries under `results/glm-sensitivity-distribution/`.

If you already have saved acceptance-floor sweep outputs and only want the
Pareto frontier for one estimator without rerunning optimization, use
`scripts/plot_saved_acceptance_floor_frontier.py`:

```bash
python scripts/plot_saved_acceptance_floor_frontier.py \
  results/glm-softmax-acceptance-floor-sweep
```

The script accepts either a direct `acceptance_floor_sweep.csv` path, a single
`acceptance_floor_frontier_<timestamp>/` directory, or the parent project output
directory. It defaults to `--estimator first_order` and writes
`pareto_objective_acceptance_first_order.png` plus
`pareto_u_acceptance_first_order.png` alongside the resolved CSV unless
`--output-dir` is provided.

To run the policy PCA-dimensionality grid over policy classes and policy-side
PCA dimensions, use:

```bash
python scripts/run_policy_pca_grid.py --n-samples 5000
```

This keeps the GLM black-box preprocessing sealed, fits configurable policy-side
preprocessors on the 19 acceptance-state columns, and writes aggregate finals,
traces, summary markdown, headline PCA/richness-gap plots, and final `u` /
acceptance-spread plots under `results/policy-pca-grid/`. The grid includes
linear-feature policies, matching softmax-wrapped feature policies, constant,
and MLP policies. The script prints per-condition progress by default; pass
`--quiet` to suppress progress output. Add `--constrained` to use `trust-constr`
with the observed GLM acceptance floor and a default 500-step cap. It accepts the
shared launch flags; `--launch slurm --array` runs one `(pca_dim, policy_class,
seed)` condition per array task.

To compare a candidate real-data CSV and its XGBoost artifacts with the
canonical dataset plus current GLM/XGBoost/spline models, use:

```bash
python scripts/analyze_real_data_model_artifacts.py \
  --candidate-dataset /path/to/df_raw_single_year_with_predictions_tp.csv \
  --candidate-artifact-dir /path/to/model_processing/artifacts
```

The analysis uses stored out-of-fold metrics as its primary performance
evidence and a deterministic 20,000-row common sample for descriptive
prediction and counterfactual-action diagnostics. It decodes covered sigmoid
curves from the trusted legacy smoothing pickle without invoking its
NumPy-incompatible inference method. Outputs include seven CSV tables, three
diagnostic plots, and `eda_summary.md` under a timestamped
`results/real-data-model-eda/` directory. This is analysis-only tooling: it
does not copy artifacts or change the runtime model registry.

To query the existing acceptance model at fixed constant actions without
running optimization, use:

```bash
python scripts/query_acceptance_at_u.py \
  --model-type glm \
  --u-count 101 \
  --n-rows 500
```

The script loads the preset objective and state batch, then reports mean
acceptance for each sampled `u` value. It writes acceptance curves and a
historical-`U` histogram with sampled constant-`u` rug marks under
`results/acceptance_queries/<model_type>/` by default. Use `--output-subdir`
to choose a subdirectory under `results/acceptance_queries/`, or pass explicit
values with `--u -0.3 0.0 0.2` instead of `--u-count`.

To benchmark GLM analytical acceptance speed, Stein-difference call counts,
objective-cache behavior, and contour-subsampling speed on the bundled real-data
objective, use:

```bash
python scripts/benchmark_experiment_speed.py --n-rows 1000 --grid-size 10
```

To evaluate a saved final policy under actual historical acceptance and observed
loss, use:

```bash
python scripts/evaluate_historical_policy_objective.py \
  --summary-json results/<slug>__<timestamp>/summary.json \
  --estimator first_order
```

The script reconstructs the saved real-data row sample from the run seed and
`n_samples`, computes final policy prices from the saved theta, and evaluates
`(1 - is_churn) * (Y_G_Loss - (u_policy + 1) * X_policy_premium)`. It prints the
theta used for manual verification and writes aggregate `summary.json` plus
row-level `per_row.csv` under `historical_policy_objective/<estimator>/` beside
the input summary by default.

For script-only checks that use historical CSV actions instead of optimized
policy actions, choose historical or model sources for acceptance and technical
price (`technical_price` is the loss term in the objective):

```bash
python scripts/evaluate_historical_policy_objective.py \
  --u-source historical \
  --model-type glm \
  --acceptance-source model \
  --technical-price-source historical

python scripts/evaluate_historical_policy_objective.py \
  --u-source historical \
  --model-type glm \
  --acceptance-source historical \
  --technical-price-source model

python scripts/evaluate_historical_policy_objective.py \
  --u-source historical \
  --model-type glm \
  --acceptance-source model \
  --technical-price-source model
```

All three evaluate
`acceptance_source(x,U_historical) * (technical_price_source(x) - (U_historical + 1) * X_policy_premium)`.
Use `--summary-json results/.../summary.json --split train|test|all` to reuse a
saved run's row sample; otherwise the script uses deterministic complete
eligible rows for `--model-type`. Outputs are written under
`historical_u_objective/<acceptance_...__technical_price_...>/`.

## Creating Config Presets

Prefer adding real-data variants through `get_config(..., overrides={...})`
rather than creating one-off preset files. Use
`src/experiments/configs/config_template.py` only for genuinely new synthetic or
non-real-data experiment families.

## Results

Each run writes artifacts under the shared external results root
(`~/projects/generali-pricing/results` by default):

- `results/<slug>__<timestamp>/` -- normal run directory
- `results/<project>/<variant>/seeds/seed-<seed>/` -- seed-sweep heavy artifacts
- `results/slurm/%x-%j.out` -- Slurm logs

Override the root with `GENERALI_RESULTS_ROOT`. Existing gitignored in-repo
`outputs/` directories are not migrated automatically.

Run artifacts include:

- `summary.json` -- full result payload
  including final trust-constr diagnostics such as `constraint_penalty`
  and any configured constant-`u` baseline evaluations; estimator results
  include both the mean objective `final_value` and summed objective
  `final_objective_sum`, plus `train` and optional `test` metric blocks;
  when a preset call site is known, `preset` records the preset name, variant,
  run seed, and serialized overrides
- `plots/optimization/steps.csv` -- per-step metrics for every estimator
- `plots/optimization/` -- loss curves, gradient norms, step sizes, and theta contour plots
- `plots/policy_train/` -- real-data final policy diagnostics on optimization rows
- `plots/policy_test/` -- real-data final policy diagnostics on held-out rows when configured

Weights & Biases integration is available for experiment tracking. See the
docstrings in `src/experiments/config.py` for W&B configuration fields.

## Adding a New Zeroth-Order Method

To add a new value-query estimator and run it through experiments:

1. Add a `GradientMethod` class in `src/optimization/gradients/methods.py`
   (follow `FiniteDifferenceGradient` / `GaussSteinGradient` /
   `SteinDifferenceGradient` / `SPSAGradient`).
2. Re-export it in `src/optimization/gradients/__init__.py`.
3. Add a solver wrapper in `src/optimization/solvers.py` that instantiates
   `Optimization(..., <YourGradientMethod>(), ...)`.
4. Register the estimator in `_ESTIMATOR_ORDER` and `_ESTIMATOR_SPECS` in
   `src/experiments/run.py`.
5. Register the estimator key in `src/experiments/config.py`
   (`allowed_estimators` in `ExperimentConfig.__post_init__`).
6. Add plot metadata in `src/reporting/visualization.py`
   (`ESTIMATOR_STYLES` and `_TRACE_ORDER`) so it renders in plots.

Finally, include your estimator key in `enabled_estimators=(...)` in a preset
under `src/experiments/configs/`.

## Reproducibility

The demo uses a fixed RNG seed (default 7, configurable per
`ExperimentConfig.seed`). For new runs, `ExperimentConfig.seed_setup` can split
that into explicit seed streams:

- `data_seed`: synthetic state sampling or real-data row subsampling.
- `split_seed`: train/test split permutation.
- `theta_seed`: random policy initialization when `theta0=None` or MLP theta is initialized.
- `noise_seed`: deterministic objective-noise surfaces such as `NoisyObjective`.
- `optimizer_seed`: mini-batches and stochastic gradient-estimator perturbations.

If `seed_setup` is omitted, all streams use the legacy `seed`. If a
`SeedSetup(run_seed=...)` is provided, any omitted stream is derived
deterministically from `run_seed`, while explicit stream seeds remain fixed
(seed logic lives in the `experiments.seeds` package). Per-estimator optimizer
RNG streams are independent of `enabled_estimators` ordering.

For seed-replicated sweeps with error bars, use the canonical
`experiments.sweep_utils.run_sweep(base_preset=..., run_seeds=(...), ...)`. It
replicates every variant across `run_seeds`; by default `vary=("theta",)` keeps
data, split, and noise identical across replicates and only reinitializes policy
`theta`. Each variant collects all its seeds under one folder
(`summary-seed-<seed>.json` plus aggregate error-bar plots and
`seed_grid_summary.csv`). A plain seed sweep is just the no-axis case. Note that
the stochastic estimators draw perturbations from `optimizer_seed`, so add
`"optimizer"` to `vary` to put error bars on their estimator noise, not just
initialization. The older `experiments.seed_repeats.run_seed_repeats(...)`
remains for its CSV-only output.

## Contributing

See `AGENTS.md` for development workflow, code organization, and testing
guidelines.
