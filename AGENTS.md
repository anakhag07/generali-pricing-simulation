# Agent Instructions

Project context: simulation and optimization repo. Users should be able to specify an experiment config and compare how different gradient methods perform across different objectives and policy types. Primary entry point is `main.py`.

## Core Working Rules

- Prefer small, focused changes with clear doc updates. Prior to making code changes, think about whether the code addition is necessary and if it keeps a clean, intuitive repo structure for the public api. 
- Keep simulation logic deterministic when a seed is set.
- When adding any new stochastic or nondeterministic process (sampling, splits,
  initialization, mini-batching, perturbations, randomized package calls, or
  parallelism), ask whether it needs its own seed stream in the experiment seed
  setup before implementation. Do not silently reuse an unrelated seed; document
  which seed controls the process.
- Include short comments or specs for functions when helpful.
- Prefer vectorized or cached computations when they preserve existing logic.
- Do not rely on prior chat context as the source of truth; repo context may be stale across terminals, worktrees, or later sessions.

## Logical Rules

- Prefer optimized math operations from online packages instead of implementing from scratch (for gradient descent methods, objective definitions, etc ...)

## Session Workflow

### Start of Session
Before editing code:

1. Read `AGENTS.md`.
2. Read `README.md`.
3. Inspect the relevant entry points and neighboring files.
4. Check recent tests related to the feature area.
5. Confirm the current branch and task scope.
6. **Check out a new branch before making any changes.** Never commit directly
   to `main`. Use a descriptive branch name (e.g., `feature/add-pca-tests`,
   `fix/armijo-edge-case`, `refactor/test-reorg`).
7. If working in a parallel terminal or worktree, assume other branches may have changed the repo recently and re-check branch state.
8. If repo structure or behavior appears inconsistent with `README.md` or `AGENTS.md`, update docs as part of the task.

### End of Session
Before finishing a build session:

1. Run relevant tests.
2. Run the demo if runtime behavior changed.
3. Update `README.md` if behavior, structure, configuration, or usage changed.
4. Update `AGENTS.md` if organization or workflow knowledge changed.
5. Leave a concise summary so another agent can continue from the branch without chat context.

## Modes

### Plan Mode
- Do not edit code.
- Propose implementation approach, file targets, and unit-test structure.
- Ask whether the proposed test structure is appropriate before implementing tests.
- For non-trivial changes, ask whether the user wants incremental commits during
  implementation. If the user approves, make small logical commits as work
  lands during build mode; otherwise leave changes uncommitted.
- Call out any expected `README.md` or `AGENTS.md` updates.

### Build Mode
- Make focused code changes.
- Keep docs in sync with implementation.
- Run validation commands after changes.
- Prepare a concise handoff summary suitable for a commit, PR, or later session.
- **Incremental commits for non-trivial changes.** When a change is larger than
  a few lines, check out a dedicated branch (if not already on one) and commit
  incrementally as logically distinct pieces of work land, rather than
  batching everything into a single end-of-session commit.
  - Prefer small, readable commits with short commit messages.
  - Group files into a commit only when they belong to the same logical change;
    if a change naturally splits by file or small file group, commit that way
    instead of batching unrelated edits together.
  - If the change is testable by the agent (unit tests, deterministic
    behavior, no manual UI/data inspection required), confirm with the user
    whether to open a PR once tests pass.
  - If the change likely needs manual testing by the user (runtime behavior,
    plots, real-data artifacts, W&B output), just commit to the branch and
    let the user drive the PR after their manual verification.

### Math Changes

When modifying or adding code that implements a mathematical formula:

1. **Identify the formula.** Find the corresponding entry in `MATH.md`. If none
   exists, add one before proceeding.
2. **Cross-reference.** Verify the implementation matches the `MATH.md` formula
   line-by-line. Pay attention to signs, index ranges, and normalization
   constants.
3. **Derive or verify.** For gradient changes, derive the gradient by hand from
   the value formula (or confirm it matches a textbook/paper reference). For
   estimator changes, confirm convergence properties (e.g., unbiasedness) hold.
4. **Suggest verification tests.** Propose 2–3 small tests the user can run to
   confirm correctness. Examples:
   - "Gradient should be zero at the known optimum u*."
   - "Finite-difference gradient should match analytical within 1e-5."
   - "Estimator variance should decrease as n_grad_samples increases."
   - "Transformed data should have identity covariance."
5. **Update MATH.md.** If the formula changed, update the `MATH.md` entry and
   the source docstring in the same commit.

## Source of Truth

When behavior and documentation disagree, use this priority:

1. `src/objective/objectives/fixed_regression.py`
2. Current implementation in the relevant source module
3. Tests
4. `MATH.md`
5. `README.md`
6. `AGENTS.md`

If lower-priority docs are stale, update them in the same task.

## Code Organization

Before adding code, inspect the surrounding module structure and choose the narrowest sensible location for the change.

Guidelines:
- Extend an existing module when responsibilities clearly match.
- Create a new file only when it introduces a reusable concept or prevents an existing file from becoming overloaded.
- Avoid scattering similar logic across multiple files.
- Prefer consistency with existing naming and folder conventions.
- When files or folders are added, moved, removed, or repurposed, record the change in `AGENTS.md`.

### Key Components

#### Objective Layer (`src/objective/`)

- **`src/objective/_math.py`** (private)
  - `_sigmoid(z)`: numerically stable vectorized sigmoid

- **`src/objective/base.py`**
  - `sample_states(rng, n, dim)`: sample n state vectors from N(0, I), returns (n, dim) array
  - `Policy`: batch-only policy interface (`value`, `grad`) operating on 2D arrays
  - `Objective`: theta-space interface class (`value`, `grad`)
  - `default_rng(seed)`: wrapper around `np.random.default_rng`

- **`src/objective/objectives/fixed_regression.py`** (source of truth for objective math)
  - `FixedRegressionObjective`: pricing objective $$f(u;x) = a(x,u)(\ell(x) - r(u))$$
  - `from_parameters` classmethod; batch evaluation via `value()`, `grad()`, `value_at_u()`

- **`src/objective/objectives/model_based.py`**
  - `ModelBasedObjective`: pricing objective $$f(u;x) = a(x,u)(\hat{Y}(x) - (u + 1) \cdot p(x))$$ backed by trained sklearn/XGBoost models
  - Takes `acceptance_model` / `loss_model` artifact bundles that can apply saved external preprocessing before calling the inner sklearn/XGBoost model
  - Owns the policy-side raw-to-processed bridge: raw `x_batch` may be a DataFrame and stays at the objective boundary; by default, the acceptance bundle's saved `FeatureProcessor` is reused internally for `u(theta, x)` and `du/dtheta`
  - Optional `policy_preprocessor` / `policy_feature_cols` decouple policy inputs from the sealed acceptance/loss artifact preprocessing; black-box model calls still receive raw `x` and use saved artifact preprocessors internally
  - Policy output `u` remains centered at 0 for the acceptance model; only the revenue term shifts to multiplier `u + 1`
  - Optional config-driven mean-acceptance floor is implemented either as a smooth penalty on the objective or directly as a SciPy `trust-constr` nonlinear constraint, depending on `step_rule`
  - Mutable diagnostic counters (`eval_counts()`, `reset_eval_counts()`) record objective value calls and acceptance/loss prediction rows for aggregate reporting
  - The canonical 052726 acceptance artifacts predict direct `p_accept` in class 1; do not flip to `1 - p_churn` for these artifacts
  - Uses extracted GLM/linear coefficients for array-native acceptance and loss predictions when available; XGB and unsupported artifacts fall back to estimator prediction calls
  - Caches policy features, GLM base logits, and loss predictions for repeated fixed `x_batch` arrays; final acceptance probabilities are not cached because they depend on `u`
  - `eval_counts()` also reports prediction/objective timing counters and cache hit/miss counters for performance diagnostics
  - `u_coef` sets the effective GLM acceptance coefficient on generated `U` for both values and analytical gradients; `None` uses the artifact coefficient or central FD for unsupported artifacts
  - `loss_source="observed"` keeps model-predicted acceptance but replaces the loss-model prediction with row-aligned historical `Y_G_Loss` carried on the real-data `x_fixed` DataFrame
  - `value()`, `grad()`, `value_at_u()`

- **`src/objective/objectives/planted_logistic.py`**
  - `PlantedLogisticObjective`: convex logistic objective with known optimum `u_star`
  - `optimal_u()` method exposes the planted optimum

- **`src/objective/policy.py`**
  - Implements `Policy` with batch methods `value(theta, x_batch)` and `grad(theta, x_batch)`
  - Feature-map classes: `IdentityFeatureMap`, `QuadraticFeatureMap`, `CubicFeatureMap`, `QuarticFeatureMap`, `CallableFeatureMap`; policies prepend the intercept internally, so custom maps return `varphi(x)`, not `[1, varphi(x)]`
  - `policy_theta_dim(policy, state_dim)`: helper for resolving theta dimension from the policy feature map
  - Concrete policies: `ConstantPolicy`, `LinearPolicy`, `SoftmaxPolicy`, `MLPPolicy`, `FeatureProcessedPolicy`
  - `SoftmaxPolicy`: bounded canonical sigmoid policy `action_low + (action_high - action_low) * sigmoid(theta^T phi(x))`; defaults to action range `(-0.5, 0.5)` and supports custom bounds such as `(-0.1, 0.2)`
  - `MLPPolicy`: two-layer MLP with `tanh` activations and bounded `0.5 - sigmoid(z)` head; default hidden width 16; flat theta layout `[W1, b1, W2, b2, W3, b3]`
  - `mlp_init_theta(rng, *, d_in, hidden)`: Glorot-uniform random init for `MLPPolicy.theta_dim`-sized theta (zero-init breaks because hidden units stay symmetric)
  - `policy_from_kind(kind)`: factory function (kinds: `constant`, `linear`, `softmax`, `mlp`)

- **`src/objective/policy_preprocessing.py`**
  - `PolicyFeaturePreprocessor`: policy-side fit-once standardization, whitening/sphering, and optional PCA truncation independent of black-box artifact preprocessing
  - `fit_policy_feature_preprocessor(...)`: convenience constructor for fitted policy preprocessors
  - `make_policy_features(...)`: applies a fitted policy preprocessor to raw policy-state rows
  - Full-state serialization via `to_state()` / `from_state(...)` persists fitted arrays such as means, scales, eigenvalues, and transform matrices for exact saved-policy replay

- **`src/objective/utils.py`**
  - `optimal_u(objective)`: public helper to extract u* from objective if available
  - Private helpers: `_theta_grad_from_u_grad`, `_mean_action`, `_action_value_at_u`

#### Data Layer (`src/data/`)

- **`src/data/dataset.csv`**
  - Canonical real-data source CSV used by all real-data loaders; the current file is the 052726 raw single-year export and both GLM/XGB configs use complete eligible rows from it

- **`src/data/dataset_metadata.py`**
  - Tracked source of truth for the 052726 dataset schema, objective X column groups, excluded target/action/lookahead columns, canonical CSV path, model artifact paths, and artifact/preprocessor notes
  - `USED_X_COLS` / `ACCEPTANCE_STATE_COLS` / `LOSS_FEATURE_COLS` are the only source covariates allowed into objective computation; `X_upcoming_premium`, historical `U`, `Y_G_Loss`, `is_churn`, IDs, and dates are excluded from objective values
  - Update this file whenever `src/data/dataset.csv`, model artifacts, model paths, column semantics, or dataset schema changes
  - When updating any `src/data` artifact or schema, confirm `dataset_metadata.py` matches the current CSV columns and model artifact paths, then run data-loader and model-artifact inference tests

- **`src/data/models/linear/`**
  - 052726 GLM/linear CV artifact pickles: acceptance classifier and financial-loss regressor; loader uses first fold and the fitted `FeatureProcessor`

- **`src/data/models/xgb/`**
  - 052726 XGBoost CV artifact pickles: acceptance classifier and financial-loss regressor; loader uses first fold and its fitted `FeatureProcessor`

- **`src/data/unused/`**
  - Legacy CSV/notebook exports not used by the current loader; retained only as temporary archive material before deletion

- **`src/data/feature_processor.py`**
  - `FeatureProcessor`: notebook-extracted whitening/PCA preprocessor used by the bundled real-data artifacts

- **`src/data/loader.py`**
  - `FEATURE_COLS_GLM` / `FEATURE_COLS_XGB`: 19-column 052726 objective X feature list shared by GLM and XGB configs; excludes lookahead `X_upcoming_premium`
  - `ACCEPTANCE_STATE_COLS`: 19 raw X cols passed to acceptance artifacts before fitted preprocessing; generated policy `U` is appended internally
  - `LOSS_FEATURE_COLS`: 18 raw X cols passed to loss artifacts before fitted preprocessing; excludes `X_policy_premium`
  - `dataset_csv_path()`: returns the canonical real-data source CSV path
  - `dataset_column_roles()`: reports used X cols, excluded lookahead X cols, target/action cols, and objective-excluded cols
  - `eligible_csv_row_indices(model_type)`: returns all complete eligible canonical dataset CSV row positions for full-row real-data bucket experiments
  - `sample_csv_row_indices(model_type, n_rows, seed)`: samples complete eligible canonical dataset CSV row positions without replacement for real-data configs
  - `load_x_frame(model_type, n_rows=5000, row_indices=None, seed=None)`: loads raw X covariates as a DataFrame, preserving categorical strings for artifact preprocessing
  - `load_x_array(...)`: compatibility wrapper returning the raw X frame as an object array; prefer `load_x_frame` for real-data configs
  - `load_observed_u_array(model_type, n_rows=5000, row_indices=None, seed=None)`: loads observed pricing multipliers from sampled canonical dataset rows for diagnostics and plots
  - `load_observed_loss_array(model_type, n_rows=5000, row_indices=None, seed=None)`: loads observed historical `Y_G_Loss` from sampled canonical dataset rows for observed-loss real-data objectives
  - `load_mean_observed_acceptance(model_type)`: computes `1 - is_churn` over complete eligible rows
  - `load_model_artifacts(model_type)`: loads first-fold `(acceptance_artifact, loss_artifact)` bundles from the 052726 CV dictionaries under `src/data/models/linear/` or `src/data/models/xgb/`
  - `ModelArtifactBundle.model_frame(raw_frame)`: converts raw notebook-space columns into the exact model-input frame expected by the bundled estimator
  - `extract_glm_u_coef(glm_pipeline)`: extracts effective d_logit(p_accept)/dU from the inner fitted GLM artifact for analytical gradient computation

#### Optimization Layer (`src/optimization/`)

- **`src/optimization/base.py`**
  - `Optimization`: class-based optimization entry point
  - `Optimization.solve(theta_start)`: dispatches to SciPy `minimize` for `step_rule="l-bfgs-b"` / `"trust-constr"` and to an internal manual gradient loop for `step_rule="constant"` / `"armijo"`; handles mini-batching, trace recording, and optional step-size history for manual rules
  - Uses separate RNG streams for mini-batch sampling (`batch_rng`) and stochastic gradient perturbations (`gradient_rng`); `rng` remains a backward-compatible fallback
  - Contains only constructor + solve orchestration; batching/objective helpers live in `src/optimization/helpers.py`
  - Solvers consume theta-level objectives only (`value(theta, x_batch)`, `grad(theta, x_batch)`)

- **`src/optimization/helpers.py`**
  - `scipy_method(...)`: maps configured algorithm string to SciPy method name
  - `sample_indices(...)`, `x_batch(...)`: mini-batch index/data helpers
  - `finite_difference_theta_grad(...)`: shared coordinate-wise finite-difference helper for value-only theta gradients
  - `objective_value_on_indices(...)`, `objective_grad_on_indices(...)`, `mean_action_on_indices(...)`: shared objective evaluation helpers used by optimizer + gradient methods

- **`src/optimization/gradients/methods.py`**
  - `GradientMethod`: base interface for pluggable gradient estimators
  - `FirstOrderGradient`: exact theta-gradient from `objective.grad(...)`
  - `FiniteDifferenceGradient`: deterministic central finite-difference theta-gradient from value queries
  - `GaussSteinGradient`: value-only theta-space Gaussian-Stein estimator
  - `SteinDifferenceGradient`: action-space Stein-SPSA hybrid estimator mapped through `policy.grad(...)`
  - `SPSAGradient`: two-sided SPSA theta-gradient estimator

- **`src/optimization/solvers.py`**
  - `run_constant_minimize(...)`, `run_first_order_minimize(...)`, `run_finite_difference_minimize(...)`, `run_gauss_stein_minimize(...)`, `run_stein_difference_minimize(...)`, `run_spsa_minimize(...)`: compatibility wrappers that instantiate `Optimization` with the corresponding gradient object and call `solve(...)`

- **`src/optimization/steps.py`**
  - `STEP_RULE_LBFGSB`, `STEP_RULE_TRUST_CONSTR`, `STEP_RULE_CONSTANT`, `STEP_RULE_ARMIJO`, `STEP_RULES`
  - `constant_step_size(step_size)`: returns the step size unchanged
  - `armijo_backtracking_step_size(...)`: Armijo line search utility used by the optimizer's manual `step_rule="armijo"` path

#### Experiment Layer (`src/experiments/`)

- **`src/experiments/config.py`**
  - `ExperimentConfig`: frozen dataclass with extensive `__post_init__` validation
  - Primary fields: `objective` (theta objective) and `theta0` (initial theta)
  - `x_fixed: np.ndarray | None = None`: when set, runner uses this 2D array as state batch instead of sampling from N(0, I)
  - `x_fixed_row_indices: np.ndarray | None = None`: source acceptance-CSV row positions for `x_fixed`; real-data configs pass this so observed-`U` reporting uses the same selected rows
  - `train_fraction` / `test_fraction`: deterministic run-level split fractions over selected rows; they must sum to `1.0`, `train_fraction` must be positive, and optimizers fit on train rows only
  - Objective/policy wiring is explicit; configs pass a concrete theta-level objective instance
  - `batch_size: int | None = None` enables stochastic mini-batch optimization when set
  - `acceptance_floor` can be enforced directly with `step_rule="trust-constr"` or via the smooth penalty path using `acceptance_penalty_weight` / `acceptance_penalty_temperature`
  - `lagrangian_lambda` enables the scalarized model-based target $$J(\theta) + \lambda(\text{floor} - \bar{a}(\theta))$$ on unconstrained step rules; experiment summaries still report the raw objective $$J(\theta)$$
  - `CorrectnessSpec`: controls how "true" gradients are computed (`"exact"`, `"numdiff"`, `"none"`)
  - `verbose: bool = False` controls terminal output of per-step metrics
  - Preset-composition helpers: `make_*_objective`, `make_softmax_policy`, `make_model_based_objective`,
    `canonical_training_block`, `canonical_runtime_block`, and `build_experiment_config`

- **`src/experiments/configs/`** (preset registry)
  - `__init__.py`: `get_config(name, overrides=None)` and `list_configs()` registry; real-data configs are exposed only as base presets plus overrides
  - `real_data_factory.py`: `build_real_data_config(...)` centralizes GLM/XGB artifact loading, row selection, policy construction, feature-order overrides, softmax action-bound overrides, policy-side no-PCA preprocessing, GLM-only `u_coef` acceptance overrides, acceptance-floor modes, estimator defaults, and theta initialization; omitted or `None` `n_samples` uses all complete eligible rows, while an integer `n_samples` samples without replacement
    - `loss_source="observed"` is an override axis that appends `Y_G_Loss` to the fixed real-data frame and configures `ModelBasedObjective` to use it as the loss term; default `"predicted"` keeps the artifact loss model
  - `first_order_runs_diff_starts.py`: planted-logistic preset configured for comparison runs across different initial starts
  - `fixed_regression_base.py`: base fixed-regression config (4D, L-BFGS-B step rule, W&B enabled)
  - `planted_logistic_base.py`: planted logistic base config (3D, L-BFGS-B step rule, 5000 steps, u*=1.1)
  - `real_data_glm_base`: registry-only base built by `real_data_factory.py`; supports `policy_kind`, `feature_order`, `policy_preprocessing`, `constraint_mode`, GLM acceptance `u_coef`, runtime, and estimator overrides
  - `real_data_xgb_base`: registry-only base built by `real_data_factory.py`; supports the same override axes, with XGB defaults excluding `first_order`
  - `config_template.py`: copy-first scaffold with `None` placeholders for all `ExperimentConfig` fields plus objective/correctness parameter blocks; not registered as a runnable preset

- **`src/experiments/defaults.py`**
  - `default_theta0(state_dim, policy=None)`: returns default initial theta sized by `policy_theta_dim(policy, state_dim)` when a policy is provided, otherwise `state_dim + 1`
  - `default_policy(state_dim)`: returns default `SoftmaxPolicy`

- **`src/experiments/helpers.py`** (largest file; orchestration + wrappers)
  - `resolve_true_grad_theta_fn(objective, correctness)`: resolves the "true" theta-gradient function from correctness spec
  - `run_constant(...)`: optimized `ConstantPolicy` baseline wrapper delegating to `optimization.solvers.run_constant_minimize`
  - `run_first_order(...)`: wrapper delegating to `optimization.solvers.run_first_order_minimize`
  - `run_finite_difference(...)`: wrapper delegating to `optimization.solvers.run_finite_difference_minimize`
  - `run_gauss_stein(...)`: wrapper delegating to `optimization.solvers.run_gauss_stein_minimize`
  - `run_stein_difference(...)`: wrapper delegating to `optimization.solvers.run_stein_difference_minimize`
  - `run_spsa(...)`: wrapper delegating to `optimization.solvers.run_spsa_minimize`
  - Uses `optimization.helpers.finite_difference_theta_grad(...)` for correctness-mode numerical theta gradients

- **`src/experiments/run.py`**
  - `run_experiment(config, step_reporter)`: main runner; uses `config.x_fixed` as state array when set, otherwise samples from N(0, I); applies the train/test split after row selection/sampling; runs enabled estimators on train rows; evaluates final policies on train and optional test rows; returns `ExperimentResult` (pure computation, no I/O)
  - `enabled_estimators` may include `"constant"`, which optimizes a one-scalar `ConstantPolicy` copy of the configured objective; `constant_u_baselines` remains fixed-action evaluation only

- **`src/experiments/sweep_utils.py`**
  - `expand_override_grid(...)`: cartesian product of override values
  - `apply_config_overrides(...)`: validates and applies top-level `ExperimentConfig` overrides
  - `generate_sweep_runs(...)`: expands a base preset into named sweep variants; real-data override grids may include factory axes such as `policy_kind` and `constraint_mode`
  - `run_preset_sweep(...)`: executes sweep variants through the standard reporter pipeline

- **`src/experiments/sensitivity_buckets.py`**
  - `median_observed_u(...)`: computes the median historical `U` over complete eligible rows
  - `glm_price_sensitivity_scores(...)`: scores rows by local GLM acceptance sensitivity $$|d p_{accept}(x, u_{ref}) / du|$$
  - `glm_price_derivative_matrix(...)`: vectorized customer-by-`u` signed GLM derivative matrix $$d p_{accept}(x, u) / du$$
  - `glm_price_sensitivity_matrix(...)`: vectorized customer-by-`u` GLM sensitivity matrix for action-grid diagnostics
  - `split_sensitivity_tertiles(...)`, `build_glm_sensitivity_buckets(...)`: split eligible rows into low/medium/high sensitivity buckets for bucketed real-data experiments

- **`src/experiments/policy_pca_grid.py`**
  - `PolicyPcaGridSpec`: configuration for the unconstrained GLM policy PCA-dimensionality grid
  - `run_policy_pca_grid(...)`: runs `pca_dim x policy_class x seed` conditions with configurable policy-side preprocessing while preserving raw-`x` black-box calls; policy classes include constant, linear/quadratic/third/fourth-order `LinearPolicy`, matching `SoftmaxPolicy` variants, and MLP
  - `write_policy_pca_outputs(...)`: writes aggregate finals/traces CSVs, summary markdown, PCA/richness-gap plots, and final `u`/acceptance spread plots

- **`src/experiments/results.py`**
  - `OptimizationTrace`: per-step trace with u values, objective values, gradient estimates, optional theta values, step sizes, and model-based mean-acceptance diagnostics
  - `EstimatorResult`: final theta, u, value, wall-clock time, and optional acceptance-constraint diagnostics
  - `PolicyEvaluation`: final policy metrics on a split (`objective_value`, `objective_sum`, mean/quantile `u`, and optional acceptance/loss/revenue diagnostics)
  - `ExperimentResult`: full result including config, train samples in `x_samples`, optional `x_test`, split row/index metadata, traces, final train/test policy metrics, and optional u_star

- **`src/experiments/seeding.py`**
  - `SeedSetup`: optional per-run seed-stream overrides (`run_seed`, `data_seed`, `split_seed`, `theta_seed`, `optimizer_seed`)
  - `resolve_seed_setup(...)`: legacy configs without `seed_setup` use `ExperimentConfig.seed` for every stream; explicit `SeedSetup` derives omitted streams from `run_seed`
  - `optimizer_rngs(...)`: derives order-independent per-estimator batch and gradient RNGs from `optimizer_seed`

- **`src/experiments/seed_repeats.py`**
  - `SeedRepeatSpec`: repeated-run orchestrator over explicit seed streams; default varies only `optimizer_seed` while fixing data/split/theta to the first run seed
  - `run_seed_repeats(...)`: runs a preset once per `run_seed`, writes normal per-run outputs plus `seed_repeats.csv` and `seed_repeats_summary.csv`

- **`src/experiments/policy_validation.py`**
  - Shared optimizer-independent policy validation helpers (`policy_u_values`, `evaluate_policy`) used by both `run_experiment()` and saved policy artifacts

- **`src/experiments/policy_artifacts.py`**
  - `PolicyArtifact`: reloadable trained-policy object saved as `policies/<estimator>/policy.json` plus `arrays.npz`
  - Separates policy input preprocessing (`raw x -> z`, including artifact preprocessing and optional policy-side whitening/PCA) from policy feature mapping (`z -> varphi(z) -> phi(z)`)
  - Persists theta, train/test/all CSV row bindings, objective/model metadata, policy head/feature-map specs, and full fitted policy-side preprocessing arrays
  - `load_policy_artifact(...).predict_u(split="train")` and `.evaluate(split="train")` rerun validation without optimizer training

- **`src/experiments/reporters.py`**
  - `RunContext`: frozen dataclass with experiment name, run directory paths, timestamp
  - `create_run_context(...)`: creates run directories under `outputs/` by default
  - `StepReporter`: protocol for per-step metric logging
  - `Reporter`: protocol with `on_start` and `on_end` hooks
  - `ReporterStack`: composite that delegates to a list of reporters; also implements `StepReporter`
  - `ConsoleReporter`: prints to terminal; per-step output controlled by `verbose`
  - `FileStepLogger`: writes per-step metrics to `plots/optimization/steps.csv`
  - `PolicyArtifactReporter`: writes reloadable trained-policy artifacts before `JsonReporter` records their relative paths
  - `JsonReporter`: writes `summary.json` on end, including estimator-level `train` and optional `test` policy metric blocks
  - `PlotReporter`: generates all matplotlib plots on end; optimization plots go under `plots/optimization/`, policy diagnostics go under `plots/policy_train/` and `plots/policy_test/`, and step-size plots are emitted whenever traces include `step_sizes`; writes per-plot timings to `plots/plot_timings.json`; theta contours for model-based objectives use a deterministic train-subsample capped at 200 rows and a 20x20 grid cap

#### Reporting Layer (`src/reporting/`)

- **`src/reporting/logging.py`**
  - `log_step(method, step, u, value, ...)`: prints one step to console
  - `log_summary(result)`: prints full experiment summary to console

- **`src/reporting/visualization.py`**
  - `ESTIMATOR_STYLES`: color/label config per estimator
  - `plot_loss_curves(...)`: objective vs step; optional |u - u*| subplot
  - `plot_gradient_norms(...)`: true theta gradient norms; optional error subplot
  - `plot_step_sizes(...)`: per-step step sizes (log scale y-axis)
  - `plot_objective_u_slice(...)`: objective vs u grid (no gradient subplot)
  - `plot_theta_objective_contours(...)`: 2D contour plot with optimization paths; use adaptive linear/log/symlog color scaling when objective ranges make a single linear scale unreadable
  - `plot_comparison_objective_curves(...)`, `plot_comparison_u_curves(...)`, `plot_comparison_final_metric(...)`: aggregate policy-comparison plots; final metrics render as grouped bars by policy with estimator colors and policy hatching
  - Model-based real-data run plots under `plots/policy_train/` and `plots/policy_test/` include `final_summary_metrics.png`, `u_histogram.png`, `acceptance_histograms.png`, `delta_u_histogram.png`, `delta_u_by_sensitivity.png`, `objective_contribution_summary.png`, and per-estimator `u_acceptance/<estimator>.png` files with binned mean acceptance, customer-level acceptance-vs-`u` scatter, and raw objective contribution histograms
  - Private sweep helpers power both lambda and trust-constrained acceptance-floor frontier plots
  - `select_theta_axes_max_variance(...)`: picks the two theta axes with highest variance for contour plots

### Entry Point (`main.py`)

- Reads `RUN_CONFIGS` list; entries may be a config name or `(config_name, overrides)` tuple passed to `get_config(config_name, overrides=overrides)`
- For each config spec: creates `RunContext`, assembles `ReporterStack`, calls `run_experiment()`, finalizes with `reporters.on_end()`
- All I/O is handled by reporters, not by the runner
- `scripts/run_sweep.py` provides optional preset-based sweep execution using top-level and real-data factory overrides
- `scripts/run_lagrangian_sweep.py` runs a lagrangian-lambda sweep and writes aggregate frontier plots under `outputs/<project>/lagrangian_frontier_<timestamp>/`
- `scripts/run_acceptance_floor_sweep.py` runs the trust-constrained softmax GLM preset over a dense acceptance-floor grid `c` and writes aggregate frontier plots under `outputs/<project>/acceptance_floor_frontier_<timestamp>/`
- `scripts/run_glm_u_coef_sweep.py` runs the softmax/no-PCA/trust-constr GLM setup over 200000 sampled rows and `u_coef in {-4, -5, -8, -10, -20}`, keeps per-run distribution plots enabled, and writes aggregate `glm_u_coef_sweep.csv` plus frontier plots under `outputs/glm-u-coef-sweep/u_coef_frontier_<timestamp>/`
- `scripts/run_glm_softmax_alpha_sweep.py` runs the trust-constrained softmax/no-PCA/linear-feature GLM setup over symmetric action bounds `[-alpha, alpha]` for `alpha in {0.5, 0.4, 0.3, 0.2, 0.15, 0.125, 0.1, 0.075}`, saves normal per-alpha policy artifacts, and writes aggregate final objective/profit plots plus artifact-replayed acceptance-threshold and per-alpha `u`-bin expected-profit summaries under `outputs/glm-softmax-alpha-sweep/alpha_sweep_<timestamp>/`
- `scripts/run_glm_sensitivity_bucket_experiment.py` buckets all complete eligible GLM rows into low/medium/high local price-sensitivity tertiles at median observed `U`, runs the softmax/no-PCA/trust-constr GLM setup on every row in each bucket, keeps per-run distribution plots enabled, and writes aggregate `glm_sensitivity_bucket_experiment.csv` plus comparison plots under `outputs/glm-sensitivity-buckets/sensitivity_bucket_summary_<timestamp>/`
- `scripts/run_glm_reference_elasticity_bucket_experiment.py` repeats the GLM bucket experiment for reference actions `u_ref in {-0.1, 0.1, 0.2, 0.3}`, ranks rows by elasticity magnitude at each reference action, runs only `first_order`, annotates summary charts with average bucket elasticity magnitude, and writes per-reference summaries under `outputs/glm-reference-elasticity-buckets/`
- `scripts/plot_glm_sensitivity_distribution.py` computes GLM customer elasticities $$d p_{accept}(x, u) / du$$ over a default `u in [-0.3, 0.3]` grid, writes a mean/quantile elasticity-by-`u` curve, selected-`u` elasticity histograms for `{-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3}` with default `0.5-99.5%` x-axis clipping marked, and CSV summaries under `outputs/glm-sensitivity-distribution/`
- `scripts/diagnose_low_sensitivity_policy_acceptance.py` rebuilds GLM sensitivity buckets, applies either a manual softmax `--theta` or exact saved-policy replay via `--policy-artifact outputs/.../policies/<estimator>/policy.json`, and writes row-level `row_index`, policy-feature, GLM acceptance-feature, policy-score, and acceptance-logit diagnostics plus histograms under `outputs/low-sensitivity-policy-acceptance-diagnostics/`; use `--bucket all` for low/medium/high, `--bucket-u-ref` to choose the scoring action, and `--bucket-row-source artifact-all|artifact-train|artifact-test` to form buckets within saved artifact rows
- `scripts/plot_policy_acceptance_grid.py` loads a saved policy artifact, scores artifact-bound rows by mean absolute acceptance sensitivity over a simulated `u` grid and by predicted loss, randomly samples clients from low/medium/high tertiles for each score, and writes two three-panel client-level acceptance-curve plots plus `sampled_clients.csv` under `outputs/policy-acceptance-grid/`; omit `--seed` to resample clients each run
- `scripts/plot_saved_acceptance_floor_frontier.py` re-plots acceptance-floor Pareto frontiers from a saved `acceptance_floor_sweep.csv` (or the latest matching frontier directory) without rerunning optimization; defaults to `first_order` and writes estimator-suffixed Pareto PNGs
- `scripts/query_acceptance_at_u.py` loads a config preset or default GLM/XGB model type and reports mean acceptance for supplied or evenly sampled constant `u` values without running optimization; writes acceptance-curve and historical-`U` rug plots under `outputs/acceptance_queries/` by default and optionally writes `u,n,mean_acceptance` CSV output
- `scripts/plot_pc_outcome_diagnostics.py` reads a saved run `summary.json`, rebuilds a real-data base preset objective with optional policy/preprocessing override flags, and writes processed-policy-component scatter diagnostics against final `f_acc`, loss, and `u`; defaults beside the summary under `pc_outcome_diagnostics/<estimator>/`
- `scripts/evaluate_historical_policy_objective.py` reads a saved run `summary.json`, reconstructs selected CSV row positions from full-eligible mode or seed/`n_samples`, prints the estimator theta used, and evaluates final policy prices under historical acceptance `1 - is_churn` and observed `Y_G_Loss`; writes aggregate `summary.json` and row-level `per_row.csv` under `historical_policy_objective/<estimator>/`
  - Prefer `--policy-artifact outputs/.../policies/<estimator>/policy.json` for new runs; `--summary-json` remains a legacy fallback for outputs created before policy artifacts existed
  - Supports `--objective model` to replay the trained model objective and `--objective historical` for the observed-outcome diagnostic; both support `--split train|test|all`, where `all` means all selected run rows before splitting
- `scripts/plot_glm_data_tsne.py` samples rows from the GLM real-data CSV, runs a standardized t-SNE/KMeans feature diagnostic, and writes embedding CSV plus color-by-feature plots under `outputs/data-tsne/`
- `scripts/run_policy_pca_grid.py` runs the GLM policy PCA-dimensionality grid over configured PCA dimensions and policy classes `(constant, linear, quadratic, third_order, fourth_order, softmax_linear, softmax_quadratic, softmax_third_order, softmax_fourth_order, mlp)`; unconstrained is default, `--constrained` uses `trust-constr` with the observed GLM acceptance floor and a 500-step default cap; outputs aggregate CSVs, summary markdown, and headline/spread plots under `outputs/policy-pca-grid/`; prints per-condition progress by default and supports `--quiet`
- `scripts/benchmark_experiment_speed.py` benchmarks GLM analytical acceptance vs sklearn `predict_proba`, Stein-difference gradient timing/call counts, repeated objective-cache behavior, and full-vs-subsampled contour grid timing; use it to quantify whether performance changes speed up real-data diagnostics without relying on flaky pytest time thresholds

## Known Issues and Dead Code

These are documented here so agents can account for them and clean them up
when appropriate.

- **`plot_dir` on `ExperimentConfig` is ignored by `PlotReporter`:**
  `PlotReporter` always uses `run_context.plots_dir` (which is
  `run_dir / "plots"`). The config field is only serialized.

## Testing

- Add or update small, focused unit tests for each change.
- Keep tests deterministic with explicit seeds.
- Avoid plotting, filesystem I/O, or long-running simulations in tests.
- Prefer testing pure functions and small components.
- Keep tests fast.
- Tests are organized into subdirectories mirroring `src/`:
  - `tests/objective/` — objective, policy, and math utility tests
  - `tests/optimization/` — gradient estimator, step rule, and helper tests
  - `tests/data/` — loader and feature processor tests
  - `tests/experiments/` — config, runner, and sweep tests
  - `tests/reporting/` — visualization and logging tests
  - `tests/integration/` — end-to-end tests
- Place new tests in the subdirectory matching the module under test.

### Current Test Coverage

#### `tests/objective/`
| Test File | Area |
|---|---|
| `test_math.py` | `_sigmoid` stability, symmetry, monotonicity, derivative |
| `test_utils.py` | `_theta_grad_from_u_grad` chain rule, `optimal_u`, `_action_value_at_u` |
| `test_objective_models.py` | FixedRegressionObjective value and gradient correctness |
| `test_objective_batch.py` | Deterministic objective private batch helpers and `value_at_u` |
| `test_objective_package_exports.py` | objective package API exports remain importable |
| `test_planted_logistic_objective.py` | Planted logistic gradient at u_star and minimum |
| `test_model_based_objective.py` | `value()`, `grad()` shape, `value_at_u()`, analytical vs FD grad agreement |
| `test_policy_batch.py` | Policy batch `value/grad` shapes, bounds, and kind labels (incl. `MLPPolicy`) |
| `test_mlp_policy_grad.py` | `MLPPolicy.grad` matches per-coordinate FD Jacobian; `mlp_init_theta` symmetry-breaking |
| `test_policy_preprocessing.py` | Policy-side standardization, whitening, PCA dimensionality, and transform validation |
| `test_policy_u_histograms.py` | Policy u-distribution visualization |

#### `tests/optimization/`
| Test File | Area |
|---|---|
| `test_helpers.py` | `_clamp_theta`, `sample_indices`, `x_batch`, `finite_difference_theta_grad` |
| `test_gradient_methods_math.py` | Gauss-Stein, SPSA, Stein-Difference convergence; FD u-space vs theta-space; SPSA variance |
| `test_step_rules.py` | Armijo sufficient decrease, edge cases, input validation |
| `test_finite_difference_gradient.py` | Finite-difference gradient accuracy and determinism |
| `test_gradient_resampling.py` | Gradient method resampling behavior |
| `test_optimization_class.py` | Class-based optimizer entry point and gradient-object behavior |
| `test_minibatch_stochasticity.py` | Mini-batch determinism and full-batch equivalence |
| `test_minimize_orders.py` | SciPy first/Gauss-Stein/Stein-difference/SPSA wrappers |
| `test_early_stopping.py` | grad_norm_tol early stopping |
| `test_trust_constr_constraint.py` | Trust-region constraint acceptance floor |

#### `tests/data/`
| Test File | Area |
|---|---|
| `test_data_loader.py` | `load_x_frame` shape/columns, model artifact types, U normalization, CSV column sets |
| `test_dataset_metadata.py` | Canonical dataset metadata matches CSV schema and artifact metadata |
| `test_feature_processor.py` | Centering, sphering, PCA whitening, inverse transform, categorical encoding |
| `test_model_artifact_inference.py` | GLM/XGB artifact inference and model-based objective smoke tests on canonical rows |

#### `tests/experiments/`
| Test File | Area |
|---|---|
| `test_config.py` | ExperimentConfig validation rules |
| `test_config_template.py` | Config template scaffold |
| `test_correctness_spec.py` | CorrectnessSpec gradient source modes |
| `test_experiment_configs.py` | Config registry (get_config, list_configs) |
| `test_real_data_config.py` | All real-data presets load; x_fixed shape; correct estimator sets |
| `test_enabled_estimators.py` | Selective estimator execution |
| `test_verbose_config.py` | verbose flag defaults and serialization |
| `test_baseline_test.py` | End-to-end smoke test with fixed_regression_base overrides |
| `test_run_context.py` | default output directory and run context paths |
| `test_train_test_split.py` | Runner train/test split, held-out policy metrics, and summary payloads |
| `test_seeding.py` | Seed setup serialization, data/split/theta stream routing, estimator-order independence |
| `test_seed_repeats.py` | Seed-repeat setup construction and aggregate CSV outputs |
| `test_sweep_utils.py` | Override-grid expansion and preset sweep config generation |
| `test_sensitivity_buckets.py` | GLM local price-sensitivity scoring and tertile construction |
| `test_sensitivity_bucket_script.py` | Sensitivity bucket experiment script constants and summaries |
| `test_softmax_alpha_sweep_script.py` | Softmax alpha sweep constants, artifact-replayed profit bins, and aggregate output plots |
| `test_plot_glm_sensitivity_distribution_script.py` | GLM elasticity distribution script summaries and plot outputs |
| `test_reference_elasticity_bucket_script.py` | Reference-u GLM elasticity bucket script constants and plot outputs |
| `test_policy_pca_grid.py` | Policy PCA grid condition construction and aggregate output writing |

#### `tests/reporting/`
| Test File | Area |
|---|---|
| `test_logging.py` | Step logging output format |
| `test_file_step_logger.py` | FileStepLogger CSV output |
| `test_split_plot_folders.py` | Split policy plot folder creation |
| `test_wandb_reporter.py` | W&B reporter integration |
| `test_reporting_theta_norms.py` | Theta norm visualization |
| `test_plot_u_star.py` | u_star selection for plotting |
| `test_lagrangian_sweep_plots.py` | lambda-wrapper and generic sweep frontier plot generation |
| `test_theta_contours.py` | Contour grid shapes, axis selection |
| `test_trace_theta_values.py` | theta_values recorded in traces |
| `test_visualization_step_sizes.py` | step_sizes plot uses log y-scale |
| `test_visualization_styles.py` | Estimator style configuration |
| `test_policy_delta_u_elasticity.py` | Delta-u histogram and reference sensitivity policy diagnostic plots |
| `test_policy_objective_contribution_summary.py` | Customer objective/profit spread diagnostic plot |
| `test_comparison_plots.py` | aggregate comparison curve plots and grouped final-metric bar charts |

#### `tests/integration/`
| Test File | Area |
|---|---|
| `test_state_vector.py` | `sample_states` shape and dtype |

## Documentation and Maintenance

### README.md
Keep README concise—a quick-start and overview only. Update when:
- setup or execution steps change
- new user-facing features are added
- basic workflow changes

Point users to `docs/` for detailed API reference.

### docs/ (pdoc-generated)
Regenerate docs when public API changes.

**Regeneration command:**

```bash
conda activate simulation_env
pdoc --math src/objective src/optimization src/experiments src/data src/reporting -o docs
```

Run this command whenever docstrings are added, updated, or removed in any
public class or function under `src/`. The command regenerates HTML for all
modules at once so cross-module links stay consistent.

Major classes and methods MUST have docstrings that render via pdoc:
- Objective classes (`FixedRegressionObjective`, `PlantedLogisticObjective`, `ModelBasedObjective`)
- Policy classes (`ConstantPolicy`, `LinearPolicy`, `SoftmaxPolicy`)
- Core interfaces (`Policy`, `Objective`) and sampling helper (`sample_states`)
- Optimization classes (`Optimization`, `GradientMethod` subclasses)
- Experiment config and results (`ExperimentConfig`, `ExperimentResult`, etc.)
- Runner functions (`run_experiment`)
- Data loaders (`load_x_array`, `load_model_artifacts`)

Docstrings should be 1-2 lines with LaTeX where it aids clarity.
Use double delimiters `$$...$$` for math rendering in pdoc (do not use single `$...$`).
Private helpers and internal utilities do not require docstrings.

Update math formulas in docstrings when objective definitions change.

### AGENTS.md
Update `AGENTS.md` when:
- files or folders are added, moved, removed, or repurposed
- module responsibilities change
- development workflow changes
- new recurring pitfalls or lessons are discovered
- new public entry points or reporting paths are introduced
- dead code or known issues are resolved

Do not leave durable organizational knowledge only in code diffs.

### Other Maintenance
- Update `requirements.txt` when dependencies change.
- Re-export public APIs in package `__init__.py` files when modules are added or moved.

## Validation

Always activate the environment before running tests or the demo.

Run the demo after changes:

```bash
conda activate simulation_env
python main.py
```

Run tests:

```bash
conda activate simulation_env
pytest -q
```
