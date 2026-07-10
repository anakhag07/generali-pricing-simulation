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

## Pull Request Reviews

- When asked to review a pull request, read `REVIEW.md` first and follow its
  architecture, workflow, docs, interface, and output-format checklist.

## Skills

- `~/skills/research-report/SKILL.md`: shared post-run/post-sweep analysis skill —
  summarize results from existing `summary.json`/CSV outputs, suggest
  pareto/sweep-axis diagnostic plots, and (when prompted) write reports to
  `results/agent-reports/`. Use it after a feature lands or a sweep finishes.

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

### Worktree Workflow

- Treat `/home/anakhag/projects/generali-pricing/generali-pricing-simulation`
  as the canonical/main checkout for this repo.
- For large feature additions or parallel feature work, create a dedicated git
  worktree before editing so multiple branches can be developed independently.
  Small documentation edits, tiny fixes, or follow-ups on an already appropriate
  feature branch do not require a new worktree.
- Keep feature worktrees outside the canonical checkout under the parent-level
  convention
  `/home/anakhag/projects/generali-pricing/worktrees/<branch-slug>`.
  Do not create nested worktrees inside the repo checkout or under an extra
  `worktrees/generali-pricing-simulation/` directory.
- Create the parent worktree directory only when a feature worktree is actually
  needed. From the canonical checkout, use a filesystem-safe slug that mirrors
  the branch name, for example:

```bash
mkdir -p ../worktrees
git worktree add ../worktrees/feature-policy-grid -b feature/policy-grid
```

- If the branch already exists, omit `-b` and pass the existing branch name.
- After entering a worktree, re-read `AGENTS.md`, confirm the branch/status, and
  assume other worktrees may have changed the repo recently.
- Gitignored data artifacts (`src/data/dataset.csv`, `src/data/models/**/*.pkl`)
  exist only in the canonical checkout, so real-data tests fail in a fresh
  worktree until they are symlinked in:

```bash
ln -sfn <canonical>/src/data/models <worktree>/src/data/models
ln -sf <canonical>/src/data/dataset.csv <worktree>/src/data/dataset.csv
```

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
- In plan mode, classify whether a task is `worktree-required` before editing.
  Use an external worktree for changes that add a new reusable `src/` module or span infrastructure areas such as seeding, config, runner behavior, package exports, or generated docs; a feature branch in the canonical checkout is not sufficient for those tasks.
- Propose implementation approach, file targets, and unit-test structure.
- Ask whether the proposed test structure is appropriate before implementing tests.
- For non-trivial changes, ask whether the user wants incremental commits during
  implementation. If the user approves, make small logical commits as work
  lands during build mode; otherwise leave changes uncommitted.
- Call out any expected `README.md` or `AGENTS.md` updates.
- For any feature that integrates with core pipeline logic, write a checklist
  plan before build mode. The plan must cover user-facing goals, implementation
  questions, integration points, the lowest-code viable design, public API/docs
  exposure, test strategy, seed/determinism implications, data/model/reporting
  effects, and migration or compatibility concerns.
- Reread core feature plans through a system-design lens before implementation:
  verify that responsibilities stay in the right modules, abstractions are not
  added prematurely, public surfaces are intentional, and the change does not
  bypass existing extension points.
- For large feature changes or code additions, scan for dead code that the new
  design would replace. In plan mode, suggest specific removals and explain why;
  only remove dead code in build mode after the user approves the removal.

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

1. Current implementation in the relevant source module
2. Tests
3. `MATH.md`
4. `README.md`
5. `AGENTS.md`

If lower-priority docs are stale, update them in the same task.

## Code Organization

Before adding code, inspect the surrounding module structure and choose the narrowest sensible location for the change.

Guidelines:
- Classify new work before choosing files. One-off or ad-hoc analyses belong in
  `scripts/`; reusable behavior that changes the experiment pipeline belongs in
  `src/` only after core-feature planning.
- `scratch/` is a tracked (committed) area for drivers of concluded experiments
  and short-lived probes. Retire a `scripts/` driver here once its experiment is
  done rather than deleting it, and drop its `scripts/` docs entries — `scratch/`
  contents are intentionally left out of the AGENTS/README scripts docs. Promote
  back to `scripts/` only if it becomes reusable tooling again.
- `scratch/plot_planted_logistic_support_bias_diagnostics.py` reads a completed
  planted-logistic support-bias sweep CSV/summaries and writes post-run true-gap
  bar charts plus constant-action oracle-vs-biased objective slices; it never
  reruns optimization.
- Keep the boundary strict: do not hide reusable pipeline logic inside a script,
  and do not promote analysis-only code into `src/` without a concrete reusable
  integration point.
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
  - `Policy`: batch-only policy interface (`value`, `grad`, `weighted_grad`) operating on 2D arrays; `weighted_grad` is the VJP-style fast path for `sum_i weights_i * d pi_theta(x_i)/dtheta`
  - `Objective`: theta-space interface class (`value`, `grad`)
  - `default_rng(seed)`: wrapper around `np.random.default_rng`

- **`src/objective/noise.py`**
  - `ObjectiveNoise`: interface for deterministic additive action-level noise fields $$\delta(x,u)$$
  - `HomoskedasticGaussianNoise`: standard/constant-std Gaussian noise keyed by exact `(x, u, seed)`, so repeated evaluations of the same row/action pair return the same noise
  - `HeteroskedasticGaussianNoise`: Gaussian noise sharing the same keyed unit-normal field with std $$\sigma_0 + \gamma\,|u - u_c|$$ growing with action distance from `u_center` (typically the planted optimum), so queries near the global minimum stay nearly noiseless; `growth=0` reproduces the homoskedastic adapter exactly
  - `NoisyObjective`: wraps any objective as $$\hat{M}(x,u)=M(x,u)+\delta(x,u)$$ for value-based optimization; intentionally raises for analytical `grad()` because the noisy objective has no analytical gradient; delegates acceptance metrics/constraints to the base objective and exposes clean `base_value()` / `base_value_at_u()` for reporting
  - `NoNoise`: zero-noise adapter for tests or disabled noise wiring

- **`src/objective/objectives/fixed_regression.py`** (source of truth for objective math)
  - `FixedRegressionObjective`: pricing objective $$f(u;x) = a(x,u)(\ell(x) - r(u))$$
  - `from_parameters` classmethod; batch evaluation via `value()`, `grad()`, `value_at_u()`

- **`src/objective/objectives/biased.py`**
  - `ActionBias`: action-level deterministic bias interface used by `BiasedObjective`; no additional seed stream is needed
  - `LinearActionBias`: global optimism wrapper term $$b(u)=-\lambda_{bias}u$$
  - `UpperSupportHingeBias`: upper-support optimism term $$b(u)=-\lambda_{bias}(u-h)_+$$ with optional smooth hinge; exact inside support and optimistic only above support
  - `BiasedObjective`: deterministic wrapper $$\hat{M}(x,u)=M(x,u)+b(x,u)$$ that exposes biased optimization values/gradients while `base_value()` / `base_value_at_u()` report the wrapped true objective

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

- **`src/objective/objectives/prepared_glm.py`**
  - `PreparedGLMBatch`: compact numeric GLM batch with columns `[base_logit, loss, premium, policy_features...]`
  - `PreparedGLMObjective`: pure NumPy GLM objective over prepared batches; no pandas/sklearn calls in the hot path
  - `prepare_glm_batch(...)` / `prepare_glm_objective(...)`: materialize a GLM-backed `ModelBasedObjective` after artifact preprocessing has run once

- **`src/objective/objectives/jax_prepared_glm.py`**
  - `JaxPreparedGLMObjective`: fixed-batch JAX version of the prepared GLM objective for SciPy callback use; transfers prepared arrays to device once and exposes NumPy-returning `value()`, `grad()`, value-only action hooks, `mean_acceptance()`, and `mean_acceptance_grad()` methods
  - `JaxPreparedGLMScipyAdapter`: explicit callback adapter with objective, gradient, constraint-margin, and constraint-Jacobian shapes for validation/benchmarking
  - `prepare_jax_glm_objective(...)`: materializes a GLM-backed `ModelBasedObjective` into a JAX objective after CPU artifact preprocessing; currently supports fixed full-batch first-order and zeroth-order GLM trust-constr/optax runs with constant policies plus linear/softmax policies over finite materializable feature maps, including built-in higher-order maps and `CallableFeatureMap`; the expanded policy design matrix is materialized once before device transfer
  - `MLPPolicy` is also supported on the JAX backend: theta is unpacked into `[W1, b1, W2, b2, W3, b3]` inside a jitted forward pass on the mapped features, so value/grad/mean_acceptance flow through autodiff (works with `first_order`, `finite_difference`, `gauss_stein`, `spsa`, `optax-adam`, and `trust-constr`). The action-space policy Jacobian (`policy_grad`/`policy_weighted_grad`, used only by `stein_difference` and diagnostics) is intentionally not materialized for MLP and raises `NotImplementedError`

- **`src/objective/objectives/planted_logistic.py`**
  - `PlantedLogisticObjective`: convex logistic objective with known optimum `u_star`
  - `optimal_u()` method exposes the planted optimum

- **`src/objective/policy.py`**
  - Implements `Policy` with batch methods `value(theta, x_batch)`, `grad(theta, x_batch)`, and `weighted_grad(theta, x_batch, weights)`
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
  - `Optimization.solve(theta_start)`: dispatches to SciPy `minimize` for `step_rule="l-bfgs-b"` / `"trust-constr"`, to the optax update loop for `step_rule="optax-adam"` / `"optax-sgd"`, and to an internal manual gradient loop for `step_rule="constant"` / `"armijo"`; handles mini-batching, trace recording, and optional step-size history for manual and optax rules
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
  - `STEP_RULE_LBFGSB`, `STEP_RULE_TRUST_CONSTR`, `STEP_RULE_CONSTANT`, `STEP_RULE_ARMIJO`, `STEP_RULE_OPTAX_ADAM`, `STEP_RULE_OPTAX_SGD`, `OPTAX_STEP_RULES`, `STEP_RULES`
  - `constant_step_size(step_size)`: returns the step size unchanged
  - `armijo_backtracking_step_size(...)`: Armijo line search utility used by the optimizer's manual `step_rule="armijo"` path

- **`src/optimization/optax_loop.py`**
  - `optax_step_rule_optimizer(algorithm, step_size)`: maps `"optax-adam"` / `"optax-sgd"` to the optax gradient transformation with `step_size` as learning rate
  - `run_optax_minimize_loop(...)`: manual optax update loop mirroring the constant/armijo path — mini-batch via `batch_rng`, theta grads from the configured `GradientMethod` (all estimators work), `grad_norm_tol` early stopping, per-step trace recording; deterministic given the gradient stream, so no new seed stream is needed
  - `optax.lbfgs` is intentionally unsupported: its linesearch requires a JAX-traceable value_fn, which NumPy objectives cannot provide; acceptance floors on optax rules go through the existing penalty/Lagrangian objective paths

#### Experiment Layer (`src/experiments/`)

- **`src/experiments/config.py`**
  - `ExperimentConfig`: frozen dataclass with extensive `__post_init__` validation
  - Primary fields: `objective` (theta objective) and `theta0` (initial theta)
  - `compute_backend`: `"numpy"` by default; `"jax"` swaps GLM training callbacks to the fixed-batch JAX prepared objective for parity/speed experiments across `first_order`, `finite_difference`, `gauss_stein`, `spsa`, and `stein_difference`; it supports `step_rule="trust-constr"` (SciPy driver) or the optax rules `"optax-adam"` / `"optax-sgd"`; JAX runs require `batch_size=None`
  - `x_fixed: np.ndarray | None = None`: when set, runner uses this 2D array as state batch instead of sampling from N(0, I)
  - `x_fixed_row_indices: np.ndarray | None = None`: source acceptance-CSV row positions for `x_fixed`; real-data configs pass this so observed-`U` reporting uses the same selected rows
  - `train_fraction` / `test_fraction`: deterministic run-level split fractions over selected rows; they must sum to `1.0`, `train_fraction` must be positive, and optimizers fit on train rows only
  - Objective/policy wiring is explicit; configs pass a concrete theta-level objective instance
  - `batch_size: int | None = None` enables stochastic mini-batch optimization when set
  - `acceptance_floor` can be enforced directly with `step_rule="trust-constr"` or via the smooth penalty path using `acceptance_penalty_weight` / `acceptance_penalty_temperature`
  - `lagrangian_lambda` enables the scalarized model-based target $$J(\theta) + \lambda(\text{floor} - \bar{a}(\theta))$$ on unconstrained step rules; experiment summaries still report the raw objective $$J(\theta)$$
  - `CorrectnessSpec`: controls how "true" gradients are computed (`"exact"`, `"denoised_exact"`, `"numdiff"`, `"none"`); `denoised_exact` uses the wrapped clean objective gradient for `NoisyObjective`
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
    - `compute_backend="jax"` keeps `step_rule="trust-constr"` and swaps supported GLM training callbacks to the fixed-batch JAX prepared objective; use with constant/linear/softmax policies, finite materializable linear/softmax feature maps, and supported estimators `first_order`, `finite_difference`, `gauss_stein`, `spsa`, and `stein_difference`
  - `real_data_xgb_base`: registry-only base built by `real_data_factory.py`; supports the same override axes, with XGB defaults excluding `first_order`
  - `config_template.py`: copy-first scaffold with `None` placeholders for all `ExperimentConfig` fields plus objective/correctness parameter blocks; not registered as a runnable preset

- **`src/experiments/initialization.py`**
  - `random_theta0(state_dim, policy, rng)`: generates policy-aware random initial theta when configs request random initialization

- **`src/experiments/correctness.py`**
  - `TrueThetaGradFn`: callable type alias for theta-gradient diagnostics
  - `resolve_true_grad_theta_fn(objective, correctness)`: resolves the "true" theta-gradient function from correctness spec
  - Uses `optimization.helpers.finite_difference_theta_grad(...)` for correctness-mode numerical theta gradients

- **`src/experiments/run.py`**
  - `run_experiment(config, step_reporter)`: main runner; uses `config.x_fixed` as state array when set, otherwise samples from N(0, I); applies the train/test split after row selection/sampling; runs enabled estimators on train rows; evaluates final policies on train and optional test rows; returns `ExperimentResult` (pure computation, no I/O)
  - Owns estimator dispatch through `_ESTIMATOR_ORDER` / `_ESTIMATOR_SPECS`, which map enabled estimator keys to `optimization.solvers` wrappers
  - `enabled_estimators` may include `"constant"`, which optimizes a one-scalar `ConstantPolicy` copy of the configured objective; `constant_u_baselines` remains fixed-action evaluation only
  - `compute_backend="jax"` converts supported GLM train batches to `JaxPreparedGLMObjective` before optimizer execution and requires `trust-constr` with full batches; `NoisyObjective(ModelBasedObjective(...))` is unwrapped, prepared as JAX GLM, and re-wrapped for zeroth-order noisy value queries

- **`src/experiments/paths.py`**
  - `results_root()`: lazy shared external results root, using `GENERALI_RESULTS_ROOT` when set and otherwise `~/projects/generali-pricing/results`; do not derive output defaults from `cwd` or `__file__`

- **`src/experiments/execution.py`**
  - `default_reporter_stack(config, *, json_reporter=None, include_plots=True)`: single source of truth for the reporter stack ordering (`PolicyArtifactReporter` before `JsonReporter`, plots before W&B upload); pass `json_reporter` to swap in a custom `JsonReporter` (e.g. seed sweeps' per-seed summary target) in the same slot and `include_plots=False` to drop the `PlotReporter`. `sweep_utils._seed_reporter_stack_factory` delegates here instead of re-hardcoding the order
  - `execute_experiment_run(name, config, runs_root=None, run_metadata=None, ...)`: creates `RunContext`, runs `run_experiment(...)`, finalizes reporters, and returns `ExecutedRun` with the result and output context; omitted `runs_root` uses `results_root()`

- **`src/experiments/launch.py`**
  - Shared local/Slurm orchestration seam for launch-aware entry points. Scripts define a `LaunchPlan` with explicit task decomposition (`run_task(index, context)`), optional serial `run_all(...)`, and optional `collect(...)`; the launcher owns `--launch auto|local|slurm`, `--array`, `--array-max-parallel`, `--task-index`, `--sweep-id`, Slurm parent/child branching, JAX GPU preflight for task jobs, task-record JSONs, and collector execution
  - Array mode never receives an opaque sweep loop: each Slurm array child resolves `SLURM_ARRAY_TASK_ID` and calls exactly one `run_task`. Local `--task-index N` uses the same task path for debugging
  - `LaunchPlan.runs_root` defaults to the shared external `results_root()` when unset; array task records live under `results/<project>/sweeps/<sweep-id>/tasks/task_<idx>.json`; collectors should read these records and rebuild aggregate CSVs/plots from normal per-task summaries or payloads

- **`src/experiments/slurm.py`**
  - Low-level ORCD Slurm adapter for launch-aware entry points; lightweight run-spec inspection selects a CPU or GPU profile before expensive config/data loading
  - NumPy-only runs auto-submit to `mit_normal` with CPU/memory/time resources; any explicit `compute_backend="jax"` run auto-submits to `mit_normal_gpu` with `--gres=gpu:l40s:1`
  - Supports single-job submissions, Slurm arrays via `SlurmArraySpec`, and dependent collector jobs (e.g. `afterany:<array_job_id>`). Slurm logs go under `results/slurm/%x-%j.out` by default (from the shared external `results_root()`); child jobs rerun the same entry point with `--no-sbatch`, activate `simulation_env`, export checkout-local `src` on `PYTHONPATH`, and require JAX GPU availability for JAX task jobs

- **`src/experiments/sweep_utils.py`**
  - `expand_override_grid(...)`: cartesian product of override values
  - `apply_config_overrides(...)`: validates and applies top-level `ExperimentConfig` overrides
  - `expand_sweep_overrides(...)`: pure `(variant_name, override_dict)` expansion (no `get_config`), so callers can merge a per-seed `seed_setup` into overrides before the config is built (required to reach the real-data factory)
  - `generate_sweep_runs(...)`: expands a base preset into named sweep variants; accepts either `override_grid` (cartesian product) or an explicit `override_list` of per-run override dicts; real-data override grids may include factory axes such as `policy_kind` and `constraint_mode`; either override form may include an `_run_name` key to set an explicit run name instead of the derived display name
  - `run_sweep(...)`: **canonical seed-aware sweep.** Replicates every variant across `run_seeds`, building each run's `SeedSetup` with `replicate_seed_setup(seed, anchor_seed, vary=..., fixed=...)`; by default `vary=("theta",)` keeps data/split/noise identical across replicates and only reinitializes policy `theta`. Per-seed runs share one variant folder (`summary-seed-<seed>.json` at the variant root, heavy artifacts under `seeds/seed-<seed>/`); writes per-variant and cross-variant aggregate error-bar plots plus `seed_grid_summary.csv`. Returns `SweepResult(project_dir, run_results, summary_rows)`. A plain seed sweep is the no-axis case (`override_list`/`override_grid` omitted). **Pitfall:** with `vary=("theta",)` the stochastic estimators (`stein_difference`/`spsa`/`gauss_stein`) reuse one perturbation stream across replicates, so their error bars capture init sensitivity only; add `"optimizer"` to `vary` to also capture estimator stochasticity.
  - `run_preset_sweep(...)`: legacy single-seed axis sweep; same `override_grid`/`override_list` signature as `generate_sweep_runs(...)`; executes variants through `execute_experiment_run(...)` and returns `SweepRunResult` records (each now also carries `run_seed`, `None` for single-seed sweeps)

- **`src/experiments/sweep_reporting.py`**
  - Aggregate sweep-output helpers for recurring scripts: timestamped aggregate directories, final estimator row collection for scalar config sweeps, CSV writing, and standard action/acceptance plus Pareto frontier plots
  - Cross-seed aggregation (home of the generalized `seed_repeats._summary_rows`): `collect_seed_grid_final_rows(...)` / `aggregate_seed_grid_rows(...)` group per-(variant, estimator) finals to mean/std/min/max over seeds; `write_seed_grid_outputs(...)` writes `seed_grid_finals.csv` + `seed_grid_summary.csv` and the aggregate error-bar plots; `objective_traces_by_estimator(...)` groups per-seed traces for loss-band plots
  - Simple sweep scripts should call this module instead of carrying local `_write_rows`, timestamp-dir, and private plotting-helper logic

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

- **`src/experiments/seeds/`** (package; replaces the former `seeding.py` module)
  - `streams.py`: seed primitives — `SeedSetup`/`ResolvedSeedSetup`, `resolve_seed_setup(...)` (legacy configs without `seed_setup` use `ExperimentConfig.seed` for every stream; explicit `SeedSetup` derives omitted streams from `run_seed`), `derive_seed`, `rng_from_seed`, and `optimizer_rngs(...)` (order-independent per-estimator batch and gradient RNGs from `optimizer_seed`)
  - `replicate.py`: seed-vary policy — `SeedStream` literal, `validate_vary(...)`, and `replicate_seed_setup(run_seed, anchor_seed, *, vary=("theta",), fixed=None)` which pins non-`vary` streams to `anchor_seed` and follows `run_seed` for `vary` streams (`vary=("all",)` leaves streams unset; `fixed` pins a stream). This is the single source of truth for "what varies across replicates"; `run_sweep` and `seed_repeats` both use it
  - `__init__.py`: re-exports the full public seed surface; import from `experiments.seeds`

- **`src/experiments/seed_repeats.py`**
  - `SeedRepeatSpec`: repeated-run orchestrator over explicit seed streams; default varies only `optimizer_seed` while fixing data/split/theta/noise to the first run seed. `seed_setup_for_repeat(...)` now delegates to `replicate_seed_setup(...)`
  - `run_seed_repeats(...)`: runs a preset once per `run_seed`, writes normal per-run outputs plus `seed_repeats.csv` and `seed_repeats_summary.csv`. Prefer `sweep_utils.run_sweep(...)` for new seed-aware sweeps

- **`src/experiments/policy_validation.py`**
  - Shared optimizer-independent policy validation helpers (`policy_u_values`, `evaluate_policy`) used by both `run_experiment()` and saved policy artifacts

- **`src/experiments/policy_artifacts.py`**
  - `PolicyArtifact`: reloadable trained-policy object saved as `policies/<estimator>/policy.json` plus `arrays.npz`
  - Separates policy input preprocessing (`raw x -> z`, including artifact preprocessing and optional policy-side whitening/PCA) from policy feature mapping (`z -> varphi(z) -> phi(z)`)
  - Persists theta, train/test/all CSV row bindings, objective/model metadata, policy head/feature-map specs, and full fitted policy-side preprocessing arrays
  - `load_policy_artifact(...).predict_u(split="train")` and `.evaluate(split="train")` rerun validation without optimizer training

- **`src/experiments/reporting/`**
  - `context.py`: `RunContext` and `create_run_context(...)` output-directory helpers; omitted `runs_root` uses `results_root()`, normal run leaves are flat `<slug>__<timestamp>`, caller-fixed `run_dir` is used verbatim for per-seed replicates, and optional `run_metadata` is carried into summaries
  - `base.py`: `StepReporter`, `Reporter`, and `ReporterStack` interfaces/composition
  - `console.py`: `ConsoleReporter` terminal output; per-step output controlled by `verbose`
  - `step_logger.py`: `FileStepLogger` writes per-step metrics to `plots/optimization/steps.csv`
  - `artifacts.py`: `PolicyArtifactReporter` writes reloadable trained-policy artifacts before `JsonReporter` records their relative paths
  - `json_summary.py`: `JsonReporter` and `build_summary_payload(...)` write `summary.json`, including estimator-level `train` and optional `test` policy metric blocks plus a `preset` block when `run_metadata` is set; `JsonReporter(summary_name=..., summary_dir=...)` parameterizes the filename/location so seed sweeps write `summary-seed-<seed>.json` at the variant root
  - `plots.py`: `PlotReporter` generates optimization and policy diagnostics, writes per-plot timings, and caps model-based theta contour subsampling/grid sizes
  - `wandb.py`: `WandbReporter` uploads run summaries and generated artifacts to W&B when enabled

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
  - Public `plot_sweep_tradeoffs(...)` and `plot_sweep_pareto_frontier(...)` power generic sweep frontier plots; lambda-specific wrappers call these helpers
  - Seed-aggregated error-bar plots for `run_sweep`: `plot_seed_grid_metric_bars(...)` (grouped bars per variant with `yerr` across seeds), `plot_seed_grid_frontier(...)` (mean acceptance vs mean objective with `xerr`/`yerr`), and `plot_seed_loss_bands(...)` (mean objective-vs-step with a +/- std band); consume `aggregate_seed_grid_rows(...)` output / per-seed traces
  - `select_theta_axes_max_variance(...)`: picks the two theta axes with highest variance for contour plots

### Entry Point (`main.py`)

- Reads `RUN_CONFIGS` list; entries may be a config name or `(config_name, overrides)` tuple passed to `get_config(config_name, overrides=overrides)`
- Builds a `LaunchPlan` with one task per `RUN_CONFIGS` entry. `python main.py` defaults to `--launch auto`, which submits through `src/experiments/launch.py`/`slurm.py` when outside Slurm; use `--launch local` or `--no-sbatch` for intentional local/debug execution
- `python main.py --launch slurm --array` submits one Slurm array task per `RUN_CONFIGS` entry; without `--array`, all configs run serially in one job/current process
- `main.py` prepends its checkout-local `src` directory to `sys.path`, and Slurm child jobs export the submitting checkout's `src` on `PYTHONPATH`, so worktree runs do not accidentally import the canonical checkout's editable install
- CPU-only specs submit to ORCD `mit_normal`; specs with `compute_backend="jax"` submit to `mit_normal_gpu` with one L40S GPU and fail fast if JAX reports only CPU in the child job
- For each config spec: delegates the run lifecycle to `experiments.execution.execute_experiment_run(...)`, which creates `RunContext`, assembles the default `ReporterStack`, calls `run_experiment()`, finalizes reporters, and stores preset/override metadata in `summary.json`
- All I/O is handled by reporters, not by the runner
- `scripts/run_sweep.py` is a minimal generic seed-aware preset sweep launcher around `experiments.sweep_utils.run_sweep(...)`. It accepts a base preset plus JSON override mapping/list/grid inputs, optional seed-vary/fixed-stream arguments, and `--requires-jax` to force the GPU Slurm profile. It should stay experiment-agnostic; dedicated scientific grids belong in their own scripts.
- `scripts/run_noisy_glm_theta_variance_sweep.py` runs the all-data trust-constrained GLM noisy-objective sweeps on GPU/JAX. It defaults to the saved first-order no-noise truth summary at `results/real_data_glm_base__20260706_124627/summary.json`, centers heteroskedastic noise at that run's final mean `u`, sweeps theta starts along the real initialization-to-truth line by L2 distance, sweeps homoskedastic/heteroskedastic noise variance for zeroth-order estimators, writes `noisy_glm_theta_variance_finals.csv` / `noisy_glm_theta_variance_summary.csv`, and regenerates theta-distance/objective-gap plots per grid project.
- `scripts/run_noise_offset_grid.py` runs the combined noise-level x theta-offset planted-logistic grid: for each noise family (homoskedastic std `sigma in {0, 0.1, 2}` new + `0.5` reused from the saved theta-offset sweep; heteroskedastic growth `gamma in {0, 0.25, 4}` new + `1.0` reused) it varies the init offset `delta` in `theta0 = theta_FO_clean + delta * 1` over 9 offsets with `RUN_SEEDS=(7, 8, 9)`, carrying its own copy of the retired planted-noise fill-in sweep driver's COMMON_OVERRIDES/seed policy (formerly in `scripts/run_sweep.py`) so runs stay comparable with the saved sweeps. Writes `noise_offset_grid_finals.csv` plus per-estimator two-panel figures (final theta distance to first-order truth | clean-objective gap on the reconstructed train batch, curves = noise level) under `results/homoskedastic-noise-offset-grid/` and `results/heteroskedastic-noise-offset-grid/`. `--plots-only` regenerates outputs from saved summaries; `--families` selects the grid group; launch-aware with auto Slurm submit
- `scripts/run_fixed_regression_noise_offset_grid.py` runs the fixed-regression noise-level x theta-offset grid: it computes an anchored clean first-order truth for `fixed_regression_base`, centers heteroskedastic noise at that truth policy's mean `u`, varies the init offset `delta` in `theta0 = theta_FO_clean + delta * 1` over 9 offsets, and uses `run_sweep(...)` one variant at a time inside warm `(family, noise-level)` launch tasks so every variant gets the canonical seed-sweep folder layout. Defaults use homoskedastic std `{0, 0.1, 0.5, 2}`, heteroskedastic growth `{0, 0.25, 1, 4}`, `RUN_SEEDS=(7, 8, 9)`, and estimators `finite_difference` / `stein_difference`. Writes `fixed_regression_noise_offset_grid_finals.csv` plus per-estimator two-panel figures under `results/fixed-regression-homoskedastic-noise-offset-grid/` and `results/fixed-regression-heteroskedastic-noise-offset-grid/`. `--plots-only` regenerates outputs from saved summaries; `--families` selects the grid group; launch-aware with auto Slurm submit
- `scripts/run_planted_logistic_action_bias_sweep.py` runs the first planted-logistic deterministic action-bias experiment: optimize `BiasedObjective(M_star, lambda_bias)` for `lambda_bias in {0, 0.01, 0.05, 0.1, 0.2}`, optimize the true `M_star` oracle baseline once on the same sampled train batch, and write `planted_logistic_action_bias_sweep.csv` with true gaps, mean actions, surrogate value, and signed optimism gap `surrogate - true`
- `scripts/run_planted_logistic_support_bias_sweep.py` runs the planted-logistic upper-support deterministic bias experiment: optimize a surrogate that is exact for `u <= u_star + support_radius` and optimistic only above support, sweep `lambda_bias in {0, 0.01, 0.025, 0.05, 0.1, 0.2}` and `support_radius in {0.02, 0.05, 0.1, 0.2}`, and write `planted_logistic_support_bias_sweep.csv` plus true-gap/support-excess plots
- `scripts/run_glm_u_coef_sweep.py` runs the softmax/no-PCA/trust-constr GLM setup over 200000 sampled rows and `u_coef in {-4, -5, -8, -10, -20}`, keeps per-run distribution plots enabled, and writes aggregate `glm_u_coef_sweep.csv` plus frontier plots under `results/glm-u-coef-sweep/u_coef_frontier_<timestamp>/`. It is launch-aware but defaults to local execution; `--launch slurm --array` runs one `u_coef` value per array task and a dependent collector combines row payloads into the existing frontier outputs
- `scripts/run_glm_softmax_alpha_sweep.py` runs the trust-constrained softmax/no-PCA/linear-feature GLM setup over symmetric action bounds `[-alpha, alpha]` for `alpha in {0.5, 0.4, 0.3, 0.2, 0.15, 0.125, 0.1, 0.075}`, saves normal per-alpha policy artifacts, and writes aggregate final objective/profit plots plus artifact-replayed acceptance-threshold and per-alpha `u`-bin expected-profit summaries under `results/glm-softmax-alpha-sweep/alpha_sweep_<timestamp>/`. It is launch-aware; `--launch slurm --array` runs one alpha per array task and a dependent collector combines final-row payloads plus artifact-replayed bin summaries into the existing aggregate outputs
- `scripts/run_glm_sensitivity_bucket_experiment.py` buckets all complete eligible GLM rows into low/medium/high local price-sensitivity tertiles at median observed `U`, runs the softmax/no-PCA/trust-constr GLM setup on every row in each bucket, keeps per-run distribution plots enabled, and writes aggregate `glm_sensitivity_bucket_experiment.csv` plus comparison plots under `results/glm-sensitivity-buckets/sensitivity_bucket_summary_<timestamp>/`. It is launch-aware; `--launch slurm --array` runs one bucket per array task and the collector recomputes bucket definitions for plots
- `scripts/run_glm_reference_elasticity_bucket_experiment.py` repeats the GLM bucket experiment for reference actions `u_ref in {-0.1, 0.1, 0.2, 0.3}`, ranks rows by elasticity magnitude at each reference action, runs only `first_order`, annotates summary charts with average bucket elasticity magnitude, and writes per-reference summaries under `results/glm-reference-elasticity-buckets/`. It is launch-aware; `--launch slurm --array` runs one `(u_ref, bucket)` pair per array task
- `scripts/plot_glm_sensitivity_distribution.py` computes GLM customer elasticities $$d p_{accept}(x, u) / du$$ over a default `u in [-0.3, 0.3]` grid, writes a mean/quantile elasticity-by-`u` curve, selected-`u` elasticity histograms for `{-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3}` with default `0.5-99.5%` x-axis clipping marked, and CSV summaries under `results/glm-sensitivity-distribution/`
- `scripts/plot_policy_acceptance_grid.py` loads a saved policy artifact, scores artifact-bound rows by mean absolute acceptance sensitivity over a simulated `u` grid and by predicted loss, randomly samples clients from low/medium/high tertiles for each score, and writes two three-panel client-level acceptance-curve plots plus `sampled_clients.csv` under `results/policy-acceptance-grid/`; omit `--seed` to resample clients each run
- `scripts/plot_saved_acceptance_floor_frontier.py` re-plots acceptance-floor Pareto frontiers from a saved `acceptance_floor_sweep.csv` (or the latest matching frontier directory) without rerunning optimization; defaults to `first_order` and writes estimator-suffixed Pareto PNGs
- `scripts/query_acceptance_at_u.py` loads a config preset or default GLM/XGB model type and reports mean acceptance for supplied or evenly sampled constant `u` values without running optimization; writes acceptance-curve and historical-`U` rug plots under `results/acceptance_queries/` by default and optionally writes `u,n,mean_acceptance` CSV output
- `scripts/evaluate_historical_policy_objective.py` reads a saved run `summary.json`, reconstructs selected CSV row positions from full-eligible mode or seed/`n_samples`, prints the estimator theta used, and evaluates final policy prices under historical acceptance `1 - is_churn` and observed `Y_G_Loss`; writes aggregate `summary.json` and row-level `per_row.csv` under `historical_policy_objective/<estimator>/`
  - Prefer `--policy-artifact results/.../policies/<estimator>/policy.json` for new runs; `--summary-json` remains a legacy fallback for outputs created before policy artifacts existed
  - Supports `--objective model` to replay the trained model objective and `--objective historical` for the observed-outcome diagnostic; both support `--split train|test|all`, where `all` means all selected run rows before splitting
  - `--u-source historical --acceptance-source historical|model --technical-price-source historical|model` runs script-only historical-`U` diagnostics without optimized policy actions; `--model-type glm|xgb` selects deterministic complete eligible rows unless `--summary-json` is supplied to reuse a saved run sample; `--acceptance-model-historical-u` is a shortcut for model acceptance plus historical technical price
- `scripts/run_policy_pca_grid.py` runs the GLM policy PCA-dimensionality grid over configured PCA dimensions and policy classes `(constant, linear, quadratic, third_order, fourth_order, softmax_linear, softmax_quadratic, softmax_third_order, softmax_fourth_order, mlp)`; unconstrained is default, `--constrained` uses `trust-constr` with the observed GLM acceptance floor and a 500-step default cap; outputs aggregate CSVs, summary markdown, and headline/spread plots under `results/policy-pca-grid/`; prints per-condition progress by default and supports `--quiet`. It is launch-aware; `--launch slurm --array` runs one `(pca_dim, policy_class, seed)` condition per array task and the collector writes combined grid outputs
- `scripts/benchmark_optax_vs_trust_constr.py` benchmarks SciPy minimize against the optax step rules: a planted-logistic group (theta dim 200 LinearPolicy; L-BFGS-B vs `optax-adam`/`optax-sgd`) and a real-data GLM group on the fixed JAX prepared batch (trust-constr with the observed acceptance floor vs optax rules on the smooth-penalty formulation of the same floor). Writes `benchmark.csv` under `results/optax-benchmark/`. On shared CPU nodes pin `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` (and `JAX_PLATFORMS=cpu` off-GPU) — JAX import plus OpenBLAS thread oversubscription inside a CPU-limited slice can slow NumPy matmuls by orders of magnitude and wash out solver timings
- `scripts/benchmark_experiment_speed.py` benchmarks GLM analytical acceptance vs sklearn `predict_proba`, Stein-difference gradient timing/call counts, repeated objective-cache behavior, and full-vs-subsampled contour grid timing; use it to quantify whether performance changes speed up real-data diagnostics without relying on flaky pytest time thresholds

## Known Issues and Dead Code

These are documented here so agents can account for them and clean them up
when appropriate.

- _None currently tracked._

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
- Tests for one-off or ad-hoc analysis scripts belong under `tests/scripts/`,
  mirroring the relevant `scripts/` filename or script domain.
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
| `test_biased_objective.py` | Deterministic linear/support action-bias wrapper value, action-gradient, theta-gradient, reporting, and serialization |
| `test_noise.py` | Gaussian noise adapters (homoskedastic + heteroskedastic keying, determinism, seed-stream wiring) and `NoisyObjective` behavior |
| `test_model_based_objective.py` | `value()`, `grad()` shape, `value_at_u()`, analytical vs FD grad agreement |
| `test_jax_prepared_glm_objective.py` | JAX prepared GLM value, per-row action-value hooks, gradient, policy-u, acceptance, and SciPy adapter parity |
| `test_policy_batch.py` | Policy batch `value/grad` shapes, bounds, and kind labels (incl. `MLPPolicy`) |
| `test_mlp_policy_grad.py` | `MLPPolicy.grad` matches per-coordinate FD Jacobian; `mlp_init_theta` symmetry-breaking |
| `test_jax_prepared_glm_objective.py` | JAX prepared GLM value, per-row action-value hooks, gradient, policy-u, acceptance, and SciPy adapter parity |
| `test_policy_preprocessing.py` | Policy-side standardization, whitening, PCA dimensionality, and transform validation |
| `test_policy_u_histograms.py` | Policy u-distribution visualization |

#### `tests/optimization/`
| Test File | Area |
|---|---|
| `test_helpers.py` | `_clamp_theta`, `sample_indices`, `x_batch`, `finite_difference_theta_grad` |
| `test_gradient_methods_math.py` | Gauss-Stein, SPSA, Stein-Difference convergence; FD u-space vs theta-space; SPSA variance |
| `test_step_rules.py` | Armijo sufficient decrease, edge cases, input validation |
| `test_optax_step_rule.py` | Optax adam/sgd loop: convergence, determinism, sgd-vs-constant parity, JAX GLM penalty run |
| `test_finite_difference_gradient.py` | Finite-difference gradient accuracy and determinism |
| `test_gradient_resampling.py` | Gradient method resampling behavior |
| `test_jax_trust_constr_callbacks.py` | JAX prepared GLM callbacks and zeroth-order gradients match CPU prepared GLM under SciPy trust-constr |
| `test_optimization_class.py` | Class-based optimizer entry point and gradient-object behavior |
| `test_minibatch_stochasticity.py` | Mini-batch determinism and full-batch equivalence |
| `test_minimize_orders.py` | SciPy first/Gauss-Stein/Stein-difference/SPSA wrappers |
| `test_early_stopping.py` | grad_norm_tol early stopping |
| `test_trust_constr_constraint.py` | Trust-region constraint acceptance floor |
| `test_jax_trust_constr_callbacks.py` | JAX prepared GLM callbacks and zeroth-order gradients match CPU prepared GLM under SciPy trust-constr |

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
| `test_execution.py` | Shared experiment run lifecycle and default reporter stack |
| `test_experiment_configs.py` | Config registry (get_config, list_configs) |
| `test_real_data_config.py` | All real-data presets load; x_fixed shape; correct estimator sets |
| `test_enabled_estimators.py` | Selective estimator execution |
| `test_verbose_config.py` | verbose flag defaults and serialization |
| `test_baseline_test.py` | End-to-end smoke test with fixed_regression_base overrides |
| `test_paths.py` | shared results-root environment override and default path |
| `test_launch.py` | Shared launch-plan local execution, Slurm array submission, collector dependency, and array-child task selection |
| `test_run_context.py` | default results-root output directory, readable run leaves, run metadata, and verbatim run_dir paths |
| `test_noisy_objective_backend.py` | Noisy objective acceptance-control propagation and JAX GLM backend re-wrapping |
| `test_run_sweep_script.py` | Generic `scripts/run_sweep.py` argument parsing, JAX launch flag detection, and launcher delegation |
| `test_slurm_launcher.py` | ORCD Slurm profile selection, command construction, array/dependency flags, autosubmit skips, and JAX GPU preflight |
| `test_train_test_split.py` | Runner train/test split, held-out policy metrics, and summary payloads |
| `test_seeding.py` | Seed setup serialization, stream routing, `replicate_seed_setup` vary/fixed policy and legacy parity |
| `test_seed_repeats.py` | Seed-repeat setup construction and aggregate CSV outputs |
| `test_sweep_run.py` | Canonical `run_sweep`: shared variant folder, fixed data/split/noise vs varying theta, non-degenerate error bars |
| `test_sweep_reporting.py` | Aggregate sweep rows, cross-seed grid aggregation, and CSV helpers |
| `test_sweep_utils.py` | Override-grid expansion and preset sweep config/result generation |
| `test_sensitivity_buckets.py` | GLM local price-sensitivity scoring and tertile construction |
| `test_sensitivity_bucket_script.py` | Sensitivity bucket experiment script constants and summaries |
| `test_softmax_alpha_sweep_script.py` | Softmax alpha sweep constants, artifact-replayed profit bins, and aggregate output plots |
| `test_plot_glm_sensitivity_distribution_script.py` | GLM elasticity distribution script summaries and plot outputs |
| `test_reference_elasticity_bucket_script.py` | Reference-u GLM elasticity bucket script constants and plot outputs |
| `test_noisy_glm_theta_variance_sweep_script.py` | Noisy GLM theta/variance truth parsing, variant construction, launch plan, and aggregate outputs |
| `test_policy_pca_grid.py` | Policy PCA grid condition construction and aggregate output writing |

#### `tests/reporting/`
| Test File | Area |
|---|---|
| `test_logging.py` | Step logging output format |
| `test_file_step_logger.py` | FileStepLogger CSV output |
| `test_json_summary_reporter.py` | JsonReporter default and parameterized `summary-seed-<seed>.json` naming/location |
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

#### `tests/scripts/`
| Test File | Area |
|---|---|
| `test_fixed_regression_noise_offset_grid_script.py` | Fixed-regression noise x theta-offset grid constants, warm `run_sweep` delegation, completion detection, metric reconstruction, and launch-plan sizing |
| `test_noise_offset_grid_script.py` | Combined noise x theta-offset grid constants, variant naming round-trip, task specs, and axis-label definitions |
| `test_planted_logistic_action_bias_sweep_script.py` | Planted-logistic action-bias sweep constants, row metrics, and CSV writing |
| `test_planted_logistic_support_bias_sweep_script.py` | Planted-logistic support-bias sweep constants, support metrics, CSV writing, and aggregate plots |

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
