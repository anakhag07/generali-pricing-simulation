# Agent Instructions

Project context: simulation and optimization repo. Users should be able to specify an experiment config and compare how different gradient methods perform across different objectives and policy types. Primary entry point is `main.py`.

## Core Working Rules

- Prefer small, focused changes with clear doc updates. Prior to making code changes, think about whether the code addition is necessary and if it keeps a clean, intuitive repo structure for the public api. 
- Keep simulation logic deterministic when a seed is set.
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
6. If working in a parallel terminal or worktree, assume other branches may have changed the repo recently and re-check branch state.
7. If repo structure or behavior appears inconsistent with `README.md` or `AGENTS.md`, update docs as part of the task.

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
- Call out any expected `README.md` or `AGENTS.md` updates.

### Build Mode
- Make focused code changes.
- Keep docs in sync with implementation.
- Run validation commands after changes.
- Prepare a concise handoff summary suitable for a commit, PR, or later session.

## Source of Truth

When behavior and documentation disagree, use this priority:

1. `src/objective/objectives/fixed_regression.py`
2. Current implementation in the relevant source module
3. Tests
4. `README.md`
5. `AGENTS.md`

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
  - `ModelBasedObjective`: pricing objective $$f(u;x) = a(x,u)(\hat{Y}(x) - u \cdot p(x))$$ backed by trained sklearn/XGBoost models
  - Takes `acceptance_model` (GLM Pipeline or XGBClassifier) and `loss_model` (LinearRegression or XGBRegressor)
  - `u_coef` enables analytical gradient for GLM; `None` triggers central FD for XGBoost
  - `value()`, `grad()`, `value_at_u()`

- **`src/objective/objectives/planted_logistic.py`**
  - `PlantedLogisticObjective`: convex logistic objective with known optimum `u_star`
  - `optimal_u()` method exposes the planted optimum

- **`src/objective/policy.py`**
  - Implements `Policy` with batch methods `value(theta, x_batch)` and `grad(theta, x_batch)`
  - Concrete policies: `ConstantPolicy`, `LinearPolicy`, `SoftmaxPolicy`
  - `policy_from_kind(kind)`: factory function

- **`src/objective/utils.py`**
  - `optimal_u(objective)`: public helper to extract u* from objective if available
  - Private helpers: `_theta_grad_from_u_grad`, `_mean_action`, `_action_value_at_u`

#### Data Layer (`src/data/`)

- **`src/data/loader.py`**
  - `FEATURE_COLS_GLM`: 12-column state feature list for GLM configs (9 base + premium + X_prev_renewal_perc + X_year)
  - `FEATURE_COLS_XGB`: 10-column state feature list for XGB configs (9 base + premium)
  - `ACCEPTANCE_STATE_COLS`: 10 cols passed to acceptance model (base + premium, no U)
  - `LOSS_FEATURE_COLS`: 9 base cols passed to loss model
  - `load_x_array(model_type, n_rows=5000)`: loads first n_rows of X features from acceptance CSV; returns (n, 12) for GLM, (n, 10) for XGB
  - `load_model_artifacts(model_type)`: loads (acceptance_model, loss_model) from pickle files
  - `extract_glm_u_coef(glm_pipeline)`: extracts effective d_logit/dU = w_U / std_U from fitted GLM Pipeline for analytical gradient computation

#### Optimization Layer (`src/optimization/`)

- **`src/optimization/base.py`**
  - `Optimization`: class-based optimization entry point
  - `Optimization.solve(theta_start)`: runs SciPy `minimize` (`L-BFGS-B`) with configured gradient method, mini-batching, and trace recording
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
  - `run_first_order_minimize(...)`, `run_finite_difference_minimize(...)`, `run_gauss_stein_minimize(...)`, `run_stein_difference_minimize(...)`, `run_spsa_minimize(...)`: compatibility wrappers that instantiate `Optimization` with the corresponding gradient object and call `solve(...)`

- **`src/optimization/steps.py`**
  - `STEP_RULE_LBFGSB`, `STEP_RULE_CONSTANT`, `STEP_RULE_ARMIJO`, `STEP_RULES`
  - `constant_step_size(step_size)`: returns the step size unchanged
  - `armijo_backtracking_step_size(...)`: Armijo line search utility (not used by SciPy first/Gauss-Stein/Stein-difference/SPSA solvers)

#### Experiment Layer (`src/experiments/`)

- **`src/experiments/config.py`**
  - `ExperimentConfig`: frozen dataclass with extensive `__post_init__` validation
  - Primary fields: `objective` (theta objective) and `theta0` (initial theta)
  - `x_fixed: np.ndarray | None = None`: when set, runner uses this 2D array as state batch instead of sampling from N(0, I)
  - Objective/policy wiring is explicit; configs pass a concrete theta-level objective instance
  - `batch_size: int | None = None` enables stochastic mini-batch optimization when set
  - `CorrectnessSpec`: controls how "true" gradients are computed (`"exact"`, `"numdiff"`, `"none"`)
  - `verbose: bool = False` controls terminal output of per-step metrics
  - Preset-composition helpers: `make_*_objective`, `make_softmax_policy`, `make_model_based_objective`,
    `canonical_training_block`, `canonical_runtime_block`, and `build_experiment_config`

- **`src/experiments/configs/`** (preset registry)
  - `__init__.py`: `get_config(name)` and `list_configs()` registry
  - `first_order_runs_diff_starts.py`: planted-logistic preset configured for comparison runs across different initial starts
  - `fixed_regression_base.py`: base fixed-regression config (4D, L-BFGS-B step rule, W&B enabled)
  - `planted_logistic_base.py`: planted logistic base config (3D, L-BFGS-B step rule, 5000 steps, u*=1.1)
  - `real_data_glm_base.py`: GLM pickle-based config; state_dim=12; all 5 estimators; analytical first-order gradient via u_coef
  - `real_data_glm_linear_base.py`: GLM pickle-based diagnostic config; same data/models as `real_data_glm_base` but with `LinearPolicy` and constant-`u=1.1` initialization to inspect behavior without softmax saturation
  - `real_data_xgb_base.py`: XGBoost pickle-based config; state_dim=10; 4 estimators (no first_order); FD for d_acceptance/du
  - `config_template.py`: copy-first scaffold with `None` placeholders for all `ExperimentConfig` fields plus objective/correctness parameter blocks; not registered as a runnable preset

- **`src/experiments/defaults.py`**
  - `default_theta0(state_dim)`: returns default initial theta with `state_dim + 1` params
  - `default_policy(state_dim)`: returns default `SoftmaxPolicy`

- **`src/experiments/helpers.py`** (largest file; orchestration + wrappers)
  - `resolve_true_grad_theta_fn(objective, correctness)`: resolves the "true" theta-gradient function from correctness spec
  - `run_first_order(...)`: wrapper delegating to `optimization.solvers.run_first_order_minimize`
  - `run_finite_difference(...)`: wrapper delegating to `optimization.solvers.run_finite_difference_minimize`
  - `run_gauss_stein(...)`: wrapper delegating to `optimization.solvers.run_gauss_stein_minimize`
  - `run_stein_difference(...)`: wrapper delegating to `optimization.solvers.run_stein_difference_minimize`
  - `run_spsa(...)`: wrapper delegating to `optimization.solvers.run_spsa_minimize`
  - Uses `optimization.helpers.finite_difference_theta_grad(...)` for correctness-mode numerical theta gradients

- **`src/experiments/run.py`**
  - `run_experiment(config, step_reporter)`: main runner; uses `config.x_fixed` as state array when set, otherwise samples from N(0, I); runs enabled estimators; returns `ExperimentResult` (pure computation, no I/O)

- **`src/experiments/sweep_utils.py`**
  - `expand_override_grid(...)`: cartesian product of override values
  - `apply_config_overrides(...)`: validates and applies top-level `ExperimentConfig` overrides
  - `generate_sweep_runs(...)`: expands a base preset into named sweep variants
  - `run_preset_sweep(...)`: executes sweep variants through the standard reporter pipeline

- **`src/experiments/results.py`**
  - `OptimizationTrace`: per-step trace with u values, objective values, gradient estimates, optional theta values and step sizes
  - `EstimatorResult`: final theta, u, value, and wall-clock time
  - `ExperimentResult`: full result including config, traces, and optional u_star

- **`src/experiments/reporters.py`**
  - `RunContext`: frozen dataclass with experiment name, run directory paths, timestamp
  - `create_run_context(...)`: creates run directories under `outputs/` by default
  - `StepReporter`: protocol for per-step metric logging
  - `Reporter`: protocol with `on_start` and `on_end` hooks
  - `ReporterStack`: composite that delegates to a list of reporters; also implements `StepReporter`
  - `ConsoleReporter`: prints to terminal; per-step output controlled by `verbose`
  - `FileStepLogger`: writes per-step metrics to `steps.csv` in the run directory
  - `JsonReporter`: writes `summary.json` on end
  - `PlotReporter`: generates all matplotlib plots on end

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
  - `plot_theta_objective_contours(...)`: 2D contour plot with optimization paths
  - `select_theta_axes_max_variance(...)`: picks the two theta axes with highest variance for contour plots

### Entry Point (`main.py`)

- Reads `RUN_CONFIGS` list (currently `["fixed_regression_base"]`)
- For each config name: loads via `get_config()`, creates `RunContext`, assembles `ReporterStack`, calls `run_experiment()`, finalizes with `reporters.on_end()`
- All I/O is handled by reporters, not by the runner
- `scripts/run_sweep.py` provides optional preset-based sweep execution using top-level overrides

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
- Keep tests in the existing flat `tests/` layout for now.

### Current Test Coverage

| Test File | Area |
|---|---|
| `test_baseline_test.py` | End-to-end smoke test with fixed_regression_base overrides |
| `test_config.py` | ExperimentConfig validation rules |
| `test_correctness_spec.py` | CorrectnessSpec gradient source modes |
| `test_early_stopping.py` | grad_norm_tol early stopping |
| `test_enabled_estimators.py` | Selective estimator execution |
| `test_experiment_configs.py` | Config registry (get_config, list_configs) |
| `test_finite_difference_gradient.py` | Finite-difference gradient accuracy and determinism |
| `test_file_step_logger.py` | FileStepLogger CSV output |
| `test_minibatch_stochasticity.py` | Mini-batch determinism and full-batch equivalence |
| `test_minimize_orders.py` | SciPy first/Gauss-Stein/Stein-difference/SPSA wrappers (decrease + seed determinism) |
| `test_optimization_class.py` | Class-based optimizer entry point and gradient-object behavior |
| `test_objective_batch.py` | Deterministic objective private batch helpers and `value_at_u` |
| `test_objective_package_exports.py` | objective package API exports remain importable |
| `test_objective_models.py` | FixedRegressionObjective value and gradient correctness |
| `test_planted_logistic_objective.py` | Planted logistic gradient at u_star and minimum |
| `test_plot_u_star.py` | u_star selection for plotting |
| `test_policy_batch.py` | Policy batch `value/grad` shapes, bounds, and kind labels |
| `test_run_context.py` | default output directory and run context paths |
| `test_state_vector.py` | `sample_states` shape and dtype |
| `test_step_rules.py` | Armijo backtracking on quadratic |
| `test_theta_contours.py` | Contour grid shapes, axis selection |
| `test_trace_theta_values.py` | theta_values recorded in first/Gauss-Stein/Stein-difference traces |
| `test_verbose_config.py` | verbose flag defaults and serialization |
| `test_visualization_step_sizes.py` | step_sizes plot uses log y-scale |
| `test_sweep_utils.py` | Override-grid expansion and preset sweep config generation |
| `test_data_loader.py` | `load_x_array` shape/dtype, model artifact types, U normalization, CSV column sets |
| `test_model_based_objective.py` | `value()`, `grad()` shape, `value_at_u()`, analytical vs FD grad agreement |
| `test_csv_objective.py` | `value_at_u`, k-fallback, `grad()` raises NotImplementedError |
| `test_real_data_config.py` | All 4 real-data presets load; x_fixed shape; correct estimator sets |

## Documentation and Maintenance

### README.md
Keep README concise—a quick-start and overview only. Update when:
- setup or execution steps change
- new user-facing features are added
- basic workflow changes

Point users to `docs/` for detailed API reference.

### docs/ (pdoc-generated)
Regenerate docs when public API changes significantly.

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
