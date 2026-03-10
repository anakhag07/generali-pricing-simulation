# Agent Instructions

Project context: pricing simulation and optimization demo. Primary entry point is `main.py`.

## Core Working Rules

- Prefer small, focused changes with clear doc updates.
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

1. `src/data/fixed_objective.py`
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

#### Data Layer (`src/data/`)

- **`src/data/models.py`**
  - `StateVector`: frozen dataclass wrapping a 1D numpy array; has `sample(rng, dim)` static method
  - `Customer`: frozen dataclass with `x: StateVector`; has `sample(rng, state_dim)` static method
  - `Contract`: frozen dataclass with `u: float` (bounds check is a no-op; see Known Issues)
  - `ObjectiveResult`: frozen dataclass with `value` and `grad_u`
  - Protocols: `AcceptanceModel`, `LossModel`, `RevenueModel`, `ObjectiveModel`
  - `default_rng(seed)`: wrapper around `np.random.default_rng`

- **`src/data/fixed_objective.py`** (source of truth for objective math)
  - `FixedRegressionAcceptance`: sigmoid acceptance with `beta_1` (positive) and `beta_2` (negative)
  - `FixedRegressionLoss`: linear loss with `beta_3` (positive)
  - `FixedRegressionRevenue`: linear revenue with `beta_4` (positive)
  - `FixedRegressionObjective`: composite objective; `from_parameters` classmethod, scalar + batch evaluation
  - `FixedRegressionBatch`: pre-computed batch for vectorized evaluation

- **`src/data/planted_logistic.py`**
  - `PlantedLogisticObjective`: convex logistic objective with known optimum `u_star`
  - `PlantedLogisticBatch`: pre-computed batch for vectorized evaluation
  - `optimal_u()` method exposes the planted optimum

#### Optimization Layer (`src/optimization/`)

- **`src/optimization/policy.py`**
  - `PolicySpec`: frozen dataclass pairing `theta` (numpy array) with `kind` string
  - Policy kinds: `POLICY_CONSTANT`, `POLICY_LINEAR`, `POLICY_SOFTMAX`
  - `phi(x)` / `phi_batch(x_array)`: prepend bias term to features
  - `policy_u(theta, x, kind)` / `policy_u_batch(...)`: compute action from policy
  - `policy_grad_theta(theta, x, kind)`: compute `du/dtheta` (defined but unused in pipeline; see Known Issues)
  - `apply_policy(policy, x)`: convenience wrapper (does not clip `u`)

- **`src/optimization/steps.py`**
  - `STEP_RULE_CONSTANT`, `STEP_RULE_ARMIJO`, `STEP_RULES`
  - `constant_step_size(step_size)`: returns the step size unchanged
  - `armijo_backtracking_step_size(...)`: Armijo line search with configurable `c`, `shrink`, `max_backtracks`, `min_step`

- **`src/optimization/common.py`**
  - `gaussian_noise(rng, shape)`: standard normal samples (used by zeroth-order estimator)
  - `U_BOUNDS`: defined but unused (see Known Issues)

- **`src/optimization/gradients/zeroth_order.py`**
  - `stein_zeroth_order_grad_batch(u_values, objective_fn, rng, n_samples, sigma)`: vectorized Stein gradient estimator (primary entry point)
  - `stein_zeroth_order_grad(...)`: scalar version (defined but unused in pipeline)

#### Experiment Layer (`src/experiments/`)

- **`src/experiments/config.py`**
  - `ExperimentConfig`: frozen dataclass with extensive `__post_init__` validation
  - `CorrectnessSpec`: controls how "true" gradients are computed (`"exact"`, `"numdiff"`, `"none"`)
  - `verbose: bool = False` controls terminal output of per-step metrics

- **`src/experiments/configs/`** (preset registry)
  - `__init__.py`: `get_config(name)` and `list_configs()` registry
  - `baseline_fixed_objective.py`: 3D fixed regression, constant step, 100 steps
  - `baseline_test.py`: minimal smoke-test config (2 samples, 1 step, plot=False)
  - `custom.py`: ad-hoc experiment (2D fixed regression, Armijo, 1000 steps)
  - `planted_logistic.py`: planted logistic, Armijo, 5000 steps, u*=1.1

- **`src/experiments/defaults.py`**
  - `default_policy_spec(state_dim)`: returns softmax policy with `state_dim + 1` theta params

- **`src/experiments/helpers.py`** (largest file; core optimization logic)
  - `build_batch_objective_fns(objective_model, x_samples)`: builds value/grad functions averaged over batch
  - `resolve_true_grad_u_fn(objective_model, correctness)`: resolves the "true" gradient function from correctness spec
  - `run_first_order(...)`: first-order theta optimizer
  - `run_zeroth_order(...)`: zeroth-order theta optimizer
  - `run_lbfgs_theta(...)`: L-BFGS-B theta optimizer via SciPy
  - Internal: `_run_estimated_grad_optimizer(...)` shared loop for first/zeroth-order, `_numdiff_grad(...)` for finite-difference gradients

- **`src/experiments/run.py`**
  - `run_experiment(config, step_reporter)`: main runner; samples customers, runs enabled estimators, returns `ExperimentResult` (pure computation, no I/O)

- **`src/experiments/results.py`**
  - `OptimizationTrace`: per-step trace with u values, objective values, gradient estimates, optional theta values and step sizes
  - `EstimatorResult`: final theta, u, value, and wall-clock time
  - `ExperimentResult`: full result including config, traces, and optional u_star

- **`src/experiments/reporters.py`**
  - `RunContext`: frozen dataclass with experiment name, run directory paths, timestamp
  - `StepReporter`: protocol for per-step metric logging
  - `Reporter`: protocol with `on_start` and `on_end` hooks
  - `ReporterStack`: composite that delegates to a list of reporters; also implements `StepReporter`
  - `ConsoleReporter`: prints to terminal; per-step output controlled by `verbose`
  - `FileStepLogger`: writes per-step metrics to `steps.csv` in the run directory
  - `JsonReporter`: writes `summary.json` on end
  - `PlotReporter`: generates all matplotlib plots on end

- **`src/experiments/logging.py`**
  - `log_step(method, step, u, value, ...)`: prints one step to console
  - `log_summary(result)`: prints full experiment summary to console

- **`src/experiments/visualization.py`**
  - `ESTIMATOR_STYLES`: color/label config per estimator
  - `plot_loss_curves(...)`: objective vs step; optional |u - u*| subplot
  - `plot_gradient_norms(...)`: true theta gradient norms; optional error subplot
  - `plot_step_sizes(...)`: per-step step sizes (log scale y-axis)
  - `plot_objective_u_slice(...)`: objective and gradient vs u grid
  - `plot_theta_objective_contours(...)`: 2D contour plot with optimization paths
  - `select_theta_axes_max_variance(...)`: picks the two theta axes with highest variance for contour plots

### Entry Point (`main.py`)

- Reads `RUN_CONFIGS` list (currently `["custom"]`)
- For each config name: loads via `get_config()`, creates `RunContext`, assembles `ReporterStack`, calls `run_experiment()`, finalizes with `reporters.on_end()`
- All I/O is handled by reporters, not by the runner

## Known Issues and Dead Code

These are documented here so agents can account for them and clean them up
when appropriate.

- **Duplicated private helpers:** `_logistic`, `_logistic_batch`, and
  `_beta_dot_x` are duplicated verbatim in `src/data/fixed_objective.py` and
  `src/data/planted_logistic.py`. These could be factored into a shared
  utility module.

- **`clip_u` is removed / commented out:** `src/optimization/policy.py` has
  a commented-out import of `clip_u` from `common.py`. `apply_policy` does
  not clip actions. The softmax policy naturally maps to `(0.5, 1.5)` but
  linear and constant policies are unbounded.

- **`U_BOUNDS` in `common.py` is unused:** The constant `(0.5, 1.5)` is
  defined but never referenced anywhere in the pipeline.

- **`constant_step` in `common.py` is dead:** A `constant_step` function
  exists in `common.py` but the real implementation is `constant_step_size`
  in `steps.py`.

- **`Contract.__post_init__` is a no-op:** The bounds check is `pass`.
  `Contract` is defined in `models.py` but is never constructed in the
  experiment pipeline (actions are raw floats).

- **`stein_zeroth_order_grad` (scalar version) is unused:** Only the batch
  version `stein_zeroth_order_grad_batch` is called in the pipeline.

- **`build_objective_fns` is unused:** Only `build_batch_objective_fns` and
  `_build_objective_batch_fns` are used in the pipeline.

- **`policy_grad_theta` is unused in the pipeline:** The gradient through
  the policy is computed inline in `_run_estimated_grad_optimizer` and
  `run_lbfgs_theta` rather than calling this function.

- **`lbfgs_seed` is set but never consumed:** It defaults to `seed + 997`
  and is serialized in `to_dict()`, but `run_lbfgs_theta` does not use any
  RNG. Reserved for future stochastic extensions.

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
| `test_baseline_test.py` | End-to-end smoke test with baseline_test config |
| `test_config.py` | ExperimentConfig validation rules |
| `test_correctness_spec.py` | CorrectnessSpec gradient source modes |
| `test_early_stopping.py` | grad_norm_tol early stopping |
| `test_enabled_estimators.py` | Selective estimator execution |
| `test_experiment_configs.py` | Config registry (get_config, list_configs) |
| `test_file_step_logger.py` | FileStepLogger CSV output |
| `test_lbfgs_theta.py` | L-BFGS-B reduces objective, trace structure |
| `test_objective_batch.py` | Batch vs scalar consistency for both objectives |
| `test_objective_models.py` | FixedRegressionObjective value and gradient correctness |
| `test_planted_logistic_objective.py` | Planted logistic gradient at u_star and minimum |
| `test_plot_u_star.py` | u_star selection for plotting |
| `test_policy_batch.py` | policy_u_batch matches scalar for all kinds |
| `test_state_vector.py` | StateVector.sample shape |
| `test_step_rules.py` | Armijo backtracking on quadratic |
| `test_theta_contours.py` | Contour grid shapes, axis selection |
| `test_trace_theta_values.py` | theta_values and step_sizes recorded in traces |
| `test_verbose_config.py` | verbose flag defaults and serialization |
| `test_visualization_step_sizes.py` | step_sizes plot uses log y-scale |

## Documentation and Maintenance

### README.md
Documentation maintenance is part of implementation, not a separate follow-up task.

Update `README.md` whenever a change affects:
- project structure
- setup or execution steps
- configuration options
- outputs, logging, or reporting behavior
- public APIs
- expected experiment workflow
- mathematical expressions or objective definitions (use LaTeX math blocks/inline math for formulas instead of plain-text code blocks)

If no README changes are needed, explicitly verify that the existing README is still accurate.

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
