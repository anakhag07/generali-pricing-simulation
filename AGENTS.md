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

1. `src/objective/fixed_objective.py`
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

- **`src/objective/base.py`**
  - `StateVector`: frozen dataclass wrapping a 1D numpy array; has `sample(rng, dim)` static method
  - `Customer`: frozen dataclass with `x: StateVector`; has `sample(rng, state_dim)` static method
  - `Contract`: frozen dataclass with `u: float` (bounds check is a no-op; see Known Issues)
  - `ObjectiveResult`: frozen dataclass with `value` and `grad_u`
  - Protocols: `AcceptanceModel`, `LossModel`, `RevenueModel`, `ObjectiveModel`
  - `default_rng(seed)`: wrapper around `np.random.default_rng`

- **`src/objective/fixed_objective.py`** (source of truth for objective math)
  - `FixedRegressionAcceptance`: sigmoid acceptance with `beta_1` (positive) and `beta_2` (negative)
  - `FixedRegressionLoss`: linear loss with `beta_3` (positive)
  - `FixedRegressionRevenue`: linear revenue with `beta_4` (positive)
  - `FixedRegressionObjective`: composite objective; `from_parameters` classmethod, scalar + batch evaluation
  - `FixedRegressionBatch`: pre-computed batch for vectorized evaluation

- **`src/objective/planted_logistic.py`**
  - `PlantedLogisticObjective`: convex logistic objective with known optimum `u_star`
  - `PlantedLogisticBatch`: pre-computed batch for vectorized evaluation
  - `optimal_u()` method exposes the planted optimum

#### Data Layer (`src/data/`)

- Reserved for dataset adapters and external data-source integrations.

#### Model Layer (`src/model/`)

- **`src/model/policy.py`**
  - `PolicySpec`: frozen dataclass pairing `theta` (numpy array) with `kind` string
  - Policy kinds: `POLICY_CONSTANT`, `POLICY_LINEAR`, `POLICY_SOFTMAX`
  - `phi(x)` / `phi_batch(x_array)`: prepend bias term to features
  - `policy_u(theta, x, kind)` / `policy_u_batch(...)`: compute action from policy
  - `policy_grad_theta(theta, x, kind)`: compute `du/dtheta` (defined but unused in pipeline; see Known Issues)
  - `apply_policy(policy, x)`: convenience wrapper (does not clip `u`)

#### Optimization Layer (`src/optimization/`)

- **`src/optimization/solvers.py`**
  - `run_first_order_minimize(...)`: SciPy `minimize` (`L-BFGS-B`) solver using analytic u-gradients chained to theta gradients
  - `run_zeroth_order_minimize(...)`: SciPy `minimize` (`L-BFGS-B`) solver using Stein-estimated u-gradients chained to theta gradients
  - `run_spsa_minimize(...)`: SciPy `minimize` (`L-BFGS-B`) solver using SPSA theta-gradient estimates
  - Internal helpers build batched objective/gradient callables, optional mini-batch sampling, and optimization traces

- **`src/optimization/steps.py`**
  - `STEP_RULE_LBFGSB`, `STEP_RULE_CONSTANT`, `STEP_RULE_ARMIJO`, `STEP_RULES`
  - `constant_step_size(step_size)`: returns the step size unchanged
  - `armijo_backtracking_step_size(...)`: Armijo line search utility (not used by SciPy first/zeroth/SPSA solvers)

- **`src/optimization/common.py`**
  - `gaussian_noise(rng, shape)`: standard normal samples (used by zeroth-order estimator)
  - `U_BOUNDS`: defined but unused (see Known Issues)

- **`src/optimization/gradients/zeroth_order.py`**
  - `stein_zeroth_order_grad_batch(u_values, objective_fn, rng, n_samples, sigma)`: vectorized Stein gradient estimator (primary entry point)
  - `stein_zeroth_order_grad(...)`: scalar version (defined but unused in pipeline)

#### Experiment Layer (`src/experiments/`)

- **`src/experiments/config.py`**
  - `ExperimentConfig`: frozen dataclass with extensive `__post_init__` validation
  - `batch_size: int | None = None` enables stochastic mini-batch optimization when set
  - `CorrectnessSpec`: controls how "true" gradients are computed (`"exact"`, `"numdiff"`, `"none"`)
  - `verbose: bool = False` controls terminal output of per-step metrics
  - Preset-composition helpers: `make_*_objective`, `make_softmax_policy_spec`,
    `canonical_training_block`, `canonical_runtime_block`, and `build_experiment_config`

- **`src/experiments/configs/`** (preset registry)
  - `__init__.py`: `get_config(name)` and `list_configs()` registry
  - `fixed_regression_base.py`: base fixed-regression config (4D, L-BFGS-B step rule, W&B enabled)
  - `planted_logistic_base.py`: planted logistic base config (3D, L-BFGS-B step rule, 5000 steps, u*=1.1)

- **`src/experiments/defaults.py`**
  - `default_policy_spec(state_dim)`: returns softmax policy with `state_dim + 1` theta params

- **`src/experiments/helpers.py`** (largest file; orchestration + wrappers)
  - `resolve_true_grad_u_fn(objective_model, correctness)`: resolves the "true" gradient function from correctness spec
  - `run_first_order(...)`: wrapper delegating to `optimization.solvers.run_first_order_minimize`
  - `run_zeroth_order(...)`: wrapper delegating to `optimization.solvers.run_zeroth_order_minimize`
  - `run_spsa(...)`: wrapper delegating to `optimization.solvers.run_spsa_minimize`
  - Internal: `_numdiff_grad(...)` for finite-difference gradients

- **`src/experiments/run.py`**
  - `run_experiment(config, step_reporter)`: main runner; samples customers, runs enabled estimators, returns `ExperimentResult` (pure computation, no I/O)

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
  - `plot_objective_u_slice(...)`: objective and gradient vs u grid
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

- **Duplicated private helpers:** `_logistic`, `_logistic_batch`, and
  `_beta_dot_x` are duplicated verbatim in `src/objective/fixed_objective.py` and
  `src/objective/planted_logistic.py`. These could be factored into a shared
  utility module.

- **`clip_u` is removed / commented out:** `src/model/policy.py` has
  a commented-out import of `clip_u` from `common.py`. `apply_policy` does
  not clip actions. The softmax policy naturally maps to `(0.5, 1.5)` but
  linear and constant policies are unbounded.

- **`U_BOUNDS` in `common.py` is unused:** The constant `(0.5, 1.5)` is
  defined but never referenced anywhere in the pipeline.

- **`constant_step` in `common.py` is dead:** A `constant_step` function
  exists in `common.py` but the real implementation is `constant_step_size`
  in `steps.py`.

- **`Contract.__post_init__` is a no-op:** The bounds check is `pass`.
  `Contract` is defined in `objective/base.py` but is never constructed in the
  experiment pipeline (actions are raw floats).

- **`stein_zeroth_order_grad` (scalar version) is unused:** Only the batch
  version `stein_zeroth_order_grad_batch` is called in the pipeline.

- **`policy_grad_theta` is unused in the pipeline:** The gradient through
  the policy is computed inline in `src/optimization/solvers.py` rather than
  calling this function.

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
| `test_file_step_logger.py` | FileStepLogger CSV output |
| `test_minibatch_stochasticity.py` | Mini-batch determinism and full-batch equivalence |
| `test_minimize_orders.py` | SciPy first/zeroth/SPSA wrappers (decrease + seed determinism) |
| `test_model_package_exports.py` | model package API exports remain importable |
| `test_objective_batch.py` | Batch vs scalar consistency for both objectives |
| `test_objective_package_exports.py` | objective package API exports remain importable |
| `test_objective_models.py` | FixedRegressionObjective value and gradient correctness |
| `test_planted_logistic_objective.py` | Planted logistic gradient at u_star and minimum |
| `test_plot_u_star.py` | u_star selection for plotting |
| `test_policy_batch.py` | policy_u_batch matches scalar for all kinds |
| `test_run_context.py` | default output directory and run context paths |
| `test_state_vector.py` | StateVector.sample shape |
| `test_step_rules.py` | Armijo backtracking on quadratic |
| `test_theta_contours.py` | Contour grid shapes, axis selection |
| `test_trace_theta_values.py` | theta_values recorded in first/zeroth-order traces |
| `test_verbose_config.py` | verbose flag defaults and serialization |
| `test_visualization_step_sizes.py` | step_sizes plot uses log y-scale |
| `test_sweep_utils.py` | Override-grid expansion and preset sweep config generation |

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
