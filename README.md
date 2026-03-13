# Generali Pricing Simulation

Pricing simulation and optimization demo using exact and Gaussian-Stein gradients.

## Quickstart


```bash
# Conda
conda create -n simulation_env python=3.11
conda activate simulation_env
pip install -e .
python main.py
```

Or with venv:

```bash
# Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
python main.py
```

Runtime dependencies live in `requirements.txt` and mirror `pyproject.toml`
(numpy >= 1.24, matplotlib >= 3.7, scipy >= 1.10, wandb >= 0.19).

To run tests:

```bash
conda activate simulation_env
pip install -e ".[dev]"
pytest -q
```

If you use a different environment name or tool, update `AGENTS.md` to match
your local setup.


Runtime dependencies live in `requirements.txt` and mirror `pyproject.toml`
(numpy >= 1.24, matplotlib >= 3.7, scipy >= 1.10, wandb >= 0.19).


## What This Does

1. Samples a batch of synthetic customer feature vectors from a standard normal
   distribution.
2. Evaluates a deterministic objective combining acceptance probability,
   expected loss, and revenue.
3. Optimizes a pricing policy parameter `theta` using one or more of:
    - First-order (exact gradient) descent
    - Zeroth-order Stein gradient estimator
    - SPSA estimator
4. Saves run artifacts under `outputs/<experiment_name>/<timestamp>/`:
   - `summary.json` -- full result payload
   - `steps.csv` -- per-step metrics for every estimator
   - `plots/` -- loss curves, gradient norms, objective slices, step sizes,
     and theta-objective contour plots

## Mathematical Model

### Decision Variables and Components

$$
x \sim \mathcal{N}(0, I),\quad \theta \in \mathbb{R}^p,\quad u = \pi_\theta(x)
$$

$$
a(x, u) \in (0, 1),\quad \ell(x) \ge 0,\quad r(u)
$$

$$
f(u; x) = a(x, u)\,\big(\ell(x) - r(u)\big),
\qquad
\min_{\theta}\; \mathbb{E}_x\big[f(\pi_\theta(x); x)\big]
$$

### Fixed Regression Objective

The default objective uses an explicit parametric form:

$$
a(x, u) = \sigma\!\left(\beta_1^\top x + \beta_2 u\right),
\qquad \beta_1 > 0,\; \beta_2 < 0
$$

$$
\ell(x) = \beta_3^\top x, \qquad \beta_3 > 0
$$

$$
r(u) = \beta_4 u, \qquad \beta_4 > 0
$$

$$
f(u; x) = a(x, u)\,\big(\ell(x) - r(u)\big)
$$

The beta values are configurable via `FixedRegressionObjective`. `beta_2` must
be negative so acceptance decreases with higher prices. When this objective is
used, the demo plots the objective and gradient over a grid of `u` values.

### Planted Logistic Objective

For estimator comparisons with a known optimum, use the planted logistic
objective:

$$
z(u, x) = \alpha u + \beta^\top x + b
$$

$$
p^*(x) = \sigma\left( \alpha u^{\*} + \beta^{\top} x + b \right)
$$

$$
L(u; x) = \log\left(1 + e^{z(u, x)}\right) - p^*(x)\,z(u, x)
$$

This function is convex in `u` and has a known minimum at `u*` for every `x`.
When this objective is active, the true optimum is exposed in logs and plots.
The preset config `planted_logistic_base` wires this in.

## Policy System

A `PolicySpec` pairs a parameter vector `theta` with a policy kind. The policy
maps `(theta, x)` to an action `u`.

| Kind | Formula | `u` range | `theta` size |
|---|---|---|---|
| `constant` | $u = \theta_0$ | unbounded | 1 |
| `linear` | $u = \theta^\top \phi(x)$ | unbounded | `state_dim + 1` |
| `softmax` | $u = 0.5 + \sigma\!\left(\theta^\top \phi(x)\right)$ | $(0.5, 1.5)$ | `state_dim + 1` |

$\phi(x)$ prepends a bias term of $1.0$ to the feature vector: $[1, x_1, \ldots, x_d]$.

The softmax policy is the default (via `default_policy_spec`). It naturally
constrains actions to `(0.5, 1.5)`. Linear and constant policies have no
built-in bounds.

## Experiment Configuration

### Config Registry

Preset configs live in `src/experiments/configs/`. The registry exposes two
functions:

```python
from experiments.configs import get_config, list_configs

list_configs()              # -> ("fixed_regression_base", "planted_logistic_base")
config = get_config("fixed_regression_base")  # -> ExperimentConfig
```

Preset files compose `ExperimentConfig` using canonical helper blocks defined in
`src/experiments/config.py` (objective, policy, training, and runtime).

Control which configs run by editing the `RUN_CONFIGS` list in `main.py`:

```python
RUN_CONFIGS = ["fixed_regression_base"]  # add or remove preset names here
```

For parameter sweeps based on one preset with top-level overrides, use:

```bash
python scripts/run_sweep.py
```

Edit `BASE_PRESET` and `OVERRIDE_GRID` in `scripts/run_sweep.py`.

### Available Presets

| Name | Objective | Key Settings |
|---|---|---|
| `fixed_regression_base` | FixedRegression (4D) | 100 samples, L-BFGS-B step rule, 50000 steps, W&B enabled |
| `planted_logistic_base` | PlantedLogistic (3D) | 20 samples, L-BFGS-B step rule, 5000 steps, u*=1.1 |

Edit `fixed_regression_base.py` for ad-hoc fixed-regression experiments.

### ExperimentConfig Fields

Each `ExperimentConfig` includes:

| Field | Default | Description |
|---|---|---|
| `state_dim` | *required* | Dimension of customer feature vector |
| `objective_model` | *required* | `FixedRegressionObjective` or `PlantedLogisticObjective` |
| `policy_spec` | *required* | `PolicySpec` with theta and kind |
| `n_samples` | *required* | Batch size for customer states |
| `batch_size` | None | Mini-batch size for stochastic optimization (`None` uses full batch) |
| `step_rule` | *required* | `"l-bfgs-b"` (recommended); `"constant"`/`"armijo"` accepted for legacy compatibility |
| `seed` | 7 | RNG seed for reproducibility |
| `t_steps` | 100 | Number of optimization steps |
| `step_size` | 0.01 | Legacy compatibility field (not used by SciPy L-BFGS-B updates) |
| `grad_norm_tol` | None | Early stopping threshold on theta gradient norm |
| `ftol` | None | SciPy L-BFGS-B relative function-improvement tolerance |
| `sigma` | 0.1 | Perturbation scale for Gauss-Stein estimator |
| `n_grad_samples` | 64 | Number of perturbations for Gauss-Stein estimator |
| `verbose` | False | Print per-step metrics to terminal |
| `plot` | True | Generate plots at end of run |
| `plot_dir` | `"plots"` | Subdirectory name for plots |
| `enabled_estimators` | `("first_order", "gauss_stein")` | Which methods to run (`"spsa"` also supported) |
| `wandb_enabled` | `False` | Enable Weights & Biases logging |
| `wandb_project` | `None` | W&B project name |
| `wandb_entity` | `None` | W&B entity/user/team |
| `wandb_group` | `None` | W&B group label for run grouping |
| `wandb_job_type` | `"experiment"` | W&B job type |
| `wandb_tags` | `()` | W&B tags |
| `wandb_mode` | `"online"` | W&B mode (`"online"`, `"offline"`, `"disabled"`) |
| `wandb_log_plots` | `True` | Upload generated PNG plots to W&B at run end |
| `wandb_estimator_allowlist` | `None` | Optional estimator filter for W&B logging only |
| `correctness` | `CorrectnessSpec()` | Controls "true" gradient computation |

### Step-Size Rules

Set `step_rule` to `"l-bfgs-b"`:

- **First-order / Gauss-Stein / SPSA:** all estimators run through SciPy
  `minimize` (`L-BFGS-B`) and rely on the solver's internal line search.
  `t_steps` is passed as `maxiter`, `grad_norm_tol` as `gtol`, and `ftol` (if
  provided) as `ftol`.
- `"constant"` and `"armijo"` remain accepted `step_rule` values for backward
  compatibility, but they do not control updates in SciPy-driven methods.

### Early Stopping

Set `grad_norm_tol` to stop when the theta gradient norm falls below the
threshold. SciPy-driven optimizers pass this as `gtol`.

Set `ftol` to tune relative objective-improvement stopping in SciPy L-BFGS-B.

### Mini-Batch Stochasticity

- Set `batch_size` to an integer in `[1, n_samples]` to run first-order,
  Gauss-Stein, and SPSA on random customer mini-batches at each objective/
  gradient call.
- Keep `batch_size=None` (default) to preserve deterministic full-batch
  behavior.
- Runs remain reproducible with fixed `seed`.

### Enabled Estimators

Control which methods run (and appear in plots/logs):

```python
enabled_estimators=("gauss_stein", "first_order", "spsa")
```

### Weights & Biases Streaming

Enable W&B by setting `wandb_enabled=True` in your preset config and logging in:

```bash
wandb login
```

Example config fields:

```python
CONFIG = ExperimentConfig(
    # ... existing fields ...
    enabled_estimators=("gauss_stein", "first_order", "spsa"),
    wandb_enabled=True,
    wandb_project="pricing-sim",
    wandb_entity=None,
    wandb_group="ablation-n-grad-samples",
    wandb_tags=("custom", "steins", "spsa"),
    wandb_mode="online",  # or "offline"
    wandb_estimator_allowlist=("gauss_stein", "spsa"),  # optional
)
```

What gets streamed:

- Per-step curves under namespaced keys such as
  `curve/gauss-stein/objective`, `curve/spsa/objective`,
  `curve/first_order/theta_grad_norm`, etc.
- Curve metrics are registered with per-estimator step axes
  (`curve/<estimator>/step`), so each estimator panel starts at step `0`.
- Final summaries under `final/<estimator>/*`.
- Plot PNGs under `plots/*` when `wandb_log_plots=True`.

This setup lets you filter/sort runs by config fields (for example
`n_grad_samples`, `n_samples`, `sigma`) and toggle estimator lines in W&B
charts without rerunning experiments.

### Correctness Settings

`CorrectnessSpec` controls how "true" gradients are computed for comparison
plots and logs:

```python
from experiments.config import CorrectnessSpec

correctness=CorrectnessSpec(
    gradient_source="numdiff",   # "exact", "numdiff", or "none"
    numdiff_method="central",    # "central", "forward", or "backward"
    numdiff_step=1e-4,
    numdiff_aggregate="per-sample",
    numdiff_bounds=(0.5, 1.5),   # optional clamp for finite-difference points
)
```

When `gradient_source="exact"`, the objective's analytic `grad_u` is used.
When `"numdiff"`, finite differences are computed per sample. When `"none"`,
no true gradient is recorded.

## Reporter System and Outputs

`main.py` assembles a `ReporterStack` that delegates to four reporters:

| Reporter | Output | Notes |
|---|---|---|
| `ConsoleReporter` | Terminal | Per-step output gated by `verbose` |
| `FileStepLogger` | `steps.csv` | CSV: method, step, u, value, grad_norm, step_size |
| `JsonReporter` | `summary.json` | Full experiment result including config and final theta L2 metrics |
| `PlotReporter` | `plots/*.png` | loss_curves, gradient_norms, objective_u_slice, step_sizes (Armijo only), theta_objective_contours (if theta dim >= 2) |

All outputs are saved under `outputs/<experiment_name>/<timestamp>/`.

## Model-to-Code Mapping

```text
x (customer features)            -> StateVector (src/objective/base.py)
customer                          -> Customer (src/objective/base.py)
a(x, u) acceptance probability    -> FixedRegressionAcceptance (src/objective/fixed_objective.py)
l(x) expected loss                -> FixedRegressionLoss (src/objective/fixed_objective.py)
r(u) revenue                      -> FixedRegressionRevenue (src/objective/fixed_objective.py)
f(u; x) fixed objective           -> FixedRegressionObjective (src/objective/fixed_objective.py)
L(u; x) planted objective         -> PlantedLogisticObjective (src/objective/planted_logistic.py)
oracle gradient API               -> FixedRegressionObjective.evaluate (src/objective/fixed_objective.py)
policy u = f(theta, x)            -> PolicySpec, apply_policy (src/model/policy.py)
Optimization entry point            -> Optimization.solve (src/optimization/base.py)
Gradient method objects             -> FirstOrderGradient, GaussSteinGradient, SPSAGradient (src/optimization/gradients/methods.py)
SciPy wrapper helpers               -> run_first_order_minimize, run_gauss_stein_minimize, run_spsa_minimize (src/optimization/solvers.py)
step-size rules (legacy support)  -> constant_step_size, armijo_backtracking_step_size (src/optimization/steps.py)
experiment runner / config        -> ExperimentConfig, run_experiment (src/experiments/run.py)
config sweep utilities            -> override grid + sweep runner helpers (src/experiments/sweep_utils.py)
optimization helper wrappers      -> run_first_order, run_gauss_stein, run_spsa (src/experiments/helpers.py)
result data structures            -> EstimatorResult, ExperimentResult, OptimizationTrace (src/experiments/results.py)
reporting / I/O                   -> ReporterStack, ConsoleReporter, etc. (src/experiments/reporters.py)
console logging helpers           -> log_step, log_summary (src/reporting/logging.py)
plots                             -> plot_loss_curves, plot_gradient_norms, etc. (src/reporting/visualization.py)
config presets                    -> src/experiments/configs/ (get_config, list_configs)
```

## Optimization Methods

The optimization entry point is the `Optimization` class. Instantiate it with
an objective, policy kind, optimization algorithm (currently `"l-bfgs-b"`),
and a gradient object (`FirstOrderGradient`, `GaussSteinGradient`, or
`SPSAGradient`), then call `solve(theta_start)`.

- **First-order exact gradient:** uses the analytic gradient of the objective
  with respect to `u`, chains through the policy gradient `du/dtheta`, and
  optimizes `theta` via SciPy `minimize` (`L-BFGS-B`).
- **Gauss-Stein estimator:** uses only objective value evaluations at
  perturbed actions, estimates gradient via
  $\mathbb{E}[f(u + \sigma\varepsilon)\,\varepsilon] / \sigma$.
  The resulting theta gradient is passed to SciPy `minimize` (`L-BFGS-B`).
- **SPSA estimator:** estimates the theta gradient with two-sided random
  perturbations. For Rademacher directions $\Delta\in\{-1,+1\}^p$,
  $$\hat g(\theta;\Delta)=\frac{J(\theta+\sigma\Delta)-J(\theta-\sigma\Delta)}{2\sigma}\,\Delta,$$
  and averages across `n_grad_samples` directions before passing the gradient
  to SciPy `minimize` (`L-BFGS-B`).
When `batch_size` is set, these estimators optimize mini-batch objectives
$J_t(\theta)$ sampled from the customer pool; with `batch_size=None`, they use
full-batch objectives.

All methods update `theta` (the policy parameter), not `u` directly.

## Project Structure

```
main.py                                 Demo entry point; RUN_CONFIGS list
scripts/
  run_sweep.py                          Preset sweep runner with override grid
src/
  data/
    __init__.py                         Reserved for dataset adapters and data-source integrations
  objective/
    base.py                             Core dataclasses/protocols (StateVector, Customer, ObjectiveModel, etc.)
    fixed_objective.py                  Fixed regression objective implementation
    planted_logistic.py                 Planted convex logistic objective with known optimum
  model/
    policy.py                           PolicySpec, policy kinds, apply_policy
  experiments/
    config.py                           ExperimentConfig and CorrectnessSpec dataclasses
    configs/                            Preset configurations
      fixed_regression_base.py          Base fixed-regression preset
      planted_logistic_base.py          Base planted-logistic preset
    defaults.py                         Default helpers (default_policy_spec)
    helpers.py                          Core optimization routines (run_first_order,
                                        run_gauss_stein, run_spsa)
    run.py                              Experiment runner (returns results, no I/O)
    sweep_utils.py                      Override-grid helpers and preset sweep execution
    results.py                          Result data structures (OptimizationTrace, etc.)
    reporters.py                        Reporter protocol, ReporterStack, RunContext,
                                        ConsoleReporter, FileStepLogger, JsonReporter,
                                        PlotReporter
  reporting/
    logging.py                          Console logging helpers (log_step, log_summary)
    visualization.py                    Matplotlib plotting utilities
  optimization/
    base.py                             Optimization class with solve() and SciPy minimize integration
    solvers.py                          Compatibility wrappers constructing Optimization + gradient objects
    steps.py                            Step-rule constants (l-bfgs-b + legacy constant/armijo)
    gradients/
      methods.py                        Gradient method classes (first-order, Gauss-Stein, SPSA)
tests/                                  Flat test layout (pytest)
```

## Testing

Tests live in `tests/` in a flat layout. Run them with:

```bash
pytest -q
```

Key test areas:
- Config validation (`test_config.py`, `test_correctness_spec.py`)
- Objective correctness (`test_objective_models.py`, `test_objective_batch.py`,
  `test_planted_logistic_objective.py`)
- Policy batch consistency (`test_policy_batch.py`)
- Step-size rules (`test_step_rules.py`)
- End-to-end smoke test (`test_baseline_test.py`, using fixed_regression_base overrides)
- Enabled estimators filtering (`test_enabled_estimators.py`)
- Early stopping (`test_early_stopping.py`)
- Visualization outputs (`test_visualization_step_sizes.py`,
  `test_theta_contours.py`, `test_plot_u_star.py`)
- Reporter I/O (`test_file_step_logger.py`)
- Sweep utilities (`test_sweep_utils.py`)

Tests use explicit seeds for determinism and avoid filesystem I/O or plotting
where possible.

## Reproducibility

The demo uses a fixed RNG seed (default 7, configurable per
`ExperimentConfig.seed`). The objective is deterministic given a fixed
configuration and state sample batch.
