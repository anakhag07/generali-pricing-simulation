# Generali Pricing Simulation

Pricing simulation and optimization demo using exact and zeroth-order Stein gradients.

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
(numpy >= 1.24, matplotlib >= 3.7, scipy >= 1.10).

To run tests:

```bash
conda activate simulation_env
pip install -e ".[dev]"
pytest -q
```

If you use a different environment name or tool, update `AGENTS.md` to match
your local setup.


Runtime dependencies live in `requirements.txt` and mirror `pyproject.toml`
(numpy >= 1.24, matplotlib >= 3.7, scipy >= 1.10).


## What This Does

1. Samples a batch of synthetic customer feature vectors from a standard normal
   distribution.
2. Evaluates a deterministic objective combining acceptance probability,
   expected loss, and revenue.
3. Optimizes a pricing policy parameter `theta` using one or more of:
   - First-order (exact gradient) descent
   - Zeroth-order Stein gradient estimator
   - L-BFGS-B baseline (SciPy)
4. Saves run artifacts under `runs/<experiment_name>/<timestamp>/`:
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
p^*(x) = \sigma\left(\alpha u^* + \beta^{\top} x + b \right)
$$

$$
L(u; x) = \log\left(1 + e^{z(u, x)}\right) - p^*(x)\,z(u, x)
$$

This function is convex in `u` and has a known minimum at `u*` for every `x`.
When this objective is active, the true optimum is exposed in logs and plots.
The preset config `planted_logistic` wires this in.

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

list_configs()              # -> ("baseline_fixed", "baseline_test", "custom", "planted_logistic")
config = get_config("custom")  # -> ExperimentConfig
```

Control which configs run by editing the `RUN_CONFIGS` list in `main.py`:

```python
RUN_CONFIGS = ["custom"]  # add or remove preset names here
```

### Available Presets

| Name | Objective | Key Settings |
|---|---|---|
| `baseline_fixed` | FixedRegression (3D) | 10 samples, constant step, 100 steps |
| `baseline_test` | FixedRegression (3D) | 2 samples, 1 step, plot=False (smoke test) |
| `custom` | FixedRegression (2D) | 100 samples, Armijo step, 1000 steps |
| `planted_logistic` | PlantedLogistic (3D) | 20 samples, Armijo step, 5000 steps, u*=1.1 |

Edit `custom.py` for ad-hoc experiments.

### ExperimentConfig Fields

Each `ExperimentConfig` includes:

| Field | Default | Description |
|---|---|---|
| `state_dim` | *required* | Dimension of customer feature vector |
| `objective_model` | *required* | `FixedRegressionObjective` or `PlantedLogisticObjective` |
| `policy_spec` | *required* | `PolicySpec` with theta and kind |
| `n_samples` | *required* | Batch size for customer states |
| `step_rule` | *required* | `"constant"` or `"armijo"` |
| `seed` | 7 | RNG seed for reproducibility |
| `t_steps` | 100 | Number of optimization steps |
| `step_size` | 0.01 | Constant step size, or initial step for Armijo |
| `grad_norm_tol` | None | Early stopping threshold on theta gradient norm |
| `sigma` | 0.1 | Perturbation scale for zeroth-order estimator |
| `n_grad_samples` | 64 | Number of perturbations for zeroth-order estimator |
| `lbfgs_maxiter` | 200 | Max iterations for L-BFGS-B |
| `verbose` | False | Print per-step metrics to terminal |
| `plot` | True | Generate plots at end of run |
| `plot_dir` | `"plots"` | Subdirectory name for plots |
| `enabled_estimators` | `("first_order", "zeroth_order", "lbfgs")` | Which methods to run |
| `correctness` | `CorrectnessSpec()` | Controls "true" gradient computation |

### Step-Size Rules

Set `step_rule` to `"constant"` or `"armijo"`:

- **Constant:** uses `step_size` as a fixed step each iteration.
- **Armijo:** backtracking line search starting from `step_size`, with
  parameters `c=1e-4`, `shrink=0.5`, `max_backtracks=20`, `min_step=1e-6`.
  When Armijo is active, a `step_sizes.png` plot is saved.

### Early Stopping

Set `grad_norm_tol` to stop when the theta gradient norm falls below the
threshold. First-order and zeroth-order optimizers check this each step;
L-BFGS-B passes the value as `gtol`.

### Enabled Estimators

Control which methods run (and appear in plots/logs):

```python
enabled_estimators=("zeroth_order", "first_order", "lbfgs")
```

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
| `JsonReporter` | `summary.json` | Full experiment result including config |
| `PlotReporter` | `plots/*.png` | loss_curves, gradient_norms, objective_u_slice, step_sizes (Armijo only), theta_objective_contours (if theta dim >= 2) |

All outputs are saved under `runs/<experiment_name>/<timestamp>/`.

## Model-to-Code Mapping

```text
x (customer features)            -> StateVector (src/data/models.py)
customer                          -> Customer (src/data/models.py)
a(x, u) acceptance probability    -> FixedRegressionAcceptance (src/data/fixed_objective.py)
l(x) expected loss                -> FixedRegressionLoss (src/data/fixed_objective.py)
r(u) revenue                      -> FixedRegressionRevenue (src/data/fixed_objective.py)
f(u; x) fixed objective           -> FixedRegressionObjective (src/data/fixed_objective.py)
L(u; x) planted objective         -> PlantedLogisticObjective (src/data/planted_logistic.py)
oracle gradient API               -> FixedRegressionObjective.evaluate (src/data/fixed_objective.py)
policy u = f(theta, x)            -> PolicySpec, apply_policy (src/optimization/policy.py)
zeroth-order Stein estimator      -> stein_zeroth_order_grad_batch (src/optimization/gradients/zeroth_order.py)
step-size rules (constant/Armijo) -> constant_step_size, armijo_backtracking_step_size (src/optimization/steps.py)
experiment runner / config        -> ExperimentConfig, run_experiment (src/experiments/run.py)
optimization helper routines      -> run_first_order, run_zeroth_order, run_lbfgs_theta (src/experiments/helpers.py)
result data structures            -> EstimatorResult, ExperimentResult, OptimizationTrace (src/experiments/results.py)
reporting / I/O                   -> ReporterStack, ConsoleReporter, etc. (src/experiments/reporters.py)
plots                             -> plot_loss_curves, plot_gradient_norms, etc. (src/experiments/visualization.py)
config presets                    -> src/experiments/configs/ (get_config, list_configs)
```

## Optimization Methods

- **First-order exact gradient:** uses the analytic gradient of the objective
  with respect to `u`, then chains through the policy gradient `du/dtheta` to
  update `theta`.
- **Zeroth-order Stein estimator:** uses only objective value evaluations at
  perturbed actions. Estimates the gradient via
  $\mathbb{E}[f(u + \sigma\varepsilon)\,\varepsilon] / \sigma$.
- **L-BFGS-B baseline:** uses SciPy's `minimize` to optimize `theta` directly
  with analytic gradients.

All three methods update `theta` (the policy parameter), not `u` directly.

## Project Structure

```
main.py                                 Demo entry point; RUN_CONFIGS list
src/
  data/
    models.py                           Core data classes (StateVector, Customer, Contract)
                                        and objective protocols (ObjectiveModel, etc.)
    fixed_objective.py                  Fixed regression objective implementation
    planted_logistic.py                 Planted convex logistic objective with known optimum
  experiments/
    config.py                           ExperimentConfig and CorrectnessSpec dataclasses
    configs/                            Preset configurations
      baseline_fixed_objective.py       3D fixed regression baseline
      baseline_test.py                  Minimal smoke-test config
      custom.py                         Ad-hoc experiment config
      planted_logistic.py              Planted logistic preset
    defaults.py                         Default helpers (default_policy_spec)
    helpers.py                          Core optimization routines (run_first_order,
                                        run_zeroth_order, run_lbfgs_theta)
    run.py                              Experiment runner (returns results, no I/O)
    results.py                          Result data structures (OptimizationTrace, etc.)
    reporters.py                        Reporter protocol, ReporterStack, RunContext,
                                        ConsoleReporter, FileStepLogger, JsonReporter,
                                        PlotReporter
    logging.py                          Console logging helpers (log_step, log_summary)
    visualization.py                    Matplotlib plotting utilities
  optimization/
    common.py                           Shared helpers (gaussian_noise)
    policy.py                           PolicySpec, policy kinds, apply_policy
    steps.py                            Step-size rules (constant, Armijo backtracking)
    gradients/
      zeroth_order.py                   Stein zeroth-order gradient estimator
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
- End-to-end smoke test (`test_baseline_test.py`)
- Enabled estimators filtering (`test_enabled_estimators.py`)
- Early stopping (`test_early_stopping.py`)
- Visualization outputs (`test_visualization_step_sizes.py`,
  `test_theta_contours.py`, `test_plot_u_star.py`)
- Reporter I/O (`test_file_step_logger.py`)

Tests use explicit seeds for determinism and avoid filesystem I/O or plotting
where possible.

## Reproducibility

The demo uses a fixed RNG seed (default 7, configurable per
`ExperimentConfig.seed`). The objective is deterministic given a fixed
configuration and state sample batch. L-BFGS-B uses a separate seed field
(`lbfgs_seed`, defaults to `seed + 997`) reserved for future stochastic
extensions.
