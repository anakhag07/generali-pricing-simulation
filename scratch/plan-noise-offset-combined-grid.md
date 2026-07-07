# Plan: combined noise x theta-offset grid (homoskedastic + heteroskedastic)

Ongoing notes for the combined 2D sweep request. Branch:
`feature/noise-offset-combined-grid`.

## Request

The four existing sweeps under `results/` are two 1D slices per noise family:

- `homoskedastic-theta-offset-sweep`: vary init offset delta at fixed noise
  std sigma = 0.5; `heteroskedastic-theta-offset-sweep`: same at fixed noise
  growth gamma = 1.0.
- `homoskedastic-noise-sweep` / `heteroskedastic-noise-sweep`: vary sigma /
  gamma at fixed theta0 = 0 (note: theta0 = 0 is NOT on the offset axis, so
  these runs cannot be reused as offset points).

Goal: one figure per (noise family, estimator) where each curve is a fixed
noise level, x = theta offset, and y = (a) final theta distance to the clean
first-order truth and (b) clean-objective gap. Separate figures for
`finite_difference` and `stein_difference`. Use a small run budget.

## Semantics pinned down (from `scripts/run_sweep.py`)

- `theta_offset` = scalar delta added to EVERY coordinate of `BASE_THETA`:
  `theta0 = theta_truth + delta * 1` where `theta_truth` is the saved clean
  first-order theta from
  `results/planted_logistic_base/first_order_truth_20260701_174139/summary.json`
  (4-dim; so ||theta0 - theta_truth||_2 = 2 * delta). Axis labels must say this.
- Homoskedastic noise: `M_hat(x,u) = M(x,u) + eps(x,u)`, eps ~ N(0, sigma^2),
  frozen field keyed by (x, u, seed).
- Heteroskedastic noise: std(u) = gamma * |u - u*| (base_std = 0, centered at
  the planted optimum u* = 0.1), so it is noiseless at the optimum.
- Runs use `planted_logistic_base` + the same `COMMON_OVERRIDES` as
  `scripts/run_sweep.py` (L-BFGS-B, t_steps=1000, n_samples=1000, sigma=0.05,
  n_grad_samples=8, perturbation_space="u", denoised_exact correctness).
- Seed policy: `vary=("optimizer", "noise")`, anchor seed 7 (data/split/theta
  anchored, estimator perturbations + noise field redrawn per seed).

## Metrics (y axes)

1. Final theta distance: `||theta_hat - theta_truth^FO_clean||_2`.
2. Clean objective gap: `J_clean(theta_hat) - J_clean(theta_truth)` evaluated
   on the reconstructed train batch (same recipe as the exploitation
   diagnostic on branch `scratch/heteroskedastic-exploitation-diagnostic`:
   rebuild `PlantedLogisticObjective` from the summary config and resample the
   train split from `data_seed`/`split_seed`). `final_value` in summaries is
   the NOISY objective, so recomputation is required.

## Grid (small budget)

- Offsets delta: {0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0} (9 points,
  subset of the existing 24-point grid so the old fixed-noise runs are reused).
- Homoskedastic sigma levels: {0, 0.1, 2.0} new + sigma=0.5 reused from
  `homoskedastic-theta-offset-sweep` -> 4 curves.
- Heteroskedastic gamma levels: {0, 0.25, 4.0} new + gamma=1.0 reused from
  `heteroskedastic-theta-offset-sweep` -> 4 curves.
- Seeds: (7, 8, 9); reused curves filtered to the same seeds so error bars are
  comparable (old sweeps ran seeds 7-11).
- New runs: 2 families x 3 noise levels x 9 offsets x 3 seeds = 162 runs
  (~0.5 s optimizer time each; minutes end to end).

## Implementation

- New driver `scripts/run_noise_offset_grid.py`, modeled on
  `scripts/run_sweep.py` and importing its reusable pieces (COMMON_OVERRIDES,
  BASE_THETA, objective builders, skip-completed helpers) so run settings stay
  identical.
- New projects: `results/homoskedastic-noise-offset-grid/` and
  `results/heteroskedastic-noise-offset-grid/`, variant names
  `noise-std-<sigma>__theta-offset-<delta>` / `noise-growth-<gamma>__theta-offset-<delta>`.
- Collector scans new grid dirs + the two old theta-offset sweep dirs (tagging
  their fixed noise level), writes `noise_offset_grid_finals.csv` and per
  estimator two-panel figures (theta distance | clean objective gap) under
  each project dir. `--plots-only` regenerates plots without running.
- Launch-aware; default `--launch auto` submits one serial CPU Slurm job.

## Axis labels (interpretable/mathematical)

- x: init offset delta with the defining formula
  `theta_0 = theta*_FO,clean + delta * 1` spelled out in the label.
- y (panel 1): `||theta_hat - theta*_FO,clean||_2`.
- y (panel 2): `J_clean(theta_hat) - J_clean(theta*_FO,clean)` (train batch).
- Curve legend: `sigma` (constant noise std) or `gamma` (noise growth in
  `std(u) = gamma |u - u*|`), with the noise model in the figure title.

## Status log

- [x] Inspected existing sweeps, pinned down theta_offset/noise semantics.
- [x] Branch `feature/noise-offset-combined-grid` created from main.
- [x] Wrote `scripts/run_noise_offset_grid.py` (162 tasks = 2 families x 3 new
      noise levels x 9 offsets x 3 seeds; 54 variants).
- [x] Smoke test passed: `--plots-only` built reused-curve figures; single
      local task (`--launch local --task-index 0`) ran clean and stayed at the
      truth theta for sigma=0, delta=0 (smoke output deleted before submit so
      the serial skip logic would not drop seeds 8/9 for that variant).
- [x] Submitted serial CPU Slurm job 17382892 (all 162 runs + plot regen).
- [ ] Verify job completion; sanity-check combined curves.
- [ ] Small test under `tests/scripts/`, AGENTS.md entry, commit + summary.

## Output locations

- `results/homoskedastic-noise-offset-grid/noise_offset_grid_{finite_difference,stein_difference}.png`
  (+ `noise_offset_grid_finals.csv`), same under
  `results/heteroskedastic-noise-offset-grid/`.
- Slurm log: `results/slurm/planted-noise-offset-grid-17382892.out` (name may
  differ; `%x-%j.out` pattern).
