# Nonlinear residual: MLP capacity and training duration

## Implementation checklist

- [ ] Preserve the original results/worktree and use the same manifest family.
- [ ] Parameterize hidden width; retain the 150/150 split and original-head bootstrap.
- [ ] Compare widths 2, 4, 8, 16 and 32 at 10000 steps, plus the original width-32 1000-step baseline and polynomial controls.
- [ ] Save solver objective/gradient diagnostics and explicit train/calibration R² and RMSE.
- [ ] Retain full-Gaussian true-residual scores and report all configurations without choosing a winner on calibration data.
- [ ] Regenerate bands, coverage, policy regret and recovery PDFs; add a compact fitting diagnostic PDF and table.
- [ ] Validate gradient/score definitions, paired data, original-head preservation, replay, PDFs and provenance.

## Fixed comparison contract

This is a continuation of `synthetic-polynomial-residual-nonlinear`. Keep n=300,
the explicit nonlinear residual, design/GP/noise/multiplier seeds, nominal 95%
asymmetric bands, B=1999 and S=100. Train on the first 150 rows; the remaining
150 are held out from fitting and calibrate the frozen head. They are reused
for model comparison, so they are not an untouched final test set. No training,
early stopping, hyperparameter choice or head replacement uses these responses.

The two SiLU hidden layers share width h. Parameter count is h²+7h+1:
19/45/121/369/1249 for widths 2/4/8/16/32. Keep L2 weight penalty 0.001,
initialization seed 20261104, gradient tolerance 1e-6 and ftol=1e-12 unchanged.
The 10000-step version restarts from the same initialization as the baseline;
the width-32 training trajectory must match its first 1000 steps. Changing width
changes the initialization array dimensions; it does not introduce a new seed.

Train and calibration scores compare predictions with the noisy residuals at
the original observed pairs. Report separate noiseless scores at those pairs.
The existing `r_squared` and `rmse` retain their original meaning: error against
the known true residual, averaged over the 150 calibration contexts and the full
Gaussian action distribution. Negative R² is permitted. Save all scores across
noise repetitions, alongside the original fixed-noise headline scores.

Reuse `mlp_head.py`, `structured_residual.py`, `structured_reporting.py` and the
existing manifest launcher. Extend only the width validator, fit diagnostics and
reporting. Keep repository optimizers for training and policy actions. For more
than three models, paginate residual slices instead of creating an excessively
wide figure. All output plots remain vector PDFs in the standard Matplotlib style.

The new named comparison lives at shared
`results/synthetic-polynomial-residual-nonlinear/capacity-comparison/`.
The parent baseline and its pinned source worktree remain unchanged. Useful
pilots are retained under shared `scratch-results/` with hash indexes.

Long training can produce nearly redundant hidden features. A numerically
rank-deficient head design is recorded as an unavailable bootstrap band; the
trained prediction and all fit scores remain available. Do not add ridge or
use a pseudoinverse. Coverage is reported conditional on an available band,
with its valid denominator, unavailable count and full-run coverage bounds.
Prediction-score averages include every repetition. An unavailable headline
band still prevents a band-derived policy; it cannot be silently substituted.
