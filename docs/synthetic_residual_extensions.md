# Interaction and nonlinear synthetic residual experiments

## Implementation checklist

- [ ] Extend additive polynomial queries with original-action interaction columns.
- [ ] Preserve the existing family, launcher, frozen GP design and paired streams.
- [ ] Keep the trained MLP and its original head fixed during calibration.
- [ ] Fit polynomial nonlinear controls on the same first 150 rows.
- [ ] Bootstrap head perturbations on the other 150; never fit observed calibration targets.
- [ ] Retain whole-line interaction coverage and label nonlinear finite-grid coverage.
- [ ] Use repository solvers for MLP fitting and every reported policy endpoint.
- [ ] Validate independent bootstrap refits, gradients, leakage, replay and domain labels.
- [ ] Produce the five established PDF reports, including context slices and a single-panel figure 05.
- [ ] Preserve historical experiments and archive useful pilot diagnostics with checksums.

## Resolved contract

Keep the original contextual quadratic, n=300, exact squared-exponential GP
restriction, observation/multiplier/optimizer seed roots, B=1999, S=100 and
asymmetric transfer. The new manifests remain in `polynomial_residual` mode.
The interaction residual adds 0.35*x1*a + 0.20*x2*a^2 to g4; compare action
degree six with interaction orders 0, 1 and 2, using all 300 rows for OLS.

The nonlinear residual is 0.2+0.1*x1+0.25*a+0.35*x1*a plus
0.8*tanh(x2)*exp(-(a-0.5)^2/(2*0.4^2)). Compare the no-interaction and
order-two polynomial bases and a two-hidden-layer, width-32 SiLU MLP.
Rows 0:150 fit each point predictor. Rows 150:300 calibrate head perturbations
and define the evaluation contexts. They are calibration data, not an independent
test set. MLP input standardization uses only training rows. Full-batch repository
L-BFGS-B fits all network weights once per repetition, with fixed initialization,
L2 weight penalty 0.001, a 1,000-step cap, gradient tolerance 1e-6 and ftol=1e-12.
These are declared first-version settings, not tuned to true residual scores.

For the frozen feature design P on calibration rows, retain the training head b0
and errors e=r-P*b0. Bootstrap perturbations are db=(P'P)^(-1)P'(signs*e).
Bootstrap head copies are b0+db; b0 is never replaced by a fit to the observed
calibration responses or an average bootstrap head. Use the existing QR
coefficient-ellipsoid quantile and asymmetric transfer. This is empirical
calibration around a frozen predictor, not an asserted OLS confidence theorem.
Record rank/conditioning and training/calibration errors; fail explicitly on
rank deficiency rather than silently applying a pseudoinverse or ridge.

Nonlinear coverage checks 150 contexts times 401 action points on
[0.5-3*sqrt(0.45),0.5+3*sqrt(0.45)], then a nested 1601-point grid with the same
band. Neither is a continuous-domain certificate. True-residual R2/RMSE and
average width integrate over the full Gaussian action distribution. Policy
optimization is continuous and unconstrained; no grid chooses an action.
Report direct J(0.5)-J(converged actions), repository convergence and any actions
outside the coverage window. Only polynomial interaction objectives receive
the existing exact polynomial global-gap check.

## Module responsibilities

`polynomial.py` owns interaction features, polynomial fibers and certificates.
`mlp_head.py` owns SiLU fitting, frozen predictions and calibration-only head
bootstrap. `structured_residual.py` owns the explicit nonlinear function,
model specifications, saved-state restoration and finite-grid evaluation.
The existing `polynomial_study.py` continues to own manifests, immutable source
contracts, repetitions and repository policies. `structured_reporting.py`
replays the new models through the established five-figure reporting pattern.
Existing manifests keep their original behavior and saved key conventions.
