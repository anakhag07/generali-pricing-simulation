# Exact Finite-Policy LCB Validation Sweep

## Summary

Validate Proposition 11.2 on the finite class
\(\Pi=\{0,0.1,\ldots,1.0\}\) with

\[
V^\pi=\pi,\qquad
\widehat V^\pi=\pi+\pi Z^\pi,\qquad
Z^\pi\overset{\mathrm{i.i.d.}}{\sim}N(0,1).
\]

Sweep \(\delta\in\{0.50,0.20,0.10,0.05,0.01\}\) over 25 paired
noise seeds. Each seed draws one vector \(Z_s\) and reuses it for every
confidence level, isolating the effect of changing the confidence radius.
Optimize by evaluating the full finite class and selecting its exact maximum
LCB, so the script-style optimization error is \(\varepsilon=0\).

## Mathematical and seed contract

For \(K=11\), compute

\[
q_\delta=\Phi^{-1}\!\left(1-\frac{\delta}{2K}\right),\qquad
\mathcal E^\pi(\delta)=2\pi q_\delta,
\]

\[
\underline V^\pi_{\delta,s}
=\pi+\pi Z_s^\pi-\pi q_\delta.
\]

For each seed, evaluate all policies, select the smallest policy attaining the
maximum LCB, record the simultaneous confidence event, and check

\[
V^{\widehat\pi_{\delta,s}}
\ge V^{\widetilde\pi}-\mathcal E^{\widetilde\pi}(\delta)-\varepsilon
\quad\text{for every comparator }\widetilde\pi,
\qquad \varepsilon=0.
\]

Use master noise seed `20260807` and run seeds `101` through `125`. Derive one
child seed per run seed only; do not include \(\delta\), so all confidence
levels reuse the same policy-noise vector. The analytic simultaneous coverage
is

\[
\Pr(A_\delta)=\left(1-\frac{\delta}{K}\right)^K\ge 1-\delta.
\]

## Implementation

- Add a pure finite-policy LCB experiment module under `src/experiments/`.
- Route manifests with `kind: "finite_policy_lcb"` through the existing
  manifest entry point without changing legacy manifest behavior.
- Add `manifests/finite_policy_lcb_validation.json`, with one launch task per
  seed and all deltas evaluated inside that task.
- Record the formulas in `MATH.md` and update README, AGENTS, and generated
  documentation.
- Write per-policy, per-selection, coverage, oracle, and aggregate CSVs plus
  seed-level and aggregate plots under
  `results/finite-policy-lcb-validation/`.

## Acceptance criteria

- Formula, quantile, envelope, and zero-policy edge-case tests pass.
- Noise is deterministic across reruns, distinct across seeds, and paired
  across deltas.
- Exact selection has LCB gap \(\varepsilon=0\).
- Proposition 11.2 has no violations on the simultaneous confidence event.
- Analytic joint coverage is at least \(1-\delta\); empirical 25-seed coverage
  is reported descriptively with a binomial interval.
- Legacy manifests remain compatible, focused tests pass, and the committed
  sweep completes locally.
