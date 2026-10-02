# Dedicated joint N/B coverage sweep

The user requests empirical coverage while increasing N and B together, using
more independent datasets. Report every setting, with no new optimization
objective, configuration selection, Pareto ranking or monotonicity constraint.

Implementation plan and representation contract:

- Keep f(a)=5a-5a^2, iid standard-normal training actions, sigma=1,
  delta=0.05, pairs resampling, original-fit standard-error denominator and
  the `higher` bootstrap quantile. Use N=B at 20,50,100,200,500,1000,2000,3000,5000.
- Use 2000 independent outer datasets. Named design, observation and row-index
  seed roots remain unchanged. Within each outer dataset, original observations
  are nested prefixes; bootstrap rows use floor(N*U[:B,:N]) from the common
  max-B by max-N uniform stream. No new stochastic process needs another seed.
- Fit shared quadratic coefficients at all real actions. The quadratic basis
  and standard-error formula define all off-grid values and dependence across
  actions. No interpolation, finite evaluation domain, optimizer queries or
  optimizer actions enter this coverage-only experiment.
- Reuse the existing observed-pairs refitter and analytic supremum certificates
  at tolerance 1e-8. Evaluate whole-real-line containment with the existing exact
  rational polynomial test. Rank-deficient resamples fail explicitly.
- Use the manifest runner and existing ORCD CPU launcher. Ten outer datasets
  share an array task, with a maximum of eight tasks running concurrently.
  Each dataset is independently checkpointed and validated for resume.
- Save original action/noise arrays, fitted coefficients, residual scales,
  bootstrap coefficients/supremum brackets, exact containment polynomials,
  source hashes and row-index hashes. Regenerate indices from the saved seeds
  and common uniform-stream dimensions, avoiding large duplicate index files.
- Validate direct original-row refits, unchanged source streams, pairing,
  polynomial containment, task partitioning, stale artifacts and collection.
  Run a small end-to-end manifest before submitting the full sweep.
- Report empirical coverage and pointwise 95% Wilson intervals in CSV/PDF.
  Each plotted line only connects evaluated settings; no smoothing is applied.
  More outer datasets reduce measurement noise. Bootstrap consistency does not
  promise monotonic coverage in N or B.

Validation completed: the 48 targeted joint-sweep, controlled-sweep, launcher
and CLI tests pass (the eight joint tests were rerun after correcting the seed
source path in provenance). A separate two-dataset preflight exercised N up to
5000 and completed through CSV/PDF collection. The runtime uses the shared
`simulation_env` and CPU partition; symbolic certificates dominate this workload.
