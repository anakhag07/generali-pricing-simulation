# Pairs-bootstrap sweep implementation

The agreed experiment resamples the original (x, y) rows with replacement.
It uses the same original-data seed roots and Gaussian observation model as
the controlled sweep. There is no held-out dataset and no new bootstrap noise.
The noise and bootstrap-count panels use N=100; baseline sigma=1 and B=2000.
The N grid extends to 5000, using prefixes of the same original dataset.

## Representation and inference contract

- Domain: the whole real line, including both tails. All curves are direct
  quadratic-basis evaluations; plotted connecting lines are display only.
- Dependence across actions comes from the same three random fitted
  coefficients. The original conditional covariance is sigma^2 p(a)' V p(b).
- The original fitted-mean standard error remains the bootstrap denominator.
  Calibration uses observed pairs only. Known truth only generates observations
  and evaluates coverage/regret.
- Bootstrap consistency gives asymptotic simultaneous coverage under the
  correctly specified random-design model and regularity conditions. Finite
  N/B coverage is empirical, not exact. Increasing B reduces Monte Carlo error,
  without a monotonic regret or width guarantee.
- The existing independent bootstrap seed stream now generates uniform row
  selections. B uses prefixes; N uses floor(N*U[:, :N]) from shared uniforms;
  sigma uses identical row indices. Design/observation streams stay unchanged.
- Rank-deficient resamples fail explicitly; they are never silently discarded
  or redrawn. Minimum agreed N=20 makes such cases very unlikely.
- Repository first-order L-BFGS-B supplies every reported action, with no bounds.
  Existing exact rational polynomial tests certify global objective gaps at
  1e-7 and bootstrap supremum brackets at relative/absolute scale 1e-8.
- Width is measured at the true reference action; whole-line maximum width is
  infinite. The coverage-event regret bound includes optimizer gap tolerances.
- New pairs-specific output names preserve old parametric artifacts. Source
  hashes, saved row indices, observations and coefficients allow exact replay.

## Implementation and verification

1. Share observed-row OLS refitting between grid bands and continuous sweeps.
2. Cache standardized pairs errors across noise scales, using OLS equivariance;
   verify against direct refits of original response rows at every scale.
3. Use one existing launch-plan task per independent dataset for ORCD arrays;
   keep dataset seeds independent of scheduling order.
4. Check resampled fits against independent least-squares calculations,
   deterministic replay/B prefixes, unchanged original-data streams, noise
   scaling, whole-line certificates, and stale-cache rejection.
5. Profile refitting versus symbolic certification before selecting CPU/GPU.
   Run a reduced manifest before launching the full 100-dataset sweep.

The GPU would only accelerate a newly ported numerical refit; the existing
symbolic certificates and SciPy repository solver run on CPU. Profile the
actual workload before allocating a GPU.

## Validation and launch

The targeted bootstrap, analytical certificate, launcher and CLI tests passed.
The reduced two-dataset manifest at `results/bootstrap-ols-pairs-preflight/`
completed with N=20,100,5000 and B up to 50. Across these calibrations, refits
took 0.028s and certificates took 0.695s (96.1% of calibration time).
The full run uses the existing ORCD CPU partition: GPU refits would address
only a small fraction of this workload, while exact symbolic certification
and the repository optimizer remain on CPU. Each array task requests one CPU
and 4 GiB; at most eight datasets run concurrently. The collector validates
all 100 dataset records before generating the three final metric PDFs.

The full manifest contains 37 axis entries and 35 distinct parameter settings
(the shared baseline appears in all three panels), not a Cartesian product.
Historical parametric outputs are preserved under their original names.
