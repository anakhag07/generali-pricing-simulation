# Fixed offline pairs bootstrap and optimization regret

This study is retired. Manifests and saved results are indexed in the
[experiment archive](../scratch/retired_experiments/README.md); consult its
recovery instructions before replay. Paths inside saved provenance remain
unchanged.

Run the manifest through the shared launcher:

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py scratch/retired_experiments/bootstrap_ols/manifests/offline_bootstrap.json --launch local
```

The experiment studies one saved offline dataset and one saved quadratic
predictor. It varies sample size N, bootstrap count B, and observation-noise
scale sigma separately around the manifest baseline. There is no Cartesian
product of those axes and no outer-dataset coverage simulation. All mathematical
definitions, including the finite-B order statistic, are in MATH.md §7.7.

The committed manifest uses baseline `N=100, B=2000, sigma=1` with these axes:

| Axis | Values | Other settings |
|---|---|---|
| N | 25, 100, 500 | Baseline B and sigma |
| B | 200, 500, 2000 | Baseline N and sigma |
| sigma | 0.25, 1, 4 | Baseline N and B |

Both `homoskedastic` and `heteroskedastic_gp` profiles run these separate axes.
The GP uses 32 Fourier frequencies, length scale 0.2, and amplitude parameter
eta=0.75. The 95% target bootstrap mass uses pointwise 99% Monte Carlo confidence.
The input observation archive is saved dataset 000 of the controlled sweep;
the saved predictor is the N=100, sigma=1 fit from dataset 001. The manifest pins
the exact archive paths, array keys, and SHA256 checksums. To use different
artifacts, copy the manifest and update those fields and the output name.

## Data and seed ownership

The input archive and predictor are identified by their saved provenance and
checksums. The predictor is held fixed across every axis and noise profile;
its plug-in action and regret consequently provide one constant baseline.
Changing N takes nested prefixes of the saved observations. Changing B takes
prefixes of the same accepted bootstrap row indices at that N.

A sigma sweep changes the observed responses, so the preparation stage saves
explicit paired synthetic response variants before fitting and calibration.
These variants reuse the archive's actions and standardized Gaussian errors.
There are no fresh population observations during calibration. Each bootstrap
replicate resamples N complete observed `(action, response)` rows with
replacement, refits quadratic OLS, and adds no new observation noise.

The bootstrap has a dedicated row-selection seed, derived separately for each
N. Profile and sigma comparisons reuse the same row indices. A separate
rejection seed supplies replacement draws when the selected design is rank
deficient. Accepted indices and retry information are retained; exhausting the
retry limit fails explicitly. This defines the bootstrap law conditional on
full column rank and must be included in any theorem for this construction.
It does not alter archived experiments' rank-deficiency behavior.

A dedicated GP seed generates the saved heteroskedastic variance profile.
Full-batch first-order optimization is deterministic with explicitly configured
starts and consumes no stochastic optimizer stream. Predictor and original
observation randomness belong to their input artifacts, not the calibration
process. Source and artifact provenance make the experiment replayable.

## What the GP represents

The homoskedastic profile is `h(a)=1`. The heteroskedastic profile is
`h(a)=exp(eta G(a))/C`, using the existing finite-Fourier GP representation.
One saved draw defines G everywhere. C is the root mean square of `exp(eta G)`
over the complete saved action reservoir, including observations outside a
particular N prefix. This is empirical variance normalization, not an exact
population expectation. It holds the average observation variance on that
reservoir fixed while changing where the noise is concentrated.

Because C depends on the complete archived action reservoir, hypothetical pairs
are not iid unconditionally. Independent Gaussian errors here are conditional
on that complete design and the saved GP draw. A future pairs-bootstrap
coverage theorem must account for this shared normalization; an iid theorem
cannot simply be applied unchanged.

The observation model is `y_i=f(a_i)+sigma h(a_i) z_i`, with independent
standard-normal z_i conditional on the saved design and GP profile. The GP
therefore defines a noise amplitude. It does not add a shared correlated error
curve to all observations. Resampling individual observed pairs would not
preserve that latter dependence model.

Off-grid profile evaluation uses the same analytic Fourier formula, exponential
transformation, and fixed C. Quadratic predictions and the fitted standard-error
weight are likewise evaluated directly. Connected sweep points in the PDFs are
visual guides; they do not define an interpolated objective or noise process.

## Radius and analytical containment

The auxiliary quadratic fit is calibrated from the saved data. Each bootstrap
statistic is the supremum, over the entire real line, of its absolute deviation
from that fit divided by the original-data fitted-mean weight s(a). If c is
the calibrated critical value, the bootstrap half-width is `w(a)=c s(a)`.
The final envelope is centered on the fixed saved predictor with radius

```text
r(a) = |fixed_predictor(a) - auxiliary_fit(a)| + w(a).
```

Under heteroskedasticity s(a) remains the specified positive calibration
weight. It must not be described as the correct heteroskedastic standard error.
The known synthetic conditional covariance supplies an oracle diagnostic after
calibration; it does not determine the bootstrap radius.

Auxiliary-band and final-envelope containment are checked separately over R,
including both tails, using exact rational polynomial signs. Final-envelope
containment can hold even when the auxiliary band fails because of the
predictor discrepancy term. The required/calibrated auxiliary critical-value
ratio shows how much radius the realized dataset requires: a certified upper
ratio at most one is sufficient for auxiliary containment. Results apply to the
represented numerical coefficients and do not enclose OLS floating-point
roundoff relative to ideal arithmetic. Bootstrap supremum brackets use the
declared 1e-8 tolerance.

Containment for a known synthetic truth is a statement about this saved
dataset. It is not an estimated sampling coverage rate or a general probability
guarantee for future datasets. The default `mc_upper` finite-B order statistic
and optional `higher` empirical quantile are distinct calibration rules. The
conservative rule controls the bootstrap Monte Carlo error through an exact
binomial tail; too few replicates for the requested confidence fail explicitly.
Its confidence is pointwise for each condition, not simultaneous across the
sweep. It guarantees conditional bootstrap probability, not coverage of unknown
truth. A finite-N bootstrap approximation theorem remains a separate proof
obligation.

## Optimization and regret

The action domain is the entire real line. Repository optimizers query the
direct objective without artificial action bounds; configured starts are
initializations, not a restricted domain. Repository solvers supply every
reported truth-reference, plug-in, and LCB action. Exact all-real polynomial
sign checks certify the returned actions' global gaps at tolerance 1e-7. The
checks never select an action. The transferred LCB has an absolute-value kink;
failed or uncertified solves remain explicit, with their attempts recorded.

The reported regret interval accounts for the truth-reference gap. On
simultaneous envelope containment, the LCB regret certificate is

```text
R_LCB <= 2 r(a_ref) + epsilon_LCB + epsilon_ref.
```

The radius is measured at the certified truth-reference action. Its maximum
over the entire real line is generally infinite. The bound is marked applicable
only when containment and the needed optimizer certificates hold. A failed
containment check cannot be replaced by an optimizer success flag.

For positive sigma scaling with a fixed profile, correctly specified OLS and
paired bootstrap indices imply linear scaling of w and an unchanged auxiliary
containment event. The fixed predictor's discrepancy prevents the same claim
for the total radius or LCB regret. Increasing N need not eliminate that
discrepancy. The regret theorem does not guarantee LCB improves on plug-in for
every realized offline dataset.

## Outputs and plots

The result records preserve observations, bootstrap indices, fitted
coefficients, critical values, numerical certificates, optimizer attempts,
seeds, and provenance. They are written under
`results/offline-bootstrap-fixed-predictor/` by default:

- `offline_snapshot.npz` and `.json`: immutable inputs, saved response variants,
  GP coefficients/profile values, normalizer, and provenance.
- `calibration/<profile>/N<N>/`: accepted bootstrap row indices, rejected-draw
  counts, fits, and analytical supremum brackets.
- `bands/`: each distinct condition's observations, fit, bootstrap coefficients,
  covariance diagnostics, and auxiliary containment polynomial.
- `metrics.csv`, `summary.json`, and `EXPERIMENT.md`: axis rows, optimizer
  attempts and certificates, inference limits, and artifact hashes.

The three vector PDFs under `plots/` are:

1. `01_analytical_containment.pdf`: auxiliary required/calibrated radius ratios
   and final-envelope containment on R, with unresolved outcomes explicit.
2. `02_radius_components.pdf`: bootstrap half-width, fixed-predictor discrepancy,
   and total transferred radius at the truth reference.
3. `03_optimization_regret.pdf`: plug-in and LCB regret intervals, applicable
   conditional bounds, and explicit optimizer/certificate status.

Each noise profile has separate panels for N, B, and sigma. These plots show
single-dataset analytical diagnostics, not averages or coverage percentages.
Historical experiment artifacts are preserved.

Ordinary reruns validate all saved hashes and reuse completed results. Use
`--force` to rebuild this experiment's snapshots, calibration checkpoints and
outputs after an implementation change. Source-archive checksums remain
mandatory even with `--force`; neither input archive is modified. Regret
interval endpoints and radius-based certificate bounds use directed rounding
of rational calculations on the represented coefficients.
