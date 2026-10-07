# Bootstrap calibration proofs

Lean verifies the finite-sample predictor-error certificate for the ideal
homoskedastic Gaussian quadratic pairs-bootstrap setup. The final theorem is
`QuadraticCalibrationDesign.finite_sample_predictor_certificate_accepted` in
[`ObservedCalibration.lean`](BootstrapCalibration/ObservedCalibration.lean).

For a fixed distinct-action design with `n ≥ 5`, an arbitrary frozen predictor,
actual full-rank pairs resampling, and measurable certified upper scores on
accepted draws, it constructs a measurable event of probability at least

`1 − δ − τ − η − ε_a / τ²`.

Here `p = 1 − δ`, `0 < τ < p ≤ 1`, `η ≥ 0`, and `0 < a < 1`. The sorted
critical value has zero-based index `j : Fin B`, chosen so that
`binomialTail p B (j.val + 1) ≤ η`.

On that event, simultaneously for every real action, the calibrated radius
bounds the frozen predictor's error and exceeds it by at most twice the
bootstrap half-width. The radius uses original-coordinate OLS, the residual
standard deviation computed from the observed responses, and the actual sorted
bootstrap critical value. `ε_a` is exactly the explicit design-dependent
formula in the LaTeX note. No bootstrap-validity or desired-coverage premise
remains at the final interface.

## Reproduce

The project pins Lean **4.19.0** in `lean-toolchain` and mathlib **v4.19.0**,
commit `c44e0c8ee63ca166450922a373c7409c5d26b00b`, in `lake-manifest.json`.
With Lean's `elan` toolchain manager available, run from this directory:

```bash
lake exe cache get
lake build
lake env lean BootstrapCalibration.lean
```

The last command prints the foundational axioms used by the exported theorems.
The intended result is only `propext`, `Classical.choice`, and `Quot.sound`;
there are no custom statistical axioms or proof placeholders.

For the initial verification on 2026-10-06, an isolated installation was used:

```bash
export ELAN_HOME=/tmp/generali-bootstrap-lean/elan
export PATH=/tmp/generali-bootstrap-lean/elan/bin:$PATH
lake build
lake env lean BootstrapCalibration.lean
```

That installation reported Lean 4.19.0, commit `6caaee842e94`, on
`x86_64-unknown-linux-gnu`. Ignored `.lake/packages/` symlinks reuse the matching
mathlib checkout and 6,641 downloaded compiled cache files under
`/tmp/generali-bootstrap-lean/`; a fresh checkout uses the ordinary commands
above. The temporary cache used
`MATHLIB_CACHE_DIR=/tmp/generali-bootstrap-lean/cache`. On this Red Hat host,
its downloader also needed
`CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt`.

## Proof map

- `Main.lean` assembles the stated closed-form bound for the original fixed
  quadratic design; `StudentizedCalibration.lean` instantiates the actual
  residual scale; `ObservedCalibration.lean` concludes the predictor-error
  certificate with an explicitly measurable confidence event.
- `AcceptedEndpoints.lean` proves that score certification and measurability
  are needed only on accepted draws; a zero extension handles rejected draws.
- `MultinomialMoments.lean` and `DesignMoments.lean` derive the count and
  leverage moments for arbitrary sample size from explicit iid uniform draws.
- `RankProbability.lean` and `QuadraticRank.lean` prove the exact singularity
  probability and positive acceptance mass for distinct quadratic actions.
- `AcceptedLaw.lean` constructs actual conditional and product laws.
  `RejectionSampling.lean` proves individual retry-failure probabilities and
  the conditional first-return law.
- `OLS.lean`, `Whitening.lean`, and `RegressionInterpretation.lean` connect the
  original OLS/pairs-refit formula, inverse-Gram standard errors, whitening,
  and the observable residual SD to the normalized coefficient statistics.
- `GaussianComparison.lean`, `GaussianLinearImage.lean`, and
  `SpectralComparison.lean` derive the Gaussian bounded-test comparison from
  density integrals, characteristic functions, and spectral identities,
  including singular covariance. The comparison is not an axiom or premise.
- `StableCovariance.lean`, `BootstrapError.lean`, and `OLSBootstrapError.lean`
  derive the inverse perturbation, covariance moments, and accepted-law error
  bound, including the distinct `1/r²` and `1/r` conditioning factors.
- `TwoReplicas.lean` and `GaussianPairsMSE.lean` derive the actual integrated
  CDF mean-square bound. `Coverage.lean` and `IntegratedCalibration.lean`
  account for dependence between the statistic and its critical value.
- `FiniteBootstrap.lean`, `OrderStatistic.lean`, and
  `OrderStatisticMeasurable.lean` prove finite-B calibration, including ties,
  measurability, conservative upper scores, and positive scale equivariance.
- `NormCDF.lean`, `SupremumStatistic.lean`, and `QuadraticStatistic.lean` treat
  the supremum over the whole real line and existence of its exact CDF
  threshold. `ResidualScale.lean` proves positive fitted scale almost surely.
- `CalibrationScale.lean` and `Envelope.lean` provide the scalar cancellation
  and deterministic transferred-radius bounds.

`FiniteSample.lean` also proves the sharper version whose error is an explicit
finite average of two-refit Gaussian covariance discrepancies;
`PairsCalibration.lean` specializes that version to the accepted pairs law.
The closed-form leverage bound is proved to upper-bound this finite average.

## Scope and limits

The result conditions on the observed actions and integrates observation noise
and bootstrap randomness. It applies to independent homoskedastic Gaussian
errors with positive noise SD and a correctly specified quadratic mean. The
frozen predictor itself can be arbitrary. Certified upper-score measurability
and domination on accepted draws remain algorithm-facing hypotheses. The
accepted-only final theorem uses the proved zero extension on rejected draws,
which does not affect the accepted resampling law.

This is a theorem for ideal exact arithmetic and exact algebraic rank. It does
not certify NumPy's tolerance-based rank decisions, floating-point OLS error,
the score-certification implementation, or the Python/SciPy rank selector.
Heteroskedastic or misspecified regression is not covered by this theorem.

The note's simplified `33*sqrt(L)` estimate, asymptotic-rate discussion, and
numerical diagnostics are outside the final audited Lean theorem. Individual
finite-retry laws are checked, but the batch abort-adjusted coverage corollary
is not assembled into that theorem. The current closed-form constants are
vacuous at the archived experiment sizes; formal correctness does not make
those numerical bounds informative. Originality remains unestablished.

See [`bootstrap_finite_sample_candidate.tex`](../docs/bootstrap_finite_sample_candidate.tex)
for the statement, derivation, diagnostics, and primary-source comparison.
`python check_multinomial_moments.py` is a separate exact-enumeration diagnostic
for N=3 through 30; the Lean moment proofs cover all N.
