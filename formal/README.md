# Bootstrap calibration proofs

This project checks selected mathematical statements for calibration of an
arbitrary frozen predictor. It does **not** yet verify finite-sample coverage of
the repository's complete pairs-bootstrap procedure.

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

## What is checked

- `Envelope.lean`: transfer from an auxiliary-model band to a frozen-predictor
  error radius, its pointwise excess-width bound, and its deterministic
  minimality under the stated interval uncertainty.
- `FiniteBootstrap.lean`: a finite bootstrap law with normalized nonnegative
  atom weights, its explicit iid product law, CDF validity with ties, the exact
  Bernoulli count recursion, and finite-`B` quantile calibration. The theorem
  `FiniteLaw.quantile_failure_le` proves
  `Pr[F_D(Q) < p] ≤ binomialTail p B k` when at least `k` sampled scores are at
  most `Q`. `FiniteLaw.quantile_mass_confidence` converts a chosen tail bound
  into a confidence statement. The recursion defining `binomialTail` is proved
  to equal the hit-count probability; it is not assumed. `FiniteLaw.toPMF_event`
  identifies the finite sums with mathlib probability-mass-function events.
- `Conditioning.lean`: elementary perturbation bounds for a conditioned law.
- `Coverage.lean`: event and moment reductions that allow the bootstrap critical
  value to depend on the same data as the estimation error.

The finite-`B` result applies to a fixed dataset and its finite resampling law,
including a law already conditioned on full design rank. It also allows a
critical value built from certified upper bounds on bootstrap scores, through
the explicit rank hypothesis. This module does not verify the Python/SciPy
implementation of the rank selector or the score-certification code.

## Remaining statistical obligation

A finite-`N` comparison between the true estimation-error distribution and the
data-dependent, rank-conditioned pairs-bootstrap distribution remains a separate
obligation. The CDF comparison or CDF mean-square estimate appearing in
`Coverage.lean` is an explicit premise, not a verified conclusion about that
bootstrap method. In particular, deriving a numerical finite-`N` error from the
Gaussian quadratic regression model, its design, and resampling mechanism is
not accomplished by compiling the event-reduction theorem.

The proposed derivation and its unverified formal obligations are in
[`bootstrap_finite_sample_candidate.tex`](../docs/bootstrap_finite_sample_candidate.tex).
`python check_multinomial_moments.py` independently checks the proposed count
moments by exact enumeration for N=3 through 30; this is not a proof for all N.

The checked finite-`B` statements therefore certify probability mass under the
conditional bootstrap law. They do not by themselves certify coverage of the
unknown true response function.
