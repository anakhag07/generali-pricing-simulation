# Fixed-design wild-bootstrap envelopes

This study compares the analytical width from thesis §1.2 with the raw-residual
wild-bootstrap width from §1.3. Both use the same contextual linear truth,
frozen predictor, and OLS fit to $R=Y-\widehat f$. The target is the conditional
mean residual $g=f-\widehat f$. The primary result is joint
auxiliary-band coverage; transferred-envelope coverage is a separate secondary
result. Mathematical definitions are in
[MATH.md §7.9](../MATH.md#79-fixed-design-wild-bootstrap-with-a-bounded-gp-variance-profile).

## Run the supplied study

From the checkout root:

```bash
conda activate simulation_env
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-wild-bootstrap.json --launch local
```

The manifest uses `kind: "wild_bootstrap_envelopes"`. Outputs go to
`results/uncertainty-envelopes-wild-bootstrap/` under the shared external results
root, normally `~/projects/generali-pricing/results`. Change
`GENERALI_RESULTS_ROOT` to select another root. Copy the manifest and choose a
new experiment name before changing its model, seeds, or sweep settings.

The supplied sweep has:

| Setting | Values |
|---|---|
| Calibration size n | 30, 60, 120, 300, 600, 1,200 |
| Bootstrap count B | 99, 499, 1,999 |
| Miscoverage level delta | 0.10, 0.05, 0.01 |
| Independent outer repetitions | 2,000 |
| Baseline for one-axis displays | n=300, B=1,999, delta=0.05 |
| Context dimension and coefficient count | d=3, p=5 |
| Diagnostic action window | [-2,2] |

Every n uses its calibration contexts as the coverage target, so increasing n
increases both the sample size and the number of contexts tested. Coverage
always includes every real action at each of those contexts. The finite
diagnostic window is used only for plots and mean-width summaries.

## Increasing n and B together

Run the paired heteroskedastic residual comparison with:

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-residual-paired-nb.json --launch slurm --array --array-max-parallel 8
```

`"sweep_mode": "paired"` zips `n_values` and `B_values`; omitted mode retains
the existing Cartesian grid. The supplied pairs are (30,99), (60,199),
(120,499), (300,999), (600,1999), and (1200,3999), with 2,000 repetitions and
confidence levels 90%, 95%, and 99%. All other contextual-model, variance-profile,
and seed settings below are unchanged. Each pair uses only its declared B
bootstrap draws. Saved `active_conditions` identifies computed array cells;
uncomputed critical values, diagnostics, and bootstrap-statistic padding are
NaN. Summary rows contain only the paired conditions.

Both methods fit residual OLS. “OLS variance bound” labels the analytical
known-variance-cap construction, which is valid with this heteroskedastic noise;
it is not a homoskedastic estimated-standard-error interval. The comparison
estimates the combined effect of increasing n and B, not their separate effects.
Coverage is checked jointly over every calibration context and every real
action. The target set therefore also grows with n. More bootstrap draws reduce
quantile simulation error; neither empirical nor true coverage must increase
monotonically along the schedule.

Six PDFs show residual coverage, transferred-envelope coverage, mean residual
half-width, residual-band slices, function-envelope slices, and the fixed noise
profile. The slice panels use the first response repetition at all pairs and
three identical X values: central, median-norm, and outer contexts selected
from the smallest-n prefix before examining responses. These values and their
original row indices are saved in `paired_examples.npz`. Shared vertical axes
within each context allow widths to be compared across rows. The plotting
window remains [-2,2]; it does not define the confidence domain.

### Context-averaged bands

Replay the completed paired sweep without new fits or bootstrap draws:

```bash
PYTHONPATH=src python scratch/plot_context_averaged_envelopes.py /home/anakhag/projects/generali-pricing/results/residual-heteroskedastic-paired-nb/summary.json
```

`plots/07_context_averaged_bands_by_pair.pdf` shows both methods at each pair,
with residual bands on the left and transferred function envelopes on the right.
Each curve averages over all n calibration contexts, using response repetition 0
and the baseline confidence level. Transfer each context's band before averaging
its endpoints; clipping an already averaged residual band gives different results.
These are empirical-context means, not bands at the mean X or averages across
response repetitions. The original simultaneous-coverage event implies containment
of the corresponding context mean. A CSV saves plotted coordinates, and a JSON
sidecar records hashes of the source, saved inputs, and outputs. Existing sweep
reports and their summary remain unchanged.

## Frozen design and variance profile

The contextual feature vector is
$$\phi(x,a)=(1,x_1,x_2,x_3,a)^{\mathsf T}.$$
The truth and predictor have fixed coefficients
$$\beta_0=(0,1,-0.5,0.25,2)^{\mathsf T},\qquad
\widehat\beta=(0.2,0.8,-0.4,0.5,1.7)^{\mathsf T},$$
so $f(x,a)=\phi(x,a)^{\mathsf T}\beta_0$ and
$\widehat f(x,a)=\phi(x,a)^{\mathsf T}\widehat\beta$.
The predictor is not refitted during the study.

Draw the largest standard-normal context matrix and logged-action vector once.
Each n uses their first n rows. A separate random seed draws one GP profile,
which is frozen for every condition and outer repetition. The profile controls
the observation standard deviation; it is not an additive correlated error
draw across observations.

The exact finite-Fourier process is
$$G(t)=\frac{1}{\sqrt m}\sum_{k=1}^{m}
\left[A_k\cos(\omega_k t)+B_k\sin(\omega_k t)\right],\qquad
A_k,B_k\stackrel{\mathrm{iid}}{\sim}N(0,1).$$
The supplied `FiniteFourierGPSpec` has `rank=16` (m=16 frequency pairs) and
lengthscale $\ell=0.7$. Its deterministic frequencies are
$$u_k=\frac{k-1/2}{m},\qquad
\omega_k=\frac{\Phi^{-1}((1+u_k)/2)}{\ell},\qquad k=1,\ldots,m.$$
Before conditioning on the saved profile, its covariance is exactly
$$\operatorname{Cov}[G(t),G(s)]=\frac1m\sum_{k=1}^{m}
\cos\!\left(\omega_k(t-s)\right).$$
This is the specified finite-rank GP, not an interpolated or exact
squared-exponential GP. Every real query uses the same Fourier formula;
connected plotted samples are visualization only.

With $c=(0.7,-0.4,0.2)^{\mathsf T}$, the standard-deviation profile is
$$\sigma(x,a)=0.15+0.85\operatorname{expit}
\!\left(2G(a+c^{\mathsf T}x)\right).$$
Thus the variance is $\sigma(x,a)^2$ and the known global standard-deviation
cap is $\sigma_{\max}=1$. This cap follows from the expit construction, not
from a sampled-grid maximum.

For each outer repetition, generate
$$Y_i=f(x_i,a_i)+\sigma(x_i,a_i)z_i,\qquad
z_i\stackrel{\mathrm{iid}}{\sim}N(0,1).$$
Conditional on the entire saved design and GP profile, the response errors are
independent, have mean zero, and have variances at most one. All reported
coverage frequencies condition on that same environment. The study does not
average coverage over newly drawn GP profiles or designs.

## Two widths from the same auxiliary fit

For design matrix P with rows $\phi(x_i,a_i)^{\mathsf T}$, let
$Q=P^{\mathsf T}P$, $V=Q^{-1}$, and
$\ell_P(x,a)=\sqrt{\phi(x,a)^{\mathsf T}V\phi(x,a)}$.
Fit OLS to $R=Y-\widehat f$ to obtain $\widetilde\beta$, residual predictions
$\widetilde g=\phi^{\mathsf T}\widetilde\beta$, and fitted noise errors
$\widetilde\epsilon=R-P\widetilde\beta$.

The analytical width is
$$w_{\delta}^{\mathrm{analytical}}(x,a)
=\sigma_{\max}\sqrt{p/\delta}\,\ell_P(x,a).$$
Its conditional finite-sample simultaneous coverage lower bound is $1-\delta$
under the stated zero-mean, independent-error, bounded-variance assumptions.

For wild-bootstrap draw b, use independent Rademacher multipliers
$\xi_{bi}\in\{-1,1\}$ and compute
$$e_b^*=\beta_b^*-\widetilde\beta
=Q^{-1}P^{\mathsf T}(\widetilde\epsilon\odot\xi_b),\qquad
T_b^*=\sup_{i,\,a\in\mathbb R}
\frac{|\phi(x_i,a)^{\mathsf T}e_b^*|}{\ell_P(x_i,a)}.$$
The raw residuals receive no HC2/HC3 leverage adjustment. The denominator is
geometric leverage, not an estimated noise standard error. Use the empirical
order statistic of rank $\lceil B(1-\delta)\rceil$ among the first B statistics
without additional Monte Carlo inflation, and define
$$w_{\delta}^{\mathrm{wild}}(x,a)
=\widehat q_{1-\delta}\,\ell_P(x,a).$$

For either method, use separate nonnegative radii:
$$r_\delta^{\mathrm{lower}}=[w_\delta-\widetilde g]_+,
\qquad r_\delta^{\mathrm{upper}}=[w_\delta+\widetilde g]_+.$$
The resulting interval contains both the shifted residual band and the frozen
predictor. Residual-band containment implies transferred-envelope containment.

## Joint coverage and continuum checks

Each outer repetition records the two events separately:
$$C_{\mathrm{aux}}=\mathbf1\{\forall i,\ \forall a\in\mathbb R:
|\widetilde g(x_i,a)-g(x_i,a)|\le w_\delta(x_i,a)\},$$
$$C_{\mathrm{transfer}}=\mathbf1\{\forall i,\ \forall a\in\mathbb R:
-r_\delta^{\mathrm{lower}}(x_i,a)\le g(x_i,a)
\le r_\delta^{\mathrm{upper}}(x_i,a)\}.$$
The empirical coverage is the mean of these joint indicators across independent
outer repetitions. Pointwise 95% Wilson intervals quantify that simulation
uncertainty; they are not simultaneous confidence intervals for the entire sweep.
Contexts within an outer repetition are not separate binomial trials, and
the study does not replace joint coverage with context-averaged coverage.

The all-real auxiliary statistic is evaluated from coefficients. For
$H_i=[(1,x_i^{\mathsf T},0)^{\mathsf T},e_a]$,
$M_i=H_i^{\mathsf T}VH_i$, and $h_i=H_i^{\mathsf T}e$,
$$T(e)^2=\max_i h_i^{\mathsf T}M_i^{-1}h_i.$$
This includes action tails. Transferred containment is checked separately on
the all-real domain. Auxiliary statistics and quantiles use floating-point
arithmetic; their comparisons are numerical evaluations, not exact certificates.
The transferred check splits signed residual inequalities into sign regions and
checks the resulting quadratics over their full intervals in extended precision.
Near-boundary cases use the existing exact-rational polynomial checker on that
context. This fallback certifies the formula defined by stored floating-point
coefficients and covariance; it does not enclose OLS fitting or evaluation
roundoff. Diagnostic samples never certify either event.
There are no downstream optimizer queries, action-boundary constraints,
reported pricing optima, or regret calculations in this study.

The analytical lower bound and bootstrap empirical coverage have different
interpretations. Bootstrap coverage is not given a finite-sample guarantee.
Increasing B reduces quantile Monte Carlo error; it does not repair a biased
finite-sample bootstrap approximation. Increasing n at fixed p can improve
that approximation under suitable regularity, including vanishing maximum
leverage. Neither coverage nor width is required to improve monotonically
over the saved sweep settings.

## Quadratic moved-point residual study

`manifests/uncertainty-envelopes-quadratic-residual.json` supplies the unchanged
truncated-normal design; `uncertainty-envelopes-quadratic-residual-outlier.json`
moves row zero to 20. Copy the latter with a new result name to use
3, 10, 40, 100, or 1000. Both use $f(a)=5a-5a^2$, the saved frozen predictor
(coefficients and artifact provenance in the manifests), $\sigma=0.5$, $n=300$,
$B=1999$, and $\delta=0.05$. Full coverage sweeps are not part of implementation
validation.

With `design.action_degree: 2` and zero contexts, features are $(1,a,a^2)$.
The exact polynomial defines every off-grid query on $\mathbb R$; plotted
connections only visualize it. All actions share the same fitted coefficients,
with covariance $\sigma^2\phi(a)^\top V\phi(b)$ under constant noise. Existing
polynomial certificates bracket the all-real bootstrap supremum; upper endpoints
are used for the empirical quantile. With tolerance $\eta=10^{-8}$, brackets
satisfy $s_+-s_-\le2\eta\max(1,s_+)$. Coverage checks include both tails and
certify stored floating-point polynomials interpreted as rationals. See
[MATH §7.12](../MATH.md#712-quadratic-residual-bands).
There are no optimizer queries or reported optimizer actions.

Both manifests reuse the existing action, observation, and multiplier streams;
only the preselected action changes. Response means and predictor values are
recomputed there. Raw residuals can collapse at high leverage, so the moved-point
study tests a possible failure of bootstrap coverage; it is not evidence of
uniform validity over arbitrary outliers. The analytical comparison retains its
conditional variance-bound guarantee.

New summaries explicitly record `band_target: residual`. Historical artifacts
without this field retain their original function-target meaning, and their
source contracts prevent resuming them with this implementation. The old
manifest spelling `quantile: higher` is accepted as an input alias but normalized
to `order_statistic`; the original NumPy quantile convention can select a
different rank for some B and delta. Use a new result name for new runs.

## Seeds, pairing, and reporting

| Seed stream | Supplied root | Role |
|---|---:|---|
| contexts | 1101 | Largest fixed context sample |
| actions | 1102 | Largest fixed logged-action sample |
| observations | 1103 | Independent outer standard-normal response errors |
| method | 1104 | Independent outer Rademacher multiplier draws |
| profile | 1105 | One frozen GP coefficient draw |

The n conditions share row prefixes of the design and standardized observation
draws. The B conditions share multiplier prefixes. Delta values and both
methods share the same observed data and auxiliary fit; delta values reuse
the same bootstrap statistics. Observation and multiplier streams are
independent across outer repetitions. This pairing supports comparisons while
leaving each condition's outer repetitions independent.

Plots of mean widths and envelope slices use the finite diagnostic window
`[-2,2]` and equal context weights. Width summaries average across all outer
repetitions, contexts, and the 101 equally weighted diagnostic actions. A
displayed mean envelope is not a proof of joint containment.
Saved first-repetition plots retain that specified
realization without selecting it for favorable coverage. All new figures are
vector PDFs, and saved artifacts retain the representation and seed information
needed to reproduce the experiment.

## Saved outputs and replay

Each result directory contains:

- `environment.npz` and `environment.json`: fixed contexts and logged actions,
  GP coefficients, noise scales, diagnostic actions, and source/manifest identity.
- `repetitions/rep-00000.npz` and corresponding JSON records: saved observation
  innovations, fits for each n, bootstrap statistics through the largest B,
  critical values, and joint auxiliary/transferred outcomes for every condition.
  The JSON records include derived repetition seeds, multiplier shape and hash,
  and the saved array artifact hash. Subsequent repetitions use sequential names.
- `coverage_width_summary.csv` and `summary.json`: coverage counts and Wilson
  intervals, mean widths and their analytical ratios, mean critical values,
  maximum design leverage, separate lower/upper radii and coverage margins,
  moved-point radii when applicable, and artifact provenance.
- `example.npz`: the first repetition's baseline truth, predictor, auxiliary
  corrected fit, residual truth/fit, asymmetric bounds, widths, and noise-profile
  arrays used for the slice plots. Saved `beta` coefficients now fit the residual.
- `EXPERIMENT.md`: this representation guide followed by the actual saved
  manifest.
- `plots/`: the seven PDFs listed below.

| PDF in `plots/` | Display |
|---|---|
| `01_auxiliary_coverage_vs_n.pdf` | Joint auxiliary coverage against n, at baseline B |
| `02_auxiliary_coverage_vs_B.pdf` | Joint auxiliary coverage against B, at baseline n |
| `03_mean_width_vs_n.pdf` | Mean auxiliary half-width against n, at baseline B |
| `04_transferred_coverage_vs_n.pdf` | Joint transferred-envelope coverage against n, at baseline B |
| `05_analytical_context_slices.pdf` | Analytical envelopes for central, median-norm, and outer contexts |
| `06_wild_bootstrap_context_slices.pdf` | Wild-bootstrap envelopes for the same contexts |
| `07_noise_profile.pdf` | Standard-deviation profile and known cap for those contexts |

The first four PDFs give separate panels for the configured delta values.
Slices use the baseline n, B, and delta and select the smallest,
lower-median-ranked, and largest Euclidean context norms with stable ties.
Centrality is a way to choose contrasting examples; it does not guarantee an
ordering of coverage difficulty. The analytical method is independent of B,
so its values repeat across that axis.

Rerunning the same manifest resumes verified saved repetitions. To collect
complete saved results and regenerate plots without new response/bootstrap draws:

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-wild-bootstrap.json --launch local --collect
```

Collection requires every requested repetition and verifies artifact hashes
and the source contract before aggregation. It does not silently discard failed
or missing datasets. Changes to the source contract or manifest require a new
experiment name, or an explicit `--force` rerun that replaces the named results.
Report rendering itself uses saved summaries and the fixed example.

The implementation lives in `src/experiments/envelopes/`: `wild.py` owns the
mathematics, `wild_study.py` owns manifest execution and saved studies, and
`wild_reporting.py` owns result plots. Tests live in
`tests/experiments/test_envelope_wild.py`,
`tests/experiments/test_envelope_wild_study.py`, and
`tests/reporting/test_wild_envelope_reporting.py`.
