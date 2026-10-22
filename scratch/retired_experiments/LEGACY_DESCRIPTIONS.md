# Retired study descriptions

Archived from the README at `b7faa26`. These describe historical configurations,
not current entry points. Original paths and commands below are retained as
historical context; consult [recovery instructions](README.md) and
[relocations.json](relocations.json) before use. Current repository optimizer
and PDF-output rules apply to any resumed work.

## Offline bootstrap, OLS and support-kernel studies

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/offline_bootstrap.json --launch local
```

This experiment freezes one saved quadratic predictor and an offline dataset,
then varies `N`, `B`, and `sigma` separately. Pairs bootstrap draws complete
observed rows with replacement; rank-deficient resamples are redrawn from the
same data. Homoskedastic noise and a saved GP-derived positive variance profile
share the same actions and standardized observation errors. Noise-scale variants
are prepared and saved before calibration.

Three PDFs report whole-real-line containment, the transferred radius's
components, and plug-in/LCB optimization regret with conditional regret bounds.
Repository optimizers supply all actions. There is no outer-dataset coverage
sweep: finite-`B` calibration certification is distinguished from the remaining
finite-`N` bootstrap approximation guarantee. See
[the offline experiment notes](../../docs/offline_bootstrap.md) and MATH.md §7.7.

### Controlled bootstrap envelope sweeps

```bash
python scripts/run_experiment_manifest.py manifests/bootstrap_ols_controlled_sweep.json --launch slurm --array
```

This pairs-bootstrap quadratic-OLS experiment varies observation-noise SD
`sigma`, training size `N`, and bootstrap count `B`, one at a time. Both the
confidence statement and unconstrained LCB optimization cover the entire real
line; normal training inputs and optimizer initialization in `[0,1]` impose
no action bounds. Each observed dataset has its own bootstrap band. The original
`(x_i,y_i)` rows are resampled with replacement, with no fresh bootstrap noise.
The baseline is `N=100, sigma=1, B=2000`; the three separate axes include
`N=20..5000`, `B=5..2000`, and `sigma=0.1..5` at the manifest's listed values.
There are 100 independent datasets, with paired prefixes across conditions.
Slurm uses one CPU task per dataset, at most eight concurrently, followed by a
collector. Analytical polynomial certification dominates this CPU workload.
See [the implementation notes](../../docs/bootstrap_pairs.md) for seeds and replay.

Three PDFs report simultaneous coverage, envelope half-width at the true
optimum, and LCB regret. All reported actions come from the repository
optimizer, with additional all-real polynomial gap certification. Unbounded or
uncertified cases have explicit counts rather than silently reported regret.
Outputs, saved fits/bootstrap perturbations, exact containment polynomials,
seeds, source hashes, and `EXPERIMENT.md` live under
`results/bootstrap-ols-pairs-controlled-sweep/`. Calibration uses no truth or grid.
See MATH.md §7.3 for pairing, width semantics, and the coverage-event regret
bound. The fixed-denominator bootstrap's sampling coverage is approximate.

For the dedicated coverage-only sweep with N and B increasing together and
2000 independent datasets, run:

```bash
python scripts/run_experiment_manifest.py manifests/bootstrap_ols_joint_coverage.json --launch slurm --array
```

The nine settings use N=B from 20 through 5000 at sigma=1. The band rule stays
fixed; the CSV and PDF report empirical coverage with pointwise Wilson
intervals. There is no ranking, smoothing or monotonicity constraint. Ten
checkpointed datasets share each CPU array task, with eight tasks concurrent.
Outputs live under `results/bootstrap-ols-pairs-joint-coverage/`; details are in
[the joint sweep notes](../../docs/bootstrap_joint_coverage.md).

To complete the 9-by-9 Cartesian N/B grid while reusing the diagonal and earlier
controlled sweep's saved bootstrap prefixes, run:

```bash
python scripts/run_experiment_manifest.py manifests/bootstrap_ols_cartesian_coverage.json --launch slurm --array
```

This retains the original 2000 datasets and fits only missing bootstrap suffixes.
Results and coverage/uncertainty heatmaps go to
`results/bootstrap-ols-pairs-cartesian-coverage/`. CPU tasks checkpoint each N;
compatible existing bootstrap draws and diagonal coverage are verified before
reuse. See [the extension notes](../../docs/bootstrap_cartesian_coverage.md).

To reconstruct the joint N/B coverage grid from the saved bootstrap prefixes
and plot its empirical Pareto frontier, run:

```bash
PYTHONPATH=src python scratch/plot_bootstrap_coverage_frontier.py
```

This writes a vector PDF, the full coverage CSV, and replay provenance under
`results/bootstrap-ols-pairs-controlled-sweep/reports/coverage_frontier/`.
The frontier minimizes N, B, and absolute coverage error relative to 95%; its
highlighted points are exploratory estimates based on 100 datasets (MATH.md §7.4).

For the paired follow-up that fixes `sigma=1` and `B=2000` while sweeping a
denser range of training sizes from `N=20` through `N=5000`, run:

```bash
python scripts/run_experiment_manifest.py manifests/bootstrap_ols_dense_n_sweep.json --launch local
```

The follow-up reuses the same all-real calibration, repository optimization,
certification, metrics, and 100-dataset design. Within each dataset, every
smaller training sample is a prefix of the `N=5000` sample, so changes across
`N` are paired. Its separate outputs live under
`results/bootstrap-ols-pairs-dense-n-sweep/` and its PDFs contain only the `N` panel.

### Gaussian-support envelope N sweep

```bash
python scripts/run_experiment_manifest.py manifests/coverage_kernel_dense_n_sweep.json --launch local
```

This separate experiment uses the same 100 paired quadratic-OLS datasets and
14 training sizes, with observation-noise SD fixed at 1. Instead of bootstrap
refits, it uses the unnormalized Gaussian support sum with fixed bandwidth
`b=0.1` and two-sided radius `q_0.975 * sigma_hat / sqrt(C_N(a))`. The nominal
Gaussian quantile does not by itself guarantee simultaneous 95% coverage;
whole-real-line coverage is measured empirically. Repository-optimizer actions
are gap-checked, and uncertain checks remain explicit rather than silently
entering regret averages. Three vector PDFs show coverage, half-width, and
regret versus `N`. Data, seeds, fits, optimizer attempts, and source hashes
are saved under `results/coverage-kernel-dense-n-sweep/`. See MATH.md §7.4.

To replay those same 100 datasets and all 14 training sizes while varying one
envelope parameter at a time, run:

```bash
python scripts/run_experiment_manifest.py manifests/coverage_kernel_q_sweep.json --launch local
python scripts/run_experiment_manifest.py manifests/coverage_kernel_b_sweep.json --launch local
```

The first sweep varies the coefficient `q` in `lambda=q*sigma_hat` at fixed
bandwidth `b=0.1`. The second varies `b` at fixed `q=1.96`. Both include the
original setting as an exact saved-output baseline, use the repository
optimizer for every other action, and write separate result trees with CSVs
and three vector PDFs. The agreed support sum is unnormalized: increasing
`b` raises support and narrows the envelope mechanically, so the bandwidth
sweep changes both smoothness and scale. These sweeps are descriptive; a
parameter chosen to reach target coverage requires evaluation on fresh data.


### Zeroth-Order Proof Validation

The `zeroth_order_proof_base` preset implements the globally strongly convex
objective $$f(x)=x^2+\frac12(\sin x-x)$$ with known $$x^*=0$$. Two committed
manifests check the predicted finite-difference and Stein-difference dependence
on perturbation radius, Stein sample count, and deterministic functional bias:

```bash
python scripts/run_experiment_manifest.py manifests/zeroth_order_baseline.json
python scripts/run_experiment_manifest.py manifests/zeroth_order_functional_bias.json
python scripts/analyze_zeroth_order_proof_validation.py \
  --baseline-manifest manifests/zeroth_order_baseline.json \
  --bias-manifest manifests/zeroth_order_functional_bias.json
```

The analyzer records $$x_0$$, the clean and biased optima, each estimator's
population fixed point, and $$x_K$$. It writes sweep-level displacement,
bias/variance/MSE, theorem-bound, and fitted-scaling tables and plots under
`results/zeroth-order-proof-validation-analysis/`.

### Finite-Policy Lower Confidence Bounds

The committed Proposition 11.2 validation evaluates the finite constant-policy
class $$\Pi=\{0,0.1,\ldots,1.0\}$$ under
$$V^\pi=\pi$$ and
$$\widehat V^\pi=\pi+\pi Z^\pi$$. It exhaustively selects the maximum lower
confidence bound, so there is no gradient method and the optimization error is
$$\varepsilon=0$$:

```bash
python scripts/run_experiment_manifest.py \
  manifests/finite_policy_lcb_validation.json
```

Each of the 25 noise seeds draws one independent Gaussian vector over policies
and reuses it for all five confidence levels. The collector writes seed-level
policy/selection tables, analytic and empirical coverage, oracle-inequality
checks, and paired seed/aggregate plots under
`results/finite-policy-lcb-validation/`.

### Continuous-Policy Lower Confidence Bounds

The continuous companion uses the same manifest launcher and seed-task output
contract over $$\Pi=[0,1]$$. Each run seed draws one scalar $$Z_s\sim N(0,1)$$
and reuses it across every policy, confidence level, estimator, and start:

$$
V(\pi)=5\pi-5\pi^2,
\qquad
\widehat V_s(\pi)=V(\pi)+\pi Z_s,
\qquad
-\underline V_{s,\delta}(\pi)=5\pi^2+(q_\delta-5-Z_s)\pi.
$$

The negative LCB is convex, with analytic constrained minimizer
$$\operatorname{clip}_{[0,1]}((5+Z_s-q_\delta)/10)$$. In particular, all
Gaussian draws within two standard deviations have interior minima for the
configured confidence levels.
Projected first-order, finite-difference, and Stein-difference runs from three
paired starts measure convergence and LCB optimization error against that exact
minimum:

```bash
python scripts/run_experiment_manifest.py \
  manifests/continuous_policy_lcb_validation.json
```

The 25 problem-noise seeds use a separate fixed Stein perturbation stream so
cross-seed spread isolates the shared Gaussian draws. Outputs under
`results/continuous-policy-lcb-validation/` include per-start and best-of-start
tables, replayable seed JSONs, trajectory CSVs, exact/empirical coverage and
oracle diagnostics, a median/IQR seed-band plot, and a separately labeled
mean/bootstrap-95% plot. Two direct objective diagnostics show the negative LCB
across problem-noise seeds at fixed $$\delta=0.05$$ and across confidence
levels with the Gaussian draw fixed at $$Z=0$$.

The continuum-wide quantile is $$q_\delta=\Phi^{-1}(1-\delta/2)$$ rather than
the finite-policy Bonferroni quantile. This is not a consequence of continuity:
it follows because the policy-indexed error is rank one,
$$\widehat V_s(\pi)-V(\pi)=\pi Z_s$$, so simultaneous coverage over every
$$\pi$$ is exactly the single event $$|Z_s|\le q_\delta$$.

### Bootstrap OLS band constructed from observations

The original finite-grid experiment below is preserved for provenance. To redo
its saved fits analytically over the **entire real line**, run:

```bash
python scripts/run_experiment_manifest.py manifests/bootstrap_ols_continuous_replay.json --launch local
```

This replay uses exactly the saved observations, OLS fits, and all bootstrap
coefficients. Every bootstrap supremum is bracketed by exact rational quartic
sign checks, including both tails, and the quantile is recalibrated from those
all-real statistics. Every observed dataset receives an analytical containment
Boolean from `c_hat²*sigma_hat²*p(a)'V*p(a) - e(a)² >= 0` for all real `a`.
The sign certificates apply to the saved numerical polynomials; repeated-data
coverage remains an empirical evaluation of an approximate bootstrap procedure.
No plot samples enter calibration or coverage, and no coefficient ellipsoid is
used. See MATH.md §7.2.

Results go to `results/bootstrap-ols-pairs-continuous-replay/`, preserving the old
experiment. Four mathematically labeled PDFs show the true/fitted/bootstrap
curves and signed error, bootstrap calibration and whole-line normalized error,
saved error realizations at `n=100` across noise SD, and empirical coverage/band
width for all original conditions. Finite display windows are explicitly
labeled; a compactified panel uses `theta=arctan(a)` to display both infinite
tails. The old across-dataset coefficient figure is omitted. A CSV records the
original coefficients, first bootstrap fit, and central 95% bootstrap ranges.

```bash
python scripts/run_experiment_manifest.py manifests/bootstrap_ols_grid_band.json --launch local
```

This two-stage experiment fits quadratic OLS to independent `x ~ N(0,1)` and
`y = 5*x - 5*x**2 + Gaussian noise`. Stage 1 uses `n=100`, noise SD `1`, and
499 original-row pairs-bootstrap refits. It calibrates a direct
maximum standardized prediction error on 1,001 points in `[0,1]`, using the
original fit's prediction standard error for every bootstrap denominator.
The 95th-percentile critical value uses `np.quantile(method="higher")`.
There is no coefficient ellipsoid. Synthetic truth enters generation and
evaluation only. One dataset gives one Boolean grid-coverage result.

Stage 2 independently repeats fitting and calibration for 100 datasets per
condition, with sample sizes `{25,100,500}` and observation noise SDs
`{0.5,1,2}`. It reports simultaneous grid-coverage fractions and Wilson intervals,
OLS coefficient spread, residual variance estimates, and band widths.
This is empirical validation of approximate bootstrap coverage on the grid,
not a finite-sample proof or a continuous-domain guarantee. The polynomial is
defined for all real inputs; only the specified grid is covered by the claim.

Outputs under `results/bootstrap-ols-pairs-grid-band/` include `EXPERIMENT.md`, a
`summary.json`, raw dataset and coverage CSVs, replayable per-condition NPZs
with all bootstrap coefficients/maxima, and five vector PDFs. Separate derived
seeds control design, response noise, and bootstrap, with independent streams
across both stages and every condition. The runner reuses matching completed
conditions; `--force` regenerates them. See MATH.md §7.1 for the exact formulas.

### Variable-Envelope Finite-Grid Lower Confidence Bounds

The variable-envelope characterization performs exact maximization on 101
points in $$[0,1]$$ for the concave objective
$$f(x)=5x-5x^2$$. Its uncertainty profile has minimum at a swept center $$m$$:

This experiment is finite: it draws one independent Gaussian value at each of
the 101 grid points and optimizes only those points. Lines joining values in
the diagnostic plots are visual guides, not interpolation and not a continuous
random-function definition.

$$
\sigma_m(x)=0.1+0.9\min\left(\frac{|x-m|}{0.5},1\right),
\qquad
\widehat f_{s,c,m}(x)=f(x)+c\sigma_m(x)Z_{s,x}.
$$

Run the committed 2,000-seed manifest with:

```bash
python scripts/run_experiment_manifest.py \
  manifests/variable_lcb_envelope_characterization.json
```

One independent Gaussian vector is drawn per run seed and reused for every
center, noise scale, Bonferroni/pointwise calibration, and selector. Thus the
cube is paired: changing $$c$$ or $$m$$ changes only the uncertainty geometry,
not the standardized noise realization. The simultaneous envelope uses
$$q=\Phi^{-1}(1-\delta/(2K))$$; the pointwise comparator uses
$$q=\Phi^{-1}(1-\delta/2)$$. Because $$c>0$$ and $$\sigma_m(x)>0$$ cancel from
the coverage inequality, the Bonferroni simultaneous event is the same
$$\max_x|Z_{s,x}|\le q$$ for every positive scale and center. At $$c=0$$,
coverage is deterministically perfect.

The outputs under `results/variable-lcb-envelope-characterization/` include
replayable seed JSONs, raw condition and selector CSVs, summaries for the noise,
calibration, envelope-shape, and center sweeps, and ten aggregate plots. The
center plots distinguish the true optimum $$x^*$$, uncertainty center $$m$$,
and deterministic penalized target
$$x^\dagger=\arg\max_x[f(x)-E(x)]$$. The constant uniform envelope is retained
as a diagnostic and must select exactly the same point as the nominal surrogate.
Here $$m$$ is specifically the point at which $$\sigma_m(x)$$ is minimized,
while $$c$$ multiplies both surrogate noise and the calibrated envelope. Every
reported regret is evaluated on the true objective:
$$R(\hat x)=f(x^*)-f(\hat x)=1.25-[5\hat x-5\hat x^2]
=5(\hat x-0.5)^2$$. The aggregate regret plots print this calculation and the
definitions of $$m$$ and $$c$$ directly below the axes.
The per-center Experiment 2 calibration-regret plot compares nominal,
pointwise-LCB, and simultaneous-LCB regret as $$c$$ changes, with 5th--95th
seed-percentile bands in every $$m$$ facet.
The matching per-center Experiment 3 plot combines the exactly identical
nominal and uniform-LCB curves, then contrasts them with the variable-LCB curve
to isolate the effect of valid envelope geometry at each $$m$$.
The additional `experiment_1_realized_landscapes.png` diagnostic fixes the
first run seed and one off-optimum uncertainty center, then shows $$f$$,
$$\widehat f$$, the two-sided envelope, its LCB edge, and both selected points
across all configured values of $$c$$.

### Continuous-GP Variable-Envelope Lower Confidence Bounds

The continuous analog defines every random path analytically rather than
interpolating finite samples. For fixed rank $$J=32$$ and lengthscale
$$\ell=0.2$$,

$$
G_s(x)=\frac1{\sqrt J}\sum_{j=1}^J
[A_{s,j}\cos(\omega_jx)+B_{s,j}\sin(\omega_jx)],
$$

where the frequencies are deterministic RBF-spectral quantiles and each seed
draws only the Gaussian coefficients. The surrogate is

$$
\widehat f_{s,c,m}(x)=5x-5x^2+c\sigma_m(x)G_s(x),
$$

with a $$C^2$$ clipped smoothstep uncertainty profile minimized at $$m$$. The
committed manifest sweeps
$$m\in\{0,0.25,0.5,0.75,1\}$$ and
$$c\in\{0,0.25,0.5,1,2\}$$:

```bash
python scripts/run_experiment_manifest.py \
  manifests/continuous_gp_variable_lcb.json
```

The single envelope multiplier is calculated before any run seed. Bonferroni
controls a 129-point covering net, and a chi-square coefficient-norm event
bounds all between-net movement, giving
$$q\approx3.908377$$ and
$$\Pr(\sup_{x\in[0,1]}|G_s(x)|\le q)\ge0.95$$. The 2,000 GP draws verify this
guarantee empirically; they do not calibrate it.

Every draw receives branch-and-bound-certified nominal and variable-LCB global
reference values. The first 200 of those same draws additionally run projected
exact-gradient, central finite-difference, and 64-perturbation antithetic
Stein-difference optimization from three starts. The ZO probes evaluate the
documented analytic extension outside `[0,1]`, while every iterate is projected
back into the feasible interval. Outputs distinguish confidence-band validity,
envelope tightness, globally achievable regret, and practical optimizer gap.

### Continuous-GP Regret Decomposition

The follow-on experiment separates the scale and spatial profile of the frozen
surrogate error from those of the lower-envelope correction:

$$
\widehat f(x)=f(x)+c_f\sigma_{m_f}(x)G_s(x),
\qquad
\underline f(x)=\widehat f(x)-c_Eq\sigma_{m_E}(x).
$$

Run its committed 200-path manifest with:

```bash
python scripts/run_experiment_manifest.py \
  manifests/continuous_gp_regret_decomposition.json
```

The one-at-a-time surrogate-scale, envelope-scale, and envelope-shape sweeps use
the certified global LCB maximizer, eliminating optimizer error. The optimizer
axis and combined grids retain checkpoints from projected central finite
difference and 64-perturbation antithetic Stein runs. All configurations within
a seed reuse the same Fourier path, starts, and fixed Stein perturbation stream.

The two frozen surrogate profiles $$m_f\in\{0.25,0.5\}$$ distinguish matching
the actual surrogate-error shape from putting the narrowest envelope point at
the true optimum. Every row records true regret, certified surrogate sup error,
the covering-net coverage level implied by $$q_{\mathrm{eff}}$$, realized
one-sided envelope validity, the optimum envelope term, and a branch-and-bound
bracket for optimizer error. The collector writes raw and best-of-start tables,
seed-grouped explanatory summaries, and five direct decomposition plots under
`results/continuous-gp-regret-decomposition/`.

### Zeroth-Order Support Envelopes

The committed support-envelope sweep reuses the same strongly convex proof
objective and compares a constant control, a constant-derivative distance
envelope, and a smooth monotone-but-nonconvex saturating envelope outside the
covered interval $$[0.75,1.25]$$:

```bash
python scripts/run_experiment_manifest.py manifests/zeroth_order_envelopes.json
python scripts/analyze_zeroth_order_envelopes.py \
  --manifest manifests/zeroth_order_envelopes.json
```

The 54 variants vary matched envelope strength, initialization, and (for the
nonconvex form) zeroth-order radius. The analyzer writes exact and
population-smoothed stationary points, final-iterate/regret tables, basin-rate
and finite-run target-error plots, and `envelope_diagnostics.png`, which shows
each envelope together with the true and upper objectives, under
`results/zeroth-order-envelope-analysis/`.

The three `seed_convergence_*.png` plots show every final $$u_K$$ across seeds
against the true minimum, exact upper-objective global minimum, and estimator
population-global minimum. Seeds vary only the optimizer RNG: finite difference
is coincident across seeds, while Stein uses a different Gaussian perturbation
stream. The two initializations are distinct sweep conditions. In the CSVs,
`x_star` is the clean-objective minimum, `exact_global_x` is the exact
upper-objective global minimum, and `population_global_x` is the
estimator-smoothed global minimum. The assigned population stationary point
(`assigned_population_x`) is the closest population local minimum to a final
iterate; it need not be population-global.

Optimization step rules:
- `l-bfgs-b` uses `scipy.minimize(method="L-BFGS-B")`.
- `trust-constr` uses `scipy.minimize(method="trust-constr")` and adds the
  acceptance-floor equation directly as a nonlinear constraint. Optional
  `initial_constr_penalty` is passed through to SciPy when configured.
- `constant` uses the repo's manual gradient loop with fixed `step_size`.
- `armijo` uses the same manual loop with Armijo backtracking seeded by `step_size`.
- `optax-adam` / `optax-sgd` run an optax update loop with `step_size` as the
  learning rate. Gradients come from the configured estimator, so all gradient
  methods work; acceptance floors use the penalty or Lagrangian objective paths
  (there is no constrained optax driver). Compare against SciPy with
  `scripts/benchmark_optax_vs_trust_constr.py`.

`enabled_estimators=("constant", ...)` optimizes a one-parameter
`ConstantPolicy` on the same objective/data as the configured policy. This is
separate from `ExperimentConfig.constant_u_baselines`, which only evaluates
fixed-action reference points such as `(-0.3, 0.0, 0.2)` and shows them in
`summary.json`, the console summary, and `plots/optimization/loss_curves.png`.
