# Maximum leverage and wild-bootstrap coverage

This fixed-design diagnostic compares raw-residual wild-bootstrap coverage
against maximum observed leverage for standard-normal logged actions truncated
to [-0.5,0.5], [-1,1], [-3,3], and [-10,10]. Actions are not centered,
standardized, or clipped. Each condition has n=300, p=2, B=1999, delta=0.05,
and 2000 independent response/bootstrap repetitions. Mathematical definitions
and representation decisions are in MATH.md section 7.10.

The feature map is phi(a)=(1,a), the true mean is f(a)=1+a, the frozen
predictor is fhat(a)=1.2+0.8a, and the observation noise is independent
N(0,0.5^2). Each design is generated once and frozen. Separate roots 1103
(observations) and 1104 (wild multipliers) derive independent repetition
streams. The actions root 1102 supplies one shared uniform vector transformed
by each truncated-normal inverse CDF, pairing the designs across bounds.
Observation innovations and multipliers are also paired across bounds.
Roots 1101 (contexts) and 1105 (variance profile) remain in the manifest for
schema compatibility and are unused in this zero-context, constant-noise case.
No additional random process or seed stream is introduced.

Coverage is the frequency of a single joint event per response dataset:
the auxiliary band contains the true function for every real action. The
existing coefficient statistic includes both tails without a finite target
grid. All real queries use the exact affine formula; there is no interpolated
random objective, optimizer query, reported pricing optimum, or optimization
reference. Double-precision statistics have the numerical limitations described
in the wild-bootstrap guide. The diagnostic grid [-2,2] is used only for the
existing width summaries, never to calibrate or certify coverage.

The existing raw-residual Rademacher bootstrap and higher quantile are retained.
At B=1999 and delta=0.05 this selects order statistic 1900, matching the updated
proof's ceil(B*(1-delta)) convention. No HC2/HC3 correction is applied.

Run the four manifests through the existing entry point, for example:

```bash
conda activate simulation_env
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/run_experiment_manifest.py \
  manifests/wild-bootstrap-leverage-bound-0p5.json --launch local
```

Repeat for suffixes `1`, `3`, and `10`. Saved studies live under the configured
results root as `wild-bootstrap-leverage-bound-<suffix>/`. The standard
checkpoint, source hash, environment, and per-repetition provenance checks
remain active. Each summary includes actual maximum leverage, auxiliary and
transferred coverage counts, and pointwise 95% Wilson intervals. Summaries
condition on each realized design; they do not average over random designs.

After all four complete, replay the combined figure with:

```bash
PYTHONPATH=src python scratch/plot_wild_bootstrap_leverage.py
```

The replay writes `wild-bootstrap-leverage/coverage_vs_leverage.pdf`, a CSV,
and input/source/output hashes. It uses the same raw measured h_n as the
simulation; centering is not needed to compare leverage. With an intercept,
an invertible affine feature change would leave all leverage values unchanged.
The four points are not connected: the proof does not supply a deterministic
or monotonic coverage function of h_n. Its leverage term controls a worst-case
approximation error, which need not be attained by these designs. The full
leverage pattern matters as well. Transferred coverage is kept separate because
the discrepancy term can conceal auxiliary bootstrap undercoverage.

At coverage near 95%, 2000 repetitions give roughly one percentage point of
marginal 95% Monte Carlo uncertainty. Small differences or overlapping
intervals do not establish that leverage has no effect. Paired outcomes can
resolve differences more precisely because conditions share random draws.

## Wide-bound comparison

The follow-up compares bounds 0.5, 1, 10, 100, 1000, and an unbounded standard
normal, retaining all settings above. Additional manifests use suffixes `100`,
`1000`, and `unbounded`. The unbounded manifest specifies `truncation_bound:
null`, interpreted as inverse-CDF sampling on the whole real line using the
same uniforms. Omitting the bound is invalid. The underlying normal location
and standard deviation remain 0 and 1; these bounds do not rescale the data.
No new seed stream is required. The legacy independent-standard-normal
generator for other experiments continues to use its original normal draws.

The earlier 0.5, 1, and 10 results are replayed from their exact saved summaries
and repetitions with original provenance. Only the three additional conditions
need new runs. This preserves the earlier comparison and its artifacts. A
changed numerical source contract prevents resuming old conditions with the
new source; do not use `--force` to overwrite them solely to make this report.

After running the three additional manifests with the existing entry point:

```bash
PYTHONPATH=src python scratch/plot_wild_bootstrap_leverage.py --wide-bounds
```

The extended report is written to `wild-bootstrap-leverage-wide-bounds/`.
It retains six rows in the CSV. Conditions whose leverage, coverage, and
interval endpoints coincide to 12 decimal places share one plotted marker
with a combined legend label; no leverage jitter is introduced. The report
provenance records those groups and hashes every input summary.

For the paired seed used here, bounds 10, 100, 1000, and unbounded produce
identical floating-point action arrays. Their realized action range is about
[-2.697,2.559]. A large permitted interval does not force observations into its
tails, so this extension is not a large-leverage perturbation of the design.
