# Uncertainty-envelopes

This experiment evaluates simultaneous coverage and radius/excess conservatism.
Truth, the frozen predictor, radius construction, containment checker, and
probability procedure are separate launch choices. The first construction uses
contextual linear OLS with a known variance bound; the shared runner and metrics
also accept nonlinear models and custom radius builders. Earlier bootstrap and
proof experiments remain separate. Mathematical definitions are in
[MATH.md §7.8](../MATH.md#78-uncertainty-envelopes-interchangeable-construction-and-evaluation).

## Run the supplied experiments

Run these commands from the checkout root after activating the environment:

```bash
conda activate simulation_env
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes.json --launch local
```

This fits once using 60 calibration observations with three contextual
coordinates. It checks the realized transferred envelope at all 60 contexts
and every real action, and records the analytical probability lower bound 0.95.
It does not estimate a sampling coverage frequency from that one realization.

The repeated-calibration launch freezes the same contexts, logged actions, and
predictor, then refits on 20 independent response vectors:

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-empirical.json --launch local
```

Twenty repetitions are a small demonstration; copy the manifest and increase
`coverage.repetitions` for a more precise empirical estimate. Slurm can assign
one repetition to each task:

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-empirical.json --launch slurm --array --array-max-parallel 4
```

A supplied constant-radius baseline demonstrates a construction with no
analytical probability claim:

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-constant.json --launch local
```

It checks radius 1 at the finite action set `[-2, -1, 0, 1, 2]` for every
calibration context. Since this baseline ignores the response data, its envelope
and containment outcome are identical across repetitions. Data-dependent
builders use the same execution path but can have varying outcomes.

Outputs use the shared external results root, normally
`~/projects/generali-pricing/results/<name>/`. Override it with `--runs-root`
or `GENERALI_RESULTS_ROOT`. The principal launch uses name
`uncertainty-envelopes`; the other examples use their filename stem.

## Change a particular construction

Copy a manifest and change its `name` whenever making a new comparison. Saved
configuration, source, and input identities are checked before reuse; changing
only the filename does not create a new output directory. For example, create
a radius-0.4 variant of the constant baseline:

```bash
python - <<'PY'
import json
from pathlib import Path

p = json.loads(Path("manifests/uncertainty-envelopes-constant.json").read_text())
p["name"] = "uncertainty-envelopes-radius-0p4"
p["envelope"] = {"type": "constant_radius", "radius": 0.4}
Path("manifests/uncertainty-envelopes-radius-0p4.json").write_text(
    json.dumps(p, indent=2) + "\n"
)
PY
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-radius-0p4.json --launch local
```

For the built-in linear transfer radius, edit the `envelope` block:

```json
{
  "type": "linear_known_variance_transfer",
  "delta": 0.05,
  "sigma_bound": 1.0,
  "width_multiplier": 1.0
}
```

This constructs `r = abs(predictor - auxiliary_OLS) + width_multiplier * w`,
where `w = sigma_bound * sqrt(p / delta) * sqrt(phi' Q_inverse phi)` and
`phi=(1,x',a)'`, `p=d+2`. With three context coordinates, `p=5`. Dimension enters
this factor and leverage, without an additional union bound over contexts or
actions. Coefficients in a linear model are ordered
`[intercept, x_1, ..., x_d, action]`.

`sigma_bound` is a known upper bound on the observation standard deviation,
not an estimated residual scale. A full-rank calibration design is required.
The built-in analytical procedure validates linear truth, independent Gaussian
noise with `std <= sigma_bound`, and `width_multiplier >= 1`. The underlying
moment proof only needs conditional zero means, zero cross-covariances, and
bounded variances. Another noise adapter requires a suitable probability
procedure, or `"procedure": "none"`.

To study a multiplier below one, set `coverage.procedure` to `"none"` and use
empirical evaluation or one realized check. The analytical probability claim
must change when its construction or assumptions change. The all-real linear
checker still applies to this scaled linear transfer construction.

## Supply your own radius

Create an importable module, for example `src/experiments/my_envelope.py`:

```python
import numpy as np
from experiments.envelopes import FittedEnvelope


def fit(data, predictor, target, settings, rng):
    """Example construction; replace the radius formula or fit from data here."""
    level = float(settings["options"]["radius"])

    def radius(X, actions):
        return np.full(len(actions), level)

    return FittedEnvelope(
        center=predictor,
        radius=radius,
        state={"radius": level},
        metadata={"construction": "my_radius"},
    )
```

Copy the constant-baseline manifest, give it a new `name`, and replace
`envelope` with:

```json
{
  "type": "python",
  "callable": "experiments.my_envelope:fit",
  "options": {"radius": 0.4}
}
```

Run the copied manifest with the same launcher. Retain its finite target,
`"containment_checker": "finite_evaluation"`, and `"procedure": "none"`
until you supply a checker or probability result justified for your formula.
The finite target is precisely the Cartesian product of saved contexts and
listed actions; coverage there does not claim coverage between actions.

The builder signature is:

```text
fit(data, predictor, target, settings, rng) -> FittedEnvelope
```

- `data` is `CalibrationData(X, actions, y)` with read-only arrays. Contexts have
  shape `(n,d)` and actions/responses have shape `(n,)`. Synthetic truth is not
  passed to the builder.
- `predictor(X, actions)` returns one finite value per paired location.
  `target.contexts` supplies the target contexts; `target.actions` is `None`
  for the real line or the finite action vector otherwise.
- `settings` is the entire `envelope` block, so custom options are under
  `settings["options"]`.
- `rng` is the dedicated method generator for this repetition. Fit or draw
  bootstrap resamples once here; retain their results in the fitted closure.
  Evaluating the fitted envelope must be deterministic and must not draw new
  randomness per query.
- `FittedEnvelope.center` and `.radius` are paired-location callables returning
  shape `(m,)`. Radii must be finite and nonnegative. `state` and `metadata`
  must be JSON-serializable, with NumPy arrays supported. Save enough state
  to reproduce the fitted construction. An optional `components(X, actions)`
  callable returns a mapping of named diagnostic arrays.

Builders, checkers, model factories, and noise adapters are trusted local Python
code. Their own source files are hashed automatically. If they import additional
implementation files, list those under top-level `source_files`, with paths
relative to the manifest directory, for example
`["../src/experiments/my_helpers.py"]`. This list is optional, but omitted
imported dependencies are not automatically tracked.

After editing a callback's source, start a new Python process before running
or collecting again. In a notebook, restart the kernel. Python caches imported
modules; otherwise an old callable can remain in memory while provenance hashes
the newly edited file. Each command-line launch above starts a fresh process.

## Change truth, predictor, or coverage procedure

Both model entries can use a Python factory. Add, for example:

```python
def nonlinear_truth(options):
    amplitude = float(options.get("amplitude", 1.0))

    def evaluate(X, actions):
        return X[:, 0] + amplitude * np.sin(actions)

    return evaluate
```

Then replace `problem.truth` in the custom/constant finite-target manifest with:

```json
{
  "type": "python",
  "callable": "experiments.my_envelope:nonlinear_truth",
  "options": {"amplitude": 0.8}
}
```

The factory receives only its `options` dictionary. Both the factory and its
returned `(X, actions) -> values` function must be deterministic: repeated
factory calls must reconstruct the same frozen model, and repeated evaluations
must return the same values. The same interface is available for the frozen
predictor. Model factories have no randomness stream; any learned or random
model must already be frozen and identified by its declared inputs.

Preparation fingerprints the design and model outputs at the logged rows and
finite diagnostic locations. These hashes detect changes at those locations;
they do not prove that two custom models agree everywhere or enforce
determinism at untested locations. The deterministic, frozen-model contract
remains the adapter author's responsibility.

Coverage has three distinct meanings in the output:

| Output | Meaning |
|---|---|
| Analytical `lower_bound` or `exact_probability` | A probability statement with its event, conditioning, assumptions, and derivation |
| `covered`, `violated`, or `unknown` | Containment decision for one realized fitted envelope on the declared target |
| Empirical estimate and Wilson interval | Frequency across independent outer calibration repetitions, conditional on the frozen design and predictor |

`coverage.mode = "analytical"` performs one realization. It does not imply that
a probability theorem is available: `procedure = "none"` is permitted.
`coverage.mode = "empirical"` requires `repetitions >= 2` and the explicit
setting `"redraw": "responses"`. It refits the entire construction each time.
An inner bootstrap count belongs to the envelope adapter and is separate from
this outer repetition count.

The supplied linear probability procedure proves at least `1-delta`; it does
not compute the exact sampling probability. Its all-real containment checker
separately tests the final transferred envelope, so failure of an auxiliary
sufficient event is not automatically reported as final-envelope failure.
Polynomial signs are exact for the stored floating-point coefficients and
covariance treated as rational numbers. This does not enclose floating-point
OLS or evaluation roundoff. The finite checker uses ordinary floating-point
evaluations at exactly the specified finite locations.

Custom coverage adapters also use `{"type":"python", "callable":"module:function"}`:

```text
checker(envelope, truth, target) -> ContainmentResult
procedure(data, truth, predictor, target, envelope_settings, noise_std) -> ProbabilityStatement
```

Place their specifications in `coverage.containment_checker` and
`coverage.procedure`, respectively. `noise_std` is the configured Gaussian
standard deviation, or `None` for a custom noise adapter. Checker/procedure
specifications select callables; their `options` are not passed separately.
A checker must explicitly return `covered`, `violated`, or `unknown` and the
correct target kind. A theorem procedure must return the typed probability
statement with assumptions and derivation. Unknown cases cannot establish
continuous coverage by falling back silently to diagnostic points.

The supplied `uncertainty-envelopes-constant-3.json` uses the Python checker
`experiments.envelopes.linear:check_constant_linear_containment`. It checks a
literal constant radius for affine truth/predictor over the whole real line:
unequal action slopes imply failure; equal slopes reduce the check to the
contextual intercept errors. This checker supplies realized containment only.

Custom noise uses `calibration.noise` with the same Python specification and
the signature `noise(X, actions, mean, options, rng) -> y`. It returns the full
response vector, including the mean. It uses observation randomness, independently
of method randomness. Its dependence and variance assumptions must be recorded
in the custom procedure and experiment specification.

## Use a saved calibration dataset

Save an NPZ containing `X` of shape `(n,d)` and vectors `actions` and `y` of
shape `(n,)`. Compute its SHA256, for example with `sha256sum calibration.npz`.
Replace `calibration` with:

```json
{
  "type": "saved",
  "path": "../calibration.npz",
  "sha256": "REPLACE_WITH_THE_64_CHARACTER_SHA256",
  "noise": {"type": "gaussian", "std": 1.0}
}
```

The path is relative to the manifest file. Analytical mode reuses its saved
`y`. Empirical mode deliberately discards that response vector and redraws
responses under the declared truth/noise law at the saved `X` and `actions`.
The checksum verifies identity, not the statistical assumptions of external
observations. Keep `noise` explicit even for analytical use because it supplies
the asserted observation model for any analytical guarantee.

## Metrics, domain, and seeds

The common pointwise quantities are radius `r`, absolute prediction error
`abs(center - truth)`, and signed excess `r - abs(center - truth)`. Negative
excess identifies noncontainment at a measured location and is never clipped.
On the auxiliary-coverage event for the linear transfer construction,
`0 <= excess <= 2*w` with the actual auxiliary width.

Coverage and diagnostic domains are separate. The default all-real coverage
target is evaluated analytically; tightness is measured at 101 actions in
`[-2,2]` for each saved context. This finite diagnostic measure is not a uniform
law on the real line. Changing its resolution cannot change the fitted envelope
or analytical coverage result. No interpolation of these points defines a
random function, and the reported metrics are not optimizer-selected actions
or global extrema.

`tightness.actions` accepts `{"type":"linspace","min":-2,"max":2,"count":101}`
or `{"type":"values","values":[-2,0,2]}`. `tightness.weights` is `"uniform"`
or a nonnegative vector with positive sum, one weight per context/action pair
in context-major order. Weights are normalized. Each repetition saves weighted
means and inverse-empirical-CDF quantiles at 0.5 and 0.9, plus the diagnostic
weight having negative excess. PDF histograms give each available repetition
equal total weight.

Four distinct named seed roots are mandatory: `contexts`, `actions`,
`observations`, and `method`. The synthetic design draws independent standard
normal contexts and logged actions once. It remains fixed across repetitions.
Observation and method seeds are derived independently by repetition index,
so scheduling order does not change datasets or fits. Within each fitted
envelope, values share the same fitted parameters; they are not independent
pointwise error draws. Off-grid built-in predictions use the explicit affine
formula, and custom models/radii must define evaluation on their whole target.
There are no optimizer queries, action-boundary rules, or global-reference
tolerances in this experiment because it performs no action optimization.

Keeping the same roots in copied manifests pairs design and observation draws
for parameter comparisons when sampling shapes/laws remain compatible. A
diagnostic-only change consumes no randomness. Use distinct names to preserve
each condition; changing roots makes a different design or noise experiment.

## Inspect and replay results

Each result directory contains:

- `contract.json`: resolved manifest, source records, input identity, prepared
  design/output fingerprints, Python and NumPy/SciPy/SymPy/Matplotlib versions,
  and a combined contract hash. Prepared fingerprints cover contexts, logged
  actions, and truth/predictor outputs at logged and finite diagnostic locations.
- `repetitions/000000/` and subsequent repetitions: `calibration.npz`,
  `fit.json`, `points.npz`, and `summary.json` with seeds, containment decisions,
  probability statements, metrics, and artifact hashes. `completion.json`
  records the summary's path, SHA256, and size after the attempt finishes.
- `summary.json`, `metrics.csv`, and `EXPERIMENT.md`: aggregate results,
  per-repetition metrics, and the run's representation/seed contract.
- `reports/01_coverage.pdf`: probability statements and realized outcome counts
  in separate panels.
- `reports/02_radius_excess.pdf`: radius and signed-excess histograms over the
  saved finite diagnostic measure.

`points.npz` retains contexts, actions, center, truth, radius, absolute error,
excess, normalized weights, and any construction components. Fit or evaluation
failures retain their failing stage and error in the repetition summary.
Resume and collection verify its `completion.json` record before trusting that
summary, then verify every artifact it lists. A completion record marks a saved
attempt, including a recorded failure; it does not imply successful fitting or
coverage. A summary without a completion record is incomplete and is counted
as missing during collection.
Unknown, failed, and missing repetitions are counted; the empirical point
estimate and interval are withheld unless every requested repetition has a
decisive containment result. Metric reports label conditioning on the successful
evaluations when others are unavailable.

Rerunning an unchanged launch resumes verified saved repetitions. To collect
and redraw reports without fitting:

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes.json --launch local --collect
```

Collection reconstructs the frozen models and design to verify the source,
configuration, runtime versions, and sampled model-output fingerprints. It
also verifies completed summaries and their artifacts, without refitting any
envelope. After code or dependency changes, use a new experiment name for a
new run; `--force` explicitly reruns/replaces that name's results. To redraw PDFs
directly from historical saved diagnostics without requiring the original
fitting source:

```bash
PYTHONPATH=src python - <<'PY'
from experiments.paths import results_root
from experiments.envelopes.reporting import write_reports

for path in write_reports(results_root() / "uncertainty-envelopes"):
    print(path)
PY
```

Direct reporting verifies the point artifacts' hashes and sizes, reads no model
or builder, and never refits. It overwrites only the two reports; it does not
update the aggregate summary's old report hashes. Full collection refreshes
those records when the source contract still matches.

The implementation lives in `src/experiments/envelopes/`: `core.py` defines
construction-neutral records and metrics, `linear.py` holds the initial
mathematics, `study.py` handles manifests/adapters/seeding/artifacts, and
`reporting.py` replays the saved diagnostics. Bootstrap calibration can later
be supplied as another builder without changing the metrics or outer runner.

## Render the constant-3 versus writeup comparison

The following paired launches retain the same 60 three-dimensional contexts,
logged actions, responses, truth, predictor, and seeds. The constant construction
uses the literal radius `r=3` (full band width 6). The writeup construction uses
`r=abs(predictor-auxiliary_OLS)+w` with `sigma_bound=1`, `delta=.05`, and `p=5`,
so `w=10*sqrt(phi' Q_inverse phi)`. Neither radius is optimized.
The frozen predictor is hand specified by the manifest coefficients; the
auxiliary OLS model is the object fitted to calibration observations.

To create the paired experiments initially:

```bash
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-constant-3.json --launch local
PYTHONPATH=src python scripts/run_experiment_manifest.py manifests/uncertainty-envelopes-writeup.json --launch local
```

To render or replace the comparison from their existing saved arrays:

```bash
PYTHONPATH=src python scratch/render_uncertainty_envelope_comparison.py
```

The rendering command preserves the source experiments, responses, fits, and
radii. It performs no refitting and writes to
`results/uncertainty-envelopes-comparison/`:

- `01_coverage_comparison.pdf`: one PDF with two panels named
  `constant radius = 3` and `analytical-from-linear-f radius`. Each panel shows
  the common truth after removing the context contribution, the mean adjusted
  frozen predictor, and the intersection of adjusted envelopes across all
  60 contexts. An on-plot note states the context adjustment.
- `adjusted_coverage.npz`: saved common lower/upper endpoints, common truth,
  mean adjusted predictor, and empty-intersection/coverage flags for the
  diagnostic actions. These derived arrays remain traceable to the original
  hashed point artifacts.
- `metrics.csv` and `comparison.json`: full 60-context diagnostics, coverage
  statements, and input/source/output identities.
- `window_certificate.json`: exact endpoint checks certifying the constant
  band's containment throughout `[-2,2]` at all 60 contexts. Convexity of the
  absolute affine error extends the endpoint checks to the whole interval.
- `03_calibration_dataset.pdf`: three panels sharing the action axis `[-2,2]`.
  The top retains the analytical construction's original adjusted common band,
  truth, and mean frozen predictor, and adds the observed `y_i-g(x_i)` points.
  The middle shows one curve, `H(a)=(common_upper(a)-common_lower(a))/2`, for
  exactly the same all-context intersection shown above; it is masked where
  that intersection is empty. The bottom shows logged-action
  counts in bins of width 0.5, plus a rug marking the individual actions.
- `calibration_action_support.json`: the displayed curves, action-support
  counts, and source/artifact provenance. Mean radius, auxiliary width, and
  predictor/auxiliary discrepancy remain saved diagnostics but are not plotted.

The calibration display shows 58 of the 60 logged actions in `[-2,2]`; the two
actions above 2 are explicitly noted. It preserves the saved data, fitted
auxiliary model, and radius construction without refitting. The noisy response
points need not lie within the shaded overlap: the envelope targets the true
mean response, not noisy individual observations. Both upper panels use the
same intersection across all 60 contexts. Its half-width is not the mean
constructed radius, and neither individual radii nor their guarantees change.
Analytical width depends on the full joint design
through `phi' Q_inverse phi`, not inverse local action density, so histogram
counts alone do not determine its shape.

To generate only the calibration plot from the saved writeup run:

```python
from experiments.envelopes.reporting import write_calibration_plot
from experiments.paths import results_root

root = results_root()
write_calibration_plot(root / "uncertainty-envelopes-writeup",
                       root / "uncertainty-envelopes-comparison")
```

[The standalone LaTeX section](uncertainty_envelopes_simulation.tex) records the
exact simulation models, sampling, fitting, radii, coverage target, and metrics.
Copy its section body into the mathematical writeup.

The previous context-slice and radius/excess comparison PDFs are archived only
after the replacement PDF renders successfully. The source experiments' own
coverage and radius/excess reports remain available.

For this additive truth, write

```text
g(x) = beta_0 + beta_x' x = x_1 - 0.5*x_2 + 0.25*x_3
f(x,a) - g(x) = h(a) = 2*a.
```

Subtracting `g(x_i)` from each saved context's truth, predictor, and envelope
endpoints puts every context on the same response scale. It leaves each radius
and coverage decision unchanged. At each saved action, the shaded overlap is

```text
common_lower(a) = max_i [predictor(x_i,a) - g(x_i) - radius(x_i,a)]
common_upper(a) = min_i [predictor(x_i,a) - g(x_i) + radius(x_i,a)].
```

The blue common-truth line is inside this intersection exactly when every
context's original envelope contains its truth at that action. If the common
lower endpoint exceeds the upper endpoint, the intersection is empty and
there is no common covered interval. The shading summarizes all contexts;
it is not any individual radius or band. In particular, even the constant-3
panel's common intersection need not have width 6, although every original
context-specific band has width 6.

The mean adjusted frozen-predictor curve gives orientation only. It is not
the center used to define coverage, and distances from that mean to the
intersection boundaries are not the constructed radii. The known synthetic
`g(x)` is used only for this evaluation display, never for fitting the envelope.
Lines join saved diagnostic actions for visualization; they do not replace the
separate continuum containment checks.

The metrics use equal weight on the same 60-by-101 diagnostic locations in
`[-2,2]` and retain the original per-context radius and signed excess. The
plotting window and metrics do not limit the all-real coverage checks. For
these models, the predictor's action slope is 1.7 and truth's is 2.0, so every
finite constant radius fails somewhere on the real line. Radius 3 covers this
displayed window. Since this baseline and predictor are fixed, its coverage
event does not change when calibration responses are redrawn. It is not a
fitted 95% calibration procedure. The writeup construction has the separate
theoretical lower bound of .95 and a realized all-real certificate.
