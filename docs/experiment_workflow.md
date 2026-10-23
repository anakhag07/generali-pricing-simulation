# Settings, bands and optimization

The existing manifest runner now supports `task: "optimize"` (the default),
`"band"`, and `"optimize_with_band"`. There is one execution pipeline and no
natural-language parser. An agent translates a request into an existing family
manifest after resolving consequential unknowns; see
[AGENTS.md](../AGENTS.md#natural-language-experiment-requests).

Keep the family name when changing settings. Use variants, matrix axes, or a
fresh run root for a revision. Saved runs are immutable: changing data, sources,
parameters or seed pairing requires a fresh location within that family.

## Where choices live

| Choice | Source of truth |
|---|---|
| Family, variants, launch resources, reporting | Existing manifest `name`, `variants`/`matrix`, `launch`, `reporting` |
| Optimization-only function, data/model, policy | Registered `objective.preset` and `objective.overrides` |
| Band inputs | Manifest `setting`: an existing synthetic study, supplied observations, or explicit Generali rows |
| Construction and assumptions | Manifest `band`: method, calibration parameters, domain; or one saved `artifact` |
| Band optimization target | `objective.preset: "band_policy"`; overrides specify `endpoint`, `sense`, `policy_kind`, bounds |
| Solver, first-/zeroth-order estimator, budget | Existing `optimizer` fields, including `step_rule`, `enabled_estimators`, `perturbation_space` |
| Randomness | Existing optimizer `seeds`; synthetic study streams or supplied bootstrap multipliers for band fitting |
| Executed configuration and evidence | `EXPERIMENT.md`, `run-contract.json`, fitted state, input/source hashes and existing per-seed reporters |

## Source layout and interfaces

- `experiments/settings.py`: `prepare_setting(spec, base_path=...)` returns
  observations, frozen predictor, contexts, provenance and (only for synthetic
  evaluation) truth. `repetition_draws` reuses the retained study's exact stream
  derivations, full draw shapes and prefix pairing.
- `experiments/envelopes/`: `fit_band(data, predictor, target, settings,
  multipliers=...)`, `fit_support_band(...)`, `save_band(...)`, `load_band(...)`.
  A `FittedEnvelope.evaluate(X, actions)` returns `center`, `lower`, `upper`, and
  diagnostic components. Asymmetric bands have `radius=None` and separate lower
  and upper radii; never silently average them for coverage or optimization.
- `objective/banded.py`: `BandObjective` composes the selected endpoint and
  policy through the existing `Objective.value/grad` interface. Maximization
  negates the objective for repository minimizers; saved endpoints retain the
  original units and sign. Acceptance constraints and scalar penalties use the
  existing objective/optimizer machinery.
- `experiments/configs/band_policy.py`: builds `ExperimentConfig` and validates
  domain/gradient capabilities. `manifest.py` and `execution.py` own orchestration;
  `run.py`, reporters and `src/optimization/` still execute optimization.
- `envelopes/quadratic.py`, `envelopes/linear.py`, `envelopes/wild.py`,
  `offline_bootstrap_math.py`, and `synthetic.py` retain shared mathematics.
  `wild_study.py`/`wild_reporting.py` retain the current study runner and reports.
  Concluded orchestration is in `scratch/retired_experiments/studies/`.

## Three flows

**Optimization only:** use the existing manifest unchanged. Its preset prepares
inputs and objective; the requested repository optimizer returns the policy;
existing reporters save theta, traces, metrics and seed metadata. To suppress
truth-derived metrics explicitly, use `truth: {"source": "none"}`.

**Band only:** set `task: "band"`, `setting`, `band`, `name`, and `launch`.
Omit optimizer and optimizer seeds. The setting supplies observations and any
bootstrap draws; fitting never receives synthetic truth. Saved state can be
loaded and evaluated without fitting, sampling or optimizing.

**Optimize with a band:** use the same setting/band and add
`objective.preset: "band_policy"`, explicit target/policy overrides,
`objective_modifications`, optimizer, and seeds. Fitting occurs once per variant;
optimizer repetitions share that exact fitted band. To vary calibration draws,
use a setting variant with a different explicit repetition. Band-based runs use
`truth.source: "none"`; known-truth coverage stays in the retained study runner.

This example selects an existing residual family setup. Values are illustrative
choices, not new scientific defaults; the source manifest still defines the
predictor, noise law, seeds and paired schedule:

```json
{
  "name": "uncertainty-envelopes-residual-paired-nb",
  "task": "band",
  "setting": {
    "source": "wild_study",
    "manifest": "uncertainty-envelopes-residual-paired-nb.json",
    "n": 30,
    "repetition": 0
  },
  "band": {"method": "residual_wild", "B": 99, "delta": 0.05},
  "launch": {"mode": "local", "array": "variant"}
}
```

Choose `n` from the referenced setup's `n_values` and `B` from the available
paired multiplier prefix. Relative input paths resolve beside the requesting
manifest. For band optimization, additionally supply (with task changed):

```json
{
  "objective": {
    "preset": "band_policy",
    "overrides": {
      "endpoint": "lower", "sense": "maximize",
      "policy_kind": "softmax", "softmax_action_bounds": [-1, 1]
    }
  },
  "objective_modifications": [],
  "optimizer": {
    "step_rule": "l-bfgs-b", "enabled_estimators": ["first_order"],
    "perturbation_space": "theta", "t_steps": 100
  },
  "seeds": {"run_seeds": [7], "anchor_seed": 7, "vary": ["theta"]}
}
```

Launch through `python scripts/run_experiment_manifest.py <manifest.json>
--runs-root <fresh-root>`. Existing local/Slurm array behavior remains available.

## Settings and method boundaries

`setting.source: "wild_study"` references the retained, validated study manifest
and an explicit `n` and `repetition`. It reproduces that design and frozen
predictor, including fixed moved-row noise. No new random process is introduced.
The actual setup is saved as `resolved_source` alongside its hash and seed roots.

`setting.source: "observations"` reads an explicit NPZ with `X`, `actions`, `y`
and optional Rademacher `multipliers`. Supply `path`, `quantity`, a `provenance`
description and frozen `predictor: {"type": "linear"|"quadratic",
"coefficients": [...]}`. Quadratic means the established action-only quadratic
basis. The caller must justify the response and calibration assumptions; an
arbitrary trained Generali model is not treated as truth or replaced by a
polynomial automatically.

`setting.source: "generali_support"` requires `model`, explicit `row_indices`,
`quantity: "profit"|"cost"|"acceptance"`, `action_grid`, `n_neighbors`, and
`action_bandwidth`. It retains source row IDs, dataset/model hashes, artifact
preprocessing and exact-spline settings where selected. No cohort is sampled.
Use `band: {"method": "support_grid", "construction": "normalized_local",
"scale": ..., "quantity": ...}` with the same quantity. This is a descriptive
support penalty, with **no probability guarantee**. The direct Python adapter
also accepts the existing `absolute_action` width construction.

Polynomial methods are `ols_transfer`, `residual_ols`, and `residual_wild`.
All require `delta`; OLS requires `sigma_bound`; manifest wild fitting requires
`B` and supplied/paired multipliers. The known-variance OLS bound and approximate
wild calibration remain distinct. Supported context domains are the calibration
contexts crossed with all real actions. Support bands use saved contexts and
the bounded action interval; local curves interpolate linearly, absolute-action
widths use the established natural cubic. Duplicate ambiguous support contexts,
unknown rows, extrapolation, unsupported methods and JAX adapters fail explicitly.

First-order evaluation uses exact represented derivatives, with zero selected
at absolute-value/positive-part kinks; this is not a smoothness claim. Zeroth-order
methods query the same band. Support optimization requires an explicitly bounded
softmax policy and theta-space perturbations to keep every query in-domain.
Automatic `constant` estimator copies and legacy `per_seed_plots` are unsupported
for band tasks; select a constant policy with the requested gradient estimator
explicitly, and use saved endpoint arrays or aggregate PDFs.
Ask about solver, bounds, confidence domain, quantity/cohort or budget whenever
these cannot be inherited from the prompt or the identified experiment.

## Outputs and replay

Each band variant saves:

- `band.json` + checksummed `band.npz`: fitted coefficients/geometry or support
  curves, observations, calibration state, resolved setting and provenance.
- `band-evaluation.npz` + `band-summary.json`: endpoint values at logged actions
  or the supplied support grid. These points do not select an optimum.
- `band-binding.json` per-seed `band-policy-seed-*.json`, and `run-contract.json`: artifact binding, executable
  inputs, source/package identity, optimizer choices and seeds.
- For optimization: ordinary `summary-seed-*.json`, per-seed theta/traces/config,
  and `seeds/seed-*/band-policy-<estimator>.npz`, evaluating the exact returned
  policy on the saved cohort. No secondary optimizer runs during this evaluation. Interrupted endpoint exports
  resume from saved theta without rerunning optimization.

Set `band: {"artifact": "path/to/band.json"}` and omit `setting` to reuse a
saved fit. `load_band(path).evaluate(X, actions)` handles standalone queries or
actions from a saved optimized policy; preserve that policy's artifact reference
in downstream reports. Evaluation does not rerun optimization. Report collection
reads saved summaries; it does not refit bands. Execution refuses a changed
contract, and generic `--force` cannot overwrite completed runs. Legacy runs
without contracts remain readable for reporting but cannot be resumed as newly
verified runs; recover their original source for exact replay.

## Retirement and recovery

Generic uncertainty-envelope, historical pairs-bootstrap/OLS, offline-bootstrap,
policy-LCB and coverage-kernel study drivers moved to tracked scratch, together
with their reporting orchestration. Current terminal/residual/fixed-noise studies,
illustration drivers, Generali models/data/configurations and shared mathematics
remain available. Recipes and results were already moved in the previous cleanup;
this source refactor does not move or delete further results.

See [archive instructions](../scratch/retired_experiments/README.md) and
[source relocation records](../scratch/retired_experiments/source_relocations_20261023.json).
Historical artifact hashes and absolute paths are unchanged. The archived runner
is for new exploratory runs at a fresh location; exact historical replay requires
the recorded source checkout and inputs.

## Refactor validation

Validation on `refactor/experiment-band-workflow` (2026-10-23):

- 117 focused retained-study, archive, band and Generali preparation checks passed;
  31 final workflow/manifest/sweep checks passed after the resume/PDF updates.
- The full suite completed with 1,268 passes and two failures. One is the already
  documented legacy proof-validation NaN plotting failure. The other was the
  environment's missing declared `Markdown` dev dependency; after installing it
  into a temporary test-only directory, all seven math-document checks passed.
- JAX initially aborted with unrestricted CPU affinity on this host. The full
  suite completed using `JAX_PLATFORMS=cpu` and `taskset -c 0-3`.
- A small `main.main` demo passed with 32 samples, three steps, seed 7, and the
  first-order, finite-difference and Stein estimators. An optimization-only CLI
  run produced valid PDFs with no raster copies. Band-only, band optimization,
  saved-fit replay and interrupted endpoint-export recovery are covered by tests.
- All 30 source relocation records match the original Git objects. Active
  manifests are unchanged; shared results and data were not modified. Both
  launchers import and expose help, and API documentation was regenerated.

The reusable band adapter is NumPy-only and intentionally validates its current
polynomial/support method and domain capabilities. New Generali response-based
calibration or a different stochastic representation still needs an explicit
scientific setup; this refactor does not supply arbitrary assumptions.
