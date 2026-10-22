# Experiment families and the downstream cleanup

An experiment family represents one scientific question. When a requested change
adjusts its noise, design, leverage, bounds, sample sizes, bootstrap count, solver,
or presentation, extend the same family. A new name is warranted by a distinct
question or an explicit request for an independent experiment.

Keep one canonical configuration entry point and reusable runner per family.
Represent comparisons through configuration overrides, named conditions, or
sweep axes. Keep completed run revisions and their configuration snapshots
identifiable within the family; Git records changes to its source. Do not silently
reuse outputs from a different configuration, force-overwrite historical runs, or
rename existing provenance records. The current specialized runners do not all
support revision subdirectories: add that capability in the later runner work
before relying on it. This archival pass does not change runner schemas or
consolidate existing result names.

## Intended reusable workflow

| User choice | Reusable responsibility | Existing foundations |
|---|---|---|
| Configure and run zeroth-order or first-order optimization | Estimator, solver, stopping rules, initialization, constraints, and trace collection | `src/optimization/`, `ExperimentConfig`, and the experiment runner |
| Construct an uncertainty band | Fit/calibrate/evaluate OLS, wild-bootstrap, or support-based bands; expose the resulting objective to the repository optimizer | `src/experiments/envelopes/`, OLS/bootstrap mathematical helpers, `src/data/coverage.py`, and `src/objective/gridded.py` |
| Specify a synthetic or Generali setting | Data, truth or fitted predictor, policy, objective, support, and explicit seed streams | `src/objective/objectives/synthetic/`, `src/objective/objectives/generali/`, `src/data/`, and config presets |

These are responsibilities for the next refactor, not a claim that a single
composable interface already exists. Extend existing boundaries first; select
module/API names after tracing their consumers. Band construction should expose
its domain, assumptions, calibration method, and guarantee separately from
empirical coverage. A support-derived radius must not inherit a bootstrap/OLS
coverage claim merely because it shares an evaluation interface.

Study-specific parameter grids, proof checks, old comparison recipes, and
one-off plots belong in `scratch/` once retired. Production code must not import
them. Shared configuration, seeding, provenance, launch, results, and reporting
remain reusable infrastructure. Preserve all Generali settings, datasets,
trained artifacts, caches, saved policies, and downstream analysis results.

## Implementation boundary

The approved first pass archives 18 historical manifests, nine legacy drivers,
and 22 corresponding result directories. See the
[archive and recovery index](../scratch/retired_experiments/README.md).
It leaves 45 active manifests and 18 files in `scripts/`.

The later source cleanup must first separate reusable mathematics from retired
study orchestration. Current wild-envelope and illustration code imports
`bootstrap_band.features`, polynomial certificates in `bootstrap_band_continuous`,
and helpers in `offline_bootstrap_math`. Wild studies also use
`policy_lcb.common` and `policy_lcb.continuous_gp_core.FourierGP`. Eager package
exports and the manifest launcher's dispatch require an audit before moving those
study modules. Keep the reusable pieces in `src/`, update consumers, and retire
the study drivers together with their exports and dispatch entries.

The 29 older uncertainty/wild-bootstrap manifests remain for a separate
selection pass. The nine paired-residual/fixed-noise manifests and two current
data-location illustrations remain available. Archive placement alone does not
mean an illustration is inactive. No extra experiments or source modules were
retired in this pass.

Before implementing new interfaces, write a focused plan covering composition,
configuration compatibility, seed ownership, saved-result identity, reporting,
and existing behavioral tests. Build in a dedicated worktree and push a PR for
review. Leave that PR open unless explicitly instructed to merge it.
