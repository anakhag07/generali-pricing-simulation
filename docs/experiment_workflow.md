# Plan: compose settings, bands and optimization

Status: source refactor is planned, not implemented. This replaces the broader
2026-10-22 proposal. Keep review on PR #182; do not merge without an explicit
request. Runtime implementation is worktree-required and should be delivered in
small, reviewable changes after the remaining scientific choices are resolved.

## Scope and user interaction

Keep the existing manifests, presets, seed machinery, local/Slurm launcher,
optimizer, reporters and saved-result organization. Natural language is how the
user specifies an experiment to the agent; it does not require a new parser or
execution framework. Inspect the existing setup first and satisfy run requests
through configuration whenever possible. Follow
[AGENTS.md](../AGENTS.md#natural-language-experiment-requests) for clarification.

Ask only about consequential choices that the prompt and identified family do
not resolve: setting/data, target, optimizer, band variant/domain, seeds and
compute budget. Preserve uncertainty, explain inherited assumptions, and wait
before launching dependent work. A missing implementation capability should be
described before proposing new runtime code.

## Implementation checklist

- [ ] **Expose reusable preparation and band construction.** Extract preparation
  from existing factories/current study drivers only where callers need it.
  Extend `experiments/envelopes/core.py` to preserve explicit lower/upper
  endpoints; reuse `linear.py`, `wild.py`, and existing support calculations.
  Keep observations separate from synthetic truth, retain Generali row/model
  identities, and save fitted numeric state for deterministic replay.
- [ ] **Compose with the existing optimizer.** Build a selected band target as
  an existing `Objective`, preserving signs, units, policy gradients, constraints,
  and theta/action perturbation semantics. Use `ExperimentConfig`,
  `execute_experiment_run` and `src/optimization/`. Validate method/domain and
  gradient capabilities before launch. Band-only execution needs no dummy
  optimization configuration.
- [ ] **Extend existing configuration, reporting and replay.** Add only the
  missing operation/band/target choices to the current manifest path; keep old
  manifests readable. Save the resolved setup, named seeds, input/source hashes,
  fitted-band reference and optimizer outputs through existing reporters.
  Strengthen generic completion checks to verify identity rather than only the
  existence of `summary-seed-*.json`. Reports replay saved state without refitting
  or optimizing; changed contracts require a fresh output location in the same
  family. No bulk manifest-schema or historical result-layout migration.
- [ ] **Move concluded study orchestration to scratch.** Audit imports first,
  extract shared mathematics only where necessary, then move selected study-only
  drivers/recipes and their private material to `scratch/retired_experiments/`.
  Outputs belong in shared `scratch-results/`. Record relocations and recovery
  instructions; production `src` must not depend on scratch. Reuse the existing
  [archive selection](../scratch/retired_experiments/README.md), not a new list
  inferred from filenames or age.

## Interfaces and file targets

These names describe proposed interfaces, not currently supported commands.

| Responsibility | Location and minimal interface |
|---|---|
| Setting | Existing `experiments/configs/` factories; `experiments/settings.py` only if shared preparation warrants extraction. `prepare_setting(preset, overrides, seeds)` returns the requested observations, predictor/function, cohorts/domain and provenance. |
| Band | Existing `experiments/envelopes/`: fit/load a `FittedEnvelope`; `evaluate(X, actions)` returns center, lower/upper endpoints and diagnostics. Add `support.py` only to adapt existing `data/coverage.py` / `objective/gridded.py` behavior. |
| Optimization target | Existing `Objective.value/grad`; add `objective/banded.py` if existing composition cannot express the selected endpoint. All reported solutions use the repository optimizer or its exact saved outputs. |
| Execution/output | Existing `manifest.py`, `execution.py`, `run.py`, `results.py`, `provenance.py` and reporting modules. Keep band mathematics out of orchestration. |

Preserve the established OLS and wild-bootstrap formulas, asymmetric residual
transfer, calibration rules and support normalization. A common interface does
not imply every method applies to every setting or has a probability guarantee.
Shared polynomial certificates and Fourier-GP representations stay reusable even
when their original study drivers move to scratch. Do not retire `wild_study.py`
or the retained terminal/residual/fixed-noise studies as part of this move.

## Where experiments are specified

| Choice | Source of truth |
|---|---|
| Stable family and conditions | Manifest `name`, `variants` / `matrix`; adjustments normally stay in the same family. |
| Synthetic/Generali setup, model, data, policy and domain | Existing `objective.preset` / `objective.overrides`, registered factories and family setup documentation. |
| Optimizer/constraints and stochastic pairing | Existing `optimizer` and explicit `seeds`; preserve current study-specific streams. |
| Band and requested stages | Proposed optional operation/band/target fields, validated by the existing manifest path; exact field names remain an implementation decision. |
| Launch and reports | Existing `launch` and `reporting`; use `scripts/run_experiment_manifest.py`. |
| Executed settings | Resolved manifest/run metadata and generated `EXPERIMENT.md`; setup guides explain meaning and assumptions without duplicating executable defaults. |

- Function optimization: prepare objective/inputs -> optimize -> save/evaluate.
- Band construction: prepare observations/predictor -> fit/load band -> save/evaluate.
- Band-based optimization: prepare -> fit/load band -> compose target -> optimize
  -> save/evaluate. Evaluation of an already optimized policy is a separate
  explicit choice and does not rerun optimization.

## Validation and remaining decisions

Use existing small fixtures and saved artifacts to check: coefficients and
asymmetric endpoints before/after extraction; observation/truth separation and
row alignment; exact seed pairing; objective/constraint and optimizer parity;
band-only execution; changed-contract rejection; and output-only replay. Confirm
the proposed test structure before adding tests, as required by AGENTS.md.
Run relevant checks and a small demo when runtime changes, with broader checks
as required. Update README/AGENTS and public exports/docstrings/pdoc as interfaces
change; update MATH references without changing established mathematics.

Before a new Generali calibration run, resolve its response/quantity (such as
profit versus acceptance), calibration cohort and justified assumptions. Before
introducing stochastic behavior, confirm seed ownership and the representation,
dependence, off-grid behavior and confidence domain required by AGENTS.md.
Never fill these scientific gaps with arbitrary defaults or treat a fitted
Generali model as known truth.

The already authorized archive moves and completed-worktree cleanup are separate
from this source refactor. Their recovery records are in
[`scratch/retired_experiments/`](../scratch/retired_experiments/README.md).
