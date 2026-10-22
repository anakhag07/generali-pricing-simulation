# Local work consolidation — 2026-10-22

This is the merge-first stage of the cleanup. Experiment drivers and result
directories have not been relocated. Preserve the Generali experiments, models,
data, caches, and saved policies for the next downstream analysis.

## Branch inventory

The initial snapshot contained 25 non-main local branches across 16 worktrees.
`main` and `origin/main` were both at `7f24a0b`. Four tips contain all 23 branches
selected for consolidation, totaling 58 existing commits:

| Tip merged | Commits ahead of initial main, including shared ancestors | Contents |
|---|---:|---|
| `feature/wild-distribution-diagnostics` (`5973fd7`) | 43 | Envelope interfaces, wild bootstrap, residual bands, paired N/B studies, finite-sample comparisons, distribution perturbations and diagnostics |
| `feature/residual-leverage-fixed-noise` (`0e9f3c0`) | 45 | Residual leverage studies and fixed moved-row noise controls |
| `fix/bootstrap-figure-titles` (`bd5c883`) | 29 | Saved radius comparisons, response-averaged diagnostics and figure titles |
| `docs/contextual-simulation-one-page` (`5bba9fa`) | 5 | Current data-location and flat contextual Stein experiments, including the mean-10 follow-up |

The following earlier branch tips are all ancestors of those four tips:

| Earlier branch | Tip |
|---|---|
| `analysis/linear-envelope-dimension-sweep` | `50f5ff2` |
| `experiment/wild-bootstrap-context-flat` | `420d0c6` |
| `experiment/wild-bootstrap-data-location` | `0581173` |
| `experiment/wild-bootstrap-mean10` | `0c81dde` |
| `feature/context-averaged-envelope-bands` | `11de7da` |
| `feature/envelope-diagnostic-reports` | `597237f` |
| `feature/residual-envelope` | `e5e10a5` |
| `feature/residual-envelope-paired-nb` | `cd4f511` |
| `feature/residual-leverage` | `55bdcc8` |
| `feature/uncertainty-envelopes` | `2fe3ea6` |
| `feature/wild-bootstrap-envelopes` | `22bc8f5` |
| `feature/wild-bootstrap-leverage` | `8cfdb38` |
| `feature/wild-bootstrap-single-outlier` | `0e5f516` |
| `feature/wild-bootstrap-wide-bounds` | `19fb73f` |
| `feature/wild-distribution-perturbations` | `79ebfdd` |
| `feature/wild-finite-sample-coverage` | `84c8b5f` |
| `fix/dimension-sweep-width-only` | `50f5ff2` |
| `fix/radius-comparison-replots` | `aa9a337` |
| `fix/simplify-envelope-comparison` | `63a7b99` |

`feature/design-bench-bridge` (`2b7965f`, PR #162) remains open. At the initial
snapshot it was 10 commits ahead and 165 behind main; it will need a separate
integration pass when resumed.

`proof/bootstrap-calibration` (`0a025db`, PR #180) is excluded from the merge
and retired at the user's request. Its four unique commits are preserved in
the sibling `scratch-results/retired-proof-bootstrap-calibration-20261022.bundle`.
The bundle requires the existing main history; verify it with `git bundle verify`
and recover it with `git fetch <bundle> proof/bootstrap-calibration:proof/bootstrap-calibration`.

## Local edits and merge resolutions

The only uncommitted source change in the simulation worktrees was the
width-only edit to `scratch/plot_linear_envelope_dimension_sweep.py` in
`worktrees/feature-envelope-diagnostic-reports`. That exact edit is included.
Generated HTML conflicts were resolved by regenerating pdoc from the combined
source. Both independently added math explanations were retained: the current
data-location and contextual illustrations are now MATH.md §§7.17–7.18, with
their README and driver references updated.

The existing math-document tests exposed unsupported notation and inline display
delimiters, including issues already present on initial main. These were
normalized without changing formulas. The development extra now declares
Markdown, which those existing tests import; runtime dependencies are unchanged.

The sibling `MEng-Thesis` checkout was clean. The sibling `model_processing`
checkout had tracked and untracked Python bytecode changes only; these are
outside this consolidation and were left untouched.

## Experiments and artifacts

The combined tree has 27 files in `scripts/`, 63 manifests, and 85 files under
`scratch/` excluding bytecode. The shared output directories initially contained
112 result directories and 45 scratch-result directories. `outputs/` inside the
canonical checkout also retains two acceptance-floor reconstructions and Slurm
logs. These locations are separate; changing branches does not move results.

| Family | Observed state | Cleanup treatment |
|---|---|---|
| Current data-location illustration | Saved summaries and traces for both `wild-data-location-20261021/` and the `-mean10/` follow-up | Keep current mean-10 outputs; review the earlier setting later |
| Current contextual illustration | Summary, per-context tables and traces in `wild-contextual-flat-20261021/`; driver already in `scratch/` | Keep available for the current write-up |
| Gaussian and centered-exponential paired residual studies | Each has 2,000 repetition records, a collected summary and plots | Completed artifacts available for later archive selection |
| Heavy-tailed t3 paired study | 1,950 of 2,000 repetitions; 39 successful tasks, task 0 failed; no collected summary | Explicitly incomplete; preserve checkpoints |
| Residual leverage and fixed-noise studies | All six outlier settings in each family have summaries; fixed-noise aggregate includes validation and provenance | Completed artifacts available for later archive selection |
| Earlier envelope, bounded-design, outlier, bootstrap, kernel and GP studies | Multiple saved summaries and report trees remain under `results/` | Review against the new proof before moving |
| Generali real-data analyses | GLM/XGB/spline, policy-capacity, support, sensitivity and saved-policy results remain available | Preserve for downstream analysis |

The t3 failure is recorded in
`results/residual-heteroskedastic-t3-design-paired-nb/sweeps/20261021_120906/tasks/task_000.json`:
two initializers shared `environment.json.tmp`, and one rename failed with
`FileNotFoundError`. The shared `write_json_atomic` helper currently uses a fixed
temporary filename. No experiments were rerun during consolidation. Resuming
only missing task 0 from its original worktree would recover the 50 absent
repetitions before collection;
concurrent initialization needs a separate fix before another fresh parallel
launch. Existing source-contract checks must be respected when resuming.

Slurm reported no active jobs at inspection time. After merging, redundant
branch pointers can be removed while keeping their worktrees detached at the
saved commits: provenance contains absolute paths into those directories.
Removing the directories belongs with the later path/provenance cleanup.

## Generali preservation and the next cleanup stage

Keep the canonical `src/data/dataset.csv`, `src/data/models/`, real-data presets,
Generali objectives, shared reporting and exact-spline caches. Keep the seven
policy-capacity/optimizer-behavior manifests, GLM and noisy-GLM launchers,
sensitivity and historical-policy evaluators, cache-building tools, and existing
Generali scratch drivers. Some older drivers were already in
`scratch/retired_real_data_analysis/` before this task; none are newly retired.

Some real-data outputs were also already archived, including
`real_data_glm_base__20260706_124627/`, `noisy-glm-homoskedastic-theta-distance-sweep/`,
`real-data-model-eda/`, `real-data-model-hierarchy-200/`, and
`xgb-logit-spline-experiment/`. Retain them. In particular, the noisy-GLM launcher
documents the first of these as its reference run under `results/`; resolve that
reference against the archive before using it downstream.

For the next stage, review older synthetic bootstrap/GP/finite-policy studies
against the new proof, then move selected outputs to `scratch-results/` and
concluded experiment drivers to `scratch/`. Keep an explicit old-to-new path map
and update readers, documentation and provenance access together. Do not infer
that an experiment is obsolete merely because its driver already lives in
`scratch/`, or because its results have an older date.

## Validation

- Full combined-tree suite: 1,256 passed, four failures. All seven math-document
  checks pass after the formatting fixes and the development dependency was
  supplied. The only remaining failure is
  `tests/scripts/test_analyze_zeroth_order_proof_validation.py::test_completed_manifest_outputs_produce_all_tables_and_plots`:
  Matplotlib encounters a NaN layout while exporting `sigma_landmarks.png`.
- The plotting failure was reproduced independently on unchanged initial main
  (`7f24a0b`) with the same environment. Its source, manifests and test are
  unchanged by this merge. No experiment or plotting code was changed to hide
  this existing failure.
- A small CPU demo through `main.main(['--launch', 'local'])` passed using
  `fixed_regression_base`, 32 samples, three steps, seed 7, and first-order,
  finite-difference and Stein-difference estimators. Its output is in the
  temporary validation directory, not the shared result tree. The default
  full-data GPU demo was not launched.
- Pdoc regenerated successfully; changed Python/JSON files parse; `git diff
  --check` passes. The existing Generali implementation, presets, data, models,
  entry point and policy-capacity manifests remain unchanged.

Validation used `simulation_env`, checkout-local `PYTHONPATH`, one BLAS/OpenMP
thread, CPU JAX and the Agg backend. Detailed logs and the original local patch
were saved in `/tmp/generali-cleanup-20261022/` for this session.
