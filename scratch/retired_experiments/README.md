# Retired experiments

This archive preserves concluded studies for recovery. The first pass on
2026-10-22 moved 18 manifests and nine drivers out of the active catalogs.
The matching shared results were moved to
`/home/anakhag/projects/generali-pricing/scratch-results/`, retaining each
project name. Results are outside Git and their moves apply across all worktrees,
including checkouts whose code predates this branch.

| Family | Manifests | Drivers | Results newly archived |
|---|---:|---:|---:|
| `zeroth_order/` — old proof/bias/support-envelope studies | 3 | 3 | 0; already archived |
| `policy_lcb/` — finite, continuous, variable-envelope and GP studies | 5 | 0 | 7, including interim reports |
| `bootstrap_ols/` — grid/continuous, controlled, dense-N, joint/Cartesian and offline pairs bootstrap | 7 | 0 | 12, including preflights and earlier parametric outputs |
| `coverage_kernel/` — N, q and bandwidth sweeps | 3 | 0 | 3 |
| `noise_and_bias/` — fixed-regression/planted-logistic noise-offset and bias grids | 0 | 6 | 0; already archived |

[relocations.json](relocations.json) records every old/new repository and result
path, original source hashes, and 23 relevant result directories already in the
archive. Original manifests are byte-for-byte preserved; driver edits only
repair imports and checkout-root discovery. The 22 result directories contain
50,875 files (12,472,870,179 bytes). Same-filesystem renames preserved every
entry's inode, relative path, mode, size, mtime, and symlink target; no saved
artifact contents or provenance records were rewritten.

The initial pass left Generali and current wild-envelope workflows active;
the follow-up selection below supersedes its result-retention choices.
Shared `src/` helpers are still available because current consumers use their
mathematics. Existing tests retain coverage and import the archived drivers or
load the archived manifests explicitly. Production code does not import scratch.

## Follow-up selection

At the user's request, the next pass moved another 29 synthetic manifests and
76 result/output directories. See
[relocations_followup_20261022.json](relocations_followup_20261022.json) for every
old/new path and original manifest hash. This supplements the first relocation
index without rewriting it.

| Archived family | Manifests moved | Result/output directories moved |
|---|---:|---:|
| Generali completed optimizations and associated analyses | 0; seven stay active | 42: 40 from shared results plus two frontier outputs from the canonical checkout |
| Generic uncertainty-envelope studies | 7 | 11, including the older generic wild-bootstrap result |
| Earlier wild-bootstrap and residual-leverage diagnostics | 22 | 23 |

The 76 moved directories contain 109,967 files (3,249,127,905 bytes). Atomic
renames preserved every file's identity and metadata; all retained result trees
have unchanged entry fingerprints. No saved contracts were rewritten. Shared
results now have 16 directories and scratch-results has 143. Cache/log directories
count toward the 16; the two top-level distribution launch metadata files remain.

Keep the terminal-distribution sweep, all four `residual-heteroskedastic*` report
folders, six fixed-noise residual-leverage runs, and three data-location
illustration folders. The active manifest catalog now has 16 entries: seven
Generali, three paired-residual distribution variants, and six fixed-noise
residual-leverage settings. The fixed-noise runs were explicitly retained with
their residual-heteroskedastic report.

Generali datasets, trained models, shared caches, source modules, and reusable
configurations were not retired. Completed optimization outputs and associated
policy/model diagnostics were moved together. Before reusing a saved Generali
policy, report, or reference run, resolve its recorded paths through the two
relocation indexes and honor the source contract. Older absolute defaults in
analysis drivers are historical locations, not archive-aware lookup rules.
No extra drivers or `src/` modules moved in this pass.

Follow-up validation: all 123 tests in the four affected envelope-study/reporting
files passed. All 29 relocated manifests and all 16 retained manifests load
through the shared launcher without executing studies. Original manifest bytes,
retained result-tree fingerprints, source/driver stability, documentation links,
and `git diff --check` were verified. The first-pass full-suite result below
remains recorded separately; no runtime code changed in this follow-up. Logs and
identity inventories are in `/tmp/generali-retire-followup-validation-20261022/`.

The [next source-refactor plan](../../docs/experiment_workflow.md) extends the
existing settings, band construction, optimizer and reporting interfaces. It
keeps the current experiment scaffolding and describes parity checks and later
movement of study-only orchestration into scratch.

## Completed worktrees

On 2026-10-23, 14 clean, completed worktrees were removed at the user's request.
Every source commit remains reachable from `main`; their only extra files were
discardable Python/pytest caches and links to canonical data. No local processes
or Slurm jobs used these worktrees, and no retained symlinks or simulation
environment installation references pointed into them. The canonical checkout,
Design-Bench worktree and active cleanup worktree remain.

[worktrees_20261023.json](worktrees_20261023.json) records every original path,
exact commit/tree, and data symlink. When an artifact requires an old absolute
source path, recreate that checkout from the canonical repository:

```bash
git worktree add --detach ../worktrees/<recorded-name> <recorded-commit>
```

Restore that record's data symlinks if present. Source commits are preserved in
Git; the worktree cleanup does not move or delete experiment results, canonical
datasets or trained models. Checkouts can be removed again after replay.

## Reading and recovery

Read summaries, tables, PDFs, and arrays directly at the new paths. Resolve an
old path by matching its project prefix in either relocation index and retaining the
suffix. Original manifests/results can describe older configurations than the
latest archived recipe; trust each saved run's contract, not similar names.
For example, the historical `bootstrap-ols-grid-band` output is not a replacement
for the current manifest's `bootstrap-ols-pairs-grid-band` project, which is not
present in the audited shared results.

Saved contracts contain absolute paths and source hashes. Changing the root
argument alone does not repair these contracts. For exact replay, recover the
recorded source revision/worktree and temporarily restore the required result
directory to its recorded location using the relocation index. Check that the
destination is absent before moving anything back; restore all referenced
dependencies together, then return them to the archive after replay. Recreate a
removed source checkout from the worktree index above when required. Do not force
stale contracts, rewrite saved provenance, or invent missing source artifacts.

The offline-bootstrap manifest specifically references datasets 000 and 001 in
`results/bootstrap-ols-pairs-controlled-sweep/`; their bytes and checksums now
live beneath `scratch-results/`. Its saved snapshots also record absolute paths.
Cartesian bootstrap reuses the joint and controlled studies; the kernel q/b
sweeps reuse the N sweep. Recover those dependencies as a group if needed.
The existing `scratch/plot_bootstrap_coverage_frontier.py` also reads the
controlled sweep and verifies its original absolute source paths; use the same
restoration procedure before that replay.

For a new exploratory run based on an archived recipe, copy its configuration,
resolve input paths against the archive, and choose a separate output location
inside that experiment family. Use the shared manifest runner with an explicit
`--runs-root`; legacy drivers honor `GENERALI_RESULTS_ROOT`, and analyzers also
accept `--runs-root`. Do not launch a fresh run into preserved results. To inspect
driver options without starting a run, from the checkout root:

```bash
conda activate simulation_env
python scratch/retired_experiments/noise_and_bias/run_noise_offset_grid.py --help
python scratch/retired_experiments/zeroth_order/analyze_zeroth_order_envelopes.py --help
```

The legacy proof-validation plotting test already failed on `main` before this
move (Matplotlib encounters a NaN layout in `sigma_landmarks.png`). Archival does
not change that plotting implementation. Historical descriptions are retained
in [LEGACY_DESCRIPTIONS.md](LEGACY_DESCRIPTIONS.md); the original files remain
recoverable from base commit `b7faa26` as well.

## Validation of this archival pass

- Full suite in `simulation_env`: 1,259 passed, one existing failure in
  `test_completed_manifest_outputs_produce_all_tables_and_plots` (the legacy
  NaN-layout failure described above). No new tests or test exclusions were added.
- All 18 archived manifests load and dispatch through the shared launcher
  without running studies; all nine relocated Python modules import, and each
  executable driver's `--help` works from its new path.
- A local `main.main` demo passed with `fixed_regression_base`, 32 samples,
  three steps, seed 7, and first-order, finite-difference, and Stein estimators.
  Its artifacts are temporary; the full-data GPU demo was not launched.
- Every archived manifest matches its original source hash. The two saved
  offline-bootstrap inputs match their pinned SHA256 hashes after relocation.
  All 45 active manifests and all tracked `src/` files are unchanged.
- Source checks found no production imports of scratch; documentation links
  resolve and `git diff --check` passes. No public API/docstrings changed, so
  generated API documentation did not need regeneration.

After the first pass the shared roots contained 90 active and 67 archived result
directories; the follow-up counts above supersede those totals.
Detailed validation logs and before/after file-identity inventories for this
session are in `/tmp/generali-retire-validation-20261022/`.
