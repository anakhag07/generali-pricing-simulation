# Heteroskedastic residual bands with one moved observation

## Confirmed experiment and implementation plan

The user selected the moved-point sweep, the full three-context residual model,
n=300, B=1999, delta=0.05, 2000 repetitions, and response repetition 0 for
band/radius illustrations. Reuse the residual-paired-nb truth, predictor,
standard-normal design, Fourier noise profile, raw Rademacher bootstrap,
empirical order statistic, and five seed roots. Override only action row 0 with
3, 10, 20, 40, 100, or 1000. The remaining 299 actions and all contexts stay
fixed. These use the first 300 rows of the original residual environment, not
the old action-only experiment's truncated-normal bulk.

- [x] Add six explicit manifests using the existing experiment runner.
- [x] Extend the existing leverage report to accept normal moved-point designs.
- [x] Replay saved environments and repetition-0 bands into distribution,
  context-averaged band, and separate lower/upper radius PDFs with CSV provenance.
- [x] Check frozen bulk/profile and paired innovations/multipliers, contextual
  leverage, and transfer-before-average semantics; run focused regression tests.
- [x] Run all conditions, regenerate reports, validate PDFs and document results.

No numerical model or bootstrap algorithm changes are needed. Reporting stays
in `src/experiments/envelopes/wild_reporting.py`; the existing replay launcher
accepts explicit summary paths. Historical artifacts remain in their own result
directories. Public reporting documentation is regenerated with pdoc.

## Representation, seeds, and confidence domain

The existing contextual affine functions are evaluated exactly for any real
action, at each of the 300 frozen calibration contexts. The finite-Fourier
noise-scale process, its covariance, and its exact off-grid evaluation are those
in `wild_bootstrap_envelopes.md` and MATH section 7.9. Conditional observation
errors are independent normals; the saved profile creates varying scales, not
correlated response errors. Recompute the moved row's mean and noise scale at
its new action. Consequently this is a design perturbation under the same noise
function, not a pure leverage perturbation holding the moved row's variance fixed.

Roots 1101/1102 draw contexts/actions once, root 1105 draws the shared profile,
and roots 1103/1104 derive observation/bootstrap repetition streams. Every
leverage condition reuses the same innovations and multiplier matrix for each
repetition. Moving a prescribed row introduces no random process or new seed
stream. The old paired-n/B experiment has a different multiplier matrix shape;
matching seed roots does not claim identical bootstrap matrices across studies.

Coverage is the joint event over every calibration context and the entire real
action line. Affine coefficient statistics include both tails; diagnostic
grids and connecting lines are visualization only. The original floating-point
comparison and exact-rational boundary fallback limitations remain unchanged.
The analytical variance-cap bound has conditional finite-sample coverage at
least 95%; wild-bootstrap coverage is empirical, without a new finite-sample
guarantee. Wilson intervals describe Monte Carlo uncertainty across 2000
responses. There are no optimizer queries, action constraints, reported optima,
or global optimization references/certification tolerances in this experiment.

## Figures and replay

`coverage_vs_leverage.pdf` preserves the earlier marker/error-bar format, using
joint residual-band coverage and actual maximum observed leverage. The moved
row's leverage is computed from the full contextual design, not an action-only
formula, and can differ from maximum leverage at the smallest moved action.

`data_distribution.pdf` displays the same 299 bulk actions, their N(0,1) parent
density, n=300, and the moved observation with its measured leverage above the
dot. A symlog action axis has a linear central region |a|<=4 so the normal bulk
and all prescribed tail locations are visible together.

`context_averaged_bands.pdf` shows residual bands and transferred function
envelopes at each condition, averaged over the same 300 contexts using response
repetition 0. `context_averaged_radii.pdf` separately shows mean residual-band
half-width w and the actual transferred lower/upper radii. Transfer each
context's band before averaging: r_lower=[w-g_tilde]+ and
r_upper=[w+g_tilde]+ (MATH section 7.9). Display actions remain [-2,2], matching
the earlier context-averaged report; this is not the confidence domain.

Run each `manifests/residual-leverage-outlier-*.json` with
`scripts/run_experiment_manifest.py --launch slurm --no-array` (put the manifest
path before the flags), then replay:

```bash
PYTHONPATH=src python scratch/plot_wild_bootstrap_leverage.py \
  --summaries /home/anakhag/projects/generali-pricing/results/residual-leverage-outlier-*/summary.json \
  --destination /home/anakhag/projects/generali-pricing/results/residual-heteroskedastic-leverage \
  --diagnostics
```

All illustrations are saved as vector PDFs with plotted CSV data and hashes of
their source artifacts. The reporting path validates the summary's environment,
saved example and repetition-0 provenance, and unchanged bulk across conditions.

## Execution notes

The 106 focused mathematical, study-runner, and reporting tests passed. A small
end-to-end runner/replay smoke study also passed. Regeneration uses the full
pdoc command; unrelated absolute-path/whitespace churn in unchanged modules is
discarded. PDF inspection renders pages only in memory; no raster artifacts are
saved.

The original residual environment was checked byte-for-byte at the array level:
all 300 contexts, all 299 unmoved actions/noise scales, and the Fourier profile
coefficients agree exactly. The first local timing pilot is preserved separately
under `results/residual-leverage-local-pilot-3/`: compute-node path aliases differ
from the local sandbox, and paths participate in the existing source contract.
Production repetitions are prepared and collected on compute nodes. Initial
short array batches were replaced by one serial job per condition to avoid
scheduler overhead, retaining successfully completed checkpoints. This changes
neither manifests nor random draws. No optimizer exception was required.

## Completed results

All six conditions completed 2000 repetitions (12000 total). The following
coverage is the joint residual-band event over all contexts and real actions.

| Moved action | Maximum leverage | Wild coverage | Pointwise 95% Wilson interval | Transferred coverage |
|---:|---:|---:|---:|---:|
| 3 | 0.056188 | 93.15% | [91.96%, 94.18%] | 95.50% |
| 10 | 0.257419 | 91.95% | [90.68%, 93.06%] | 94.35% |
| 20 | 0.577627 | 91.95% | [90.68%, 93.06%] | 94.30% |
| 40 | 0.845438 | 93.75% | [92.60%, 94.73%] | 94.80% |
| 100 | 0.971640 | 83.65% | [81.96%, 85.21%] | 88.80% |
| 1000 | 0.999709 | 71.85% | [69.84%, 73.78%] | 82.65% |

The analytical variance-cap band covered in all 2000 repetitions at every
condition; this observed frequency is distinct from its theoretical lower bound.
Wild coverage is below nominal and drops substantially at the two most extreme
positions. The intermediate behavior is not monotonic. In particular, the
moved point has noise SD about 0.184 at action 40, 0.663 at action 100, and
0.903 at action 1000, so leverage alone is not a controlled explanation of
every difference. Displayed context means are repetition-0 illustrations, not
coverage averages over repetitions or evidence of joint containment.

Production jobs 26513568, 26513576, 26513579, 26513585, 26513592, and 26513595
completed after permitting the checkpointed workload to use `mit_preemptable`
as well as `mit_normal`, with one CPU and a ten-minute wall limit per condition.
All four final vector PDFs were regenerated from the committed reporting source,
opened with a PDF parser, and visually inspected in memory for clipping,
overlap, readability, and labels. No raster copies were saved.
