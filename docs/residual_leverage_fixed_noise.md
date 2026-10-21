# Residual leverage sweep with the moved observation's scale fixed

## Authorized setup and implementation plan

Repeat the six moved-action conditions 3, 10, 20, 40, 100, 1000 from
`residual-heteroskedastic-leverage`, holding only the moved observation's noise
SD fixed. Use its original pre-move value, 0.6991976375961079, recorded at action
0.6002754512595134 in row 0 of the saved
`results/residual-heteroskedastic-paired-nb/environment.npz`. All other inputs,
truth/predictor coefficients, GP coefficients, seeds, n=300, B=1999, delta=0.05,
and 2000 repetitions remain the same. New result names preserve prior studies.

- [x] Add a validated scalar `noise.moved_observation_sigma` option to the
  existing runner, applying it only to `design.action_override.index`.
- [x] Add six explicit manifests with the frozen scale and original seeds.
- [x] Test the single-row change, unchanged bulk/scales/geometry, response
  pairing across locations, fresh repetitions, and rejection above the known cap.
- [x] Run through the existing manifest entry point on six one-CPU Slurm jobs.
- [x] Replay the same four PDFs, validate all artifacts and inspect the PDFs.

No bootstrap, optimizer, figure-layout, or seed-system change is needed.
The option is absent by default, preserving legacy behavior. The numerical
source contract changes, so new runs use new result directories. Update the
study guide, MATH section 7.9, README, AGENTS, and generated public API docs.

## Representation and confidence domain

For moved row j=0 and prescribed action A, draw
Y_j(A)=f(x_j,A)+s_j Z_j with s_j=0.6991976375961079. Other rows retain their saved
GP-based SDs. Each repetition draws fresh independent standard normals Z_i;
the same innovation vector and bootstrap multipliers are paired across all six
conditions and with the earlier changing-scale experiment. No new random process
or seed stream is introduced. Roots 1101/1102/1105 remain contexts/actions/profile,
and 1103/1104 remain observations/multipliers.

The fixed row-level scale overrides observation generation only. It does not
redefine the continuous finite-Fourier GP or its covariance/off-grid evaluation.
Any diagnostic GP grid still depicts the background profile; the saved `sigma`
array records the actual observation SDs. The overridden SD is validated to be
positive and at most the existing global sigma_max=1, retaining the analytical
variance-cap bound's assumptions. Errors remain independent conditional on the
fixed design/profile and have mean zero.

The target remains the conditional mean residual g=f-fhat at the 300 fixed
contexts and every real action. All-real affine coefficient statistics and the
existing boundary/numerical handling are unchanged; grids/interpolated display
lines do not certify containment. Bootstrap coverage remains empirical, not a
new finite-sample guarantee. There are no optimizer queries, action constraints,
reported optima, global optimization references, or new certification tolerances.

Coverage uses all 2000 response datasets per condition. Band/radius pictures
use repetition 0, transferring per-context endpoints before averaging them.
The mean-band/radius display window remains [-2,2]; the distribution picture
retains its common symlog action axis. Moved-row leverage and maximum leverage
are distinguished exactly as before. Holding observation scales fixed does not
impose monotonicity on empirical coverage.

## Run and replay

Run every `manifests/residual-leverage-fixed-noise-outlier-*.json` with
`scripts/run_experiment_manifest.py <manifest> --launch slurm --no-array`.
Use one CPU per condition; the short checkpointed jobs can use mit_preemptable.

```bash
PYTHONPATH=src python scratch/plot_wild_bootstrap_leverage.py \
  --summaries /home/anakhag/projects/generali-pricing/results/residual-leverage-fixed-noise-outlier-*/summary.json \
  --destination /home/anakhag/projects/generali-pricing/results/residual-heteroskedastic-leverage-fixed-noise \
  --diagnostics
```

The output includes coverage, labeled distributions, averaged bands, and
separate residual half-width/transferred radii as vector PDFs, plus CSV data,
source/artifact provenance, and this experiment document.

Reference environment SHA-256: `d4bc0cd734468ae5d3279e26fbd8a399151837348d146a68f9946376abe9e250`.

## Completed results

All six conditions completed 2000 repetitions (12000 total). The moved row
has SD 0.6991976375961079 everywhere; all other rows retain their original
heteroskedastic scales. Coverage below is joint across all calibration contexts
and all real actions.

| Moved action | Maximum leverage | Residual coverage | Pointwise 95% Wilson interval | Transferred coverage |
|---:|---:|---:|---:|---:|
| 3 | 0.056188 | 93.05% | [91.85%, 94.08%] | 95.65% |
| 10 | 0.257419 | 92.40% | [91.16%, 93.48%] | 94.75% |
| 20 | 0.577627 | 89.50% | [88.08%, 90.77%] | 92.70% |
| 40 | 0.845438 | 85.15% | [83.52%, 86.64%] | 89.75% |
| 100 | 0.971640 | 81.75% | [80.00%, 83.38%] | 87.75% |
| 1000 | 0.999709 | 81.35% | [79.58%, 83.00%] | 87.45% |

Residual coverage decreases along this realized sweep, from 93.05% to 81.35%.
The prior rebound at action 40 disappears. This empirical sequence does not
establish a universal monotonicity theorem. The analytical variance-cap bands
covered all 2000 datasets in every condition. The four PDF layouts and display
windows match the earlier study; band/radius illustrations still use repetition 0.

Validation: 116 focused tests passed; a two-condition manifest-runner and
reporting smoke test passed; all four regenerated PDFs opened as vector PDFs
and were visually inspected in memory, with no raster artifacts saved.
`execution.json` records the six one-CPU Slurm jobs and manifest hashes;
`repetition_validation.json` records the full saved-dataset audit.
