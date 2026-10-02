# Cartesian N/B coverage extension

User request: fill the 9 by 9 Cartesian product, retain 2000 outer datasets,
reuse completed work, and regenerate the coverage plot. Keep sigma=1, delta=.05,
original pairs resampling, fixed-denominator standard error, higher quantiles,
and analytic whole-real-line containment. No objectives or ranking are added.

- Use the same independent seed roots and common 5000 by 5000 bootstrap uniform
  array per dataset. Reuse the completed diagonal experiment and compatible
  sigma=1 draws from the 100-dataset controlled experiment.
- Check input hashes, data/seed/model/domain/tolerance contracts and uniform
  dimensions before reuse. Verify overlap of saved bootstrap sequences.
- Reuse each original fitted model. For each N, retain the longest available
  bootstrap prefix; refit and certify only its missing suffix through B=5000.
  Recompute quantiles for all B prefixes, never repeating saved bootstrap fits.
- Checkpoint each N separately, and preserve the old result directories. The
  new directory records hashes/provenance for all reused inputs and new arrays.
- The shared manifest launcher submits CPU array tasks and a dependent collector.
  This implementation has no GPU kernels; the expensive certificates run on CPU.
- Plot a discrete N/B heatmap with viridis and empirical coverage percentages.
  Include pointwise Wilson bounds in CSV and a PDF of interval half-widths.
  No interpolation or monotonicity constraints. Highlight no preferred cells.
- Validate prefix equality, strict reuse without refits, exact missing-refit
  counts, unchanged diagonal coverage, corruption detection and full collection.

Validation before launch: 22 focused tests passed (Cartesian reuse/collection,
existing joint coverage, shared manifest launcher). A two-dataset preflight
reused actual controlled/diagonal artifacts at N=20 and N=5000, extended only
the missing N=20 bootstrap suffixes, and generated valid, visually inspected
coverage and uncertainty PDFs. The production manifest uses 200 CPU array
tasks (10 outer datasets each), with at most 64 concurrent one-CPU tasks.
