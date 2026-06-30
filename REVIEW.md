# Pull Request Review Instructions

Use these instructions when asking Claude or another coding agent to review a
pull request in this repo.

## Required Lens

- Invoke `/improve-codebase-architecture` as the architecture review lens.
- Use `/codebase-design` vocabulary exactly: module, interface, depth, seam,
  adapter, leverage, and locality.
- Keep the review finding-first: list bugs, regressions, architectural friction,
  and missing tests before any summary.
- If present, read `CONTEXT.md` and relevant ADRs under `docs/adr/` before
  judging domain seams.

## Workflow Fit

Check that changes land in the right logical workflow:

- Data: loaders, dataset metadata, artifacts, preprocessing, schema/path updates.
- Experiment configuration: config factories, presets, overrides, seeding,
  train/test split, and sweeps.
- Objective: objective math, policy/objective bridge, gradients, and `MATH.md`
  parity.
- Optimization: policy optimization, gradient estimators, solvers, step rules,
  and RNG streams.
- Reporting: results, reporters, plots, artifacts, and summaries.

## Architecture Checks

- Prefer the lowest-code solution that preserves behavior.
- Apply the deletion test to new modules and helpers.
- Flag shallow modules whose interface is nearly as complex as their
  implementation.
- Suggest removing unnecessary abstractions, pass-through helpers, compatibility
  layers, and dead code.
- Suggest deduplicating logic when one deeper module would improve locality and
  leverage.
- Avoid adding a seam unless variation is concrete; one adapter is usually only
  a hypothetical seam.

## Docs And Interfaces

- Keep `README.md` concise: quick-start, core workflow, and user-facing behavior.
- Keep `MATH.md` synchronized with objective, gradient, and estimator formulas.
- Keep generated `docs/` and public docstrings concise when public interfaces
  change.
- Expose only interfaces that callers need; keep implementation details private.

## Output Format

- Findings first, ordered by severity, with file/line references.
- For each finding, state the issue, impact, and recommended fix.
- Call out test gaps for each changed workflow.
- If there are no findings, say so and list residual risks.
