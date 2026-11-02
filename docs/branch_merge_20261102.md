# Branch audit — 2026-11-02

## Integration update

Merge [PR #193](https://github.com/anakhag07/generali-pricing-simulation/pull/193)
(`pr/generali-reporting-stack`) first, then
[PR #194](https://github.com/anakhag07/generali-pricing-simulation/pull/194)
(`pr/generali-action-powers`). Design-Bench remains deferred at the user's request.
The reporting PR includes the saved slide refinements without changing the
original dirty source worktree. The action-power PR ports only the two extension
and replay commits while preserving main's solver interruption handling.

The user selected "keep both" for all three documentation conflicts: slideshow
instructions precede MLP diagnostic instructions; checkpoint/resume and
action-power study instructions both remain; frozen-feature ellipsoids stay in
MATH section 7.20 and action powers follow in section 7.21 with updated references.
Original source worktrees and artifact contracts remain unchanged. The initial
audit and proposed sequence below are retained as the historical snapshot.

## Initial audit

Remote heads were checked with `git ls-remote --heads origin`: `main` is
`43c3c35`, and the only other remote branch is `feature/design-bench-bridge`
at `2b7965f`. The Generali feature branches are local. This is a branch/ref
audit and merge simulation, not a code review or a live pull-request status check.

| Branch | State relative to main | Recommended treatment |
| --- | --- | --- |
| `feature/generali-wild-elasticity-slices` | 3 commits ahead, 0 behind; clean | Ancestor of slides; integrate as part of that stack. |
| `feature/generali-slide-figures` | 4 ahead, 0 behind; 6 modified and 2 untracked files | Commit reviewed pending layout/shading/docs changes before integration. |
| `feature/generali-action-powers` | 16 ahead, 46 behind; clean | 13 commits are patch-equivalent to main. Port only `df19ad5` and `56b40f4`. |
| `feature/design-bench-bridge` | 10 ahead, 299 behind; clean and matches remote | Separate integration branch/PR after updating against main; docs conflicts. |
| Both `feature/generali-residual-bases*` branches | 13 ahead, 46 behind; dirty source worktrees | Already represented by merged PRs #188/#189 and fix #191. Keep for provenance; do not merge old branch tips. |

The new `feature/generali-mlp-action-diagnostics` branch starts at slide commit
`ac635a9`, so its ancestry includes all committed slide and elasticity changes.
It does not include the slide worktree's pending edits. No existing branch was
merged, rebased, reset or deleted during this audit.

## Suggested sequence

1. In `worktrees/feature-generali-slide-figures`, review and commit the six tracked
   edits (`AGENTS.md`, `MATH.md`, `README.md`, `docs/experiment_workflow.md`,
   `scratch/generali_slide_report.py`, `scratch/generali_slideshow.json`) and the
   two new relocation indexes (`generali_slideshow_layout_20261030.json` and
   `generali_slideshow_policy_shading_20261030.json` in `scratch/retired_experiments/`).
   Review before staging; these are preserved pre-existing edits.
2. In the canonical checkout on `feature/generali-mlp-action-diagnostics`, merge
   that newly committed `feature/generali-slide-figures` tip. Resolve any additive
   reporting/docs overlap and rerun focused tests. Submit the combined stack
   against main. A separate elasticity merge is unnecessary because it is an
   ancestor. At the audited tips, main → slides is a clean fast-forward.
3. From the updated main, create an action-power integration worktree and apply
   just the actual extension and replay fix:

   ```bash
   git worktree add ../worktrees/integrate-generali-action-powers \
     -b integrate/generali-action-powers main
   cd ../worktrees/integrate-generali-action-powers
   git cherry-pick df19ad5 56b40f4
   ```

   Skip `a3a095c`: it reintroduces the inherited residual-basis implementation
   already integrated and subsequently fixed on main. A three-way trial of
   `df19ad5` against the audited main conflicted only in `README.md` and `MATH.md`;
   source and tests merged automatically. Retain both documentation additions,
   renumber the action-power math section after the existing frozen-feature
   section 7.20, and fix its stale internal section references. Keep main's
   `_solve` interruption handling: save an interrupted diagnostic and raise
   before finalizing an unfinished network or policy. The replay fix `56b40f4`
   preserves exact CSV action coordinates; retain it. Test the resulting tree;
   later changes to main can alter the conflicts.

   A wholesale merge has **10 conflicted files**, including add/add conflicts
   in the residual adapter, reusable math and tests. Taking entire feature-side
   files would revert main's checkpoint fixes. Regenerate pdoc from the resolved
   source if public docs change; do not choose a stale generated-doc side.
4. Integrate Design-Bench separately on a branch from updated main, merging
   `feature/design-bench-bridge`. At the audited tips, only `AGENTS.md` and
   `README.md` conflicted; `pyproject.toml` merged automatically. Keep both sets
   of documentation and current dependencies. Run its focused tests and its
   separate pinned-environment smoke check before merging the PR.

Suggested Generali checks in `simulation_env`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest -q \
  tests/experiments/test_feature_wild.py \
  tests/scripts/test_generali_profit_residuals.py \
  tests/scripts/test_generali_wild_policy.py \
  tests/scripts/test_generali_wild_slices.py
```

The two dirty residual-basis worktrees and the detached action-power source
worktree at `df19ad5` are required by saved absolute source contracts. Preserve
them and their current contents even after integration. Do not use a blanket
stash, clean, branch deletion or worktree cleanup. See the existing 2026-10-28
worktree recovery index for provenance constraints.
