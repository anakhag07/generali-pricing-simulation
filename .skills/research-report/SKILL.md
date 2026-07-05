---
name: research-report
description: Analyze a finished run or sweep from its summary.json/CSV outputs, suggest pareto-frontier and sweep-axis diagnostic plots, and (when prompted) write a short next-steps report to results/agent-reports/.
---

# Research Report

Turn a finished experiment run or sweep into a **short, concise analysis with
suggested next steps**, driven by the data already on disk. Never rerun or
re-launch experiments just to summarize them; only suggest a rerun when a needed
quantity is genuinely absent from the existing outputs (see the
data-availability reference below).

## Two outputs

1. **Analysis** (default): what moved, how estimators compare, what the
   tradeoffs look like, and what to try next. Deliver inline; write it to a file
   only when prompted (see report format).
2. **Plot-generation plan** (only when asked for plots, and in plan mode): for
   each suggested plot, name the plot type, the exact data source
   (summary.json key or CSV column), and the existing
   `src/reporting/visualization.py` helper that draws it. Do not render plots
   as part of this skill — produce the plan.

## Workflow

1. **Locate outputs.** Resolve the results base via
   `experiments.paths.results_root()` (honors `GENERALI_RESULTS_ROOT`, defaults
   to `~/projects/generali-pricing/results`) — never `cwd` or `__file__`. Older
   archived outputs may live under the canonical checkout's `outputs/` dir.
   Sweep layout: `<project>/<variant>/summary-seed-<seed>.json` at the variant
   root (heavy artifacts under `seeds/seed-<seed>/`; older sweeps instead have
   `seeds/seed-<seed>/summary.json`), plus `seed_grid_finals.csv` and
   `seed_grid_summary.csv` at the variant and project levels. Single runs:
   `<slug>__<timestamp>/summary.json`.
2. **Read summaries first.** Pull everything possible from `summary.json` and
   the aggregate CSVs before considering anything else.
3. **Analyze.** Compare estimators on final objective, mean acceptance, and
   `u`; read the sweep axis (which parameter varied and how metrics respond);
   check seed spread (`_std` columns) for whether differences are real; flag
   anomalies — `optimizer_success=false`, bad `optimizer_status`/`message`,
   `constraint_violation > 0`, train/test gaps.
4. **Report (opt-in).** If prompted, write
   `results/agent-reports/<YYYY-MM-DD>-<short-summary-of-topic>.md` (create the
   dir if missing). **Gate:** only write a report when enough results exist to
   support conclusions; otherwise summarize inline and say why a report would be
   premature.

## Suggesting plots

Default to two families and map every suggestion to a real helper:

- **Pareto frontier** — mean acceptance vs. objective (or `u`), colored by the
  sweep axis: `plot_sweep_pareto_frontier(points, ..., sweep_key, y_key, ...)`;
  seed-aggregated version with error bars:
  `plot_seed_grid_frontier(summary_rows, ...)`.
- **Sweep-axis diagnostics** — metric vs. the swept parameter:
  `plot_sweep_tradeoffs(points, ..., sweep_key, ...)` for config-axis sweeps;
  `plot_seed_grid_metric_bars(summary_rows, ..., metric, ...)` for grouped bars
  with cross-seed error bars; `plot_seed_loss_bands(...)` for objective-vs-step
  mean ± std bands (needs per-step traces — see below).

Inputs come from `experiments.sweep_reporting`:
`collect_seed_grid_final_rows` / `aggregate_seed_grid_rows` (seed grids),
`collect_config_sweep_final_rows` (config-axis sweeps),
`objective_traces_by_estimator` (loss bands, in-memory traces only).

## Data availability (what's already on disk vs. needs a rerun)

**In `summary.json` — never suggest a rerun for these.** Per estimator:
`final_u`, `final_value`, `final_objective_sum`, `runtime_sec`, `theta`,
`theta_l2_norm`, `theta_delta_l2_norm`; optional `mean_acceptance`,
`constraint_violation`, `acceptance_multiplier`; `train`/`test` policy blocks
(`objective_value`, `objective_sum`, `mean_u`, `u_q25`, `u_q75`,
`mean_acceptance`, `projected_loss`, `projected_revenue`); optimizer outcome
(`optimizer_success`/`status`/`message`). Top level: `trace_summary`
(`steps`, `final_objective`, `min_objective` per estimator), `initial_value`,
`u_star`/`value_at_u_star`, `split` counts, and the full `config` block (seeds,
step rule, estimators, objective/policy spec). Present only sometimes — treat as
optional: `preset` (name + overrides), `model_coefficients`,
`constant_u_baselines`/`best_constant_u_baseline`, `policy_artifacts`.

**In sweep CSVs:** `seed_grid_finals.csv` (one row per
variant × seed × estimator: finals, train/test metrics) and
`seed_grid_summary.csv` (per variant × estimator:
`<metric>_{mean,std,min,max}` over seeds).

**Requires a rerun (or a saved policy artifact):** full per-step
objective/gradient/theta trajectories beyond the 3-number `trace_summary`
(per-step CSVs exist only if the run wrote `plots/optimization/steps.csv`);
per-sample `x`/`u`/acceptance distributions (check `policy_artifacts` first —
`load_policy_artifact(...).predict_u(...)` / `.evaluate(...)` replays a saved
policy without optimizer training); anything needing the raw
`ExperimentResult`. Suggest the narrowest option that fills the gap.

## Report format

Keep it under ~1 page:

- **Title + date**, one-line takeaway.
- **Pointer** to the run/sweep dir(s) analyzed.
- **Key results** — small table: estimator (or variant) → final objective /
  mean acceptance / mean `u` (± std when seed-aggregated).
- **Read** — 2–4 sentences on tradeoffs and anomalies.
- **Suggested plots** — each with data source + helper (only if asked).
- **Next steps** — concrete follow-up runs or analyses; note explicitly when a
  suggestion requires new data vs. existing summaries.
