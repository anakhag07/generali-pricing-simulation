# Retired real-data analysis drivers

These files preserve the exact source used for concluded 2026 real-data plots
and policy diagnostics. They are retained for provenance; they are not supported
production entry points and are intentionally omitted from the repository's
script catalog.

New or repeated analyses must use the existing generic experiment-manifest
workflow in `scripts/run_experiment_manifest.py`. Reusable model evaluation,
coverage, response-cache, policy, provenance, and plotting behavior now lives in
`src/`. In particular, the `real_data_profit_dispersion` reporting recipe accepts
model aliases such as `glm`, `xgb`, `monotone_spline_xgb`, and
`exact_spline_xgb`, so the comparison can be changed in the manifest rather than
by copying a driver.

The historical drivers still use the repository optimizer for every marked or
reported optimum. They should not be copied back into `scripts/`; promote only a
reusable interface into `src/` and invoke it from a manifest.
