# Model artifact hierarchy

Runtime model artifacts and conversion inputs are deliberately separate:

```text
src/data/
├── models/
│   ├── linear/
│   ├── xgb/
│   ├── xgb_logit_spline/
│   └── xgb_monotone_spline/
└── model_sources/
    └── acceptance/
```

Files under `models/` are the only experiment-runtime artifacts. Files under
`model_sources/` are trusted, executable pickle inputs used only by explicit
conversion scripts. Portable curve artifacts use NPZ arrays and load with
`allow_pickle=False`.

## Acceptance lineage

```text
glm_20260527

xgb_20260527
└── xgb_logit_spline_20260706       (200 policy-specific curves)

xgb_20260728
└── xgb_monotone_spline_20260728    (200 policy-specific curves)
```

The two curve artifacts cover disjoint sets of 200 canonical policies. They are
not conditional models that can generate a curve for an unseen policy. For a
covered policy, covariates are frozen and the stored curve maps generated price
action `U` to churn; runtime returns acceptance as `1 - churn`. Unknown policy
IDs fail explicitly.

All 200 monotone curves remain in the runtime artifact. One covered canonical
row is missing `X_age` and `X_driving_license_years`, so experiment row
selection uses the 199-policy intersection of curve coverage and complete
objective inputs. No new imputation rule is introduced.

The monotone source wrapper contains one base XGBoost classifier, its
preprocessor, and 200 unique PCHIP curves. Its embedded classifier and
preprocessor exactly match `xgb_20260728`, so the portable artifact stores only
curve arrays and provenance rather than a second base-model copy.

| Artifact ID | Runtime format | Base model | Coverage | Source SHA-256 |
|---|---|---|---:|---|
| `xgb_logit_spline_20260706` | NPZ logit-space B-splines | `xgb_20260527` | 200 | `b92296dda5a11d7a7d84983fb3e2de4089c6cd5083c706c4206213150ccf9db3` |
| `xgb_monotone_spline_20260728` | NPZ probability-space PCHIP | `xgb_20260728` | 200 | `87a4851b241f5f2eb2edf587028770c060fcc87e75581322b4e373ccc26cf0b9` |

The shifted-sigmoid smoother was hard-removed. The global GLM acceptance model
remains: despite having a logistic link, it is one model over policy features
and action rather than a lookup table of fitted per-policy sigmoid curves.
