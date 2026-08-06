# Model artifact hierarchy

There are exactly three runtime model families:

```text
src/data/models/
├── linear/
│   ├── acceptance.pkl
│   └── loss.pkl
├── xgb/
│   ├── acceptance.pkl
│   └── loss.pkl
└── monotone-spline-xgb/
    └── acceptance-curves.npz
```

`linear` contains the date-free logistic-acceptance and Ridge-loss runtime
models. `xgb` contains fold index 0 exported from the newest available
acceptance and loss CV bundles. The exports retain source filenames, hashes,
metrics, and `source_fold=0` inside the pickle rather than in filenames.

`monotone-spline-xgb` is not another independently trained XGB model. Its NPZ
stores only portable PCHIP curve coefficients for 200 deterministic policy
profiles and a hash of `xgb/acceptance.pkl`. At runtime the adapter composes
those curves with `xgb/acceptance.pkl`; policies outside the cache fall back to
that raw XGB model, matching the original wrapper contract. Its loss side is
always `xgb/loss.pkl`.

The cached curves are fitted in churn-probability space using a weighted
smoothing spline, dense-grid isotonic regression, and shape-preserving PCHIP.
They are bounded in `[0, 1]` and monotone non-decreasing in churn (therefore
non-increasing in acceptance). Logit-spline, shifted-sigmoid, dated duplicate
models, conversion-only source pickles, and the old 199-policy comparison
manifest are not part of the runtime hierarchy.
