# Mathematical Reference

This file records the mathematics implemented by the repository. It documents
verified behavior; it does not override the implementation.

## Outline

1. [Purpose, precedence, and conventions](#1-purpose-precedence-and-conventions)
2. [Real-data inputs and feature processing](#2-real-data-inputs-and-feature-processing)
3. [Policies and feature maps](#3-policies-and-feature-maps)
4. [Pricing objectives](#4-pricing-objectives)
5. [Objective composition and constraints](#5-objective-composition-and-constraints)
6. [Real-data analysis quantities](#6-real-data-analysis-quantities)
7. [Uncertainty and lower bounds](#7-uncertainty-and-lower-bounds)
8. [Gradients and estimators](#8-gradients-and-estimators)
9. [Optimization rules](#9-optimization-rules)
10. [Implementation and verification index](#10-implementation-and-verification-index)

## 1. Purpose, Precedence, and Conventions

Resolve disagreements in this order:

1. current implementation in the relevant source module;
2. tests;
3. this file;
4. `README.md`;
5. `AGENTS.md`.

A mismatch is a maintenance bug: reconcile the lower-priority source in the
same change. The repository minimizes objectives. For customer $i$, $x_i$ is
state, $u_i=\pi_\theta(x_i)$ is the relative price change, and $\theta$ is the
policy parameter. Reported profit is the negative of pricing cost. Population
averages use $n^{-1}\sum_i$ unless stated otherwise.

The stable sigmoid is defined separately on the two numerical branches:

$$
\sigma(z)=(1+e^{-z})^{-1},\qquad z\geq 0,
$$

$$
\sigma(z)=e^{z}(1+e^{z})^{-1},\qquad z<0.
$$

Its derivative is $\sigma'(z)=\sigma(z)(1-\sigma(z))$.

Source: `src/objective/_math.py::_sigmoid`.

## 2. Real-Data Inputs and Feature Processing

### 2.1 Immutable rows and column roles

`src/data/dataset.csv` is the canonical semicolon-separated source. Loaders
never modify it. They select columns and stable zero-based CSV row positions,
then transform copies in memory.

A row is eligible when every `REQUIRED_DATASET_COLUMNS` value is present.
Seeded cohorts are sorted samples without replacement from eligible positions.
Saved row positions plus their digest identify a replayable cohort.

- Acceptance state is the 19 `ACCEPTANCE_STATE_COLS`, including premium.
- Loss state is the 18 `LOSS_FEATURE_COLS`, excluding premium.
- `X_policy_premium` is also read separately for revenue.
- Policy-generated `U` is appended only to acceptance-model input.
- Historical `U`, `is_churn`, `Y_G_Loss`, IDs, dates, and
  `X_upcoming_premium` are diagnostic or lookup fields, not objective state.

An `id` may be retained only as a spline-curve lookup key.

Sources: `src/data/dataset_metadata.py`, `src/data/loader.py`.

### 2.2 Saved artifact transform

Let $r\in\mathbb{R}^{d}$ be numeric source columns, $\mu$ the saved training mean,
and

$$
\Sigma=Q\mathrm{diag}(\lambda_1,\ldots,\lambda_d)Q^{\top},
\qquad \tilde\lambda_j=\max(\lambda_j,\varepsilon).
$$

Without PCA,

$$
z_{\rm num}=(r-\mu)Q\mathrm{diag}(\tilde\lambda_j^{-1/2})Q^{\top}.
$$

With $k$ PCA components,

$$
z_{\rm num}=(r-\mu)Q_{[:,1:k]}
\mathrm{diag}(\tilde\lambda_1^{-1/2},\ldots,\tilde\lambda_k^{-1/2}).
$$

For categorical column $c$, training-order categories define mapping $m_c$.
Missing values become `__MISSING__`; unseen values receive code $|m_c|$. The
encoded value is

$$
z_c=\frac{m_c(c)}{\max(|m_c|,1)}.
$$

Numeric outputs precede categorical outputs. Estimator input is reindexed to
`feature_names_in_` when present.

Source: `src/data/feature_processor.py::FeatureProcessor`.

### 2.3 Model and policy frames

Acceptance follows

$$
x_{\rm raw,acc}\longrightarrow z_{\rm acc}
\longrightarrow[z_{\rm acc},U]\longrightarrow\widehat{a}(x,U).
$$

Loss follows

$$
x_{\rm raw,loss}\longrightarrow z_{\rm loss}\longrightarrow\widehat{L}(x)
$$

and never receives `U`. Class 1 is interpreted using the artifact's recorded
acceptance/churn target orientation.

By default the policy reuses acceptance-side processed state. Optional policy
preprocessing applies a second fitted transform
$z_{\rm acc}\mapsto z_{\rm policy}$ without changing black-box model input.
Saved policies record both preprocessing stages and the feature map separately.

Sources: `src/data/loader.py::ModelArtifactBundle.model_frame`,
`src/objective/objectives/generali/model_based.py`,
`src/objective/policy_preprocessing.py`, `src/experiments/policy_artifacts.py`.

## 3. Policies and Feature Maps

Feature maps return $\varphi(x)$; linear and bounded heads prepend an intercept:

$$
\phi(x)=[1,\varphi(x)].
$$

Identity uses $\varphi(x)=x$. For total degree $D$,

$$
\mathcal{A}_D=\{\alpha\in\mathbb{N}_{0}^{d}:1\leq|\alpha|_1\leq D\},
\qquad \varphi_D(x)=[x^{\alpha}:\alpha\in\mathcal{A}_D],
$$

giving $\binom{d+D}{D}$ head parameters including the intercept.

The additive Chebyshev map uses

$$
t_j=\mathrm{clip}(x_j/s,-1,1),\quad
T_0=1,\quad T_1=t,\quad T_k=2tT_{k-1}-T_{k-2},
$$

and concatenates $T_1(t_1),\ldots,T_D(t_d)$ by degree. It has $dD$ mapped
features and no interactions.

Policy heads are

$$
u_{\rm constant}=\theta_0,
\qquad
u_{\rm linear}=\theta^{\top}\phi(x),
$$

$$
u_{\rm bounded}=l+(h-l)\sigma(\theta^{\top}\phi(x)),
$$

with

$$
\nabla_\theta u_{\rm bounded}
=(h-l)\sigma(z)(1-\sigma(z))\phi(x).
$$

The MLP policy is

$$
h_1=\tanh(W_1\varphi(x)+b_1),\quad
h_2=\tanh(W_2h_1+b_2),\quad
u=0.5-\sigma(W_3h_2+b_3).
$$

Source: `src/objective/policy.py`.

## 4. Pricing Objectives

### 4.1 Fixed regression benchmark

$$
a=\sigma(\beta_1^{\top}x+\beta_2u),\quad
L=\beta_3^{\top}x,\quad R=\beta_4u,
$$

$$
f(u;x)=a(L-R),
\qquad
\frac{\partial f}{\partial u}=\beta_2a(1-a)(L-R)-a\beta_4.
$$

Source: `src/objective/objectives/synthetic/fixed_regression.py`.

### 4.2 Planted logistic benchmark

Let $z=\alpha u+\beta^{\top}x+b$ and
$p^{*}(x)=\sigma(\alpha u^{*}+\beta^{\top}x+b)$. Then

$$
f(u;x)=\log(1+e^{z})-p^{*}(x)z,
\qquad
\frac{\partial f}{\partial u}=\alpha(\sigma(z)-p^{*}(x)).
$$

The unique action optimum is $u^{*}$.

Source: `src/objective/objectives/synthetic/planted_logistic.py`.

### 4.3 Real-data model objective

For premium $p(x)$, acceptance $a(x,u)$, and predicted or observed loss $L(x)$,

$$
f(u;x)=a(x,u)[L(x)-(1+u)p(x)].
$$

The optimizer minimizes $J(\theta)=n^{-1}\sum_i f(\pi_\theta(x_i);x_i)$.
Displayed customer profit is

$$
P_i(u)=-f(u;x_i)=a(x_i,u)[(1+u)p(x_i)-L(x_i)].
$$

For GLM logit $g(x)+\beta_u u$,

$$
\frac{\partial a}{\partial u}=\beta_u a(1-a),
\qquad
\frac{\partial f}{\partial u}
=\frac{\partial a}{\partial u}[L-(1+u)p]-ap.
$$

Spline acceptance is one minus monotone churn. Exact analysis splines use
constant-left and clipped-linear-right behavior outside $[0,0.16]$.

Sources: `src/objective/objectives/generali/model_based.py`,
`src/data/monotone_spline_xgb.py`, `src/reporting/real_data.py`.

### 4.4 Synthetic ladder and proof benchmark

The strongly convex rung is

$$
f(w)=\frac{1}{2}(w-w^{*})^{\top}A(w-w^{*}),
\qquad \nabla f(w)=A(w-w^{*}),
$$

where $A=Q\mathrm{diag}(\lambda)Q^{\top}$ has eigenvalues in
$[\mu,\mu\kappa]$.

The smoothed nonconvex rung is

$$
f(w)=\frac{1}{2}\|w-w^{*}\|^{2}
-a_0e^{-\|w-w^{*}\|^{2}/(2s_0^{2})}
-\sum_j a_j\psi\!\left(\frac{\|w-c_j\|^{2}}{\rho_j^{2}}\right),
$$

where $\psi(s)=e^{1-1/(1-s)}$ for $0\leq s<1$ and zero otherwise. Disjoint
supports, positive clearance from $w^{*}$, and
$a_j<\frac{1}{2}(\|c_j-w^{*}\|-\rho_j)^{2}$ preserve the unique global minimum.
Piecewise convex and double-well rungs remain explicit structural stubs.

The proof-validation objective is

$$
f(x)=x^{2}+\frac{1}{2}(\sin x-x),
$$

with $f''(x)\in[1.5,2.5]$, $x^{*}=0$, and $|f'''(x)|\leq0.5$.

Sources: `src/objective/objectives/synthetic/ladder.py` and
`proof_validation.py`.

## 5. Objective Composition and Constraints

Manifest modifications are applied in listed order. `base_value` methods expose
the wrapped unmodified objective for reporting.

An action bias gives $\widehat{M}(x,u)=M(x,u)+b(x,u)$. Implemented fields include

$$
b_{\rm linear}(u)=-\lambda u,
\qquad
b_{\rm hinge}(u)=-\lambda(u-h)_+.
$$

For knots $(v_j,b_j)$, `NaturalCubicActionBias` uses natural cubic spline $S_b$:

$$
b(u)=\lambda S_b(\mathrm{clip}(u,v_1,v_m)).
$$

Its derivative is $\lambda S_b'(u)$ inside $(v_1,v_m)$ and zero outside.

Noise gives $\widehat{M}=M+\delta$. Homoskedastic noise has scale $\sigma_0$;
heteroskedastic noise scales the same query-keyed unit-normal field by
$\sigma_0+\gamma|u-u_c|$. Noisy objectives intentionally have no analytical
gradient.

For mean acceptance $\bar a(\theta)$ and floor $a_{\min}$, the smooth penalty is

$$
s=\tau\log(1+e^{(a_{\min}-\bar a)/\tau}),
\qquad
J_{\rm pen}=J+\lambda s^{2}.
$$

The Lagrangian form is

$$
J_{\rm lag}=J+\lambda(a_{\min}-\bar a).
$$

Direct trust-constr enforcement instead solves

$$
\min_\theta J(\theta)
\quad\text{subject to}\quad
\bar{a}(\theta)\geq a_{\min}.
$$

Sources: `src/objective/modifications/`.

## 6. Real-Data Analysis Quantities

For profit matrix $P_{ij}=P_i(u_j)$,

$$
\mu_j=\frac{1}{n}\sum_iP_{ij},
\qquad
s_j=\sqrt{\frac{1}{n}\sum_i(P_{ij}-\mu_j)^{2}},
$$

$$
m_j=\mathrm{median}_iP_{ij},
\qquad
\mathrm{MAD}_j=\mathrm{median}_i|P_{ij}-m_j|.
$$

MAD is raw unless a displayed quantity explicitly multiplies it by $1.4826$.

Customer/action support uses clipped saved-whitened numeric coordinates and
one-hot categorical coordinates divided by $\sqrt{2}$. For neighbors $N_i$,
state weights $q_{ik}$, action bandwidth $b$, and historical action $U_k$,

$$
S_i(u)=\sum_{k\in N_i}q_{ik}
\exp\left[-\frac{1}{2}\left(\frac{U_k-u}{b}\right)^{2}\right].
$$

The normalized coverage penalty is

$$
W_i(u)=c\left(1-\frac{S_i(u)}{\max_vS_i(v)}\right).
$$

Marginal action support uses

$$
n_{\rm eff}(u)=\frac{(\sum_iw_i(u))^{2}}{\sum_iw_i(u)^{2}},
\qquad
w_i(u)=\exp\left[-\frac{1}{2}\left(\frac{U_i-u}{b}\right)^{2}\right].
$$

Customer response grids use piecewise-linear interpolation; the derivative is
the slope of the containing interval. A grid may render or interpolate an
objective but never selects a reported optimum.

Sources: `src/reporting/profit_dispersion.py`, `src/reporting/real_data.py`,
`src/data/coverage.py`, `src/objective/gridded.py`.

## 7. Uncertainty and Lower Bounds

For finite policy class $\Pi$, simultaneous error widths $\mathcal{E}^{\pi}$ give

$$
V_{\mathrm{LCB}}^{\pi}=\widehat{V}^{\pi}-\frac{1}{2}\mathcal{E}^{\pi}.
$$

On the simultaneous coverage event, an $\varepsilon$-optimal LCB policy obeys

$$
V^{\widehat{\pi}}\geq
V^{\widetilde{\pi}}-\mathcal{E}^{\widetilde{\pi}}-\varepsilon
$$

for every comparator $\widetilde{\pi}\in\Pi$. The finite Gaussian validation uses
$V^{\pi}=\pi$, $\widehat{V}^{\pi}=\pi+\pi Z^{\pi}$, and Bonferroni quantile
$q=\Phi^{-1}(1-\delta/(2|\Pi|))$.

The continuous rank-one validation uses

$$
V(\pi)=5\pi-5\pi^{2},
\qquad
\widehat{V}_s(\pi)=V(\pi)+\pi Z_s,
$$

so $\sup_{\pi>0}|\widehat{V}_s(\pi)-V(\pi)|/\pi=|Z_s|$ and
$q=\Phi^{-1}(1-\delta/2)$ needs no finite-class factor.

Finite-Fourier GP experiments define one analytic path

$$
G_s(x)=\frac{1}{\sqrt{J}}\sum_{j=1}^{J}
[A_{s,j}\cos(\omega_jx)+B_{s,j}\sin(\omega_jx)].
$$

Its covariance is

$$
k_J(x,x')=\frac{1}{J}\sum_j\cos(\omega_j(x-x')).
$$

Optimizer queries evaluate this formula directly; plotted connections are not
an off-grid definition. The spline/XGBoost support cloud is explicitly a
support-risk proxy, not a calibrated confidence interval.

Sources: `src/experiments/policy_lcb/`,
`src/objective/modifications/regularization.py`.

## 8. Gradients and Estimators

For a differentiable action objective and policy, the population chain rule is

$$
\nabla_\theta J(\theta)=\frac{1}{n}\sum_i
\frac{\partial f}{\partial u}(u_i;x_i)
\nabla_\theta\pi_\theta(x_i).
$$

The first-order method uses the objective's exact gradient. With coordinate
vector $e_k$ and smoothing scale $\sigma$, central finite differences use

$$
\widehat{g}_k=
\frac{J(\theta+\sigma e_k)-J(\theta-\sigma e_k)}{2\sigma}.
$$

For $m$ independent standard Gaussian vectors $\varepsilon_j$, the one-sided
Gaussian Stein estimator is

$$
\widehat{g}_{\rm GS}=
\frac{1}{m\sigma}\sum_{j=1}^{m}
J(\theta+\sigma\varepsilon_j)\varepsilon_j.
$$

For independent Rademacher vectors $\Delta_j\in\{-1,1\}^{d}$, SPSA uses

$$
\widehat{g}_{\rm SPSA}=
\frac{1}{m}\sum_{j=1}^{m}
\frac{J(\theta+\sigma\Delta_j)-J(\theta-\sigma\Delta_j)}{2\sigma}
\Delta_j.
$$

The Stein-difference estimator replaces $\Delta_j$ by standard Gaussian
$\varepsilon_j$ in the same two-sided expression. In `u` perturbation space,
these scalar-action estimators are evaluated customer by customer and mapped
back through $\nabla_\theta\pi_\theta(x_i)$. Random estimators use explicitly
seeded generators; batch and perturbation streams can be separated.

Sources: `src/optimization/gradients/methods.py`,
`src/objective/utils.py`.

## 9. Optimization Rules

For constant-step descent,

$$
\theta_{t+1}=\theta_t-\alpha\widehat{g}_t.
$$

Armijo backtracking chooses $\alpha=\alpha_0\rho^{k}$ until

$$
J(\theta_t-\alpha\widehat{g}_t)
\leq J(\theta_t)-c\alpha\|\widehat{g}_t\|^{2}.
$$

`l-bfgs-b` delegates unconstrained or box-constrained minimization to the
repository solver wrapper. `trust-constr` is the repository path for nonlinear
constraints such as the mean-acceptance floor. Optax SGD and Adam use the same
configured gradient method in the repository update loop.

Every optimum, optimizer action, or optimizer shift reported by a script,
notebook, or analysis must come from `src/optimization/`, or replay an exact
saved optimizer artifact with provenance. A plot grid may evaluate and display
an objective, but `argmin`, `argmax`, sorting, or an equivalent grid scan may
not select the reported solution.

Sources: `src/optimization/base.py`, `src/optimization/solvers.py`,
`src/optimization/steps.py`, `src/optimization/optax_loop.py`.

## 10. Implementation and Verification Index

This index is navigational. When a formula disagrees with code or a test, use
the precedence in Section 1 and update this document.

| Topic | Primary implementation | Representative verification |
|---|---|---|
| Dataset columns and cohorts | `src/data/dataset_metadata.py`, `src/data/loader.py` | `tests/data/test_data_loader.py`, `tests/data/test_dataset_metadata.py` |
| Saved feature transforms | `src/data/feature_processor.py` | `tests/data/test_feature_processor.py` |
| Policy preprocessing and maps | `src/objective/policy_preprocessing.py`, `src/objective/policy.py` | `tests/objective/test_policy_preprocessing.py`, `tests/objective/test_feature_maps.py`, `tests/objective/test_policy_batch.py` |
| Real-data objective | `src/objective/objectives/generali/model_based.py` | `tests/objective/test_model_based_objective.py` |
| Synthetic objectives | `src/objective/objectives/synthetic/` | `tests/objective/` |
| Bias, noise, and constraints | `src/objective/modifications/` | `tests/objective/test_objective_modifications.py`, `tests/objective/test_biased_objective.py` |
| Coverage and grid interpolation | `src/data/coverage.py`, `src/objective/gridded.py` | `tests/data/test_coverage.py`, `tests/objective/test_gridded.py` |
| Reusable real-data reports | `src/reporting/real_data.py`, `src/reporting/profit_dispersion.py` | `tests/reporting/test_real_data_reporting.py` |
| Gradient methods | `src/optimization/gradients/methods.py` | `tests/optimization/test_gradient_methods_math.py` |
| Solvers and step rules | `src/optimization/` | `tests/optimization/` |
| Manifest and report dispatch | `src/experiments/manifest.py`, `src/experiments/reporting/recipes.py` | `tests/experiments/test_manifest.py`, `tests/experiments/test_reporting_recipes.py` |
| Provenance and exact-spline cache | `src/experiments/provenance.py`, `src/reporting/exact_spline_cache.py` | `tests/reporting/test_exact_spline_cache.py` |

The implementation and tests above are authoritative. This file is the single
mathematical reference for maintainers, not a second implementation.
