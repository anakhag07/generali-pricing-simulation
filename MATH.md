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
$p^{\star}(x)=\sigma(\alpha u^{\star}+\beta^{\top}x+b)$. Then

$$
f(u;x)=\log(1+e^{z})-p^{\star}(x)z,
\qquad
\frac{\partial f}{\partial u}=\alpha(\sigma(z)-p^{\star}(x)).
$$

The unique action optimum is $u^{\star}$.

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
f(w)=\frac{1}{2}(w-w^{\star})^{\top}A(w-w^{\star}),
\qquad \nabla f(w)=A(w-w^{\star}),
$$

where $A=Q\mathrm{diag}(\lambda)Q^{\top}$ has eigenvalues in
$[\mu,\mu\kappa]$.

The smoothed nonconvex rung is

$$
f(w)=\frac{1}{2}\|w-w^{\star}\|^{2}
-a_0e^{-\|w-w^{\star}\|^{2}/(2s_0^{2})}
-\sum_j a_j\psi\!\left(\frac{\|w-c_j\|^{2}}{\rho_j^{2}}\right),
$$

where $\psi(s)=e^{1-1/(1-s)}$ for $0\leq s<1$ and zero otherwise. Disjoint
supports, positive clearance from $w^{\star}$, and
$a_j<\frac{1}{2}(\|c_j-w^{\star}\|-\rho_j)^{2}$ preserve the unique global minimum.
Piecewise convex and double-well rungs remain explicit structural stubs.

The proof-validation objective is

$$
f(x)=x^{2}+\frac{1}{2}(\sin x-x),
$$

with $f''(x)\in[1.5,2.5]$, $x^{\star}=0$, and $|f'''(x)|\leq0.5$.

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

### 7.1 Quadratic OLS bootstrap band on a finite grid

The bootstrap construction uses independent training inputs $x_i\sim N(0,1)$
and errors $\varepsilon_i\sim N(0,\sigma^2)$, with

$$
y_i=5x_i-5x_i^2+\varepsilon_i,\qquad
p(a)=(1,a,a^2)^\top,\qquad P_{i,:}=p(x_i)^\top.
$$

Quadratic OLS fits all three coefficients, including the intercept. With
full column rank and $n>3$,

$$
\widehat\beta=(P^\top P)^{-1}P^\top y,\qquad
\widehat\sigma^2=\frac{\|y-P\widehat\beta\|^2}{n-3},\qquad
\widehat s(a)=\widehat\sigma\sqrt{p(a)^\top(P^\top P)^{-1}p(a)}.
$$

The original fit uses reduced QR solves. Conditional on the observed dataset,
each pairs-bootstrap replicate samples $n$ row indices independently and
uniformly with replacement. Both the input and response use the same indices:

$$
I_{bi}\sim\mathrm{Uniform}\{1,\ldots,n\},\qquad
P^*_{b,i,:}=P_{I_{bi},:},\qquad y^*_{b,i}=y_{I_{bi}},\qquad
\widehat\beta_b^*=(P_b^{*\top}P_b^*)^{-1}P_b^{*\top}y_b^*.
$$

Refits use numerical least squares without normal equations. Rank-deficient
resamples fail explicitly rather than being redrawn or discarded. No Gaussian
noise is generated inside the bootstrap. The calibration statistic is:

$$
T_b^{\ast}=\max_{a\in\mathcal{G}}
\frac{|p(a)^\top(\widehat\beta_b^{\ast}-\widehat\beta)|}{\widehat s(a)}.
$$

Here $\mathcal{G}$ is the manifest's inclusive equally spaced grid in $[0,1]$.
The original fit's $\widehat s$ stays fixed in all bootstrap denominators.
Let $\widehat c$ be the empirical $(1-\delta)$ quantile using NumPy's
`method="higher"` (zero-based sorted index `ceil((B-1)*(1-delta))`). Then

$$
r_\delta(a)=\widehat c\,\widehat s(a),\qquad
C=\mathbf{1}\left[
\max_{a\in\mathcal{G}}\frac{|\widehat f(a)-f(a)|}{\widehat s(a)}
\leq\widehat c\right].
$$

Calibration takes only observations and bootstrap randomness; truth is used
only to generate data and evaluate coverage. Stage 1 reports one Boolean $C$.
Stage 2 redraws inputs and errors independently, reconstructs each band, and
reports the fraction of $R$ covered datasets with a 95% Wilson interval.
The target is approximate bootstrap coverage on the grid, not a finite-sample
proof or a continuous-domain guarantee. In particular this fixed-denominator
bootstrap does not reproduce the sampling variation of the residual variance
estimate in the observed statistic. Polynomial evaluations exist off-grid,
but grid calibration does not certify between-grid coverage. No coefficient
ellipsoid or optimizer/action selection is involved.

Sources: `src/experiments/bootstrap_band.py`,
`src/experiments/bootstrap_band_reporting.py`.

### 7.2 Analytical replay on the entire real line

The all-real replay reads exactly the observations, OLS coefficients, residual
scale estimates, and bootstrap coefficients saved by section 7.1. It does not
draw new data. Write $V=(P^\top P)^{-1}$ and $v(a)=p(a)^\top Vp(a)$. Then

$$
e(a)=p(a)^\top(\widehat\beta-\beta_0)
=p(a)^\top\left(\sum_i p_i p_i^\top\right)^{-1}\sum_i p_i\varepsilon_i.
$$

For every coefficient difference $d$, replace the grid statistic by

$$
T(d)=\sup_{a\in\mathbb{R}}\frac{|p(a)^\top d|}{\widehat\sigma\sqrt{v(a)}}.
$$

For $h(a)=p(a)^\top d$ and $q(a)=\widehat\sigma^2v(a)$, finite nonzero
stationary values of $h(a)^2/q(a)$ satisfy $2h'(a)q(a)-h(a)q'(a)=0$.
The nominal degree-five term cancels, leaving degree at most four. Both tails
have the limit $d_2^2/(\widehat\sigma^2 V_{22})$. Numerical polynomial roots
and this tail limit propose the statistic; exact rational polynomial sign
tests certify lower and upper bounds. The final saved interval has width at
most twice the configured tolerance times $\max(1,T)$, apart from rounding.
No grid selects or checks the statistic. These are user-requested analytical
coverage statistics, not optimizer actions or pricing optima.

Specifically, for every tested threshold $t\geq0$,

$$
T(d)\leq t\quad\Longleftrightarrow\quad
H_{t,d}(a)=t^2q(a)-h(a)^2\geq0\quad\forall a\in\mathbb{R}.
$$

The exact sign test handles zero and constant polynomials, checks leading sign
and degree, and counts real roots of the odd-multiplicity square-free factors.
A positive-leading even-degree polynomial is nonnegative on the real line
exactly when all its real roots have even multiplicity. Saved binary floating
point inputs are interpreted as exact rationals for these sign decisions;
OLS itself remains a floating-point fit. Ambiguous numerical proposals are
refined until the certificate succeeds, or fail explicitly.

Calibration uses $T(\widehat\beta_b^{\ast}-\widehat\beta)$ only. Its empirical
quantile uses certified upper bounds (a conservatively rounded approximation
within the stored quantile bracket), retaining `method="higher"` and the
original fit's standard error in all denominators. Truth is used afterward:

$$
C_r=\mathbf{1}\left[
\widehat c^2\widehat\sigma^2v(a)-e_r(a)^2\geq0
\quad\forall a\in\mathbb{R}\right],\qquad
\widehat{\mathrm{Coverage}}=\frac{1}{R}\sum_{r=1}^{R}C_r.
$$

Each realized containment decision is algebraic over the whole real line.
The repeated-dataset coverage rate remains empirical validation of approximate
bootstrap coverage, not a theorem of exact 95% sampling coverage. Figures show
finite display windows only. The observed fitting error is checked against
the known observation-error identity; residuals are not substituted for
$\varepsilon_i$ in that identity.

Source: `src/experiments/bootstrap_band_continuous.py`.

### 7.3 Controlled all-real bootstrap sweeps and LCB regret

The controlled experiment keeps $a_i\sim N(0,1)$ and
$f(a)=5a-5a^2$ on all of $\mathbb{R}$. Independent seed streams generate
training actions, standardized observation errors, and bootstrap row-selection
uniforms. Original dataset seeds are unchanged from the parametric experiment.
Training samples are nested across $N$. For shared iid $U_{bi}\sim U[0,1)$,
zero-based indices $I_{bi}^{(N)}=\lfloor N U_{bi}\rfloor$ select the same
observed input/response row. The first $N$ columns are used at size $N$, and
bootstrap samples use prefixes across $B$. Noise settings reuse these indices.

For each design, first fit and resample its actual responses at noise scale one.
With $V=(P^\top P)^{-1}$, define standardized pairs-refit perturbations

$$
d_b=\frac{\widehat\beta_b^*-\widehat\beta}{\widehat\sigma},\qquad
T_b^*=\sup_{a\in\mathbb R}
\frac{|p(a)^\top d_b|}{\sqrt{p(a)^\top Vp(a)}}.
$$

This is an actual refit on observed pairs, not an independent perturbation
of the fitted model. Calibration receives only observed inputs, responses and
row indices. Section 7.2 certifies its all-real suprema and containment.
For full-rank resamples of the correctly specified quadratic model and paired
positive noise scales, OLS equivariance gives
$\widehat\beta_{b,\sigma}^*=\widehat\beta_\sigma+\widehat\sigma_\sigma d_b$.
Thus $e_\sigma=\sigma e_1$ and $r_{\delta,\sigma}=\sigma r_{\delta,1}$ (up to
rounding), allowing cached standardized refits/certificates across sigma.
Coverage indicators are identical across paired sigma settings.

Increasing $B$ estimates the same empirical-bootstrap quantile more precisely;
it need not decrease regret or width. Increasing $N$ supplies more information.
Under iid sampling, finite moments, a nonsingular population design and a
continuous limiting supremum distribution, as both $N,B\to\infty$,

$$
\Pr\{ |\widehat f(a)-f(a)|\le r_\delta(a)\ \forall a\in\mathbb R\}
\longrightarrow 1-\delta.
$$

For this fixed-dimensional quadratic family the normalized basis has finite
tail limits, so coefficient-bootstrap consistency extends to the whole-line
standardized supremum. Unstandardized root-N uniform error/width rates apply
only on compact intervals, or pointwise; the whole-line maximum width is
infinite. Finite-N/B coverage remains approximate, distinct from exact
polynomial verification of each represented band's containment.

The repository optimizer minimizes $-L(a)$ without action bounds, initialized
at the manifest's starts in $[0,1]$, where

$$
L(a)=p(a)^\top\widehat\beta-k\sqrt{v(a)},\quad
k=\widehat c\widehat\sigma,\quad v(a)=p(a)^\top Vp(a),
$$

$$
L'(a)=p'(a)^\top\widehat\beta-k\frac{p'(a)^\top Vp(a)}{\sqrt{v(a)}}.
$$

The leading tail coefficient is $\widehat\beta_2-k\sqrt{V_{22}}$.
A positive value means the LCB is unbounded above, a negative value means
both tails tend to minus infinity, and the exactly zero case is explicitly
flagged as degenerate rather than silently treated as coercive.

Optimizer actions and the true reference action both come from
`src/optimization/`. Global verification never selects an alternative action:
for a proposed level $u$, let $g(a)=p(a)^\top\widehat\beta-u$ and
$S(a)=k^2v(a)-g(a)^2$. Then

$$
L(a)\leq u\ \forall a\quad\Longleftrightarrow\quad
\left[g(a)\leq 0\ \mathrm{or}\ S(a)\geq 0\right]\ \forall a.
$$

Exact rational real-root isolation of $gS$ determines the signs on every
open sign cell, including both tails. Strict simultaneous violations are
open, so root points require no separate test. A candidate from the repository
optimizer is accepted only if its certified lower value plus the manifest
gap tolerance is an all-real upper bound. Floating coefficients are treated
as exact rationals, as in section 7.2. Unbounded, degenerate, or uncertified
optimization cases are counted and never silently pooled as successful regret.

The three metrics are $C_r$ from section 7.2,
$W_r=r_{\delta,r}(a^\star)$, and
$R_{\mathrm{LCB},r}=f(a^\star)-f(\widehat a_r)$.
Whole-line maximum width is infinite, hence the explicitly local width $W_r$.
On the coverage event, an LCB solution with objective gap at most $\eta$ obeys
$R_{\mathrm{LCB},r}\leq 2W_r+\eta$ (plus the certified true-reference tolerance).
The plotted $2W_r$ is a bound benchmark, not an unconditional mean theorem.
Coverage uses Wilson intervals; width/regret means use standard errors across
independent datasets. Regret summaries state their successful-case denominator.
The finite-$B$, fixed-denominator bootstrap remains approximate sampling
coverage, not an exact finite-sample confidence theorem.

The dense sample-size follow-up uses the same definitions with
$\sigma=1$, $B=2000$, and
$N\in\{20,25,30,50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000\}$.
For each dataset index, all conditions use prefixes of the same length-5000
training-action and observation-error streams, while each design receives 2000
pairs-bootstrap refits. It therefore isolates the effect of training sample size and
writes a distinct result tree without changing the original three-axis sweep.

Source: `src/experiments/bootstrap_band_sweep.py`.

### 7.4 Empirical coverage Pareto replay

At fixed $\sigma=1$, the saved 2000 bootstrap supremum brackets at each $N$
allow reconstruction of every manifest $B$ using its prefix and the same
`higher` quantile of the certified upper endpoints. The observed statistic's
certified bracket determines containment whenever it lies strictly to one
side of the threshold; ambiguous cases use the exact polynomial test in §7.2.
All original N/B sweep containment indicators must match this reconstruction.
No bootstrap draw, model fit, or optimizer action is generated by this replay.

Write $\widehat C(N,B)=R^{-1}\sum_{r=1}^R C_r(N,B)$. The descriptive empirical
Pareto objectives, all minimized, are

$$
\left(N,\ B,\ |\widehat C(N,B)-(1-\delta)|\right).
$$

A measured configuration is dominated when another has no larger value in any
component and a strictly smaller value in at least one. Thus overcoverage and
undercoverage are penalized symmetrically. This is a comparison among measured
configurations, not optimization of an action objective. Integer covered counts
and an exact rational target preserve ties above and below the target.
The highlighted frontier depends on the 100-dataset coverage estimates; it is
exploratory, not a certified population frontier. Wilson intervals are pointwise
and do not account for selecting configurations from the full grid. Connected
or interpolated continuous coverage surfaces are not assumed.

Source: `scratch/plot_bootstrap_coverage_frontier.py`.

### 7.5 Dedicated joint N/B coverage sweep

The dedicated experiment applies the unchanged construction of §7.3 to
explicit pairs $(N_j,B_j)$, with both coordinates increasing. The default path
has $N_j=B_j\in\{20,50,100,200,500,1000,2000,3000,5000\}$, $\sigma=1$,
$\delta=0.05$, and $R=2000$ independent outer datasets. It reports only

$$
\widehat C_j=\frac{1}{R}\sum_{r=1}^R
\mathbf{1}\{ |\widehat f_{r,N_j}(a)-f(a)|\leq
\widehat c_{r,N_j,B_j}\widehat s_{r,N_j}(a)\quad\forall a\in\mathbb R\}.
$$

Each bootstrap quantile uses the same certified upper endpoints and `higher`
order-statistic convention as §7.3. Confidence intervals for $\widehat C_j$
are pointwise Wilson intervals over independent outer datasets. Settings share
nested observations and bootstrap uniform prefixes, so comparisons across
settings are paired. Neither $\widehat C_j$ nor its population counterpart is
constrained to be monotone. No ranking or additional minimization objective is
introduced. Fixed confidence level targets asymptotic coverage $1-\delta$.

Source: `src/experiments/bootstrap_joint_coverage.py`.

### 7.6 Cartesian N/B coverage replay and extension

The Cartesian extension evaluates the same $C_r(N,B)$ in §7.5 at all 81
combinations of the nine N and B values. Original fitted coefficients and
standard errors are retained. For each N, reuse the longest compatible saved
bootstrap sequence of length $K_N$. Compute only replicates $K_N+1,\ldots,5000$
from the original uniform stream, then calibrate each B using the first B
certified supremum brackets and the unchanged `higher` quantile. Reusing
prefixes introduces no new statistical approximation. Exact saved containment
outcomes on the diagonal must agree with replay. No optimizer action is chosen.

Every cell reports $\widehat C(N,B)=R^{-1}\sum_r C_r(N,B)$ with $R=2000$ and
pointwise Wilson intervals. Heatmap cells are discrete settings, not samples of
an interpolated coverage surface. Lower/upper interval bounds and half-widths
quantify Monte Carlo uncertainty; they are not simultaneous across grid cells.

Source: `src/experiments/bootstrap_cartesian_coverage.py`.

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
