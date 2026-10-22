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

### 7.4 Fixed-bandwidth Gaussian-support envelope

This separate, non-bootstrap experiment uses the same quadratic truth,
$p(a)=(1,a,a^2)^\top$, iid $a_j\sim N(0,1)$, and
$y_j=p(a_j)^\top\beta_0+\sigma_\varepsilon\epsilon_j$ with iid
$\epsilon_j\sim N(0,1)$ as section 7.3. It fixes $\sigma_\varepsilon=1$ and
reuses the same named design/observation seed streams and nested training
prefixes across the dense $N$ axis. One OLS coefficient vector is shared by
all action queries; the estimation errors at different actions are therefore
dependent, with covariance $\sigma_\varepsilon^2p(a)^\top Vp(a')$.

The action and coverage domain is the entire real line. The fixed Gaussian
bandwidth is $b=0.1$, with no $N$-dependent bandwidth or kernel-density
normalization. Direct off-grid evaluation is

$$
C_N(a)=\sum_{j=1}^{N}\exp\!\left[-\frac{(a_j-a)^2}{2b^2}\right],\qquad
r_N(a)=\frac{q_{1-\alpha/2}\widehat\sigma}{\sqrt{C_N(a)}},\qquad
\widehat f_{\mathrm{LCB},N}(a)=p(a)^\top\widehat\beta-r_N(a),
$$

where $\alpha=0.05$, $q_{0.975}=\Phi^{-1}(0.975)$, and
$\widehat\sigma^2=\|y-P\widehat\beta\|^2/(N-3)$. Log-sum-exp is only a
numerically stable evaluation of this same finite Gaussian sum. The repository
first-order optimizer minimizes $-\widehat f_{\mathrm{LCB},N}$ without action
bounds; starts are initializations, not a finite candidate grid. Both the
true-reference and lower-envelope reported actions come from that optimizer.
The support penalty tends to infinity in either tail, so the lower envelope
tends to minus infinity. A conservative numerical interval-bound check may
verify the optimizer's global objective gap; unlike the polynomial bootstrap
certificate, it is not exact-arithmetic proof. Uncertified results are labeled.

The two-sided simultaneous coverage event for each dataset is

$$
C_r=\mathbf1\left\{
  [p(a)^\top(\widehat\beta-\beta_0)]^2C_N(a)
  \le q_{0.975}^2\widehat\sigma^2\quad\forall a\in\mathbb R
\right\}.
$$

An adaptive interval upper bound checks the complete finite line; an
analytical Gaussian-tail bound covers both infinite tails. A point violating
the inequality proves noncoverage; if numerical bounds cannot resolve a case,
it remains indeterminate rather than being classified by a display grid.
Plots sample a finite display only and do not define the objective or coverage.
Coverage rates use resolved datasets with explicit unresolved counts and Wilson
intervals. Width is $r_N(a^\star)$ and regret is
$f(a^\star)-f(\widehat a_{\mathrm{LCB},N})$. The pointwise Gaussian quantile
and kernel-support heuristic do **not** imply 95% simultaneous coverage; the
sweep empirically evaluates that claim.

Source: `src/experiments/coverage_kernel_sweep.py`.

The paired $q$ follow-up replays the exact saved actions, observations, OLS
coefficients, residual scales, and repository-optimizer true reference from
the dense $N$ sweep. It changes only the standardized envelope coefficient
$q>0$ in $\lambda_{r,N}(q)=q\widehat\sigma_{r,N}$, using the same support
$C_{r,N}(a)$ and the same two-sided, all-real coverage event. The original
$q=\Phi^{-1}(0.975)$ row is replayed verbatim from its saved optimizer output.
For each fixed dataset and $N$, the width at $a^\star$ is linear in $q$ and
the coverage indicator is nondecreasing in $q$; true regret need not be
monotone because the optimizing action can change. Coverage and regret remain
empirical evaluations on the same datasets used to display this sweep, not
an independently validated choice of $q$ that guarantees coverage.

Source: `src/experiments/coverage_kernel_parameter_sweep.py`.

The paired $b$ follow-up holds $q=\Phi^{-1}(0.975)$ and changes only the
Gaussian bandwidth in the same **unnormalized** support sum. For fixed
$a,a_j$, each kernel term increases with $b$; hence $C_{r,N,b}(a)$ is
nondecreasing, the radius is nonincreasing, and the two-sided coverage event
is nonincreasing in $b$. The smoothing shape and overall support scale change
together: this sweep does not isolate a normalized kernel-density bandwidth.
Regret need not be monotone. The original $b=0.1$ rows are likewise replayed
verbatim with optimizer and artifact provenance.

Source: `src/experiments/coverage_kernel_parameter_sweep.py`.
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

### 7.7 Fixed offline pairs bootstrap, transferred envelope, and regret

This experiment loads a checksummed archive of observations and a separately
saved quadratic predictor. No population observations are drawn during fitting
or bootstrap calibration. N takes nested prefixes of the archived rows; B takes
prefixes of row resamples; sigma changes only explicitly saved synthetic response
variants. The three axes vary separately around one baseline. Historical results
are preserved. The predictor is fixed across every condition.

Preparation alone may use synthetic truth to derive paired response snapshots:
$$y_i^{(\sigma,h)}=p(a_i)^\top\beta_0+\sigma h(a_i)z_i,$$
where actions and standardized errors are replayed from the archive. For a
baseline archive at sigma0, z_i=(y_i-p(a_i)'beta0)/sigma0. The homoskedastic profile
is h=1. The heteroskedastic profile is h(a)=exp(eta G(a))/C, with the exact
finite-Fourier GP of the existing GP core. A dedicated GP seed generates its
coefficients once. C is the root mean square of exp(eta G) over the entire saved
action reservoir, so the two profiles have the same average variance there;
this is explicitly empirical normalization, not a population integral. Values
off the logged actions use the same analytic formula and fixed C. Conditional
on the saved action design and profile, original Gaussian errors are independent.
The GP defines a variance profile, not a shared additive observation error.
Because C uses the complete archived action reservoir, hypothetical observation
pairs are not iid unconditionally. The independent-error covariance statement
conditions on that complete design and the GP draw. An iid pairs-bootstrap
theorem cannot be applied without accounting for this normalization dependence.

Calibration has only x, y, and independent row-selection randomness as inputs.
Each replicate resamples N complete observed pairs with replacement, then refits
quadratic OLS. Rank-deficient resamples are redrawn from that same dataset using
a separate rejection stream; attempts and accepted indices are saved. A finite
retry limit fails explicitly. Thus the specified bootstrap law is conditioned
on full column rank, and any bootstrap validity theorem must cover that law.
N-specific seed derivation preserves B prefixes and reuses indices across sigma
and noise profiles. No fresh observation noise is added by bootstrap resampling.

The original auxiliary fit supplies beta_tilde, V=(P'P)^(-1), and the fixed
weight s(a)=sigma_hat sqrt(p(a)'V p(a)). Bootstrap calibration uses
$$T_b^*=\sup_{a\in\mathbb R}|p(a)^\top(\widetilde\beta_b^*-\widetilde\beta)|/s(a),
\quad w(a)=c s(a),\quad r(a)=|\widehat f(a)-\widetilde f(a)|+w(a).$$
Under heteroskedasticity s is a positive weighting function, not a consistent
heteroskedastic standard error. The exact sampling covariance benchmark is
$$V_{\rm oracle}=(P^\top P)^{-1}P^\top\operatorname{diag}(\sigma^2h(a_i)^2)P(P^\top P)^{-1}.$$
It is used only after calibration for diagnostics, never to set the radius.

The empirical higher quantile is available. The default conservative quantile
uses the smallest k<=B satisfying Pr(Binomial(B,1-delta)<=k-1)>=1-eta_mc.
Using the kth order statistic of certified upper supremum bounds then contains
at least 1-delta of the specified conditional bootstrap law with Monte Carlo
confidence at least 1-eta_mc (ties are conservative). Insufficient B is rejected.
This is NOT a probability guarantee for unknown truth. Such a guarantee still
requires a finite-N bootstrap approximation bound. No outer-data coverage rate
or numerical value for that unproved error term is reported.

All functions and containment certificates cover R, including both tails;
plots on [0,1] are visualization only. Let b1=beta_tilde and
b2=2 beta_hat-beta_tilde, and q(a)=c^2 s(a)^2. Then
$$L(a)=\min\{p(a)'b_1,p(a)'b_2\}-\sqrt{q(a)},\qquad
U(a)=\max\{p(a)'b_1,p(a)'b_2\}+\sqrt{q(a)}.$$
L>t is equivalent to the four strict polynomial inequalities
p'b_j-t>0 and (p'b_j-t)^2-q>0, j=1,2. Exact rational sign cells therefore verify
global LCB levels without selecting actions. Substituting t=f(a), and similarly
testing f>U, certifies final-envelope containment. Auxiliary containment uses
q-[p'(beta_tilde-beta0)]^2>=0. Certificates apply to represented coefficients;
they do not enclose OLS roundoff relative to an ideal exact-arithmetic fit.

Repository solvers supply the true-reference, fixed-predictor and LCB actions;
all-real sign tests verify a global objective gap <=1e-7. No action grid, roots,
or direct optimizer substitutes select a reported action. The bootstrap
supremum brackets use tolerance 1e-8. The LCB has an absolute-value kink;
unsuccessful or uncertified solves retain status/attempts rather than masquerade
as certified optima. On simultaneous coverage, for any comparator a,
$$f(a)-f(a_{\rm LCB})\le 2r(a)+\epsilon_{\rm LCB}.$$
For an approximate repository truth reference a_ref with gap epsilon_ref,
$$R_{\rm LCB}\le 2r(a_{\rm ref})+\epsilon_{\rm LCB}+\epsilon_{\rm ref}.$$
Reported regret is bracketed by the value difference to a_ref and that difference
plus epsilon_ref. The plug-in bound is r(a_ref)+r(a_plugin)+epsilon_plugin
+epsilon_ref. These certificates are conditional on containment; a failed
containment check is not repaired by an optimizer success flag. Pure positive
sigma scaling preserves the auxiliary coverage event and scales w linearly;
the fixed predictor's discrepancy means neither total r nor regret need scale
linearly or monotonically. Increasing N need not remove the discrepancy floor.

Sources: `src/experiments/offline_bootstrap.py`,
`src/experiments/offline_bootstrap_math.py` and their reporting companion.

### 7.8 Uncertainty-envelopes: interchangeable construction and evaluation

The `uncertainty_envelopes` manifest separates truth, a frozen predictor,
envelope construction, containment checking, and probability evaluation.
For any nonnegative finite radius, the common diagnostics are
$$b(x,a)=|\widehat f(x,a)-f(x,a)|,\qquad s(x,a)=r(x,a)-b(x,a).$$
Signed excess s is never clipped. Means and quantiles use explicitly saved
finite diagnostic locations and weights, not an implicit uniform law on R.
Finite diagnostic locations do not determine the continuous coverage domain.

The initial adapter has phi(x,a)=(1,x',a)', p=d+2, full-column-rank
P with rows phi(x_i,a_i)', and Q=P'P. Conditional on the frozen logged
design Z, errors have zero means, zero cross-covariances, and variances at
most the known sigma_bound squared. The truth equals phi'beta0 on the target.
QR solves produce the auxiliary OLS fit beta_tilde; no residual variance is
substituted for the known variance bound. The construction is
$$w(x,a)^2=\sigma_{bound}^2(p/\delta)\phi(x,a)'Q^{-1}\phi(x,a),
\qquad r(x,a)=|\widehat f(x,a)-\widetilde f(x,a)|+w(x,a).$$
With e=beta_tilde-beta0, E[e'Qe|Z]<=p sigma_bound^2. Markov and
Cauchy--Schwarz imply simultaneous auxiliary containment with probability
at least 1-delta, hence transferred-envelope containment. This is a lower
bound, not an exact probability. On auxiliary containment,
$$0\le r-|\widehat f-f|\le 2w.$$
At logged rows, sum_i phi_i'Q^{-1}phi_i=p, providing a normalization check.
Context dimension enters p and leverage, without an additional union bound.

The one-off linear dimension sweep keeps sigma and delta fixed and adds
zero-coefficient standard-normal covariates to both truth and predictor.
The explicit square-root-p factor does not describe all dimension dependence:
leverage is typically of order p/n, giving auxiliary width of order
p/sqrt(n). Thus fixed n and n proportional to p have different scaling.
On the logged design rows the exact identity is
$$\sqrt{\frac1n\sum_i w_\delta(X_i,A_i)^2}
  =\frac{\sigma p}{\sqrt{n\delta}}.$$
This RMS identity is a numerical check, not an equality for arithmetic mean
width on the context-by-action plotting grid. That mean is computed directly
using the repository's fitted envelope. The total radius also contains the
absolute predictor/auxiliary discrepancy. Square-root-p reference curves
are anchored at p=5 rather than fitted to the sweep. The sweep uses
n=12p with p=5,10,...,100; p=d+2. The original three-dimensional
signal is retained, and all added coefficients are zero.

The built-in optional width_multiplier multiplies w, not the discrepancy.
Multipliers at least one retain this bound; smaller multipliers are allowed
only without claiming this analytical guarantee. A constant-radius adapter
and user-provided builders support evaluation without inheriting the theorem.

For a literal constant radius c>=0 and affine truth/predictor, write their
difference on each fixed context as h_i(a)=u_i+v a. All-real containment
holds exactly when v=0 and |u_i|<=c for every context; if v!=0, the absolute
error eventually exceeds every finite c. For a declared bounded interval
[L,U], checking |h_i(L)|<=c and |h_i(U)|<=c certifies the entire interval,
because absolute affine error is convex. These are containment decisions,
not optimizer actions or choices of an optimal radius. In the supplied
synthetic example the slopes are 1.7 and 2.0, so literal c=3 fails on R even
when it covers a plotted bounded window. A fixed constant radius does not
depend on the calibration responses: conditional on fixed truth, predictor,
and design, its coverage probability is zero or one on the declared domain.
Truth-based verification of that event is separate from the writeup's
distributional lower bound of 1-delta for its data-dependent construction.

For the paired coverage display, decompose the additive contextual truth as
$$f(x,a)=g(x)+h(a),\qquad
g(x)=\beta_0+\beta_x'x,\qquad h(a)=\beta_a a.$$
At every saved diagnostic action, subtract the known synthetic context term
g(x_i) from each context's saved truth, predictor, and envelope endpoints.
The adjusted intervals and their intersection across all target contexts are
$$I_i(a)=[\widehat f(x_i,a)-g(x_i)-r(x_i,a),\,
          \widehat f(x_i,a)-g(x_i)+r(x_i,a)],$$
$$L(a)=\max_i\{\widehat f(x_i,a)-g(x_i)-r(x_i,a)\},\qquad
  U(a)=\min_i\{\widehat f(x_i,a)-g(x_i)+r(x_i,a)\}.$$
The common interval is empty when L(a)>U(a). Otherwise the common true response
h(a) lies inside [L(a),U(a)] if and only if every context's original envelope
covers its truth at that action. Taking endpoint maxima/minima here is set
intersection across contexts, not action optimization or radius selection.
The shaded intersection is not an individual context's radius or envelope,
and its width must not be interpreted as twice a representative radius.
The mean adjusted frozen predictor
$$\overline m(a)=\frac1n\sum_{i=1}^n[\widehat f(x_i,a)-g(x_i)]$$
provides orientation only; neither coverage nor the intersection endpoints are
defined relative to that mean. The g adjustment uses truth only in evaluation,
never in fitting or construction. In the supplied example g(x)=x_1-0.5x_2+0.25x_3
and h(a)=2a. Its frozen predictor is hand specified; the auxiliary OLS model is
the fitted object. This view reuses saved responses, fits, radii, and diagnostic
locations without refitting. Connected finite diagnostic values visualize the
intersection; continuum guarantees still come from the separate certificates.

The paired `02_radius_excess.pdf` uses that same context collapse. For the
nonempty common interval, its midpoint and half-width are
$$M(a)=(L(a)+U(a))/2,\qquad H(a)=(U(a)-L(a))/2.$$
The collapsed signed excess is
$$S(a)=\min\{h(a)-L(a),\ U(a)-h(a)\}
      =\min_i\{r(x_i,a)-|\widehat f(x_i,a)-f(x_i,a)|\}.$$
For nonempty intervals, S=H-|M-h|; for empty intervals the endpoint-margin
formula still gives negative S, while H is masked rather than shown as a
negative radius. Zero S includes boundary coverage. Thus S>=0 exactly when
every context is covered at that diagnostic action. Neither curve averages
over contexts; H is the common band's half-width, not an individual radius.
Original pointwise radius/excess summaries remain unchanged. Endpoint and
margin reductions aggregate contexts only and do not select an action optimum.

The optional `04_simple_comparison.pdf` uses the original saved pointwise
diagnostics directly, without context adjustment or interval intersection:
$$C(a)=\min_{1\le i\le n}s(x_i,a),\qquad
  \overline r(a)=\frac1n\sum_{i=1}^n r(x_i,a).$$
Here C equals S above, but its definition does not require an additive truth.
C>=0 means all saved contexts are covered at that diagnostic action; negative
C means at least one is missed. The mean radius uses equal context weights
and equals 3 for the constant-3 construction. These are diagnostic curves for
one saved realization, not estimates of coverage probability or a replacement
for continuum certificates. Lines connecting their samples are visualization.

Analytical runs use one saved calibration realization and a separately typed
probability statement. Empirical runs freeze Z and the predictor and redraw
independent response vectors; each repetition refits the entire construction.
Any method randomness is independent of response randomness. Inner bootstrap
resamples, when supplied by an adapter, are not outer coverage repetitions.
Unknown containment and failed fits remain in counts; no empirical point
estimate is emitted unless all requested repetitions are decisively evaluated.

All-real linear containment is decided from polynomial signs on each observed
context fiber. It certifies the mathematical polynomial defined by stored
floating-point fit coefficients/covariance interpreted as exact rationals;
it does not enclose floating-point OLS or evaluation error. Failure of the
coefficient sufficient event does not imply failed envelope containment.
Finite-target checking makes only a finite-set floating-point claim.

Sources: `src/experiments/envelopes/`; usage: `docs/uncertainty_envelopes.md`.

For the calibration action-support display, plot the same common-overlap
band above, with observed responses adjusted as Y_i-g(x_i). These are noisy
observations, not the truth values tested by mean-response coverage. All axes
share the existing [-2,2] diagnostic window; counts explicitly record the
calibration actions outside it. Histogram bins count logged actions only and
do not affect fitting or radius evaluation. At each saved diagnostic action,
plot arithmetic means across all target contexts of r_i(a), w_i(a), and
D_i(a)=|fhat_i(a)-ftilde_i(a)|, so mean(r)=mean(w)+mean(D). The common-overlap
band is an intersection and its half-width is not mean(r).

The analytical width uses global joint-design leverage, not inverse local
histogram density. Write B=[1,X], b=(1,x')', gamma=(B'B)^(-1)B'a_logged,
and S=||a_logged-B gamma||^2. Block inversion of Q gives
$$phi(x,a)'Q^{-1}phi(x,a)
 = b'(B'B)^{-1}b + (a-b'gamma)^2/S.$$
This explains wider uncertainty away from the design's fitted action trend;
the histogram alone cannot determine leverage. Display averages replay saved
width/discrepancy/radius arrays without new observations, model fits, or PCA.

The reusable per-experiment diagnostic reports replay the original coordinates
without subtracting context effects or intersecting intervals. For repetition
j and target context x_i, define simultaneous action coverage on the declared
coverage domain A by
$$c_{ji}=\mathbf{1}\{\forall a\in A:\;
 |\widehat f_j(x_i,a)-f(x_i,a)|\le r_{\delta,j}(x_i,a)\},\qquad
 \overline c=\frac{1}{Jn}\sum_{j=1}^{J}\sum_{i=1}^{n}c_{ji}.$$
The coverage bar uses saved containment certificates, or all declared locations
for a finite target, rather than checking only the diagnostic action window.
Unknown, failed, and missing evaluations remain in the requested denominator;
an exact value is shown only when every c_ji is known. This is the average
fraction of contexts whose entire action domain is covered, not the joint
all-context event $\prod_i c_{ji}$. A single saved realization is not a coverage
probability estimate. Any repeated-response interpretation retains the original
fixed-design sampling contract and is distinct from an analytical lower bound.

Mean and slice curves use one saved successful repetition (the first by default,
or an explicit repetition index). For that repetition, arithmetic context means
at each saved diagnostic action are
$$\overline f(a)=\frac1n\sum_i f(x_i,a),\qquad
 \overline{\widehat f}(a)=\frac1n\sum_i\widehat f(x_i,a),\qquad
 \overline r_\delta(a)=\frac1n\sum_i r_\delta(x_i,a).$$
The mean envelope is $[\overline{\widehat f}-\overline r_\delta,
\overline{\widehat f}+\overline r_\delta]$. Its containment of $\overline f$
does not imply containment at every context. These means use equal context
weights, not the flattened context/action diagnostic weights and not averages
across response repetitions. Where native transfer components are saved,
$$\overline D(a)=\frac1n\sum_i
 |\widehat f(x_i,a)-\widetilde f(x_i,a)|,\qquad
 \overline w_\delta(a)=\frac1n\sum_i w_\delta(x_i,a),\qquad
 \overline r_\delta(a)=\overline D(a)+\overline w_\delta(a).$$
The absolute discrepancy is averaged before summing with the auxiliary width;
it is not the absolute difference of the mean fitted functions. A directly
specified constant radius c has no native auxiliary fit or such decomposition.
An optional matched analytical project's auxiliary fit may be shown as an
explicit external reference in its slices, but must not be used to decompose c.
Matching requires identical saved contexts, diagnostic actions, truth, center,
and calibration data; it consumes no randomness and fits no additional model.

Context slices select the lowest, lower-median, and highest ranks of
$\|x_i\|_2$ with stable ties resolved by the original row index. The zero-based
median rank is $\lfloor(n-1)/2\rfloor$; duplicate selected indices are removed
for small n. For standard-normal contexts, this is a centrality display, not a
claim that radius or coverage difficulty is monotone in the norm. Each slice
shows unadjusted truth, predictor, available auxiliary fit, and predictor plus
or minus its own radius. Connected saved action samples are visualization only;
all-real or other continuous-domain claims still use the saved checker. The
reports introduce no stochastic process, optimizer, interpolation rule, or
new confidence guarantee. Saved report data records source provenance,
repetition selection, action window, selected rows, and the plotted arrays.

### 7.9 Fixed-design wild bootstrap with a bounded GP variance profile

The `wild_bootstrap_envelopes` study compares thesis sections 1.2 and 1.3
using the same contextual affine truth, frozen predictor, and auxiliary OLS
fit. Hold p=d+2 fixed, draw the largest design once, and use row prefixes
across n. All coverage events include every calibration context and every
real action. The diagnostic action window [-2,2] only defines displays and
equal-weight mean widths, never calibration or containment.

The independently seeded environment is one exact finite-rank Gaussian
process on t=a+c'x:
$$G(t)=m^{-1/2}\sum_{k=1}^m[A_k\cos(\omega_k t)+B_k\sin(\omega_k t)],
\quad A_k,B_k\stackrel{iid}{\sim}N(0,1),$$
with deterministic frequencies from `FiniteFourierGPSpec`. Its covariance is
$m^{-1}\sum_k\cos(\omega_k(t-s))$; it is not an interpolated squared-exponential
GP. At every real query use this same Fourier formula and
$$\sigma(x,a)=\sigma_{\min}+(\sigma_{\max}-\sigma_{\min})
\operatorname{expit}(\eta G(a+c'x)),\qquad
Y_i=\phi_i'\beta_0+\sigma(x_i,a_i)z_i,\quad z_i\stackrel{iid}{\sim}N(0,1).$$
Condition on the entire saved design and GP profile. Responses are independent
under that conditioning, with zero-mean errors and variances bounded globally
by sigma_max squared. The known cap does not come from a sampled grid maximum.
The environment, contexts, actions, observations, and wild multipliers have
distinct seed streams; n prefixes pair observations, B prefixes pair
multipliers, and all delta values and methods share the same fitted data.

An optional `noise.moved_observation_sigma=s_j` overrides only the observation
SD of row j=`design.action_override.index`. For moved action A,
$$Y_j(A)=\phi(x_j,A)'\beta_0+s_j z_j,\qquad 0<s_j\le\sigma_{\max}.$$
Every other row retains its GP-based scale. The same z vector is paired across
locations, with fresh independent vectors across response repetitions. This is
a row-level observation override, not a change to the continuous GP profile or
its off-grid evaluation. The fixed-noise leverage experiment takes s_j from
that row's original pre-move saved scale (see `docs/residual_leverage_fixed_noise.md`).

For new runs, set R=Y-fhat at the logged rows and target g=f-fhat.
Let Q=P'P, V=Q^{-1}, beta_tilde=Q^{-1}P'R, e_tilde=R-P beta_tilde, and
ell_P(x,a)=sqrt(phi'V phi). Section 1.3 uses raw residuals (no HC2/HC3
adjustment) and independent Rademacher multipliers:
$$\beta_b^*-\widetilde\beta=Q^{-1}P'(\widetilde\epsilon\odot\xi_b),
\qquad T_b^*=\sup_{i,a\in\mathbb R}
\frac{|\phi(x_i,a)'(\beta_b^*-\widetilde\beta)|}{\ell_P(x_i,a)}.$$
Take order statistic ceil(B*(1-delta)) of the first B statistics, with no
additional Monte Carlo inflation (the historical NumPy higher rule differs
for some B and delta). The width is q_hat ell_P; the analytical
width is sigma_max sqrt(p/delta) ell_P. No noise standard error is estimated.
Both use residual radii r_lower=[w-g_tilde]_+, r_upper=[w+g_tilde]_+.
The residual-band event is |g_tilde-g|<=w everywhere; the transferred event
is -r_lower<=g<=r_upper everywhere. Save and measure both separately.
Historical artifacts without band_target=residual retain their function-target
fit and symmetric transfer; they cannot be resumed under the new source contract.

The all-real statistic is the norm of coefficient projections: with
H_i=[(1,x_i',0)',e_a], M_i=H_i' V H_i and h_i=H_i'e,
$$T(e)^2=\max_i h_i'M_i^{-1}h_i.$$
Equivalently its threshold event is that q^2 M_i-h_i h_i' is positive
semidefinite for every i. This includes both action tails; no sampled action
grid defines the statistic. These are coefficient-norm/containment calculations,
not selection of a downstream optimizer action. This study reports no pricing
optimum, optimizer shift, or regret. Numerical boundary checks must not turn
an ambiguous floating-point decision into a claimed exact certificate.

One outer repetition supplies one joint auxiliary-coverage indicator and one
joint transferred-envelope indicator. Report binomial Wilson intervals across
independent response/multiplier repetitions at the frozen environment; never
substitute context-averaged coverage. Increasing B reduces quantile Monte Carlo
error only. Increasing n at fixed p can improve bootstrap approximation under
regularity (including vanishing maximum leverage); neither finite-n coverage
nor monotone convergence is guaranteed. The analytical method retains the
finite-sample lower bound 1-delta, while bootstrap validity remains asymptotic
and conditional on the required approximation. Simulated performance is not
a proof of that approximation. The saved first repetition supplies slices;
it is not selected for favorable coverage.

Sources: `src/experiments/envelopes/wild.py`,
`src/experiments/envelopes/wild_study.py`, and its reporting companion.

### 7.10 Truncated-normal leverage and wild-bootstrap coverage

The fixed-n leverage diagnostic specializes section 7.9 to no contexts,
$$\phi(a)=(1,a)^\mathsf T,\quad f(a)=1+a,\quad
\widehat f(a)=1.2+0.8a,\quad Y_i=f(a_i)+0.5Z_i,$$
where the $Z_i$ are independent standard normal observation errors. For each
bound $c\in\{0.5,1,3,10\}$, generate $n=300$ logged actions by
$$a_i^{(c)}=F_{[-c,c]}^{-1}(U_i),\qquad U_i\stackrel{\mathrm{iid}}\sim U(0,1),$$
where $F_{[-c,c]}$ is the standard-normal CDF conditioned on $[-c,c]$.
The actions seed supplies one shared vector of uniforms across bounds. Freeze
each design; reuse observation innovations and independent Rademacher draws
across bounds, with independent seeds for each outer repetition. No centering
or standardization is applied. Context and profile seeds are unused in this
special case, retained only for compatibility with the existing seed schema.

The wide-bound extension uses $c\in\{0.5,1,10,100,1000,\infty\}$.
An explicit JSON `null` bound represents $c=\infty$, with
$F_{[-\infty,\infty]}^{-1}=\Phi^{-1}$. It uses the same uniforms and seed
streams as finite bounds. Normal location and scale remain 0 and 1;
increasing the truncation bound does not rescale the underlying distribution.
Existing finite-bound artifacts may be replayed with their original provenance.
The original standard-normal RNG path is unchanged for other studies.

With $S_{aa}=\sum_i(a_i-\bar a)^2$, observed leverage is
$$h_i=1/n+(a_i-\bar a)^2/S_{aa},\qquad h_n=\max_i h_i.$$
Each response dataset contributes one simultaneous auxiliary coverage event
$$\mathbf1\{T(\widetilde\beta-\beta^\star)\le\widehat q_{0.95}\},$$
using the existing all-real-action statistic and raw-residual bootstrap.
For $B=1999$, NumPy's higher empirical quantile at 0.95 selects order
statistic 1900, equal to $\lceil B(1-\delta)\rceil$ for $\delta=0.05$.
The diagnostic uses 2000 response/bootstrap repetitions per fixed design and
pointwise 95% Wilson intervals for each coverage frequency. Coverage concerns
all $a\in\mathbb R$, not a plotting grid; exact affine functions define all
off-grid queries. Numerical coefficient statistics retain section 7.9's
floating-point limitations. There is no stochastic objective surface,
downstream optimization, or reported optimizer action in this diagnostic.

The plot compares the four realized designs, not a universal coverage function
of $h_n$. Maximum leverage bounds bootstrap approximation error; it does not
determine that error or guarantee monotonic empirical coverage. Constant noise,
sample size, feature count, bootstrap count, and confidence level remove those
sources of variation. The target action domain remains the same across bounds.

### 7.11 One moved observation with a fixed truncated-normal bulk

Use the same 300 inverse-CDF draws from $N(0,1)$ conditioned on [-3,3]
and replace the preselected row 0 action with
$$t\in\{3,10,20,40,100,1000\}.$$
All other 299 actions stay exactly fixed. This is a deterministic design
modification, so it introduces no new random process or seed stream. The
truth, frozen predictor, independent constant-variance Gaussian observation
errors, bootstrap scheme, 2000 repetitions, and all-real coverage event remain
as in section 7.10. Every condition shares observation innovations and wild
multipliers; the moved row retains its own response noise and its response mean
is recomputed from the same true linear model. It is a high-leverage design
row, not a response contamination or model misspecification.

Let $m=n-1$, let $\mu$ be the fixed bulk mean, and let
$S=\sum_{i\ne0}(a_i-\mu)^2>0$ be its centered sum of squares. Define
$u=t-\mu$. The full design's centered sum of squares and moved-row leverage are
$$S_{aa}=S+\frac{m}{n}u^2,\qquad
h_0(t)=\frac1n+\frac{(m/n)^2u^2}{S+(m/n)u^2}.$$
Therefore $h_0(t)\to1$ as $|t|\to\infty$. All observed leverages are diagonal
entries of the orthogonal projection $H=X(X^\mathsf TX)^{-1}X^\mathsf T$,
so $0\le h_i\le1$ and $h_n=\max_i h_i\le1$. Values 2 or 10 are impossible
for the document's definition. The observed maximum is computed over all rows;
the moved row is not assumed to be the largest before checking.

The primary diagnostic remains simultaneous auxiliary-band coverage against
the actual $h_n$, with pointwise 95% Wilson intervals. The full transferred
envelope coverage is retained separately. Under constant noise variance,
$$\mathbb E[\widetilde\varepsilon_i^2\mid X]=\sigma^2(1-h_i),$$
which explains why a highly influential row has a suppressed raw bootstrap
residual. No leverage adjustment, target-grid approximation, or downstream
optimizer is introduced. The supplied outlier locations are prescribed design
conditions, not optimized actions. This experiment conditions on one bulk
realization; it does not establish a universal coverage curve determined only
by maximum leverage.

### 7.12 Quadratic residual bands

`wild_bootstrap_envelopes` accepts `design.action_degree=2` for zero-context
quadratic fits, alongside the existing contextual affine basis. The supplied
predictor remains fixed; subtract its values before fitting or bootstrapping.
The quadratic manifest fixes f(u)=5u-5u² and the saved independent predictor
coefficients [0.10142434006958467, 4.824476863339917, -5.127996631955374].
The residual coefficients are truth minus predictor coefficients.

Use QR OLS, phi=(1,u,u²), and the existing QuadraticErrorCertificate with
sigma_hat=1: the denominator is sqrt(phi'V phi), without a second noise scale.
Each bootstrap statistic uses the certified upper endpoint of a supremum
bracket with hi-lo <= 2*tol*max(1,hi), tol=1e-8. Calibration and containment
cover the entire real line including both tails. Values at off-grid inputs
use the exact polynomial formulas, never interpolation. This is a coefficient
statistic, not a downstream optimizer action or an optimization reference.

For g_tilde=phi'beta, the signed residual g, and w²=q² phi'V phi, write
 d=g_tilde-g. Auxiliary containment is w²-d²>=0 for every real u.
Transferred containment fails exactly when either strict conjunction is feasible:
(-g>0, d>0, d²-w²>0) or (g>0, -d>0, d²-w²>0).
Use the existing exact rational polynomial sign-cell checker for these events;
do not use grid coverage or the old symmetric-radius containment checker.
The affine case uses the same signed inequalities on each context fiber.

Saved artifacts include residual coefficients, residual and transferred coverage,
both radii and signed margins, the fixed predictor, and target metadata. The
asymmetric function band is [fhat-r_lower,fhat+r_upper]. Reporting preserves
historical function-target artifacts based on their saved target metadata.

Representation/implementation checklist:
- [x] Retain the existing data, multiplier, profile, and repetition seed streams;
  subtracting predictor values and changing features introduces no random process.
  Designs are fixed per condition; n and B use shared prefixes; delta and moved
  positions reuse the same draws. Observation noise is independent conditional
  on the frozen design/profile. Fits share coefficients across queried inputs.
- [x] Reuse wild.py geometry, wild_study.py launch/provenance, polynomial
  certificates, and wild_reporting.py rather than adding another runner.
- [x] Keep old symmetric helpers for explicitly historical replay. New residual
  runs use a distinct source contract and result names, never relabel old fits.
- [x] Check residual refits against independent least squares, signed all-real
  containment including tail/equality cases, and a quadratic manifest/replay.
- [x] Update public API docs and verify saved plots show the asymmetric bounds.

Theoretical coverage is the thesis's conditional bootstrap-approximation bound,
not an unconditional finite-sample 95% assertion. High leverage can suppress
raw fitted residuals; increasing B does not repair that design failure.

### 7.13 Paired calibration-size and bootstrap-count sweep

The heteroskedastic residual experiment can use `sweep_mode: paired` to run
only the zipped `(n_values, B_values)` conditions. The approved schedule is
(30,99), (60,199), (120,499), (300,999), (600,1999), (1200,3999), with 2,000
response repetitions and delta in {0.10,0.05,0.01}. Both axes strictly increase;
the baseline (300,999,0.05) is one of those conditions. This measures their
combined effect, without identifying separate effects of n and B.

Retain §7.9's d=3 normal design, contextual affine f and fixed predictor,
shared finite-Fourier variance profile, and residual target g=f-fhat. All
queries use their exact formulas, with no interpolation. Random inputs at
different actions share the same fitted coefficients and frozen variance profile;
observation innovations are independent conditional on that profile. Existing
seed roots 1101–1105 remain unchanged. Every n uses a prefix of one maximum
size design and innovation vector; every B uses the matching prefix of the
same maximum-shape Rademacher matrix. No new stochastic process is introduced.

The analytical OLS comparator is sigma_max sqrt(p/delta) ell, which remains
valid under the known heteroskedastic variance cap. Both methods fit OLS to R;
they differ in calibration of w. Coverage remains simultaneous over all n
calibration contexts and every real action, checked by the existing coefficient
and polynomial calculations, with their existing roundoff qualifications.
Increasing n also enlarges this target set. Neither coverage nor interval
width is required to change monotonically, and larger B reduces quantile
Monte Carlo error rather than repairing finite-n bootstrap bias.

For interpretation, save the first response repetition at every pair and
select the central, median-norm, and outer context from the smallest-n prefix,
using the existing deterministic selector. Hold these exact contexts fixed
across pairs, without selecting on fit or coverage. Show residual bands and
transferred function envelopes separately. A 101-action grid on [-2,2] renders
these slices and width summaries; it never determines coverage or an optimizer
action. There are no optimizer queries or reported optima in this experiment.

Implementation/validation checklist:
- [x] Reuse the manifest runner, Slurm launcher, seed streams, and certificates;
  omit off-pair conditions and mark uncomputed array slots explicitly.
- [x] Keep grid mode compatible; test paired-prefix replay and reject mismatched
  axis lengths/baselines before sampling.
- [x] Save shared-context slices and show OLS versus wild bootstrap for each
  pair; verify plot inputs against saved bounds and regenerate PDF reports.
- [x] Regenerate public docs, run the paired sweep, and analyze saved summaries.

For the context-averaged display, retain response repetition 0 and average
over all n calibration contexts at each pair. Writing $\langle h\rangle_n(a)
=n^{-1}\sum_{i=1}^n h(x_i,a)$, the residual band is
$$[\langle\widetilde g-w\rangle_n(a),
   \langle\widetilde g+w\rangle_n(a)],$$
and the transferred envelope is
$$[\langle\widehat f-[w-\widetilde g]_+\rangle_n(a),
   \langle\widehat f+[w+\widetilde g]_+\rangle_n(a)].$$
Transfer each context's band before averaging: positive-part clipping does not
commute with averaging. These empirical-context averages inherit containment
on the original simultaneous-coverage event; no new calibration is performed.
They are neither bands evaluated at the mean context nor averages over response
repetitions. Replay saved coefficients and critical values on the existing
action grid, introducing no random draws or optimizer queries.

### 7.14 Finite-sample wild-bootstrap coverage reference

Replay the same paired residual sweep and its existing joint coverage counts.
The thesis `uncertainty-envelope.tex`, equations
`wild-bootstrap-main-coverage` and `wild-bootstrap-rho-tau`, gives
$$L_{n,B,\delta}=1-\delta-\rho_n-\tau_n-\frac{1-\delta}{B+1},
\qquad L^{\mathrm{order}}_{n,B,\delta}
=\frac{\lceil B(1-\delta)\rceil}{B+1}-\rho_n-\tau_n\ge L_{n,B,\delta}.$$
Both the residual band and transferred envelope have this same lower bound;
transfer preserves the residual-containment event. Use the expanded $L$ for
the requested plotted reference and retain both versions in the numeric table.
For the existing independent Gaussian observation errors,
$E[\varepsilon_i^4\mid\mathcal Z]=3\sigma_i^4$; thus the manifest's global
scale bounds give $K=3^{1/4}\sigma_{\max}$ and
$\underline\sigma=\sigma_{\min}$ (equal to $\sigma_{\max}$ for constant noise).
These are known model bounds, not estimated residual moments. With feature
dimension $p$ and observed-design maximum leverage $h_n$,
$$\kappa=K/\underline\sigma,\quad c_p=42p^{1/4}+16,\quad
d_p=\sqrt p+(2\cdot3^{1/4}+\sqrt3)p,\quad
C_{p,\kappa}=7c_pp\kappa^3+2d_p\kappa^2,$$
$$\rho_n=2^{1/3}C_{p,\kappa}^{2/3}h_n^{1/3},\qquad
\tau_n=2^{-2/3}C_{p,\kappa}^{2/3}h_n^{1/3}.$$
Verify saved $h_n$ against the frozen design's QR leverage. Preserve raw terms
even when they exceed one; plot $\max(0,L)$ as the probability lower bound.
For the completed sweep, $p=5$, $\sigma_{\min}=0.15$, $\sigma_{\max}=1$ and
$C_{p,\kappa}\approx1.867\times10^6$: every raw bound is negative. This is a
vacuous guarantee, not evidence of a finite-sample violation or a numerical
coverage estimate. No new draws, fitting, optimization, or coverage checks are
needed. Keep the same three deltas, paired sizes, and 95% Wilson intervals;
omit the analytical comparator from these two replayed coverage figures.

### 7.15 Paired distribution perturbations

Keep $g=f-\widehat f$, the affine features, raw-residual Rademacher bootstrap,
and paired $(n,B)$ schedule unchanged. Transform the original independent
standard-normal streams $Z$ (CDF $\Phi$) deterministically:

* Noise variant: $\xi=-\log\Phi(-Z)-1\sim\mathrm{Exp}(1)-1$ and
  $\epsilon_i=\sigma(X_i,A_i)\xi_i$. Mean, variance, third and fourth central
  moments are $0,1,2,9$, so $K=9^{1/4}\sigma_{\max}$ in Section 7.14,
  replacing the Gaussian $3^{1/4}\sigma_{\max}$.
* Design variant: independently transform every context coordinate and logged
  action to $F_{t_3}^{-1}(\Phi(Z))/\sqrt3$. Mean zero, variance one, infinite
  fourth moment. Observation innovations stay normal. The fixed-design theorem
  requires full column rank and controls leverage; it needs no fourth moment
  for the design distribution.

No clipping or empirical normalization is used. Evaluate t quantiles using the
smaller Gaussian tail and symmetry, and exponential innovations with
`log_ndtr(-Z)`. Existing context, action, observation, profile and multiplier
seed streams retain their roles; no additional random process is introduced.
The variants reuse normal ranks, prefixes and multipliers. Each law freezes one
design; 2,000 repetitions vary responses and bootstrap draws. The same frozen
Fourier GP scale function is evaluated analytically at the new design points;
observation errors remain conditionally independent.

Coverage is joint over calibration contexts and every real action, determined
by coefficient checks. The window $[-2,2]$ is only for rendering/width summaries,
with no interpolation in the proof. No objective optimization is involved.
Use $\max\{0,1-\delta-\rho_n-\tau_n-(1-\delta)/(B+1)\}$ with the appropriate
$K$, recording the untruncated bound too. No Gaussian-only refinement is used
for the exponential noise.

Implementation plan: extend the existing manifest runner's distribution choices;
verify unchanged Gaussian replay, paired transforms and moment constants with
small complete local studies. A post-collection replay compares Gaussian and
perturbed wild coverage with Wilson intervals and law-specific finite bounds.
Radius, context-slice and context-averaged figures contain the perturbed study
only, retaining OLS and wild overlays. Submit the two production arrays and
dependent report jobs after validation.

### 7.16 Distribution and mean-width diagnostics from saved repetitions

Replay the artifacts from Section 7.15 without generating observations or
bootstrap draws. Show the normal density against $e^{-(z+1)}1\{z\ge-1\}$
for standardized noise, and histogram the saved heteroskedastic errors
$\epsilon_i=\sigma_i\xi_i$ from response repetition 0. For the design variant,
the univariate source density is $\sqrt3\,p_{t_3}(\sqrt3 v)$. Compare its
density with the normal density and plot the empirical two-sided tail frequency
of all saved context coordinates and logged actions. The density window is a
visualization only; the empirical tail panel includes every saved observation.

For repetition $r$, pair $j=(n_j,B_j)$ and confidence parameter $\delta$,
reconstruct the original mean residual-band half-width exactly as
$$W_{r,j,\delta}=q^*_{r,j,\delta}\,\overline\ell_j,\qquad
\overline\ell_j=\frac{1}{n_j|G|}\sum_{i=1}^{n_j}\sum_{a\in G}
\sqrt{\phi(X_i,a)^\top(\Phi_j^\top\Phi_j)^{-1}\phi(X_i,a)},$$
where $G$ is the original 101-point action grid on $[-2,2]$, and both $q^*$
and $\overline\ell_j$ are saved artifacts. Average $W$ over the $R=2000$
repetitions; verify this reproduces `summary.metrics.mean_width`. These are
residual half-widths, not the asymmetric transferred radii.

Use the shared repetition seeds to form $D_r=W_r^{\mathrm{variant}}-
W_r^{\mathrm{Gaussian}}$. Report $\overline D$, relative change
$100\overline D/\overline W^{\mathrm{Gaussian}}$, and Monte Carlo standard
error $s_D/\sqrt R$. Pointwise approximate 95% intervals use
$\overline D\pm t_{R-1,.975}s_D/\sqrt R$. Also report Bonferroni-adjusted
intervals with quantile $t_{R-1,1-.05/(2m)}$, where $m$ counts all displayed
variant/pair/delta comparisons (36 for these two sweeps). Width curves use
pointwise mean intervals; paired-difference panels use the adjusted intervals.
These describe Monte Carlo precision conditional on the frozen designs and
profile; they are not the finite-sample band-coverage guarantee, nor uncertainty
over newly drawn designs. No optimizer is used.

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
