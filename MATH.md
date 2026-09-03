# Math Reference

Central reference for every mathematical formula implemented in the codebase.
Each entry lists the formula, the implementing source file and function, and
a brief note where helpful.

---

## 1. Shared Utilities

### 1.1 Numerically Stable Sigmoid

$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & z \ge 0 \\ \frac{e^z}{1 + e^z} & z < 0 \end{cases}$$

- **Source:** `src/objective/_math.py` :: `_sigmoid(z)`
- **Notes:** Two-branch form avoids overflow in `exp()`. Output is always in
  $(0, 1)$. Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

---

## 2. Policies

All policies map parameters $\theta$ and state $x$ to a scalar action $u$.

### 2.1 Feature Construction

Policies use a configurable state feature map $$\varphi: \mathbb{R}^d \to \mathbb{R}^q$$.
The policy layer prepends the intercept internally:

$$\phi(x) = [1,\; \varphi_1(x),\; \dots,\; \varphi_q(x)]$$

Therefore $$\theta \in \mathbb{R}^{q+1}$$. User-supplied feature maps return
only $$\varphi(x)$$; they should not include the leading intercept column.

Built-in feature maps:

$$\varphi_{\text{identity}}(x) = [x_1,\; \dots,\; x_d]$$

$$\varphi_{\text{quadratic}}(x) = [x_1,\; \dots,\; x_d,\; x_1^2,\; x_1x_2,\; \dots,\; x_d^2]$$

$$\varphi_{\text{cubic}}(x) = [x_1,\; \dots,\; x_d,\; x_i x_j x_k\; \text{for}\; 1 \le i \le j \le k \le d]$$

$$\varphi_{\text{quartic}}(x) = [x_1,\; \dots,\; x_d,\; x_i x_j x_k x_l\; \text{for}\; 1 \le i \le j \le k \le l \le d]$$

For a gradually nested, interaction-free capacity ladder, first define

$$t_j(x)=\operatorname{clip}\!\left(\frac{x_j}{s},-1,1\right),\qquad
T_0(t)=1,\quad T_1(t)=t,\quad T_k(t)=2tT_{k-1}(t)-T_{k-2}(t).$$

The degree-$$D$$ additive Chebyshev map is ordered by degree,

$$\varphi_{\mathrm{cheb},D}(x)=
[T_1(t_1),\ldots,T_1(t_d),T_2(t_1),\ldots,T_D(t_d)],$$

so it has $$dD$$ mapped features and the bounded policy has
$$1+dD$$ parameters including its intercept. The capacity experiment uses
$$s=3$$ after train-only standardization. It contains no feature interactions,
and degree $$D$$ is an exact prefix of degree $$D+1$$.

For a nested interaction-capable polynomial ladder, let

$$\mathcal A_D = \{\alpha\in\mathbb N_0^d:1\le |\alpha|_1\le D\},
\qquad x^\alpha=\prod_{j=1}^d x_j^{\alpha_j}.$$

The total-degree polynomial map is

$$\varphi_{\mathrm{poly},D}(x)=[x^\alpha:\alpha\in\mathcal A_D],$$

ordered first by total degree and then by deterministic
combinations-with-replacement order. It has
$$\binom{d+D}{D}-1$$ mapped features, so a linear or bounded policy has
$$\binom{d+D}{D}$$ parameters including its intercept. Degree $$D$$ is an
exact prefix of degree $$D+1$$ and contains every interaction whose total
degree is at most $$D$$.

- **Source:** `src/objective/policy.py` :: `FeatureMap`, `AdditiveChebyshevFeatureMap`, `TotalDegreePolynomialFeatureMap`, `IdentityFeatureMap`,
  `QuadraticFeatureMap`, `CubicFeatureMap`, `QuarticFeatureMap`,
  `CallableFeatureMap`, `_phi(x_batch, feature_map)`
- **Notes:** `IdentityFeatureMap` preserves the previous default behavior
  $$\phi(x) = [1, x]$$. Higher-order maps include linear terms plus exact-degree
  monomials. Interaction terms use deterministic combinations-with-replacement
  ordering; for degree $$k$$ the non-linear width is $$\binom{d+k-1}{k}$$.

### 2.2 Constant Policy

$$u = \theta_0$$

- **Gradient:** $\frac{\partial u}{\partial \theta} = [1, 0, \dots, 0]$
- **Source:** `src/objective/policy.py` :: `ConstantPolicy.value()`, `ConstantPolicy.grad()`, `ConstantPolicy.weighted_grad()`

### 2.3 Linear Policy

$$u = \theta^\top \phi(x)$$

- **Gradient:** $\frac{\partial u}{\partial \theta} = \phi(x)$
- **Source:** `src/objective/policy.py` :: `LinearPolicy.value()`, `LinearPolicy.grad()`, `LinearPolicy.weighted_grad()`

### 2.4 Softmax (Bounded) Policy

With lower action bound $$l$$ and upper action bound $$h$$:

$$u = l + (h-l)\,\sigma(\theta^\top \phi(x)) \;\in\; (l,\; h)$$

The default is $$l=-0.5$$ and $$h=0.5$$.

- **Gradient:** $\frac{\partial u}{\partial \theta} = (h-l)\,\sigma(z)(1 - \sigma(z))\;\phi(x)$
  where $z = \theta^\top \phi(x)$
- **Source:** `src/objective/policy.py` :: `SoftmaxPolicy.value()`, `SoftmaxPolicy.grad()`, `SoftmaxPolicy.weighted_grad()`

### 2.5 MLP (Two-Layer) Policy

A two-layer MLP with $\tanh$ activations and a bounded sigmoid head:

$$h_1 = \tanh(W_1\,\varphi(x) + b_1),\quad
h_2 = \tanh(W_2\,h_1 + b_2),\quad
z = W_3\,h_2 + b_3,\quad
u = 0.5 - \sigma(z)$$

with $W_1\in\mathbb{R}^{H\times d_{in}}$, $W_2\in\mathbb{R}^{H\times H}$,
$W_3\in\mathbb{R}^{1\times H}$. Theta is the row-major flat concatenation
$[\,W_1, b_1, W_2, b_2, W_3, b_3\,]$, so
$\dim(\theta) = d_{in}H + H + H^2 + H + H + 1$.

- **Gradient:** standard chain rule via reverse-mode through both layers, with
  $\partial u/\partial z = -\sigma(z)(1-\sigma(z))$ and
  $\tanh'(z_\ell) = 1 - h_\ell^2$ at each hidden layer.
- **Source:** `src/objective/policy.py` :: `MLPPolicy.value()`, `MLPPolicy.grad()`, `MLPPolicy.weighted_grad()`

### 2.6 Feature-Processed Policy

Wrapper that applies a saved `FeatureProcessor` to raw state $x$ before
delegating to an inner policy (Constant, Linear, Softmax, or MLP).

- **Source:** `src/objective/policy.py` :: `FeatureProcessedPolicy`

---

## 3. Objectives

### 3.1 Isotropic Quadratic Objective

For a configured parameter dimension $$d$$:

$$J(\theta) = \frac{1}{2}\|\theta\|_2^2 = \frac{1}{2}\sum_{j=1}^{d}\theta_j^2$$

**Gradient and Hessian:**

$$\nabla J(\theta) = \theta, \qquad \nabla^2J(\theta) = I_d$$

- **Source:** `src/objective/objectives/synthetic/ladder.py` ::
  `StronglyConvexQuadratic.isotropic(dim)`
- **Notes:** This is a direct theta-space objective and does not compose through
  a policy. It is 1-strongly convex and 1-smooth, with unique minimizer
  $$\theta^*=0$$ and minimum value $$J(\theta^*)=0$$. The required `x_batch`
  argument is ignored. It is the $$w^*=0,\, A=I$$ case of ladder rung 1
  (section 3.7.1), which is why the former standalone `QuadraticObjective` was
  folded into the ladder.

### 3.2 Fixed Regression Objective

$$f(u;\, x) = a(x, u)\,\bigl(\ell(x) - r(u)\bigr)$$

where:
- Acceptance: $a(x, u) = \sigma(\beta_1^\top x + \beta_2\, u)$
- Loss: $\ell(x) = \beta_3^\top x$
- Revenue: $r(u) = \beta_4\, u$

**Gradient w.r.t. $u$:**

$$\frac{\partial f}{\partial u} = \frac{\partial a}{\partial u}\,(\ell - r) - a\,\beta_4$$

where $\frac{\partial a}{\partial u} = a(1 - a)\,\beta_2$.

- **Source:** `src/objective/objectives/synthetic/fixed_regression.py` :: `FixedRegressionObjective`
  - `_value_batch()` — per-sample values
  - `_grad_u_batch()` — per-sample $\partial f/\partial u$

### 3.3 Planted Logistic Objective

$$L(u;\, x) = \log(1 + e^z) - p^*(x)\, z$$

where:
- $z = \alpha\, u + \beta^\top x + b$
- $z^* = \alpha\, u^* + \beta^\top x + b$
- $p^*(x) = \sigma(z^*)$

**Gradient w.r.t. $u$:**

$$\frac{\partial L}{\partial u} = \alpha\,\bigl(\sigma(z) - p^*(x)\bigr)$$

- **Source:** `src/objective/objectives/synthetic/planted_logistic.py` :: `PlantedLogisticObjective`
  - `_value_batch()` — uses `np.logaddexp(0, z)` for numerical stability
  - `_grad_u_batch()` — zero at $u = u^*$ by construction
- **Notes:** Convex in $u$. Known optimum $u^*$ is planted at construction.

### 3.4 Model-Based Objective

$$f(u;\, x) = a(x, u)\,\bigl(L(x) - (u + 1)\, p(x)\bigr)$$

where:
- $a(x, u) = p_{\text{accept}}(x, u)$ — acceptance from trained classifier
- $L(x)$ — loss term; by default $L(x)=\hat{Y}(x)$ from the loss model, while
  real-data configs with `loss_source="observed"` use $L(x)=Y_G_Loss$
- $p(x)$ — policy premium extracted from state column `premium_col`
- $(u + 1)\, p(x)$ — revenue (centered: $u = 0$ is baseline multiplier)

For GLM/linear artifacts with extractable coefficients, the implementation uses
the equivalent array formulas:

$$p_{\text{accept}}(x, u) = \sigma\bigl(\beta_0 + \beta_x^\top z_{\text{acc}}(x) + \beta_u^{\text{eff}} u\bigr)$$

$$\hat{Y}(x) = \gamma_0 + \gamma_x^\top z_{\text{loss}}(x)$$

where $$z_{\text{acc}}$$ and $$z_{\text{loss}}$$ are the artifact-preprocessed model
features. By default $$\beta_u^{\text{eff}}$$ is the extracted artifact coefficient;
GLM real-data configs may override it with `u_coef` for counterfactual acceptance
sensitivity sweeps. If coefficients cannot be extracted, the objective falls back
to the bundled estimator's `predict_proba` / `predict` methods. In observed-loss
mode the loss-model path is bypassed and `Y_G_Loss` must be present in the
real-data batch.

**Gradient w.r.t. $u$:**

$$\frac{\partial f}{\partial u} = \frac{\partial a}{\partial u}\,(L - (u+1)\,p) - a\, p$$

Acceptance derivative:
- **GLM direct acceptance (analytical):** $\frac{\partial a}{\partial u} = a(1-a)\;\beta_u^{\text{eff}}$
- **Legacy churn artifacts (analytical):** $\frac{\partial a}{\partial u} = -a(1-a)\;\beta_u^{\text{eff}}$
- **Per-policy XGBoost logit spline (analytical inside support):**
  $\frac{\partial a_i}{\partial u} = -q_i(1-q_i)S_i'(u)$
- **XGBoost (numerical):** central FD with $\epsilon = 10^{-4}$

**Per-policy XGBoost logit-spline acceptance:**

For each covered insurance policy $$i$$, the source XGBoost ensemble is evaluated
on the fixed action grid $$u_j=j/100$$ for $$j=0,\ldots,16$$. Its direct
acceptance output is converted to churn probability $$q_{ij}=1-a^{XGB}_{ij}$$,
then projected onto a non-decreasing sequence with weighted isotonic regression.
A cubic smoothing spline $$S_i(u)$$ is fitted to the clipped logits of that
sequence. Inside the fitted support $$[u_{min},u_{max}]=[0,0.16]$$:

$$q_i(u)=\sigma(S_i(u)), \qquad a_i(u)=1-q_i(u)$$

and therefore

$$\frac{\partial a_i}{\partial u}
=-\sigma(S_i(u))\bigl(1-\sigma(S_i(u))\bigr)S_i'(u)
=-q_i(u)(1-q_i(u))S_i'(u).$$

Below support, churn is held constant at $$q_i(u_{min})$$, so
$$\partial a_i/\partial u=0$$. Above support, churn uses the source artifact's
tangent rule

$$q_i(u)=\operatorname{clip}\left(q_i(u_{max})
+s_i^{max}(u-u_{max}),0,1\right),$$

so $$\partial a_i/\partial u=-s_i^{max}$$ while the tangent is unclipped and is
zero after clipping. The piecewise extension can be nondifferentiable exactly at
the support boundaries or clipping points; the implementation uses the interior
derivative at the boundaries and zero on clipped regions. The real-data preset
keeps policy actions within the fitted support.

**Per-policy monotone PCHIP XGBoost acceptance:**

For each covered policy $$i$$, the 20260728 artifact stores the coefficients of
a shape-preserving cubic Hermite interpolator $$P_i(u)$$ fitted to a
non-decreasing churn curve on a shared action grid. On the fitted support,

$$q_i(u)=P_i(u), \qquad a_i(u)=1-q_i(u),
\qquad u\in[u_{min},u_{max}].$$

The stored artifact is validated at its knots and within every interval: churn
must remain in $$[0,1]$$ and must be non-decreasing. Consequently acceptance is
bounded in $$[0,1]$$ and non-increasing on the fitted support. Its derivative is

$$\frac{\partial a_i}{\partial u}=-P_i'(u).$$

Below support, churn is held constant at $$q_i(u_{min})$$. Above support, it uses
the stored non-negative upper tangent $$s_i^{max}$$:

$$q_i(u)=\operatorname{clip}\left(q_i(u_{max})
+s_i^{max}(u-u_{max}),0,1\right).$$

Thus $$\partial a_i/\partial u=0$$ below support, and equals
$$-s_i^{max}$$ above support while the tangent is strictly inside $$(0,1)$$,
then zero after clipping. At the support boundaries the implementation uses the
interior derivative. The hierarchy preset constrains actions to $$[0,0.16]$$ and
rejects policies absent from the artifact.

**GLM/XGBoost policy-capacity experiment:**

For the shared 19-dimensional, train-standardized policy input $$z(x)$$, the
degree-$$D$$ policy is

$$u_{\theta,D}(x)=-0.1+0.3\,\sigma\!\left(
\theta_0+\sum_{k=1}^{D}\sum_{j=1}^{19}\theta_{kj}
T_k\!\left(\operatorname{clip}(z_j(x)/3,-1,1)\right)
\right).$$

There are no interactions and the parameter count is $$p_D=1+19D$$. Every fit
starts at $$\theta_0=-\log 2$$ and all other coefficients zero, which gives
$$u(x)=0$$ for every customer. For evaluator $$m\in\{\mathrm{GLM},\mathrm{XGB}\}$$,

$$J_m(\theta;S)=\frac1{|S|}\sum_{i\in S}
a_i^m(u_{\theta,D}(x_i))\left[L_i^m-(1+u_{\theta,D}(x_i))p_i\right],$$

$$\bar a_m(\theta;S)=\frac1{|S|}\sum_{i\in S}a_i^m(u_{\theta,D}(x_i)),$$

and L-BFGS-B minimizes the fixed-floor penalized training target

$$Q_m(\theta;S)=J_m(\theta;S)+10^6\left[
10^{-3}\log\!\left(1+\exp\!\left(
\frac{0.8787745289312372-\bar a_m(\theta;S)}{10^{-3}}
\right)\right)\right]^2.$$

Reported objective performance is the unpenalized $$J_m$$ (or profit $$-J_m$$).
The floor is fixed and is not a sweep axis. The XGBoost arm builds an
experiment-specific 31-knot raw query grid from $$-0.10$$ through $$0.20$$,
then applies the same smoothing-spline, isotonic, and PCHIP construction; the
policy bounds keep all evaluations inside that fitted support. This grid widens
the canonical spline range $$[0,0.16]$$, not the raw XGBoost training-data range:
the saved acceptance-training notebook reports observed $$U$$ from approximately
$$-0.1144$$ through $$0.4327$$ after its modeling filters. Those aggregate
endpoints do not establish dense conditional support for every customer profile.
In particular, tree predictions in sparsely observed tail/profile combinations
can be flat leaf-boundary values. The manifest must therefore set
`curve_cache.widened_xgb_tail_acknowledged=true`, and results outside
$$[0,0.16]$$ are interpreted as tail-sensitivity analysis rather than validated
empirical or causal extrapolation. Post-fit spline monotonicity and probability
bounds establish numerical shape constraints only.

- **Source:** `src/experiments/policy_capacity.py`,
  `manifests/policy_capacity_glm_xgb.json`

The full-customer analysis cache stores the same PCHIP exactly as shared-grid
cubic Hermite data rather than one Python polynomial object per customer. For
an interval $$[x_j,x_{j+1}]$$, let
$$t=(u-x_j)/h_j$$, $$h_j=x_{j+1}-x_j$$, stored knot values
$$y_j=P_i(x_j)$$, and stored knot derivatives $$m_j=P_i'(x_j)$$. Evaluation is

$$
P_i(u)=h_{00}(t)y_j+h_{10}(t)h_jm_j
       +h_{01}(t)y_{j+1}+h_{11}(t)h_jm_{j+1},
$$

where $$h_{00}=2t^3-3t^2+1$$, $$h_{10}=t^3-2t^2+t$$,
$$h_{01}=-2t^3+3t^2$$, and $$h_{11}=t^3-t^2$$. Differentiating these four
basis functions gives the cached analytical derivative. The representation is
mathematically the same PCHIP returned by the canonical fitter; only float32
storage introduces approximation, which the cache collector bounds against
fresh canonical fits for both values and derivatives. Tail equations remain
the ones above.

**Local price-sensitivity bucket score:**

For GLM sensitivity-bucket experiments, customers are ranked by local acceptance
sensitivity at the median observed historical action $$u_{ref}$$:

$$s_i = \left|\frac{\partial a(x_i, u_{ref})}{\partial u}\right| = |\beta_u^{\text{eff}}|\,a_i(1-a_i), \quad a_i = a(x_i, u_{ref})$$

Rows are split into low/medium/high tertiles by $$s_i$$. With no explicit
interaction terms between `U` and `X`, heterogeneity in this score comes from
where each customer sits on the logistic acceptance curve.

**Elasticity distribution over action values:**

For GLM elasticity-distribution diagnostics, elasticity is the signed local
acceptance derivative with respect to the centered action. The customer-by-action
matrix is

$$D_{ij} = \frac{\partial a(x_i, u_j)}{\partial u}$$

For direct-acceptance GLM artifacts,

$$D_{ij} = \beta_u^{\text{eff}}\,a_{ij}(1-a_{ij}), \quad a_{ij} = a(x_i, u_j)$$

For legacy churn-probability artifacts, the sign flips because
$$a = 1 - p_{churn}$$:

$$D_{ij} = -\beta_u^{\text{eff}}\,a_{ij}(1-a_{ij})$$

For bucket construction only, the absolute sensitivity score matrix is

$$S_{ij} = \left|\frac{\partial a(x_i, u_j)}{\partial u}\right| = |\beta_u^{\text{eff}}|\,a_{ij}(1-a_{ij}), \quad a_{ij} = a(x_i, u_j)$$

For saved-policy acceptance-grid diagnostics, the representative sensitivity
score averages absolute sensitivity over the simulated action grid:

$$s_i^{grid} = \frac{1}{m}\sum_{j=1}^{m}\left|\frac{\partial a(x_i, u_j)}{\partial u}\right|$$

The plotted average elasticity curve summarizes customers within each action
bin using signed derivatives:

$$\bar{D}(u_j) = \frac{1}{n}\sum_{i=1}^n D_{ij}$$

Selected fixed actions show the empirical cross-customer distribution of signed
$$D_{ij}$$ as histograms. Histogram x-axes are clipped for display by default at
the `0.5` and `99.5` percentiles and those clipping thresholds are marked on the
chart; CSV summaries retain the unclipped values.

**Delta-u by reference sensitivity diagnostic:**

For final real-data policy diagnostics, each estimator/customer point plots

$$\Delta u_i = \pi_\theta(x_i) - u_i^{\text{historical}}$$

against absolute local acceptance sensitivity evaluated at reference action
$$u_{ref}=0.08$$:

$$s_i(0.08) = \left|\frac{\partial a(x_i, u)}{\partial u}\right|_{u=0.08}$$

One aggregate sensitivity scatter and one $$\Delta u_i$$ histogram are written
per train/test split with all estimators overlaid.

**Expected profit contribution diagnostic:**

The per-customer objective contribution is $$M_i = f(\pi_\theta(x_i); x_i)$$.
Because the optimizer minimizes $$M$$, the reporting plot uses the sign-flipped
expected profit contribution

$$P_i = -M_i$$

so $$P_i > 0$$ means predicted money made on customer $$i$$ and $$P_i < 0$$
means predicted money lost. The plot shows the cross-customer distribution of
$$P_i$$ and a scatter of $$P_i$$ against predicted acceptance $$a(x_i,\pi_\theta(x_i))$$.

**Acceptance penalty** (smooth floor enforcement):

$$\text{penalty} = w \cdot \bigl[\tau\,\log(1 + e^{g/\tau})\bigr]^2$$

where $g = \text{floor} - \bar{a}(\theta)$ and $\tau$ is temperature.

$$\frac{\partial\,\text{penalty}}{\partial\,\bar{a}} = -2w\,\text{softplus}(g/\tau)\,\sigma(g/\tau)$$

- **Source:** `src/objective/objectives/generali/model_based.py` :: `ModelBasedObjective`
  - `_value_batch()` — per-sample values
  - `_glm_acceptance_proba()` — coefficient-backed GLM acceptance probability when available
  - `_grad_u_batch()` — per-sample $\partial f/\partial u$
  - `_d_acceptance_du_batch()` — analytical or FD acceptance derivative
  - `_acceptance_penalty()` — penalty value and gradient scale
- **Source:** `src/objective/objectives/generali/prepared_glm.py` :: `PreparedGLMObjective`, `PreparedGLMBatch`, `prepare_glm_objective()`
  - Uses the same GLM formulas after materializing `base_logit`, `loss`, `premium`, and policy features into a compact numeric batch.
- **Source:** `src/objective/objectives/generali/jax_prepared_glm.py` :: `JaxPreparedGLMObjective`, `JaxPreparedGLMScipyAdapter`, `prepare_jax_glm_objective()`
  - Uses the same prepared GLM formulas in JAX for fixed-batch SciPy callbacks. The explicit constraint-margin adapter uses $$\bar{a}(\theta) - \alpha$$, equivalent to SciPy's existing lower-bound form $$\bar{a}(\theta) \ge \alpha$$.
- **Source:** `src/experiments/sensitivity_buckets.py` :: `glm_price_derivative_matrix()`, `glm_price_sensitivity_scores()`, `glm_price_sensitivity_matrix()`, `split_sensitivity_tertiles()`
- **Source:** `src/reporting/visualization.py` :: `_plot_policy_delta_u_histograms()`, `_plot_policy_delta_u_by_elasticity()`, `_plot_policy_objective_contribution_summary()`

**Lagrangian scalarization** (lambda sweep path):

$$J_{\lambda}(\theta) = J(\theta) + \lambda\,(\text{floor} - \bar{a}(\theta))$$

where $$J(\theta) = \mathbb{E}[f(\pi_\theta(x); x)]$$ and
$$\bar{a}(\theta) = \mathbb{E}[a(x, \pi_\theta(x))]$$.

**Gradient w.r.t. $\theta$:**

$$\nabla_\theta J_{\lambda}(\theta) = \nabla_\theta J(\theta) - \lambda\,\nabla_\theta \bar{a}(\theta)$$

- **Source:** `src/objective/objectives/generali/model_based.py` :: `ModelBasedObjective.value()`, `ModelBasedObjective.grad()`, `ModelBasedObjective._lagrangian_adjustment()`
- **Notes:** `base_value()` and `base_value_at_u()` keep exposing the raw objective $$J$$ for experiment summaries and sweep frontier plots while optimization uses $$J_\lambda$$.

### 3.5 Noisy Objective Wrapper

`NoisyObjective` wraps an action-level objective with additive deterministic
noise:

$$\hat{M}(x, u) = M(x, u) + \delta(x, u)$$

and the theta-space value oracle is

$$\hat{J}(\theta) = \frac{1}{n}\sum_{i=1}^n \hat{M}(x_i, \pi_\theta(x_i)).$$

The initial homoskedastic Gaussian noise adapter uses

$$\delta(x_i, u_i) = \sigma_\delta\,\varepsilon(x_i, u_i; s), \qquad \varepsilon \sim N(0, 1)$$

where $$\varepsilon(x_i, u_i; s)$$ is generated by a stable hash of the experiment
noise seed $$s$$, the exact row $$x_i$$, and the exact action $$u_i$$. Therefore
the same $$(x_i, u_i)$$ pair receives the same noise on every objective call, while
different actions for the same row generally receive different noise.

For a policy-free theta-space objective such as a synthetic ladder rung, the same
homoskedastic adapter instead provides one scalar noise value per exact
parameter vector:

$$\hat{J}_s(\theta) = J(\theta) + \sigma_\delta\,\varepsilon(\theta; s),
\qquad \varepsilon(\theta; s) \sim N(0,1).$$

Here $$\varepsilon(\theta;s)$$ is generated by a stable hash of the noise seed
and the exact float64 parameter vector. Repeated queries at the same $$\theta$$
therefore agree, while different parameter vectors generally receive different
noise. This path adds one objective-level draw rather than averaging one draw
per dummy state row; the policy-free objective does not depend on `x_batch`.

The heteroskedastic Gaussian adapter scales the same unit-normal field by an
action-dependent standard deviation that grows linearly with distance from a
noise center $$u_c$$ (typically the planted optimum $$u^*$$):

$$\delta(x_i, u_i) = \big(\sigma_0 + \gamma\,|u_i - u_c|\big)\,\varepsilon(x_i, u_i; s)$$

so value queries near the global minimum stay nearly noiseless while queries far
from it become increasingly noisy. Because both adapters share the hash-keyed
field $$\varepsilon(x_i, u_i; s)$$, setting $$\gamma = 0$$ reproduces the
homoskedastic adapter with $$\sigma_\delta = \sigma_0$$ exactly. The noise is
zero-mean at every $$(x, u)$$, so it perturbs value oracles without biasing the
objective in expectation. Heteroskedastic noise remains action-only because its
scale depends on distance from an action center $$u_c$$.

- **Source:** `src/objective/noise.py` :: `NoisyObjective`, `HomoskedasticGaussianNoise`, `HeteroskedasticGaussianNoise`
- **Notes:** This wrapper intentionally exposes no analytical gradient for its
  noisy value oracle. Use zeroth-order estimators for optimization, or
  call the wrapped `base_objective.grad(...)` to inspect the true non-noisy
  objective gradient. `CorrectnessSpec(gradient_source="denoised_exact")` uses
  this wrapped-objective gradient for diagnostics, while `"exact"` remains the
  optimizer-facing objective gradient source.

### 3.6 Biased Objective Wrapper

`BiasedObjective` wraps an action-level objective with a deterministic additive
action bias:

$$
\hat{M}(x, u) = M(x, u) + b(x, u)
$$

The default `LinearActionBias` preserves the original global linear action bias:

$$b(u) = - \lambda_{bias}\,u$$

and the theta-space value oracle is

$$\hat{J}(\theta) = J(\theta) - \lambda_{bias}\,\frac{1}{n}\sum_{i=1}^n \pi_\theta(x_i).$$

For minimization and $$\lambda_{bias} > 0$$, larger actions look artificially
better because they reduce $$\hat{M}$$.

**Gradient w.r.t. $u$:**

$$\frac{\partial \hat{M}}{\partial u} = \frac{\partial M}{\partial u} - \lambda_{bias}$$

**Gradient w.r.t. $\theta$:**

$$\nabla_\theta \hat{J}(\theta) = \nabla_\theta J(\theta) - \lambda_{bias}\,\frac{1}{n}\sum_{i=1}^n \nabla_\theta\pi_\theta(x_i)$$

`UpperSupportHingeBias` instead leaves an upper action-support band exact and
adds optimism only above support. Let $$h = u_c + r$$ be the upper support
boundary, where $$u_c$$ is the support center and $$r \ge 0$$ is the support
radius. The hard hinge is

$$b(u) = -\lambda_{bias}\,(u-h)_+,$$

so the surrogate equals the true objective for $$u \le h$$ and becomes
optimistic only when actions exceed support. Its action-gradient is

$$\frac{\partial b}{\partial u} = -\lambda_{bias}\,\mathbb{1}\{u > h\}.$$

When `smooth_tau = \tau > 0`, the hinge excess is replaced by

$$\tau\log\left(1 + \exp\left(\frac{u-h}{\tau}\right)\right),$$

with action-gradient

$$\frac{\partial b}{\partial u} = -\lambda_{bias}\,\sigma\left(\frac{u-h}{\tau}\right).$$

- **Source:** `src/objective/objectives/biased.py` :: `ActionBias`,
  `LinearActionBias`, `UpperSupportHingeBias`, `BiasedObjective`
- **Notes:** `base_value()` and `base_value_at_u()` expose the wrapped true
  objective for reporting, while optimization uses the biased surrogate through
  `value()` and `grad()`. The bias is deterministic and introduces no new seed
  stream.

#### 3.6.1 Policy-Free Theta Biases

`ThetaBiasedObjective` adds a deterministic scalar bias directly to a
one-dimensional theta-space objective:

$$\widetilde J(x)=J(x)+b(x), \qquad
\widetilde J'(x)=J'(x)+b'(x).$$

The zeroth-order proof-validation experiment uses three bias fields:

$$b_{\rm linear}(x)=\alpha x, \qquad b'_{\rm linear}(x)=\alpha,$$

$$b_{\rm arctan}(x)=\alpha\arctan x, \qquad
b'_{\rm arctan}(x)=\frac{\alpha}{1+x^2},$$

$$b_{\rm remainder}(x)=\alpha(x-\arctan x), \qquad
b'_{\rm remainder}(x)=\frac{\alpha x^2}{1+x^2}.$$

The remainder is cubic near the clean minimum because
$$x-\arctan x=x^3/3+O(x^5)$$. All three obey
$$\sup_x|b'(x)|\le |\alpha|$$. For the two nonlinear fields,

$$\sup_x|b''(x)|=\frac{3\sqrt3}{8}|\alpha|, \qquad
\sup_x|b'''(x)|=2|\alpha|.$$

- **Source:** `src/objective/modifications/bias.py` :: `ThetaBias`,
  `LinearThetaBias`, `ArctanThetaBias`, `ArctanRemainderThetaBias`,
  `ThetaBiasedObjective`
- **Notes:** Theta biases are separate from action biases: they apply to direct
  theta-space objectives without requiring a policy. `base_value()` preserves
  clean-objective reporting, and no new seed stream is introduced.

#### 3.6.2 Policy-Free Support Envelopes

For the proof objective

$$f(u)=u^2+\frac12(\sin u-u), \qquad u^\star=0,$$

let $$C=[\ell,h]$$ be the covered interval and
$$d(u,C)=\max(\ell-u,0,u-h)$$. The envelope sweep optimizes the upper objective

$$F(u)=f(u)+\phi(u)$$

for three deterministic envelope forms.

The constant control is

$$\phi_{\mathrm{const}}(u)=A, \qquad
\phi_{\mathrm{const}}'(u)=0.$$

It shifts every value equally and therefore leaves the exact and zeroth-order
trajectories unchanged. In particular, it does not identify a unique envelope
minimum: every $$u$$ minimizes the envelope, while $$F$$ retains the clean
minimum.

The constant-derivative increasing envelope is the interval-distance penalty

$$\phi_{\mathrm{lin}}(u)=\lambda d(u,C), \qquad
\phi_{\mathrm{lin}}'(u)=
\begin{cases}
-\lambda,&u<\ell,\\
0,&\ell<u<h,\\
\lambda,&u>h.
\end{cases}$$

When the clean minimum lies below coverage, as here, the exact minimizer solves
$$f'(u)=\lambda$$ below $$\ell$$ while $$\lambda<f'(\ell)$$ and is pinned to
the left coverage boundary $$u=\ell$$ once $$\lambda\ge f'(\ell)$$. Thus the
upper objective selects the truth-favorable edge of the envelope's flat
minimum, not an arbitrary interior point.

The smooth saturating envelope is zero on coverage and increases monotonically
with distance outside it:

$$
\phi_{\mathrm{nc}}(u)=
\begin{cases}
0,&d(u,C)=0,\\
A\exp\left[-\left(\frac{s}{d(u,C)}\right)^2\right],&d(u,C)>0.
\end{cases}
$$

This function is $$C^\infty$$ at the interval boundary and bounded above by
$$A$$. For $$d>0$$ its derivative magnitude is

$$
\left|\phi_{\mathrm{nc}}'(u)\right|
=\frac{2A}{s}\left(\frac{s}{d}\right)^3
\exp\left[-\left(\frac{s}{d}\right)^2\right].
$$

It reaches its maximum at $$s/d=\sqrt{3/2}$$:

$$
\max_u|\phi_{\mathrm{nc}}'(u)|
=2(3/2)^{3/2}e^{-3/2}\frac{A}{s}
\approx 0.8198326\frac{A}{s}.
$$

Although $$\phi_{\mathrm{nc}}$$ is monotone in distance, its slope first grows
and then decays, so the envelope and the resulting upper objective need not be
convex. The sweep matches the linear slope to this maximum,
$$\lambda=0.8198326A/s$$, and uses $$C=[0.75,1.25]$$, $$s=0.25$$, and
$$A\in\{0,0.25,0.35,0.42,0.60,0.70\}$$. These values cross the creation of a
truth-side minimum/maximum pair, the global-minimum switch between truth- and
coverage-side basins, and the later disappearance of the outer basin.

The estimator population landscapes used by the analyzer are

$$
F_{\mathrm{FD},\sigma}'(u)
=\frac{F(u+\sigma)-F(u-\sigma)}{2\sigma}
$$

and

$$
F_{\mathrm{Stein},\sigma}'(u)
=\mathbb E\left[F'(u+\sigma Z)\right],
\qquad Z\sim N(0,1).
$$

The former is the derivative of a uniform convolution and the latter of a
Gaussian convolution. Comparing their stationary points with finite-run
$$u_K$$ separates deterministic envelope geometry, zeroth-order smoothing,
initialization-dependent basin selection, and finite-sample optimizer noise.

- **Source:** `src/objective/modifications/regularization.py` ::
  `ConstantThetaRegularizer`, `IntervalDistanceThetaRegularizer`,
  `SmoothSaturatingIntervalThetaRegularizer`, `RegularizedObjective`;
  `manifests/zeroth_order_envelopes.json`;
  `scripts/analyze_zeroth_order_envelopes.py`
- **Notes:** These regularizers are reusable theta-space objective
  modifications. The scalar stationary-point and convolution tooling is kept
  experiment-side because it is specific to one-dimensional landscape
  analysis.

#### 3.6.3 Finite-Policy Lower Confidence Bounds

Let the finite class of constant policies be

$$\Pi=\{0,0.1,\ldots,1.0\},\qquad K=|\Pi|=11,$$

with true value and Gaussian surrogate

$$V^\pi=\pi,$$

$$\widehat V_s^\pi
=V^\pi+\pi Z_s^\pi
=\pi+\pi Z_s^\pi,
\qquad Z_s^\pi\overset{\mathrm{i.i.d.}}{\sim}N(0,1).$$

Thus

$$\mathbb E[\widehat V_s^\pi]=V^\pi,
\qquad \operatorname{Var}(\widehat V_s^\pi)=\pi^2.$$

For confidence level $$1-\delta$$, define

$$q_\delta
=\Phi^{-1}\!\left(1-\frac{\delta}{2K}\right),$$

$$\mathcal E^\pi(\delta)=2\pi q_\delta,$$

and the lower confidence bound

$$\underline V_{\delta,s}^\pi
=\widehat V_s^\pi-\frac12\mathcal E^\pi(\delta)
=\pi+\pi Z_s^\pi-\pi q_\delta.$$

The Gaussian tail probability and a union bound give

$$\Pr\!\left(
|\widehat V_s^\pi-V^\pi|
\le \frac12\mathcal E^\pi(\delta)
\text{ for every }\pi\in\Pi
\right)\ge 1-\delta.$$

Because the policy-level Gaussian variables are independent, the exact joint
coverage is

$$\Pr(A_{\delta,s})
=\left(1-\frac{\delta}{K}\right)^K
\ge 1-\delta.$$

The finite optimizer evaluates every policy and selects

$$\widehat\pi_{\delta,s}
\in\arg\max_{\pi\in\Pi}\underline V_{\delta,s}^\pi.$$

It is exact, so its script-style optimization error is $$\varepsilon=0$$.
On the simultaneous confidence event, every comparator
$$\widetilde\pi\in\Pi$$ satisfies the Proposition 11.2 oracle inequality

$$V^{\widehat\pi_{\delta,s}}
\ge
V^{\widetilde\pi}
-\mathcal E^{\widetilde\pi}(\delta)
-\varepsilon.$$

Each run seed draws one vector $$(Z_s^\pi)_{\pi\in\Pi}$$ and reuses it for
every configured $$\delta$$. This paired design changes only the confidence
radius within a seed; different run seeds use independently derived noise
streams.

- **Source:** `src/experiments/policy_lcb/finite.py` (with compatibility import
  `src/experiments/finite_policy_lcb.py`);
  `manifests/finite_policy_lcb_validation.json`
- **Notes:** There is no gradient method, theta initialization, data split, or
  optimizer RNG. Exhaustive policy evaluation is the optimizer, and only the
  noise seed varies.

#### 3.6.4 Continuous-Policy Lower Confidence Bounds

For the continuous class $$\Pi=[0,1]$$, one scalar Gaussian draw is shared by
every policy within run seed $$s$$:

$$
V(\pi)=5\pi-5\pi^2,
\qquad
\widehat V_s(\pi)=V(\pi)+\pi Z_s,
\qquad
Z_s\sim N(0,1).
$$

The quadratic mean changes the optimizer but not the error process:

$$
\widehat V_s(\pi)-V(\pi)=\pi Z_s.
$$

Because the same $$Z_s$$ is used throughout the interval, every positive
policy has the same standardized error and

$$
\sup_{\pi\in(0,1]}
\frac{|\widehat V_s(\pi)-V(\pi)|}{\pi}
=|Z_s|.
$$

Consequently, the simultaneous two-sided quantile has no finite-class
Bonferroni factor:

$$
q_\delta=\Phi^{-1}(1-\delta/2),
\qquad
\mathcal E^\pi(\delta)=2\pi q_\delta.
$$

Continuity itself does not remove the multiplicity correction. The finite
experiment has $$K$$ distinct Gaussian coordinates and controls their union
with $$q_{\delta,K}=\Phi^{-1}(1-\delta/(2K))$$. Here the continuum has a
rank-one error process, so the policy-indexed intersection is exactly the
single event $$|Z_s|\le q_\delta$$. A nonconstant Gaussian process
$$Z_s(\pi)$$ would instead require a bound for its supremum, typically involving
the complexity of the policy class rather than this scalar quantile.

The experiment minimizes the negative lower confidence bound

$$
F_{s,\delta}(\pi)
=-\underline V_{s,\delta}(\pi)
=5\pi^2+(q_\delta-5-Z_s)\pi,
$$

whose exact derivative and constrained minimizer are

$$
F'_{s,\delta}(\pi)=10\pi+q_\delta-5-Z_s,
$$

$$
\pi^*_{s,\delta}
=\operatorname{clip}_{[0,1]}
\left(\frac{5+Z_s-q_\delta}{10}\right).
$$

For the configured confidence levels, $$q_\delta\in[0.674,2.576]$$. Thus all
draws $$Z_s\in[-2,2]$$ have a strictly interior analytic minimizer.

The single event $$|Z_s|\le q_\delta$$ covers every policy simultaneously, so

$$
\Pr\!\left(
|\widehat V_s(\pi)-V(\pi)|\le \tfrac12\mathcal E^\pi(\delta)
\text{ for every }\pi\in[0,1]
\right)=1-\delta.
$$

Projected first-order, finite-difference, and Stein-difference updates retain
feasible iterates in $$[0,1]$$. Finite-difference and Stein probes evaluate the
same quadratic formula outside the feasible interval before the updated policy
is projected.
If an optimizer returns $$\widehat\pi$$, its measured LCB optimization error is

$$
\varepsilon
=F_{s,\delta}(\widehat\pi)-F_{s,\delta}(\pi^*_{s,\delta})
=\underline V_{s,\delta}(\pi^*_{s,\delta})
-\underline V_{s,\delta}(\widehat\pi)\ge 0.
$$

- **Source:** `src/experiments/policy_lcb/continuous.py`
- **Notes:** Problem-noise seeds vary across runs. The Stein perturbation
  stream is separate and deliberately paired across run seeds, confidence
  levels, and starts so cross-seed spread isolates the shared Gaussian draw.

#### 3.6.5 Variable-Envelope Finite-Grid Lower Confidence Bounds

Let the finite optimization class be the inclusive grid

$$
\mathcal X=\{x_1,\ldots,x_K\}\subset[0,1]
$$

with true value and grid optimum

$$
f(x)=5x-5x^2,
\qquad
x^*\in\arg\max_{x\in\mathcal X}f(x).
$$

For an uncertainty center $$m$$, the clipped distance-ramp scale is

$$
\sigma_m(x)=\sigma_{\min}
+(\sigma_{\max}-\sigma_{\min})
\min\!\left(\frac{|x-m|}{r},1\right).
$$

One run seed draws a vector of independent standard Gaussians and reuses it for
every noise magnitude, uncertainty center, and envelope calibration:

$$
Z_{s,x}\overset{\mathrm{i.i.d.}}{\sim}N(0,1),
\qquad
\widehat f_{s,c,m}(x)=f(x)+c\sigma_m(x)Z_{s,x}.
$$

For failure probability $$\delta$$, the simultaneous Bonferroni and pointwise
two-sided quantiles are

$$
q_{\mathrm{sim}}=\Phi^{-1}\!\left(1-\frac{\delta}{2K}\right),
\qquad
q_{\mathrm{point}}=\Phi^{-1}\!\left(1-\frac{\delta}{2}\right),
$$

and either calibration defines the half-width and lower confidence bound

$$
E_{c,m,q}(x)=c q\sigma_m(x),
\qquad
\underline f_{s,c,m,q}(x)=\widehat f_{s,c,m}(x)-E_{c,m,q}(x).
$$

For Bonferroni calibration, a union bound gives

$$
\Pr\!\left(
|\widehat f_{s,c,m}(x)-f(x)|\le E_{c,m,q_{\mathrm{sim}}}(x)
\ \forall x\in\mathcal X
\right)\ge1-\delta.
$$

Under the configured independent Gaussian coordinates, its exact coverage is
$$ (1-\delta/K)^K $$. Pointwise calibration instead covers each fixed point
with probability $$1-\delta$$, so its expected covered fraction is
$$1-\delta$$ while its exact simultaneous coverage is $$(1-\delta)^K$$.

The exhaustive selectors are

$$
\widehat x_{\mathrm{nom}}
\in\arg\max_x\widehat f_{s,c,m}(x),
\qquad
\widehat x_{\mathrm{var}}
\in\arg\max_x\underline f_{s,c,m,q}(x).
$$

The uniform half-width $$E_{\mathrm{unif}}=\max_x E_{c,m,q}(x)$$ is constant
over $$x$$, hence

$$
\arg\max_x[\widehat f_{s,c,m}(x)-E_{\mathrm{unif}}]
=\arg\max_x\widehat f_{s,c,m}(x).
$$

The deterministic penalized target

$$
x^\dagger_{c,m,q}\in\arg\max_x[f(x)-E_{c,m,q}(x)]
$$

separates envelope geometry from surrogate randomness. Regret is

$$
R(\widehat x)=f(x^*)-f(\widehat x).
$$

On the simultaneous two-sided coverage event, exact variable-LCB maximization
obeys

$$
R(\widehat x_{\mathrm{var}})\le 2E_{c,m,q_{\mathrm{sim}}}(x^*).
$$

Because the same standardized $$Z_{s,x}$$ is paired across positive $$c$$ and
all $$\sigma_m(x)>0$$, the coverage event reduces to
$$\{|Z_{s,x}|\le q\ \forall x\}$$ for every configured center and every
positive noise magnitude. No additional center multiplicity correction is
needed for this paired construction.

- **Source:** `src/experiments/policy_lcb/finite_grid.py`;
  `manifests/variable_lcb_envelope_characterization.json`
- **Notes:** The grid search is exact and has no optimizer or reporting RNG.
  The master noise seed is the only stochastic stream. The random surrogate is
  defined only at the $$K$$ grid points. Lines connecting those values in plots
  are visualization aids, not an interpolation rule or a continuous random
  function.

#### 3.6.6 Continuous Finite-Fourier GP Lower Confidence Bounds

The continuous optimization class is $$\mathcal X=[0,1]$$, with

$$
f(x)=5x-5x^2,
\qquad x^*=\tfrac12,
\qquad R(x)=f(x^*)-f(x)=5(x-\tfrac12)^2.
$$

Let

$$
h(t)=
\begin{cases}
6t^5-15t^4+10t^3,&0\le t<1,\\
1,&t\ge1,
\end{cases}
$$

and define the smooth clipped uncertainty scale

$$
\sigma_m(x)=\sigma_{\min}
+(\sigma_{\max}-\sigma_{\min})
h\!\left(\frac{|x-m|}{r}\right).
$$

The configured values are $$\sigma_{\min}=0.1$$,
$$\sigma_{\max}=1$$, and $$r=0.5$$. The polynomial has zero first and
second derivatives at both endpoints, so $$\sigma_m$$ is $$C^2$$ at its
minimum $$m$$ and at the clipped plateau. The point $$m$$ minimizes marginal
uncertainty; it need not minimize a realized absolute error.

For rank $$J$$ and lengthscale $$\ell$$, use deterministic half-normal
spectral quantiles

$$
p_j=\frac{j-\tfrac12}{J},
\qquad
\omega_j=\ell^{-1}\Phi^{-1}\!\left(\frac{1+p_j}{2}\right),
\qquad j=1,\ldots,J.
$$

One run seed draws the coefficient vector

$$
\xi_s=(A_{s,1},B_{s,1},\ldots,A_{s,J},B_{s,J})
\sim N(0,I_{2J})
$$

and defines the analytic random function

$$
G_s(x)=\frac1{\sqrt J}\sum_{j=1}^J
\left[A_{s,j}\cos(\omega_jx)+B_{s,j}\sin(\omega_jx)\right].
$$

This is an exact finite-rank GP with covariance

$$
k_J(x,x')=\frac1J\sum_{j=1}^J\cos(\omega_j(x-x')),
\qquad k_J(x,x)=1.
$$

It is evaluated from this formula at every optimizer query. Plotting grids do
not define or interpolate the path. The nonstationary surrogate and LCB are

$$
\widehat f_{s,c,m}(x)=f(x)+c\sigma_m(x)G_s(x),
$$

$$
E_{c,m}(x)=cq\sigma_m(x),
\qquad
\underline f_{s,c,m}(x)=\widehat f_{s,c,m}(x)-E_{c,m}(x).
$$

To certify one continuum-wide multiplier, write
$$G_s(x)=\phi(x)^\top\xi_s$$. Then

$$
\|\phi'(x)\|_2
=L_\phi
=\sqrt{\frac1J\sum_{j=1}^J\omega_j^2}.
$$

For an equally spaced $$N$$-point covering net with radius
$$\rho=1/[2(N-1)]$$, split the failure probability equally between the net
and coefficient-norm events:

$$
q_{\mathrm{net}}
=\Phi^{-1}\!\left(1-\frac{\delta}{4N}\right),
\qquad
r_{\mathrm{coef}}
=\sqrt{\chi^2_{2J,1-\delta/2}},
$$

$$
q=q_{\mathrm{net}}+\rho L_\phi r_{\mathrm{coef}}.
$$

Bonferroni gives

$$
\Pr\!\left(\max_{t\in T}|G_s(t)|\le q_{\mathrm{net}}\right)
\ge1-\delta/2,
$$

while the chi-square event has probability $$1-\delta/2$$ and implies, for
the nearest net point $$t$$,

$$
|G_s(x)-G_s(t)|
\le |x-t|L_\phi\|\xi_s\|_2
\le\rho L_\phi r_{\mathrm{coef}}.
$$

A union bound therefore proves

$$
\Pr\!\left(\sup_{x\in[0,1]}|G_s(x)|\le q\right)\ge1-\delta.
$$

For $$J=32$$, $$\ell=0.2$$, $$N=129$$, and $$\delta=0.05$$,
$$L_\phi\approx4.9505$$, $$q_{\mathrm{net}}\approx3.7270$$,
$$r_{\mathrm{coef}}\approx9.3810$$, and $$q\approx3.9084$$. This multiplier
is computed before and independently of all run seeds. The seeds only verify
the guaranteed coverage empirically.

For positive $$c$$ and $$\sigma_m(x)$$, the coverage inequality reduces to
$$|G_s(x)|\le q$$. Reusing one $$G_s$$ across every $$m$$ and $$c$$ therefore
requires no additional multiplicity correction. On this event, the exact LCB
maximizer obeys

$$
R(\widehat x_{\mathrm{LCB}})\le2E_{c,m}(x^*).
$$

If an iterative optimizer has LCB objective gap $$\varepsilon$$ relative to
the global LCB maximum, then

$$
R(\widehat x_{\mathrm{method}})
\le2E_{c,m}(x^*)+\varepsilon.
$$

Global reference values are certified numerically in one dimension. If
$$|F''(x)|\le M_I$$ on $$I=[a,b]$$, linear-interpolation error gives

$$
\sup_{x\in I}F(x)
\le\max\{F(a),F(b)\}+\frac{M_I(b-a)^2}{8}.
$$

Branch-and-bound subdivides only intervals whose upper bound can beat the
incumbent and stops when the global upper/lower value gap is at most the
manifest tolerance. This certifies the reference value, not global convergence
of first-order or zeroth-order iterates.

- **Source:** `src/experiments/policy_lcb/continuous_gp.py`;
  `manifests/continuous_gp_variable_lcb.json`
- **Notes:** The GP coefficient seed, fixed Stein perturbation seed, and
  reporting seed are separate. Formulas are extended to all real $$x$$ for
  zeroth-order probes, while every optimizer iterate is projected onto
  $$[0,1]$$.

#### 3.6.7 Continuous-GP Regret Decomposition

The decomposition experiment retains the objective, analytic Fourier draw,
uncertainty family, and global-reference construction from Section 3.6.6, but
separates the surrogate-error and lower-envelope parameters:

$$
\widehat f(x)=f(x)+c_f\sigma_{m_f}(x)G_s(x),
\qquad
\underline f(x)=\widehat f(x)-c_Eq\sigma_{m_E}(x).
$$

Here $$c_f$$ controls the global magnitude of the frozen surrogate error and
$$m_f$$ controls its spatial amplitude profile. Independently, $$c_E$$ controls
the global confidence-correction magnitude and $$m_E$$ controls where that
correction is narrowest. The standardization of $$\sigma_m$$ remains fixed at
minimum $$0.1$$ and maximum $$1$$, so the scale and shape parameters are
identifiable.

For $$c_f>0$$, define the certified shape ratio and effective GP threshold

$$
r_{\min}(m_f,m_E)=\inf_{x\in[0,1]}
\frac{\sigma_{m_E}(x)}{\sigma_{m_f}(x)},
\qquad
q_{\mathrm{eff}}=q\frac{c_E}{c_f}r_{\min}(m_f,m_E).
$$

The two-sided event $$\sup_x|G_s(x)|\le q_{\mathrm{eff}}$$ is sufficient for
$$\underline f(x)\le f(x)$$ everywhere. Its certified probability is obtained
by inverting the same covering-net construction as Section 3.6.6: for failure
probability $$\delta'$$,

$$
q(\delta')=
\Phi^{-1}\!\left(1-\frac{\delta'}{4N}\right)
+\rho L_\phi\sqrt{\chi^2_{2J,1-\delta'/2}}.
$$

Thus $$p_{\mathrm{cert}}(t)=1-\delta_t$$ when
$$q(\delta_t)=t$$. If the threshold is below the smallest value certifiable by
this construction, the reported lower bound is zero. When $$c_f=0$$ and
$$c_E\ge0$$ the envelope is deterministically valid, while $$c_E=0<c_f$$ has
certified level zero. The matched unit-scale case has
$$q_{\mathrm{eff}}=q(0.05)$$ and therefore certified level $$0.95$$.

For one realized path, envelope validity is checked independently by certifying
the maximum of

$$
v(x)=\underline f(x)-f(x)
=c_f\sigma_{m_f}(x)G_s(x)-c_Eq\sigma_{m_E}(x).
$$

No value of $$f$$ is used to clip or alter the optimized lower envelope.
Surrogate error is summarized by the certified quantity

$$
\|\widehat f-f\|_\infty
=c_f\sup_{x\in[0,1]}|\sigma_{m_f}(x)G_s(x)|.
$$

For an optimizer checkpoint $$\widehat x$$, define

$$
T=f(x^*)-\underline f(x^*),
\qquad
\varepsilon=\max_x\underline f(x)-\underline f(\widehat x).
$$

Whenever the realized lower envelope is valid over the domain,

$$
R(\widehat x)=f(x^*)-f(\widehat x)
\le T+\varepsilon.
$$

Branch-and-bound supplies lower and upper brackets for the global values, so
the stored surrogate error, realized violation, and optimizer error retain
their numerical certification gaps rather than being described as exact real
numbers.

- **Source:** `src/experiments/policy_lcb/continuous_gp_core.py`;
  `src/experiments/policy_lcb/continuous_gp_decomposition.py`;
  `manifests/continuous_gp_regret_decomposition.json`
- **Notes:** Each run seed owns one Fourier coefficient draw reused by every
  condition. One dedicated optimizer seed fixes the antithetic Stein
  perturbations across paths, conditions, and starts. Zeroth-order probes use
  the analytic real-line extension and iterates are projected to $$[0,1]$$.

### 3.7 Synthetic Ladder Objectives

Direct theta-space benchmark functions over the decision vector $$w = \theta$$
with globally known minimizers by construction; `x_batch` is ignored and there
is no policy or action space. Each instance is deterministic given its
construction seed (`from_seed`), so true-gap metrics
$$f(w) - f(w^*)$$ and $$\|w - w^*\|_2$$ need no reference runs.

#### 3.7.1 Strongly Convex Quadratic (rung 1)

$$f(w) = \tfrac{1}{2}(w - w^*)^\top A (w - w^*), \qquad
A = Q\,\mathrm{diag}(\lambda)\,Q^\top,$$

with $$Q$$ a seeded random rotation and eigenvalues log-spaced in
$$[\mu, \mu\kappa]$$ for condition number $$\kappa$$.

**Gradient:** $$\nabla f(w) = A(w - w^*)$$. The function is
$$\mu$$-strongly convex and $$\mu\kappa$$-smooth with unique minimizer
$$w^*$$ and minimum value 0.

- **Source:** `src/objective/objectives/synthetic/ladder.py` :: `StronglyConvexQuadratic`

#### 3.7.2 Smoothed Nonconvex With Known Global Minimum (rung 2)

$$f(w) = \tfrac{1}{2}\|w - w^*\|^2
 - a_0\, e^{-\|w - w^*\|^2/(2 s_0^2)}
 - \sum_j a_j\, \psi\!\left(\frac{\|w - c_j\|^2}{\rho_j^2}\right),$$

with the compactly supported $$C^\infty$$ mollifier
$$\psi(s) = e^{1 - 1/(1 - s)}$$ on $$[0, 1)$$ and $$\psi \equiv 0$$ for
$$s \ge 1$$, so trap $$j$$ affects only $$\{\|w - c_j\| < \rho_j\}$$.

**Gradient:** with $$d = w - w^*$$, $$d_j = w - c_j$$, and
$$s_j = \|d_j\|^2/\rho_j^2$$,

$$\nabla f(w) = \left(1 + \frac{a_0}{s_0^2}\, e^{-\|d\|^2/(2 s_0^2)}\right) d
 - \sum_j \frac{2 a_j}{\rho_j^2}\, \psi'(s_j)\, d_j,
\qquad \psi'(s) = -\frac{\psi(s)}{(1 - s)^2}.$$

**Global-minimum guarantee.** Let
$$g(r) = \tfrac{1}{2} r^2 - a_0 e^{-r^2/(2 s_0^2)}$$ be the trap-free radial
profile; $$g'(r) = r\,(1 + (a_0/s_0^2) e^{-r^2/(2 s_0^2)}) > 0$$ for
$$r > 0$$, so $$g$$ is strictly increasing with unique minimum
$$g(0) = -a_0$$. Construction enforces:

1. clearance $$\gamma_j = \|c_j - w^*\| - \rho_j > 0$$ (no trap support
   touches $$w^*$$, hence $$f(w^*) = -a_0$$ exactly and
   $$\nabla f(w^*) = 0$$);
2. pairwise disjoint trap supports
   ($$\|c_i - c_j\| > \rho_i + \rho_j$$), so at most one trap is active at
   any point;
3. per-trap depth budget $$a_j < \tfrac{1}{2}\gamma_j^2$$.

For $$w$$ in the support of trap $$j$$, $$r = \|w - w^*\| \ge \gamma_j$$ and
$$f(w) \ge g(r) - a_j \ge g(0) + \tfrac{1}{2}\gamma_j^2 - a_j > f(w^*)$$;
outside all supports $$f = g(r) > g(0)$$ for $$r > 0$$. Hence $$w^*$$ is the
unique global minimizer.

**Whether the traps actually trap is a separate, unenforced condition.** The
budget above only guarantees that $$w^*$$ stays global; it says nothing about a
trap admitting a local minimum. A trap center is never itself a critical point
($$\psi'(0)$$ contributes nothing at $$w = c_j$$, leaving
$$\nabla f(c_j) \propto c_j - w^*\neq 0$$), so a basin exists only when the trap
is steep enough to overcome the quadratic pull. `from_seed` sets
$$a_j = \texttt{depth\_fraction} \cdot \tfrac{1}{2}\gamma_j^2$$: empirically at the
default 0.9 every trap is a genuine local minimum, at 0.5-0.3 only some are, and
by 0.1 none are and the rung is unimodal despite remaining formally nonconvex.
Both ends are pinned by tests in `tests/objective/test_synthetic_functions.py`.

- **Source:** `src/objective/objectives/synthetic/ladder.py` :: `SmoothedNonconvex`

#### 3.7.3 Piecewise Convex (rung 3, planned — structural stub)

Intended form: with rotated coordinates $$v = Q^\top (w - w^*)$$ (identity
when unrotated), $$f(w) = \sum_i h_i(v_i)$$ where

$$h_i(v) = \begin{cases}
 \tfrac{1}{2} c_i v^2 & |v| \le k_i \\
 \tfrac{1}{2} c_i k_i^2 + m_i (|v| - k_i) & |v| > k_i
\end{cases}$$

with $$m_i > c_i k_i$$ producing kinks at $$\pm k_i$$; convexity requires
$$m_i \ge c_i k_i$$. `kink_at_optimum` collapses $$k_i = 0$$ (weighted-L1
behavior, nonsmooth at the optimum). `grad()` returns the right derivative at
kinks. `_f`/`_grad_f` raise `NotImplementedError` until implemented.

- **Source:** `src/objective/objectives/synthetic/ladder.py` :: `PiecewiseConvex`

#### 3.7.4 Piecewise Nonconvex Double Well (rung 4, planned — structural stub)

Intended form: with rotated coordinates $$v = Q^\top (w - w^*)$$, masked
coordinates use

$$h_i(v) = \min\!\left(\tfrac{1}{2} c_i v^2,\;
 \tfrac{1}{2} d_i (v - b_i)^2 + \delta_i\right), \qquad \delta_i > 0,$$

and unmasked coordinates stay purely quadratic. The decoy well at
$$v = b_i$$ sits $$\delta_i$$ above the true well, so the global minimum is
$$w^*$$ with value 0 (sum of coordinate-wise minima); the min of two parabolas
is nonconvex with kinks at the crossing points. `_f`/`_grad_f` raise
`NotImplementedError` until implemented.

- **Source:** `src/objective/objectives/synthetic/ladder.py` :: `PiecewiseNonconvexDoubleWell`

### 3.8 Zeroth-Order Proof-Validation Objective

The one-dimensional policy-free objective is

$$f(x)=x^2+\frac12(\sin x-x).$$

Its first three derivatives are

$$f'(x)=2x+\frac12(\cos x-1),$$

$$f''(x)=2-\frac12\sin x \in [1.5,2.5],$$

$$f'''(x)=-\frac12\cos x, \qquad |f'''(x)|\le0.5.$$

Thus $$f$$ is globally $$\mu=1.5$$ strongly convex and $$L=2.5$$ smooth,
with unique minimizer $$x^\star=0$$ and third-derivative bound $$\rho=0.5$$.
For central finite difference,

$$D_\sigma f(x)=2x+\frac12\left(\frac{\sin\sigma}{\sigma}\cos x-1\right).$$

For the two-sided Gaussian Stein-difference estimator, its population mean is

$$\mathbb E_W\!\left[
\frac{f(x+\sigma W)-f(x-\sigma W)}{2\sigma}W
\right]
=2x+\frac12\left(e^{-\sigma^2/2}\cos x-1\right),
\qquad W\sim N(0,1).$$

The corresponding estimator fixed point $$x^\star_{\rm est}$$ is the unique
root of the appropriate population-gradient equation. Both roots move
$$O(\sigma^2)$$ from $$x^\star$$, so their squared displacement is
$$O(\sigma^4)$$.

- **Source:** `src/objective/objectives/synthetic/proof_validation.py` ::
  `ZerothOrderProofObjective`

---

## 4. Chain Rule (Theta-Gradient from Action-Gradient)

$$\nabla_\theta J = \mathbb{E}\!\left[\frac{\partial f}{\partial u}\;\frac{\partial u}{\partial \theta}\right] = \frac{1}{n}\sum_{i=1}^n \frac{\partial f}{\partial u_i}\;\nabla_\theta \pi_\theta(x_i)$$

- **Source:** `src/objective/utils.py` :: `_theta_grad_from_u_grad()`
- **Notes:** Used by all three objectives to compose the action-level gradient
  with the policy Jacobian. Policies may implement `weighted_grad()` to compute
  the vector-Jacobian product directly instead of materializing the full
  `(n, theta_dim)` Jacobian.

---

## 5. Gradient Estimators

All estimators produce $\hat{g} \approx \nabla_\theta J(\theta)$. Each has a
theta-space and a u-space variant. The u-space variant applies the estimator to
the action-level objective $M(x, u)$ and chain-rules back to theta via
$\nabla_\ theta \pi_\theta(x)$.

### 5.1 First-Order (Exact) Gradient

$$\hat{g} = \nabla_\theta J(\theta) \quad\text{(analytical, from \texttt{objective.grad})}$$

- **Source:** `src/optimization/gradients/methods.py` :: `FirstOrderGradient`
- **Cost:** 1 objective gradient evaluation.

### 5.2 Finite Difference

**Theta-space (central):**

$$\hat{g}_k = \frac{J(\theta + \sigma e_k) - J(\theta - \sigma e_k)}{2\sigma}$$

- **Cost:** $2d$ objective evaluations ($d = \dim\theta$).
- **Source:** `src/optimization/helpers.py` :: `finite_difference_theta_grad()` (also supports forward and backward variants)

**U-space:**

$$\hat{g}_{u,i} = \frac{M(x_i, u_i + \sigma) - M(x_i, u_i - \sigma)}{2\sigma}$$

then chain-rule: $\hat{g}_\theta = \frac{1}{n}\sum_i \hat{g}_{u,i}\;\nabla_\theta\pi(x_i)$.

- **Cost:** 2 batch evaluations regardless of $d$.
- **Source:** `src/optimization/gradients/methods.py` :: `FiniteDifferenceGradient._u_grad()`

### 5.3 Gauss-Stein (Score Function) Estimator

**Theta-space (one-sided):**

$$\hat{g} = \frac{1}{m}\sum_{j=1}^m \frac{J(\theta + \sigma\varepsilon_j)}{\sigma}\;\varepsilon_j, \qquad \varepsilon_j \sim \mathcal{N}(0, I^d)$$

- **Cost:** $m$ = `n_grad_samples` evaluations.
- **Source:** `src/optimization/gradients/methods.py` :: `GaussSteinGradient._theta_grad()`

**U-space (one-sided):**

Same estimator applied per sample with scalar $w_j \sim \mathcal{N}(0,1)$,
chain-ruled to theta.

- **Source:** `src/optimization/gradients/methods.py` :: `GaussSteinGradient._u_grad()`

### 5.4 SPSA (Rademacher) Estimator

**Theta-space (two-sided):**

$$\hat{g} = \frac{1}{m}\sum_{j=1}^m \frac{J(\theta + \sigma\Delta_j) - J(\theta - \sigma\Delta_j)}{2\sigma}\;\Delta_j, \qquad \Delta_j \sim \{+1, -1\}^d$$

- **Cost:** $2m$ evaluations.
- **Source:** `src/optimization/gradients/methods.py` :: `SPSAGradient._theta_grad()`

**U-space (two-sided):**

Same with scalar $\Delta_j \sim \{+1, -1\}$, chain-ruled to theta.

- **Source:** `src/optimization/gradients/methods.py` :: `SPSAGradient._u_grad()`

### 5.5 Stein-Difference Estimator

**Theta-space (two-sided Gaussian):**

$$\hat{g} = \frac{1}{m}\sum_{j=1}^m \frac{J(\theta + \sigma\varepsilon_j) - J(\theta - \sigma\varepsilon_j)}{2\sigma}\;\varepsilon_j, \qquad \varepsilon_j \sim \mathcal{N}(0, I^d)$$

- **Cost:** $2m$ evaluations.
- **Source:** `src/optimization/gradients/methods.py` :: `SteinDifferenceGradient._theta_grad()`

**U-space (two-sided Gaussian):**

$$\hat{g}_{u,i} = \frac{1}{m}\sum_{j=1}^m \frac{M(x_i, u_i + \sigma w_j) - M(x_i, u_i - \sigma w_j)}{2\sigma}\; w_j, \qquad w_j \sim \mathcal{N}(0,1)$$

chain-ruled to theta via $\nabla_\theta\pi(x_i)$.

- **Source:** `src/optimization/gradients/methods.py` :: `SteinDifferenceGradient._u_grad()`

---

## 6. Step Rules

### 6.1 Constant Step Size

$$\alpha_t = \alpha \quad\text{(identity)}$$

- **Source:** `src/optimization/steps.py` :: `constant_step_size()`

### 6.2 Armijo Backtracking

Find the largest $\alpha = \alpha_0\, \rho^i$ satisfying the sufficient decrease
condition:

$$J(\theta + \alpha\, d) \;\le\; J(\theta) + c\,\alpha\,\nabla J^\top d$$

where $d = -\nabla J$ (steepest descent), $\rho$ = `shrink` $\in (0, 1)$,
$c$ = `1e-4`.

- **Source:** `src/optimization/steps.py` :: `armijo_backtracking_step_size()`
- **Notes:** Falls back to `min_step` if the condition is never met within
  `max_backtracks` iterations.

### 6.3 SciPy L-BFGS-B

For unconstrained SciPy runs, the optimizer solves

$$
\min_{\theta} J(\theta)
$$

where

$$
J(\theta) = \frac{1}{n}\sum_{i=1}^n f(\pi_\theta(x_i); x_i).
$$

- **Source:** `src/optimization/base.py` :: `Optimization.solve()`
- **Notes:** The repo passes `method="L-BFGS-B"` through `scipy.minimize`. In the
  current implementation, this path is unconstrained; if an acceptance floor is
  configured under `l-bfgs-b`, it is enforced only through the separate smooth
  penalty added inside `ModelBasedObjective.value()`.

### 6.4 SciPy Trust-Constr With Acceptance Constraint

For constrained SciPy runs, the optimizer solves

$$
\min_{\theta} J(\theta)
\quad \text{subject to} \quad
\bar{a}(\theta) \ge \alpha_{\min},
$$

where

$$
\bar{a}(\theta) = \frac{1}{n}\sum_{i=1}^n a\bigl(x_i, \pi_\theta(x_i)\bigr)
$$

is the batch mean acceptance and $$\alpha_{\min}$$ is `acceptance_floor`.

The constraint Jacobian is

$$
\nabla_\theta \bar{a}(\theta) = \frac{1}{n}\sum_{i=1}^n \frac{\partial a(x_i,u_i)}{\partial u}\,\nabla_\theta \pi_\theta(x_i), \qquad u_i = \pi_\theta(x_i).
$$

- **Source:** `src/optimization/base.py` :: `Optimization.solve()`
  - `trust_constr_constraint()` builds the SciPy `NonlinearConstraint`
- **Constraint value source:** `src/objective/objectives/generali/model_based.py` :: `mean_acceptance()`
- **Constraint gradient source:** `src/objective/objectives/generali/model_based.py` :: `mean_acceptance_grad()`
- **Notes:** The repo passes `method="trust-constr"` through `scipy.minimize`
  and enforces the acceptance floor directly as a nonlinear inequality
  constraint, rather than via the smooth penalty path.

---

## 8. Feature Processing

### 8.1 Centering

$$x_{\text{centered}} = x - \mu$$

where $\mu$ is the column-wise mean from `fit()`.

- **Source:** `src/data/feature_processor.py` :: `FeatureProcessor.fit()` / `.transform()`

### 8.2 Sphering (Without PCA)

$$x_{\text{out}} = (x - \mu)\, S, \qquad S = V\,\text{diag}(1/\sqrt{\lambda})\,V^\top$$

where $V, \lambda$ are eigenvectors/eigenvalues of the sample covariance
(sorted descending). Eigenvalues are floored at `regularization` to avoid
division by zero.

- **Source:** `src/data/feature_processor.py` :: `FeatureProcessor.fit()` (with `use_pca=False`)

### 8.3 PCA Whitening

$$x_{\text{out}} = (x - \mu)\, V_k\,\text{diag}(1/\sqrt{\lambda_k})$$

where $V_k$ is the top-$k$ eigenvectors, selected by `n_components` or
`explained_variance_threshold`.

- **Source:** `src/data/feature_processor.py` :: `FeatureProcessor.fit()` (with `use_pca=True`)

### 8.4 PCA Inverse Transform

$$\hat{x}_{\text{raw}} = x_{\text{out}}\, V_k^\top + \mu$$

- **Source:** `src/data/feature_processor.py` :: `FeatureProcessor.inverse_transform_numeric()`
- **Notes:** Only available when `use_pca=True`.

### 8.5 Categorical Encoding

Each category $c$ in column $j$ is mapped to $\frac{\text{label}(c)}{|\text{categories}_j|}$
where $\text{label}(c) \in \{0, 1, \dots\}$. Unknown categories receive code
$|\text{categories}_j|$.

- **Source:** `src/data/feature_processor.py` :: `FeatureProcessor.fit()` / `.transform()`

---

## 9. GLM Coefficient Extraction

### 9.1 Effective U Coefficient

$$\beta_u = \frac{d\,\text{logit}(p_{\text{accept}})}{dU}$$

For current direct-acceptance GLM artifacts, this is the fitted logistic
coefficient whose feature label is `U`. Legacy pipeline artifacts may compute the
same effective coefficient as $w_U / \text{std}_U$ when `U` was standardized.
`build_real_data_config(u_coef=...)` can override this value for GLM-only
counterfactual acceptance sweeps.

- **Source:** `src/data/loader.py` :: `extract_glm_u_coef()`

### 9.2 Processed-Space Acceptance Coefficients

$$\text{logit}(p_{\text{accept}}) = \beta_0 + \beta_x^\top z_{\text{acc}}(x) + \beta_u\,u$$

Returns the processed model-feature coefficients used by the GLM acceptance
artifact, excluding the generated `U` column from `x_feature_names` and reporting
the `U` coefficient separately.

- **Source:** `src/data/loader.py` :: `extract_glm_acceptance_coefficients()`
- **Notes:** Legacy pipeline artifacts may report churn coefficients instead.

### 9.3 Linear Loss Coefficients

$$\hat{Y}(x) = \gamma_0 + \gamma_x^\top x$$

Extracts intercept and per-feature coefficients from a fitted
`LinearRegression`.

- **Source:** `src/data/loader.py` :: `extract_linear_loss_coefficients()`
