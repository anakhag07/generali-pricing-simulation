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

- **Source:** `src/objective/policy.py` :: `FeatureMap`, `IdentityFeatureMap`,
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

### 3.1 Quadratic Objective

For a configured parameter dimension $$d$$:

$$J(\theta) = \frac{1}{2}\|\theta\|_2^2 = \frac{1}{2}\sum_{j=1}^{d}\theta_j^2$$

**Gradient and Hessian:**

$$\nabla J(\theta) = \theta, \qquad \nabla^2J(\theta) = I_d$$

- **Source:** `src/objective/objectives/quadratic.py` :: `QuadraticObjective`
- **Notes:** This is a direct theta-space objective and does not compose through
  a policy. It is 1-strongly convex and 1-smooth, with unique minimizer
  $$\theta^*=0$$ and minimum value $$J(\theta^*)=0$$. The required `x_batch`
  argument is ignored.

### 3.2 Fixed Regression Objective

$$f(u;\, x) = a(x, u)\,\bigl(\ell(x) - r(u)\bigr)$$

where:
- Acceptance: $a(x, u) = \sigma(\beta_1^\top x + \beta_2\, u)$
- Loss: $\ell(x) = \beta_3^\top x$
- Revenue: $r(u) = \beta_4\, u$

**Gradient w.r.t. $u$:**

$$\frac{\partial f}{\partial u} = \frac{\partial a}{\partial u}\,(\ell - r) - a\,\beta_4$$

where $\frac{\partial a}{\partial u} = a(1 - a)\,\beta_2$.

- **Source:** `src/objective/objectives/fixed_regression.py` :: `FixedRegressionObjective`
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

- **Source:** `src/objective/objectives/planted_logistic.py` :: `PlantedLogisticObjective`
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
- **XGBoost (numerical):** central FD with $\epsilon = 10^{-4}$

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

- **Source:** `src/objective/objectives/model_based.py` :: `ModelBasedObjective`
  - `_value_batch()` — per-sample values
  - `_glm_acceptance_proba()` — coefficient-backed GLM acceptance probability when available
  - `_grad_u_batch()` — per-sample $\partial f/\partial u$
  - `_d_acceptance_du_batch()` — analytical or FD acceptance derivative
  - `_acceptance_penalty()` — penalty value and gradient scale
- **Source:** `src/objective/objectives/prepared_glm.py` :: `PreparedGLMObjective`, `PreparedGLMBatch`, `prepare_glm_objective()`
  - Uses the same GLM formulas after materializing `base_logit`, `loss`, `premium`, and policy features into a compact numeric batch.
- **Source:** `src/objective/objectives/jax_prepared_glm.py` :: `JaxPreparedGLMObjective`, `JaxPreparedGLMScipyAdapter`, `prepare_jax_glm_objective()`
  - Uses the same prepared GLM formulas in JAX for fixed-batch SciPy callbacks. The explicit constraint-margin adapter uses $$\bar{a}(\theta) - \alpha$$, equivalent to SciPy's existing lower-bound form $$\bar{a}(\theta) \ge \alpha$$.
- **Source:** `src/experiments/sensitivity_buckets.py` :: `glm_price_derivative_matrix()`, `glm_price_sensitivity_scores()`, `glm_price_sensitivity_matrix()`, `split_sensitivity_tertiles()`
- **Source:** `src/reporting/visualization.py` :: `_plot_policy_delta_u_histograms()`, `_plot_policy_delta_u_by_elasticity()`, `_plot_policy_objective_contribution_summary()`

**Lagrangian scalarization** (lambda sweep path):

$$J_{\lambda}(\theta) = J(\theta) + \lambda\,(\text{floor} - \bar{a}(\theta))$$

where $$J(\theta) = \mathbb{E}[f(\pi_\theta(x); x)]$$ and
$$\bar{a}(\theta) = \mathbb{E}[a(x, \pi_\theta(x))]$$.

**Gradient w.r.t. $\theta$:**

$$\nabla_\theta J_{\lambda}(\theta) = \nabla_\theta J(\theta) - \lambda\,\nabla_\theta \bar{a}(\theta)$$

- **Source:** `src/objective/objectives/model_based.py` :: `ModelBasedObjective.value()`, `ModelBasedObjective.grad()`, `ModelBasedObjective._lagrangian_adjustment()`
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

For a policy-free theta-space objective such as `QuadraticObjective`, the same
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
- **Constraint value source:** `src/objective/objectives/model_based.py` :: `mean_acceptance()`
- **Constraint gradient source:** `src/objective/objectives/model_based.py` :: `mean_acceptance_grad()`
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
