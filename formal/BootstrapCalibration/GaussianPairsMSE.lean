import BootstrapCalibration.GaussianLinearImage
import BootstrapCalibration.TwoReplicas

noncomputable section
open MeasureTheory Matrix
open scoped BigOperators Classical

namespace BootstrapCalibration

variable {ι κ α : Type} [Fintype ι] [Fintype κ] [DecidableEq κ] [Fintype α]

omit [DecidableEq κ] in
private theorem gaussian_indicator_mean (g : (κ → ℝ) → ℝ) (hg : Measurable g) (t : ℝ) :
    (∫ x, (if g x ≤ t then (1 : ℝ) else 0) ∂standardGaussianMeasure κ) =
      (standardGaussianMeasure κ).real {x | g x ≤ t} := by
  have hs : MeasurableSet {x | g x ≤ t} := measurableSet_le hg measurable_const
  simpa only [Set.indicator, integral_const, smul_eq_mul, mul_one, measureReal_restrict_apply,
    MeasurableSet.univ, Set.univ_inter] using
    (integral_indicator (μ := standardGaussianMeasure κ)
      (f := fun _ : κ → ℝ => (1 : ℝ)) hs)

omit [DecidableEq κ] in
private theorem gaussian_indicator_integrable (g : (κ → ℝ) → ℝ) (hg : Measurable g) (t : ℝ) :
    Integrable (fun x => if g x ≤ t then (1 : ℝ) else 0) (standardGaussianMeasure κ) := by
  simpa only [Set.indicator] using
    (integrable_const (1 : ℝ)).indicator (measurableSet_le hg measurable_const : MeasurableSet {x | g x ≤ t})

/-- Exact two-replica Gaussian comparison for any measurable coefficient
statistic. It compares the actual shared-noise linear images with independent
reference Gaussians, and takes no bootstrap-validity assumption. -/
theorem gaussian_pair_centered_bound (A B : Matrix κ ι ℝ)
    (g : (κ → ℝ) → ℝ) (hg : Measurable g) (t p : ℝ)
    (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (hpmean : (standardGaussianMeasure κ).real {x | g x ≤ t} = p) :
    (∫ z, ((if g (A *ᵥ z) ≤ t then (1 : ℝ) else 0) - p) *
      ((if g (B *ᵥ z) ≤ t then (1 : ℝ) else 0) - p) ∂standardGaussianMeasure ι) ≤
      min 1 (Real.sqrt (∑ i, ∑ j,
        (((fromRows A B * (fromRows A B)ᵀ - 1) : Matrix (κ ⊕ κ) (κ ⊕ κ) ℝ) i j)^2)) := by
  let f : (κ → ℝ) → ℝ := fun x => (if g x ≤ t then 1 else 0) - p
  have hf : Measurable f :=
    (Measurable.ite (measurableSet_le hg measurable_const) measurable_const measurable_const).sub
      measurable_const
  have hfmean : (∫ x, f x ∂standardGaussianMeasure κ) = 0 := by
    rw [integral_sub (gaussian_indicator_integrable g hg t) (integrable_const p),
      gaussian_indicator_mean g hg t, integral_const]
    simp [hpmean]
  let h : ((κ ⊕ κ) → ℝ) → ℝ :=
    fun x => f (fun i => x (Sum.inl i)) * f (fun i => x (Sum.inr i))
  have hh : Measurable h :=
    (hf.comp (measurable_pi_lambda _ (fun i => measurable_pi_apply (Sum.inl i)))).mul
      (hf.comp (measurable_pi_lambda _ (fun i => measurable_pi_apply (Sum.inr i))))
  have hlo : ∀ x, -(p * (1 - p)) ≤ h x := by
    intro x
    dsimp [h, f]
    split_ifs <;> nlinarith [mul_nonneg hp (sub_nonneg.mpr hp1)]
  have hhi : ∀ x, h x ≤ -(p * (1 - p)) + 1 := by
    intro x
    dsimp [h, f]
    split_ifs <;> nlinarith
  have href : (∫ x, h x ∂standardGaussianMeasure (κ ⊕ κ)) = 0 := by
    rw [standardGaussian_sum_integral_product f f, hfmean, zero_mul]
  have hc := gaussian_linearImage_bounded_test_le (fromRows A B) h hh
    (-(p * (1 - p))) hlo hhi
  rw [href, sub_zero, integral_map (measurable_matrix_mulVec _).aemeasurable
    hh.aestronglyMeasurable] at hc
  have hle := (le_abs_self _).trans hc
  simpa only [h, f, Matrix.fromRows_mulVec, Sum.elim_inl, Sum.elim_inr] using hle

/-- The finite-sample conditional-CDF MSE follows from the actual Gaussian
replica covariance errors averaged over the actual finite resampling law. -/
theorem gaussian_bootstrap_cdf_mse (P : FiniteLaw α) (K : α → Matrix κ ι ℝ)
    (g : (κ → ℝ) → ℝ) (hg : Measurable g) (t p : ℝ)
    (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (hpmean : (standardGaussianMeasure κ).real {x | g x ≤ t} = p) :
    (∫ z, (P.cdf (fun a => g (K a *ᵥ z)) t - p)^2 ∂standardGaussianMeasure ι) ≤
      ∑ a, ∑ b, P.weight a * P.weight b *
        min 1 (Real.sqrt (∑ i, ∑ j,
          (((fromRows (K a) (K b) * (fromRows (K a) (K b))ᵀ - 1) :
            Matrix (κ ⊕ κ) (κ ⊕ κ) ℝ) i j)^2)) := by
  apply P.cdf_mse_le_pair_bounds (standardGaussianMeasure ι)
    (fun z a => g (K a *ᵥ z) ≤ t)
    (fun a => measurableSet_le (hg.comp (measurable_matrix_mulVec _)) measurable_const)
    p hp hp1
  exact fun a b => gaussian_pair_centered_bound (K a) (K b) g hg t p hp hp1 hpmean

end BootstrapCalibration
