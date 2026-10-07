import BootstrapCalibration.GaussianPairsMSE
import BootstrapCalibration.IntegratedCalibration
import BootstrapCalibration.QuadraticStatistic

noncomputable section
open MeasureTheory Matrix
open scoped BigOperators

namespace BootstrapCalibration

variable {ι α : Type} [Fintype ι] [Fintype α] [MeasurableSpace α]

/-- An explicit finite-design error bound: average covariance discrepancy of
two independent resampling draws sharing the same observation noise. -/
def covarianceCalibrationError (P : FiniteLaw α) (K : α → Matrix (Fin 3) ι ℝ) : ℝ :=
  ∑ a, ∑ b, P.weight a * P.weight b *
    min 1 (Real.sqrt (∑ i, ∑ j,
      (((fromRows (K a) (K b) * (fromRows (K a) (K b))ᵀ - 1) :
        Matrix (Fin 3 ⊕ Fin 3) (Fin 3 ⊕ Fin 3) ℝ) i j)^2))

/-- Gaussian orthonormal projection preserves the coefficient reference law. -/
theorem standardGaussian_projection (U : Matrix (Fin 3) ι ℝ) (hU : U * Uᵀ = 1) :
    (standardGaussianMeasure ι).map (fun z => U *ᵥ z) = standardGaussianMeasure (Fin 3) := by
  have h := standard_linearImage_eq_of_gram_eq U (1 : Matrix (Fin 3) (Fin 3) ℝ)
    (by simpa using hU)
  simpa using h

/-- A complete finite-sample Gaussian calibration theorem with an explicit
finite-sum design error, not an assumed bootstrap approximation. The score
is the actual supremum over all real quadratic actions. `hcutoffRank` permits
the sorted conservative score bounds used by calibration. -/
theorem finite_sample_gaussian_calibration
    (P : FiniteLaw α) (U : Matrix (Fin 3) ι ℝ) (hU : U * Uᵀ = 1)
    (K : α → Matrix (Fin 3) ι ℝ)
    (L : QuadraticSpace ≃L[ℝ] QuadraticSpace)
    (B k : ℕ) (cutoff : (ι → ℝ) → Draws α B → ℝ)
    (hcutoffMeas : Measurable (fun z : (ι → ℝ) × Draws α B => cutoff z.1 z.2))
    (hcutoffRank : ∀ z x, k ≤ countHits
      (fun a => quadraticPiSupremum L (K a *ᵥ z) ≤ cutoff z x) B x)
    (p tau eta : ℝ) (hp1 : p ≤ 1) (htau : 0 < tau) (htaup : tau < p)
    (heta : 0 ≤ eta) (hrank : binomialTail p B k ≤ eta) :
    p - tau - eta - covarianceCalibrationError P K / tau^2 ≤
      ((standardGaussianMeasure ι).prod (P.iid B).toPMF.toMeasure).real
        {z | quadraticPiSupremum L (U *ᵥ z.1) ≤ cutoff z.1 z.2} := by
  have hp : 0 ≤ p := by linarith
  have hpt : 0 < p - tau := sub_pos.mpr htaup
  have hpt1 : p - tau < 1 := by linarith
  have hg := (quadraticPiSupremum_continuous L).measurable
  have hac : standardGaussianMeasure (Fin 3) ≪ volume :=
    withDensity_absolutelyContinuous _ _
  obtain ⟨t, ht⟩ := quadraticPiSupremum_threshold (standardGaussianMeasure (Fin 3))
    hac L (p - tau) hpt hpt1
  have htrue : (standardGaussianMeasure ι).real
      {z | quadraticPiSupremum L (U *ᵥ z) ≤ t} = p - tau := by
    rw [← standardGaussian_projection U hU] at ht
    simpa only [measureReal_def,
      Measure.map_apply (measurable_matrix_mulVec U) (measurableSet_le hg measurable_const)] using ht
  apply P.coverage_of_pair_comparisons (standardGaussianMeasure ι)
    (fun z => quadraticPiSupremum L (U *ᵥ z)) (hg.comp (measurable_matrix_mulVec U))
    (fun z a => quadraticPiSupremum L (K a *ᵥ z))
    (fun a => hg.comp (measurable_matrix_mulVec (K a)))
    B k cutoff hcutoffMeas hcutoffRank t p tau eta (covarianceCalibrationError P K)
    htau hp hp1 hpt.le heta htrue hrank
  · exact fun a b => gaussian_pair_centered_bound (K a) (K b)
      (quadraticPiSupremum L) hg t (p - tau) hpt.le hpt1.le ht
  · rfl

end BootstrapCalibration
