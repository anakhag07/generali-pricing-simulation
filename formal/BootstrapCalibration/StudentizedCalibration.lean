import BootstrapCalibration.Main
import BootstrapCalibration.ResidualScale

/-! The actual residual studentizer closes the arbitrary-scale interface in
`Main`: dividing by its zero value is harmless because every pairs perturbation
then vanishes. Thus certified upper endpoints of the studentized score provide
the actual finite-sample sorted calibration bound. -/

noncomputable section
open MeasureTheory Matrix
namespace BootstrapCalibration.QuadraticCalibrationDesign

variable (D : QuadraticCalibrationDesign)

/-- The fitted residual standard deviation divided by the true noise scale. -/
def residualScale (z : Fin D.n → ℝ) : ℝ :=
  ResidualScale.fittedScaleRatio D.n D.normalized z

theorem residualScale_measurable : Measurable D.residualScale := by
  unfold residualScale ResidualScale.fittedScaleRatio
  exact (((EuclideanSpace.equiv (Fin D.n) ℝ).symm.continuous.measurable).comp
    (measurable_matrix_mulVec (OLS.residual D.normalized))).norm.div_const _

theorem residualScale_nonneg (z : Fin D.n → ℝ) : 0 ≤ D.residualScale z :=
  div_nonneg (norm_nonneg _) (Real.sqrt_nonneg _)

theorem residualScale_positive_ae :
    ∀ᵐ z ∂standardGaussianMeasure (Fin D.n), 0 < D.residualScale z :=
  ResidualScale.fittedScaleRatio_pos_ae D.n (by have := D.sampleSize; omega)
    D.normalized D.orthonormal

theorem residual_zero_of_scale_zero (z : Fin D.n → ℝ) (hz : D.residualScale z = 0) :
    OLS.residual D.normalized *ᵥ z = 0 := by
  by_contra h
  have hpos := ResidualScale.fittedScaleRatio_pos D.n
    (by have := D.sampleSize; omega) D.normalized z h
  exact (ne_of_gt hpos) hz

/-- The nonstudentized supremum of the actual accepted-pairs coefficient perturbation. -/
def rawScore (z : Fin D.n → ℝ) (w : Draws (Fin D.n) D.n) : ℝ :=
  quadraticPiSupremum D.directions (pairsCoefficientMap D.normalized w *ᵥ z)

/-- The score computed with the one shared fitted residual standard deviation. -/
def studentizedScore (z : Fin D.n → ℝ) (w : Draws (Fin D.n) D.n) : ℝ :=
  D.rawScore z w / D.residualScale z

theorem rawScore_measurable (w : Draws (Fin D.n) D.n) :
    Measurable (fun z => D.rawScore z w) :=
  (quadraticPiSupremum_continuous D.directions).measurable.comp
    (measurable_matrix_mulVec (pairsCoefficientMap D.normalized w))

theorem studentizedScore_measurable (w : Draws (Fin D.n) D.n) :
    Measurable (fun z => D.studentizedScore z w) :=
  (D.rawScore_measurable w).div D.residualScale_measurable

theorem rawScore_nonneg (z : Fin D.n → ℝ) (w : Draws (Fin D.n) D.n) :
    0 ≤ D.rawScore z w := apply_nonneg (quadraticPiSupremum D.directions) _

theorem studentizedScore_nonneg (z : Fin D.n → ℝ) (w : Draws (Fin D.n) D.n) :
    0 ≤ D.studentizedScore z w :=
  div_nonneg (D.rawScore_nonneg z w) (D.residualScale_nonneg z)

/-- Zero residual scale forces every pairs perturbation to vanish, even on
singular count vectors to which Lean assigns a total inverse. -/
theorem rawScore_zero_of_scale_zero (z : Fin D.n → ℝ) (hz : D.residualScale z = 0)
    (w : Draws (Fin D.n) D.n) : D.rawScore z w = 0 := by
  have hres := D.residual_zero_of_scale_zero z hz
  unfold rawScore pairsCoefficientMap OLS.coefficientMap
  rw [← Matrix.mulVec_mulVec, hres, Matrix.mulVec_zero, map_zero]

/-- Exact cancellation holds on every data vector, including the zero-scale
null set, using Lean's total convention for division by zero. -/
theorem studentizedScore_reconstruct (z : Fin D.n → ℝ) (w : Draws (Fin D.n) D.n) :
    D.residualScale z * D.studentizedScore z w = D.rawScore z w := by
  by_cases hz : D.residualScale z = 0
  · rw [hz, zero_mul, D.rawScore_zero_of_scale_zero z hz w]
  · unfold studentizedScore
    exact mul_div_cancel₀ _ hz

/-- A certified upper endpoint for the studentized score gives the precise
scaled upper bound required by the finite-sample probability theorem. -/
theorem certified_studentized_upper
    (upper : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (hupper : ∀ z w, D.studentizedScore z w ≤ upper z w) :
    ∀ z w, quadraticPiSupremum D.directions (pairsCoefficientMap D.normalized w *ᵥ z) ≤
      D.residualScale z * upper z w := by
  intro z w
  change D.rawScore z w ≤ _
  rw [← D.studentizedScore_reconstruct z w]
  exact mul_le_mul_of_nonneg_left (hupper z w) (D.residualScale_nonneg z)

/-- Finite-sample coverage with the actual fitted residual studentizer and the
sorted conservative score endpoints Tplus. All error terms are explicit
functions of the original fixed quadratic design and the accepted pairs law. -/
theorem finite_sample_studentized_bound
    (Tplus : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (hTplusMeas : ∀ w, Measurable (fun z => Tplus z w))
    (hTplus : ∀ z w, D.studentizedScore z w ≤ Tplus z w)
    (B : ℕ) (j : Fin B) (p tau eta a : ℝ)
    (hp1 : p ≤ 1) (htau : 0 < tau) (htaup : tau < p) (heta : 0 ≤ eta)
    (ha0 : 0 < a) (ha1 : a < 1) (hrank : binomialTail p B (j.val + 1) ≤ eta) :
    p - tau - eta - D.error a / tau^2 ≤
      ((standardGaussianMeasure (Fin D.n)).prod (D.law.iid B).toPMF.toMeasure).real
        {z | quadraticPiSupremum D.directions (D.normalizedᵀ *ᵥ z.1) ≤
          scaledSortedCutoff D.residualScale Tplus B j z.1 z.2} :=
  D.finite_sample_bound D.residualScale D.residualScale_measurable D.residualScale_nonneg
    Tplus hTplusMeas (D.certified_studentized_upper Tplus hTplus)
    B j p tau eta a hp1 htau htaup heta ha0 ha1 hrank

/-- Exact supremum scores are a measurable valid instance of the conservative
endpoint interface; numerical certification may increase any endpoint. -/
theorem finite_sample_exact_studentized_bound
    (B : ℕ) (j : Fin B) (p tau eta a : ℝ)
    (hp1 : p ≤ 1) (htau : 0 < tau) (htaup : tau < p) (heta : 0 ≤ eta)
    (ha0 : 0 < a) (ha1 : a < 1) (hrank : binomialTail p B (j.val + 1) ≤ eta) :
    p - tau - eta - D.error a / tau^2 ≤
      ((standardGaussianMeasure (Fin D.n)).prod (D.law.iid B).toPMF.toMeasure).real
        {z | quadraticPiSupremum D.directions (D.normalizedᵀ *ᵥ z.1) ≤
          scaledSortedCutoff D.residualScale D.studentizedScore B j z.1 z.2} :=
  D.finite_sample_studentized_bound D.studentizedScore D.studentizedScore_measurable
    (fun _ _ => le_rfl) B j p tau eta a hp1 htau htaup heta ha0 ha1 hrank

end BootstrapCalibration.QuadraticCalibrationDesign
