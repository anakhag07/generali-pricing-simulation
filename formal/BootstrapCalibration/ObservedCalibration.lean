import BootstrapCalibration.StudentizedCalibration
import BootstrapCalibration.RegressionInterpretation
import BootstrapCalibration.AcceptedEndpoints

noncomputable section
open MeasureTheory Matrix

namespace BootstrapCalibration.QuadraticCalibrationDesign
open RegressionInterpretation

variable (D : QuadraticCalibrationDesign)

def responses (β : Fin 3 → ℝ) (σ : ℝ) (z : Fin D.n → ℝ) : Fin D.n → ℝ :=
  D.matrix *ᵥ β + σ • z

def fittedMean (β : Fin 3 → ℝ) (σ : ℝ) (z : Fin D.n → ℝ) (u : ℝ) : ℝ :=
  prediction (olsCoefficient D.matrix (D.responses β σ z)) u

/-- The band is computed using the residual SD of the observed responses,
the sorted certified bootstrap scores, and the inverse-Gram standard error. -/
def bandHalfWidth (β : Fin 3 → ℝ) (σ : ℝ)
    (Tplus : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (B : ℕ) (j : Fin B) (z : Fin D.n → ℝ) (x : Draws (Draws (Fin D.n) D.n) B) (u : ℝ) : ℝ :=
  fittedResidualStd D.n D.matrix (D.responses β σ z) *
    sortedCritical (Tplus z) B x j * standardErrorShape D.matrix u

def predictorRadius (fixed : ℝ → ℝ) (β : Fin 3 → ℝ) (σ : ℝ)
    (Tplus : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (B : ℕ) (j : Fin B) (z : Fin D.n → ℝ) (x : Draws (Draws (Fin D.n) D.n) B) (u : ℝ) : ℝ :=
  radius (fixed u) (D.fittedMean β σ z u) (D.bandHalfWidth β σ Tplus B j z x u)

/-- The complete original-coordinate confidence certificate. The measurable
event has the claimed joint probability over observation noise and accepted
bootstrap draws; on it the calibrated radius bounds the frozen predictor's
error at every real action, with excess at most twice the bootstrap half-width.
No coverage or bootstrap-approximation premise appears in this statement. -/
theorem finite_sample_predictor_certificate
    (β : Fin 3 → ℝ) (σ : ℝ) (hσ : 0 < σ) (fixed : ℝ → ℝ)
    (Tplus : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (hTplusMeas : ∀ w, Measurable (fun z => Tplus z w))
    (hTplus : ∀ z w, D.studentizedScore z w ≤ Tplus z w)
    (B : ℕ) (j : Fin B) (p tau eta a : ℝ)
    (hp1 : p ≤ 1) (htau : 0 < tau) (htaup : tau < p) (heta : 0 ≤ eta)
    (ha0 : 0 < a) (ha1 : a < 1) (hrank : binomialTail p B (j.val + 1) ≤ eta) :
    ∃ good : Set ((Fin D.n → ℝ) × Draws (Draws (Fin D.n) D.n) B),
      MeasurableSet good ∧
      p - tau - eta - D.error a / tau^2 ≤
        ((standardGaussianMeasure (Fin D.n)).prod (D.law.iid B).toPMF.toMeasure).real good ∧
      ∀ z ∈ good, ∀ u : ℝ,
        |fixed u - prediction β u| ≤ D.predictorRadius fixed β σ Tplus B j z.1 z.2 u ∧
        (0 ≤ D.predictorRadius fixed β σ Tplus B j z.1 z.2 u - |fixed u - prediction β u| ∧
          D.predictorRadius fixed β σ Tplus B j z.1 z.2 u - |fixed u - prediction β u| ≤
            2 * D.bandHalfWidth β σ Tplus B j z.1 z.2 u) := by
  let good : Set ((Fin D.n → ℝ) × Draws (Draws (Fin D.n) D.n) B) :=
    {z | quadraticPiSupremum D.directions (D.normalizedᵀ *ᵥ z.1) ≤
      scaledSortedCutoff D.residualScale Tplus B j z.1 z.2}
  have hgood : MeasurableSet good :=
    measurableSet_le
      ((quadraticPiSupremum_continuous D.directions).measurable.comp
        ((measurable_matrix_mulVec D.normalizedᵀ).comp measurable_fst))
      ((D.residualScale_measurable.comp measurable_fst).mul
        (measurable_sortedCritical_joint Tplus hTplusMeas B j))
  refine ⟨good, hgood,
    D.finite_sample_studentized_bound Tplus hTplusMeas hTplus B j p tau eta a
      hp1 htau htaup heta ha0 ha1 hrank, ?_⟩
  intro z hz u
  have h := original_regression_envelope D.matrix D.gram_isUnit β z.1 σ
    (scaledSortedCutoff D.residualScale Tplus B j z.1 z.2) hσ.le hz fixed u
  have hsd := fittedResidualStd_eq D.n D.matrix D.gram_isUnit β z.1 σ hσ.le
  simpa only [predictorRadius, bandHalfWidth, fittedMean, responses, hsd,
    scaledSortedCutoff, residualScale, mul_assoc] using h

/-- The complete certificate with numerical endpoint obligations restricted
to accepted resamples. Rejected draws use a proved harmless zero extension. -/
theorem finite_sample_predictor_certificate_accepted
    (β : Fin 3 → ℝ) (σ : ℝ) (hσ : 0 < σ) (fixed : ℝ → ℝ)
    (Tplus : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (hTplusMeas : ∀ w, pairsAccept D.normalized w → Measurable (fun z => Tplus z w))
    (hTplus : ∀ z w, pairsAccept D.normalized w → D.studentizedScore z w ≤ Tplus z w)
    (B : ℕ) (j : Fin B) (p tau eta a : ℝ)
    (hp1 : p ≤ 1) (htau : 0 < tau) (htaup : tau < p) (heta : 0 ≤ eta)
    (ha0 : 0 < a) (ha1 : a < 1) (hrank : binomialTail p B (j.val + 1) ≤ eta) :
    ∃ good : Set ((Fin D.n → ℝ) × Draws (Draws (Fin D.n) D.n) B),
      MeasurableSet good ∧
      p - tau - eta - D.error a / tau^2 ≤
        ((standardGaussianMeasure (Fin D.n)).prod (D.law.iid B).toPMF.toMeasure).real good ∧
      ∀ z ∈ good, ∀ u : ℝ,
        |fixed u - prediction β u| ≤ D.predictorRadius fixed β σ (D.acceptedEndpoints Tplus) B j z.1 z.2 u ∧
        (0 ≤ D.predictorRadius fixed β σ (D.acceptedEndpoints Tplus) B j z.1 z.2 u - |fixed u - prediction β u| ∧
          D.predictorRadius fixed β σ (D.acceptedEndpoints Tplus) B j z.1 z.2 u - |fixed u - prediction β u| ≤
            2 * D.bandHalfWidth β σ (D.acceptedEndpoints Tplus) B j z.1 z.2 u) := by
  exact D.finite_sample_predictor_certificate β σ hσ fixed (D.acceptedEndpoints Tplus)
    (D.acceptedEndpoints_measurable Tplus hTplusMeas)
    (D.acceptedEndpoints_upper Tplus hTplus)
    B j p tau eta a hp1 htau htaup heta ha0 ha1 hrank

end BootstrapCalibration.QuadraticCalibrationDesign
