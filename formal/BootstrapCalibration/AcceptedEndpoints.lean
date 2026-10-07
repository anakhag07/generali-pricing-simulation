import BootstrapCalibration.StudentizedCalibration

/-! Numerical score certificates need only be provided on accepted resamples.
Rejected count vectors receive a zero extension, which suffices for the finite
law theorem because their normalized coefficient map is zero. -/

noncomputable section
open MeasureTheory Matrix
open scoped Classical
namespace BootstrapCalibration

theorem pairsCoefficientMap_eq_zero_of_rejected {n : ℕ}
    (Q : Matrix (Fin n) (Fin 3) ℝ) (w : Draws (Fin n) n)
    (hw : ¬ pairsAccept Q w) : pairsCoefficientMap Q w = 0 := by
  have hdet : ¬ IsUnit (OLS.gram Q (pairsCountMatrix w)).det := by
    intro h
    exact hw ((Matrix.isUnit_iff_isUnit_det _).mpr h)
  simp only [pairsCoefficientMap, OLS.coefficientMap,
    Matrix.nonsing_inv_apply_not_isUnit _ hdet, Matrix.zero_mul]

namespace QuadraticCalibrationDesign
variable (D : QuadraticCalibrationDesign)

theorem studentizedScore_eq_zero_of_rejected
    (z : Fin D.n → ℝ) (w : Draws (Fin D.n) D.n)
    (hw : ¬ pairsAccept D.normalized w) : D.studentizedScore z w = 0 := by
  simp only [studentizedScore, rawScore,
    pairsCoefficientMap_eq_zero_of_rejected D.normalized w hw,
    Matrix.zero_mulVec, map_zero, zero_div]

/-- Preserve certified endpoints on accepted draws and fill unused rejected
draws with zero. No score or measurability obligation is imposed there. -/
def acceptedEndpoints
    (upper : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (z : Fin D.n → ℝ) (w : Draws (Fin D.n) D.n) : ℝ :=
  if pairsAccept D.normalized w then upper z w else 0

theorem acceptedEndpoints_eq_of_accepted
    (upper : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (z : Fin D.n → ℝ) (w : Draws (Fin D.n) D.n)
    (hw : pairsAccept D.normalized w) : D.acceptedEndpoints upper z w = upper z w := by
  simp only [acceptedEndpoints, if_pos hw]

theorem acceptedEndpoints_measurable
    (upper : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (hupper : ∀ w, pairsAccept D.normalized w → Measurable (fun z => upper z w))
    (w : Draws (Fin D.n) D.n) : Measurable (fun z => D.acceptedEndpoints upper z w) := by
  by_cases hw : pairsAccept D.normalized w
  · simpa only [acceptedEndpoints, if_pos hw] using hupper w hw
  · simp only [acceptedEndpoints, if_neg hw]
    exact measurable_const

/-- An accepted-only certificate supplies the complete score hypothesis of
the finite-sample calibration theorem after the harmless zero extension. -/
theorem acceptedEndpoints_upper
    (upper : (Fin D.n → ℝ) → Draws (Fin D.n) D.n → ℝ)
    (hupper : ∀ z w, pairsAccept D.normalized w → D.studentizedScore z w ≤ upper z w) :
    ∀ z w, D.studentizedScore z w ≤ D.acceptedEndpoints upper z w := by
  intro z w
  by_cases hw : pairsAccept D.normalized w
  · rw [D.acceptedEndpoints_eq_of_accepted upper z w hw]
    exact hupper z w hw
  · rw [D.studentizedScore_eq_zero_of_rejected z w hw]
    simp only [acceptedEndpoints, if_neg hw, le_refl]

end QuadraticCalibrationDesign
end BootstrapCalibration
