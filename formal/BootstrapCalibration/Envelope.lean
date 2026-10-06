import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-!
# A frozen predictor's error certificate

These are deterministic lemmas. They do not assume or prove that a bootstrap
band has the asserted probability of covering truth. The statistical coverage
event is an explicit hypothesis of the transfer lemmas.
-/

namespace BootstrapCalibration

/-- Radius of the smallest symmetric interval, centered at `fixed`, which
contains the interval centered at `auxiliary` with nonnegative half-width `w`. -/
def radius (fixed auxiliary w : ℝ) : ℝ := |fixed - auxiliary| + w

theorem error_le_radius (fixed auxiliary truth w : ℝ)
    (h : |auxiliary - truth| ≤ w) :
    |fixed - truth| ≤ radius fixed auxiliary w := by
  calc
    |fixed - truth| = |(fixed - auxiliary) + (auxiliary - truth)| := by congr 1; ring
    _ ≤ |fixed - auxiliary| + |auxiliary - truth| := abs_add _ _
    _ ≤ radius fixed auxiliary w := add_le_add_left h _

/-- On auxiliary coverage, the excess certificate width is at most `2*w`. -/
theorem radius_accuracy (fixed auxiliary truth w : ℝ)
    (h : |auxiliary - truth| ≤ w) :
    0 ≤ radius fixed auxiliary w - |fixed - truth| ∧
      radius fixed auxiliary w - |fixed - truth| ≤ 2 * w := by
  constructor
  · exact sub_nonneg.mpr (error_le_radius fixed auxiliary truth w h)
  · have ht : |fixed - auxiliary| ≤ |fixed - truth| + w := by
      calc
        |fixed - auxiliary| = |(fixed - truth) + (truth - auxiliary)| := by congr 1; ring
        _ ≤ |fixed - truth| + |truth - auxiliary| := abs_add _ _
        _ = |fixed - truth| + |auxiliary - truth| := by rw [abs_sub_comm truth auxiliary]
        _ ≤ |fixed - truth| + w := add_le_add_left h _
    unfold radius
    linarith

/-- Minimality uses only containment of both endpoints of the auxiliary band. -/
theorem radius_minimal (fixed auxiliary w R : ℝ) (hw : 0 ≤ w)
    (hl : |fixed - (auxiliary - w)| ≤ R)
    (hu : |fixed - (auxiliary + w)| ≤ R) :
    radius fixed auxiliary w ≤ R := by
  unfold radius
  by_cases h : 0 ≤ fixed - auxiliary
  · rw [abs_of_nonneg h]
    have hs : 0 ≤ fixed - (auxiliary - w) := by linarith
    rw [abs_of_nonneg hs] at hl
    linarith
  · have hn : fixed - auxiliary ≤ 0 := le_of_not_ge h
    rw [abs_of_nonpos hn]
    have hs : fixed - (auxiliary + w) ≤ 0 := by linarith
    rw [abs_of_nonpos hs] at hu
    linarith

theorem interval_containment (fixed auxiliary w value : ℝ)
    (h : |auxiliary - value| ≤ w) :
    |fixed - value| ≤ radius fixed auxiliary w :=
  error_le_radius fixed auxiliary value w h

/-- Function-wide transfer applies on an arbitrary domain, including all ℝ. -/
theorem simultaneous_transfer {A : Type*} (fixed auxiliary truth w : A → ℝ)
    (h : ∀ u, |auxiliary u - truth u| ≤ w u) :
    ∀ u, |fixed u - truth u| ≤ radius (fixed u) (auxiliary u) (w u) := by
  intro u
  exact error_le_radius _ _ _ _ (h u)

end BootstrapCalibration
