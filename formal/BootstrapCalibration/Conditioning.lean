import Mathlib.Tactic

/-!
# Effect of rejecting singular resamples

For an event A and accepted-resample event E, write r=P(E),
x=P(A ∩ E), and y=P(A ∩ Eᶜ). The probability after rejection is x/r;
the unconditional probability is x+y. The bound below is uniform in A.
It is the algebraic part of the total-variation conditioning bound.
-/

namespace BootstrapCalibration

theorem conditioning_error (r x y : ℝ)
    (hr : 0 < r) (hr1 : r ≤ 1)
    (hx0 : 0 ≤ x) (hxr : x ≤ r)
    (hy0 : 0 ≤ y) (hyr : y ≤ 1 - r) :
    |x / r - (x + y)| ≤ 1 - r := by
  apply abs_le.mpr
  have hxdiv : x / r * r = x := div_mul_cancel₀ x (ne_of_gt hr)
  have hmul : 0 ≤ x * (1 - r) := mul_nonneg hx0 (sub_nonneg.mpr hr1)
  have hmul_le : x * (1 - r) ≤ r * (1 - r) :=
    mul_le_mul_of_nonneg_right hxr (sub_nonneg.mpr hr1)
  constructor <;> nlinarith

/-- Algebraic specialization when the joint acceptance probability is r². -/
theorem two_draw_conditioning_error (r x y : ℝ)
    (hr : 0 < r) (hr1 : r ≤ 1)
    (hx0 : 0 ≤ x) (hxr : x ≤ r ^ 2)
    (hy0 : 0 ≤ y) (hyr : y ≤ 1 - r ^ 2) :
    |x / r ^ 2 - (x + y)| ≤ 1 - r ^ 2 := by
  apply conditioning_error (r ^ 2) x y (sq_pos_of_pos hr) ?_ hx0 hxr hy0 hyr
  nlinarith only [hr, hr1, sq_nonneg (r - 1)]

end BootstrapCalibration
