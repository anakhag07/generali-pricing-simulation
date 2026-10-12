import BootstrapCalibration.FiniteBootstrap

namespace BootstrapCalibration

/-- A shared nonnegative scale preserves the conservative rank condition.
Thus upper bounds on studentized bootstrap scores calibrate the corresponding
unstudentized coefficient statistic with the scaled critical value. -/
theorem scaled_conservative_rank {α : Type} (B k : ℕ) (x : Draws α B)
    (T upper : α → ℝ) (scale c : ℝ) (hscale : 0 ≤ scale)
    (hupper : ∀ a, T a ≤ scale * upper a)
    (hrank : k ≤ countHits (fun a => upper a ≤ c) B x) :
    k ≤ countHits (fun a => T a ≤ scale * c) B x := by
  apply hrank.trans
  apply countHits_mono
  intro a ha
  exact (hupper a).trans (mul_le_mul_of_nonneg_left ha hscale)

/-- The positive fitted-scale ratio cancels exactly in the actual band width. -/
theorem scale_cancellation (σ fitted c s : ℝ) (hσ : σ ≠ 0) :
    σ * ((fitted / σ) * c) * s = fitted * c * s := by
  field_simp

end BootstrapCalibration
