import BootstrapCalibration.StableCovariance

/-!
Finite-law aggregation of the stable covariance estimate. Conditioning two
accepted resamples costs the square of the one-draw acceptance probability on
the unstable event, but only its first power in the stable Cauchy--Schwarz term.
-/

noncomputable section
open scoped BigOperators Classical
namespace BootstrapCalibration.BootstrapError
open StableCovariance
variable {Ω : Type} [Fintype Ω]

/-- Cauchy--Schwarz under an actual finite probability law. -/
theorem expectation_le_rms (P : FiniteLaw Ω) (f : Ω → ℝ) :
    P.expectation f ≤ rms P f := by
  have hs := Finset.sum_sq_le_sum_mul_sum_of_sq_eq_mul Finset.univ
    (r := fun x => P.weight x * f x)
    (f := fun x => P.weight x * (f x)^2) (g := P.weight)
    (fun x _ => mul_nonneg (P.nonneg x) (sq_nonneg (f x)))
    (fun x _ => P.nonneg x) (fun x _ => by ring)
  change (P.expectation f)^2 ≤ P.expectation (fun x => (f x)^2) * (∑ x, P.weight x) at hs
  rw [P.total, mul_one, ← rms_sq] at hs
  have hn := rms_nonneg P f
  nlinarith

/-- A stable event contained in acceptance retains its unconditional moment,
divided by the actual acceptance mass. -/
theorem accepted_stable_second_moment (P : FiniteLaw Ω) (A S : Ω → Prop)
    (hr : 0 < P.mass A) (hSA : ∀ x, S x → A x) (f : Ω → ℝ) :
    (P.accepted A hr).expectation (fun x => (if S x then f x else 0)^2) =
      P.expectation (fun x => if S x then (f x)^2 else 0) / P.mass A := by
  rw [FiniteLaw.accepted_expectation]
  congr 1
  apply P.expectation_congr
  intro x
  by_cases hs : S x
  · simp [hs, hSA x hs]
  · simp [hs]

/-- The stable contribution after conditioning is bounded by the square root
of its unconditional second moment, divided by the one-draw acceptance mass. -/
theorem accepted_stable_expectation_bound (P : FiniteLaw Ω) (A S : Ω → Prop)
    (r v : ℝ) (hr : 0 < r) (hmass : P.mass A = r^2)
    (hSA : ∀ x, S x → A x) (f : Ω → ℝ) (hv : 0 ≤ v)
    (hmoment : P.expectation (fun x => if S x then (f x)^2 else 0) ≤ v) :
    (P.accepted A (by rw [hmass]; positivity)).expectation
        (fun x => if S x then f x else 0) ≤ Real.sqrt v / r := by
  let Q := P.accepted A (by rw [hmass]; positivity)
  let g : Ω → ℝ := fun x => if S x then f x else 0
  have hs : (rms Q g)^2 ≤ v / r^2 := by
    rw [rms_sq]
    change (P.accepted A _).expectation (fun x => (if S x then f x else 0)^2) ≤ _
    rw [accepted_stable_second_moment P A S _ hSA f, hmass]
    exact div_le_div_of_nonneg_right hmoment (sq_nonneg r)
  have ht : (Real.sqrt v / r)^2 = v / r^2 := by
    rw [div_pow, Real.sq_sqrt hv]
  have htarget : 0 ≤ Real.sqrt v / r := div_nonneg (Real.sqrt_nonneg v) hr.le
  have hnorm : rms Q g ≤ Real.sqrt v / r := by
    rw [← ht] at hs
    exact (sq_le_sq₀ (rms_nonneg Q g) htarget).mp hs
  exact (expectation_le_rms Q g).trans hnorm

/-- Actual accepted-law expectation of the clipped covariance discrepancy.
Here `A` is two-draw acceptance with mass `r²`, and `S` is a stable subevent.
No conditional moment or approximation inequality is assumed. -/
theorem accepted_clipped_error_bound (P : FiniteLaw Ω) (A S : Ω → Prop)
    (r b v : ℝ) (hr : 0 < r) (hmass : P.mass A = r^2)
    (hSA : ∀ x, S x → A x) (f : Ω → ℝ)
    (hv : 0 ≤ v) (hbad : P.mass (fun x => ¬ S x) ≤ b)
    (hmoment : P.expectation (fun x => if S x then (f x)^2 else 0) ≤ v) :
    (P.accepted A (by rw [hmass]; positivity)).expectation (fun x => min 1 (f x)) ≤
      min 1 (b / r^2 + Real.sqrt v / r) := by
  let Q := P.accepted A (by rw [hmass]; positivity)
  have hupper : Q.expectation (fun x => min 1 (f x)) ≤ 1 := by
    have h := Q.expectation_mono (fun x => min_le_left (1 : ℝ) (f x))
    simpa only [FiniteLaw.expectation_const] using h
  have hsplit : Q.expectation (fun x => min 1 (f x)) ≤
      Q.mass (fun x => ¬ S x) + Q.expectation (fun x => if S x then f x else 0) := by
    have hp (x : Ω) : min 1 (f x) ≤
        (if ¬ S x then 1 else 0) + (if S x then f x else 0) := by
      by_cases hs : S x
      · simp [hs]
      · simp [hs]
    have h := Q.expectation_mono hp
    have hind : Q.expectation (fun x => if ¬ S x then (1 : ℝ) else 0) =
        Q.mass (fun x => ¬ S x) := by
      unfold FiniteLaw.expectation FiniteLaw.mass
      apply Finset.sum_congr rfl
      intro x hx
      by_cases hs : S x <;> simp [hs]
    rw [FiniteLaw.expectation_add, hind] at h
    exact h
  have hbadQ : Q.mass (fun x => ¬ S x) ≤ b / r^2 := by
    change (P.accepted A _).mass _ ≤ _
    rw [FiniteLaw.accepted_mass, hmass]
    apply div_le_div_of_nonneg_right _ (sq_nonneg r)
    apply le_trans _ hbad
    unfold FiniteLaw.mass
    apply Finset.sum_le_sum
    intro x hx
    by_cases hs : S x <;> by_cases ha : A x <;> simp [hs, ha, P.nonneg x]
  have hstable := accepted_stable_expectation_bound P A S r v hr hmass hSA f hv hmoment
  exact le_min hupper (hsplit.trans (add_le_add hbadQ hstable))

/-- The explicit covariance constants used in the finite-N theorem. -/
theorem accepted_covariance_error_bound (P : FiniteLaw Ω) (A S : Ω → Prop)
    (r a α d e : ℝ) (hr : 0 < r) (_ha : a < 1) (hmass : P.mass A = r^2)
    (hSA : ∀ x, S x → A x) (f : Ω → ℝ)
    (hbad : P.mass (fun x => ¬ S x) ≤ 2 * α / a^2)
    (hmoment : P.expectation (fun x => if S x then (f x)^2 else 0) ≤
      (2 * d^2 + 2 * e^2) / (1 - a)^4) :
    (P.accepted A (by rw [hmass]; positivity)).expectation (fun x => min 1 (f x)) ≤
      min 1 (2 * α / (a^2 * r^2) +
        Real.sqrt (2 * d^2 + 2 * e^2) / ((1 - a)^2 * r)) := by
  have hv : 0 ≤ (2 * d^2 + 2 * e^2) / (1 - a)^4 := by positivity
  have h := accepted_clipped_error_bound P A S r (2 * α / a^2)
    ((2 * d^2 + 2 * e^2) / (1 - a)^4) hr hmass hSA f hv hbad hmoment
  have hsqrt : Real.sqrt ((2 * d^2 + 2 * e^2) / (1 - a)^4) =
      Real.sqrt (2 * d^2 + 2 * e^2) / (1 - a)^2 := by
    rw [Real.sqrt_div (by positivity), show (1 - a)^4 = ((1 - a)^2)^2 by ring,
      Real.sqrt_sq (sq_nonneg _)]
  rw [hsqrt, div_div, div_div] at h
  exact h

end BootstrapCalibration.BootstrapError
