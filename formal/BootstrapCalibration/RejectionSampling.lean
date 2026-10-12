import BootstrapCalibration.AcceptedLaw

/-!
Finite retry sampling from explicit iid attempts. Failure is an actual outcome;
conditional on returning a value, the return law is exactly the accepted law.
-/

noncomputable section
open scoped BigOperators Classical
namespace BootstrapCalibration

/-- Return the first accepted value, or none when the attempt budget is exhausted. -/
def firstAccepted {α : Type} (A : α → Prop) : (m : ℕ) → Draws α m → Option α
  | 0, _ => none
  | m+1, x => if A x.1 then some x.1 else firstAccepted A m x.2

/-- The event that finite retry sampling returns a value in E. -/
def returnsIn {α : Type} (A E : α → Prop) (m : ℕ) (x : Draws α m) : Prop :=
  ∃ a, firstAccepted A m x = some a ∧ E a

namespace FiniteLaw
variable {α : Type} [Fintype α] (P : FiniteLaw α)

lemma iid_mass_step (m : ℕ) (E : Draws α (m+1) → Prop) :
    (P.iid (m+1)).mass E = ∑ a, P.weight a * (P.iid m).mass (fun x => E (a,x)) := by
  change (∑ x : α × Draws α m, if E x then P.weight x.1 * (P.iid m).weight x.2 else 0) = _
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro a ha
  unfold mass
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro x hx
  by_cases h : E (a,x) <;> simp [h]

lemma firstAccepted_failure_step (A : α → Prop) (m : ℕ) :
    (P.iid (m+1)).mass (fun x => firstAccepted A (m+1) x = none) =
      (1 - P.mass A) * (P.iid m).mass (fun x => firstAccepted A m x = none) := by
  rw [iid_mass_step, ← P.mass_compl A]
  unfold mass
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro a ha
  by_cases hA : A a <;> simp [firstAccepted, hA]
  left
  apply Finset.sum_congr rfl
  intro x hx
  by_cases h : firstAccepted A m x = none <;> simp [h]

/-- Exact failure probability after m independent attempts. -/
theorem firstAccepted_failure (A : α → Prop) (m : ℕ) :
    (P.iid m).mass (fun x => firstAccepted A m x = none) = (1-P.mass A)^m := by
  induction m with
  | zero => simp [firstAccepted]
  | succ m ih => rw [firstAccepted_failure_step, ih, pow_succ]; ring

lemma firstAccepted_return_step (A E : α → Prop) (m : ℕ) :
    (P.iid (m+1)).mass (returnsIn A E (m+1)) = P.mass (fun a => A a ∧ E a) +
      (1-P.mass A) * (P.iid m).mass (returnsIn A E m) := by
  rw [iid_mass_step, ← P.mass_compl A]
  unfold mass
  rw [Finset.sum_mul, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro a ha
  by_cases hA : A a
  · by_cases hE : E a <;> simp [returnsIn, firstAccepted, hA, hE, (P.iid m).total]
  · simp [returnsIn, firstAccepted, hA]
    left
    apply Finset.sum_congr rfl
    intro x hx
    by_cases h : ∃ a, firstAccepted A m x = some a ∧ E a <;> simp [h]

/-- Exact subprobability of returning in E before exhausting m attempts. -/
theorem firstAccepted_return (A E : α → Prop) (m : ℕ) :
    (P.iid m).mass (returnsIn A E m) =
      P.mass (fun a => A a ∧ E a) * ∑ k ∈ Finset.range m, (1-P.mass A)^k := by
  induction m with
  | zero => simp [mass, returnsIn, firstAccepted]
  | succ m ih =>
    rw [firstAccepted_return_step, ih, Finset.sum_range_succ']
    simp only [pow_zero, pow_succ]
    rw [← Finset.sum_mul]
    ring

/-- Returning any value is precisely the complement of failure. -/
lemma firstAccepted_success_compl (A : α → Prop) (m : ℕ) :
    (P.iid m).mass (returnsIn A (fun _ => True) m) =
      1 - (P.iid m).mass (fun x => firstAccepted A m x = none) := by
  rw [← (P.iid m).mass_compl]
  congr 1
  funext x
  apply propext
  cases h : firstAccepted A m x <;> simp [returnsIn, h]

/-- Exact probability that the finite attempt budget produces a value. -/
theorem firstAccepted_success (A : α → Prop) (m : ℕ) :
    (P.iid m).mass (returnsIn A (fun _ => True) m) = 1-(1-P.mass A)^m := by
  rw [firstAccepted_success_compl, firstAccepted_failure]

/-- Conditional on success, every return event has exactly the accepted-law mass. -/
theorem firstAccepted_conditional_return (A E : α → Prop) (hr : 0 < P.mass A)
    (m : ℕ) (hm : 0 < m) :
    (P.iid m).mass (returnsIn A E m) /
      (P.iid m).mass (returnsIn A (fun _ => True) m) = (P.accepted A hr).mass E := by
  rw [firstAccepted_return, firstAccepted_return, accepted_mass]
  have htrue : P.mass (fun a => A a ∧ True) = P.mass A := by congr 1; funext a; simp
  rw [htrue]
  have hg : 0 < ∑ k ∈ Finset.range m, (1-P.mass A)^k := by
    apply Finset.sum_pos'
    · intro k hk
      exact pow_nonneg (sub_nonneg.mpr (P.mass_le_one A)) k
    · exact ⟨0, Finset.mem_range.mpr hm, by simp⟩
  field_simp
  ring

end FiniteLaw
end BootstrapCalibration
