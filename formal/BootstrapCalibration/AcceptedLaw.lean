import BootstrapCalibration.MultinomialMoments
import BootstrapCalibration.Conditioning

/-!
Conditioning a finite resampling law on acceptance, including two independent
accepted draws. These bounds apply to every test taking values in [0,1].
-/

noncomputable section
open scoped BigOperators Classical
namespace BootstrapCalibration.FiniteLaw
variable {α β : Type} [Fintype α] [Fintype β]

@[ext] theorem ext_weights {P Q : FiniteLaw α} (h : ∀ a, P.weight a = Q.weight a) : P = Q := by
  cases P
  cases Q
  congr
  exact funext h

variable (P : FiniteLaw α)

lemma expectation_nonneg {f : α → ℝ} (hf : ∀ a, 0 ≤ f a) : 0 ≤ P.expectation f := by
  exact Finset.sum_nonneg (fun a _ => mul_nonneg (P.nonneg a) (hf a))

lemma expectation_mono {f g : α → ℝ} (h : ∀ a, f a ≤ g a) :
    P.expectation f ≤ P.expectation g := by
  exact Finset.sum_le_sum (fun a _ => mul_le_mul_of_nonneg_left (h a) (P.nonneg a))

lemma expectation_indicator (A : α → Prop) :
    P.expectation (fun a => if A a then 1 else 0) = P.mass A := by
  unfold expectation mass
  apply Finset.sum_congr rfl
  intro a ha
  by_cases h : A a <;> simp [h]

/-- The exact law of one resample conditioned on satisfying acceptance A. -/
def accepted (A : α → Prop) (hr : 0 < P.mass A) : FiniteLaw α where
  weight := fun a => if A a then P.weight a / P.mass A else 0
  nonneg := fun a => by split_ifs; exact div_nonneg (P.nonneg a) hr.le; exact le_rfl
  total := by
    have heq : (∑ a, if A a then P.weight a / P.mass A else 0) = P.mass A / P.mass A := by
      unfold mass
      rw [Finset.sum_div]
      apply Finset.sum_congr rfl
      intro a ha
      by_cases h : A a <;> simp [h]
    rw [heq, div_self (ne_of_gt hr)]

@[simp] lemma accepted_weight (A : α → Prop) (hr : 0 < P.mass A) (a : α) :
    (P.accepted A hr).weight a = if A a then P.weight a / P.mass A else 0 := rfl

lemma accepted_mass (A E : α → Prop) (hr : 0 < P.mass A) :
    (P.accepted A hr).mass E = P.mass (fun a => A a ∧ E a) / P.mass A := by
  unfold mass
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro a ha
  by_cases hA : A a <;> by_cases hE : E a <;> simp [accepted, hA, hE, mass]

lemma accepted_expectation (A : α → Prop) (hr : 0 < P.mass A) (f : α → ℝ) :
    (P.accepted A hr).expectation f =
      P.expectation (fun a => if A a then f a else 0) / P.mass A := by
  unfold expectation
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro a ha
  by_cases hA : A a <;> simp [accepted, hA, div_mul_eq_mul_div]

lemma restricted_expectation_bounds (A : α → Prop) (f : α → ℝ)
    (h0 : ∀ a, 0 ≤ f a) (h1 : ∀ a, f a ≤ 1) :
    0 ≤ P.expectation (fun a => if A a then f a else 0) ∧
    P.expectation (fun a => if A a then f a else 0) ≤ P.mass A := by
  constructor
  · apply expectation_nonneg
    intro a
    split_ifs
    · exact h0 a
    · exact le_rfl
  · rw [← P.expectation_indicator A]
    apply expectation_mono
    intro a
    split_ifs
    · exact h1 a
    · exact le_rfl

lemma expectation_acceptance_partition (A : α → Prop) (f : α → ℝ) :
    P.expectation f = P.expectation (fun a => if A a then f a else 0) +
      P.expectation (fun a => if ¬ A a then f a else 0) := by
  rw [← expectation_add]
  apply expectation_congr
  intro a
  by_cases h : A a <;> simp [h]

/-- Conditioning changes the expectation of any [0,1] test by at most rejection mass. -/
theorem accepted_expectation_error (A : α → Prop) (hr : 0 < P.mass A) (f : α → ℝ)
    (h0 : ∀ a, 0 ≤ f a) (h1 : ∀ a, f a ≤ 1) :
    |(P.accepted A hr).expectation f - P.expectation f| ≤ 1 - P.mass A := by
  have hx := P.restricted_expectation_bounds A f h0 h1
  have hy0 : 0 ≤ P.expectation (fun a => if ¬ A a then f a else 0) := by
    apply expectation_nonneg
    intro a
    by_cases ha : A a <;> simp [ha, h0 a]
  have hy1 : P.expectation (fun a => if ¬ A a then f a else 0) ≤ 1 - P.mass A := by
    rw [← P.mass_compl A]
    unfold expectation mass
    apply Finset.sum_le_sum
    intro a ha
    by_cases hA : A a
    · simp [hA]
    · simp only [hA, not_false_eq_true, ↓reduceIte]
      simpa using mul_le_mul_of_nonneg_left (h1 a) (P.nonneg a)
  rw [accepted_expectation, P.expectation_acceptance_partition A f]
  exact BootstrapCalibration.conditioning_error (P.mass A) _ _ hr (P.mass_le_one A)
    hx.1 hx.2 hy0 hy1

/-- In particular, the bound applies uniformly to all events. -/
theorem accepted_mass_error (A E : α → Prop) (hr : 0 < P.mass A) :
    |(P.accepted A hr).mass E - P.mass E| ≤ 1 - P.mass A := by
  have h := P.accepted_expectation_error A hr (fun a => if E a then 1 else 0)
    (by intro a; dsimp only; split_ifs <;> norm_num) (by intro a; dsimp only; split_ifs <;> norm_num)
  simpa only [expectation_indicator] using h

/-- Independent product of two arbitrary finite probability laws. -/
def product (Q : FiniteLaw β) : FiniteLaw (α × β) where
  weight := fun x => P.weight x.1 * Q.weight x.2
  nonneg := fun x => mul_nonneg (P.nonneg x.1) (Q.nonneg x.2)
  total := by
    rw [Fintype.sum_prod_type]
    simp only [← Finset.mul_sum, Q.total, mul_one, P.total]

@[simp] lemma product_weight (Q : FiniteLaw β) (x : α × β) :
    (P.product Q).weight x = P.weight x.1 * Q.weight x.2 := rfl

lemma product_mass (Q : FiniteLaw β) (A : α → Prop) (B : β → Prop) :
    (P.product Q).mass (fun x => A x.1 ∧ B x.2) = P.mass A * Q.mass B := by
  unfold mass
  rw [Fintype.sum_prod_type, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro a ha
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro b hb
  by_cases hA : A a <;> by_cases hB : B b <;> simp [product, hA, hB]

lemma product_expectation (Q : FiniteLaw β) (f : α × β → ℝ) :
    (P.product Q).expectation f = ∑ a, ∑ b, P.weight a * Q.weight b * f (a,b) := by
  exact Fintype.sum_prod_type _

lemma product_accepted (Q : FiniteLaw β) (A : α → Prop) (B : β → Prop)
    (hA : 0 < P.mass A) (hB : 0 < Q.mass B) :
    (P.accepted A hA).product (Q.accepted B hB) =
      (P.product Q).accepted (fun x => A x.1 ∧ B x.2)
        (by rw [product_mass]; exact mul_pos hA hB) := by
  apply ext_weights
  intro x
  simp only [product_weight, accepted_weight, product_mass]
  by_cases ha : A x.1 <;> by_cases hb : B x.2 <;>
    simp [ha, hb, div_mul_div_comm]

/-- Two independently accepted draws change a [0,1] expectation by at most 1-r². -/
theorem two_accepted_expectation_error (A : α → Prop) (hr : 0 < P.mass A)
    (f : α × α → ℝ) (h0 : ∀ x, 0 ≤ f x) (h1 : ∀ x, f x ≤ 1) :
    |((P.accepted A hr).product (P.accepted A hr)).expectation f -
      (P.product P).expectation f| ≤ 1 - (P.mass A)^2 := by
  rw [product_accepted]
  have h := (P.product P).accepted_expectation_error (fun x => A x.1 ∧ A x.2)
    (by rw [product_mass]; exact mul_pos hr hr) f h0 h1
  simpa only [product_mass, ← sq] using h

/-- The corresponding two-draw bound for any event. -/
theorem two_accepted_mass_error (A : α → Prop) (hr : 0 < P.mass A)
    (E : α × α → Prop) :
    |((P.accepted A hr).product (P.accepted A hr)).mass E -
      (P.product P).mass E| ≤ 1 - (P.mass A)^2 := by
  have h := P.two_accepted_expectation_error A hr (fun x => if E x then 1 else 0)
    (by intro x; dsimp only; split_ifs <;> norm_num) (by intro x; dsimp only; split_ifs <;> norm_num)
  simpa only [expectation_indicator] using h

end BootstrapCalibration.FiniteLaw
