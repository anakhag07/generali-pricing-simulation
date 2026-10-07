import BootstrapCalibration.FiniteBootstrap

/-!
Exact multinomial moments from the finite iid row-sampling law.
No multinomial moment identity is taken as a hypothesis.
-/

noncomputable section
open scoped BigOperators Classical
namespace BootstrapCalibration
namespace FiniteLaw
variable {α : Type} [Fintype α] (P : FiniteLaw α)

/-- Expectation under the explicit normalized finite law. -/
def expectation (f : α → ℝ) : ℝ := ∑ a, P.weight a * f a

@[simp] lemma expectation_const (c : ℝ) : P.expectation (fun _ => c) = c := by
  simp only [expectation, ← Finset.sum_mul, P.total, one_mul]

lemma expectation_congr {f g : α → ℝ} (h : ∀ a, f a = g a) :
    P.expectation f = P.expectation g := by
  exact congrArg P.expectation (funext h)

@[simp] lemma expectation_add (f g : α → ℝ) :
    P.expectation (fun a => f a + g a) = P.expectation f + P.expectation g := by
  simp only [expectation, mul_add, Finset.sum_add_distrib]

@[simp] lemma expectation_sub (f g : α → ℝ) :
    P.expectation (fun a => f a - g a) = P.expectation f - P.expectation g := by
  simp only [expectation, mul_sub, Finset.sum_sub_distrib]

@[simp] lemma expectation_mul_const (f : α → ℝ) (c : ℝ) :
    P.expectation (fun a => f a * c) = P.expectation f * c := by
  simp only [expectation, ← mul_assoc, Finset.sum_mul]

@[simp] lemma expectation_const_mul (c : ℝ) (f : α → ℝ) :
    P.expectation (fun a => c * f a) = c * P.expectation f := by
  simpa only [mul_comm] using P.expectation_mul_const f c

lemma expectation_iid_step (n : ℕ) (f : Draws α (n+1) → ℝ) :
    (P.iid (n+1)).expectation f =
      P.expectation (fun a => (P.iid n).expectation (fun x => f (a, x))) := by
  change (∑ x : α × Draws α n, P.weight x.1 * (P.iid n).weight x.2 * f x) = _
  rw [Fintype.sum_prod_type]
  unfold expectation
  simp only [Finset.mul_sum, mul_assoc]

/-- Independence is proved by factorization of the explicit iid atom weights. -/
lemma expectation_iid_prod (n : ℕ) (h : α → ℝ) (f : Draws α n → ℝ) :
    (P.iid (n+1)).expectation (fun x => h x.1 * f x.2) =
      P.expectation h * (P.iid n).expectation f := by
  change (∑ x : α × Draws α n, (P.weight x.1 * (P.iid n).weight x.2) *
      (h x.1 * f x.2)) = _
  rw [Fintype.sum_prod_type]
  calc
    (∑ a, ∑ x, P.weight a * (P.iid n).weight x * (h a * f x)) =
      ∑ a, ∑ x, (P.weight a * h a) * ((P.iid n).weight x * f x) := by
        apply Finset.sum_congr rfl
        intro a ha
        apply Finset.sum_congr rfl
        intro x hx
        ring
    _ = P.expectation h * (P.iid n).expectation f := by
      simp only [← Finset.mul_sum, ← Finset.sum_mul, expectation]

@[simp] lemma expectation_iid_head (n : ℕ) (h : α → ℝ) :
    (P.iid (n+1)).expectation (fun x => h x.1) = P.expectation h := by
  simpa using P.expectation_iid_prod n h (fun _ => 1)

@[simp] lemma expectation_iid_tail (n : ℕ) (f : Draws α n → ℝ) :
    (P.iid (n+1)).expectation (fun x => f x.2) = (P.iid n).expectation f := by
  simpa using P.expectation_iid_prod n (fun _ => 1) f
end FiniteLaw

/-- Sum a real-valued row statistic over an explicit iid sample. -/
def sumDraws {α : Type} (h : α → ℝ) : (n : ℕ) → Draws α n → ℝ
  | 0, _ => 0
  | n+1, x => h x.1 + sumDraws h n x.2

namespace FiniteLaw
variable {α : Type} [Fintype α] (P : FiniteLaw α)

lemma expectation_sumDraws (h : α → ℝ) (n : ℕ) :
    (P.iid n).expectation (sumDraws h n) = n * P.expectation h := by
  induction n with
  | zero => simp [sumDraws, expectation, iid, Draws]
  | succ n ih =>
    change (P.iid (n+1)).expectation (fun x => h x.1 + sumDraws h n x.2) = _
    rw [expectation_add, expectation_iid_head, expectation_iid_tail, ih]
    push_cast
    ring

lemma expectation_sumDraws_sq (h : α → ℝ) (hmean : P.expectation h = 0) (n : ℕ) :
    (P.iid n).expectation (fun x => (sumDraws h n x)^2) =
      n * P.expectation (fun a => (h a)^2) := by
  induction n with
  | zero => simp [sumDraws, expectation, iid, Draws]
  | succ n ih =>
    have hex : (fun x : Draws α (n+1) => (sumDraws h (n+1) x)^2) =
        (fun x => (h x.1)^2 + 2 * (h x.1 * sumDraws h n x.2) + (sumDraws h n x.2)^2) := by
      funext x
      simp only [sumDraws]
      ring
    rw [hex, expectation_iid_step]
    dsimp only
    simp only [expectation_add, expectation_const, expectation_mul_const, expectation_const_mul, expectation_iid_head,
      expectation_iid_tail, expectation_iid_prod, expectation_sumDraws, hmean, zero_mul, add_zero, ih]
    push_cast
    ring

lemma expectation_sumDraws_mul (h g : α → ℝ)
    (hh : P.expectation h = 0) (hg : P.expectation g = 0) (n : ℕ) :
    (P.iid n).expectation (fun x => sumDraws h n x * sumDraws g n x) =
      n * P.expectation (fun a => h a * g a) := by
  induction n with
  | zero => simp [sumDraws, expectation, iid, Draws]
  | succ n ih =>
    have hex : (fun x : Draws α (n+1) => sumDraws h (n+1) x * sumDraws g (n+1) x) =
        (fun x => h x.1 * g x.1 + h x.1 * sumDraws g n x.2 +
          g x.1 * sumDraws h n x.2 + sumDraws h n x.2 * sumDraws g n x.2) := by
      funext x
      simp only [sumDraws]
      ring
    rw [hex, expectation_iid_step]
    dsimp only
    simp only [expectation_add, expectation_const, expectation_mul_const, expectation_const_mul, expectation_iid_head, expectation_iid_tail,
      expectation_iid_prod, expectation_sumDraws, hh, hg, zero_mul, add_zero, ih]
    push_cast
    ring

lemma expectation_sumDraws_fourth (h : α → ℝ) (hmean : P.expectation h = 0) (n : ℕ) :
    (P.iid n).expectation (fun x => (sumDraws h n x)^4) =
      n * P.expectation (fun a => (h a)^4) +
      3 * n * (n-1 : ℝ) * (P.expectation (fun a => (h a)^2))^2 := by
  induction n with
  | zero => simp [sumDraws, expectation, iid, Draws]
  | succ n ih =>
    have hex : (fun x : Draws α (n+1) => (sumDraws h (n+1) x)^4) =
        (fun x => (h x.1)^4 + 4 * ((h x.1)^3 * sumDraws h n x.2) +
          6 * ((h x.1)^2 * (sumDraws h n x.2)^2) +
          4 * (h x.1 * (sumDraws h n x.2)^3) + (sumDraws h n x.2)^4) := by
      funext x
      simp only [sumDraws]
      ring
    rw [hex, expectation_iid_step]
    dsimp only
    simp only [expectation_add, expectation_const, expectation_mul_const, expectation_const_mul, expectation_iid_head,
      expectation_iid_tail, expectation_iid_prod, expectation_sumDraws,
      expectation_sumDraws_sq P h hmean, hmean, zero_mul, mul_zero, add_zero, ih]
    push_cast
    ring

lemma expectation_sumDraws_sq_mul_sq (h g : α → ℝ)
    (hh : P.expectation h = 0) (hg : P.expectation g = 0) (n : ℕ) :
    (P.iid n).expectation (fun x => (sumDraws h n x)^2 * (sumDraws g n x)^2) =
      n * P.expectation (fun a => (h a)^2 * (g a)^2) + n * (n-1 : ℝ) *
        (P.expectation (fun a => (h a)^2) * P.expectation (fun a => (g a)^2) +
          2 * (P.expectation (fun a => h a * g a))^2) := by
  induction n with
  | zero => simp [sumDraws, expectation, iid, Draws]
  | succ n ih =>
    have hex : (fun x : Draws α (n+1) => (sumDraws h (n+1) x)^2 * (sumDraws g (n+1) x)^2) =
        (fun x => (h x.1)^2 * (g x.1)^2 +
          2 * (((h x.1)^2 * g x.1) * sumDraws g n x.2) +
          (h x.1)^2 * (sumDraws g n x.2)^2 +
          2 * ((h x.1 * (g x.1)^2) * sumDraws h n x.2) +
          4 * ((h x.1 * g x.1) * (sumDraws h n x.2 * sumDraws g n x.2)) +
          2 * (h x.1 * (sumDraws h n x.2 * (sumDraws g n x.2)^2)) +
          (g x.1)^2 * (sumDraws h n x.2)^2 +
          2 * (g x.1 * ((sumDraws h n x.2)^2 * sumDraws g n x.2)) +
          (sumDraws h n x.2)^2 * (sumDraws g n x.2)^2) := by
      funext x
      simp only [sumDraws]
      ring
    rw [hex, expectation_iid_step]
    dsimp only
    simp only [expectation_add, expectation_const, expectation_mul_const, expectation_const_mul, expectation_iid_head,
      expectation_iid_tail, expectation_iid_prod, expectation_sumDraws,
      expectation_sumDraws_sq P h hh, expectation_sumDraws_sq P g hg,
      expectation_sumDraws_mul P h g hh hg, hh, hg, zero_mul, mul_zero, add_zero, ih]
    push_cast
    ring

/-- Centered indicator for one category in a finite row-sampling law. -/
def centeredAtom (i : α) (a : α) : ℝ := (if a = i then 1 else 0) - P.weight i

lemma expectation_atom (i : α) : P.expectation (fun a => if a = i then 1 else 0) = P.weight i := by
  simp [expectation, mul_ite]

@[simp] lemma expectation_centeredAtom (i : α) : P.expectation (P.centeredAtom i) = 0 := by
  change P.expectation (fun a => (if a = i then 1 else 0) - P.weight i) = 0
  rw [expectation_sub, expectation_atom, expectation_const]
  ring

lemma expectation_centeredAtom_sq (i : α) :
    P.expectation (fun a => (P.centeredAtom i a)^2) = P.weight i - (P.weight i)^2 := by
  have hex : (fun a => (P.centeredAtom i a)^2) =
      (fun a => (if a = i then 1 else 0) * (1 - 2 * P.weight i) + (P.weight i)^2) := by
    funext a
    by_cases h : a = i <;> simp [centeredAtom, h]
    ring
  rw [hex]
  simp only [expectation_add, expectation_mul_const, expectation_atom, expectation_const]
  ring

lemma expectation_centeredAtom_fourth (i : α) :
    P.expectation (fun a => (P.centeredAtom i a)^4) =
      P.weight i - 4*(P.weight i)^2 + 6*(P.weight i)^3 - 3*(P.weight i)^4 := by
  have hex : (fun a => (P.centeredAtom i a)^4) =
      (fun a => (if a = i then 1 else 0) *
        (1 - 4*P.weight i + 6*(P.weight i)^2 - 4*(P.weight i)^3) + (P.weight i)^4) := by
    funext a
    by_cases h : a = i <;> simp [centeredAtom, h] <;> ring
  rw [hex]
  simp only [expectation_add, expectation_mul_const, expectation_atom, expectation_const]
  ring

lemma expectation_centeredAtom_mul (i j : α) (hij : i ≠ j) :
    P.expectation (fun a => P.centeredAtom i a * P.centeredAtom j a) =
      -(P.weight i * P.weight j) := by
  have hex : (fun a => P.centeredAtom i a * P.centeredAtom j a) =
      (fun a => (if a = i then 1 else 0) * (-P.weight j) +
        (if a = j then 1 else 0) * (-P.weight i) + P.weight i * P.weight j) := by
    funext a
    by_cases hi : a = i
    · subst a
      simp [centeredAtom, hij]
      ring
    · by_cases hj : a = j <;> simp [centeredAtom, hi, hj, hij, Ne.symm hij]
      ring
  rw [hex]
  simp only [expectation_add, expectation_mul_const, expectation_atom, expectation_const]
  ring

lemma expectation_centeredAtom_sq_mul_sq (i j : α) (hij : i ≠ j) :
    P.expectation (fun a => (P.centeredAtom i a)^2 * (P.centeredAtom j a)^2) =
      P.weight i * (P.weight j)^2 + (P.weight i)^2 * P.weight j -
        3*(P.weight i)^2*(P.weight j)^2 := by
  have hex : (fun a => (P.centeredAtom i a)^2 * (P.centeredAtom j a)^2) =
      (fun a => (if a = i then 1 else 0) * ((P.weight j)^2 * (1-2*P.weight i)) +
        (if a = j then 1 else 0) * ((P.weight i)^2 * (1-2*P.weight j)) +
        (P.weight i)^2 * (P.weight j)^2) := by
    funext a
    by_cases hi : a = i
    · subst a
      simp [centeredAtom, hij]
      ring
    · by_cases hj : a = j <;> simp [centeredAtom, hi, hj, hij, Ne.symm hij]
      ring
  rw [hex]
  simp only [expectation_add, expectation_mul_const, expectation_atom, expectation_const]
  ring

lemma sumDraws_centeredAtom (i : α) (n : ℕ) (x : Draws α n) :
    sumDraws (P.centeredAtom i) n x = (countHits (fun a => a = i) n x : ℝ) - n * P.weight i := by
  induction n with
  | zero => simp [sumDraws, countHits]
  | succ n ih =>
    simp only [sumDraws, countHits, centeredAtom, ih]
    push_cast
    by_cases h : x.1 = i <;> simp [h] <;> ring

end FiniteLaw

/-- Uniform sampling of one of n observed rows, with replacement across draws. -/
def uniformRows (n : ℕ) (hn : 0 < n) : FiniteLaw (Fin n) where
  weight := fun _ => 1/(n : ℝ)
  nonneg := fun _ => by positivity
  total := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
    field_simp

/-- Multinomial row count minus one, for a size-n pairs-bootstrap resample. -/
def centeredRowCount {n : ℕ} (i : Fin n) (x : Draws (Fin n) n) : ℝ :=
  (countHits (fun a => a = i) n x : ℝ) - 1

lemma uniformRows_sumDraws_centeredAtom (n : ℕ) (hn : 0 < n) (i : Fin n)
    (x : Draws (Fin n) n) :
    sumDraws ((uniformRows n hn).centeredAtom i) n x = centeredRowCount i x := by
  rw [FiniteLaw.sumDraws_centeredAtom]
  simp only [uniformRows, centeredRowCount]
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  field_simp

/-- The diagonal covariance of unconditioned size-n multinomial row counts. -/
theorem multinomial_second (n : ℕ) (hn : 0 < n) (i : Fin n) :
    ((uniformRows n hn).iid n).expectation (fun x => (centeredRowCount i x)^2) =
      1 - 1/(n : ℝ) := by
  have h := (uniformRows n hn).expectation_sumDraws_sq
    ((uniformRows n hn).centeredAtom i)
    ((uniformRows n hn).expectation_centeredAtom i) n
  simp_rw [uniformRows_sumDraws_centeredAtom] at h
  simp only [
    FiniteLaw.expectation_centeredAtom_sq, uniformRows] at h
  refine h.trans ?_
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  field_simp
  ring

/-- The off-diagonal covariance of unconditioned size-n multinomial row counts. -/
theorem multinomial_cross (n : ℕ) (hn : 0 < n) (i j : Fin n) (hij : i ≠ j) :
    ((uniformRows n hn).iid n).expectation (fun x => centeredRowCount i x * centeredRowCount j x) =
      -1/(n : ℝ) := by
  have h := (uniformRows n hn).expectation_sumDraws_mul
    ((uniformRows n hn).centeredAtom i) ((uniformRows n hn).centeredAtom j)
    ((uniformRows n hn).expectation_centeredAtom i)
    ((uniformRows n hn).expectation_centeredAtom j) n
  simp_rw [uniformRows_sumDraws_centeredAtom] at h
  simp only [
    FiniteLaw.expectation_centeredAtom_mul _ i j hij, uniformRows] at h
  refine h.trans ?_
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  field_simp

/-- The exact fourth moment; n is arbitrary, not a finite enumeration range. -/
theorem multinomial_fourth (n : ℕ) (hn : 0 < n) (i : Fin n) :
    ((uniformRows n hn).iid n).expectation (fun x => (centeredRowCount i x)^4) =
      4 - 13/(n : ℝ) + 15/(n : ℝ)^2 - 6/(n : ℝ)^3 := by
  have h := (uniformRows n hn).expectation_sumDraws_fourth
    ((uniformRows n hn).centeredAtom i)
    ((uniformRows n hn).expectation_centeredAtom i) n
  simp_rw [uniformRows_sumDraws_centeredAtom] at h
  simp only [
    FiniteLaw.expectation_centeredAtom_fourth, FiniteLaw.expectation_centeredAtom_sq,
    uniformRows] at h
  refine h.trans ?_
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  field_simp
  ring

/-- The exact cross moment of squared distinct multinomial count deviations. -/
theorem multinomial_square_cross (n : ℕ) (hn : 0 < n) (i j : Fin n) (hij : i ≠ j) :
    ((uniformRows n hn).iid n).expectation
      (fun x => (centeredRowCount i x)^2 * (centeredRowCount j x)^2) =
      1 - 3/(n : ℝ) + 7/(n : ℝ)^2 - 6/(n : ℝ)^3 := by
  have h := (uniformRows n hn).expectation_sumDraws_sq_mul_sq
    ((uniformRows n hn).centeredAtom i) ((uniformRows n hn).centeredAtom j)
    ((uniformRows n hn).expectation_centeredAtom i)
    ((uniformRows n hn).expectation_centeredAtom j) n
  simp_rw [uniformRows_sumDraws_centeredAtom] at h
  simp only [
    FiniteLaw.expectation_centeredAtom_sq_mul_sq _ i j hij,
    FiniteLaw.expectation_centeredAtom_mul _ i j hij,
    FiniteLaw.expectation_centeredAtom_sq, uniformRows] at h
  refine h.trans ?_
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  field_simp
  ring

/-- Covariance in a form that includes both diagonal and off-diagonal entries. -/
theorem multinomial_covariance (n : ℕ) (hn : 0 < n) (i j : Fin n) :
    ((uniformRows n hn).iid n).expectation (fun x => centeredRowCount i x * centeredRowCount j x) =
      (if i = j then 1 else 0) - 1/(n : ℝ) := by
  by_cases hij : i = j
  · subst j
    simpa only [↓reduceIte, ← sq] using multinomial_second n hn i
  · simpa only [hij, ↓reduceIte, zero_sub, neg_div] using multinomial_cross n hn i j hij

/-- The variance of the squared multinomial count deviation. -/
theorem multinomial_square_variance (n : ℕ) (hn : 0 < n) (i : Fin n) :
    ((uniformRows n hn).iid n).expectation
      (fun x => ((centeredRowCount i x)^2 - (1-1/(n : ℝ)))^2) =
      3 - 11/(n : ℝ) + 14/(n : ℝ)^2 - 6/(n : ℝ)^3 := by
  have hex : (fun x : Draws (Fin n) n => ((centeredRowCount i x)^2 - (1-1/(n : ℝ)))^2) =
      (fun x => (centeredRowCount i x)^4 -
        2 * ((centeredRowCount i x)^2 * (1-1/(n : ℝ))) + (1-1/(n : ℝ))^2) := by
    funext x
    ring
  rw [hex]
  simp only [FiniteLaw.expectation_add, FiniteLaw.expectation_sub,
    FiniteLaw.expectation_const_mul, FiniteLaw.expectation_mul_const,
    FiniteLaw.expectation_const, multinomial_fourth, multinomial_second]
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  field_simp
  ring

/-- Covariance of squared deviations for distinct categories. -/
theorem multinomial_square_covariance (n : ℕ) (hn : 0 < n) (i j : Fin n) (hij : i ≠ j) :
    ((uniformRows n hn).iid n).expectation
      (fun x => ((centeredRowCount i x)^2 - (1-1/(n : ℝ))) *
        ((centeredRowCount j x)^2 - (1-1/(n : ℝ)))) =
      -1/(n : ℝ) + 6/(n : ℝ)^2 - 6/(n : ℝ)^3 := by
  have hex : (fun x : Draws (Fin n) n =>
        ((centeredRowCount i x)^2 - (1-1/(n : ℝ))) *
        ((centeredRowCount j x)^2 - (1-1/(n : ℝ)))) =
      (fun x => (centeredRowCount i x)^2 * (centeredRowCount j x)^2 -
        (centeredRowCount i x)^2 * (1-1/(n : ℝ)) -
        (centeredRowCount j x)^2 * (1-1/(n : ℝ)) + (1-1/(n : ℝ))^2) := by
    funext x
    ring
  rw [hex]
  simp only [FiniteLaw.expectation_add, FiniteLaw.expectation_sub,
    FiniteLaw.expectation_mul_const, FiniteLaw.expectation_const,
    multinomial_square_cross n hn i j hij, multinomial_second]
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  field_simp
  ring

end BootstrapCalibration
