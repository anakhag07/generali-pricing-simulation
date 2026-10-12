import BootstrapCalibration.RejectionSampling

/-! Exact occupancy probability of an iid pairs-bootstrap sample. The recursion
below counts newly appearing row indices from explicit finite uniform weights. -/

noncomputable section
open scoped BigOperators Classical
namespace BootstrapCalibration

/-- The set of observed row indices appearing in the resample. -/
def drawSupport {α : Type} [DecidableEq α] : (m : ℕ) → Draws α m → Finset α
  | 0, _ => ∅
  | m+1, x => insert x.1 (drawSupport m x.2)

lemma drawSupport_nonempty {α : Type} [DecidableEq α] (m : ℕ) (x : Draws α (m+1)) :
    (drawSupport (m+1) x).Nonempty := Finset.insert_nonempty _ _

namespace FiniteLaw
variable {α : Type} [Fintype α] (P : FiniteLaw α)

lemma iid_mass_step_tail (m : ℕ) (E : Draws α (m+1) → Prop) :
    (P.iid (m+1)).mass E = ∑ x, (P.iid m).weight x * P.mass (fun a => E (a,x)) := by
  rw [P.iid_mass_step]
  unfold mass
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro x hx
  apply Finset.sum_congr rfl
  intro a ha
  by_cases h : E (a,x) <;> simp [h]
  ring
end FiniteLaw

/-- A deterministic constant sequence, used only to witness a nonempty acceptance event. -/
def constantDraws {α : Type} (a : α) : (m : ℕ) → Draws α m
  | 0 => ()
  | m+1 => (a, constantDraws a m)

namespace RankProbability
variable (n : ℕ) (hn : 0 < n)

lemma uniform_mass_mem (S : Finset (Fin n)) :
    (uniformRows n hn).mass (fun a => a ∈ S) = (S.card : ℝ)/(n : ℝ) := by
  simp only [FiniteLaw.mass, uniformRows, ← Finset.sum_filter]
  simp
  ring

lemma insert_card_one {α : Type} [DecidableEq α] (S : Finset α) (hS : S.Nonempty) (a : α) :
    (insert a S).card = 1 ↔ S.card = 1 ∧ a ∈ S := by
  have hpos := Finset.card_pos.mpr hS
  by_cases ha : a ∈ S
  · simp [Finset.insert_eq_of_mem ha, ha]
  · rw [Finset.card_insert_of_not_mem ha]
    simp only [ha, and_false]
    constructor
    · intro heq
      omega
    · intro hfalse
      exact hfalse.elim

lemma insert_card_two {α : Type} [DecidableEq α] (S : Finset α) (hS : S.Nonempty) (a : α) :
    (insert a S).card ≤ 2 ↔ S.card = 1 ∨ (S.card = 2 ∧ a ∈ S) := by
  have hpos := Finset.card_pos.mpr hS
  by_cases ha : a ∈ S
  · simp only [Finset.insert_eq_of_mem ha, ha, and_true]
    omega
  · rw [Finset.card_insert_of_not_mem ha]
    simp only [ha, and_false, or_false]
    omega

lemma next_one (S : Finset (Fin n)) (hS : S.Nonempty) :
    (uniformRows n hn).mass (fun a => (insert a S).card = 1) =
      if S.card = 1 then 1/(n : ℝ) else 0 := by
  by_cases hs : S.card = 1
  · have he : (fun a => (insert a S).card = 1) = (fun a => a ∈ S) := by
      funext a
      apply propext
      simp [insert_card_one S hS, hs]
    rw [he, uniform_mass_mem]
    simp [hs]
  · have he : (fun a => (insert a S).card = 1) = (fun _ => False) := by
      funext a
      apply propext
      simp [insert_card_one S hS, hs]
    rw [he]
    simp [hs]

lemma next_two (S : Finset (Fin n)) (hS : S.Nonempty) :
    (uniformRows n hn).mass (fun a => (insert a S).card ≤ 2) =
      2/(n : ℝ) * (if S.card ≤ 2 then 1 else 0) +
        (1-2/(n : ℝ)) * (if S.card = 1 then 1 else 0) := by
  by_cases h1 : S.card = 1
  · have he : (fun a => (insert a S).card ≤ 2) = (fun _ => True) := by
      funext a
      apply propext
      simp [insert_card_two S hS, h1]
    rw [he]
    simp [h1]
  · by_cases h2 : S.card = 2
    · have he : (fun a => (insert a S).card ≤ 2) = (fun a => a ∈ S) := by
        funext a
        apply propext
        simp [insert_card_two S hS, h1, h2]
      rw [he, uniform_mass_mem]
      simp [h1, h2]
    · have hnot : ¬ S.card ≤ 2 := by have hp := Finset.card_pos.mpr hS; omega
      have he : (fun a => (insert a S).card ≤ 2) = (fun _ => False) := by
        funext a
        apply propext
        simp [insert_card_two S hS, h1, h2]
      rw [he]
      simp [h1, hnot]

/-- Probability of drawing exactly one distinct row. -/
def oneMass (m : ℕ) : ℝ :=
  ((uniformRows n hn).iid m).mass (fun x => (drawSupport m x).card = 1)

/-- Probability of drawing at most two distinct rows. -/
def twoMass (m : ℕ) : ℝ :=
  ((uniformRows n hn).iid m).mass (fun x => (drawSupport m x).card ≤ 2)

@[simp] lemma oneMass_one : oneMass n hn 1 = 1 := by
  unfold oneMass
  have h : (fun x : Draws (Fin n) 1 => (drawSupport 1 x).card = 1) = (fun _ => True) := by
    funext x
    simp [drawSupport]
  rw [h]
  simp

@[simp] lemma twoMass_one : twoMass n hn 1 = 1 := by
  unfold twoMass
  have h : (fun x : Draws (Fin n) 1 => (drawSupport 1 x).card ≤ 2) = (fun _ => True) := by
    funext x
    simp [drawSupport]
  rw [h]
  simp

lemma oneMass_step (m : ℕ) : oneMass n hn (m+2) = 1/(n : ℝ) * oneMass n hn (m+1) := by
  unfold oneMass
  rw [FiniteLaw.iid_mass_step_tail]
  have h (x : Draws (Fin n) (m+1)) := next_one n hn (drawSupport (m+1) x) (drawSupport_nonempty m x)
  change (∑ x : Draws (Fin n) (m+1), ((uniformRows n hn).iid (m+1)).weight x *
    (uniformRows n hn).mass (fun a => (insert a (drawSupport (m+1) x)).card = 1)) = _
  simp_rw [h]
  unfold FiniteLaw.mass
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro x hx
  by_cases hs : (drawSupport (m+1) x).card = 1 <;> simp [hs]
  ring

lemma twoMass_step (m : ℕ) : twoMass n hn (m+2) =
    2/(n : ℝ) * twoMass n hn (m+1) + (1-2/(n : ℝ)) * oneMass n hn (m+1) := by
  unfold twoMass oneMass
  rw [FiniteLaw.iid_mass_step_tail]
  have h (x : Draws (Fin n) (m+1)) := next_two n hn (drawSupport (m+1) x) (drawSupport_nonempty m x)
  change (∑ x : Draws (Fin n) (m+1), ((uniformRows n hn).iid (m+1)).weight x *
    (uniformRows n hn).mass (fun a => (insert a (drawSupport (m+1) x)).card ≤ 2)) = _
  simp_rw [h]
  unfold FiniteLaw.mass
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro x hx
  by_cases hs : (drawSupport (m+1) x).card ≤ 2 <;>
    by_cases h1 : (drawSupport (m+1) x).card = 1 <;> simp [hs, h1] <;> ring

/-- Exact probability of one occupied category, derived by the occupancy recursion. -/
theorem oneMass_exact (m : ℕ) : oneMass n hn (m+1) = (1/(n : ℝ))^m := by
  induction m with
  | zero => simp
  | succ m ih => rw [oneMass_step, ih, pow_succ]; ring

/-- Exact probability of at most two occupied categories, derived from the actual
uniform iid law rather than an assumed subset count. -/
theorem twoMass_exact (m : ℕ) : twoMass n hn (m+1) =
    ((n : ℝ)-1)*(2/(n : ℝ))^m - ((n : ℝ)-2)*(1/(n : ℝ))^m := by
  induction m with
  | zero => simp; ring
  | succ m ih =>
    rw [twoMass_step, ih, oneMass_exact, pow_succ, pow_succ]
    have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
    field_simp
    ring

/-- The occupancy formula in binomial-coefficient form, for any positive attempt count. -/
theorem twoMass_choose (m : ℕ) : twoMass n hn (m+1) =
    ((n : ℝ) + (n.choose 2 : ℝ) * ((2 : ℝ)^(m+1)-2)) / (n : ℝ)^(m+1) := by
  rw [twoMass_exact, Nat.cast_choose_two]
  simp only [div_pow, one_pow, pow_succ]
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  field_simp
  ring

/-- Exact probability that a size-n pairs resample contains at most two rows.
For distinct quadratic actions, this is the candidate's singular-resample count
probability; the connection from occupancy to matrix rank is a separate lemma. -/
theorem uniform_rows_support_le_two :
    ((uniformRows n hn).iid n).mass (fun x => (drawSupport n x).card ≤ 2) =
      ((n : ℝ) + (n.choose 2 : ℝ) * ((2 : ℝ)^n-2)) / (n : ℝ)^n := by
  have hm : n-1+1 = n := Nat.sub_add_cancel hn
  have h := twoMass_choose n hn (n-1)
  rw [hm] at h
  exact h

/-- Exact complementary probability that a size-n resample uses at least three rows. -/
theorem uniform_rows_support_ge_three :
    ((uniformRows n hn).iid n).mass (fun x => 3 ≤ (drawSupport n x).card) =
      1 - ((n : ℝ) + (n.choose 2 : ℝ) * ((2 : ℝ)^n-2)) / (n : ℝ)^n := by
  have h := ((uniformRows n hn).iid n).mass_compl (fun x => (drawSupport n x).card ≤ 2)
  have he : (fun x : Draws (Fin n) n => ¬ (drawSupport n x).card ≤ 2) =
      (fun x => 3 ≤ (drawSupport n x).card) := by
    funext x
    apply propext
    omega
  rw [he, uniform_rows_support_le_two] at h
  exact h

lemma uniform_iid_weight (m : ℕ) (x : Draws (Fin n) m) :
    ((uniformRows n hn).iid m).weight x = (1/(n : ℝ))^m := by
  induction m with
  | zero => simp [FiniteLaw.iid]
  | succ m ih =>
    change (1/(n : ℝ)) * ((uniformRows n hn).iid m).weight x.2 = _
    rw [ih, pow_succ]
    ring

/-- At least three-row support has positive probability for n≥3, proved by an
explicit sample containing rows 0,1,2 rather than an assumed acceptance rate. -/
theorem support_ge_three_positive (hn3 : 3 ≤ n) :
    0 < ((uniformRows n hn).iid n).mass (fun x => 3 ≤ (drawSupport n x).card) := by
  obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_le hn3
  have hk' : n = k+3 := by omega
  clear hk
  subst n
  let i0 : Fin (k+3) := ⟨0, by omega⟩
  let i1 : Fin (k+3) := ⟨1, by omega⟩
  let i2 : Fin (k+3) := ⟨2, by omega⟩
  let x : Draws (Fin (k+3)) (k+3) := (i0, (i1, (i2, constantDraws i2 k)))
  have hsub : ({i0,i1,i2} : Finset (Fin (k+3))) ⊆ drawSupport (k+3) x := by
    intro a ha
    simp only [Finset.mem_insert, Finset.mem_singleton] at ha
    simp only [x, drawSupport, Finset.mem_insert]
    rcases ha with h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (Or.inl h))
  have hcard : ({i0,i1,i2} : Finset (Fin (k+3))).card = 3 := by
    simp [i0, i1, i2, Fin.ext_iff]
  have hx : 3 ≤ (drawSupport (k+3) x).card := by
    simpa only [hcard] using Finset.card_le_card hsub
  have hw : 0 < ((uniformRows (k+3) hn).iid (k+3)).weight x := by
    rw [uniform_iid_weight]
    positivity
  unfold FiniteLaw.mass
  apply lt_of_lt_of_le hw
  have hs := Finset.single_le_sum
    (f := fun a : Draws (Fin (k+3)) (k+3) =>
      if (fun x => 3 ≤ (drawSupport (k+3) x).card) a then
        ((uniformRows (k+3) hn).iid (k+3)).weight a else 0)
    (s := Finset.univ)
    (by
      intro y hy
      dsimp only
      split_ifs
      · exact ((uniformRows (k+3) hn).iid (k+3)).nonneg y
      · exact le_rfl)
    (Finset.mem_univ x)
  convert hs using 1
  · simp [hx]
  · apply Finset.sum_congr rfl
    intro y hy
    by_cases h : 3 ≤ (drawSupport (k+3) y).card <;> simp [h]


end RankProbability
end BootstrapCalibration
