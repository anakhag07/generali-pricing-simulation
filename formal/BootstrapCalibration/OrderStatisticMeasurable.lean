import BootstrapCalibration.OrderStatistic
import BootstrapCalibration.IntegratedCalibration

/-!
# Measurability of the actual sorted bootstrap critical value

Sublevel sets of the merge-sorted sample order statistic are exactly finite
threshold-count events. This proves measurability from the input scores without
assuming measurability or validity of a sorting black box.
-/

noncomputable section
open MeasureTheory
open scoped Classical

namespace BootstrapCalibration

theorem sorted_list_get_le_iff (l : List ℝ) (hl : l.Sorted (· ≤ ·))
    (j : Fin l.length) (t : ℝ) :
    l.get j ≤ t ↔ j.val + 1 ≤ l.countP (fun x => decide (x ≤ t)) := by
  induction l with
  | nil => exact j.elim0
  | cons a l ih =>
    have hp := List.pairwise_cons.mp hl
    have hnone (ha : ¬ a ≤ t) : l.countP (fun x => decide (x ≤ t)) = 0 := by
      apply List.countP_eq_zero.mpr
      intro b hb
      simp only [decide_eq_true_eq]
      exact fun hbt => ha ((hp.1 b hb).trans hbt)
    cases j using Fin.cases with
    | zero =>
      by_cases ha : a ≤ t
      · simp [List.countP_cons, ha]
      · simp [List.countP_cons, ha, hnone ha]
    | succ j =>
      by_cases ha : a ≤ t
      · have hiff := ih hp.2 j
        simp [List.countP_cons, ha] at hiff ⊢
        exact hiff
      · have hnot : ¬ l.get j ≤ t := by
          intro hj
          exact ha ((hp.1 _ (List.get_mem l j)).trans hj)
        simpa [List.countP_cons, ha, hnone ha] using hnot

/-- Exact threshold-count characterization for the implemented mergeSort nth. -/
theorem sortedCritical_le_iff {α : Type} (T : α → ℝ) (B : ℕ)
    (x : Draws α B) (j : Fin B) (t : ℝ) :
    sortedCritical T B x j ≤ t ↔
      j.val + 1 ≤ countHits (fun a => T a ≤ t) B x := by
  let l := (drawScores T B x).mergeSort (· ≤ ·)
  have hsort : l.Sorted (· ≤ ·) := List.sorted_mergeSort' (· ≤ ·) _
  have hj : j.val < l.length := by
    simpa only [l, List.length_mergeSort, length_drawScores] using j.isLt
  change l.get ⟨j.val, hj⟩ ≤ t ↔ _
  rw [sorted_list_get_le_iff l hsort ⟨j.val, hj⟩ t]
  have hperm := List.mergeSort_perm (drawScores T B x) (· ≤ ·)
  rw [hperm.countP_eq, drawScores_count]

theorem measurable_countHits_threshold {Ω : Type*} [MeasurableSpace Ω] {α : Type}
    (T : Ω → α → ℝ) (hT : ∀ a, Measurable (fun ω => T ω a))
    (B : ℕ) (x : Draws α B) (t : ℝ) :
    Measurable (fun ω => countHits (fun a => T ω a ≤ t) B x) := by
  induction B with
  | zero => exact measurable_const
  | succ B ih =>
    change Measurable (fun ω => (if T ω x.1 ≤ t then 1 else 0) +
      countHits (fun a => T ω a ≤ t) B x.2)
    exact (Measurable.ite (measurableSet_le (hT x.1) measurable_const)
      measurable_const measurable_const).add (ih x.2)

/-- Measurability in the observations for every fixed bootstrap draw sequence. -/
theorem measurable_sortedCritical {Ω : Type*} [MeasurableSpace Ω] {α : Type}
    (T : Ω → α → ℝ) (hT : ∀ a, Measurable (fun ω => T ω a))
    (B : ℕ) (x : Draws α B) (j : Fin B) :
    Measurable (fun ω => sortedCritical (T ω) B x j) := by
  apply measurable_of_Iic
  intro t
  have hcount := measurable_countHits_threshold T hT B x t
  have hset := measurableSet_le (measurable_const (a := j.val + 1)) hcount
  change MeasurableSet {ω | sortedCritical (T ω) B x j ≤ t}
  simp_rw [sortedCritical_le_iff]
  exact hset

/-- Joint measurability in observations and finite discrete bootstrap draws. -/
theorem measurable_sortedCritical_joint {Ω : Type*} [MeasurableSpace Ω] {α : Type}
    [Fintype α] [MeasurableSpace α] [MeasurableSingletonClass α]
    (T : Ω → α → ℝ) (hT : ∀ a, Measurable (fun ω => T ω a))
    (B : ℕ) (j : Fin B) :
    Measurable (fun z : Ω × Draws α B => sortedCritical (T z.1) B z.2 j) := by
  apply measurable_from_prod_countable
  intro x
  exact measurable_sortedCritical T hT B x j

/-- Replacing every bootstrap score by a certified upper endpoint can only
increase the actual sorted critical value. -/
theorem sortedCritical_mono {α : Type} (T S : α → ℝ) (hTS : ∀ a, T a ≤ S a)
    (B : ℕ) (x : Draws α B) (j : Fin B) :
    sortedCritical T B x j ≤ sortedCritical S B x j := by
  apply (sortedCritical_le_iff T B x j _).2
  exact (sortedCritical_rank S B x j).trans
    (countHits_mono (fun a ha => (hTS a).trans ha) B x)

/-- A common positive rescaling commutes exactly with the implemented sorted
critical value. This is the order-statistic step in fixed-scale cancellation. -/
theorem sortedCritical_pos_mul {α : Type} (T : α → ℝ) (c : ℝ) (hc : 0 < c)
    (B : ℕ) (x : Draws α B) (j : Fin B) :
    sortedCritical (fun a => c * T a) B x j = c * sortedCritical T B x j := by
  apply le_antisymm
  · apply (sortedCritical_le_iff (fun a => c * T a) B x j _).2
    simpa only [mul_le_mul_left hc] using sortedCritical_rank T B x j
  · have hdiv : sortedCritical T B x j ≤ sortedCritical (fun a => c * T a) B x j / c := by
      apply (sortedCritical_le_iff T B x j _).2
      simpa only [le_div_iff₀ hc, mul_comm] using sortedCritical_rank (fun a => c * T a) B x j
    simpa only [mul_comm] using (le_div_iff₀ hc).mp hdiv

end BootstrapCalibration
