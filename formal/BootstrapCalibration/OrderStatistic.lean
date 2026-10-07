import BootstrapCalibration.FiniteBootstrap

noncomputable section
open scoped Classical

namespace BootstrapCalibration

def drawScores {α : Type} (T : α → ℝ) : (B : ℕ) → Draws α B → List ℝ
  | 0, _ => []
  | B+1, x => T x.1 :: drawScores T B x.2

@[simp] theorem length_drawScores {α : Type} (T : α → ℝ) (B : ℕ) (x : Draws α B) :
    (drawScores T B x).length = B := by
  induction B with
  | zero => rfl
  | succ B ih => simp [drawScores, ih]

theorem drawScores_count {α : Type} (T : α → ℝ) (B : ℕ) (x : Draws α B) (c : ℝ) :
    (drawScores T B x).countP (fun t => decide (t ≤ c)) =
      countHits (fun a => T a ≤ c) B x := by
  induction B with
  | zero => rfl
  | succ B ih =>
      simp only [drawScores, List.countP_cons, countHits, ih]
      by_cases h : T x.1 ≤ c <;> simp [h, Nat.add_comm]

theorem sorted_list_rank (l : List ℝ) (hl : l.Sorted (· ≤ ·)) (j : Fin l.length) :
    j.val + 1 ≤ l.countP (fun t => decide (t ≤ l.get j)) := by
  induction l with
  | nil => exact j.elim0
  | cons a l ih =>
      cases j using Fin.cases with
      | zero => simp
      | succ j =>
          have hp := List.pairwise_cons.mp hl
          have ha : a ≤ l.get j := hp.1 _ (List.get_mem l j)
          have ha' : a ≤ l[j.val] := ha
          have h := ih hp.2 j
          simpa [List.countP_cons, ha', Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
            using Nat.succ_le_succ h

/-- The ordinary sorted sample order statistic, using zero-based index j. -/
def sortedCritical {α : Type} (T : α → ℝ) (B : ℕ) (x : Draws α B) (j : Fin B) : ℝ :=
  let l := (drawScores T B x).mergeSort (· ≤ ·)
  l.get ⟨j.val, by simpa only [l, List.length_mergeSort, length_drawScores] using j.isLt⟩

/-- Sorting, including tied values, discharges the quantile theorem's rank
condition for the actual order statistic. -/
theorem sortedCritical_rank {α : Type} (T : α → ℝ) (B : ℕ) (x : Draws α B) (j : Fin B) :
    j.val + 1 ≤ countHits (fun a => T a ≤ sortedCritical T B x j) B x := by
  let l := (drawScores T B x).mergeSort (· ≤ ·)
  have hsort : l.Sorted (· ≤ ·) := List.sorted_mergeSort' (· ≤ ·) _
  have hj : j.val < l.length := by
    simpa only [l, List.length_mergeSort, length_drawScores] using j.isLt
  have hr := sorted_list_rank l hsort ⟨j.val, hj⟩
  have hperm := List.mergeSort_perm (drawScores T B x) (· ≤ ·)
  rw [← drawScores_count]
  rw [← hperm.countP_eq]
  exact hr

end BootstrapCalibration
