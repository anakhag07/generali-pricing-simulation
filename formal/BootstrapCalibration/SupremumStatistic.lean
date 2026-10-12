import BootstrapCalibration.NormCDF

noncomputable section
open MeasureTheory MeasureTheory.Measure Set
open scoped Topology

namespace BootstrapCalibration

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Positive levels of a continuous seminorm lie on the frontier of its
convex sublevel set. Radial scaling excludes interior points. -/
theorem seminorm_level_subset_frontier (q : Seminorm ℝ E) (hq : Continuous q)
    (t : ℝ) (ht : 0 < t) :
    {x | q x = t} ⊆ frontier {x | q x ≤ t} := by
  intro x hx
  have hclosed : IsClosed {x | q x ≤ t} := isClosed_le hq continuous_const
  change x ∈ closure {x | q x ≤ t} ∧ x ∉ interior {x | q x ≤ t}
  constructor
  · rw [hclosed.closure_eq]
    exact hx.le
  · intro hi
    obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp (mem_interior_iff_mem_nhds.mp hi)
    let c : ℝ := ε / (2 * (‖x‖ + 1))
    have hc : 0 < c := div_pos hε (by positivity)
    have hcx : c * ‖x‖ < ε := by
      have hden : 0 < 2 * (‖x‖ + 1) := by positivity
      dsimp [c]
      apply (div_mul_eq_mul_div _ _ _).trans_lt
      apply (div_lt_iff₀ hden).2
      nlinarith [norm_nonneg x]
    have hdist : dist ((1 + c) • x) x < ε := by
      rw [dist_eq_norm]
      have heq : (1 + c) • x - x = c • x := by rw [add_smul, one_smul]; abel
      rw [heq, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
      exact hcx
    have hle := hball hdist
    change q ((1 + c) • x) ≤ t at hle
    rw [map_smul_eq_mul, Real.norm_eq_abs, abs_of_pos (by linarith : 0 < 1 + c), hx] at hle
    nlinarith

/-- A continuous definite seminorm has null level sets under any absolutely
continuous finite-dimensional law, even when it is not the ambient norm. -/
theorem noAtoms_map_seminorm [FiniteDimensional ℝ E] [Nontrivial E]
    [MeasurableSpace E] [BorelSpace E] (μ ν : Measure E) [IsAddHaarMeasure ν]
    (hμ : μ ≪ ν) (q : Seminorm ℝ E) (hq : Continuous q)
    (hdef : ∀ x, q x = 0 → x = 0) : NoAtoms (μ.map q) := by
  apply noAtoms_map_of_null_levels μ q hq.measurable
  intro t
  apply hμ
  rcases lt_trichotomy t 0 with ht | ht | ht
  · have heq : {x | q x = t} = (∅ : Set E) := by
      ext x
      simp only [mem_setOf_eq, mem_empty_iff_false, iff_false]
      intro hx
      linarith [apply_nonneg q x]
    rw [heq, measure_empty]
  · subst t
    have heq : {x | q x = 0} = ({0} : Set E) := by
      ext x
      constructor
      · exact hdef x
      · intro hx
        simpa only [Set.mem_singleton_iff.mp hx] using map_zero q
    rw [heq]
    exact measure_singleton 0
  · apply measure_mono_null (seminorm_level_subset_frontier q hq t ht)
    have hconv : Convex ℝ {x | q x ≤ t} := by
      simpa only [Seminorm.closedBall_zero_eq] using q.convex_closedBall 0 t
    exact hconv.addHaar_frontier ν

section Directional

variable {F ι : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F] [Nonempty ι]

/-- Absolute directional evaluation as a seminorm. -/
def directionSeminorm (v : F) : Seminorm ℝ F :=
  (normSeminorm ℝ ℝ).comp (innerSL ℝ v).toLinearMap

@[simp] theorem directionSeminorm_apply (v x : F) :
    directionSeminorm v x = |inner (𝕜 := ℝ) v x| := rfl

theorem directionSeminorm_le_norm (v : F) (hv : ‖v‖ ≤ 1) :
    directionSeminorm v ≤ normSeminorm ℝ F := by
  intro x
  change |inner (𝕜 := ℝ) v x| ≤ ‖x‖
  calc
    _ ≤ ‖v‖ * ‖x‖ := abs_real_inner_le_norm _ _
    _ ≤ ‖x‖ := mul_le_of_le_one_left (norm_nonneg x) hv

omit [Nonempty ι] in
theorem bounded_directionSeminorm (d : ι → F) (hd : ∀ i, ‖d i‖ ≤ 1) :
    BddAbove (range (fun i => directionSeminorm (d i))) := by
  refine ⟨normSeminorm ℝ F, ?_⟩
  rintro _ ⟨i, rfl⟩
  exact directionSeminorm_le_norm (d i) (hd i)

/-- Supremum over an arbitrary family of bounded directions. -/
def directionSupremum (d : ι → F) : Seminorm ℝ F := ⨆ i, directionSeminorm (d i)

omit [Nonempty ι] in
theorem directionSupremum_apply (d : ι → F) (hd : ∀ i, ‖d i‖ ≤ 1) (x : F) :
    directionSupremum d x = ⨆ i, |inner (𝕜 := ℝ) (d i) x| := by
  exact Seminorm.iSup_apply (bounded_directionSeminorm d hd)

theorem directionSupremum_continuous (d : ι → F) (hd : ∀ i, ‖d i‖ ≤ 1) :
    Continuous (directionSupremum d) := by
  apply Seminorm.continuous_of_le (q := normSeminorm ℝ F) continuous_norm
  exact ciSup_le (fun i => directionSeminorm_le_norm (d i) (hd i))

omit [Nonempty ι] in
theorem directionSupremum_definite (d : ι → F) (hd : ∀ i, ‖d i‖ ≤ 1)
    (hsep : ∀ x, (∀ i, inner (𝕜 := ℝ) (d i) x = 0) → x = 0)
    (x : F) (hx : directionSupremum d x = 0) : x = 0 := by
  apply hsep x
  intro i
  have hle : directionSeminorm (d i) ≤ directionSupremum d :=
    le_ciSup (bounded_directionSeminorm d hd) i
  have habs : |inner (𝕜 := ℝ) (d i) x| ≤ 0 := by simpa only [directionSeminorm_apply, hx] using hle x
  exact abs_eq_zero.mp (le_antisymm habs (abs_nonneg _))

/-- The continuum supremum statistic is atomless under an absolutely
continuous law when its normalized directions separate vectors. -/
theorem noAtoms_map_directionSupremum [FiniteDimensional ℝ F] [Nontrivial F]
    [MeasurableSpace F] [BorelSpace F] (μ ν : Measure F) [IsAddHaarMeasure ν]
    (hμ : μ ≪ ν) (d : ι → F) (hd : ∀ i, ‖d i‖ ≤ 1)
    (hsep : ∀ x, (∀ i, inner (𝕜 := ℝ) (d i) x = 0) → x = 0) :
    NoAtoms (μ.map (directionSupremum d)) :=
  noAtoms_map_seminorm μ ν hμ _ (directionSupremum_continuous d hd)
    (directionSupremum_definite d hd hsep)

end Directional
end BootstrapCalibration
