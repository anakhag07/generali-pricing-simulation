import Mathlib

/-! Continuity and existence of deterministic CDF thresholds. These statements
derive CDF continuity from null level sets, rather than taking it as a premise.
The final supremum statistic still needs to be identified with its norm. -/

noncomputable section
open MeasureTheory MeasureTheory.Measure ProbabilityTheory Set Filter
open scoped Topology

namespace BootstrapCalibration

/-- An atomless real probability law has a continuous CDF. -/
theorem continuous_cdf_of_noAtoms (μ : Measure ℝ) [IsProbabilityMeasure μ]
    [NoAtoms μ] : Continuous (cdf μ) := by
  apply continuous_iff_continuousAt.mpr
  intro t
  apply (cdf μ).mono.continuousAt_iff_leftLim_eq_rightLim.mpr
  rw [(cdf μ).rightLim_eq]
  have hz : ENNReal.ofReal (cdf μ t - Function.leftLim (cdf μ) t) = 0 := by
    rw [← (cdf μ).measure_singleton, measure_cdf]
    exact measure_singleton t
  have hle := ENNReal.ofReal_eq_zero.mp hz
  have hleft := (cdf μ).mono.leftLim_le (le_refl t)
  linarith

/-- Every interior probability level is attained by an atomless CDF. -/
theorem exists_cdf_threshold (μ : Measure ℝ) [IsProbabilityMeasure μ]
    [NoAtoms μ] (p : ℝ) (hp : 0 < p) (hp1 : p < 1) :
    ∃ t, μ.real (Iic t) = p := by
  have hlo : ∃ t, cdf μ t ≤ p := by
    obtain ⟨t, ht⟩ := ((tendsto_cdf_atBot μ).eventually (gt_mem_nhds hp)).exists
    exact ⟨t, ht.le⟩
  have hhi : ∃ t, p ≤ cdf μ t := by
    obtain ⟨t, ht⟩ := ((tendsto_cdf_atTop μ).eventually (lt_mem_nhds hp1)).exists
    exact ⟨t, ht.le⟩
  obtain ⟨t, ht⟩ := mem_range_of_exists_le_of_exists_ge
    (continuous_cdf_of_noAtoms μ) hlo hhi
  exact ⟨t, by simpa only [cdf_eq_real] using ht⟩

/-- A measurable statistic with null level sets has an atomless pushforward. -/
theorem noAtoms_map_of_null_levels {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (T : Ω → ℝ) (hT : Measurable T)
    (hlevels : ∀ t, μ {ω | T ω = t} = 0) : NoAtoms (μ.map T) := by
  constructor
  intro t
  rw [Measure.map_apply hT (measurableSet_singleton t)]
  exact hlevels t

/-- The norm under an absolutely continuous finite-dimensional law has no
atoms. Haar-null spheres supply the needed geometric fact. -/
theorem noAtoms_map_norm {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] [Nontrivial E] [MeasurableSpace E] [BorelSpace E]
    (μ ν : Measure E) [IsAddHaarMeasure ν] (hμ : μ ≪ ν) :
    NoAtoms (μ.map (fun x : E => ‖x‖)) := by
  apply noAtoms_map_of_null_levels μ _ continuous_norm.measurable
  intro t
  apply hμ
  have heq : {x : E | ‖x‖ = t} = Metric.sphere 0 t := by
    ext x
    simp [Metric.mem_sphere, dist_zero_right]
  rw [heq]
  exact addHaar_sphere ν 0 t

/-- A continuous law's norm admits the exact threshold needed by the coverage
reduction; no independence from the bootstrap critical value is asserted. -/
theorem exists_norm_threshold {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] [Nontrivial E] [MeasurableSpace E] [BorelSpace E]
    (μ ν : Measure E) [IsProbabilityMeasure μ] [IsAddHaarMeasure ν]
    (hμ : μ ≪ ν) (p : ℝ) (hp : 0 < p) (hp1 : p < 1) :
    ∃ t, μ.real {x : E | ‖x‖ ≤ t} = p := by
  letI := noAtoms_map_norm μ ν hμ
  haveI : IsProbabilityMeasure (μ.map (fun x : E => ‖x‖)) :=
    isProbabilityMeasure_map continuous_norm.measurable.aemeasurable
  obtain ⟨t, ht⟩ := exists_cdf_threshold (μ.map (fun x : E => ‖x‖)) p hp hp1
  refine ⟨t, ?_⟩
  simpa only [measureReal_def, Measure.map_apply continuous_norm.measurable measurableSet_Iic,
    mem_Iic, Set.preimage_setOf_eq] using ht

end BootstrapCalibration
