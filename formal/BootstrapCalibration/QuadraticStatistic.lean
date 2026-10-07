import BootstrapCalibration.SupremumStatistic
import BootstrapCalibration.Envelope

noncomputable section
open MeasureTheory MeasureTheory.Measure ProbabilityTheory Set
open scoped BigOperators

namespace BootstrapCalibration

abbrev QuadraticSpace := EuclideanSpace ℝ (Fin 3)

def quadraticFeatures (u : ℝ) : QuadraticSpace := !₂[1, u, u^2]

theorem quadraticFeatures_ne_zero (u : ℝ) : quadraticFeatures u ≠ 0 := by
  intro h
  have h0 := congrArg (fun x : QuadraticSpace => x 0) h
  norm_num [quadraticFeatures] at h0

theorem quadraticFeatures_inner (u : ℝ) (x : QuadraticSpace) :
    inner (𝕜 := ℝ) (quadraticFeatures u) x = x 0 + u * x 1 + u^2 * x 2 := by
  simp [quadraticFeatures, PiLp.inner_apply, Fin.sum_univ_succ, mul_comm, add_assoc]

theorem quadraticFeatures_separate (x : QuadraticSpace)
    (hx : ∀ u : ℝ, inner (𝕜 := ℝ) (quadraticFeatures u) x = 0) : x = 0 := by
  have h0 := hx 0
  have h1 := hx 1
  have hn := hx (-1)
  simp only [quadraticFeatures_inner] at h0 h1 hn
  norm_num at h0 h1 hn
  ext i
  fin_cases i
  · exact h0
  · change x 1 = 0
    linarith
  · change x 2 = 0
    linarith

/-- Normalized whitened polynomial evaluation direction. An invertible
whitening map preserves separation over the whole real action line. -/
def quadraticDirection (L : QuadraticSpace ≃L[ℝ] QuadraticSpace) (u : ℝ) :
    QuadraticSpace := ‖L (quadraticFeatures u)‖⁻¹ • L (quadraticFeatures u)

theorem quadraticDirection_norm (L : QuadraticSpace ≃L[ℝ] QuadraticSpace) (u : ℝ) :
    ‖quadraticDirection L u‖ = 1 := by
  have hn : ‖L (quadraticFeatures u)‖ ≠ 0 := by
    apply norm_ne_zero_iff.mpr
    exact fun h => quadraticFeatures_ne_zero u (L.injective (by simpa using h))
  simp [quadraticDirection, norm_smul, Real.norm_eq_abs, abs_of_nonneg (norm_nonneg _), hn]

theorem quadraticDirection_separate (L : QuadraticSpace ≃L[ℝ] QuadraticSpace)
    (x : QuadraticSpace) (hx : ∀ u : ℝ, inner (𝕜 := ℝ) (quadraticDirection L u) x = 0) :
    x = 0 := by
  have hraw (u : ℝ) : inner (𝕜 := ℝ) (L (quadraticFeatures u)) x = 0 := by
    have h := hx u
    have hn : ‖L (quadraticFeatures u)‖ ≠ 0 := by
      apply norm_ne_zero_iff.mpr
      exact fun hz => quadraticFeatures_ne_zero u (L.injective (by simpa using hz))
    simpa [quadraticDirection, inner_smul_left, hn] using h
  have hadj : L.toContinuousLinearMap.adjoint x = 0 := by
    apply quadraticFeatures_separate
    intro u
    rw [ContinuousLinearMap.adjoint_inner_right]
    exact hraw u
  have hinner := L.toContinuousLinearMap.adjoint_inner_right (L.symm x) x
  rw [hadj, inner_zero_right] at hinner
  apply (inner_self_eq_zero (𝕜 := ℝ)).mp
  simpa using hinner.symm

def quadraticSupremum (L : QuadraticSpace ≃L[ℝ] QuadraticSpace) :
    Seminorm ℝ QuadraticSpace := directionSupremum (quadraticDirection L)

theorem quadraticSupremum_apply (L : QuadraticSpace ≃L[ℝ] QuadraticSpace)
    (x : QuadraticSpace) :
    quadraticSupremum L x =
      ⨆ u : ℝ, |inner (𝕜 := ℝ) (L (quadraticFeatures u)) x| /
        ‖L (quadraticFeatures u)‖ := by
  rw [quadraticSupremum, directionSupremum_apply _
    (fun u => (quadraticDirection_norm L u).le)]
  congr 1
  funext u
  simp [quadraticDirection, inner_smul_left, abs_mul, abs_inv,
    abs_of_nonneg (norm_nonneg _), div_eq_mul_inv, mul_comm]

theorem quadraticSupremum_dominates (L : QuadraticSpace ≃L[ℝ] QuadraticSpace)
    (x : QuadraticSpace) (u : ℝ) :
    |inner (𝕜 := ℝ) (L (quadraticFeatures u)) x| ≤
      quadraticSupremum L x * ‖L (quadraticFeatures u)‖ := by
  have hle : directionSeminorm (quadraticDirection L u) ≤ quadraticSupremum L :=
    le_ciSup (bounded_directionSeminorm _ (fun v => (quadraticDirection_norm L v).le)) u
  have h := hle x
  simp only [directionSeminorm_apply, quadraticDirection, inner_smul_left,
    RCLike.conj_to_real, abs_mul, abs_inv, abs_norm] at h
  have hn : 0 < ‖L (quadraticFeatures u)‖ := by
    apply norm_pos_iff.mpr
    exact fun hz => quadraticFeatures_ne_zero u (L.injective (by simpa using hz))
  apply (div_le_iff₀ hn).mp
  simpa only [div_eq_mul_inv, mul_comm] using h

/-- Coefficient-statistic coverage yields the frozen predictor's entire-line
error certificate and its two-width excess bound. -/
theorem quadraticSupremum_envelope (L : QuadraticSpace ≃L[ℝ] QuadraticSpace)
    (x : QuadraticSpace) (σ q : ℝ) (hσ : 0 ≤ σ) (hq : quadraticSupremum L x ≤ q)
    (fixed auxiliary truth : ℝ → ℝ)
    (hmodel : ∀ u, auxiliary u - truth u = σ * inner (𝕜 := ℝ) (L (quadraticFeatures u)) x) :
    ∀ u, let w := σ * q * ‖L (quadraticFeatures u)‖
      |fixed u - truth u| ≤ radius (fixed u) (auxiliary u) w ∧
      (0 ≤ radius (fixed u) (auxiliary u) w - |fixed u - truth u| ∧
        radius (fixed u) (auxiliary u) w - |fixed u - truth u| ≤ 2 * w) := by
  intro u
  have hband : |auxiliary u - truth u| ≤ σ * q * ‖L (quadraticFeatures u)‖ := by
    rw [hmodel, abs_mul, abs_of_nonneg hσ]
    have h := mul_le_mul_of_nonneg_left
      ((quadraticSupremum_dominates L x u).trans
        (mul_le_mul_of_nonneg_right hq (norm_nonneg _))) hσ
    simpa only [mul_assoc] using h
  exact ⟨error_le_radius _ _ _ _ hband, radius_accuracy _ _ _ _ hband⟩

/-- The actual all-real quadratic supremum admits every interior CDF
threshold under an absolutely continuous coefficient law. -/
theorem quadraticSupremum_threshold (μ : Measure QuadraticSpace)
    [IsProbabilityMeasure μ] (hμ : μ ≪ volume)
    (L : QuadraticSpace ≃L[ℝ] QuadraticSpace) (p : ℝ) (hp : 0 < p) (hp1 : p < 1) :
    ∃ t, μ.real {x | quadraticSupremum L x ≤ t} = p := by
  have hc : Continuous (quadraticSupremum L) :=
    directionSupremum_continuous _ (fun u => (quadraticDirection_norm L u).le)
  letI : NoAtoms (μ.map (quadraticSupremum L)) :=
    noAtoms_map_directionSupremum μ volume hμ _
      (fun u => (quadraticDirection_norm L u).le) (quadraticDirection_separate L)
  haveI : IsProbabilityMeasure (μ.map (quadraticSupremum L)) :=
    isProbabilityMeasure_map hc.measurable.aemeasurable
  obtain ⟨t, ht⟩ := exists_cdf_threshold (μ.map (quadraticSupremum L)) p hp hp1
  refine ⟨t, ?_⟩
  simpa only [measureReal_def, Measure.map_apply hc.measurable measurableSet_Iic,
    mem_Iic, Set.preimage_setOf_eq] using ht


/-- The same statistic in ordinary coefficient coordinates; the ambient Pi
norm is irrelevant because this is the explicit directional seminorm. -/
def quadraticPiSupremum (L : QuadraticSpace ≃L[ℝ] QuadraticSpace) :
    Seminorm ℝ (Fin 3 → ℝ) :=
  (quadraticSupremum L).comp (EuclideanSpace.equiv (Fin 3) ℝ).symm.toLinearMap

theorem quadraticPiSupremum_continuous (L : QuadraticSpace ≃L[ℝ] QuadraticSpace) :
    Continuous (quadraticPiSupremum L) :=
  (directionSupremum_continuous _ (fun u => (quadraticDirection_norm L u).le)).comp
    (EuclideanSpace.equiv (Fin 3) ℝ).symm.continuous

theorem quadraticPiSupremum_definite (L : QuadraticSpace ≃L[ℝ] QuadraticSpace)
    (x : Fin 3 → ℝ) (hx : quadraticPiSupremum L x = 0) : x = 0 := by
  have hz := directionSupremum_definite (quadraticDirection L)
    (fun u => (quadraticDirection_norm L u).le) (quadraticDirection_separate L)
    ((EuclideanSpace.equiv (Fin 3) ℝ).symm x) hx
  simpa using congrArg (EuclideanSpace.equiv (Fin 3) ℝ) hz

theorem quadraticPiSupremum_threshold (μ : Measure (Fin 3 → ℝ))
    [IsProbabilityMeasure μ] (hμ : μ ≪ volume)
    (L : QuadraticSpace ≃L[ℝ] QuadraticSpace) (p : ℝ) (hp : 0 < p) (hp1 : p < 1) :
    ∃ t, μ.real {x | quadraticPiSupremum L x ≤ t} = p := by
  have hc := quadraticPiSupremum_continuous L
  letI := noAtoms_map_seminorm μ volume hμ (quadraticPiSupremum L) hc
    (quadraticPiSupremum_definite L)
  haveI : IsProbabilityMeasure (μ.map (quadraticPiSupremum L)) :=
    isProbabilityMeasure_map hc.measurable.aemeasurable
  obtain ⟨t, ht⟩ := exists_cdf_threshold (μ.map (quadraticPiSupremum L)) p hp hp1
  refine ⟨t, ?_⟩
  simpa only [measureReal_def, Measure.map_apply hc.measurable measurableSet_Iic,
    mem_Iic, Set.preimage_setOf_eq] using ht

end BootstrapCalibration
