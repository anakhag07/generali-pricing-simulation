import BootstrapCalibration.FiniteBootstrap

open MeasureTheory
open scoped BigOperators Classical

namespace BootstrapCalibration.FiniteLaw

variable {Ω α : Type} [MeasurableSpace Ω] [Fintype α]

/-- Measurability of a conditional CDF formed from a fixed finite law. -/
theorem measurable_mass (P : FiniteLaw α) (E : Ω → α → Prop)
    (hE : ∀ a, MeasurableSet {ω | E ω a}) :
    Measurable (fun ω => P.mass (E ω)) := by
  unfold mass
  exact Finset.measurable_sum _ (fun a _ =>
    Measurable.ite (hE a) measurable_const measurable_const)

/-- Bounded CDF discrepancies have an integrable square. -/
theorem mass_centered_square_integrable (P : FiniteLaw α) (μ : Measure Ω)
    [IsFiniteMeasure μ] (E : Ω → α → Prop)
    (hE : ∀ a, MeasurableSet {ω | E ω a}) (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    Integrable (fun ω => (P.mass (E ω) - p)^2) μ := by
  have hm : Measurable (fun ω => (P.mass (E ω) - p)^2) :=
    ((P.measurable_mass E hE).sub measurable_const).pow_const 2
  have hLp : MemLp (fun ω => (P.mass (E ω) - p)^2) 1 μ :=
    MemLp.of_bound hm.aestronglyMeasurable 1 (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      have hb : |P.mass (E ω) - p| ≤ 1 := by
        apply abs_le.mpr
        constructor <;> linarith [P.mass_nonneg (E ω), P.mass_le_one (E ω)]
      nlinarith [sq_abs (P.mass (E ω) - p), abs_nonneg (P.mass (E ω) - p)])
  exact hLp.integrable le_rfl

omit [Fintype α] in
private theorem centered_pair_integrable (μ : Measure Ω) [IsFiniteMeasure μ]
    (E : Ω → α → Prop) (hE : ∀ a, MeasurableSet {ω | E ω a})
    (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) (a b : α) :
    Integrable (fun ω => ((if E ω a then (1 : ℝ) else 0) - p) *
      ((if E ω b then (1 : ℝ) else 0) - p)) μ := by
  classical
  have hm (c : α) : Measurable (fun ω => (if E ω c then (1 : ℝ) else 0) - p) :=
    (Measurable.ite (hE c) measurable_const measurable_const).sub measurable_const
  have hb (c : α) (ω : Ω) : |(if E ω c then (1 : ℝ) else 0) - p| ≤ 1 := by
    split_ifs <;> apply abs_le.mpr <;> constructor <;> linarith
  have hLp : MemLp (fun ω => ((if E ω a then (1 : ℝ) else 0) - p) *
      ((if E ω b then (1 : ℝ) else 0) - p)) 1 μ :=
    MemLp.of_bound ((hm a).mul (hm b)).aestronglyMeasurable 1
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_mul]
        nlinarith [hb a ω, hb b ω,
          abs_nonneg ((if E ω a then (1 : ℝ) else 0) - p),
          abs_nonneg ((if E ω b then (1 : ℝ) else 0) - p)])
  exact hLp.integrable le_rfl

/-- The conditional-CDF square is the centered product of two independent resampling laws.
The observation variable remains shared between the replicas. -/
theorem mass_centered_square (P : FiniteLaw α) (E : α → Prop) (p : ℝ) :
    (P.mass E - p)^2 = ∑ a, ∑ b, P.weight a * P.weight b *
      (((if E a then (1 : ℝ) else 0) - p) * ((if E b then (1 : ℝ) else 0) - p)) := by
  classical
  have hs : (∑ a, P.weight a * ((if E a then (1 : ℝ) else 0) - p)) =
      P.mass E - p := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, P.total, one_mul]
    congr 1
    apply Finset.sum_congr rfl
    intro a ha
    by_cases he : E a <;> simp [he]
  rw [← hs, pow_two, Finset.sum_mul]
  simp_rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a ha
  apply Finset.sum_congr rfl
  intro b hb
  ring

/-- Exact second-moment identity, including integration over random observations. -/
theorem integrated_mass_centered_square (P : FiniteLaw α) (μ : Measure Ω)
    [IsFiniteMeasure μ] (E : Ω → α → Prop)
    (hE : ∀ a, MeasurableSet {ω | E ω a}) (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    (∫ ω, (P.mass (E ω) - p)^2 ∂μ) =
      ∑ a, ∑ b, P.weight a * P.weight b *
        ∫ ω, ((if E ω a then (1 : ℝ) else 0) - p) *
          ((if E ω b then (1 : ℝ) else 0) - p) ∂μ := by
  classical
  have hi (a b : α) := (centered_pair_integrable μ E hE p hp hp1 a b).const_mul
    (P.weight a * P.weight b)
  simp_rw [P.mass_centered_square]
  rw [integral_finset_sum _ (fun a _ => integrable_finset_sum _ (fun b _ => hi a b))]
  apply Finset.sum_congr rfl
  intro a ha
  rw [integral_finset_sum _ (fun b _ => hi a b)]
  apply Finset.sum_congr rfl
  intro b hb
  exact integral_const_mul _ _

/-- Pairwise distribution comparisons yield the CDF mean-square bound.
The pair comparisons remain explicit hypotheses until instantiated for Gaussian OLS. -/
theorem cdf_mse_le_pair_bounds (P : FiniteLaw α) (μ : Measure Ω)
    [IsFiniteMeasure μ] (E : Ω → α → Prop)
    (hE : ∀ a, MeasurableSet {ω | E ω a}) (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (d : α → α → ℝ)
    (hd : ∀ a b, (∫ ω, ((if E ω a then (1 : ℝ) else 0) - p) *
      ((if E ω b then (1 : ℝ) else 0) - p) ∂μ) ≤ d a b) :
    (∫ ω, (P.mass (E ω) - p)^2 ∂μ) ≤ ∑ a, ∑ b, P.weight a * P.weight b * d a b := by
  rw [P.integrated_mass_centered_square μ E hE p hp hp1]
  apply Finset.sum_le_sum
  intro a ha
  apply Finset.sum_le_sum
  intro b hb
  exact mul_le_mul_of_nonneg_left (hd a b) (mul_nonneg (P.nonneg a) (P.nonneg b))

end BootstrapCalibration.FiniteLaw
