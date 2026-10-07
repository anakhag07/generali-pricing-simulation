import BootstrapCalibration.FiniteBootstrap
import BootstrapCalibration.Coverage
import BootstrapCalibration.TwoReplicas

open MeasureTheory
open scoped BigOperators Classical

namespace BootstrapCalibration
namespace FiniteLaw

variable {Ω α : Type} [MeasurableSpace Ω] [Fintype α]
  [MeasurableSpace α]

/-- Integrating a conditional finite-law bound preserves its failure probability. -/
theorem prod_event_le (P : FiniteLaw α) (μ : Measure Ω) [IsProbabilityMeasure μ]
    (E : Set (Ω × α)) (hE : MeasurableSet E) (eta : ℝ) (heta : 0 ≤ eta)
    (hbound : ∀ ω, P.mass (fun a => (ω, a) ∈ E) ≤ eta) :
    (μ.prod P.toPMF.toMeasure).real E ≤ eta := by
  have hslice (ω : Ω) :
      P.toPMF.toMeasure (Prod.mk ω ⁻¹' E) =
        ENNReal.ofReal (P.mass (fun a => (ω, a) ∈ E)) := by
    rw [P.toPMF.toMeasure_apply_eq_toOuterMeasure_apply _
      (hE.preimage measurable_prodMk_left)]
    exact P.toPMF_event _
  have hprob : μ.prod P.toPMF.toMeasure E ≤ ENNReal.ofReal eta := by
    rw [Measure.prod_apply hE]
    simp_rw [hslice]
    calc
      (∫⁻ ω, ENNReal.ofReal (P.mass (fun a => (ω, a) ∈ E)) ∂μ)
          ≤ ∫⁻ _, ENNReal.ofReal eta ∂μ :=
        lintegral_mono (fun ω => ENNReal.ofReal_le_ofReal (hbound ω))
      _ = ENNReal.ofReal eta := by simp
  have hreal := ENNReal.toReal_mono (by simp) hprob
  simpa only [measureReal_def, ENNReal.toReal_ofReal heta] using hreal

end FiniteLaw

instance drawsMeasurableSpace (α : Type) [MeasurableSpace α] :
    (n : ℕ) → MeasurableSpace (Draws α n)
  | 0 => inferInstanceAs (MeasurableSpace Unit)
  | n + 1 => @Prod.instMeasurableSpace α (Draws α n) _ (drawsMeasurableSpace α n)

instance drawsMeasurableSingletonClass (α : Type) [MeasurableSpace α]
    [MeasurableSingletonClass α] : (n : ℕ) → MeasurableSingletonClass (Draws α n)
  | 0 => inferInstanceAs (MeasurableSingletonClass Unit)
  | n + 1 => by
      letI := drawsMeasurableSingletonClass α n
      exact inferInstanceAs (MeasurableSingletonClass (α × Draws α n))

namespace FiniteLaw

variable {Ω α : Type} [MeasurableSpace Ω] [Fintype α]
  [MeasurableSpace α]

/-- The finite-B calibration bound holds jointly over random data and bootstrap draws.
No coverage or bootstrap-approximation premise is used in this theorem. -/
theorem unconditional_quantile_failure (P : FiniteLaw α) (μ : Measure Ω)
    [IsProbabilityMeasure μ] (T : Ω → α → ℝ) {p eta : ℝ}
    (hp : 0 ≤ p) (hp1 : p ≤ 1) (heta : 0 ≤ eta)
    (B k : ℕ) (Q : Ω → Draws α B → ℝ)
    (hQ : ∀ ω x, k ≤ countHits (fun a => T ω a ≤ Q ω x) B x)
    (hE : MeasurableSet {z : Ω × Draws α B | P.cdf (T z.1) (Q z.1 z.2) < p})
    (hrank : binomialTail p B k ≤ eta) :
    (μ.prod (P.iid B).toPMF.toMeasure).real
      {z | P.cdf (T z.1) (Q z.1 z.2) < p} ≤ eta := by
  apply (P.iid B).prod_event_le μ _ hE eta heta
  intro ω
  exact (P.quantile_failure_le (T ω) hp hp1 B k (Q ω) (hQ ω)).trans hrank

/-- Complete finite-resampling coverage reduction, from pairwise comparisons
under the observation law to coverage jointly over observations and bootstrap
draws. The remaining model-specific premise is `hpair`; for Gaussian pairs OLS
it must be obtained from the covariance/density comparison, not assumed. -/
theorem coverage_of_pair_comparisons (P : FiniteLaw α) (μ : Measure Ω)
    [IsProbabilityMeasure μ] (S : Ω → ℝ) (hS : Measurable S)
    (T : Ω → α → ℝ) (hT : ∀ a, Measurable (fun ω => T ω a))
    (B k : ℕ) (Q : Ω → Draws α B → ℝ)
    (hQmeas : Measurable (fun z : Ω × Draws α B => Q z.1 z.2))
    (hQ : ∀ ω x, k ≤ countHits (fun a => T ω a ≤ Q ω x) B x)
    (t p tau eta error : ℝ) (htau : 0 < tau)
    (hp : 0 ≤ p) (hp1 : p ≤ 1) (hptau : 0 ≤ p - tau) (heta : 0 ≤ eta)
    (htruth : μ.real {ω | S ω ≤ t} = p - tau)
    (hrank : binomialTail p B k ≤ eta)
    (d : α → α → ℝ)
    (hpair : ∀ a b, (∫ ω,
      ((if T ω a ≤ t then (1 : ℝ) else 0) - (p - tau)) *
      ((if T ω b ≤ t then (1 : ℝ) else 0) - (p - tau)) ∂μ) ≤ d a b)
    (hweighted : (∑ a, ∑ b, P.weight a * P.weight b * d a b) ≤ error) :
    p - tau - eta - error / tau^2 ≤
      (μ.prod (P.iid B).toPMF.toMeasure).real {z | S z.1 ≤ Q z.1 z.2} := by
  have hptau1 : p - tau ≤ 1 := by linarith
  have hE (a : α) : MeasurableSet {ω | T ω a ≤ t} :=
    measurableSet_le (hT a) measurable_const
  have hEj (a : α) : MeasurableSet
      {z : Ω × Draws α B | T z.1 a ≤ t} :=
    measurableSet_le ((hT a).comp measurable_fst) measurable_const
  have hmcmeas : MeasurableSet
      {z : Ω × Draws α B | P.cdf (T z.1) (Q z.1 z.2) < p} := by
    apply measurableSet_lt _ measurable_const
    exact P.measurable_mass _ (fun a =>
      measurableSet_le ((hT a).comp measurable_fst) hQmeas)
  apply coverage_of_cdf_mean_square (μ.prod (P.iid B).toPMF.toMeasure)
    (fun z => S z.1) (fun z => Q z.1 z.2) (hS.comp measurable_fst) hQmeas
    (fun z => P.cdf (T z.1)) t p tau error eta htau
  · have heq : {z : Ω × Draws α B | S z.1 ≤ t} =
        {ω | S ω ≤ t} ×ˢ Set.univ := by ext z; simp
    rw [heq, measureReal_prod_prod]
    simpa using htruth
  · intro z
    exact P.cdf_mono (T z.1)
  · exact P.mass_centered_square_integrable _ _ hEj (p - tau) hptau hptau1
  · rw [integral_fun_fst (fun ω : Ω => (P.cdf (T ω) t - (p - tau))^2)]
    simp only [measureReal_univ_eq_one, one_smul]
    exact (P.cdf_mse_le_pair_bounds μ _ hE (p - tau) hptau hptau1 d hpair).trans
      hweighted
  · exact P.unconditional_quantile_failure μ T hp hp1 heta B k Q hQ hmcmeas hrank

end FiniteLaw
end BootstrapCalibration
