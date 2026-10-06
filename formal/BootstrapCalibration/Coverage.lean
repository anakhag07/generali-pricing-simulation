import Mathlib.MeasureTheory.Measure.Real
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Linarith

/-!
# Statistical reduction for a bootstrap critical value

The critical value may depend on the same data as the true error statistic.
The reduction therefore uses an event inclusion, not an independence argument.
The CDF comparison and its exceptional-event probability are hypotheses: this
file does not establish a finite-sample approximation for pairs bootstrap.
-/

open MeasureTheory

namespace BootstrapCalibration

variable {Ω : Type*} [MeasurableSpace Ω]

/-- A union-bound transfer, valid also for outer measures of arbitrary events. -/
theorem coverage_of_event_transfer (μ : Measure Ω) [IsProbabilityMeasure μ]
    (safe bad target : Set Ω) (alpha rho : ℝ)
    (htransfer : safe ⊆ target ∪ bad)
    (hsafe : 1 - alpha ≤ μ.real safe) (hbad : μ.real bad ≤ rho) :
    1 - alpha - rho ≤ μ.real target := by
  have hmono : μ.real safe ≤ μ.real (target ∪ bad) :=
    measureReal_mono htransfer
  have hunion : μ.real (target ∪ bad) ≤ μ.real target + μ.real bad :=
    measureReal_union_le target bad
  linarith

/-- A random critical value loses at most its probability of falling below a
deterministic threshold. Measurability makes every displayed measure a probability;
no independence of the statistic and critical value is assumed. -/
theorem random_threshold_coverage (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T q : Ω → ℝ) (_hT : Measurable T) (_hq : Measurable q)
    (t alpha rho : ℝ)
    (htruth : 1 - alpha ≤ μ.real {ω | T ω ≤ t})
    (hcutoff : μ.real {ω | q ω < t} ≤ rho) :
    1 - alpha - rho ≤ μ.real {ω | T ω ≤ q ω} := by
  apply coverage_of_event_transfer μ {ω | T ω ≤ t} {ω | q ω < t}
    {ω | T ω ≤ q ω} alpha rho ?_ htruth hcutoff
  intro ω hω
  by_cases hq : q ω < t
  · exact Or.inr hq
  · exact Or.inl (le_trans hω (le_of_not_gt hq))

/-- An upper comparison of a bootstrap CDF with a reference CDF prevents a
calibrated critical value from falling below a deterministic threshold.
The strict left-threshold condition accommodates CDF jumps without assuming
continuity or treating the value at a jump as a left limit. -/
theorem threshold_le_of_cdf_comparison (F G : ℝ → ℝ)
    (q t p error : ℝ)
    (hleft : ∀ x < t, F x + error < p)
    (hcomparison : G q ≤ F q + error) (hcalibration : p ≤ G q) :
    t ≤ q := by
  by_contra h
  have hbelow : q < t := lt_of_not_ge h
  have hstrict := hleft q hbelow
  linarith

/-- Conditional statistical reduction from a reference CDF and bootstrap CDF
comparison outside a bad event. In an application `error` and the probability
`rho` must be proved for the actual resampling law; they are not certified here. -/
theorem coverage_of_cdf_comparison (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T q : Ω → ℝ) (hT : Measurable T) (hq : Measurable q)
    (F : ℝ → ℝ) (G : Ω → ℝ → ℝ) (bad : Set Ω)
    (t p error alpha rho : ℝ)
    (hcdf : F t = μ.real {ω | T ω ≤ t})
    (hlevel : 1 - alpha ≤ F t)
    (hleft : ∀ x < t, F x + error < p)
    (hcomparison : ∀ ω ∉ bad, G ω (q ω) ≤ F (q ω) + error)
    (hcalibration : ∀ ω ∉ bad, p ≤ G ω (q ω))
    (hbad : μ.real bad ≤ rho) :
    1 - alpha - rho ≤ μ.real {ω | T ω ≤ q ω} := by
  have hcutoff : {ω | q ω < t} ⊆ bad := by
    intro ω hbelow
    by_contra hgood
    have hthreshold := threshold_le_of_cdf_comparison F (G ω)
      (q ω) t p error hleft (hcomparison ω hgood) (hcalibration ω hgood)
    exact (not_lt_of_ge hthreshold) hbelow
  apply random_threshold_coverage μ T q hT hq t alpha rho
  · simpa only [hcdf] using hlevel
  · exact le_trans (measureReal_mono hcutoff) hbad

/-- Markov's inequality applied to a squared error, with an explicit positive
threshold and an integrability hypothesis. -/
theorem squared_moment_tail (μ : Measure Ω) [IsProbabilityMeasure μ]
    (Z : Ω → ℝ) (tau error : ℝ) (htau : 0 < tau)
    (hintegrable : Integrable (fun ω => (Z ω) ^ 2) μ)
    (hmoment : (∫ ω, (Z ω) ^ 2 ∂μ) ≤ error) :
    μ.real {ω | tau ^ 2 ≤ (Z ω) ^ 2} ≤ error / tau ^ 2 := by
  have hmarkov := mul_meas_ge_le_integral_of_nonneg
    (Filter.Eventually.of_forall (fun ω => sq_nonneg (Z ω))) hintegrable (tau ^ 2)
  apply (le_div_iff₀ (sq_pos_of_pos htau)).2
  nlinarith

/-- A finite-sample reduction from a mean-square CDF approximation at one
deterministic threshold. The approximation bound `hmse` is an explicit premise,
not a proved property of a bootstrap method. The statistic and critical value
may be dependent. The separate `hmc` premise accounts for bootstrap Monte Carlo
calibration failure. -/
theorem coverage_of_cdf_mean_square (μ : Measure Ω) [IsProbabilityMeasure μ]
    (T q : Ω → ℝ) (hT : Measurable T) (hq : Measurable q)
    (H : Ω → ℝ → ℝ) (t p tau error eta : ℝ)
    (htau : 0 < tau)
    (htruth : μ.real {ω | T ω ≤ t} = p - tau)
    (hmonotone : ∀ ω, Monotone (H ω))
    (hintegrable : Integrable (fun ω => (H ω t - (p - tau)) ^ 2) μ)
    (hmse : (∫ ω, (H ω t - (p - tau)) ^ 2 ∂μ) ≤ error)
    (hmc : μ.real {ω | H ω (q ω) < p} ≤ eta) :
    p - tau - eta - error / tau ^ 2 ≤ μ.real {ω | T ω ≤ q ω} := by
  have htail := squared_moment_tail μ (fun ω => H ω t - (p - tau))
    tau error htau hintegrable hmse
  have hsubset : {ω | q ω < t} ⊆
      {ω | H ω (q ω) < p} ∪ {ω | tau ^ 2 ≤ (H ω t - (p - tau)) ^ 2} := by
    intro ω hbelow
    by_cases hfailure : H ω (q ω) < p
    · exact Or.inl hfailure
    · apply Or.inr
      have hcalibration : p ≤ H ω (q ω) := le_of_not_gt hfailure
      have horder : H ω (q ω) ≤ H ω t := hmonotone ω (le_of_lt hbelow)
      have hgap : tau ≤ H ω t - (p - tau) := by linarith
      exact (sq_le_sq₀ (le_of_lt htau) (le_trans (le_of_lt htau) hgap)).2 hgap
  have hcutoff : μ.real {ω | q ω < t} ≤ eta + error / tau ^ 2 := by
    calc
      μ.real {ω | q ω < t} ≤
          μ.real ({ω | H ω (q ω) < p} ∪
            {ω | tau ^ 2 ≤ (H ω t - (p - tau)) ^ 2}) := measureReal_mono hsubset
      _ ≤ μ.real {ω | H ω (q ω) < p} +
          μ.real {ω | tau ^ 2 ≤ (H ω t - (p - tau)) ^ 2} := measureReal_union_le _ _
      _ ≤ eta + error / tau ^ 2 := add_le_add hmc htail
  have hbase : 1 - (1 - (p - tau)) ≤ μ.real {ω | T ω ≤ t} := by
    rw [htruth]
    linarith
  have hcoverage := random_threshold_coverage μ T q hT hq t
    (1 - (p - tau)) (eta + error / tau ^ 2) hbase hcutoff
  linarith

end BootstrapCalibration
