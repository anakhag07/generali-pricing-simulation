import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.MeasureTheory.Integral.Pi
import Mathlib.Probability.Distributions.Gaussian
import Mathlib.Tactic

/-!
# Density and Gaussian comparison

Analytic comparison lemmas used in the bootstrap covariance argument. Every
result in this file is proved; a covariance-to-Gaussian-law identification is
not silently introduced as a hypothesis asserting the desired TV inequality.
-/

noncomputable section

open MeasureTheory
open scoped BigOperators NNReal

namespace BootstrapCalibration

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

theorem l2_inner_toLp (f g : Ω → ℝ) (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) :
    inner (𝕜 := ℝ) (hf.toLp f) (hg.toLp g) = ∫ x, f x * g x ∂μ := by
  rw [L2.inner_def]
  apply integral_congr_ae
  filter_upwards [hf.coeFn_toLp, hg.coeFn_toLp] with x hx hy
  simp [hx, hy, mul_comm]

/-- Cauchy--Schwarz for integrable products of real square-integrable functions. -/
theorem integral_cauchy_schwarz_sq (f g : Ω → ℝ)
    (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) :
    (∫ x, f x * g x ∂μ) ^ 2 ≤
      (∫ x, (f x) ^ 2 ∂μ) * (∫ x, (g x) ^ 2 ∂μ) := by
  have h := real_inner_mul_inner_self_le (hf.toLp f) (hg.toLp g)
  rw [l2_inner_toLp f g hf hg, l2_inner_toLp f f hf hf,
    l2_inner_toLp g g hg hg] at h
  simpa only [pow_two] using h

/-- The squared Hellinger-affinity bound for the half-L1 distance between
normalized densities `f²` and `g²`, represented by their nonnegative roots. -/
theorem density_half_l1_sq_le (f g : Ω → ℝ)
    (hf : MemLp f 2 μ) (hg : MemLp g 2 μ)
    (hf0 : ∀ x, 0 ≤ f x) (hg0 : ∀ x, 0 ≤ g x)
    (hf1 : (∫ x, (f x) ^ 2 ∂μ) = 1)
    (hg1 : (∫ x, (g x) ^ 2 ∂μ) = 1) :
    ((∫ x, |(f x) ^ 2 - (g x) ^ 2| ∂μ) / 2) ^ 2 ≤
      1 - (∫ x, f x * g x ∂μ) ^ 2 := by
  have hfg : Integrable (fun x => f x * g x) μ := hf.integrable_mul hg
  have hsum : Integrable (fun x => (f x) ^ 2 + (g x) ^ 2) μ :=
    hf.integrable_sq.add hg.integrable_sq
  have hminus : MemLp (fun x => |f x - g x|) 2 μ := by
    simpa only [Real.norm_eq_abs] using (hf.sub hg).norm
  have hplus : MemLp (fun x => f x + g x) 2 μ := hf.add hg
  have hfactor : ∀ x, |f x - g x| * (f x + g x) = |(f x) ^ 2 - (g x) ^ 2| := by
    intro x
    calc
      |f x - g x| * (f x + g x) = |(f x - g x) * (f x + g x)| := by
        rw [abs_mul, abs_of_nonneg (add_nonneg (hf0 x) (hg0 x))]
      _ = |(f x) ^ 2 - (g x) ^ 2| := by congr 1; ring
  have hmi : (∫ x, |f x - g x| ^ 2 ∂μ) = 2 - 2 * (∫ x, f x * g x ∂μ) := by
    calc
      _ = ∫ x, ((f x) ^ 2 + (g x) ^ 2) - 2 * (f x * g x) ∂μ := by
        apply integral_congr_ae
        exact Filter.Eventually.of_forall (fun x => by dsimp; rw [sq_abs]; ring)
      _ = _ := by
        rw [integral_sub hsum (hfg.const_mul 2),
          integral_add hf.integrable_sq hg.integrable_sq, integral_const_mul, hf1, hg1]
        ring
  have hpl : (∫ x, (f x + g x) ^ 2 ∂μ) = 2 + 2 * (∫ x, f x * g x ∂μ) := by
    calc
      _ = ∫ x, ((f x) ^ 2 + (g x) ^ 2) + 2 * (f x * g x) ∂μ := by
        apply integral_congr_ae
        exact Filter.Eventually.of_forall (fun x => by ring)
      _ = _ := by
        rw [integral_add hsum (hfg.const_mul 2),
          integral_add hf.integrable_sq hg.integrable_sq, integral_const_mul, hf1, hg1]
        ring
  have hcs := integral_cauchy_schwarz_sq (fun x => |f x - g x|)
    (fun x => f x + g x) hminus hplus
  simp_rw [hfactor] at hcs
  rw [hmi, hpl] at hcs
  nlinarith

/-- For normalized densities, every measurable-event discrepancy is at most
the half-L1 distance. -/
theorem setIntegral_sub_le_half_l1 (f g : Ω → ℝ)
    (hf : Integrable f μ) (hg : Integrable g μ)
    (hf1 : (∫ x, f x ∂μ) = 1) (hg1 : (∫ x, g x ∂μ) = 1)
    (s : Set Ω) (hs : MeasurableSet s) :
    |(∫ x in s, f x ∂μ) - (∫ x in s, g x ∂μ)| ≤
      (∫ x, |f x - g x| ∂μ) / 2 := by
  have hdiff : Integrable (fun x => f x - g x) μ := hf.sub hg
  have hzero : (∫ x, f x - g x ∂μ) = 0 := by
    rw [integral_sub hf hg, hf1, hg1, sub_self]
  have hsplit := integral_add_compl hs hdiff
  rw [hzero] at hsplit
  have hcomplement : (∫ x in sᶜ, f x - g x ∂μ) = -(∫ x in s, f x - g x ∂μ) := by
    linarith
  have hfirst := abs_integral_le_integral_abs (μ := μ.restrict s)
    (f := fun x => f x - g x)
  have hsecond := abs_integral_le_integral_abs (μ := μ.restrict sᶜ)
    (f := fun x => f x - g x)
  rw [hcomplement, abs_neg] at hsecond
  have habssplit := integral_add_compl hs hdiff.abs
  rw [← integral_sub hf.restrict hg.restrict]
  linarith

/-- Event form of the affinity bound, avoiding any assumption of the desired
Gaussian comparison. The two displayed set integrals are the event masses. -/
theorem density_event_sq_le (f g : Ω → ℝ)
    (hf : MemLp f 2 μ) (hg : MemLp g 2 μ)
    (hf0 : ∀ x, 0 ≤ f x) (hg0 : ∀ x, 0 ≤ g x)
    (hf1 : (∫ x, (f x) ^ 2 ∂μ) = 1)
    (hg1 : (∫ x, (g x) ^ 2 ∂μ) = 1)
    (s : Set Ω) (hs : MeasurableSet s) :
    ((∫ x in s, (f x) ^ 2 ∂μ) - (∫ x in s, (g x) ^ 2 ∂μ)) ^ 2 ≤
      1 - (∫ x, f x * g x ∂μ) ^ 2 := by
  have hevent := setIntegral_sub_le_half_l1 (fun x => (f x) ^ 2)
    (fun x => (g x) ^ 2) hf.integrable_sq hg.integrable_sq hf1 hg1 s hs
  have hl1 := density_half_l1_sq_le f g hf hg hf0 hg0 hf1 hg1
  have hnonneg : 0 ≤ (∫ x, |(f x) ^ 2 - (g x) ^ 2| ∂μ) / 2 := by
    positivity
  have hsquare := (sq_le_sq₀ (abs_nonneg _) hnonneg).2 hevent
  rw [sq_abs] at hsquare
  exact hsquare.trans hl1

/-- An integrable normalized-density difference tested against a measurable
function of range length at most one is controlled by half its L1 norm. -/
theorem bounded_test_sub_le_half_l1 (f g h : Ω → ℝ)
    (hf : Integrable f μ) (hg : Integrable g μ) (hh : Measurable h)
    (hf1 : (∫ x, f x ∂μ) = 1) (hg1 : (∫ x, g x ∂μ) = 1)
    (a : ℝ) (hlo : ∀ x, a ≤ h x) (hhi : ∀ x, h x ≤ a + 1) :
    |∫ x, h x * (f x - g x) ∂μ| ≤ (∫ x, |f x - g x| ∂μ) / 2 := by
  have hdiff : Integrable (fun x => f x - g x) μ := hf.sub hg
  have hcenter : ∀ x, |h x - (a + 1 / 2)| ≤ 1 / 2 := by
    intro x
    exact abs_le.mpr ⟨by linarith [hlo x], by linarith [hhi x]⟩
  have hcentermeas : AEStronglyMeasurable (fun x => h x - (a + 1 / 2)) μ :=
    (hh.sub measurable_const).aestronglyMeasurable
  have hcenterint : Integrable (fun x => (h x - (a + 1 / 2)) * (f x - g x)) μ :=
    hdiff.bdd_mul hcentermeas ⟨1 / 2, fun x => by simpa using hcenter x⟩
  have hzero : (∫ x, f x - g x ∂μ) = 0 := by
    rw [integral_sub hf hg, hf1, hg1, sub_self]
  have hshift : (∫ x, h x * (f x - g x) ∂μ) =
      ∫ x, (h x - (a + 1 / 2)) * (f x - g x) ∂μ := by
    calc
      _ = ∫ x, (h x - (a + 1 / 2)) * (f x - g x) +
          (a + 1 / 2) * (f x - g x) ∂μ := by
        apply integral_congr_ae
        exact Filter.Eventually.of_forall (fun x => by ring)
      _ = _ := by
        rw [integral_add hcenterint (hdiff.const_mul _), integral_const_mul, hzero]
        ring
  rw [hshift]
  calc
    _ ≤ ∫ x, |(h x - (a + 1 / 2)) * (f x - g x)| ∂μ := abs_integral_le_integral_abs
    _ ≤ ∫ x, (1 / 2) * |f x - g x| ∂μ := by
      apply integral_mono hcenterint.abs (hdiff.abs.const_mul _)
      intro x
      dsimp
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right (hcenter x) (abs_nonneg _)
    _ = _ := by rw [integral_const_mul]; ring

/-- Affinity controls arbitrary measurable tests whose range has length one. -/
theorem density_bounded_test_sq_le (f g h : Ω → ℝ)
    (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) (hh : Measurable h)
    (hf0 : ∀ x, 0 ≤ f x) (hg0 : ∀ x, 0 ≤ g x)
    (hf1 : (∫ x, (f x) ^ 2 ∂μ) = 1)
    (hg1 : (∫ x, (g x) ^ 2 ∂μ) = 1)
    (a : ℝ) (hlo : ∀ x, a ≤ h x) (hhi : ∀ x, h x ≤ a + 1) :
    (∫ x, h x * ((f x) ^ 2 - (g x) ^ 2) ∂μ) ^ 2 ≤
      1 - (∫ x, f x * g x ∂μ) ^ 2 := by
  have htest := bounded_test_sub_le_half_l1 (fun x => (f x) ^ 2)
    (fun x => (g x) ^ 2) h hf.integrable_sq hg.integrable_sq hh hf1 hg1 a hlo hhi
  have hl1 := density_half_l1_sq_le f g hf hg hf0 hg0 hf1 hg1
  have hnonneg : 0 ≤ (∫ x, |(f x) ^ 2 - (g x) ^ 2| ∂μ) / 2 := by positivity
  have hsquare := (sq_le_sq₀ (abs_nonneg _) hnonneg).2 htest
  rw [sq_abs] at hsquare
  exact hsquare.trans hl1

/-- Product deficit is controlled by the sum of factor deficits. -/
theorem one_sub_prod_le_sum {ι : Type*} (s : Finset ι) (a : ι → ℝ)
    (ha0 : ∀ i ∈ s, 0 ≤ a i) (ha1 : ∀ i ∈ s, a i ≤ 1) :
    1 - ∏ i ∈ s, a i ≤ ∑ i ∈ s, (1 - a i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | @insert i s hi ih =>
    simp only [Finset.prod_insert hi, Finset.sum_insert hi]
    have h0 := ha0 i (Finset.mem_insert_self _ _)
    have h1 := ha1 i (Finset.mem_insert_self _ _)
    have ht0 : ∀ j ∈ s, 0 ≤ a j := fun j hj => ha0 j (Finset.mem_insert_of_mem hj)
    have ht1 : ∀ j ∈ s, a j ≤ 1 := fun j hj => ha1 j (Finset.mem_insert_of_mem hj)
    have hprod : (∏ j ∈ s, a j) ≤ 1 := Finset.prod_le_one ht0 ht1
    have hind := ih ht0 ht1
    nlinarith [mul_nonneg (sub_nonneg.mpr h1) (sub_nonneg.mpr hprod)]

/-- The scalar Gaussian squared-affinity factor is in `[0,1]`. -/
theorem gaussian_affinity_factor_bounds (v : ℝ) (hv : 0 ≤ v) :
    0 ≤ 2 * Real.sqrt v / (1 + v) ∧ 2 * Real.sqrt v / (1 + v) ≤ 1 := by
  have hden : 0 < 1 + v := by linarith
  constructor
  · positivity
  · apply (div_le_one hden).2
    nlinarith [sq_nonneg (Real.sqrt v - 1), Real.sq_sqrt hv]

/-- A scalar eigenvalue controls its squared-affinity deficit. -/
theorem gaussian_affinity_factor_deficit (v : ℝ) (hv : 0 ≤ v) :
    1 - 2 * Real.sqrt v / (1 + v) ≤ (v - 1) ^ 2 := by
  have hden : 0 < 1 + v := by linarith
  have hroot : 0 ≤ Real.sqrt v := Real.sqrt_nonneg v
  have hsquare : (Real.sqrt v) ^ 2 = v := Real.sq_sqrt hv
  have hfac : 1 ≤ (1 + v) * (1 + Real.sqrt v) ^ 2 := by
    nlinarith [mul_nonneg hv (sq_nonneg (1 + Real.sqrt v))]
  have hidentity :
      (1 - 2 * Real.sqrt v / (1 + v)) * ((1 + v) * (1 + Real.sqrt v) ^ 2) =
        (v - 1) ^ 2 := by
    field_simp
    linear_combination -(1 + v) * (3 + 2 * Real.sqrt v - v) * hsquare
  have hdeficit : 0 ≤ 1 - 2 * Real.sqrt v / (1 + v) :=
    sub_nonneg.mpr (gaussian_affinity_factor_bounds v hv).2
  nlinarith [mul_nonneg hdeficit (sub_nonneg.mpr hfac)]

/-- Finite eigenvalue-product algebra in the Gaussian covariance comparison. -/
theorem gaussian_affinity_product_deficit {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    (hv : ∀ i ∈ s, 0 ≤ v i) :
    1 - ∏ i ∈ s, (2 * Real.sqrt (v i) / (1 + v i)) ≤ ∑ i ∈ s, (v i - 1) ^ 2 := by
  calc
    _ ≤ ∑ i ∈ s, (1 - 2 * Real.sqrt (v i) / (1 + v i)) :=
      one_sub_prod_le_sum s _ (fun i hi => (gaussian_affinity_factor_bounds _ (hv i hi)).1)
        (fun i hi => (gaussian_affinity_factor_bounds _ (hv i hi)).2)
    _ ≤ _ := Finset.sum_le_sum (fun i hi => gaussian_affinity_factor_deficit _ (hv i hi))

/-- Nonnegative square root of the centered Gaussian density proportional to
`exp (-a*x²)`. Its variance is `1/(2*a)` when `a>0`. -/
def gaussianRootPrecision (a x : ℝ) : ℝ :=
  Real.sqrt (Real.sqrt (a / Real.pi)) * Real.exp (-(a / 2) * x ^ 2)

theorem gaussianRootPrecision_nonneg (a x : ℝ) : 0 ≤ gaussianRootPrecision a x := by
  unfold gaussianRootPrecision
  positivity

theorem gaussianRootPrecision_sq (a x : ℝ) :
    gaussianRootPrecision a x ^ 2 =
      Real.sqrt (a / Real.pi) * Real.exp (-a * x ^ 2) := by
  unfold gaussianRootPrecision
  rw [mul_pow, Real.sq_sqrt (Real.sqrt_nonneg _), pow_two, ← Real.exp_add]
  congr 1
  congr 1
  ring

theorem gaussianRootPrecision_memLp (a : ℝ) (ha : 0 < a) :
    MemLp (gaussianRootPrecision a) 2 (volume : Measure ℝ) := by
  have hcontinuous : Continuous (gaussianRootPrecision a) := by
    unfold gaussianRootPrecision
    fun_prop
  apply (memLp_two_iff_integrable_sq hcontinuous.aestronglyMeasurable).2
  simp_rw [gaussianRootPrecision_sq]
  exact (integrable_exp_neg_mul_sq ha).const_mul _

/-- Normalization is obtained by evaluating the Gaussian integral. -/
theorem gaussianRootPrecision_normalized (a : ℝ) (ha : 0 < a) :
    (∫ x : ℝ, gaussianRootPrecision a x ^ 2) = 1 := by
  simp_rw [gaussianRootPrecision_sq]
  rw [integral_const_mul, integral_gaussian,
    ← Real.sqrt_mul (div_nonneg ha.le Real.pi_pos.le)]
  have hcancel : a / Real.pi * (Real.pi / a) = 1 := by
    field_simp
  rw [hcancel, Real.sqrt_one]

/-- Direct one-dimensional Gaussian affinity integral, with no assumed
normal-density integration formula beyond mathlib's evaluated Gaussian integral. -/
theorem gaussianRootPrecision_affinity_sq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    (∫ x : ℝ, gaussianRootPrecision a x * gaussianRootPrecision b x) ^ 2 =
      2 * Real.sqrt (a * b) / (a + b) := by
  have hab : 0 < a + b := add_pos ha hb
  have hprod : ∀ x, gaussianRootPrecision a x * gaussianRootPrecision b x =
      (Real.sqrt (Real.sqrt (a / Real.pi)) * Real.sqrt (Real.sqrt (b / Real.pi))) *
        Real.exp (-((a + b) / 2) * x ^ 2) := by
    intro x
    unfold gaussianRootPrecision
    rw [mul_mul_mul_comm, ← Real.exp_add]
    congr 1
    congr 1
    ring
  simp_rw [hprod]
  rw [integral_const_mul, integral_gaussian, mul_pow, mul_pow,
    Real.sq_sqrt (Real.sqrt_nonneg _), Real.sq_sqrt (Real.sqrt_nonneg _),
    Real.sq_sqrt (div_nonneg Real.pi_pos.le (by linarith : 0 ≤ (a + b) / 2))]
  have hnorm : Real.sqrt (a / Real.pi) * Real.sqrt (b / Real.pi) * Real.pi =
      Real.sqrt (a * b) := by
    rw [Real.sqrt_div ha.le, Real.sqrt_div hb.le, Real.sqrt_mul ha.le]
    field_simp
  calc
    _ = 2 * (Real.sqrt (a / Real.pi) * Real.sqrt (b / Real.pi) * Real.pi) / (a + b) := by
      field_simp
      ring
    _ = _ := by rw [hnorm]

/-- Gaussian density root parameterized by variance. -/
def gaussianRootVariance (v x : ℝ) : ℝ := gaussianRootPrecision (2 * v)⁻¹ x

/-- The explicit density root agrees with mathlib's centered normal density. -/
theorem gaussianRootVariance_sq (v : ℝ≥0) (x : ℝ) :
    gaussianRootVariance v x ^ 2 = ProbabilityTheory.gaussianPDFReal 0 v x := by
  rw [gaussianRootVariance, gaussianRootPrecision_sq,
    ProbabilityTheory.gaussianPDFReal]
  simp only [sub_zero]
  congr 1
  · rw [← Real.sqrt_inv]
    congr 1
    simp only [div_eq_mul_inv, mul_inv_rev]
    ring
  · congr 1
    ring

/-- The one-dimensional standard-normal comparison factor, proved from the
evaluated Gaussian integral. -/
theorem gaussianRootVariance_affinity_sq (v : ℝ) (hv : 0 < v) :
    (∫ x : ℝ, gaussianRootVariance v x * gaussianRootVariance 1 x) ^ 2 =
      2 * Real.sqrt v / (1 + v) := by
  unfold gaussianRootVariance
  rw [gaussianRootPrecision_affinity_sq _ _ (by positivity) (by positivity)]
  have hprod : (2 * v)⁻¹ * (2 * (1 : ℝ))⁻¹ = (4 * v)⁻¹ := by ring
  rw [hprod, Real.sqrt_inv, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 4)]
  norm_num
  field_simp
  nlinarith [Real.sq_sqrt hv.le]

/-- Product of nonnegative Gaussian density roots on a finite coordinate space. -/
def diagonalGaussianRoot {ι : Type*} [Fintype ι] (v : ι → ℝ) (x : ι → ℝ) : ℝ :=
  ∏ i, gaussianRootVariance (v i) (x i)

theorem diagonalGaussianRoot_nonneg {ι : Type*} [Fintype ι] (v : ι → ℝ) (x : ι → ℝ) :
    0 ≤ diagonalGaussianRoot v x :=
  Finset.prod_nonneg (fun _ _ => gaussianRootPrecision_nonneg _ _)

theorem diagonalGaussianRoot_memLp {ι : Type*} [Fintype ι]
    (v : ι → ℝ) (hv : ∀ i, 0 < v i) :
    MemLp (diagonalGaussianRoot v) 2 (volume : Measure (ι → ℝ)) := by
  have hcontinuous : Continuous (diagonalGaussianRoot v) := by
    unfold diagonalGaussianRoot gaussianRootVariance gaussianRootPrecision
    fun_prop
  apply (memLp_two_iff_integrable_sq hcontinuous.aestronglyMeasurable).2
  simp only [diagonalGaussianRoot, ← Finset.prod_pow]
  exact Integrable.fintype_prod (fun i =>
    (gaussianRootPrecision_memLp (2 * v i)⁻¹
      (inv_pos.mpr (mul_pos (by norm_num) (hv i)))).integrable_sq)

theorem diagonalGaussianRoot_normalized {ι : Type*} [Fintype ι]
    (v : ι → ℝ) (hv : ∀ i, 0 < v i) :
    (∫ x : ι → ℝ, diagonalGaussianRoot v x ^ 2) = 1 := by
  simp only [diagonalGaussianRoot, ← Finset.prod_pow]
  rw [integral_fintype_prod_eq_prod ι (fun i x => gaussianRootVariance (v i) x ^ 2)]
  apply Finset.prod_eq_one
  intro i hi
  exact gaussianRootPrecision_normalized _ (inv_pos.mpr (mul_pos (by norm_num) (hv i)))

/-- Tensorization computes the Gaussian affinity, rather than postulating it. -/
theorem diagonalGaussianRoot_affinity_sq {ι : Type*} [Fintype ι]
    (v : ι → ℝ) (hv : ∀ i, 0 < v i) :
    (∫ x : ι → ℝ, diagonalGaussianRoot v x * diagonalGaussianRoot (fun _ => 1) x) ^ 2 =
      ∏ i, (2 * Real.sqrt (v i) / (1 + v i)) := by
  simp only [diagonalGaussianRoot, ← Finset.prod_mul_distrib]
  rw [integral_fintype_prod_eq_prod ι
    (fun i x => gaussianRootVariance (v i) x * gaussianRootVariance 1 x), ← Finset.prod_pow]
  apply Finset.prod_congr rfl
  intro i hi
  exact gaussianRootVariance_affinity_sq (v i) (hv i)

/-- Measurable-event comparison for a positive diagonal Gaussian covariance.
Both integrals are actual normalized Gaussian density event probabilities. -/
theorem diagonal_gaussian_event_sq_le {ι : Type*} [Fintype ι]
    (v : ι → ℝ) (hv : ∀ i, 0 < v i)
    (s : Set (ι → ℝ)) (hs : MeasurableSet s) :
    ((∫ x in s, diagonalGaussianRoot v x ^ 2) -
      (∫ x in s, diagonalGaussianRoot (fun _ => 1) x ^ 2)) ^ 2 ≤
      ∑ i, (v i - 1) ^ 2 := by
  have h := density_event_sq_le (diagonalGaussianRoot v)
    (diagonalGaussianRoot (fun _ => 1))
    (diagonalGaussianRoot_memLp v hv) (diagonalGaussianRoot_memLp _ (by simp))
    (diagonalGaussianRoot_nonneg v) (diagonalGaussianRoot_nonneg _)
    (diagonalGaussianRoot_normalized v hv) (diagonalGaussianRoot_normalized _ (by simp)) s hs
  rw [diagonalGaussianRoot_affinity_sq v hv] at h
  exact h.trans (gaussian_affinity_product_deficit Finset.univ v (fun i _ => (hv i).le))

/-- Positive-diagonal Gaussian comparison for a measurable test of range
length at most one, including the centered two-indicator test. -/
theorem diagonal_gaussian_bounded_test_sq_le {ι : Type*} [Fintype ι]
    (v : ι → ℝ) (hv : ∀ i, 0 < v i) (h : (ι → ℝ) → ℝ) (hh : Measurable h)
    (a : ℝ) (hlo : ∀ x, a ≤ h x) (hhi : ∀ x, h x ≤ a + 1) :
    (∫ x : ι → ℝ, h x * (diagonalGaussianRoot v x ^ 2 -
      diagonalGaussianRoot (fun _ => 1) x ^ 2)) ^ 2 ≤ ∑ i, (v i - 1) ^ 2 := by
  have ht := density_bounded_test_sq_le (diagonalGaussianRoot v)
    (diagonalGaussianRoot (fun _ => 1)) h
    (diagonalGaussianRoot_memLp v hv) (diagonalGaussianRoot_memLp _ (by simp)) hh
    (diagonalGaussianRoot_nonneg v) (diagonalGaussianRoot_nonneg _)
    (diagonalGaussianRoot_normalized v hv) (diagonalGaussianRoot_normalized _ (by simp))
    a hlo hhi
  rw [diagonalGaussianRoot_affinity_sq v hv] at ht
  exact ht.trans (gaussian_affinity_product_deficit Finset.univ v (fun i _ => (hv i).le))

/-- Unsquared expectation form of the positive-diagonal comparison. -/
theorem diagonal_gaussian_bounded_test_le {ι : Type*} [Fintype ι]
    (v : ι → ℝ) (hv : ∀ i, 0 < v i) (h : (ι → ℝ) → ℝ) (hh : Measurable h)
    (a : ℝ) (hlo : ∀ x, a ≤ h x) (hhi : ∀ x, h x ≤ a + 1) :
    |(∫ x : ι → ℝ, h x * diagonalGaussianRoot v x ^ 2) -
      (∫ x : ι → ℝ, h x * diagonalGaussianRoot (fun _ => 1) x ^ 2)| ≤
      Real.sqrt (∑ i, (v i - 1) ^ 2) := by
  have hbound : ∃ C, ∀ x, ‖h x‖ ≤ C := by
    refine ⟨|a| + 1, fun x => ?_⟩
    rw [Real.norm_eq_abs]
    exact abs_le.mpr ⟨by linarith [neg_abs_le a, hlo x], by linarith [le_abs_self a, hhi x]⟩
  have hfirst := (diagonalGaussianRoot_memLp v hv).integrable_sq.bdd_mul
    hh.aestronglyMeasurable hbound
  have hsecond := (diagonalGaussianRoot_memLp (fun _ : ι => (1 : ℝ))
    (by simp)).integrable_sq.bdd_mul hh.aestronglyMeasurable hbound
  have ht := diagonal_gaussian_bounded_test_sq_le v hv h hh a hlo hhi
  simp_rw [mul_sub] at ht
  rw [integral_sub hfirst hsecond] at ht
  exact Real.abs_le_sqrt ht

end BootstrapCalibration
