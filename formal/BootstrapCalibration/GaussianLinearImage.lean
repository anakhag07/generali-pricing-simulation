import BootstrapCalibration.GaussianComparison
import BootstrapCalibration.SpectralComparison
import Mathlib.MeasureTheory.Measure.CharacteristicFunction
import Mathlib.LinearAlgebra.Matrix.BilinearForm
import Mathlib.LinearAlgebra.Matrix.PosDef

/-!
# Gaussian linear images

The vector Gaussian law is defined by an explicitly normalized density. Its
characteristic function is computed from mathlib's scalar Gaussian law and
finite-product integration; law identification uses uniqueness of characteristic
functions, not an assumed multivariate-normal distribution formula.
-/

noncomputable section
open MeasureTheory Matrix
open scoped BigOperators NNReal ENNReal

namespace BootstrapCalibration

variable {ι : Type*} [Fintype ι]

theorem continuous_diagonalGaussianRoot (v : ι → ℝ) : Continuous (diagonalGaussianRoot v) := by
  unfold diagonalGaussianRoot gaussianRootVariance gaussianRootPrecision
  fun_prop

/-- The positive diagonal Gaussian distribution, defined by its product density. -/
def diagonalGaussianMeasure (v : ι → ℝ) : Measure (ι → ℝ) :=
  volume.withDensity (fun x => ENNReal.ofReal (diagonalGaussianRoot v x ^ 2))

theorem diagonalGaussianMeasure_probability (v : ι → ℝ) (hv : ∀ i, 0 < v i) :
    IsProbabilityMeasure (diagonalGaussianMeasure v) := by
  constructor
  rw [diagonalGaussianMeasure, withDensity_apply _ MeasurableSet.univ,
    Measure.restrict_univ,
    ← ofReal_integral_eq_lintegral_ofReal (diagonalGaussianRoot_memLp v hv).integrable_sq
      (Filter.Eventually.of_forall (fun x => sq_nonneg _)),
    diagonalGaussianRoot_normalized v hv, ENNReal.ofReal_one]

theorem integral_diagonalGaussianMeasure {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (v : ι → ℝ) (f : (ι → ℝ) → E) :
    (∫ x, f x ∂diagonalGaussianMeasure v) =
      ∫ x, diagonalGaussianRoot v x ^ 2 • f x := by
  rw [diagonalGaussianMeasure,
    integral_withDensity_eq_integral_toReal_smul
      (((continuous_diagonalGaussianRoot v).pow 2).measurable.ennreal_ofReal)
      (Filter.Eventually.of_forall (fun _ => ENNReal.ofReal_lt_top))]
  simp only [ENNReal.toReal_ofReal (sq_nonneg _)]

def standardGaussianMeasure (ι : Type*) [Fintype ι] : Measure (ι → ℝ) :=
  diagonalGaussianMeasure (fun _ => 1)

instance standardGaussianMeasure_probability : IsProbabilityMeasure (standardGaussianMeasure ι) :=
  diagonalGaussianMeasure_probability _ (by simp)

/-- The characteristic function written in ordinary finite-coordinate notation. -/
def dotCharFun (μ : Measure (ι → ℝ)) (t : ι → ℝ) : ℂ :=
  ∫ x, Complex.exp ((dotProduct x t : ℝ) * Complex.I) ∂μ

theorem scalar_gaussian_density_character (v : ℝ) (hv : 0 < v) (t : ℝ) :
    (∫ x : ℝ, (gaussianRootVariance v x ^ 2 : ℂ) *
      Complex.exp ((x * t : ℝ) * Complex.I)) = Complex.exp (-(v : ℂ) * t ^ 2 / 2) := by
  have hvar : (⟨v, hv.le⟩ : ℝ≥0) ≠ 0 := by
    intro heq
    exact (ne_of_gt hv) (congrArg (fun w : ℝ≥0 => (w : ℝ)) heq)
  have h := ProbabilityTheory.charFun_gaussianReal (μ := 0) (v := ⟨v, hv.le⟩) t
  rw [charFun_apply_real,
    ProbabilityTheory.integral_gaussianReal_eq_integral_smul hvar] at h
  simpa [Complex.real_smul, ← gaussianRootVariance_sq, mul_comm, mul_left_comm, mul_assoc,
    neg_div] using h

theorem dotCharFun_diagonalGaussian (v : ι → ℝ) (hv : ∀ i, 0 < v i) (t : ι → ℝ) :
    dotCharFun (diagonalGaussianMeasure v) t =
      Complex.exp (-((∑ i, v i * (t i) ^ 2 : ℝ) : ℂ) / 2) := by
  rw [dotCharFun, integral_diagonalGaussianMeasure]
  have hfactor : ∀ x : ι → ℝ,
      diagonalGaussianRoot v x ^ 2 • Complex.exp ((dotProduct x t : ℝ) * Complex.I) =
      ∏ i, (gaussianRootVariance (v i) (x i) ^ 2 : ℂ) *
        Complex.exp ((x i * t i : ℝ) * Complex.I) := by
    intro x
    simp only [Complex.real_smul, diagonalGaussianRoot, ← Finset.prod_pow, dotProduct,
      Complex.ofReal_prod, Complex.ofReal_sum, Complex.ofReal_pow, Finset.sum_mul, Complex.exp_sum,
      Finset.prod_mul_distrib]
  simp_rw [hfactor]
  rw [integral_fintype_prod_eq_prod ι
    (fun i x => (gaussianRootVariance (v i) x ^ 2 : ℂ) * Complex.exp ((x * t i : ℝ) * Complex.I))]
  simp_rw [scalar_gaussian_density_character _ (hv _) _, ← Complex.exp_sum]
  congr 1
  simp only [Complex.ofReal_sum, Complex.ofReal_mul, Complex.ofReal_pow,
    Finset.sum_div, Finset.sum_neg_distrib, neg_mul, neg_div]

/-- Characteristic functions in finite-coordinate notation determine finite laws. -/
theorem measure_eq_of_dotCharFun_eq (μ ν : Measure (ι → ℝ))
    [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (h : ∀ t, dotCharFun μ t = dotCharFun ν t) : μ = ν := by
  classical
  let L := Matrix.toBilin' (1 : Matrix ι ι ℝ)
  have hL : Continuous (fun p : (ι → ℝ) × (ι → ℝ) => L p.1 p.2) := by
    simp only [L, Matrix.toBilin'_apply', one_mulVec, dotProduct]
    fun_prop
  have hsep : ∀ x : ι → ℝ, x ≠ 0 → L x ≠ 0 := by
    intro x hx hzero
    apply hx
    apply dotProduct_eq
    intro u
    have hu := congrArg (fun z : (ι → ℝ) →ₗ[ℝ] ℝ => z u) hzero
    simpa only [L, Matrix.toBilin'_apply', one_mulVec, LinearMap.zero_apply, zero_dotProduct] using hu
  apply ext_of_integral_char_eq Real.continuous_probChar Real.probChar_ne_one hsep hL
  intro t
  simpa only [BoundedContinuousFunction.char_apply, Real.probChar_apply,
    L, Matrix.toBilin'_apply', one_mulVec, dotCharFun] using h t

theorem measurable_matrix_mulVec {κ : Type*} [Fintype κ] (A : Matrix κ ι ℝ) :
    Measurable (fun x : ι → ℝ => A *ᵥ x) := by
  unfold Matrix.mulVec dotProduct
  fun_prop

/-- Characteristic function under an arbitrary rectangular linear image. -/
theorem dotCharFun_map_matrix {κ : Type*} [Fintype κ]
    (μ : Measure (ι → ℝ)) (A : Matrix κ ι ℝ) (t : κ → ℝ) :
    dotCharFun (μ.map (fun x => A *ᵥ x)) t = dotCharFun μ (Aᵀ *ᵥ t) := by
  have hcontinuous : Continuous (fun x : κ → ℝ =>
      Complex.exp ((dotProduct x t : ℝ) * Complex.I)) := by
    unfold dotProduct
    fun_prop
  rw [dotCharFun, integral_map (measurable_matrix_mulVec A).aemeasurable
    hcontinuous.aestronglyMeasurable, dotCharFun]
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro x
  have hdot : dotProduct (A *ᵥ x) t = dotProduct x (Aᵀ *ᵥ t) := by
    rw [dotProduct_comm, dotProduct_mulVec, ← mulVec_transpose, dotProduct_comm]
  dsimp
  rw [hdot]

/-- Computed characteristic function of an actual linear image of independent
standard Gaussian coordinates, valid for rectangular matrices. -/
theorem dotCharFun_standard_linearImage {κ : Type*} [Fintype κ]
    (A : Matrix κ ι ℝ) (t : κ → ℝ) :
    dotCharFun ((standardGaussianMeasure ι).map (fun x => A *ᵥ x)) t =
      Complex.exp (-((∑ j, ((Aᵀ *ᵥ t) j) ^ 2 : ℝ) : ℂ) / 2) := by
  rw [dotCharFun_map_matrix, standardGaussianMeasure, dotCharFun_diagonalGaussian _ (by simp)]
  simp only [one_mul]

theorem transpose_mulVec_sq_sum {κ : Type*} [Fintype κ]
    (A : Matrix κ ι ℝ) (t : κ → ℝ) :
    (∑ j, ((Aᵀ *ᵥ t) j) ^ 2) = dotProduct t ((A * Aᵀ) *ᵥ t) := by
  rw [← mulVec_mulVec, dotProduct_mulVec, ← mulVec_transpose]
  simp only [dotProduct, pow_two]

/-- Equality of covariance matrices identifies the actual Gaussian image laws. -/
theorem standard_linearImage_eq_of_gram_eq {κ ν : Type*} [Fintype κ] [Fintype ν]
    (A : Matrix κ ι ℝ) (B : Matrix κ ν ℝ) (hgram : A * Aᵀ = B * Bᵀ) :
    (standardGaussianMeasure ι).map (fun x => A *ᵥ x) =
      (standardGaussianMeasure ν).map (fun x => B *ᵥ x) := by
  apply measure_eq_of_dotCharFun_eq
  intro t
  rw [dotCharFun_standard_linearImage, dotCharFun_standard_linearImage,
    transpose_mulVec_sq_sum, transpose_mulVec_sq_sum, hgram]

/-- Scaling independent standard coordinates yields the diagonal Gaussian law. -/
theorem standard_diagonal_scale_eq_diagonalGaussian
    [DecidableEq ι] (v : ι → ℝ) (hv : ∀ i, 0 < v i) :
    (standardGaussianMeasure ι).map (fun x => (diagonal (fun i => Real.sqrt (v i))) *ᵥ x) =
      diagonalGaussianMeasure v := by
  classical
  letI := diagonalGaussianMeasure_probability v hv
  apply measure_eq_of_dotCharFun_eq
  intro t
  rw [dotCharFun_standard_linearImage, dotCharFun_diagonalGaussian v hv]
  simp only [diagonal_transpose, mulVec_diagonal, mul_pow, Real.sq_sqrt (hv _).le]

/-- Orthogonal invariance follows from the computed characteristic functions. -/
theorem standardGaussian_orthogonal_invariant
    [DecidableEq ι] (U : Matrix ι ι ℝ) (hU : U * Uᵀ = 1) :
    (standardGaussianMeasure ι).map (fun x => U *ᵥ x) = standardGaussianMeasure ι := by
  classical
  have h := standard_linearImage_eq_of_gram_eq U (1 : Matrix ι ι ℝ) (by simpa using hU)
  simpa only [one_mulVec, Measure.map_id'] using h

/-- A covariance factorization identifies the Gaussian image with the indicated
linear image of independent diagonal Gaussian coordinates. -/
theorem linearImage_eq_diagonal_image {κ : Type*} [Fintype κ] [DecidableEq κ]
    (A : Matrix κ ι ℝ) (U : Matrix κ κ ℝ) (v : κ → ℝ) (hv : ∀ i, 0 < v i)
    (hcov : A * Aᵀ = U * diagonal v * Uᵀ) :
    (standardGaussianMeasure ι).map (fun x => A *ᵥ x) =
      (diagonalGaussianMeasure v).map (fun x => U *ᵥ x) := by
  let D : Matrix κ κ ℝ := diagonal (fun i => Real.sqrt (v i))
  have hD : D * D = diagonal v := by
    dsimp only [D]
    rw [Matrix.diagonal_mul_diagonal]
    apply congrArg diagonal
    funext i
    exact Real.mul_self_sqrt (hv i).le
  have hfactor : (U * D) * (U * D)ᵀ = U * diagonal v * Uᵀ := by
    rw [transpose_mul]
    have hsym : Dᵀ = D := diagonal_transpose _
    rw [hsym]
    calc
      U * D * (D * Uᵀ) = U * (D * D) * Uᵀ := by simp only [Matrix.mul_assoc]
      _ = _ := by rw [hD]
  have hlaw := standard_linearImage_eq_of_gram_eq A (U * D) (hcov.trans hfactor.symm)
  rw [← standard_diagonal_scale_eq_diagonalGaussian v hv,
    Measure.map_map (measurable_matrix_mulVec U) (measurable_matrix_mulVec _)]
  simpa only [Function.comp_def, mulVec_mulVec, D] using hlaw

/-- Positive spectral-factor comparison obtained by law identification and
orthogonal invariance, not by assuming a Gaussian TV inequality. -/
theorem linearImage_bounded_test_le_of_factor {κ : Type*} [Fintype κ] [DecidableEq κ]
    (A : Matrix κ ι ℝ) (U : Matrix κ κ ℝ) (v : κ → ℝ) (hv : ∀ i, 0 < v i)
    (hU : U * Uᵀ = 1) (hcov : A * Aᵀ = U * diagonal v * Uᵀ)
    (h : (κ → ℝ) → ℝ) (hh : Measurable h)
    (a : ℝ) (hlo : ∀ x, a ≤ h x) (hhi : ∀ x, h x ≤ a + 1) :
    |(∫ x, h x ∂((standardGaussianMeasure ι).map (fun z => A *ᵥ z))) -
      (∫ x, h x ∂standardGaussianMeasure κ)| ≤ Real.sqrt (∑ i, (v i - 1) ^ 2) := by
  have hfirst : (∫ x, h x ∂((standardGaussianMeasure ι).map (fun z => A *ᵥ z))) =
      ∫ x, h (U *ᵥ x) ∂diagonalGaussianMeasure v := by
    rw [linearImage_eq_diagonal_image A U v hv hcov,
      integral_map (measurable_matrix_mulVec U).aemeasurable hh.aestronglyMeasurable]
  have hsecond : (∫ x, h x ∂standardGaussianMeasure κ) =
      ∫ x, h (U *ᵥ x) ∂standardGaussianMeasure κ := by
    calc
      _ = ∫ x, h x ∂((standardGaussianMeasure κ).map (fun z => U *ᵥ z)) := by
        rw [standardGaussian_orthogonal_invariant U hU]
      _ = _ := integral_map (measurable_matrix_mulVec U).aemeasurable hh.aestronglyMeasurable
  rw [hfirst, hsecond, integral_diagonalGaussianMeasure, standardGaussianMeasure,
    integral_diagonalGaussianMeasure]
  simpa only [smul_eq_mul, mul_comm] using
    diagonal_gaussian_bounded_test_le v hv (fun x => h (U *ᵥ x))
      (hh.comp (measurable_matrix_mulVec U)) a (fun x => hlo _) (fun x => hhi _)

/-- A probability expectation stays in the deterministic range of a bounded test. -/
theorem integral_mem_bounded_interval {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (h : Ω → ℝ) (hh : Measurable h)
    (a : ℝ) (hlo : ∀ x, a ≤ h x) (hhi : ∀ x, h x ≤ a + 1) :
    a ≤ (∫ x, h x ∂μ) ∧ (∫ x, h x ∂μ) ≤ a + 1 := by
  have hbdd : ∃ C, ∀ x, ‖h x‖ ≤ C := by
    refine ⟨|a| + 1, fun x => ?_⟩
    rw [Real.norm_eq_abs]
    exact abs_le.mpr ⟨by linarith [neg_abs_le a, hlo x], by linarith [le_abs_self a, hhi x]⟩
  have hint : Integrable h μ := by
    simpa only [mul_one] using (integrable_const (1 : ℝ)).bdd_mul hh.aestronglyMeasurable hbdd
  constructor
  · simpa using integral_mono (integrable_const a) hint hlo
  · simpa using integral_mono hint (integrable_const (a + 1)) hhi

theorem probability_bounded_test_le_one {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (h : Ω → ℝ) (hh : Measurable h)
    (a : ℝ) (hlo : ∀ x, a ≤ h x) (hhi : ∀ x, h x ≤ a + 1) :
    |(∫ x, h x ∂μ) - (∫ x, h x ∂ν)| ≤ 1 := by
  have hm := integral_mem_bounded_interval μ h hh a hlo hhi
  have hn := integral_mem_bounded_interval ν h hh a hlo hhi
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-- Full covariance comparison for actual rectangular linear images of the
standard Gaussian law. Singular covariance is covered by the probability bound
one; positive covariance is diagonalized and compared by the proved affinity
integral. No Gaussian distribution-identification hypothesis is present. -/
theorem gaussian_linearImage_bounded_test_le {κ : Type*} [Fintype κ] [DecidableEq κ]
    (A : Matrix κ ι ℝ) (h : (κ → ℝ) → ℝ) (hh : Measurable h)
    (a : ℝ) (hlo : ∀ x, a ≤ h x) (hhi : ∀ x, h x ≤ a + 1) :
    |(∫ x, h x ∂((standardGaussianMeasure ι).map (fun z => A *ᵥ z))) -
      (∫ x, h x ∂standardGaussianMeasure κ)| ≤
      min 1 (Real.sqrt (∑ i, ∑ j, (((A * Aᵀ - 1) : Matrix κ κ ℝ) i j) ^ 2)) := by
  letI := isProbabilityMeasure_map (μ := standardGaussianMeasure ι)
    (measurable_matrix_mulVec A).aemeasurable
  have hone := probability_bounded_test_le_one
    ((standardGaussianMeasure ι).map (fun z => A *ᵥ z)) (standardGaussianMeasure κ)
    h hh a hlo hhi
  apply le_min hone
  rw [SpectralComparison.gram_covariance_error_eq_eigenvalues A]
  let hS := Matrix.isHermitian_mul_conjTranspose_self A
  let v := hS.eigenvalues
  change _ ≤ Real.sqrt (∑ i, (v i - 1) ^ 2)
  by_cases hv : ∀ i, 0 < v i
  · let U : Matrix κ κ ℝ := hS.eigenvectorUnitary
    have hU : U * Uᵀ = 1 := by
      simpa only [U, unitary.coe_star, Matrix.star_eq_conjTranspose,
        Matrix.conjTranspose_eq_transpose_of_trivial]
        using unitary.coe_mul_star_self hS.eigenvectorUnitary
    have hcov : A * Aᵀ = U * diagonal v * Uᵀ := by
      simpa only [U, v, RCLike.ofReal_real_eq_id, Function.id_comp, Matrix.star_eq_conjTranspose,
        Matrix.conjTranspose_eq_transpose_of_trivial] using hS.spectral_theorem
    exact linearImage_bounded_test_le_of_factor A U v hv hU hcov h hh a hlo hhi
  · obtain ⟨i, hi⟩ := not_forall.mp hv
    have hnonneg : 0 ≤ v i := SpectralComparison.gram_eigenvalues_nonneg A i
    have hz : v i = 0 := le_antisymm (le_of_not_gt hi) hnonneg
    have hsum : 1 ≤ ∑ j, (v j - 1) ^ 2 := by
      have hterm := Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) => sq_nonneg (v j - 1))
        (Finset.mem_univ i)
      simpa only [hz, zero_sub, neg_sq, one_pow] using hterm
    exact hone.trans (Real.le_sqrt_of_sq_le (by simpa using hsum))

/-- Exact factorization of expectations from disjoint coordinate blocks of the
standard Gaussian law, proved by density factorization and Fubini. -/
theorem standardGaussian_sum_integral_product {κ : Type*} [Fintype κ]
    (f : (ι → ℝ) → ℝ) (g : (κ → ℝ) → ℝ) :
    (∫ z : (ι ⊕ κ) → ℝ, f (fun i => z (Sum.inl i)) * g (fun j => z (Sum.inr j))
      ∂standardGaussianMeasure (ι ⊕ κ)) =
      (∫ x, f x ∂standardGaussianMeasure ι) * (∫ y, g y ∂standardGaussianMeasure κ) := by
  simp only [standardGaussianMeasure, integral_diagonalGaussianMeasure, smul_eq_mul]
  have hroot : ∀ z : (ι ⊕ κ) → ℝ, diagonalGaussianRoot (fun _ => 1) z =
      diagonalGaussianRoot (fun _ : ι => 1) (fun i => z (Sum.inl i)) *
        diagonalGaussianRoot (fun _ : κ => 1) (fun j => z (Sum.inr j)) := by
    intro z
    exact Fintype.prod_sum_type _
  rw [← (volume_measurePreserving_sumPiEquivProdPi_symm (fun _ : ι ⊕ κ => ℝ)).integral_comp']
  simp only [MeasurableEquiv.coe_sumPiEquivProdPi_symm, Equiv.sumPiEquivProdPi_symm_apply]
  simp_rw [hroot]
  simp only [Sum.elim_inl, Sum.elim_inr, mul_pow]
  change (∫ z : (ι → ℝ) × (κ → ℝ),
    (diagonalGaussianRoot (fun _ : ι => 1) z.1 ^ 2 *
      diagonalGaussianRoot (fun _ : κ => 1) z.2 ^ 2) * (f z.1 * g z.2)) = _
  have hfactor : ∀ z : (ι → ℝ) × (κ → ℝ),
      (diagonalGaussianRoot (fun _ : ι => 1) z.1 ^ 2 *
        diagonalGaussianRoot (fun _ : κ => 1) z.2 ^ 2) * (f z.1 * g z.2) =
      (diagonalGaussianRoot (fun _ : ι => 1) z.1 ^ 2 * f z.1) *
        (diagonalGaussianRoot (fun _ : κ => 1) z.2 ^ 2 * g z.2) := by
    intro z
    ring
  simp_rw [hfactor]
  rw [Measure.volume_eq_prod]
  exact integral_prod_mul (fun x => diagonalGaussianRoot (fun _ : ι => 1) x ^ 2 * f x)
    (fun y => diagonalGaussianRoot (fun _ : κ => 1) y ^ 2 * g y)

end BootstrapCalibration
