import BootstrapCalibration.OLS
import BootstrapCalibration.GaussianLinearImage
import BootstrapCalibration.Whitening
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar

/-! Strict positivity of the fitted Gaussian residual scale follows from the
nullity of a proper linear subspace. No chi-square law or independence of the
fit and residual is assumed. -/

noncomputable section
open MeasureTheory Matrix
open scoped BigOperators
namespace BootstrapCalibration.ResidualScale

/-- The explicitly defined standard Gaussian density is absolutely continuous. -/
theorem standardGaussian_absolutelyContinuous (ι : Type*) [Fintype ι] :
    standardGaussianMeasure ι ≪ (volume : Measure (ι → ℝ)) :=
  withDensity_absolutelyContinuous _ _

/-- A nonzero real matrix has a Gaussian-null kernel, including rectangular maps. -/
theorem gaussian_matrix_kernel_null {ι κ : Type*} [Fintype ι] [Fintype κ]
    [DecidableEq ι] (A : Matrix κ ι ℝ) (hA : A ≠ 0) :
    standardGaussianMeasure ι {z | A *ᵥ z = 0} = 0 := by
  have hmap : A.mulVecLin ≠ 0 := by
    intro h
    apply hA
    apply Matrix.toLin'.injective
    simpa only [Matrix.toLin'_apply', Matrix.mulVecLin_zero] using h
  have hker : LinearMap.ker A.mulVecLin ≠ ⊤ :=
    fun h => hmap (LinearMap.ker_eq_top.mp h)
  exact standardGaussian_absolutelyContinuous ι
    (Measure.addHaar_submodule volume (LinearMap.ker A.mulVecLin) hker)

/-- The residual projection has the expected positive trace for n>3. -/
theorem trace_residual (n : ℕ) (Q : Matrix (Fin n) (Fin 3) ℝ)
    (hQ : Qᵀ * Q = 1) :
    (OLS.residual Q).trace = (n : ℝ) - 3 := by
  rw [OLS.residual, Matrix.trace_sub, Matrix.trace_mul_comm, hQ]
  simp

theorem residual_ne_zero (n : ℕ) (hn : 3 < n) (Q : Matrix (Fin n) (Fin 3) ℝ)
    (hQ : Qᵀ * Q = 1) : OLS.residual Q ≠ 0 := by
  intro hzero
  have htrace := trace_residual n Q hQ
  rw [hzero, Matrix.trace_zero] at htrace
  have hn' : (3 : ℝ) < n := by exact_mod_cast hn
  linarith

/-- Standardized residual standard deviation, with n-3 residual degrees of freedom. -/
def fittedScaleRatio (n : ℕ) (Q : Matrix (Fin n) (Fin 3) ℝ) (z : Fin n → ℝ) : ℝ :=
  ‖(EuclideanSpace.equiv (Fin n) ℝ).symm (OLS.residual Q *ᵥ z)‖ /
    Real.sqrt ((n : ℝ) - 3)

/-- The fitted-scale ratio is positive whenever the residual vector is nonzero. -/
theorem fittedScaleRatio_pos (n : ℕ) (hn : 3 < n)
    (Q : Matrix (Fin n) (Fin 3) ℝ) (z : Fin n → ℝ)
    (hz : OLS.residual Q *ᵥ z ≠ 0) : 0 < fittedScaleRatio n Q z := by
  have hn' : (3 : ℝ) < n := by exact_mod_cast hn
  apply div_pos
  · apply norm_pos_iff.mpr
    intro hzero
    apply hz
    have h := congrArg (EuclideanSpace.equiv (Fin n) ℝ) hzero
    simpa using h
  · exact Real.sqrt_pos.mpr (sub_pos.mpr hn')

/-- The actual Gaussian data law makes the studentizing denominator positive a.s. -/
theorem fittedScaleRatio_pos_ae (n : ℕ) (hn : 3 < n)
    (Q : Matrix (Fin n) (Fin 3) ℝ) (hQ : Qᵀ * Q = 1) :
    ∀ᵐ z ∂standardGaussianMeasure (Fin n), 0 < fittedScaleRatio n Q z := by
  have hnz : ∀ᵐ z ∂standardGaussianMeasure (Fin n), OLS.residual Q *ᵥ z ≠ 0 := by
    rw [ae_iff]
    simpa only [not_not] using
      gaussian_matrix_kernel_null (OLS.residual Q) (residual_ne_zero n hn Q hQ)
  exact hnz.mono (fun z hz => fittedScaleRatio_pos n hn Q z hz)

/-- Original distinct quadratic actions inherit positive residual scale after whitening. -/
theorem original_quadratic_fittedScaleRatio_pos_ae (n : ℕ) (hn : 3 < n)
    (u : Fin n → ℝ) (hu : Function.Injective u) :
    ∀ᵐ z ∂standardGaussianMeasure (Fin n),
      0 < fittedScaleRatio n (Whitening.normalizedDesign (QuadraticRank.design u)) z :=
  fittedScaleRatio_pos_ae n hn _ (Whitening.normalizedDesign_orthonormal _
    (Whitening.quadratic_original_gram_isUnit n hn.le u hu))

end BootstrapCalibration.ResidualScale
